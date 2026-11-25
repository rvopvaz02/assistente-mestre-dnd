"""Avaliação do RAG sobre o conjunto estratificado data/eval/rag_testset.jsonl.

1. Recuperação: Recall@k (k=1,3,5) por configuração (bm25, denso, híbrido, híbrido+reranker) e por estrato.
2. Portão de abstenção: limiar calibrado no split "dev" e reportado no split "teste" (sem espiar o teste).
3. Ponta a ponta (se houver GEMINI_API_KEY): abstenção correta, acerto da resposta, auditoria de
   citações, latência p50/p95 e tokens por pergunta.

Uso:  uv run python scripts/eval_rag.py [--sem-llm]
Saída: results/rag_eval.json e results/rag_eval.md
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from mestre import config
from mestre.rag.index import PERMISSOES, HybridIndex
from mestre.rag.ingest import Chunk, load_chunks

TESTSET = config.DATA_DIR / "eval" / "rag_testset.jsonl"
CONFIGS = {"bm25": ("bm25", False), "denso": ("dense", False), "hibrido": ("hybrid", False),
           "hibrido+rerank": ("hybrid", True)}
KS = (1, 3, 5)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def matches(chunk: Chunk, spec: dict) -> bool:
    if "secao" in spec and norm(chunk.secao) != norm(spec["secao"]):
        return False
    if "contem" in spec and norm(spec["contem"]) not in norm(chunk.texto):
        return False
    return True


def group_hit(chunks: list[Chunk], group: list[dict]) -> bool:
    return any(matches(c, spec) for c in chunks for spec in group)


def validate(testset: list[dict], chunks: list[Chunk]) -> None:
    bad = []
    for q in testset:
        allowed = PERMISSOES[q["perfil"]]
        pool = [c for c in chunks if c.permissao in allowed]
        for g in q["evidencias"]:
            if not group_hit(pool, g):
                bad.append((q["id"], g))
    if bad:
        sys.exit(f"Conjunto de teste referencia evidências que não existem no corpus: {bad}")


def pct(x: float) -> str:
    return f"{x:.1%}"


def eval_retrieval(index: HybridIndex, testset: list[dict]) -> tuple[dict, dict]:
    out: dict = {}
    top_scores: dict[str, float] = {}
    answerable = [q for q in testset if q["evidencias"]]
    for name, (mode, rr) in CONFIGS.items():
        per_q = {}
        for q in testset:
            hits = index.search(q["pergunta"], k=max(KS), mode=mode, rerank=rr, perfil=q["perfil"])
            if name == "hibrido+rerank":
                top_scores[q["id"]] = hits[0].score if hits else 0.0
            if not q["evidencias"]:
                continue
            per_q[q["id"]] = {k: sum(group_hit([h.chunk for h in hits[:k]], g) for g in q["evidencias"])
                              / len(q["evidencias"]) for k in KS}
        by_stratum = defaultdict(list)
        for q in answerable:
            by_stratum["geral"].append(per_q[q["id"]])
            by_stratum[q["estrato"]].append(per_q[q["id"]])
        out[name] = {s: {f"recall@{k}": round(statistics.mean(v[k] for v in vals), 4) for k in KS} | {"n": len(vals)}
                     for s, vals in by_stratum.items()}
    return out, top_scores


def calibrate(testset: list[dict], scores: dict[str, float]) -> dict:
    def acc(th: float, split: str) -> dict:
        qs = [q for q in testset if q["split"] == split]
        tp = sum(1 for q in qs if q["deve_abster"] and scores[q["id"]] < th)
        fn = sum(1 for q in qs if q["deve_abster"] and scores[q["id"]] >= th)
        tn = sum(1 for q in qs if not q["deve_abster"] and scores[q["id"]] >= th)
        fp = sum(1 for q in qs if not q["deve_abster"] and scores[q["id"]] < th)
        rec_abst = tp / max(tp + fn, 1)
        rec_resp = tn / max(tn + fp, 1)
        return {"limiar": th, "abstencao_correta": round(rec_abst, 4), "respondeu_quando_devia": round(rec_resp, 4),
                "acuracia_balanceada": round((rec_abst + rec_resp) / 2, 4), "n": len(qs)}

    # O portão é a 1ª de duas camadas (a 2ª é o LLM com NAO_SEI). Critério: o maior limiar que ainda
    # deixa passar >= 95% das perguntas respondíveis do dev. Bloquear pergunta boa é o erro mais caro aqui.
    grid = sorted({round(s - 1e-4, 4) for s in scores.values()} | {0.001, 0.01, 0.05, 0.1})
    rows = [acc(t, "dev") for t in grid]
    safe = [r for r in rows if r["respondeu_quando_devia"] >= 0.95] or rows
    best = max(safe, key=lambda r: (r["abstencao_correta"], r["limiar"]))
    th = best["limiar"]
    best_bal = max(rows, key=lambda r: r["acuracia_balanceada"])
    return {"criterio": "maior limiar com >=95% de respostas no dev", "dev": best, "teste": acc(th, "teste"),
            "alternativa_acuracia_balanceada": {"dev": best_bal, "teste": acc(best_bal["limiar"], "teste")}}


def eval_end_to_end(index: HybridIndex, testset: list[dict], threshold: float) -> dict:
    from mestre.llm import GeminiLLM, QuotaExhausted
    from mestre.rag.answer import RulesAssistant

    llm = GeminiLLM()
    bot = RulesAssistant(index, llm, threshold=threshold)
    rows = []
    interrompido = None
    for q in testset:
        try:
            a = bot.ask(q["pergunta"], perfil=q["perfil"])
        except QuotaExhausted:
            interrompido = f"cota diária do {llm.model} esgotada após {len(rows)} perguntas"
            print("  " + interrompido, file=sys.stderr)
            break
        txt = norm(a.texto)
        ok_abst = a.abstencao == q["deve_abster"]
        has = all(norm(s) in txt for s in q.get("resposta_contem", [])) if not a.abstencao else False
        hasnt = not any(norm(s) in txt for s in q.get("resposta_nao_contem", []))
        cited = [a.trechos[i - 1].chunk for i in a.citacoes if 0 < i <= len(a.trechos)]
        cite_ok = all(group_hit(cited, g) for g in q["evidencias"]) if q["evidencias"] and not a.abstencao else None
        correct = ok_abst and hasnt and (q["deve_abster"] or has)
        rows.append({"id": q["id"], "estrato": q["estrato"], "split": q["split"], "pergunta": q["pergunta"],
                     "resposta": a.texto, "abstencao": a.abstencao, "motivo": a.motivo, "fontes": a.fontes(),
                     "correta": correct, "abstencao_ok": ok_abst, "citou_evidencia": cite_ok,
                     "auditoria_taxa_ok": a.auditoria.get("taxa_ok"), "latencia_s": round(a.latencia_s, 3),
                     "tokens_entrada": a.tokens_entrada, "tokens_saida": a.tokens_saida})
        print(f"  {q['id']}: {'OK ' if correct else 'ERRO'} {a.motivo}")
    by = defaultdict(list)
    for r in rows:
        by["geral"].append(r)
        by[r["estrato"]].append(r)
    summary = {}
    for s, rs in by.items():
        lat = sorted(r["latencia_s"] for r in rs)
        cites = [r["citou_evidencia"] for r in rs if r["citou_evidencia"] is not None]
        aud = [r["auditoria_taxa_ok"] for r in rs if r["auditoria_taxa_ok"] is not None]
        summary[s] = {"n": len(rs), "acerto": round(float(sum(r["correta"] for r in rs) / len(rs)), 4),
                      "abstencao_ok": round(float(sum(r["abstencao_ok"] for r in rs) / len(rs)), 4),
                      "citou_evidencia": round(float(statistics.mean(map(float, cites))), 4) if cites else None,
                      "auditoria_citacoes": round(float(statistics.mean(aud)), 4) if aud else None,
                      "latencia_p50_s": round(statistics.median(lat), 3),
                      "latencia_p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 3)}
    if rows:
        summary["geral"]["tokens_medios"] = {
            "entrada": round(statistics.mean(r["tokens_entrada"] for r in rows), 1),
            "saida": round(statistics.mean(r["tokens_saida"] for r in rows), 1)}
    return {"modelo": llm.model, "interrompido": interrompido, "resumo": summary, "respostas": rows}


def to_markdown(res: dict) -> str:
    L = ["## Recuperação (Recall@k por estrato)", ""]
    strata = ["geral", "rotineira", "composta", "adversarial"]
    L += ["| Configuração | " + " | ".join(f"{s} R@1 / R@3 / R@5" for s in strata) + " |",
          "|---|" + "---|" * len(strata)]
    for name, r in res["recuperacao"].items():
        cells = []
        for s in strata:
            v = r.get(s)
            cells.append(" / ".join(pct(v[f"recall@{k}"]) for k in KS) if v else "-")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    c = res["abstencao_portao"]
    L += ["", "## Portão de abstenção (score do reranker)", "",
          f"Limiar calibrado no dev: **{c['dev']['limiar']}**", "",
          "| Split | n | Abstenção correta | Respondeu quando devia | Acurácia balanceada |", "|---|---|---|---|---|"]
    for split in ("dev", "teste"):
        v = c[split]
        L.append(f"| {split} | {v['n']} | {pct(v['abstencao_correta'])} | {pct(v['respondeu_quando_devia'])} | {pct(v['acuracia_balanceada'])} |")
    if "ponta_a_ponta" in res:
        e2e = res["ponta_a_ponta"]
        nota = f" **Parcial: {e2e['interrompido']}.**" if e2e.get("interrompido") else ""
        L += ["", f"## Ponta a ponta (`{e2e['modelo']}`)", "", nota.strip(), "",
              "| Estrato | n | Acerto | Abstenção ok | Citou a evidência | Auditoria de citações | p50 (s) | p95 (s) |",
              "|---|---|---|---|---|---|---|---|"]
        for s, v in res["ponta_a_ponta"]["resumo"].items():
            f = lambda x: pct(x) if isinstance(x, float) else "-"  # noqa: E731
            L.append(f"| {s} | {v['n']} | {f(v['acerto'])} | {f(v['abstencao_ok'])} | {f(v['citou_evidencia'])} | "
                     f"{f(v['auditoria_citacoes'])} | {v['latencia_p50_s']} | {v['latencia_p95_s']} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-llm", action="store_true", help="avalia só recuperação e portão de abstenção")
    a = ap.parse_args()
    testset = [json.loads(l) for l in TESTSET.read_text(encoding="utf-8").splitlines() if l.strip()]
    chunks = load_chunks()
    validate(testset, chunks)
    index = HybridIndex(chunks)
    res: dict = {"n_perguntas": len(testset), "n_chunks": len(chunks), "embed_model": index.embed_model_name,
                 "rerank_model": index.rerank_model_name}
    res["recuperacao"], scores = eval_retrieval(index, testset)
    res["abstencao_portao"] = calibrate(testset, scores)
    res["scores_top1"] = scores
    if not a.sem_llm and os.getenv("GEMINI_API_KEY"):
        res["ponta_a_ponta"] = eval_end_to_end(index, testset, res["abstencao_portao"]["dev"]["limiar"])
    elif not a.sem_llm:
        print("Sem GEMINI_API_KEY: avaliação ponta a ponta pulada.", file=sys.stderr)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "rag_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    md = to_markdown(res)
    (config.RESULTS_DIR / "rag_eval.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
