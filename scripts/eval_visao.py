"""Comparação direta: responder pelo texto extraído x pelo texto + leitura visual das mesmas páginas.

Também resume o custo da visão: páginas lidas, tokens por página, latência e cache.

Uso:  uv run python scripts/eval_visao.py
Saída: results/visao_eval.json e results/visao_eval.md
"""

from __future__ import annotations

import json
import statistics
import sys

from mestre import config
from mestre.llm import GeminiLLM, QuotaExhausted
from mestre.rag.answer import RulesAssistant
from mestre.rag.index import HybridIndex
from mestre.rag.ingest import build_chunks
from mestre.vision.pages import CACHE_DIR, PageReading

TESTSET = config.DATA_DIR / "eval" / "visao_testset.jsonl"


def norm(s: str) -> str:
    return s.lower().replace("–", "-").replace("−", "-")


def run_condition(name: str, chunks, testset, llm) -> list[dict]:
    bot = RulesAssistant(HybridIndex(chunks, persist=False), llm)
    rows = []
    for q in testset:
        try:
            a = bot.ask(q["pergunta"])
        except QuotaExhausted:
            print("  cota esgotada", file=sys.stderr)
            break
        ok = (not a.abstencao) and all(norm(s) in norm(a.texto) for s in q["resposta_contem"])
        used_visual = any(a.trechos[i - 1].chunk.modalidade == "visao" for i in a.citacoes if 0 < i <= len(a.trechos))
        rows.append({"id": q["id"], "elemento": q["elemento"], "correta": ok, "abstencao": a.abstencao,
                     "citou_visao": used_visual, "resposta": a.texto, "fontes": a.fontes(),
                     "latencia_s": round(a.latencia_s, 3)})
        print(f"  [{name}] {q['id']}: {'OK ' if ok else 'ERRO'} {a.texto[:90]!r}")
    return rows


def vision_costs() -> dict:
    out = {}
    for f in CACHE_DIR.glob("*.json"):
        r = PageReading(**json.loads(f.read_text(encoding="utf-8")))
        out.setdefault(r.modelo, []).append(r)
    summary = {}
    for model, rs in out.items():
        summary[model] = {
            "paginas": len(rs),
            "tokens_entrada_por_pagina": round(statistics.mean(r.tokens_entrada for r in rs)),
            "tokens_saida_por_pagina": round(statistics.mean(r.tokens_saida for r in rs)),
            "latencia_media_s": round(statistics.mean(r.latencia_s for r in rs), 1),
            "capitulares_lidas": sorted({c.get("palavra_completa") for r in rs
                                         for c in r.leitura.get("capitulares") or []} - {None}),
        }
    return summary


def main() -> None:
    testset = [json.loads(l) for l in TESTSET.read_text(encoding="utf-8").splitlines() if l.strip()]
    llm = GeminiLLM()
    so_texto = build_chunks(visao=False)
    com_visao = build_chunks(visao=True)
    res = {"modelo_resposta": llm.model, "modelo_visao": config.VISION_PAGES_MODEL,
           "chunks": {"texto": len(so_texto), "texto+visao": len(com_visao)}}
    res["texto"] = run_condition("texto", so_texto, testset, llm)
    res["texto+visao"] = run_condition("texto+visao", com_visao, testset, llm)
    for k in ("texto", "texto+visao"):
        rows = res[k]
        res[f"acerto_{k}"] = round(sum(r["correta"] for r in rows) / max(len(rows), 1), 4)
    res["custo_visao"] = vision_costs()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "visao_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    by_el: dict[str, dict[str, list[bool]]] = {}
    for k in ("texto", "texto+visao"):
        for r in res[k]:
            by_el.setdefault(r["elemento"], {}).setdefault(k, []).append(r["correta"])
    L = [f"## Texto extraído x texto + visão ({len(testset)} perguntas sobre tabelas, `{llm.model}`)", "",
         "| Elemento | n | Só texto | Texto + visão |", "|---|---|---|---|"]
    for el, v in by_el.items():
        n = len(v.get("texto", []))
        L.append(f"| {el} | {n} | {sum(v.get('texto', []))}/{n} | {sum(v.get('texto+visao', []))}/{n} |")
    L.append(f"| **Geral** | {len(testset)} | **{res['acerto_texto']:.0%}** | **{res['acerto_texto+visao']:.0%}** |")
    L += ["", "## Custo da leitura visual (por página)", "",
          "| Modelo | Páginas | Tokens entrada | Tokens saída | Latência média | Capitulares lidas |", "|---|---|---|---|---|---|"]
    for m, v in res["custo_visao"].items():
        L.append(f"| {m} | {v['paginas']} | {v['tokens_entrada_por_pagina']} | {v['tokens_saida_por_pagina']} | "
                 f"{v['latencia_media_s']} s | {', '.join(v['capitulares_lidas'])} |")
    md = "\n".join(L) + "\n"
    (config.RESULTS_DIR / "visao_eval.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
