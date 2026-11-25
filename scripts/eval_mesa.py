"""Acurácia da leitura de dados em fotos da mesa (data/mesa/gabarito.csv).

Métricas
  foto_exata  : todos os dados da foto lidos certo (tipo e valor), sem dado inventado.
  precisao    : dos dados que o modelo informou, quantos existem no gabarito.
  revocacao   : dos dados do gabarito, quantos o modelo leu certo.
  tipo_certo  : acerto do tipo do dado (d6, d20...), independentemente do valor.
Por estrato (coluna `condicao`) e por tipo de dado.

Uso:  uv run python scripts/eval_mesa.py [--modelo gemini-3.5-flash]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict

from mestre import config
from mestre.llm import GeminiLLM, QuotaExhausted
from mestre.vision.table import dice_multiset, read_table

DIR = config.DATA_DIR / "mesa"


def score(expected: list[str], got: list[str]) -> dict:
    e, g = Counter(expected), Counter(got)
    tp = sum((e & g).values())
    et, gt = Counter(x.split(":")[0] for x in expected), Counter(x.split(":")[0] for x in got)
    return {"tp": tp, "n_esperado": len(expected), "n_lido": len(got), "exata": e == g,
            "tipo_tp": sum((et & gt).values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=None)
    a = ap.parse_args()
    rows = list(csv.DictReader((DIR / "gabarito.csv").open(encoding="utf-8")))
    rows = [r for r in rows if (DIR / r["arquivo"]).exists()]
    if not rows:
        raise SystemExit("Nenhuma foto com gabarito em data/mesa/. Veja data/mesa/README.md.")
    llm = GeminiLLM(model=a.modelo)
    results = []
    for r in rows:
        expected = sorted(x.strip().lower() for x in r["dados"].split(";") if x.strip())
        try:
            t = read_table(DIR / r["arquivo"], llm)
        except QuotaExhausted:
            print("cota esgotada; resultado parcial")
            break
        got = dice_multiset(t.leitura)
        s = score(expected, got)
        results.append({**r, "esperado": expected, "lido": got, **s, "latencia_s": round(t.latencia_s, 2),
                        "tokens": t.tokens_entrada + t.tokens_saida})
        print(f"{r['arquivo']}: {'OK ' if s['exata'] else 'ERRO'} esperado={expected} lido={got}")

    def agg(rs):
        tp, ne, nl = (sum(x[k] for x in rs) for k in ("tp", "n_esperado", "n_lido"))
        return {"n_fotos": len(rs), "foto_exata": round(sum(x["exata"] for x in rs) / len(rs), 4),
                "precisao": round(tp / nl, 4) if nl else None, "revocacao": round(tp / ne, 4) if ne else None,
                "tipo_certo": round(sum(x["tipo_tp"] for x in rs) / ne, 4) if ne else None,
                "latencia_p50_s": statistics.median(x["latencia_s"] for x in rs)}

    by = defaultdict(list)
    for x in results:
        by["geral"].append(x)
        by[f"condicao={x['condicao']}"].append(x)
    out = {"modelo": llm.model, "resumo": {k: agg(v) for k, v in by.items()}, "fotos": results}
    (config.RESULTS_DIR / "mesa_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    L = [f"## Leitura de dados em fotos (`{llm.model}`)", "",
         "| Estrato | Fotos | Foto 100% certa | Precisão | Revocação | Tipo certo | p50 |", "|---|---|---|---|---|---|---|"]
    f = lambda v: f"{v:.0%}" if isinstance(v, float) else "-"  # noqa: E731
    for k, v in out["resumo"].items():
        L.append(f"| {k} | {v['n_fotos']} | {f(v['foto_exata'])} | {f(v['precisao'])} | {f(v['revocacao'])} | "
                 f"{f(v['tipo_certo'])} | {v['latencia_p50_s']} s |")
    md = "\n".join(L) + "\n"
    (config.RESULTS_DIR / "mesa_eval.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
