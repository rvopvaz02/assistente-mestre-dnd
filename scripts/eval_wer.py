"""Avaliação do ASR: WER em 4 condições, por amostra e por estrato (ambiente/locutor).

Condições
  base           Whisper puro.
  prompt         Whisper + vocabulário de domínio como initial_prompt/hotwords (sem vazamento).
  base+dic_loo   Whisper puro + dicionário de correções, leave-one-out.
  prompt+dic_loo As duas estratégias juntas, leave-one-out.
  base+dic_vaz   Dicionário aplicado SEM leave-one-out: só para mostrar quanto o vazamento infla o número.
  prompt+llm     Prompt + pós-correção pelo Gemini com a lista genérica de termos (só se GEMINI_API_KEY existir).

Uso:  uv run python scripts/eval_wer.py [--model large-v3-turbo] [--model base]
Saída: results/wer_<modelo>.json e results/wer_<modelo>.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from mestre import config
from mestre.asr.transcriber import Transcriber
from mestre.asr.vocab import CorrectionDict, build_prompt, llm_correct, load_terms
from mestre.asr.wer import sample_wer, term_hits, wer


def load_refs(audio_dir: Path) -> list[dict]:
    with (audio_dir / "refs.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    present = [r for r in rows if (audio_dir / r["arquivo"]).exists()]
    missing = [r["arquivo"] for r in rows if r not in present]
    if missing:
        print(f"Aviso: {len(missing)} áudios listados não encontrados: {', '.join(missing)}", file=sys.stderr)
    return present


def run(model: str, audio_dir: Path) -> dict:
    rows = load_refs(audio_dir)
    if not rows:
        sys.exit(f"Nenhum áudio encontrado em {audio_dir}. Veja data/audio/README.md.")
    terms = load_terms(config.VOCAB_TERMS)
    prompt = build_prompt(terms)
    dic = CorrectionDict.load(config.VOCAB_CORRECTIONS)
    llm = None
    if os.getenv("GEMINI_API_KEY"):
        from mestre.llm import GeminiLLM
        llm = GeminiLLM()
    else:
        print("Sem GEMINI_API_KEY: condição prompt+llm será pulada.", file=sys.stderr)
    if any(not c.origin for c in dic.items):
        print("Aviso: há correções sem 'origem'; elas entram em todas as amostras e podem vazar.", file=sys.stderr)

    asr = Transcriber(model)
    print(f"Whisper {model} em {asr.device}")
    per_sample = []
    for r in rows:
        path = audio_dir / r["arquivo"]
        base = asr.transcribe(path)
        withp = asr.transcribe(path, prompt=prompt)
        hyps = {
            "base": base.text,
            "prompt": withp.text,
            "base+dic_loo": dic.apply(base.text, exclude_origin=r["arquivo"]),
            "prompt+dic_loo": dic.apply(withp.text, exclude_origin=r["arquivo"]),
            "base+dic_vaz": dic.apply(base.text),
        }
        if llm:
            hyps["prompt+llm"] = llm_correct(withp.text, terms, llm)
        per_sample.append({
            **r,
            "rtf": round(base.seconds_processing / max(base.seconds_audio, 1e-6), 3),
            "hipoteses": hyps,
            "wer": {k: round(sample_wer(r["referencia"], h), 4) for k, h in hyps.items()},
        })
        print(f"  {r['arquivo']}: base={per_sample[-1]['wer']['base']:.2%} prompt={per_sample[-1]['wer']['prompt']:.2%}")

    conds = list(per_sample[0]["hipoteses"])
    refs = [s["referencia"] for s in per_sample]
    overall = {c: round(wer(refs, [s["hipoteses"][c] for s in per_sample]), 4) for c in conds}
    strata: dict[str, dict] = {}
    for key in ("ambiente", "locutor"):
        groups = defaultdict(list)
        for s in per_sample:
            groups[f"{key}={s.get(key, '?')}"].append(s)
        for g, items in groups.items():
            strata[g] = {"n": len(items), **{c: round(wer([i["referencia"] for i in items],
                                                          [i["hipoteses"][c] for i in items]), 4) for c in conds}}
    terms_acc = {}
    for c in conds:
        hits = [term_hits(s["referencia"], s["hipoteses"][c], terms) for s in per_sample]
        found, total = sum(h[0] for h in hits), sum(h[1] for h in hits)
        terms_acc[c] = round(found / total, 4) if total else None
    return {"modelo": model, "dispositivo": asr.device, "n_amostras": len(per_sample), "prompt": prompt,
            "wer_geral": overall, "acerto_termos": terms_acc,
            "n_termos": sum(term_hits(s["referencia"], s["referencia"], terms)[1] for s in per_sample),
            "estratos": strata, "amostras": per_sample}


def to_markdown(res: dict) -> str:
    conds = list(res["wer_geral"])
    lines = [f"## WER: Whisper `{res['modelo']}` ({res['dispositivo']}), {res['n_amostras']} amostras", "",
             "| Estrato | n | " + " | ".join(conds) + " |", "|---|---|" + "---|" * len(conds),
             "| **Geral** | " + str(res["n_amostras"]) + " | " + " | ".join(f"{res['wer_geral'][c]:.1%}" for c in conds) + " |"]
    for g, v in res["estratos"].items():
        lines.append(f"| {g} | {v['n']} | " + " | ".join(f"{v[c]:.1%}" for c in conds) + " |")
    lines.append(f"| **Acerto de termos de D&D** ↑ | {res['n_termos']} termos | "
                 + " | ".join(f"{res['acerto_termos'][c]:.1%}" for c in conds) + " |")
    lines += ["", "`dic_loo` = dicionário avaliado com leave-one-out; `dic_vaz` = com vazamento (só referência).", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", help="tamanho do Whisper (pode repetir)")
    ap.add_argument("--audio-dir", type=Path, default=config.AUDIO_DIR)
    a = ap.parse_args()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    for m in a.model or [config.WHISPER_MODEL]:
        res = run(m, a.audio_dir)
        stem = config.RESULTS_DIR / f"wer_{m}"
        stem.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        md = to_markdown(res)
        stem.with_suffix(".md").write_text(md, encoding="utf-8")
        print(md)


if __name__ == "__main__":
    main()
