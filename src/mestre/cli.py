"""Linha de comando do assistente.

  mestre indexar                       # lê data/corpus/manifest.json e gera os chunks
  mestre perguntar "pergunta" [--mestre]
  mestre voz caminho/do/audio.wav [--falar saida.mp3] [--mestre]
"""

from __future__ import annotations

import argparse
import logging
import sys


def _assistant():
    from mestre.llm import GeminiLLM
    from mestre.rag.answer import RulesAssistant
    from mestre.rag.index import HybridIndex
    from mestre.rag.ingest import load_chunks

    return RulesAssistant(HybridIndex(load_chunks()), GeminiLLM())


def _print_answer(a) -> None:
    print(f"\n{a.texto}\n")
    if a.abstencao:
        print(f"(abstenção: {a.motivo})")
    for f in a.fontes():
        print("  " + f)
    print(f"\n[{a.latencia_s:.2f}s | tokens {a.tokens_entrada}+{a.tokens_saida}]")


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(prog="mestre")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("indexar")
    p = sub.add_parser("perguntar")
    p.add_argument("pergunta")
    p.add_argument("--mestre", action="store_true", help="inclui fontes com permissão 'mestre'")
    v = sub.add_parser("voz")
    v.add_argument("audio")
    v.add_argument("--falar", help="salva a resposta falada neste .mp3")
    v.add_argument("--mestre", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "indexar":
        from mestre.rag.index import HybridIndex
        from mestre.rag.ingest import build_chunks, save_chunks

        chunks = build_chunks()
        path = save_chunks(chunks)
        HybridIndex(chunks)
        print(f"{len(chunks)} chunks salvos em {path} e indexados.")
        return

    perfil = "mestre" if a.mestre else "jogadores"
    if a.cmd == "perguntar":
        _print_answer(_assistant().ask(a.pergunta, perfil=perfil))
    elif a.cmd == "voz":
        from mestre import config
        from mestre.asr.transcriber import Transcriber
        from mestre.asr.vocab import build_prompt, load_terms

        t = Transcriber().transcribe(a.audio, prompt=build_prompt(load_terms(config.VOCAB_TERMS)))
        print(f"Transcrição ({t.seconds_processing:.1f}s em {t.device}): {t.text}")
        ans = _assistant().ask(t.text, perfil=perfil)
        _print_answer(ans)
        if a.falar:
            import re

            from mestre.asr.tts import speak

            speak(re.sub(r"\[\d+\]", "", ans.texto), a.falar)
            print(f"Resposta falada salva em {a.falar}")


if __name__ == "__main__":
    main()
