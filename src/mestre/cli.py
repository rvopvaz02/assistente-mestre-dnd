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


def _read_pages() -> None:
    import json
    from pathlib import Path

    from mestre import config
    from mestre.llm import GeminiLLM
    from mestre.vision.pages import read_page, select_pages

    manifest = json.loads(config.CORPUS_MANIFEST.read_text(encoding="utf-8"))
    llm = GeminiLLM(model=config.VISION_PAGES_MODEL)
    total_in = total_out = 0
    for src in manifest["fontes"]:
        path = Path(src["caminho"])
        path = path if path.is_absolute() else config.CORPUS_MANIFEST.parent / path
        if path.suffix.lower() != ".pdf" or not path.exists():
            continue
        for sel in select_pages(path, src["paginas"]):
            r = read_page(path, src["id"], sel.pdf_page, llm, motivos=sel.motivos)
            lt = r.leitura
            print(f"p{sel.pdf_page:03d} {'cache' if r.do_cache else f'{r.latencia_s:.1f}s'} "
                  f"tipo={lt.get('tipo_pagina')} tabelas={len(lt.get('tabelas') or [])} "
                  f"capitulares={[c.get('palavra_completa') for c in lt.get('capitulares') or []]} "
                  f"figuras_regra={sum(bool(f.get('relevante_para_regras')) for f in lt.get('figuras') or [])} "
                  f"tokens={r.tokens_entrada}+{r.tokens_saida}  ({'; '.join(sel.motivos)})")
            if not r.do_cache:
                total_in += r.tokens_entrada
                total_out += r.tokens_saida
    print(f"Tokens gastos nesta execução: {total_in} entrada + {total_out} saída. Rode 'mestre indexar' em seguida.")


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
    sub.add_parser("ler-paginas", help="lê com visão as páginas do índice visual (usa cache)")
    m = sub.add_parser("mesa", help="lê uma foto da mesa: dados, miniaturas e mapa")
    m.add_argument("foto")
    a = ap.parse_args(argv)

    if a.cmd == "ler-paginas":
        _read_pages()
        return
    if a.cmd == "mesa":
        from mestre.llm import GeminiLLM
        from mestre.vision.table import read_table

        r = read_table(a.foto, GeminiLLM())
        import json

        print(json.dumps(r.leitura, ensure_ascii=False, indent=2))
        print(f"\n[{r.latencia_s:.2f}s | tokens {r.tokens_entrada}+{r.tokens_saida}]")
        return

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
