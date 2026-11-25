"""Cálculo de WER com normalização explícita (documentada, para o número ser reprodutível)."""

from __future__ import annotations

import re
import unicodedata

import jiwer

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


_NUM = {w: str(i) for i, w in enumerate(
    "zero um dois três quatro cinco seis sete oito nove dez onze doze treze catorze quinze "
    "dezesseis dezessete dezoito dezenove vinte".split())}
_NUM.update({"uma": "1", "duas": "2", "quatorze": "14"})


def normalize(text: str) -> str:
    """minúsculas, sem pontuação, espaços colapsados. Acentos são mantidos (mudam significado em PT).

    - "D20", "d-20" e "d 20" viram "d20" para não punir grafia equivalente.
    - Números por extenso de 0 a 20 viram dígitos ("dezessete" == "17"). A regra vale para os dois
      lados (referência e hipótese), então "um" artigo também vira "1" sem distorcer a métrica.
      Números compostos ("cento e trinta e cinco") devem ser escritos em dígitos na referência.
    """
    t = unicodedata.normalize("NFC", text).lower()
    t = re.sub(r"\bd\s*-?\s*(\d+)\b", r"d\1", t)
    t = _PUNCT.sub(" ", t)
    t = " ".join(_NUM.get(w, w) for w in t.split())
    return re.sub(r"\s+", " ", t).strip()


def wer(references: list[str], hypotheses: list[str]) -> float:
    """WER agregado no corpus (soma de erros / soma de palavras de referência)."""
    refs = [normalize(r) for r in references]
    hyps = [normalize(h) for h in hypotheses]
    return jiwer.wer(refs, hyps)


def sample_wer(reference: str, hypothesis: str) -> float:
    return jiwer.wer(normalize(reference), normalize(hypothesis))
