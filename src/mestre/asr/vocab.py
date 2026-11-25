"""Vocabulário de domínio para o ASR.

Três estratégias, avaliadas separadamente em scripts/eval_wer.py:

1. **Prompt de domínio** (`build_prompt`): a lista de termos vai como `initial_prompt` do Whisper,
   que passa a preferir essas grafias. Não depende de nenhum erro observado, então não há vazamento.
   (Usar `hotwords` junto com o prompt fez o large-v3-turbo entrar em laço de repetição; por isso não usamos.)
2. **Dicionário de correções** (`CorrectionDict`): pares "erro -> correto" anotados a partir de erros
   reais. Como ele é construído olhando as amostras, só vale medir com *leave-one-out*: ao avaliar a
   amostra X, as correções que vieram da própria amostra X são descartadas.
3. **Pós-correção por LLM** (`llm_correct`): o LLM recebe a transcrição e a lista genérica de termos e
   corrige só grafias de termos do jogo. Também não usa erros observados, então não há vazamento.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


def load_terms(path: str | Path) -> list[str]:
    terms = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            terms.append(line)
    return terms


def build_prompt(terms: list[str], max_chars: int = 600) -> str:
    """Prompt curto em estilo de transcrição (o Whisper imita o estilo do prompt).

    O limite de ~224 tokens do Whisper é respeitado cortando a lista em `max_chars`.
    """
    head = "Sessão de RPG de mesa, Dungeons & Dragons 5ª edição. Termos: "
    out = head
    for t in terms:
        piece = t + ", "
        if len(out) + len(piece) > max_chars:
            break
        out += piece
    return out.rstrip(", ") + "."


@dataclass
class Correction:
    wrong: str
    right: str
    origin: str  # id da amostra de onde o erro foi observado ("" = genérico)


class CorrectionDict:
    def __init__(self, items: list[Correction]):
        # Frases mais longas primeiro, para "seado lixo" vencer "lixo".
        self.items = sorted(items, key=lambda c: -len(c.wrong))

    @classmethod
    def load(cls, path: str | Path) -> "CorrectionDict":
        p = Path(path)
        if not p.exists():
            return cls([])
        with p.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        return cls([Correction(r["errado"].strip(), r["correto"].strip(), (r.get("origem") or "").strip())
                    for r in rows if r.get("errado")])

    def apply(self, text: str, exclude_origin: str | None = None) -> str:
        for c in self.items:
            if exclude_origin and c.origin == exclude_origin:
                continue
            text = re.sub(rf"(?<!\w){re.escape(c.wrong)}(?!\w)", c.right, text, flags=re.IGNORECASE)
        return text


LLM_CORRECT_SYSTEM = (
    "Você corrige transcrições automáticas de fala de sessões de Dungeons & Dragons em português. "
    "O reconhecimento de voz costuma errar nomes do jogo, trocando-os por palavras de som parecido. "
    "Corrija SOMENTE palavras que sejam claramente um termo da lista escrito errado. "
    "Não reescreva, não pontue de novo, não troque números, não acrescente nada. "
    "Responda apenas com a transcrição corrigida."
)


def llm_correct(text: str, terms: list[str], llm) -> str:
    prompt = f"Termos do jogo: {', '.join(terms)}\n\nTranscrição:\n{text}"
    out = llm.generate(prompt, system=LLM_CORRECT_SYSTEM).text
    return out or text
