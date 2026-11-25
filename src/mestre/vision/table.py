"""Leitura de uma foto da mesa: dados rolados, miniaturas e mapa tático."""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

SYSTEM = """Você olha fotos de uma mesa de RPG (D&D 5e) e descreve o estado do jogo.
Devolva SOMENTE um JSON:
{
  "dados": [{"tipo": "d4|d6|d8|d10|d12|d20|d100", "valor": int, "cor": str, "confianca": "alta|media|baixa"}],
  "miniaturas": [{"descricao": str, "posicao": str}],
  "mapa": {"tem_grade": bool, "descricao": str},
  "observacoes": str
}
Regras:
- "dados": só dados já parados. "valor" é a face voltada para cima. No d4, é o número na base
  ou no vértice de cima, conforme o modelo do dado. Não chute: se não der para ler, use confianca "baixa".
- O tipo do dado vem da forma (d4 pirâmide, d6 cubo, d8 octaedro, d10 pião, d12 dodecaedro, d20 icosaedro).
- "posicao": se houver grade, use coordenada estilo xadrez (coluna letra, linha número a partir do canto
  superior esquerdo da foto); senão, descreva ("perto da porta", "canto inferior direito").
- Se algo não estiver na foto, devolva lista vazia; não invente.
"""


@dataclass
class TableReading:
    leitura: dict
    tokens_entrada: int
    tokens_saida: int
    latencia_s: float


def _load_image(path: str | Path, max_side: int = 1600) -> bytes:
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def read_table(photo: str | Path, llm) -> TableReading:
    image = _load_image(photo)
    r = llm.generate("Descreva o estado desta mesa.", system=SYSTEM, images=[(image, "image/jpeg")], json_mode=True)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", r.text.strip())
    return TableReading(json.loads(text), r.input_tokens, r.output_tokens, r.latency_s)


def dice_multiset(reading: dict) -> list[str]:
    """['d20:17', 'd6:3'] ordenado, para comparar com o gabarito."""
    return sorted(f"{d.get('tipo', '').lower()}:{d.get('valor')}" for d in reading.get("dados") or [])
