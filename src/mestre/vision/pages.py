"""Leitura visual de páginas do PDF, só onde o texto extraído não basta.

Pipeline
  1. `select_pages`  : índice de elementos visuais. Heurísticas baratas sobre o texto extraído decidem
                       quais páginas precisam de visão (tabela, capitular, página sem texto) e por quê.
  2. `read_page`     : renderiza a página e pede ao LLM multimodal um JSON com tabelas, títulos como
                       impressos e figuras relevantes para regras.
  3. cache em disco  : chave = hash(imagem + modelo + versão do prompt). Reprocessar o corpus não paga
                       a visão de novo; mudar o prompt invalida o cache de propósito.
  4. `visual_chunks` / `capitular_fixes`: o que a visão leu vira trechos "visao" no RAG e correções
                       nos trechos de texto da mesma página.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mestre import config

PROMPT_VERSION = "v1"
CACHE_DIR = config.DATA_DIR / "processed" / "visao"

SYSTEM = """Você lê páginas digitalizadas do Livro do Jogador de D&D 5e em português.
Devolva SOMENTE um JSON com este formato:
{
  "tipo_pagina": "texto" | "texto_com_tabela" | "ilustracao",
  "tabelas": [{"titulo": str, "colunas": [str], "linhas": [[str]]}],
  "titulos": [str],
  "capitulares": [{"palavra_completa": str}],
  "figuras": [{"descricao": str, "relevante_para_regras": bool}]
}
Regras:
- "tabelas": transcreva cada tabela célula a célula, exatamente como impressa (sinais como "–" e "+" inclusos).
  Uma tabela com dois blocos lado a lado (ex.: Valor/Modificador repetido) deve virar uma única lista de linhas.
- "titulos": títulos de seção exatamente como impressos.
- "capitulares": quando um parágrafo começa com letra decorativa grande (capitular), escreva a primeira
  palavra completa incluindo a capitular, ex.: "Seis".
- "figuras": ilustrações. "relevante_para_regras" só é true se a figura traz informação de regra
  (diagrama, área de efeito, mapa com medidas). Arte decorativa é false.
- Não invente: se não houver tabelas, use [].
"""


@dataclass
class PageSelection:
    pdf_page: int
    motivos: list[str]


@dataclass
class PageReading:
    fonte: str
    pdf_page: int
    pagina: int | None
    modelo: str
    leitura: dict
    tokens_entrada: int
    tokens_saida: int
    latencia_s: float
    do_cache: bool = False
    motivos: list[str] = field(default_factory=list)


# ---------- 1. seleção de páginas (índice de elementos visuais) ----------

def select_pages(pdf_path: Path, ranges: list[list[int]]) -> list[PageSelection]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    out = []
    for a, b in ranges:
        for p in range(a, b + 1):
            text = reader.pages[p - 1].extract_text() or ""
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            motivos = []
            if len(lines) <= 3:
                motivos.append("página sem texto extraível")
            # Linha de tabela: curta e terminando em número/traço ("Difícil 20"), ou com 3+ números
            # ("Rápido 120 metros 6 km 45 km"). A 1ª linha é o número da página e fica de fora.
            def is_row(l: str) -> bool:
                nums = re.findall(r"[–\-+]?\d+(?:[,.]\d+)?", l)
                short = len(l) < 45 and not l.endswith(".")
                return (short and (re.search(r"(\d|–)\s*$", l) is not None or len(nums) >= 2)) or len(nums) >= 3

            rows = [l for l in lines[1:] if is_row(l)]
            if len(rows) >= 4 or (re.search(r"\btabela\b", text, re.I) and len(rows) >= 3):
                motivos.append(f"tabela provável ({len(rows)} linhas tabulares)")
            if re.search(r"CAP[ÍI]TULO\s+\d+", text[:300]):
                motivos.append("abertura de capítulo (capitular)")
            if motivos:
                out.append(PageSelection(p, motivos))
    return out


# ---------- 2/3. leitura com cache ----------

def render_page(pdf_path: Path, pdf_page: int, scale: float = 1.6) -> bytes:
    import pypdfium2 as pdfium

    img = pdfium.PdfDocument(str(pdf_path))[pdf_page - 1].render(scale=scale).to_pil()
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


def read_page(pdf_path: Path, fonte: str, pdf_page: int, llm, *, pagina: int | None = None,
              motivos: list[str] | None = None, cache_dir: Path = CACHE_DIR) -> PageReading:
    image = render_page(pdf_path, pdf_page)
    key = hashlib.sha256(image + llm.model.encode() + PROMPT_VERSION.encode()).hexdigest()[:16]
    cache = cache_dir / f"{fonte}_p{pdf_page:03d}_{key}.json"
    if cache.exists():
        r = PageReading(**json.loads(cache.read_text(encoding="utf-8")))
        r.do_cache = True
        return r
    res = llm.generate("Leia esta página.", system=SYSTEM, images=[(image, "image/jpeg")], json_mode=True)
    reading = PageReading(fonte, pdf_page, pagina, llm.model, _parse_json(res.text), res.input_tokens,
                          res.output_tokens, round(res.latency_s, 3), False, motivos or [])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(asdict(reading), ensure_ascii=False, indent=1), encoding="utf-8")
    return reading


def load_readings(fonte: str, cache_dir: Path = CACHE_DIR, model: str | None = None) -> dict[int, PageReading]:
    """Leituras já feitas do modelo de visão configurado (a mais recente por página). Não chama API."""
    model = model or config.VISION_PAGES_MODEL
    out: dict[int, PageReading] = {}
    if not cache_dir.exists():
        return out
    for f in sorted(cache_dir.glob(f"{fonte}_p*.json"), key=lambda p: p.stat().st_mtime):
        r = PageReading(**json.loads(f.read_text(encoding="utf-8")))
        if r.modelo == model:
            out[r.pdf_page] = r
    return out


# ---------- 4. integração com o RAG ----------

def table_to_text(t: dict) -> str:
    """Tabela -> frases "coluna: valor", que o BM25, o denso e o LLM leem sem ambiguidade de colunas."""
    cols = t.get("colunas") or []
    rows = []
    for row in t.get("linhas") or []:
        if cols and len(cols) == len(row):
            rows.append("; ".join(f"{c}: {v}" for c, v in zip(cols, row)))
        else:
            rows.append(" | ".join(map(str, row)))
    return f"Tabela {t.get('titulo', '').strip()}. " + ". ".join(rows) + "."


def visual_texts(reading: PageReading) -> list[tuple[str, str]]:
    """(seção, texto) de cada elemento visual que vale indexar."""
    out = []
    for t in reading.leitura.get("tabelas") or []:
        if t.get("linhas"):
            out.append((f"Tabela: {t.get('titulo', '').strip().title()}", table_to_text(t)))
    for f in reading.leitura.get("figuras") or []:
        if f.get("relevante_para_regras") and f.get("descricao"):
            out.append(("Figura", f["descricao"]))
    return out


def capitular_fixes(reading: PageReading, extracted_text: str) -> list[tuple[str, str]]:
    """Pares (como saiu no texto, correto) para palavras cuja capitular sumiu na extração.

    Ex.: visão lê "Seis"; o texto extraído começa com "EIS HABILIDADES" -> ("EIS", "SEIS").
    Só aceita a correção se a palavra extraída for exatamente a palavra completa sem a 1ª letra.
    """
    fixes = []
    words = re.findall(r"[A-Za-zÀ-ÿ]+", extracted_text)
    for c in reading.leitura.get("capitulares") or []:
        full = (c.get("palavra_completa") or "").strip()
        if not full:
            continue
        if len(full) == 1:
            # Capitular que é uma palavra inteira ("O barulho", "A magia"): some sem deixar rastro.
            if words and words[0].lower() != full.lower():
                fixes.append((words[0], f"{full} {words[0]}"))
            continue
        cut = full[1:]
        for w in words:
            if w.lower() == cut.lower() and w.lower() != full.lower():
                fixes.append((w, full.upper() if w.isupper() else full))
                break
    return fixes
