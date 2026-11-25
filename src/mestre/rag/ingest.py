"""Ingestão governada: PDF/Markdown -> seções -> chunks com metadados de versão, vigência e permissão.

O chunking respeita fronteiras naturais: primeiro por título de seção (no livro, linhas em CAIXA ALTA;
no Markdown, linhas '#'), e só quebra uma seção longa entre frases. Uma regra nunca é cortada no meio
de uma frase.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from mestre import config

ENCLITICS = {"lo", "la", "los", "las", "se", "me", "te", "lhe", "lhes", "nos", "vos", "o", "a", "os", "as"}


@dataclass
class Chunk:
    id: str
    texto: str
    fonte: str
    titulo_fonte: str
    versao: str
    vigente_desde: str | None
    vigente_ate: str | None
    permissao: str
    capitulo: str
    secao: str
    pagina: int | None  # número impresso no livro (o que o Mestre vai procurar)
    pagina_pdf: int | None
    modalidade: str = "texto"  # "texto" (extraído do PDF) | "visao" (lido da imagem da página)

    def citation(self) -> str:
        pg = f", p. {self.pagina}" if self.pagina else ""
        vis = " (leitura visual)" if self.modalidade == "visao" else ""
        return f"{self.titulo_fonte}, {self.secao}{pg}{vis}"


# ---------- limpeza de texto ----------

def _join_lines(lines: list[str]) -> str:
    """Junta linhas quebradas pelo layout de duas colunas, tratando hifenização."""
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if out.endswith("-"):
            m = re.match(r"(\w+)", ln)
            nxt = m.group(1).lower() if m else ""
            # "empurrá-" + "lo" mantém o hífen; "conjura-" + "ção" é hifenização de layout
            out = out + ln if nxt in ENCLITICS else out[:-1] + ln
        else:
            out = (out + " " + ln) if out else ln
    return re.sub(r"\s+", " ", out).strip()


def _is_heading(line: str) -> bool:
    s = line.strip()
    letters = re.sub(r"[^A-Za-zÀ-ÿ]", "", s)
    return 3 <= len(s) <= 60 and len(letters) >= 3 and s.upper() == s and not re.match(r"^\d+$", s)


def _split_sentences(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sents = re.split(r"(?<=[.!?:])\s+(?=[A-ZÀ-Ý0-9•\-])", text)
    parts, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > max_chars:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


# ---------- leitores ----------

def _sections_from_pdf(path: Path, ranges: list[list[int]], chapters: dict[str, str]):
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    chap_starts = sorted((int(k), v) for k, v in chapters.items())
    sec_title, sec_lines, sec_page, sec_pdf, chapter = "Introdução", [], None, None, ""
    for a, b in ranges:
        for pdf_pg in range(a, b + 1):
            for start, name in chap_starts:
                if pdf_pg >= start:
                    chapter = name
            raw = reader.pages[pdf_pg - 1].extract_text() or ""
            lines = raw.splitlines()
            printed = None
            if lines and re.fullmatch(r"\s*\d{1,3}\s*", lines[0]):
                printed = int(lines[0])
                lines = lines[1:]
            for ln in lines:
                if _is_heading(ln):
                    if sec_lines:
                        yield chapter, sec_title, sec_page, sec_pdf, _join_lines(sec_lines)
                    sec_title, sec_lines, sec_page, sec_pdf = ln.strip().title(), [], printed, pdf_pg
                else:
                    if not sec_lines:
                        sec_page, sec_pdf = printed, pdf_pg
                    sec_lines.append(ln)
    if sec_lines:
        yield chapter, sec_title, sec_page, sec_pdf, _join_lines(sec_lines)


def _sections_from_md(path: Path):
    chapter, title, buf = "", "", []
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(#+)\s+(.*)", ln)
        if m:
            if buf:
                yield chapter, title, None, None, _join_lines(buf)
                buf = []
            if len(m.group(1)) == 1:
                chapter = m.group(2).strip()
            title = m.group(2).strip()
        else:
            buf.append(ln)
    if buf:
        yield chapter, title, None, None, _join_lines(buf)


# ---------- pipeline ----------

def build_chunks(manifest_path: Path | None = None, max_chars: int | None = None,
                 visao: bool = True) -> list[Chunk]:
    """Gera os chunks. Com `visao=True`, usa as leituras visuais já em cache (não chama API):
    acrescenta trechos de tabelas/figuras e corrige títulos que perderam a capitular."""
    from mestre.vision.pages import capitular_fixes, load_readings, visual_texts

    manifest_path = manifest_path or config.CORPUS_MANIFEST
    max_chars = max_chars or config.CHUNK_MAX_CHARS
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    for src in manifest["fontes"]:
        path = Path(src["caminho"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if not path.exists():
            print(f"Aviso: fonte '{src['id']}' não encontrada em {path}; pulando.")
            continue
        readings = load_readings(src["id"]) if visao else {}
        if path.suffix.lower() == ".pdf":
            sections = _sections_from_pdf(path, src["paginas"], src.get("capitulos", {}))
        else:
            sections = _sections_from_md(path)
        base = dict(fonte=src["id"], titulo_fonte=src["titulo"], versao=src["versao"],
                    vigente_desde=src.get("vigente_desde"), vigente_ate=src.get("vigente_ate"),
                    permissao=src.get("permissao", "jogadores"))
        chapter_of_page: dict[int, tuple[str, int | None]] = {}
        for chapter, title, page, pdf_page, text in sections:
            # A capitular abre só a 1ª seção da página (a que vem logo após o título do capítulo).
            if pdf_page in readings and pdf_page not in chapter_of_page:
                for wrong, right in capitular_fixes(readings[pdf_page], title):
                    if title.lower().startswith(wrong.lower()):
                        title = right.title() + title[len(wrong):]
            if pdf_page:
                chapter_of_page.setdefault(pdf_page, (chapter, page))
            if len(text) < 40:
                continue
            for i, part in enumerate(_split_sentences(text, max_chars)):
                h = hashlib.sha1(f"{src['id']}|{src['versao']}|{title}|{pdf_page}|{i}|{part[:80]}".encode()).hexdigest()[:12]
                chunks.append(Chunk(id=f"{src['id']}-{h}", texto=part, capitulo=chapter, secao=title,
                                    pagina=page, pagina_pdf=pdf_page, **base))
        for pdf_page, r in sorted(readings.items()):
            chapter, page = chapter_of_page.get(pdf_page, ("", r.pagina))
            for j, (secao, texto) in enumerate(visual_texts(r)):
                h = hashlib.sha1(f"{src['id']}|visao|{pdf_page}|{j}|{texto[:80]}".encode()).hexdigest()[:12]
                chunks.append(Chunk(id=f"{src['id']}-v{h}", texto=texto, capitulo=chapter, secao=secao,
                                    pagina=r.pagina or page, pagina_pdf=pdf_page, modalidade="visao", **base))
    return chunks


def save_chunks(chunks: list[Chunk], path: Path | None = None) -> Path:
    path = path or config.CHUNKS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    return path


def load_chunks(path: Path | None = None) -> list[Chunk]:
    path = path or config.CHUNKS_PATH
    return [Chunk(**json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
