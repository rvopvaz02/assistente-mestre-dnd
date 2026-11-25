"""Configuração central. Tudo pode ser sobrescrito por variáveis de ambiente ou pelo arquivo .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# --- LLM (Gemini) ---
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-3.5-flash-lite")

# --- Voz ---
WHISPER_MODEL = _env("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = _env("WHISPER_DEVICE", "auto")  # auto | cuda | cpu
TTS_VOICE = _env("TTS_VOICE", "pt-BR-AntonioNeural")
VOCAB_TERMS = DATA_DIR / "vocab" / "termos_dnd.txt"
VOCAB_CORRECTIONS = DATA_DIR / "vocab" / "correcoes.csv"
AUDIO_DIR = DATA_DIR / "audio"

# --- RAG ---
CORPUS_MANIFEST = DATA_DIR / "corpus" / "manifest.json"
CHUNKS_PATH = DATA_DIR / "processed" / "chunks.jsonl"
CHROMA_DIR = DATA_DIR / "processed" / "chroma"
EMBED_MODEL = _env("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
RERANK_MODEL = _env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
CHUNK_MAX_CHARS = int(_env("CHUNK_MAX_CHARS", "1200"))
# Limiar do portão de abstenção (score do reranker, 0..1). Calibrado em scripts/eval_rag.py no split "dev".
ABSTAIN_THRESHOLD = float(_env("ABSTAIN_THRESHOLD", "0.017"))
