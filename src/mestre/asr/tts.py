"""Resposta falada com edge-tts (vozes neurais pt-BR, sem custo, requer internet)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from mestre import config


async def _speak(text: str, out: Path, voice: str) -> None:
    import edge_tts

    await edge_tts.Communicate(text, voice).save(str(out))


def speak(text: str, out: str | Path, voice: str | None = None) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(_speak(text, out, voice or config.TTS_VOICE))
    return out
