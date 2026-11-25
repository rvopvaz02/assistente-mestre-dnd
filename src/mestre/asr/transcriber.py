"""Transcrição com faster-whisper (CTranslate2). Usa GPU se houver; cai para CPU sem quebrar."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from mestre import config

log = logging.getLogger(__name__)


def _preload_cuda_dlls() -> None:
    """No Windows, importar o torch registra as DLLs de cuBLAS/cuDNN que o CTranslate2 precisa."""
    try:
        import torch  # noqa: F401
    except Exception:  # pragma: no cover
        pass


@dataclass
class Transcription:
    text: str
    seconds_audio: float
    seconds_processing: float
    device: str


class Transcriber:
    def __init__(self, model_size: str | None = None, device: str | None = None):
        from faster_whisper import WhisperModel

        model_size = model_size or config.WHISPER_MODEL
        device = device or config.WHISPER_DEVICE
        self.model_size = model_size
        attempts = []
        if device in ("auto", "cuda"):
            attempts.append(("cuda", "float16"))
        if device in ("auto", "cpu"):
            attempts.append(("cpu", "int8"))
        last_err: Exception | None = None
        for dev, ctype in attempts:
            try:
                if dev == "cuda":
                    _preload_cuda_dlls()
                self.model = WhisperModel(model_size, device=dev, compute_type=ctype)
                # Força uma inferência mínima para detectar erro de DLL/arquitetura já aqui.
                if dev == "cuda":
                    import numpy as np
                    list(self.model.transcribe(np.zeros(16000, dtype="float32"), language="pt")[0])
                self.device = dev
                log.info("Whisper %s carregado em %s (%s)", model_size, dev, ctype)
                return
            except Exception as e:  # noqa: BLE001
                log.warning("Falha ao carregar Whisper em %s: %s", dev, e)
                last_err = e
        raise RuntimeError(f"Não foi possível carregar o Whisper: {last_err}")

    def transcribe(self, audio: str | Path, *, prompt: str | None = None,
                   hotwords: str | None = None, language: str = "pt") -> Transcription:
        t0 = time.perf_counter()
        segments, info = self.model.transcribe(
            str(audio),
            language=language,
            initial_prompt=prompt,
            hotwords=hotwords,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return Transcription(text, info.duration, time.perf_counter() - t0, self.device)
