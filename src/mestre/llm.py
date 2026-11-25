"""Cliente de LLM. Gemini por padrão; qualquer objeto com `generate()` serve (útil para testes)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Protocol

from mestre import config


class LLM(Protocol):
    def generate(self, prompt: str, *, system: str | None = None,
                 images: list[tuple[bytes, str]] | None = None) -> "LLMResult": ...


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    model: str = ""
    meta: dict = field(default_factory=dict)


class GeminiLLM:
    """Wrapper fino sobre o SDK `google-genai`. Requer GEMINI_API_KEY no ambiente ou no .env."""

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        from google import genai  # import tardio: o resto do projeto roda sem a chave

        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Defina GEMINI_API_KEY no arquivo .env (veja .env.example).")
        self._client = genai.Client(api_key=key)
        self.model = model or config.GEMINI_MODEL
        self.temperature = temperature

    def generate(self, prompt: str, *, system: str | None = None,
                 images: list[tuple[bytes, str]] | None = None) -> LLMResult:
        from google.genai import types

        parts: list = [types.Part.from_bytes(data=b, mime_type=m) for b, m in (images or [])]
        parts.append(prompt)
        cfg = types.GenerateContentConfig(system_instruction=system, temperature=self.temperature)
        t0 = time.perf_counter()
        resp = self._call_with_retry(parts, cfg)
        dt = time.perf_counter() - t0
        usage = getattr(resp, "usage_metadata", None)
        return LLMResult(
            text=(resp.text or "").strip(),
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            latency_s=dt,
            model=self.model,
            meta={"tentativas": self._last_attempts},
        )

    def _call_with_retry(self, parts, cfg, attempts: int = 5, base_delay: float = 2.0):
        """Repete em 429 (cota) e 5xx (sobrecarga) com backoff exponencial; outros erros sobem na hora."""
        from google.genai import errors

        for i in range(attempts):
            try:
                self._last_attempts = i + 1
                return self._client.models.generate_content(model=self.model, contents=parts, config=cfg)
            except errors.APIError as e:
                retriable = e.code == 429 or (e.code or 0) >= 500
                # Cota diária esgotada (free tier: poucas dezenas de chamadas/dia): repetir só gasta tempo.
                if e.code == 429 and "PerDay" in str(e):
                    raise QuotaExhausted(str(e)) from e
                if not retriable or i == attempts - 1:
                    raise
                time.sleep(base_delay * 2 ** i)


class QuotaExhausted(RuntimeError):
    """Cota diária do provedor esgotada."""
