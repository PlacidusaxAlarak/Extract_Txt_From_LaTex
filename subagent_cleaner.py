# -*- coding: utf-8 -*-
"""LLM-based cleanup for flattened LaTeX content."""

from __future__ import annotations

import logging
import re
from typing import Iterable, List, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

logger = logging.getLogger(__name__)


def _chunk_text(text: str, max_chars: int) -> List[str]:
    if not text:
        return []
    max_chars = max(200, int(max_chars))

    parts = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for part in parts:
        if part is None:
            continue
        part = part.strip()
        if not part:
            continue

        if len(part) > max_chars:
            if buf:
                chunks.append("\n\n".join(buf))
                buf = []
                buf_len = 0
            for idx in range(0, len(part), max_chars):
                chunk = part[idx : idx + max_chars]
                if chunk.strip():
                    chunks.append(chunk)
            continue

        extra = len(part) + (2 if buf else 0)
        if buf_len + extra > max_chars and buf:
            chunks.append("\n\n".join(buf))
            buf = [part]
            buf_len = len(part)
        else:
            if buf:
                buf_len += 2 + len(part)
            else:
                buf_len = len(part)
            buf.append(part)

    if buf:
        chunks.append("\n\n".join(buf))

    return chunks


def _build_system_prompt() -> str:
    return (
        "You clean LaTeX into plain text. Output only cleaned text. "
        "Do not add explanations or Markdown. Keep math in $...$ or $$...$$ when present. "
        "Remove LaTeX commands and formatting but preserve readable content."
    )


def clean_latex_with_llm(text: str, config) -> Optional[str]:
    """Return cleaned text via LLM, or None on failure."""
    if not text:
        return None
    if OpenAI is None:
        logger.warning("openai package not available; skipping LLM cleanup.")
        return None

    max_chars = int(getattr(config, "SUBAGENT_MAX_CHARS", 200000))
    if len(text) > max_chars:
        logger.warning("Input too large for LLM cleanup; skipping.")
        return None

    backend = str(getattr(config, "SUBAGENT_BACKEND", "vllm")).lower()
    if backend in {"none", "off", "disabled"}:
        return None

    base_url = getattr(config, "SUBAGENT_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = getattr(config, "SUBAGENT_API_KEY", "EMPTY")
    model = getattr(config, "SUBAGENT_MODEL", "Qwen2.5-7B-Instruct")
    timeout_sec = int(getattr(config, "SUBAGENT_TIMEOUT_SEC", 120))
    chunk_chars = int(getattr(config, "SUBAGENT_CHUNK_CHARS", 8000))

    if backend == "openai":
        client = OpenAI(api_key=api_key, timeout=timeout_sec)
    else:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_sec)
    system_prompt = _build_system_prompt()

    chunks = _chunk_text(text, chunk_chars)
    if not chunks:
        return None

    outputs: List[str] = []
    for chunk in chunks:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk},
                ],
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning(f"LLM cleanup failed: {exc}")
            return None

        content = response.choices[0].message.content if response.choices else ""
        if content:
            cleaned = content.strip()
            if cleaned:
                outputs.append(cleaned)

    if not outputs:
        return None
    return "\n\n".join(outputs).strip()
