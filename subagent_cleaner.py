# -*- coding: utf-8 -*-
"""LLM-based cleanup for LaTeX content."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

logger = logging.getLogger(__name__)


def _resolve_cleanup_log_file(config) -> Optional[Path]:
    enabled = str(getattr(config, "SUBAGENT_CLEAN_LOG_ENABLED", "1")).strip().lower()
    if enabled in {"0", "false", "no", "off", "disabled"}:
        return None

    log_dir_raw = str(getattr(config, "SUBAGENT_CLEAN_LOG_DIR", "logs"))
    log_file_name = str(getattr(config, "SUBAGENT_CLEAN_LOG_FILE", "llm_cleanup.log"))

    log_dir = Path(log_dir_raw)
    if not log_dir.is_absolute():
        log_dir = Path(__file__).resolve().parent / log_dir

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning(f"Failed to create cleanup log directory {log_dir}: {exc}")
        return None

    return log_dir / log_file_name


def _append_cleanup_log(log_file: Optional[Path], text: str) -> None:
    if log_file is None:
        return
    try:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
    except Exception as exc:
        logger.warning(f"Failed to write cleanup log {log_file}: {exc}")


def _build_system_prompt() -> str:
    return (
        "You are a strict LaTeX cleaner. Output only cleaned text content. "
        "Do not output explanations or code fences. "
        "Keep Markdown headings if they already exist in the text. "
        "Rule 1: Remove citation/cross-reference/bibliography tokens such as \\cite*, \\ref, \\eqref, \\label, \\bibliography. "
        "Rule 2: Remove image information completely: delete \\includegraphics, figure environment wrappers, image file paths, width/height/alignment options, and rendering-only image commands. "
        "If a figure caption exists, keep only one plain line like 'Figure: <caption>'. "
        "Rule 3: Preserve table layout and tabular formatting/content; do not flatten or discard table structure. "
        "Rule 4: Remove custom macro declarations and custom macro symbols (for example \\newcommand, \\def, \\renewcommand, \\spmath, \\RR, \\NN). "
        "If a custom macro clearly represents a plain word, keep the word without the backslash; otherwise delete the macro token. "
        "Rule 5: Preserve readable prose and equations in $...$ or $$...$$ when possible while removing non-semantic command shells."
    )


def _build_client_and_model(config):
    if OpenAI is None:
        logger.warning("openai package not available; skipping LLM cleanup.")
        return None, None

    backend = str(getattr(config, "SUBAGENT_BACKEND", "vllm")).lower()
    if backend in {"none", "off", "disabled"}:
        return None, None

    base_url = getattr(config, "SUBAGENT_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = getattr(config, "SUBAGENT_API_KEY", "EMPTY")
    model = getattr(config, "SUBAGENT_MODEL", "Qwen2.5-7B-Instruct")
    timeout_sec = int(getattr(config, "SUBAGENT_TIMEOUT_SEC", 120))

    try:
        if backend == "openai":
            client = OpenAI(api_key=api_key, timeout=timeout_sec)
        else:
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_sec)
    except Exception as exc:
        logger.warning(f"Failed to initialize LLM client: {exc}")
        return None, None

    return client, model


def clean_latex_chunks_with_llm(
    chunks: Sequence[Tuple[str, str]], config
) -> Optional[str]:
    """Return cleaned text by cleaning each source chunk with LLM."""
    if not chunks:
        return None

    normalized_chunks: List[Tuple[str, str]] = []
    for source_name, source_text in chunks:
        text = (source_text or "").strip()
        if text:
            normalized_chunks.append((source_name or "latex", text))

    if not normalized_chunks:
        return None

    log_file = _resolve_cleanup_log_file(config)
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
    _append_cleanup_log(
        log_file,
        (
            f"\n===== LLM CLEANUP RUN START {run_id} =====\n"
            f"chunk_count={len(normalized_chunks)}\n"
        ),
    )

    client, model = _build_client_and_model(config)
    if client is None or model is None:
        _append_cleanup_log(
            log_file,
            f"run={run_id} status=failed reason=client_or_model_unavailable\n"
            f"===== LLM CLEANUP RUN END {run_id} =====\n",
        )
        return None

    system_prompt = _build_system_prompt()

    def _run_llm_once(prompt: str, payload: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": payload},
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content if response.choices else ""
        return content.strip() if content else ""

    outputs: List[str] = []
    total = len(normalized_chunks)
    for index, (source_name, source_text) in enumerate(normalized_chunks, start=1):
        try:
            cleaned = _run_llm_once(system_prompt, source_text)
            if not cleaned:
                # Second attempt: force coverage instead of dropping content.
                retry_prompt = (
                    system_prompt
                    + " IMPORTANT: Do not omit content. If cleaning would remove everything, "
                    + "return the input unchanged."
                )
                cleaned = _run_llm_once(retry_prompt, source_text)
        except Exception as exc:
            logger.warning(f"LLM cleanup failed on source '{source_name}': {exc}")
            _append_cleanup_log(
                log_file,
                (
                    f"run={run_id} status=failed chunk={index}/{total} source={source_name} "
                    f"input_chars={len(source_text)} error={exc}\n"
                    f"===== LLM CLEANUP RUN END {run_id} =====\n"
                ),
            )
            return None

        if cleaned:
            outputs.append(cleaned)
            _append_cleanup_log(
                log_file,
                (
                    f"run={run_id} status=ok chunk={index}/{total} source={source_name} "
                    f"input_chars={len(source_text)} output_chars={len(cleaned)}\n"
                    f"--- CLEANED RESULT START ({source_name}) ---\n"
                    f"{cleaned}\n"
                    f"--- CLEANED RESULT END ({source_name}) ---\n"
                ),
            )
        else:
            _append_cleanup_log(
                log_file,
                (
                    f"run={run_id} status=empty chunk={index}/{total} source={source_name} "
                    f"input_chars={len(source_text)}\n"
                ),
            )

    if not outputs:
        _append_cleanup_log(
            log_file,
            f"run={run_id} status=failed reason=no_outputs\n"
            f"===== LLM CLEANUP RUN END {run_id} =====\n",
        )
        return None

    merged = "\n\n".join(outputs).strip()
    _append_cleanup_log(
        log_file,
        (
            f"run={run_id} status=success merged_output_chars={len(merged)}\n"
            f"--- MERGED RESULT START ---\n"
            f"{merged}\n"
            f"--- MERGED RESULT END ---\n"
            f"===== LLM CLEANUP RUN END {run_id} =====\n"
        ),
    )
    return merged


def clean_latex_with_llm(text: str, config, force_chunks: bool = False) -> Optional[str]:
    """Backward-compatible wrapper that cleans a single text payload."""
    _ = force_chunks
    if not text:
        return None
    return clean_latex_chunks_with_llm([("flattened.tex", text)], config)
