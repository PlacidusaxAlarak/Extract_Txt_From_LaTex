# -*- coding: utf-8 -*-
"""LLM-only extractor (optional dependency)."""

from __future__ import annotations

import logging
from pathlib import Path

import config
import pandoc_parser
import subagent_cleaner

logger = logging.getLogger(__name__)


def _prepare_chunks(main_tex_file: Path):
    return pandoc_parser._prepare_llm_source_chunks_raw(main_tex_file)


def parse_project_to_text(main_tex_file: Path) -> str:
    if not main_tex_file.exists():
        raise FileNotFoundError(f"Main .tex file not found: {main_tex_file}")

    logger.info("Collecting LaTeX project chunks by source .tex files...")
    source_chunks = _prepare_chunks(main_tex_file)
    if not source_chunks:
        raise RuntimeError("No source chunks collected from LaTeX project.")

    logger.info("Running LLM cleanup file-by-file...")
    llm_text = subagent_cleaner.clean_latex_chunks_with_llm(source_chunks, config)
    if not llm_text:
        raise RuntimeError("LLM cleanup failed or produced no output.")

    pandoc_parser._write_debug_text("llm_only", main_tex_file, llm_text)
    return llm_text.strip()
