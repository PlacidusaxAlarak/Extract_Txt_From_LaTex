# -*- coding: utf-8 -*-
"""plasTeX-based extractor (optional dependency)."""

from __future__ import annotations

import logging
import re
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import config
import pandoc_parser

logger = logging.getLogger(__name__)


def _node_name(node) -> str:
    return (getattr(node, "nodeName", "") or getattr(node, "tagName", "") or "").lower()


def _node_text(node) -> str:
    text_content = getattr(node, "textContent", None)
    if isinstance(text_content, str):
        return " ".join(text_content.split()).strip()
    children = getattr(node, "childNodes", None)
    if not children:
        return ""
    parts = []
    for child in children:
        child_text = _node_text(child)
        if child_text:
            parts.append(child_text)
    return " ".join(parts).strip()


def _get_section_title(node) -> str:
    for child in getattr(node, "childNodes", []) or []:
        if _node_name(child) == "title":
            return _node_text(child)
    return ""


def _walk(node, state: Dict[str, Optional[int]], lines: List[str], abstract_holder: Dict[str, str]) -> None:
    if state.get("stop"):
        return

    name = _node_name(node)

    if name in {"thebibliography", "bibliography"}:
        return

    if name in {"appendix"}:
        state["stop"] = True
        return

    if name == "abstract":
        if config.KEEP_ABSTRACT and not abstract_holder.get("text"):
            abstract_holder["text"] = _node_text(node)
        return

    section_levels = {
        "part": 0,
        "chapter": 0,
        "section": 1,
        "subsection": 2,
        "subsubsection": 3,
        "paragraph": 4,
        "subparagraph": 5,
    }

    if name in section_levels:
        level = section_levels[name]
        title = _get_section_title(node)

        if state.get("skip_level") is not None and level <= state["skip_level"]:
            state["skip_level"] = None

        if state.get("skip_level") is not None:
            return

        if title and pandoc_parser._is_excluded_section(title):
            state["skip_level"] = level
            return

        if title:
            lines.append(title)
            lines.append("")

    if state.get("skip_level") is not None:
        return

    if name in {"p", "para", "paragraph"}:
        text = _node_text(node)
        if text:
            lines.append(text)
            lines.append("")
        return

    if name in {"math", "equation", "displaymath"} and config.KEEP_MATH:
        text = _node_text(node)
        if text:
            lines.append(text)
            lines.append("")
        return

    if name in {"footnote", "note"} and config.KEEP_FOOTNOTES:
        text = _node_text(node)
        if text:
            lines.append(f"[Footnote: {text}]")
            lines.append("")
        return

    for child in getattr(node, "childNodes", []) or []:
        _walk(child, state, lines, abstract_holder)


def parse_project_to_text(main_tex_file: Path) -> str:
    try:
        from plasTeX.TeX import TeX
    except Exception as e:
        raise RuntimeError("plasTeX is required for the plastex engine.") from e

    if not main_tex_file.exists():
        raise FileNotFoundError(f"Main .tex file not found: {main_tex_file}")

    logger.info("Flattening LaTeX project for plasTeX...")
    flattened_content = pandoc_parser._resolve_and_flatten_tex(main_tex_file)
    flattened_content = pandoc_parser._strip_latex_comments(flattened_content)
    flattened_content = pandoc_parser._remove_bibliography_content(flattened_content)
    flattened_content = pandoc_parser._truncate_at_appendix(flattened_content)
    flattened_content = pandoc_parser._strip_cjk_envs(flattened_content)

    match_begin = re.search(r"\\begin\{document\}", flattened_content, flags=re.IGNORECASE)
    preamble = flattened_content[:match_begin.start()] if match_begin else ""
    body = flattened_content[match_begin.end() :] if match_begin else flattened_content

    title = pandoc_parser._extract_command_arg(preamble, "title") if config.KEEP_TITLE else ""
    preamble_abstract = ""
    if config.KEEP_ABSTRACT:
        match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", preamble, flags=re.DOTALL | re.IGNORECASE)
        if match:
            preamble_abstract = match.group(1).strip()

    logger.info("Parsing LaTeX with plasTeX...")
    tex = TeX()
    buffer = StringIO(body)
    buffer.name = str(main_tex_file.name)
    tex.input(buffer)
    document = tex.parse()

    lines: List[str] = []
    abstract_holder: Dict[str, str] = {"text": ""}
    state: Dict[str, Optional[int]] = {"skip_level": None, "stop": False}

    _walk(document, state, lines, abstract_holder)

    output_parts: List[str] = []
    if title:
        output_parts.append(f"Title: {title}")
    abstract_text = abstract_holder.get("text") or preamble_abstract
    if abstract_text and config.KEEP_ABSTRACT:
        output_parts.append(f"Abstract:\n{abstract_text}")
    if lines:
        output_parts.append("--- Body ---\n\n" + "\n".join(lines).strip())

    combined = "\n\n".join(output_parts)
    combined = pandoc_parser._postprocess_text(combined)
    logger.info("plasTeX extraction complete.")
    return combined
