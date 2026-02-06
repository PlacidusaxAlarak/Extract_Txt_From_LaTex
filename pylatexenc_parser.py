# -*- coding: utf-8 -*-
"""pylatexenc-based extractor (optional dependency)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import config
import pandoc_parser

logger = logging.getLogger(__name__)


def _get_arg_nodes(node) -> List:
    if not getattr(node, "nodeargd", None):
        return []
    argnlist = getattr(node.nodeargd, "argnlist", None)
    if not argnlist:
        return []
    return [arg.nodelist for arg in argnlist if arg is not None and getattr(arg, "nodelist", None) is not None]


def _nodes_to_text(nodes, state: Dict[str, Optional[int]]) -> str:
    try:
        from pylatexenc.latexwalker import (
            LatexCharsNode,
            LatexCommentNode,
            LatexEnvironmentNode,
            LatexGroupNode,
            LatexMacroNode,
            LatexMathNode,
            LatexSpecialsNode,
        )
    except Exception as e:
        raise RuntimeError("pylatexenc is required for the pylatexenc engine.") from e

    parts: List[str] = []

    section_levels = {
        "part": 0,
        "chapter": 0,
        "section": 1,
        "subsection": 2,
        "subsubsection": 3,
        "paragraph": 4,
        "subparagraph": 5,
    }

    for node in nodes:
        if state.get("stop"):
            break

        if isinstance(node, LatexCommentNode):
            continue

        if isinstance(node, LatexCharsNode):
            parts.append(node.chars)
            continue

        if isinstance(node, LatexSpecialsNode):
            parts.append(node.specials_chars)
            continue

        if isinstance(node, LatexMathNode):
            if config.KEEP_MATH:
                try:
                    parts.append(node.latex_verbatim())
                except Exception:
                    try:
                        parts.append(node.nodelist.latex_verbatim())
                    except Exception:
                        pass
            continue

        if isinstance(node, LatexGroupNode):
            parts.append(_nodes_to_text(node.nodelist, state))
            continue

        if isinstance(node, LatexMacroNode):
            name = node.macroname

            if name == "appendix":
                state["stop"] = True
                break

            if name in {"cite", "citet", "citep", "autocite", "ref", "eqref", "label"}:
                continue

            if name in section_levels:
                level = section_levels[name]
                arg_nodes = _get_arg_nodes(node)
                title = _nodes_to_text(arg_nodes[0], state) if arg_nodes else ""

                if state.get("skip_level") is not None and level <= state["skip_level"]:
                    state["skip_level"] = None

                if state.get("skip_level") is not None:
                    continue

                if title and pandoc_parser._is_excluded_section(title):
                    state["skip_level"] = level
                    continue

                if title:
                    parts.append("\n" + title + "\n\n")
                continue

            if state.get("skip_level") is not None:
                continue

            if name == "footnote" and config.KEEP_FOOTNOTES:
                arg_nodes = _get_arg_nodes(node)
                note_text = _nodes_to_text(arg_nodes[0], state) if arg_nodes else ""
                if note_text:
                    parts.append(f" [Footnote: {note_text}]")
                continue

            if name == "item":
                parts.append("\n- ")
                continue

            arg_nodes = _get_arg_nodes(node)
            for arg in arg_nodes:
                parts.append(_nodes_to_text(arg, state))
            continue

        if isinstance(node, LatexEnvironmentNode):
            env = node.environmentname

            if env in {"thebibliography", "bibliography"}:
                continue

            if env == "abstract":
                if config.KEEP_ABSTRACT and not state.get("abstract"):
                    state["abstract"] = _nodes_to_text(node.nodelist, state).strip()
                continue

            if env in {"equation", "align", "alignat", "eqnarray", "gather", "multline", "flalign"}:
                if config.KEEP_MATH:
                    try:
                        parts.append(node.latex_verbatim())
                    except Exception:
                        parts.append(_nodes_to_text(node.nodelist, state))
                continue

            if env in {"figure", "table", "tabular", "tabularx", "longtable", "algorithm", "algorithmic", "tikzpicture"}:
                continue

            if state.get("skip_level") is not None:
                continue

            parts.append(_nodes_to_text(node.nodelist, state))
            continue

    return "".join(parts)


def parse_project_to_text(main_tex_file: Path) -> str:
    try:
        from pylatexenc.latexwalker import LatexWalker
    except Exception as e:
        raise RuntimeError("pylatexenc is required for the pylatexenc engine.") from e

    if not main_tex_file.exists():
        raise FileNotFoundError(f"Main .tex file not found: {main_tex_file}")

    logger.info("Flattening LaTeX project for pylatexenc...")
    flattened_content = pandoc_parser._resolve_and_flatten_tex(main_tex_file)
    flattened_content = pandoc_parser._strip_latex_comments(flattened_content)
    macros = pandoc_parser._extract_simple_macros(flattened_content)
    flattened_content = pandoc_parser._strip_macro_definitions(flattened_content)
    flattened_content = pandoc_parser._apply_macros(flattened_content, macros)
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

    logger.info("Parsing LaTeX with pylatexenc...")
    walker = LatexWalker(body)
    nodes, _, _ = walker.get_latex_nodes()

    state: Dict[str, Optional[int]] = {"skip_level": None, "stop": False, "abstract": ""}
    body_text = _nodes_to_text(nodes, state)

    output_parts: List[str] = []
    if title:
        output_parts.append(f"Title: {title}")
    abstract_text = state.get("abstract") or preamble_abstract
    if abstract_text and config.KEEP_ABSTRACT:
        output_parts.append(f"Abstract:\n{abstract_text}")

    body_text = body_text.strip()
    if body_text:
        output_parts.append("--- Body ---\n\n" + body_text)

    combined = "\n\n".join(output_parts)
    protected, placeholders = pandoc_parser._protect_math(combined)
    cleaned = pandoc_parser._postprocess_text(protected)
    for key, value in placeholders.items():
        cleaned = cleaned.replace(key, value)

    logger.info("pylatexenc extraction complete.")
    return cleaned.strip()
