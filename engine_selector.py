# -*- coding: utf-8 -*-
"""Engine selector for LaTeX text extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandoc_parser

try:
    import latexml_parser
except Exception:  # optional dependency
    latexml_parser = None

try:
    import plastex_parser
except Exception:  # optional dependency
    plastex_parser = None

try:
    import pylatexenc_parser
except Exception:  # optional dependency
    pylatexenc_parser = None


ENGINE_ALIASES: Dict[str, str] = {
    "pandoc": "pandoc",
    "latexml": "latexml",
    "plastex": "plastex",
    "pylatexenc": "pylatexenc",
}

AVAILABLE_ENGINES = ("pandoc", "latexml", "plastex", "pylatexenc")


def normalize_engine(engine: str) -> str:
    if not engine:
        return "pandoc"
    key = engine.strip().lower()
    return ENGINE_ALIASES.get(key, key)


def parse_project_to_text(main_tex_file: Path, engine: str) -> str:
    selected = normalize_engine(engine)
    if selected == "pandoc":
        return pandoc_parser.parse_project_to_text(main_tex_file)
    if selected == "latexml":
        if latexml_parser is None:
            raise RuntimeError("latexml_parser not available. Please check dependencies.")
        return latexml_parser.parse_project_to_text(main_tex_file)
    if selected == "plastex":
        if plastex_parser is None:
            raise RuntimeError("plastex_parser not available. Please check dependencies.")
        return plastex_parser.parse_project_to_text(main_tex_file)
    if selected == "pylatexenc":
        if pylatexenc_parser is None:
            raise RuntimeError("pylatexenc_parser not available. Please check dependencies.")
        return pylatexenc_parser.parse_project_to_text(main_tex_file)
    raise ValueError(f"Unknown engine: {engine}")
