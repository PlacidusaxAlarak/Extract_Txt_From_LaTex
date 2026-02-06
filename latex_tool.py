# -*- coding: utf-8 -*-
"""Tool wrapper to extract text from LaTeX projects."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import engine_selector
import pandoc_parser
import unpacker
import latex_extractor

logger = logging.getLogger(__name__)

ARCHIVE_EXTS = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".bz2",
    ".gz",
    ".xz",
)


def _is_archive(path: Path) -> bool:
    lower = str(path).lower()
    return any(lower.endswith(ext) for ext in ARCHIVE_EXTS)


def _decompress_archive(archive_path: Path, extract_to_dir: Path) -> Path:
    if hasattr(unpacker.decompress_project, "invoke"):
        unpacker.decompress_project.invoke(
            {"archive_path": str(archive_path), "extract_to_dir": str(extract_to_dir), "overwrite": True}
        )
    else:
        unpacker.decompress_project(archive_path=str(archive_path), extract_to_dir=extract_to_dir, overwrite=True)
    return extract_to_dir


def _extract_from_dir(dir_path: Path, engine: str, fallback_mode: Optional[str]) -> str:
    main_tex = None
    if hasattr(latex_extractor, "_find_main_tex_file"):
        main_tex = latex_extractor._find_main_tex_file(dir_path)
    if not main_tex:
        return f"Error: no .tex file found under {dir_path}"

    selected = engine_selector.normalize_engine(engine)
    if selected == "pandoc":
        text, _meta = pandoc_parser.parse_project_to_text_with_meta(main_tex, fallback_mode=fallback_mode)
        return text
    return engine_selector.parse_project_to_text(main_tex, selected)


def latex_project_to_text(project_path: str, engine: str = "pandoc", fallback_mode: Optional[str] = None) -> str:
    """Extract plain text from a LaTeX project directory or archive."""
    if not project_path:
        return "Error: project_path is required."

    path = Path(project_path).expanduser()
    if not path.exists():
        return f"Error: path does not exist: {path}"

    try:
        if path.is_dir():
            return _extract_from_dir(path, engine, fallback_mode)

        if path.is_file() and _is_archive(path):
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                _decompress_archive(path, tmp_path)
                return _extract_from_dir(tmp_path, engine, fallback_mode)

        if path.is_file():
            return f"Error: unsupported file type (not an archive): {path}"

        return f"Error: invalid path: {path}"
    except Exception as exc:
        logger.exception("latex_project_to_text failed.")
        return f"Error: {exc}"
