# -*- coding: utf-8 -*-
"""LaTeXML-based extractor (optional dependency)."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import config
import pandoc_parser

logger = logging.getLogger(__name__)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _element_text(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    return " ".join("".join(elem.itertext()).split()).strip()


def _find_first_text(root: ET.Element, tag_names: Iterable[str]) -> str:
    targets = {name.lower() for name in tag_names}
    for elem in root.iter():
        if _local_name(elem.tag).lower() in targets:
            text = _element_text(elem)
            if text:
                return text
    return ""


def _find_title(root: ET.Element) -> str:
    front = root.find(".//{*}frontmatter")
    if front is not None:
        title = _find_first_text(front, ("title",))
        if title:
            return title
    return _find_first_text(root, ("title",))


def _find_abstract(root: ET.Element) -> str:
    front = root.find(".//{*}frontmatter")
    if front is not None:
        abstract = _find_first_text(front, ("abstract",))
        if abstract:
            return abstract
    return _find_first_text(root, ("abstract",))


def _is_excluded_header(header: str) -> bool:
    return pandoc_parser._is_excluded_section(header)


def _collect_text_from_section(elem: ET.Element) -> str:
    lines: List[str] = []

    for child in elem.iter():
        name = _local_name(child.tag).lower()
        if name in {"section", "subsection", "subsubsection", "chapter", "part"}:
            title_elem = child.find(".//{*}title")
            title = _element_text(title_elem)
            if title:
                if _is_excluded_header(title):
                    return "\n".join(lines)
                lines.append(title)
                lines.append("")
            continue

        if name in {"thebibliography", "bibliography", "appendix"}:
            return "\n".join(lines)

        if name in {"p", "para", "paragraph"}:
            text = _element_text(child)
            if text:
                lines.append(text)
                lines.append("")
        elif name in {"math", "m", "mathml"} and config.KEEP_MATH:
            text = _element_text(child)
            if text:
                lines.append(text)
                lines.append("")
        elif name in {"footnote", "note"} and config.KEEP_FOOTNOTES:
            text = _element_text(child)
            if text:
                lines.append(f"[Footnote: {text}]")
                lines.append("")

    return "\n".join(lines).strip()


def _run_latexml(tex_path: Path, output_path: Path) -> None:
    latexmlc = shutil.which("latexmlc")
    latexml = shutil.which("latexml")
    if latexmlc:
        cmd = [latexmlc, "--format=xml", f"--dest={output_path}", str(tex_path)]
    elif latexml:
        cmd = [latexml, f"--dest={output_path}", str(tex_path)]
    else:
        raise RuntimeError("LaTeXML is not installed or not in PATH.")

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "LaTeXML failed to convert LaTeX."
        raise RuntimeError(stderr)


def parse_project_to_text(main_tex_file: Path) -> str:
    if not main_tex_file.exists():
        raise FileNotFoundError(f"Main .tex file not found: {main_tex_file}")

    logger.info("Flattening LaTeX project for LaTeXML...")
    flattened_content = pandoc_parser._resolve_and_flatten_tex(main_tex_file)
    flattened_content = pandoc_parser._strip_latex_comments(flattened_content)
    flattened_content = pandoc_parser._remove_bibliography_content(flattened_content)
    flattened_content = pandoc_parser._truncate_at_appendix(flattened_content)
    flattened_content = pandoc_parser._strip_cjk_envs(flattened_content)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        tex_path = tmp_dir_path / "main.tex"
        xml_path = tmp_dir_path / "out.xml"

        tex_path.write_text(flattened_content, encoding="utf-8")
        logger.info("Running LaTeXML conversion...")
        _run_latexml(tex_path, xml_path)

        logger.info("Parsing LaTeXML XML output...")
        tree = ET.parse(xml_path)
        root = tree.getroot()

    title = _find_title(root) if config.KEEP_TITLE else ""
    abstract = _find_abstract(root) if config.KEEP_ABSTRACT else ""

    body_elem = root.find(".//{*}body") or root
    body_text = _collect_text_from_section(body_elem)

    output_parts: List[str] = []
    if title:
        output_parts.append(f"Title: {title}")
    if abstract:
        output_parts.append(f"Abstract:\n{abstract}")
    if body_text:
        output_parts.append("--- Body ---\n\n" + body_text)

    combined = "\n\n".join(output_parts)
    combined = pandoc_parser._postprocess_text(combined)
    logger.info("LaTeXML extraction complete.")
    return combined
