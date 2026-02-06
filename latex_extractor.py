# -*- coding: utf-8 -*-
# main.py

import argparse
import sys
import logging
import shutil
from pathlib import Path
from typing import Optional
from langchain.tools import tool

import config
import unpacker
import engine_selector

# --- 配置 ---
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
FALLBACK_DATA_DIR = Path(__file__).parent / "data_runs"
OUTPUT_FILE = Path(__file__).parent / "extracted_text.txt"
DEFAULT_ENGINE = getattr(config, "DEFAULT_ENGINE", "pandoc")


@tool
def extract_latex_text_from_folder(project_directory: str, engine: str = DEFAULT_ENGINE) -> str:
    """
    从包含 LaTeX 项目的目录中提取纯文本。
    这个目录应该是已经解压过的。
    返回一个包含提取文本的字符串。
    """
    selected_engine = engine_selector.normalize_engine(engine)
    logger.info(f"从目录 '{project_directory}' 提取 LaTeX 文本 (engine={selected_engine})...")
    project_path = Path(project_directory)

    if not project_path.is_dir():
        error_msg = f"错误: 提供的路径 '{project_directory}' 不是一个有效的目录。"
        logger.error(error_msg)
        return error_msg

    main_tex_file = _find_main_tex_file(project_path)

    if not main_tex_file:
        error_msg = f"错误: 在目录 '{project_directory}' 中找不到任何 .tex 文件。"
        logger.error(error_msg)
        return error_msg

    logger.info(f"找到主文件: {main_tex_file.relative_to(project_path)}。正在解析项目为文本...")
    extracted_text = engine_selector.parse_project_to_text(main_tex_file, selected_engine)
    return extracted_text


def check_dependencies(engine: str) -> None:
    """检查所选引擎的依赖项。"""
    selected_engine = engine_selector.normalize_engine(engine)
    if selected_engine == "pandoc":
        if not shutil.which("pandoc"):
            logger.critical("Critical dependency missing: Pandoc is not installed or not in PATH.")
            print("\nPlease install Pandoc and ensure it is in your system's PATH.", file=sys.stderr)
            sys.exit(1)
        logger.info("Pandoc dependency is satisfied.")
        return

    if selected_engine == "latexml":
        if not (shutil.which("latexmlc") or shutil.which("latexml")):
            logger.critical("Critical dependency missing: LaTeXML is not installed or not in PATH.")
            print("\nPlease install LaTeXML and ensure it is in your system's PATH.", file=sys.stderr)
            sys.exit(1)
        logger.info("LaTeXML dependency is satisfied.")
        return

    if selected_engine == "plastex":
        try:
            import plasTeX  # noqa: F401
        except Exception:
            logger.critical("Critical dependency missing: plasTeX is not installed.")
            print("\nPlease install plasTeX (pip install plasTeX).", file=sys.stderr)
            sys.exit(1)
        logger.info("plasTeX dependency is satisfied.")
        return

    if selected_engine == "pylatexenc":
        try:
            import pylatexenc  # noqa: F401
        except Exception:
            logger.critical("Critical dependency missing: pylatexenc is not installed.")
            print("\nPlease install pylatexenc (pip install pylatexenc).", file=sys.stderr)
            sys.exit(1)
        logger.info("pylatexenc dependency is satisfied.")
        return

    raise ValueError(f"Unknown engine: {engine}")


def _find_main_tex_file(dir_path: Path) -> Optional[Path]:
    """
    在目录中找到主 .tex 文件。
    策略：优先寻找包含 \\begin{document} 的文件，并偏好常见的主文件名。
    """
    main_files = [
        p
        for p in dir_path.rglob("*.tex")
        if "\\begin{document}" in p.read_text(encoding="utf-8", errors="ignore")
    ]

    if not main_files:
        logger.warning("No .tex file with \\begin{document} found. Looking for any .tex file.")
        all_tex_files = list(dir_path.rglob("*.tex"))
        return all_tex_files[0] if all_tex_files else None

    # 优先选择常见的主文件名
    for preferred_name in ["main.tex", "paper.tex", "article.tex", "ms.tex", "report.tex"]:
        for f in main_files:
            if f.name.lower() == preferred_name:
                return f

    # 如果没有找到，返回第一个包含 \\begin{document} 的文件
    return main_files[0]


def _select_data_dir() -> Path:
    """Select a writable data directory, falling back when permissions are restricted."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        test_file = DATA_DIR / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()

        if any(DATA_DIR.iterdir()):
            if _has_main_tex(DATA_DIR):
                return DATA_DIR
            logger.warning(f"Data directory missing a main .tex file, skipping: {DATA_DIR}")
        else:
            return DATA_DIR
    except Exception as e:
        logger.warning(f"Data directory not writable: {DATA_DIR} ({e})")

    import time

    fallback_dir = FALLBACK_DATA_DIR / f"run_{int(time.time())}"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    logger.warning(f"Using fallback data directory: {fallback_dir}")
    return fallback_dir


def _has_main_tex(dir_path: Path) -> bool:
    for tex_file in dir_path.rglob("*.tex"):
        try:
            content = tex_file.read_text(encoding="utf-8", errors="ignore")
            if "\\begin{document}" in content:
                return True
        except Exception:
            continue
    return False


def run_extraction(engine: str) -> None:
    """
    执行完整的文本提取流程。
    """
    selected_engine = engine_selector.normalize_engine(engine)

    data_dir = _select_data_dir()

    if not data_dir.exists() or not any(data_dir.iterdir()):
        logger.info(f"'{data_dir}' is empty or non-existent. Searching for a project archive...")
        script_dir = Path(__file__).parent
        supported_ext = (".zip", ".tar.gz", ".tar", ".tgz", ".bz2")
        archives = [p for p in script_dir.iterdir() if str(p).lower().endswith(supported_ext)]

        if not archives:
            logger.error(f"Error: No project archive found in '{script_dir}'. Please place one here.")
            sys.exit(1)

        if len(archives) > 1:
            logger.warning(f"Multiple archives found. Using the first one: {archives[0].name}")

        archive_path = archives[0]
        logger.info(f"Found archive: '{archive_path.name}'. Decompressing to '{data_dir}'...")
        unpacker.decompress_project.invoke(
            {
                "archive_path": str(archive_path),
                "extract_to_dir": str(data_dir),
                "overwrite": True,
            }
        )
    else:
        logger.info(f"Using existing files in '{data_dir}'.")

    logger.info(f"Searching for main .tex file in '{data_dir}'...")
    main_tex_file = _find_main_tex_file(data_dir)

    if not main_tex_file:
        logger.error("Could not find any .tex file in the data directory.")
        sys.exit(1)

    logger.info(
        f"Found main file: {main_tex_file.relative_to(data_dir)}. Parsing project to text (engine={selected_engine})..."
    )

    extracted_text = engine_selector.parse_project_to_text(main_tex_file, selected_engine)

    try:
        OUTPUT_FILE.write_text(extracted_text, encoding="utf-8")
        logger.info(f"Successfully extracted text and saved to '{OUTPUT_FILE}'")
        print(f"\nExtraction complete. Output saved to: {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"Failed to write output file: {e}")
        sys.exit(1)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract text from a LaTeX project.")
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        help="Choose extraction engine: pandoc | latexml(latexML) | plastex(plasTex) | pylatexenc",
    )
    return parser


def main() -> None:
    """
    主函数：检查依赖、初始化并运行提取流程。
    """
    args = _build_arg_parser().parse_args()
    try:
        check_dependencies(args.engine)
        run_extraction(args.engine)
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        logger.critical(f"A configuration or file error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
