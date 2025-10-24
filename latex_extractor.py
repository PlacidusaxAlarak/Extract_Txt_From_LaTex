# main.py

import sys
import logging
import shutil
from pathlib import Path
from typing import Optional
from langchain.tools import tool

# 导入自定义模块
import config
import unpacker
import pandoc_parser

# --- 配置 ---
logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义项目根目录下的 data 目录
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = Path(__file__).parent / "extracted_text.txt"


@tool
def extract_latex_text_from_folder(project_directory: str) -> str:
    """
    从包含LaTeX项目的目录中提取纯文本。
    这个目录应该是已经解压过的。
    返回一个包含提取文本的字符串。
    """
    logger.info(f"从目录 '{project_directory}' 提取LaTeX文本...")
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
    extracted_text = pandoc_parser.parse_project_to_text(main_tex_file)
    return extracted_text


def check_dependencies():
    """检查所有必要的外部依赖项。"""
    if not shutil.which('pandoc'):
        logger.critical("Critical dependency missing: Pandoc is not installed or not in PATH.")
        print("\nPlease install Pandoc and ensure it is in your system's PATH.", file=sys.stderr)
        sys.exit(1)
    logger.info("Pandoc dependency is satisfied.")


def _find_main_tex_file(dir_path: Path) -> Optional[Path]:
    """
    在目录中找到主 .tex 文件。
    策略：优先寻找包含 \begin{document} 的文件，并偏好常见的主文件名。
    """
    main_files = [p for p in dir_path.rglob('*.tex') if
                  '\\begin{document}' in p.read_text(encoding='utf-8', errors='ignore')] # Corrected: escaped backslash for 
    
    if not main_files:
        logger.warning("No .tex file with \begin{document} found. Looking for any .tex file.")
        all_tex_files = list(dir_path.rglob('*.tex'))
        return all_tex_files[0] if all_tex_files else None

    # 优先选择常见的主文件名
    for preferred_name in ['main.tex', 'paper.tex', 'article.tex', 'ms.tex', 'report.tex']:
        for f in main_files:
            if f.name.lower() == preferred_name:
                return f
    
    # 如果没有找到，返回第一个找到的包含 \begin{document} 的文件
    return main_files[0]


def run_extraction():
    """
    执行完整的文本提取流程。
    """
    # 1. 准备 data 目录
    # 如果 data 目录不存在或为空，则尝试解压项目压缩包
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        logger.info(f"'{DATA_DIR}' is empty or non-existent. Searching for a project archive...")
        script_dir = Path(__file__).parent
        supported_ext = ('.zip', '.tar.gz', '.tar', '.tgz', '.bz2')
        archives = [p for p in script_dir.iterdir() if str(p).lower().endswith(supported_ext)]

        if not archives:
            logger.error(f"Error: No project archive found in '{script_dir}'. Please place one here.")
            sys.exit(1)
        
        if len(archives) > 1:
            logger.warning(f"Multiple archives found. Using the first one: {archives[0].name}")
        
        archive_path = archives[0]
        logger.info(f"Found archive: '{archive_path.name}'. Decompressing to '{DATA_DIR}'...")
        # 使用新的 unpacker 工具，并允许覆盖
        unpacker.decompress_project(archive_path, DATA_DIR, overwrite=True)
    else:
        logger.info(f"Using existing files in '{DATA_DIR}'.")

    # 2. 查找主 .tex 文件
    logger.info(f"Searching for main .tex file in '{DATA_DIR}'...")
    main_tex_file = _find_main_tex_file(DATA_DIR)

    if not main_tex_file:
        logger.error("Could not find any .tex file in the data directory.")
        sys.exit(1)

    logger.info(f"Found main file: {main_tex_file.relative_to(DATA_DIR)}. Parsing project to text...")
    
    # 3. 解析为纯文本
    extracted_text = pandoc_parser.parse_project_to_text(main_tex_file)

    # 4. 保存输出
    try:
        OUTPUT_FILE.write_text(extracted_text, encoding='utf-8')
        logger.info(f"Successfully extracted text and saved to '{OUTPUT_FILE}'")
        print(f"\nExtraction complete. Output saved to: {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"Failed to write output file: {e}")
        sys.exit(1)


def main():
    """
    主函数：检查依赖、初始化并运行提取流程。
    """
    try:
        check_dependencies()
        run_extraction()
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        logger.critical(f"A configuration or file error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
