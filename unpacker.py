# -*- coding: utf-8 -*-
# unpacker.py

import shutil
import logging
import tarfile
from pathlib import Path
from typing import Union
from langchain.tools import tool

logger = logging.getLogger(__name__)


def _is_within_directory(base_dir: Path, target_path: Path) -> bool:
    try:
        base_dir = base_dir.resolve()
        target_path = target_path.resolve()
        return str(target_path).startswith(str(base_dir))
    except Exception:
        return False


def _safe_extract_tar(archive_path: Path, extract_to_dir: Path) -> None:
    """Extract tar archives while skipping binary assets and preventing path traversal."""
    allowed_exts = {
        ".tex",
        ".bib",
        ".bbl",
        ".cls",
        ".sty",
        ".bst",
        ".cfg",
        ".def",
        ".clo",
        ".txt",
        ".md",
    }
    with tarfile.open(archive_path) as tar:
        members = []
        for member in tar.getmembers():
            name = member.name
            if not name:
                continue
            if member.isdir():
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in allowed_exts:
                continue
            target_path = extract_to_dir / name
            if not _is_within_directory(extract_to_dir, target_path):
                continue
            members.append(member)
        tar.extractall(path=extract_to_dir, members=members)


@tool
def decompress_project(archive_path: Union[str, Path], extract_to_dir: Path, overwrite: bool = False) -> str:
    """
    解压一个压缩文件到指定目录。

    此工具利用 shutil.unpack_archive，可自动处理多种压缩格式，
    通常支持 .zip, .tar, .tar.gz, .tar.bz2, .tar.xz 等。

    Args:
        archive_path (Union[str, Path]): 要解压的压缩文件的完整路径。
        extract_to_dir (Path): 文件将被解压到的目标目录的完整路径。
        overwrite (bool, optional): 如果为 True 且目标目录已存在，则会先删除该目录再解压。
                                   默认为 False，如果目标目录存在且非空，会引发错误。

    Returns:
        str: 成功解压后，返回目标目录的绝对路径。

    Raises:
        FileNotFoundError: 如果指定的压缩文件不存在。
        FileExistsError: 如果目标目录已存在且不为空，并且 overwrite 为 False。
        ValueError: 如果文件格式不支持或文件已损坏。
    """
    archive_path = Path(archive_path).resolve()
    extract_to_dir = Path(extract_to_dir).resolve()

    if not archive_path.exists():
        raise FileNotFoundError(f"压缩文件未找到: {archive_path}")

    if extract_to_dir.exists() and any(extract_to_dir.iterdir()):
        if overwrite:
            logger.warning(f"目标目录 '{extract_to_dir}' 已存在且非空。根据参数将执行覆盖操作。")
            shutil.rmtree(extract_to_dir)
        else:
            raise FileExistsError(
                f"目标目录 '{extract_to_dir}' 已存在且非空。请使用 'overwrite=True' 参数进行覆盖，或选择其他目录。"
            )

    try:
        extract_to_dir.mkdir(parents=True, exist_ok=True)
        if archive_path.suffix in {".tar", ".gz", ".tgz", ".bz2", ".xz"}:
            _safe_extract_tar(archive_path, extract_to_dir)
        else:
            shutil.unpack_archive(archive_path, extract_to_dir)
        logger.info(f"成功将 '{archive_path.name}' 解压到 '{extract_to_dir}'")
        return str(extract_to_dir)

    except shutil.ReadError as e:
        supported_formats = [fmt[0] for fmt in shutil.get_unpack_formats()]
        error_msg = (
            f"无法解压 '{archive_path}'。可能是文件损坏或格式不支持。"
            f"支持的格式: {', '.join(supported_formats)}。详细信息: {e}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        # 在其他错误发生时，清理部分解压的文件
        if extract_to_dir.exists():
            shutil.rmtree(extract_to_dir)
        logger.error(f"解压 '{archive_path}' 时发生意外错误: {e}")
        raise
