# -*- coding: utf-8 -*-
# file_processor.py

import shutil
import logging
from pathlib import Path

# 使用具名 logger
logger = logging.getLogger(__name__)


def cleanup_directory(dir_path: Path) -> None:
    """
    安全地删除指定的目录及其所有内容。
    """
    if dir_path and dir_path.exists() and dir_path.is_dir():
        try:
            shutil.rmtree(dir_path)
            logger.info(f"Successfully cleaned up directory: {dir_path}")
        except OSError as e:
            # 对于大型目录，shutil.rmtree 可能会很慢，但对于单个项目来说通常可接受。
            logger.error(f"Error cleaning up directory {dir_path}: {e}")
    else:
        logger.warning(f"Cleanup skipped: Directory '{dir_path}' does not exist or is not a directory.")
