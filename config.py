# -*- coding: utf-8 -*-
# config.py

import os

# 从环境变量获取日志级别（默认 INFO）
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# 默认引擎
DEFAULT_ENGINE = os.getenv("LATEX_ENGINE", "pandoc").lower()

# 抽取设置
KEEP_TITLE = True
KEEP_ABSTRACT = True
KEEP_FOOTNOTES = True
KEEP_MATH = True

# 需要跳过的章节关键词（触发后跳过，直到下一个同级或更高层级标题）
SECTION_EXCLUDE_KEYWORDS = [
    "references",
    "bibliography",
    "appendix",
    "supplementary",
    "supplemental",
    "acknowledg",
    "funding",
    "author contributions",
    "conflict of interest",
    "参考文献",
    "附录",
    "致谢",
    "资助",
    "作者贡献",
    "利益冲突",
    "补充材料",
]

# 预处理开关
REMOVE_BIBLIOGRAPHY_ENV = True
REMOVE_BIBLIOGRAPHY_COMMAND = True
TRUNCATE_AT_APPENDIX = True

# Pandoc 解析失败时的恢复策略
PANDOC_RECOVERY_MAX_ATTEMPTS = int(os.getenv("PANDOC_RECOVERY_MAX_ATTEMPTS", "5"))
PANDOC_RECOVERY_WINDOW = int(os.getenv("PANDOC_RECOVERY_WINDOW", "3"))
