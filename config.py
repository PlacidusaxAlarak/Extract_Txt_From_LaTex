# config.py

import os

# 从环境变量获取日志级别，默认为 INFO
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
