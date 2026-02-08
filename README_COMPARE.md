# Pandoc+LLM vs LLM-only 对比测试

该脚本用于对同一个 LaTeX 项目进行两种清洗路径的对比：

- `pandoc+llm`：先 Pandoc 提取，再 LLM 二次清洗
- `llm_only`：按 `.tex` 文件分块，直接 LLM 清洗

脚本文件：`compare_pandoc_llm_vs_llm.py`

## 快速开始

```bash
python compare_pandoc_llm_vs_llm.py "<你的项目路径或压缩包路径>"
```

示例：

```bash
python compare_pandoc_llm_vs_llm.py "G:\Test\latex_project"
python compare_pandoc_llm_vs_llm.py "G:\Test\latex_project.zip"
```

## 常用参数

- `--output-dir benchmark_results`：结果输出目录（默认）
- `--skip-server-start`：不自动启动 vLLM，要求服务已在 `http://127.0.0.1:8000/v1` 可用
- `--judge-with-llm`：额外调用一次 LLM 作为裁判给出 winner/reason
- `--tools-dir <path>`：显式指定工具模块目录（需包含 `config.py`、`llm_parser.py`、`pandoc_parser.py`、`subagent_cleaner.py`）

如果脚本与上述模块不在同一目录，可使用：

```bash
python compare_pandoc_llm_vs_llm.py --tools-dir "/path/to/Extract_Text_From_LaTex" "<project_path>"
```

也可设置环境变量：

```bash
export LATEX_TOOLS_DIR="/path/to/Extract_Text_From_LaTex"
python compare_pandoc_llm_vs_llm.py "<project_path>"
```

## 输出结果

每次运行会创建一个时间戳目录，例如：

`benchmark_results/20260207_170000/`

其中包含：

- `pandoc_plus_llm.txt`：方法一输出
- `llm_only.txt`：方法二输出
- `report.json`：完整指标、耗时、winner、错误信息
- `summary.md`：摘要报告
- `clean_logs/llm_cleanup.log`：每次 LLM 清洗日志（含分块结果）
