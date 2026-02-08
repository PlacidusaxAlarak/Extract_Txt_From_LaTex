# Extract_Text_From_LaTex

Tools to convert LaTeX projects into cleaned plain text, with two main paths:

- **`pandoc + LLM`**: Pandoc parses LaTeX into an AST, then optional LLM polishing.
- **`LLM-only`**: Raw LaTeX body (only `\begin{document}`..`\end{document}`) is split by
  sections and cleaned by an LLM.

This repo also includes a benchmarking script to compare both approaches.

---

## Quick Start

### 1) Install requirements

Minimal runtime:

- Python 3.9+
- `openai` Python package

Optional but recommended:

- `pandoc` (or `pypandoc`) for the Pandoc path
- vLLM server (OpenAI-compatible API)

```bash
python -m pip install openai
```

If you rely on Pandoc:

```bash
# System package manager (example)
pandoc --version
```

### 2) Run the benchmark script

The script compares `pandoc_plus_llm` vs `llm_only` on a LaTeX project or archive:

```bash
python compare_pandoc_llm_vs_llm.py /path/to/project_or_archive.tar.gz
```

You can also pass:

```bash
python compare_pandoc_llm_vs_llm.py --project-path /path/to/project
```

Outputs are saved under `benchmark_results/YYYYmmdd_HHMMSS/`:

- `pandoc_plus_llm.txt`
- `llm_only.txt`
- `report.json`
- `summary.md`

---

## How It Works

### LLM-only path

1. Reads the main `.tex`.
2. Recursively expands `\input`, `\include`, `\subfile`.
3. Keeps only `\begin{document}`..`\end{document}`.
4. Splits by section commands and injects Markdown headings.
5. Cleans each chunk with an LLM.

This is implemented in `pandoc_parser.py` and `llm_parser.py`.

### Pandoc + LLM path

1. Pandoc converts LaTeX to JSON AST.
2. Text is extracted from the AST.
3. Optional LLM polish is applied.

Fallback: if Pandoc fails, it uses the same LLM-only chunks.

---

## LLM Cleaning Prompt

The LLM cleaner lives in `subagent_cleaner.py`. The current prompt enforces:

- Remove citations and references (`\cite`, `\ref`, `\label`, etc.)
- Remove image rendering info; keep figure captions as `Figure: <caption>`
- Preserve table layout and tabular content
- Remove custom macro definitions and tokens
- Preserve readable prose and equations

You can edit `_build_system_prompt()` to tune behavior.

---

## vLLM Server

The benchmark script can start a local vLLM server, or reuse an existing one:

```bash
# Environment variables used by the script
export BENCH_MODEL_PATH=/path/to/model
export BENCH_SERVED_MODEL_NAME=Qwen2.5-7B-Instruct
export BENCH_HOST=127.0.0.1
export BENCH_PORT=8000
```

If a server is already running at `http://127.0.0.1:8000/v1`, it will reuse it.

---

## Configuration (config.py)

Key options:

- `KEEP_TITLE`, `KEEP_ABSTRACT`, `KEEP_FOOTNOTES`, `KEEP_MATH`
- `REMOVE_BIBLIOGRAPHY_ENV`, `REMOVE_BIBLIOGRAPHY_COMMAND`
- `TRUNCATE_AT_APPENDIX`
- `PANDOC_RECOVERY_MAX_ATTEMPTS`, `PANDOC_RECOVERY_WINDOW`

LLM logging:

- `SUBAGENT_CLEAN_LOG_ENABLED`
- `SUBAGENT_CLEAN_LOG_DIR`
- `SUBAGENT_CLEAN_LOG_FILE`

Logs are appended to `logs/llm_cleanup.log` by default.

---

## Troubleshooting

**1) Pandoc errors**

If Pandoc fails, the parser auto-recovers by dropping problematic lines or
paragraphs and retries. It logs warnings like:

```
Pandoc failed at line N. Dropping with strategy=...
```

**2) Missing content**

If you see missing sections, enable section headings (already injected by
default) and re-run. This improves coverage by preventing the LLM from
dropping whole sections.

**3) Import errors**

If you see errors like:

```
Could not import 'latex_tool'
```

That module is not part of this repo. Use the provided scripts instead, or
create a wrapper around `pandoc_parser.parse_project_to_text_with_meta`.

---

## License

No license is defined in this repository. Add one if you plan to publish.
