#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

config = None
llm_parser = None
pandoc_parser = None
subagent_cleaner = None


def _candidate_tools_dirs(explicit_dir: Optional[Path]) -> List[Path]:
    dirs: List[Path] = []

    if explicit_dir is not None:
        dirs.append(explicit_dir.expanduser().resolve())

    env_dir = os.getenv("LATEX_TOOLS_DIR")
    if env_dir:
        dirs.append(Path(env_dir).expanduser().resolve())

    dirs.extend([TOOLS_DIR.resolve(), TOOLS_DIR.parent.resolve(), Path.cwd().resolve()])

    unique_dirs: List[Path] = []
    seen = set()
    for item in dirs:
        key = str(item)
        if key not in seen:
            seen.add(key)
            unique_dirs.append(item)
    return unique_dirs


def _load_tool_modules(explicit_dir: Optional[Path]) -> Path:
    global config, llm_parser, pandoc_parser, subagent_cleaner

    required_files = [
        "config.py",
        "llm_parser.py",
        "pandoc_parser.py",
        "subagent_cleaner.py",
    ]
    errors: List[str] = []

    for candidate in _candidate_tools_dirs(explicit_dir):
        if not candidate.is_dir():
            continue
        if not all((candidate / filename).exists() for filename in required_files):
            continue

        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

        try:
            config = importlib.import_module("config")
            llm_parser = importlib.import_module("llm_parser")
            pandoc_parser = importlib.import_module("pandoc_parser")
            subagent_cleaner = importlib.import_module("subagent_cleaner")
            return candidate
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    searched = "\n  - ".join(str(path) for path in _candidate_tools_dirs(explicit_dir))
    detail = "\n".join(errors) if errors else "(no import attempts succeeded)"
    raise ImportError(
        "Failed to import required modules (config/llm_parser/pandoc_parser/subagent_cleaner).\n"
        f"Searched directories:\n  - {searched}\n"
        f"Details:\n{detail}\n"
        "Hint: pass --tools-dir or set LATEX_TOOLS_DIR to the directory containing these .py files."
    )

# ================= Runtime configuration =================

MODEL_PATH = os.getenv(
    "BENCH_MODEL_PATH",
    "/inspire/hdd/project/exploration-topic/public/downloaded_ckpts/Qwen2.5-7B-Instruct",
)
SERVED_MODEL_NAME = os.getenv("BENCH_SERVED_MODEL_NAME", "Qwen2.5-7B-Instruct")

VLLM_GPU_ID = os.getenv("BENCH_VLLM_GPU_ID", "0")
VLLM_TP_SIZE = int(os.getenv("BENCH_VLLM_TP_SIZE", "1"))

HOST = os.getenv("BENCH_HOST", "127.0.0.1")
PORT = int(os.getenv("BENCH_PORT", "8000"))
SERVER_URL = f"http://{HOST}:{PORT}/v1"

READY_TIMEOUT_SEC = int(os.getenv("BENCH_READY_TIMEOUT_SEC", "300"))
READY_POLL_SEC = int(os.getenv("BENCH_READY_POLL_SEC", "2"))


def _start_vllm_server() -> subprocess.Popen:
    print(f"[System] Starting vLLM server at {HOST}:{PORT} ...")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = VLLM_GPU_ID

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_PATH,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--tensor-parallel-size",
        str(VLLM_TP_SIZE),
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--trust-remote-code",
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.9",
    ]

    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def _is_server_ready() -> bool:
    url = f"{SERVER_URL}/models"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def _wait_for_ready(timeout_sec: int = READY_TIMEOUT_SEC) -> None:
    print("[System] Waiting for vLLM server to become ready...")
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if _is_server_ready():
            print("[System] vLLM server is ready.")
            return
        time.sleep(READY_POLL_SEC)
    raise TimeoutError(f"vLLM server not ready after {timeout_sec} seconds.")


def _configure_subagent_runtime(clean_log_dir: Path) -> None:
    config.SUBAGENT_BACKEND = "vllm"
    config.SUBAGENT_BASE_URL = SERVER_URL
    config.SUBAGENT_API_KEY = "EMPTY"
    config.SUBAGENT_MODEL = SERVED_MODEL_NAME
    config.SUBAGENT_TIMEOUT_SEC = int(getattr(config, "SUBAGENT_TIMEOUT_SEC", 240))

    config.SUBAGENT_CLEAN_LOG_ENABLED = "1"
    config.SUBAGENT_CLEAN_LOG_DIR = str(clean_log_dir)
    config.SUBAGENT_CLEAN_LOG_FILE = "llm_cleanup.log"


def _extract_archive_to_temp(archive_path: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="latex_compare_"))
    try:
        shutil.unpack_archive(str(archive_path), str(temp_dir))
    except Exception as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to unpack archive '{archive_path}': {exc}") from exc
    return temp_dir


def _read_text_safely(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _find_main_tex(project_root: Path) -> Path:
    tex_files = sorted(project_root.rglob("*.tex"))
    if not tex_files:
        raise FileNotFoundError(f"No .tex file found under: {project_root}")

    preferred_names = ["main.tex", "paper.tex", "article.tex", "ms.tex"]
    lowered_name_to_path = {path.name.lower(): path for path in tex_files}
    for preferred in preferred_names:
        if preferred in lowered_name_to_path:
            return lowered_name_to_path[preferred]

    def _score(path: Path) -> Tuple[int, int, int]:
        text = _read_text_safely(path)
        has_begin_document = 1 if "\\begin{document}" in text else 0
        include_count = len(re.findall(r"\\(?:input|include|subfile)\b", text))
        return has_begin_document, include_count, path.stat().st_size

    return max(tex_files, key=_score)


def _resolve_main_tex(project_path: Path) -> Tuple[Path, Optional[Path]]:
    if not project_path.exists():
        raise FileNotFoundError(f"Path does not exist: {project_path}")

    if project_path.is_dir():
        return _find_main_tex(project_path), None

    if project_path.suffix.lower() == ".tex":
        return project_path, None

    extracted_dir = _extract_archive_to_temp(project_path)
    return _find_main_tex(extracted_dir), extracted_dir


def _compute_quality_metrics(text: str) -> Dict[str, Any]:
    stripped_text = (text or "").strip()
    words = re.findall(r"\S+", stripped_text)
    lines = stripped_text.splitlines() if stripped_text else []
    non_empty_lines = [line for line in lines if line.strip()]

    latex_command_count = len(re.findall(r"\\[a-zA-Z@]+\*?", stripped_text))
    structural_latex_count = len(
        re.findall(
            r"\\(?:begin|end|cite|ref|eqref|label|bibliography|bibliographystyle)\b",
            stripped_text,
        )
    )
    brace_imbalance = abs(stripped_text.count("{") - stripped_text.count("}"))
    placeholder_count = len(re.findall(r"@@[A-Z0-9_]+@@", stripped_text))

    coverage_score = min(40.0, len(words) / 250.0)
    structure_score = min(20.0, len(non_empty_lines) / 8.0)
    penalty = (
        latex_command_count * 0.12
        + structural_latex_count * 0.8
        + brace_imbalance * 0.3
        + placeholder_count * 1.0
    )
    quality_score = max(0.0, min(100.0, 40.0 + coverage_score + structure_score - penalty))

    return {
        "char_count": len(stripped_text),
        "word_count": len(words),
        "line_count": len(lines),
        "non_empty_line_count": len(non_empty_lines),
        "latex_command_count": latex_command_count,
        "structural_latex_count": structural_latex_count,
        "brace_imbalance": brace_imbalance,
        "placeholder_count": placeholder_count,
        "quality_score": round(quality_score, 3),
    }


def _compare_by_heuristics(
    pandoc_metrics: Dict[str, Any], llm_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    pandoc_score = float(pandoc_metrics.get("quality_score", 0.0))
    llm_score = float(llm_metrics.get("quality_score", 0.0))
    diff = round(abs(pandoc_score - llm_score), 3)

    if diff <= 1.0:
        winner = "tie"
    else:
        winner = "pandoc_plus_llm" if pandoc_score > llm_score else "llm_only"

    return {
        "winner": winner,
        "score_diff": diff,
        "pandoc_plus_llm_score": pandoc_score,
        "llm_only_score": llm_score,
    }


def _truncate_for_judge(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    head_size = max_chars // 2
    tail_size = max_chars - head_size
    return f"{text[:head_size]}\n\n...[TRUNCATED]...\n\n{text[-tail_size:]}"


def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", raw_text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _judge_with_llm(
    client: OpenAI,
    pandoc_text: str,
    llm_text: str,
) -> Dict[str, Any]:
    prompt = (
        "Compare two cleaned texts from the same LaTeX project.\n"
        "Text A: pandoc_plus_llm\n"
        "Text B: llm_only\n"
        "Evaluate readability, completeness, and remaining LaTeX noise.\n"
        "Return strict JSON with keys: winner, confidence, reason.\n"
        "winner must be one of: pandoc_plus_llm, llm_only, tie."
    )

    user_content = (
        "[Text A: pandoc_plus_llm]\n"
        f"{_truncate_for_judge(pandoc_text)}\n\n"
        "[Text B: llm_only]\n"
        f"{_truncate_for_judge(llm_text)}"
    )

    try:
        response = client.chat.completions.create(
            model=SERVED_MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
        }

    raw_content = response.choices[0].message.content if response.choices else ""
    parsed = _extract_json_object(raw_content)
    if parsed is None:
        return {
            "status": "parse_failed",
            "raw": raw_content,
        }

    winner = str(parsed.get("winner", "tie")).strip().lower()
    if winner not in {"pandoc_plus_llm", "llm_only", "tie"}:
        winner = "tie"

    return {
        "status": "ok",
        "winner": winner,
        "confidence": parsed.get("confidence"),
        "reason": parsed.get("reason"),
        "raw": raw_content,
    }


def _run_pandoc_plus_llm(main_tex_file: Path) -> Tuple[str, Dict[str, Any]]:
    start_time = time.perf_counter()
    pandoc_text, pandoc_meta = pandoc_parser.parse_project_to_text_with_meta(
        main_tex_file,
        fallback_mode="rule_then_llm",
    )

    llm_polished_text = subagent_cleaner.clean_latex_with_llm(pandoc_text, config)
    final_text = (llm_polished_text or pandoc_text or "").strip()

    duration_sec = round(time.perf_counter() - start_time, 3)
    meta = {
        "mode": "pandoc_plus_llm",
        "duration_sec": duration_sec,
        "post_llm_polish_used": bool(llm_polished_text),
        "pandoc_meta": pandoc_meta,
    }
    return final_text, meta


def _run_llm_only(main_tex_file: Path) -> Tuple[str, Dict[str, Any]]:
    start_time = time.perf_counter()
    text = llm_parser.parse_project_to_text(main_tex_file).strip()
    duration_sec = round(time.perf_counter() - start_time, 3)
    meta = {
        "mode": "llm_only",
        "duration_sec": duration_sec,
    }
    return text, meta


def _safe_run(name: str, func) -> Dict[str, Any]:
    try:
        text, meta = func()
        return {
            "name": name,
            "status": "ok",
            "text": text,
            "meta": meta,
            "metrics": _compute_quality_metrics(text),
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "error",
            "error": str(exc),
        }


def _write_text(path: Path, content: str) -> None:
    path.write_text(content or "", encoding="utf-8")


def _resolve_project_path_from_args(args: argparse.Namespace) -> Path:
    project_raw = args.project_path_opt or args.project_path or os.getenv("BENCH_PROJECT_PATH", "").strip()

    if not project_raw:
        try:
            project_raw = input("[Input] project_path (dir/.tex/archive): ").strip()
        except Exception:
            project_raw = ""

    if not project_raw:
        raise ValueError(
            "project_path is required. Use positional project_path, --project-path, "
            "or set BENCH_PROJECT_PATH."
        )

    return Path(project_raw).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare cleaning quality: pandoc+llm vs llm_only"
    )
    parser.add_argument(
        "--tools-dir",
        type=str,
        default=None,
        help="Directory containing config.py, llm_parser.py, pandoc_parser.py, subagent_cleaner.py",
    )
    parser.add_argument(
        "project_path",
        nargs="?",
        type=str,
        default=None,
        help="Path to LaTeX project directory, main .tex, or archive (zip/tar.*)",
    )
    parser.add_argument(
        "--project-path",
        dest="project_path_opt",
        type=str,
        default=None,
        help="Alternative named argument for project path (same meaning as positional project_path)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save outputs and report",
    )
    parser.add_argument(
        "--skip-server-start",
        action="store_true",
        help="Do not start vLLM server automatically",
    )
    parser.add_argument(
        "--judge-with-llm",
        action="store_true",
        help="Use an extra LLM call as judge after both methods finish",
    )
    args = parser.parse_args()

    try:
        modules_dir = _load_tool_modules(Path(args.tools_dir) if args.tools_dir else None)
    except Exception as exc:
        print(f"[Error] {exc}")
        sys.exit(1)
    print(f"[System] Loaded parser modules from: {modules_dir}")

    output_root = Path(args.output_dir).resolve()
    run_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    started_server: Optional[subprocess.Popen] = None
    extracted_dir: Optional[Path] = None

    try:
        if _is_server_ready():
            print(f"[System] Reusing existing vLLM server: {SERVER_URL}")
        else:
            if args.skip_server_start:
                raise RuntimeError(
                    f"vLLM server is not ready at {SERVER_URL} and --skip-server-start is set."
                )
            started_server = _start_vllm_server()
            _wait_for_ready()

        _configure_subagent_runtime(run_dir / "clean_logs")

        project_path = _resolve_project_path_from_args(args)
        main_tex_file, extracted_dir = _resolve_main_tex(project_path)
        print(f"[System] Main TeX selected: {main_tex_file}")

        pandoc_result = _safe_run(
            "pandoc_plus_llm",
            lambda: _run_pandoc_plus_llm(main_tex_file),
        )
        llm_result = _safe_run(
            "llm_only",
            lambda: _run_llm_only(main_tex_file),
        )

        if pandoc_result.get("status") == "ok":
            _write_text(run_dir / "pandoc_plus_llm.txt", pandoc_result.get("text", ""))
        if llm_result.get("status") == "ok":
            _write_text(run_dir / "llm_only.txt", llm_result.get("text", ""))

        heuristic_comparison: Dict[str, Any] = {
            "winner": "unknown",
            "reason": "one_or_both_methods_failed",
        }
        if pandoc_result.get("status") == "ok" and llm_result.get("status") == "ok":
            heuristic_comparison = _compare_by_heuristics(
                pandoc_result["metrics"],
                llm_result["metrics"],
            )

        llm_judge_result: Optional[Dict[str, Any]] = None
        if (
            args.judge_with_llm
            and pandoc_result.get("status") == "ok"
            and llm_result.get("status") == "ok"
        ):
            judge_client = OpenAI(base_url=SERVER_URL, api_key="EMPTY")
            llm_judge_result = _judge_with_llm(
                judge_client,
                pandoc_result["text"],
                llm_result["text"],
            )

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "server_url": SERVER_URL,
            "model": SERVED_MODEL_NAME,
            "project_path": str(project_path),
            "main_tex_file": str(main_tex_file),
            "pandoc_plus_llm": {
                "status": pandoc_result.get("status"),
                "meta": pandoc_result.get("meta"),
                "metrics": pandoc_result.get("metrics"),
                "error": pandoc_result.get("error"),
                "output_file": "pandoc_plus_llm.txt"
                if pandoc_result.get("status") == "ok"
                else None,
            },
            "llm_only": {
                "status": llm_result.get("status"),
                "meta": llm_result.get("meta"),
                "metrics": llm_result.get("metrics"),
                "error": llm_result.get("error"),
                "output_file": "llm_only.txt" if llm_result.get("status") == "ok" else None,
            },
            "heuristic_comparison": heuristic_comparison,
            "llm_judge": llm_judge_result,
            "clean_log_file": str((run_dir / "clean_logs" / "llm_cleanup.log").resolve()),
        }

        report_path = run_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        summary_lines = [
            "# Cleaning Comparison Result",
            "",
            f"- project_path: `{project_path}`",
            f"- main_tex_file: `{main_tex_file}`",
            f"- model: `{SERVED_MODEL_NAME}`",
            f"- server_url: `{SERVER_URL}`",
            f"- clean_log_file: `{(run_dir / 'clean_logs' / 'llm_cleanup.log').resolve()}`",
            "",
            "## Status",
            f"- pandoc_plus_llm: `{pandoc_result.get('status')}`",
            f"- llm_only: `{llm_result.get('status')}`",
            "",
            "## Heuristic Winner",
            f"- winner: `{heuristic_comparison.get('winner')}`",
            f"- score_diff: `{heuristic_comparison.get('score_diff')}`",
            "",
            f"See full details in `{report_path}`.",
        ]

        summary_path = run_dir / "summary.md"
        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

        print("\n[Result] Benchmark completed.")
        print(f"[Result] Run dir: {run_dir}")
        print(f"[Result] Report: {report_path}")
        print(f"[Result] Summary: {summary_path}")

    finally:
        if extracted_dir is not None:
            shutil.rmtree(extracted_dir, ignore_errors=True)

        if started_server is not None and started_server.poll() is None:
            print("[System] Shutting down vLLM server started by this script...")
            started_server.terminate()
            try:
                started_server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                started_server.kill()


if __name__ == "__main__":
    main()
