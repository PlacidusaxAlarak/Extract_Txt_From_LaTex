# pandoc_parser.py

import logging
import json
try:
    import pypandoc
except ModuleNotFoundError:
    pypandoc = None
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

import config

logger = logging.getLogger(__name__)


def _strip_latex_comments(text: str) -> str:
    """Removes LaTeX comments while preserving escaped percent signs."""
    lines = []
    for line in text.splitlines():
        lines.append(re.sub(r'(?<!\\)%.*$', '', line))
    return "\n".join(lines)


def _strip_cjk_envs(text: str) -> str:
    """Remove CJK environment wrappers while preserving inner content."""
    text = re.sub(r'\\begin\{CJK\*?\}(?:\[[^\]]*\])?(?:\{[^}]*\}){0,2}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\\end\{CJK\*?\}', '', text, flags=re.IGNORECASE)
    return text


def _extract_simple_macros(text: str) -> Dict[str, str]:
    """Extract simple zero-argument macros from \\newcommand/\\renewcommand/\\def."""
    macros: Dict[str, str] = {}
    for match in re.finditer(r'\\(?:re)?newcommand\s*\{\\([a-zA-Z]+)\}\s*\{([^{}]*)\}', text):
        macros[match.group(1)] = match.group(2)
    for match in re.finditer(r'\\def\\([a-zA-Z]+)\s*\{([^{}]*)\}', text):
        macros[match.group(1)] = match.group(2)
    return macros


def _apply_macros(text: str, macros: Dict[str, str]) -> str:
    if not macros:
        return text
    for name, value in macros.items():
        pattern = r'\\' + re.escape(name) + r'\b'
        text = re.sub(pattern, lambda _m, v=value: v, text)
    return text


def _strip_macro_definitions(text: str) -> str:
    text = re.sub(r'\\(?:re)?newcommand\s*\{\\[a-zA-Z]+\}\s*\{[^{}]*\}', '', text)
    text = re.sub(r'\\def\\[a-zA-Z]+\s*\{[^{}]*\}', '', text)
    return text

def _normalize_header_text(text: str) -> str:
    normalized = re.sub(r'[\s\W_]+', ' ', text.lower()).strip()
    return normalized


def _is_excluded_section(header_text: str) -> bool:
    normalized = _normalize_header_text(header_text)
    for keyword in config.SECTION_EXCLUDE_KEYWORDS:
        if keyword and keyword.lower() in normalized:
            return True
    return False


def _remove_bibliography_content(text: str) -> str:
    if config.REMOVE_BIBLIOGRAPHY_ENV:
        text = re.sub(
            r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}',
            '',
            text,
            flags=re.DOTALL | re.IGNORECASE
        )
    if config.REMOVE_BIBLIOGRAPHY_COMMAND:
        text = re.sub(r'\\bibliography\{[^}]*\}', '', text, flags=re.IGNORECASE)
    return text


def _truncate_at_appendix(text: str) -> str:
    if not config.TRUNCATE_AT_APPENDIX:
        return text
    match = re.search(r'\\appendix\b', text, flags=re.IGNORECASE)
    if not match:
        return text
    head = text[:match.start()]
    tail = text[match.start():]
    if re.search(r'\\end\{document\}', tail, flags=re.IGNORECASE):
        head += "\n\\end{document}\n"
    return head

def _resolve_and_flatten_tex(main_file_path: Path, processed_files: Set[Path] = None) -> str:
    """
    Recursively reads a main .tex file and flattens its content by replacing
    \input{} and \include{} commands, while ignoring commented out commands.
    """
    if processed_files is None:
        processed_files = set()

    if not main_file_path.exists():
        logger.warning(f"Attempted to include a non-existent file: {main_file_path}")
        return f"% [File not found: {main_file_path.name}]"

    abs_path = main_file_path.resolve()
    if abs_path in processed_files:
        logger.warning(f"Circular dependency detected: {abs_path} has already been included. Skipping.")
        return ""

    processed_files.add(abs_path)

    try:
        content = main_file_path.read_text(encoding='utf-8', errors='ignore')
        content = content.replace('~', ' ')
        content = _strip_latex_comments(content)
    except Exception as e:
        logger.error(f"Could not read file {main_file_path}: {e}")
        return f"% [Error reading file: {main_file_path.name}]"

    input_pattern = re.compile(r'\\(?:input|include|subfile)\b(?:\s*\{([^}]+)\}|\s+([^\s%]+))')

    def replacer(match):
        relative_path_str = match.group(1) or match.group(2)
        if not relative_path_str.endswith('.tex'):
            relative_path_str += '.tex'
        
        included_file_path = main_file_path.parent / relative_path_str
        return _resolve_and_flatten_tex(included_file_path, processed_files)

    return input_pattern.sub(replacer, content)


def _inlines_to_text(inlines: List[Dict[str, Any]], math_store: Optional[Dict[str, str]] = None) -> str:
    """Converts a list of Pandoc inline elements to a plain text string."""
    text = ""
    if not isinstance(inlines, list):
        return text

    for inline in inlines:
        if not isinstance(inline, dict):
            continue
        t = inline.get('t')
        c = inline.get('c')
        if t == 'Str':
            text += c
        elif t == 'Space':
            text += ' '
        elif t in ('Emph', 'Strong', 'Strikeout', 'Superscript', 'Subscript', 'Quoted', 'SmallCaps', 'Underline'):
            # For Quoted, c is [QuoteType, [Inlines]]
            content = c[1] if t == 'Quoted' else c
            text += _inlines_to_text(content, math_store)
        elif t == 'Span':
            # c is [Attr, [Inlines]]
            text += _inlines_to_text(c[1], math_store)
        elif t == 'Link':
            # c is [Attr, [Inlines], Target]
            text += _inlines_to_text(c[1], math_store)
        elif t == 'Code':
            # c is [Attr, String]
            text += c[1]
        elif t == 'Math':
            # c is [MathType, String]
            if config.KEEP_MATH:
                math_content = c[1]
                if math_store is not None:
                    key = f"@@MATH{len(math_store)}@@"
                    math_store[key] = math_content
                    text += key
                else:
                    text += math_content
        elif t == 'Note':
            if config.KEEP_FOOTNOTES:
                note_text = _process_blocks_for_text(c, math_store, section_filter=None, state={"skip_level": None})
                note_text = note_text.strip()
                if note_text:
                    text += f" [Footnote: {note_text}]"
        elif t in ('SoftBreak', 'LineBreak'):
            text += ' '
        # Ignore other types like 'Image', 'RawInline', 'Cite'

    return text


def _extract_caption_text(caption: Any, math_store: Optional[Dict[str, str]] = None) -> str:
    if not caption:
        return ""
    if isinstance(caption, dict) and caption.get('t') == 'Caption':
        short, long = caption.get('c', [None, None])
        if isinstance(long, list):
            return _process_blocks_for_text(long, math_store, section_filter=None, state={"skip_level": None}).strip()
        if isinstance(short, list):
            return _inlines_to_text(short, math_store).strip()
        return ""
    if isinstance(caption, list):
        if caption and isinstance(caption[0], dict) and caption[0].get('t'):
            return _inlines_to_text(caption, math_store).strip()
        if caption and isinstance(caption[0], dict) and caption[0].get('t') in (
            'Para', 'Plain', 'Header', 'BlockQuote', 'Div', 'BulletList', 'OrderedList'
        ):
            return _process_blocks_for_text(caption, math_store, section_filter=None, state={"skip_level": None}).strip()
        if len(caption) >= 2:
            return _extract_caption_text(caption[1], math_store)
    return ""


def _process_blocks_for_text(
    blocks: List[Dict[str, Any]],
    math_store: Optional[Dict[str, str]] = None,
    section_filter: Optional[List[str]] = None,
    state: Optional[Dict[str, Optional[int]]] = None
) -> str:
    """Recursively processes Pandoc blocks to extract only plain text."""
    text = ""
    if state is None:
        state = {"skip_level": None}

    for block in blocks:
        if not isinstance(block, dict):
            continue
        t = block.get('t')
        c = block.get('c')

        if t == 'Header':
            # c is [level, Attr, [Inlines]]
            level = c[0]
            header_text = _inlines_to_text(c[2], math_store)

            if state["skip_level"] is not None and level <= state["skip_level"]:
                state["skip_level"] = None

            if state["skip_level"] is not None:
                continue

            if section_filter and _is_excluded_section(header_text):
                state["skip_level"] = level
                continue

            text += header_text + '\n\n'
            continue

        if state["skip_level"] is not None:
            continue

        if t == 'Para':
            # c is [[Inlines]]
            text += _inlines_to_text(c, math_store) + '\n\n'
        elif t == 'Plain':
            # c is [[Inlines]]
            text += _inlines_to_text(c, math_store) + '\n'
        elif t == 'BulletList':
            # c is [[[Blocks]]]
            for list_item in c:
                text += "- " + _process_blocks_for_text(list_item, math_store, section_filter, state).strip() + '\n'
            text += '\n'
        elif t == 'OrderedList':
            # c is [ListAttributes, [[[Blocks]]]]
            items = c[1] if isinstance(c, list) and len(c) > 1 else []
            for idx, list_item in enumerate(items, 1):
                item_text = _process_blocks_for_text(list_item, math_store, section_filter, state).strip()
                if item_text:
                    text += f"{idx}. {item_text}\n"
            text += '\n'
        elif t == 'DefinitionList':
            # c is [ [ [Inlines], [ [Blocks] ] ], ... ]
            for term, definitions in c:
                term_text = _inlines_to_text(term, math_store).strip()
                if term_text:
                    text += f"{term_text}\n"
                for definition_blocks in definitions:
                    def_text = _process_blocks_for_text(definition_blocks, math_store, section_filter, state).strip()
                    if def_text:
                        text += f"- {def_text}\n"
            text += '\n'
        elif t == 'BlockQuote':
            # c is [[Blocks]]
            quoted_text = _process_blocks_for_text(c, math_store, section_filter, state)
            text += '\n'.join([f"> {line}" for line in quoted_text.splitlines()]) + '\n\n'
        elif t == 'Div':
            # c is [Attr, [Blocks]]
            text += _process_blocks_for_text(c[1], math_store, section_filter, state)
        elif t == 'Table':
            caption_text = _extract_caption_text(c[1] if isinstance(c, list) and len(c) > 1 else None, math_store)
            if caption_text:
                text += f"Table: {caption_text}\n\n"
        elif t == 'Figure':
            caption_text = _extract_caption_text(c[1] if isinstance(c, list) and len(c) > 1 else None, math_store)
            if caption_text:
                text += f"Figure: {caption_text}\n\n"
        # Explicitly ignore non-text blocks
        elif t in ('CodeBlock', 'RawBlock', 'HorizontalRule'):
            continue

    return text


def _postprocess_text(text: str, math_store: Optional[Dict[str, str]] = None) -> str:
    """Removes residual LaTeX commands and cleans whitespace while preserving math placeholders."""
    if not text:
        return text

    cleaned = text
    cleaned = cleaned.replace(r'\%', '%').replace(r'\_', '_').replace(r'\&', '&').replace(r'\#', '#').replace(r'\@', '')
    cleaned = re.sub(r'\\[a-zA-Z]+\\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?', '', cleaned)
    cleaned = re.sub(r'\{\s*\}', '', cleaned)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    if math_store:
        for key, value in math_store.items():
            cleaned = cleaned.replace(key, value)

    return cleaned.strip()


def _protect_math(text: str) -> tuple[str, Dict[str, str]]:
    """Protect math segments to avoid being stripped by cleanup."""
    placeholders: Dict[str, str] = {}

    def _store(match: re.Match) -> str:
        key = f"@@MATHPROT{len(placeholders)}@@"
        placeholders[key] = match.group(0)
        return key

    patterns = [
        r'\$\$.*?\$\$',
        r'\\\[.*?\\\]',
        r'\\\(.*?\\\)',
        r'(?<!\\)\$.*?(?<!\\)\$',
    ]
    protected = text
    for pattern in patterns:
        protected = re.sub(pattern, _store, protected, flags=re.DOTALL)

    return protected, placeholders


def _extract_command_arg(text: str, command: str) -> str:
    pattern = r'\\' + re.escape(command) + r'\s*\{'
    match = re.search(pattern, text)
    if not match:
        return ""
    idx = match.end() - 1
    depth = 0
    for i in range(idx, len(text)):
        if text[i] == '{':
            depth += 1
            if depth == 1:
                start = i + 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i]
    return ""


def _fallback_plain_text(content: str) -> str:
    """Fallback extraction when Pandoc fails."""
    body = content
    match_begin = re.search(r'\\begin\{document\}', body, flags=re.IGNORECASE)
    preamble = body[:match_begin.start()] if match_begin else ""
    if match_begin:
        body = body[match_begin.end():]
    match_end = re.search(r'\\end\{document\}', body, flags=re.IGNORECASE)
    if match_end:
        body = body[:match_end.start()]

    body = _strip_cjk_envs(body)
    body = _remove_bibliography_content(body)
    body = _truncate_at_appendix(body)

    title = _extract_command_arg(preamble, "title") if config.KEEP_TITLE else ""
    abstract = ""
    abstract_in_body = False
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', body, flags=re.DOTALL | re.IGNORECASE)
    if abstract_match:
        abstract_in_body = True
    else:
        abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', preamble, flags=re.DOTALL | re.IGNORECASE)

    if abstract_match and config.KEEP_ABSTRACT:
        abstract = abstract_match.group(1).strip()
        if abstract_in_body:
            body = body[:abstract_match.start()] + body[abstract_match.end():]

    if config.KEEP_FOOTNOTES:
        body = re.sub(r'\\footnote\{([^{}]*)\}', r'[Footnote: \1]', body)

    for env in ["equation", "align", "alignat", "eqnarray", "gather", "multline", "flalign"]:
        body = re.sub(rf'\\begin\{{{env}\*?\}}', '$$', body)
        body = re.sub(rf'\\end\{{{env}\*?\}}', '$$', body)

    body = re.sub(r'\\caption\*?(?:\[[^\]]*\])?\{[^{}]*\}', '', body)
    body = re.sub(r'\\(label|ref|eqref|citep?|citet|autocite)\*?\{[^{}]*\}', '', body)

    for env in ["table", "tabular", "tabularx", "longtable", "figure", "algorithm", "algorithmic", "tikzpicture"]:
        body = re.sub(
            rf'\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}',
            '',
            body,
            flags=re.DOTALL | re.IGNORECASE
        )

    body = re.sub(r'\\begin\{[^}]+\}(?:\[[^\]]*\])?', '', body)
    body = re.sub(r'\\end\{[^}]+\}', '', body)

    # Promote section titles to plain text
    body = re.sub(
        r'\\(?:chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?\{([^{}]*)\}',
        r'\n\1\n',
        body
    )

    # Drop common preamble-like commands and environments
    lines = []
    in_brace_block = False
    for line in body.splitlines():
        stripped = line.strip()
        if in_brace_block:
            if "}" in stripped:
                in_brace_block = False
            continue
        if stripped.startswith("{") and "}" not in stripped:
            in_brace_block = True
            continue
        if not stripped:
            lines.append("")
            continue
        if re.fullmatch(r'\[[^\]]*\]', stripped):
            continue
        if stripped in ("{", "}"):
            continue
        if stripped.startswith("{") and stripped.endswith("}"):
            continue
        if stripped in {"\\tiny", "\\scriptsize", "\\footnotesize", "\\small", "\\normalsize", "\\large", "\\Large", "\\LARGE", "\\huge", "\\Huge"}:
            continue
        if stripped.startswith("\\"):
            if stripped.startswith("\\item"):
                lines.append(stripped.replace("\\item", "- ", 1))
            else:
                lines.append(stripped)
        else:
            lines.append(stripped)
    body = "\n".join(lines)

    header_parts = []
    if title:
        header_parts.append(f"Title: {title}")
    if abstract:
        header_parts.append(f"Abstract:\n{abstract}")
    if header_parts:
        body = "\n\n".join(header_parts) + "\n\n" + body

    protected, math_placeholders = _protect_math(body)
    cleaned = _postprocess_text(protected)
    for key, value in math_placeholders.items():
        cleaned = cleaned.replace(key, value)

    cleaned_lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}") and len(stripped) < 300:
            continue
        if stripped.startswith("{") and len(stripped) < 120:
            continue
        cleaned_lines.append(line)

    filtered_lines = []
    for line in cleaned_lines:
        stripped = line.strip()
        if stripped and len(stripped) <= 80 and _is_excluded_section(stripped):
            break
        filtered_lines.append(line)

    return "\n".join(filtered_lines).strip()


def _run_pandoc_json(cleaned_content: str, extra_args: List[str]) -> str:
    if pypandoc is not None:
        return pypandoc.convert_text(
            cleaned_content,
            "json",
            format="latex",
            extra_args=extra_args,
        )

    import subprocess

    pandoc_cmd = ["pandoc", "--from=latex", "--to=json"] + extra_args
    result = subprocess.run(
        pandoc_cmd,
        input=cleaned_content,
        text=True,
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Pandoc failed to produce JSON AST.")
    return result.stdout


def _extract_pandoc_error_line(error_text: str) -> Optional[int]:
    if not error_text:
        return None
    match = re.search(r'line\s+(\d+)', error_text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _find_paragraph_bounds(lines: List[str], idx: int) -> Tuple[int, int]:
    start = idx
    while start > 0 and lines[start].strip():
        start -= 1
    if not lines[start].strip():
        start += 1

    end = idx
    while end < len(lines) and lines[end].strip():
        end += 1

    return start, end


def _drop_problem_segment(text: str, line_num: int, window: int, attempt: int) -> Tuple[str, str]:
    lines = text.splitlines()
    if not lines:
        return text, "empty"

    idx = max(0, min(len(lines) - 1, line_num - 1))

    if attempt == 0:
        del lines[idx : idx + 1]
        return "\n".join(lines), "line"

    para_start, para_end = _find_paragraph_bounds(lines, idx)
    para_len = max(0, para_end - para_start)

    begin_idx = None
    end_idx = None
    env_name = None

    for i in range(idx, -1, -1):
        match = re.search(r'\\begin\{([^}]+)\}', lines[i])
        if match:
            begin_idx = i
            env_name = match.group(1)
            break

    if env_name:
        end_pattern = re.compile(rf'\\end\{{{re.escape(env_name)}\}}')
        for j in range(idx, len(lines)):
            if end_pattern.search(lines[j]):
                end_idx = j
                break

    candidates: List[Tuple[str, int, int, int]] = []
    if para_len > 0:
        candidates.append(("paragraph", para_start, para_end, para_len))
    if begin_idx is not None and end_idx is not None and end_idx >= begin_idx:
        env_len = end_idx - begin_idx + 1
        candidates.append((f"environment:{env_name}", begin_idx, end_idx + 1, env_len))

    if candidates and attempt in (1, 2):
        candidates.sort(key=lambda item: item[3])
        chosen = candidates[0] if attempt == 1 else candidates[-1]
        _, start, end, _ = chosen
        del lines[start:end]
        return "\n".join(lines), chosen[0]

    scale = max(1, attempt - 2)
    window_size = window * scale
    start = max(0, idx - window_size)
    end = min(len(lines), idx + window_size + 1)
    del lines[start:end]
    return "\n".join(lines), f"window:{window_size}"


def _pandoc_to_ast_with_recovery(cleaned_content: str, extra_args: List[str]) -> Tuple[Optional[Dict[str, Any]], str]:
    max_attempts = max(0, int(getattr(config, "PANDOC_RECOVERY_MAX_ATTEMPTS", 5)))
    window = max(1, int(getattr(config, "PANDOC_RECOVERY_WINDOW", 3)))

    working = cleaned_content
    for attempt in range(max_attempts + 1):
        try:
            pandoc_ast_str = _run_pandoc_json(working, extra_args)
            return json.loads(pandoc_ast_str), working
        except Exception as e:
            error_text = str(e)
            line_num = _extract_pandoc_error_line(error_text)
            if line_num is None or attempt >= max_attempts:
                logger.error(f"Pandoc conversion failed after recovery attempts: {error_text}")
                return None, working

            new_text, strategy = _drop_problem_segment(working, line_num, window, attempt)
            logger.warning(
                f"Pandoc failed at line {line_num}. Dropping with strategy={strategy} and retrying "
                f"(attempt {attempt + 1}/{max_attempts})."
            )
            working = new_text


def parse_project_to_text(main_tex_file: Path) -> str:
    """
    Parses a LaTeX project, flattens it, and converts it to a single plain text string,
    preserving main content while skipping appendix and references.
    """
    if not main_tex_file.exists():
        raise FileNotFoundError(f"Main .tex file not found: {main_tex_file}")

    logger.info(f"Flattening LaTeX project starting from {main_tex_file.name}...")
    flattened_content = _resolve_and_flatten_tex(main_tex_file)
    flattened_content = _strip_latex_comments(flattened_content)
    macros = _extract_simple_macros(flattened_content)
    flattened_content = _strip_macro_definitions(flattened_content)
    flattened_content = _apply_macros(flattened_content, macros)
    flattened_content = _remove_bibliography_content(flattened_content)
    flattened_content = _truncate_at_appendix(flattened_content)
    flattened_content = _strip_cjk_envs(flattened_content)

    # Remove complex \texttt macro if it exists, as it can break Pandoc parsing
    texttt_pattern = re.compile(r'\\renewcommand{\\texttt}.*?^}', re.DOTALL | re.MULTILINE)
    cleaned_content = texttt_pattern.sub('', flattened_content)

    logger.info("Converting flattened LaTeX to Pandoc JSON AST...")
    # Define path to the Lua filter and add it to pandoc's arguments if it exists
    script_dir = Path(__file__).parent
    lua_filter_path = script_dir / "filters" / "remove_commands.lua"

    extra_args = ["--quiet"]
    if lua_filter_path.exists():
        logger.info(f"Using Lua filter: {lua_filter_path}")
        extra_args.append(f"--lua-filter={lua_filter_path}")
    else:
        logger.warning(f"Lua filter not found at {lua_filter_path}. Proceeding without it.")

    ast, recovered_content = _pandoc_to_ast_with_recovery(cleaned_content, extra_args)
    if ast is None:
        # Fallback to a best-effort plain-text extraction on the recovered content
        return _fallback_plain_text(recovered_content)

    cleaned_content = recovered_content

    logger.info("Extracting text from Pandoc AST...")
    
    math_store: Dict[str, str] = {}

    # Extract title and abstract from metadata
    title = _inlines_to_text(ast['meta'].get('title', {}).get('c', []), math_store) if config.KEEP_TITLE else ""
    abstract = _inlines_to_text(ast['meta'].get('abstract', {}).get('c', []), math_store) if config.KEEP_ABSTRACT else ""

    # Extract text from the main body
    body_text = _process_blocks_for_text(
        ast['blocks'],
        math_store=math_store,
        section_filter=config.SECTION_EXCLUDE_KEYWORDS
    )

    full_text = ""
    if title:
        full_text += f"Title: {title}\n\n"
    if abstract:
        full_text += f"Abstract:\n{abstract}\n\n"
    full_text += "--- Body ---\n\n" + body_text

    full_text = _postprocess_text(full_text, math_store)

    logger.info("Successfully extracted text from the LaTeX project.")
    return full_text
