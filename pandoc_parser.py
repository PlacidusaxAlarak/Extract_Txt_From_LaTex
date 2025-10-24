# pandoc_parser.py

import logging
import json
import pypandoc
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger(__name__)


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
    except Exception as e:
        logger.error(f"Could not read file {main_file_path}: {e}")
        return f"% [Error reading file: {main_file_path.name}]"

    input_pattern = re.compile(r'(?<!%)\s*\\(?:input|include)\{([^}]+)\}')

    def replacer(match):
        relative_path_str = match.group(1)
        if not relative_path_str.endswith('.tex'):
            relative_path_str += '.tex'
        
        included_file_path = main_file_path.parent / relative_path_str
        return _resolve_and_flatten_tex(included_file_path, processed_files)

    return input_pattern.sub(replacer, content)


def _inlines_to_text(inlines: List[Dict[str, Any]]) -> str:
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
        elif t in ('Emph', 'Strong', 'Strikeout', 'Superscript', 'Subscript', 'Quoted'):
            # For Quoted, c is [QuoteType, [Inlines]]
            content = c[1] if t == 'Quoted' else c
            text += _inlines_to_text(content)
        elif t == 'Link':
            # c is [Attr, [Inlines], Target]
            text += _inlines_to_text(c[1])
        elif t == 'Code':
            # c is [Attr, String]
            text += c[1]
        elif t == 'Math':
            # c is [MathType, String]
            text += c[1]
        elif t == 'LineBreak':
            text += '\n'
        # Ignore other types like 'Image', 'RawInline', 'Note'

    return text


def _process_blocks_for_text(blocks: List[Dict[str, Any]]) -> str:
    """Recursively processes Pandoc blocks to extract only plain text."""
    text = ""
    for block in blocks:
        t = block.get('t')
        c = block.get('c')

        if t == 'Header':
            # c is [level, Attr, [Inlines]]
            text += _inlines_to_text(c[2]) + '\n\n'
        elif t == 'Para':
            # c is [[Inlines]]
            text += _inlines_to_text(c) + '\n\n'
        elif t == 'Plain':
            # c is [[Inlines]]
            text += _inlines_to_text(c) + '\n'
        elif t in ('BulletList', 'OrderedList'):
            # c is [[[Blocks]]]
            for list_item in c:
                text += "- " + _process_blocks_for_text(list_item)
        elif t == 'BlockQuote':
            # c is [[Blocks]]
            quoted_text = _process_blocks_for_text(c)
            # Add a '>' prefix to each line
            text += '\n'.join([f"> {line}" for line in quoted_text.splitlines()]) + '\n\n'
        elif t == 'Div':
            # c is [Attr, [Blocks]]
            text += _process_blocks_for_text(c[1])
        # Explicitly ignore non-text blocks
        elif t in ('Table', 'Figure', 'CodeBlock', 'RawBlock', 'HorizontalRule'):
            continue

    return text


def parse_project_to_text(main_tex_file: Path) -> str:
    """
    Parses a LaTeX project, flattens it, and converts it to a single plain text string,
    ignoring all images, tables, and code blocks.
    """
    if not main_tex_file.exists():
        raise FileNotFoundError(f"Main .tex file not found: {main_tex_file}")

    logger.info(f"Flattening LaTeX project starting from {main_tex_file.name}...")
    flattened_content = _resolve_and_flatten_tex(main_tex_file)

    # Remove complex \texttt macro if it exists, as it can break Pandoc parsing
    texttt_pattern = re.compile(r'\\renewcommand{\\texttt}.*?^}', re.DOTALL | re.MULTILINE)
    cleaned_content = texttt_pattern.sub('', flattened_content)

    logger.info("Converting flattened LaTeX to Pandoc JSON AST...")
    try:
        # Define path to the Lua filter and add it to pandoc's arguments if it exists
        script_dir = Path(__file__).parent
        lua_filter_path = script_dir / "filters" / "remove_commands.lua"

        extra_args = ['--quiet']
        if lua_filter_path.exists():
            logger.info(f"Using Lua filter: {lua_filter_path}")
            extra_args.append(f'--lua-filter={lua_filter_path}')
        else:
            logger.warning(f"Lua filter not found at {lua_filter_path}. Proceeding without it.")

        # We don't need to extract media, so the command is simpler
        pandoc_ast_str = pypandoc.convert_text(
            cleaned_content,
            'json',
            format='latex',
            extra_args=extra_args
        )
        ast = json.loads(pandoc_ast_str)
    except Exception as e:
        logger.error(f"Pandoc conversion to JSON AST failed: {e}", exc_info=True)
        # As a fallback, return the raw flattened content
        return cleaned_content

    logger.info("Extracting text from Pandoc AST...")
    
    # Extract title and abstract from metadata
    title = _inlines_to_text(ast['meta'].get('title', {}).get('c', []))
    abstract = _inlines_to_text(ast['meta'].get('abstract', {}).get('c', []))

    # Extract text from the main body
    body_text = _process_blocks_for_text(ast['blocks'])

    full_text = ""
    if title:
        full_text += f"Title: {title}\n\n"
    if abstract:
        full_text += f"Abstract:\n{abstract}\n\n"
    full_text += "--- Body ---\\n\n" + body_text

    logger.info("Successfully extracted text from the LaTeX project.")
    return full_text.strip()
