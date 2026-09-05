"""Markdown parsing utilities for code blocks, fences, and JSON payloads."""

from __future__ import annotations

import io
import json
import re
import tokenize
from typing import Any


def find_markdown_fences(markdown: str) -> list[dict[str, Any]]:
    """Extract code block contents and line bounds from markdown without Python-specific tokenization.

    Returns a list of dicts:
        - "content": the text inside the code block
        - "start_line": 1-indexed line of the opening fence
        - "end_line": 1-indexed line of the closing fence
    """
    lines = markdown.splitlines()
    blocks: list[dict[str, Any]] = []
    in_block = False
    current_lines: list[str] = []
    start_line: int | None = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not in_block:
            if stripped.startswith("```"):
                in_block = True
                current_lines = []
                start_line = i + 1
        else:
            if stripped == "```" or stripped.startswith("```"):
                in_block = False
                blocks.append(
                    {
                        "content": "\n".join(current_lines),
                        "start_line": start_line or (i + 1),
                        "end_line": i + 1,
                    }
                )
                current_lines = []
            else:
                current_lines.append(line)

    return blocks


def find_code_blocks(markdown: str) -> list[dict[str, Any]]:
    """Find Python code blocks in markdown using tokenize to handle nested Python quotes and comments.

    Returns a list of dicts:
        - "content": the code block string
        - "start_line": 1-indexed line of opening fence
        - "end_line": 1-indexed line of closing fence
    """
    lines = markdown.splitlines()
    found_blocks: list[dict[str, Any]] = []

    in_code_block = False
    current_block_content: list[str] = []
    current_block_start_line: int | None = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not in_code_block:
            if stripped.startswith("```"):
                in_code_block = True
                current_block_content = []
                current_block_start_line = i + 1
        else:
            if stripped == "```":
                block_so_far = "\n".join(current_block_content)

                try:
                    list(tokenize.generate_tokens(io.StringIO(block_so_far).readline))

                    in_code_block = False
                    found_blocks.append(
                        {
                            "content": block_so_far,
                            "start_line": current_block_start_line,
                            "end_line": i + 1,
                        }
                    )
                    current_block_content = []

                except tokenize.TokenError:
                    current_block_content.append(line)
            else:
                current_block_content.append(line)

    return found_blocks


def extract_json_block(content: str) -> dict[str, Any]:
    """Extract a valid JSON dictionary from Markdown code blocks or raw text."""
    blocks = find_markdown_fences(content)

    # 1. Check code blocks for TaskSpec dict or structured JSON
    for block in blocks:
        raw = str(block.get("content", "")).strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and ("schema_version" in parsed or "name" in parsed):
                return parsed
        except json.JSONDecodeError:
            continue

    # 2. Check any generic code blocks containing valid json dict
    for block in blocks:
        raw = str(block.get("content", "")).strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    # 3. Fall back to outermost { ... }
    content_stripped = content.strip()
    match = re.search(r"\{[\s\S]*\}", content_stripped)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON object from model response: {content[:300]}...")


def extract_json_block_optional(content: str) -> dict[str, Any] | None:
    """Attempt to extract a JSON dictionary from content, returning None if no valid JSON is found."""
    try:
        return extract_json_block(content)
    except ValueError:
        return None
