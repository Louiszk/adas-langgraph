"""Unit tests for adas_core.markdown_parser."""

from __future__ import annotations

import json
import pytest

from adas_core.markdown_parser import (
    extract_json_block,
    extract_json_block_optional,
    find_code_blocks,
    find_markdown_fences,
)


def test_find_markdown_fences_basic():
    text = """Some prefix text
```json
{
  "key": "value"
}
```
Some middle text
```python
def foo():
    return 42
```
Suffix text"""
    fences = find_markdown_fences(text)
    assert len(fences) == 2
    assert "key" in fences[0]["content"]
    assert fences[0]["start_line"] == 2
    assert fences[0]["end_line"] == 6
    assert "def foo():" in fences[1]["content"]


def test_find_markdown_fences_handles_non_python_syntax():
    text = """```json
{
  "query": "What's the meaning of $100 and unclosed quote ' inside string?",
  "valid": true,
  "nothing": null
}
```"""
    fences = find_markdown_fences(text)
    assert len(fences) == 1
    parsed = json.loads(fences[0]["content"])
    assert parsed["valid"] is True
    assert parsed["nothing"] is None


def test_find_code_blocks_python():
    text = '```python\ndef hello():\n    """Docstring with ``` inside"""\n    return True\n```'
    blocks = find_code_blocks(text)
    assert len(blocks) == 1
    assert "def hello():" in blocks[0]["content"]


def test_find_code_blocks_extracts_multiple_blocks():
    text = (
        "Here is Python code:\n"
        "```python\n"
        "def foo():\n"
        "    return 42\n"
        "```\n"
        "And another block:\n"
        "```python\n"
        "x = 10 + 20\n"
        "```"
    )
    blocks = find_code_blocks(text)
    assert len(blocks) == 2
    assert "def foo():" in str(blocks[0]["content"])
    assert "x = 10 + 20" in str(blocks[1]["content"])
    assert "start_line" in blocks[0] and "end_line" in blocks[0]


def test_extract_json_block_markdown():
    text = """Here is the config:
```json
{
  "name": "TestSystem",
  "schema_version": "1.0"
}
```
Please inspect it."""
    data = extract_json_block(text)
    assert data["name"] == "TestSystem"
    assert data["schema_version"] == "1.0"


def test_extract_json_block_raw():
    text = 'Prefix {"name": "TestSystem", "count": 5} suffix'
    data = extract_json_block(text)
    assert data["name"] == "TestSystem"
    assert data["count"] == 5


def test_extract_json_block_invalid():
    with pytest.raises(ValueError, match="Could not extract valid JSON object"):
        extract_json_block("No json object here!")


def test_extract_json_block_optional():
    assert extract_json_block_optional("plain text") is None
    res = extract_json_block_optional('```json\n{"status": "ok"}\n```')
    assert res == {"status": "ok"}
