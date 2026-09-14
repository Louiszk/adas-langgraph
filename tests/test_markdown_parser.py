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


def test_find_code_blocks_opening_fence_attached_to_text():
    text = (
        "## Actions\n"
        "Create the movie expert node.```python\n"
        '@manage_node(action="create")\n'
        "def movie_expert_node(state):\n"
        "    return state\n"
        "```"
    )
    blocks = find_code_blocks(text)
    assert len(blocks) == 1
    assert "Create the movie expert node" not in blocks[0]["content"]
    assert "```" not in blocks[0]["content"]
    assert blocks[0]["content"] == '@manage_node(action="create")\ndef movie_expert_node(state):\n    return state'


def test_find_markdown_fences_opening_fence_attached_to_text():
    text = 'Some prefix text.```json\n{\n  "key": "value"\n}\n```'
    fences = find_markdown_fences(text)
    assert len(fences) == 1
    assert "Some prefix text" not in fences[0]["content"]
    assert fences[0]["content"] == '{\n  "key": "value"\n}'


def test_find_code_blocks_ignores_inline_triple_backticks_in_text():
    text = "You should use ```code``` as an inline example.\nNow here is real code:\n```python\nx = 42\n```"
    blocks = find_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0]["content"] == "x = 42"


def test_find_code_blocks_ignores_single_line_triple_backticks_and_does_not_invert():
    text = (
        "## Plan & Diagnosis\n"
        "```foo```\n"
        "Here is some text describing the plan.\n"
        "## Actions\n"
        "```python\n"
        "def my_func():\n"
        "    return 100\n"
        "```\n"
        "End of response."
    )
    blocks = find_code_blocks(text)
    assert len(blocks) == 1
    assert "Here is some text" not in blocks[0]["content"]
    assert "## Actions" not in blocks[0]["content"]
    assert blocks[0]["content"] == "def my_func():\n    return 100"


def test_find_markdown_fences_ignores_single_line_triple_backticks():
    text = 'Prefix\n```foo```\nMiddle text\n```json\n{"valid": true}\n```\nSuffix'
    fences = find_markdown_fences(text)
    assert len(fences) == 1
    assert fences[0]["content"] == '{"valid": true}'
