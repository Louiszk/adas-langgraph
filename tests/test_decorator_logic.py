"""
Specification tests for markdown block parsing, decorator tool call extraction, and execution logic.
"""

import textwrap
from dataclasses import dataclass
from typing import Any

from adas_core.decorator_logic import (
    build_decorator_signatures,
    execute_decorator_tool_calls,
    find_code_blocks,
    parse_arguments,
    parse_decorator_tool_calls,
)
from meta_system.tools import (
    manage_conditional_edge,
    manage_edge,
    manage_node,
    manage_tool,
    manage_utilities,
)


class TestMarkdownCodeBlockParsing:
    def test_find_code_blocks_extracts_multiple_blocks(self):
        """Contract: Must extract valid Python code blocks with line bounds using tokenizer."""
        markdown_text = textwrap.dedent("""
            Here is Python code:
            ```python
            def foo():
                return 42
            ```
            And another block:
            ```python
            x = 10 + 20
            ```
        """)
        blocks = find_code_blocks(markdown_text)
        assert len(blocks) == 2
        assert "def foo():" in str(blocks[0]["content"])
        assert "x = 10 + 20" in str(blocks[1]["content"])
        assert "start_line" in blocks[0] and "end_line" in blocks[0]


class TestDecoratorArgumentParsing:
    def test_parse_arguments_positional_and_keyword(self):
        """Must correctly parse mixed positional literals and keyword arguments."""
        args_str = '"node", name="researcher", retry=True, limit=5'
        args, kwargs = parse_arguments(args_str)
        assert args == ("node",)
        assert kwargs == {"name": "researcher", "retry": True, "limit": 5}

    def test_parse_arguments_empty_or_none(self):
        """Empty argument string should safely return empty tuple and dict."""
        assert parse_arguments("") == ((), {})
        assert parse_arguments(None) == ((), {})


class TestDecoratorToolCallExtraction:
    def test_parse_decorator_tool_calls_with_associated_code(self):
        """
        Contract: Decorators starting with @@ inside code blocks (e.g. @@manage_node)
        extract camelCased tool name and attach following code block to kwargs.
        """
        code_block = textwrap.dedent("""
            @@manage_node(action="create", name="analyzer")
            def analyzer(state: dict) -> dict:
                return {"status": "analyzed"}
        """).strip()
        code_related_tools = {"manage_node": "function_code"}
        calls = parse_decorator_tool_calls(code_block, code_related_tools)

        assert len(calls) == 1
        call = calls[0]
        assert call["name"] == "ManageNode"
        assert call["kw_args"]["name"] == "analyzer"
        assert "def analyzer(state: dict)" in call["kw_args"]["function_code"]


class TestDecoratorSignatureBuilding:
    def test_manage_decorator_signatures_hide_framework_and_code_parameters(self):
        """The meta-agent prompt exposes the current management API, not parser internals."""
        signatures = build_decorator_signatures(
            [manage_node, manage_tool, manage_conditional_edge, manage_edge, manage_utilities],
            {
                "manage_node": "function_code",
                "manage_tool": "function_code",
                "manage_conditional_edge": "function_code",
                "manage_utilities": "utility_code",
            },
        )

        assert "@@manage_node(action: Literal['create', 'update', 'delete'], name: str" in signatures
        assert "@@manage_tool(action: Literal['create', 'update', 'delete'], name: str" in signatures
        assert "@@manage_conditional_edge(action: Literal['create', 'update', 'delete'], source: str" in signatures
        assert "@@manage_edge(action: Literal['create', 'delete'], source: str, target: str)" in signatures
        assert "@@manage_utilities(action: Literal['create', 'update', 'delete'], definitions:" in signatures
        assert "function_code" not in signatures
        assert "utility_code" not in signatures
        assert "state:" not in signatures
        assert "Creates, updates, or deletes typed utility definitions." in signatures


class TestDecoratorExecutionEngine:
    def test_execute_decorator_tool_calls_dispatches_to_handlers(self):
        """
        Contract: execute_decorator_tool_calls extracts code blocks and dispatches parsed tool calls
        to matching tool instances in available_tools and returns message + results list.
        """
        execution_log = []

        @dataclass
        class MockTool:
            def func(self, *args: Any, **kwargs: Any) -> str:
                execution_log.append(kwargs)
                return f"Successfully updated {kwargs.get('name')}."

        tools_dict = {"ManageNode": MockTool()}
        markdown_input = textwrap.dedent("""
            ```python
            @@manage_node(action="create", name="filter_node")
            def filter_node(state): return state
            ```
        """)
        state = {}
        human_msg, results = execute_decorator_tool_calls(
            response_content=markdown_input,
            available_tools=tools_dict,
            code_related_tools={"manage_node": "function_code"},
            state=state,
        )

        assert len(execution_log) == 1
        assert execution_log[0]["name"] == "filter_node"
        assert len(results) == 1
        assert results[0][0] == "ManageNode"
        assert human_msg is not None
        assert "Successfully updated filter_node." in str(human_msg.content)
