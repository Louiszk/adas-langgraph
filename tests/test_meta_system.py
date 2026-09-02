"""
Specification tests for the meta_system package.
Verifies workflow assembly, node logic, conditions, prompts, and tool bindings.
"""

import ast

from langgraph.graph import END

from adas_core.virtual_agentic_system import VirtualAgenticSystem
from meta_system.graph import (
    create_meta_workflow,
    design_completed_condition,
    hardening_condition,
    workflow,
)
from meta_system.helpers import ignored_nodes_message, normalize_response_content, parse_validation_code
from meta_system.nodes import formatting_function
from meta_system.prompts import (
    agentic_system_documentation,
    build_meta_agent_prompt,
    decorator_reminder,
    decorator_tool_prompt,
    hardening_prompt,
    test_reminder,
    trimming_message,
    validation_prompt,
)
from meta_system.state import MetaState
from meta_system.tools import function_signatures, tools


class TestMetaSystemWorkflow:
    def test_workflow_compiled_and_has_nodes(self):
        """Workflow must compile successfully and have all required nodes."""
        assert workflow is not None
        compiled_nodes = workflow.nodes
        assert "Formatting" in compiled_nodes
        assert "Validation" in compiled_nodes
        assert "InitialTestRunner" in compiled_nodes
        assert "MetaAgent" in compiled_nodes
        assert "ToolExecution" in compiled_nodes

    def test_create_meta_workflow_factory(self):
        """Factory function must construct a valid, independent compiled workflow."""
        wf = create_meta_workflow()
        assert wf is not None
        assert "Formatting" in wf.nodes

    def test_tools_dictionary_is_populated(self):
        """Registered tools dict must contain all expected decorator tools."""
        expected_tool_names = [
            "InstallPackage",
            "SetImports",
            "SetState",
            "ManageNode",
            "ManageTool",
            "ManageConditionalEdge",
            "ManageEdge",
            "ManageUtilities",
            "TestSystem",
            "EndDesign",
        ]
        for tool_name in expected_tool_names:
            assert tool_name in tools
            assert callable(getattr(tools[tool_name], "invoke", None))


class TestMetaSystemPromptsAndSignatures:
    def test_prompts_contain_core_references(self):
        """Prompt constants must contain essential instructions and documentation."""
        assert "LangGraph + ADAS Core Reference" in agentic_system_documentation
        assert "You validate agentic systems" in validation_prompt
        assert "TARGET_SYSTEM_TEST_CASES" in hardening_prompt
        assert "@@decorator_name" in decorator_reminder
        assert "Analyze these test result logs" in test_reminder
        assert "{trimmed_iterations}" in trimming_message
        assert "@@test_system()" in decorator_tool_prompt

    def test_build_meta_agent_prompt(self):
        """build_meta_agent_prompt must embed function signatures and documentation."""
        prompt = build_meta_agent_prompt(function_signatures)
        assert "You are an expert AI software engineer" in prompt
        assert "@@manage_node" in prompt
        assert "LangGraph + ADAS Core Reference" in prompt


class TestMetaSystemHelpers:
    def test_normalize_response_content(self):
        """normalize_response_content must handle strings, lists of dicts, and None."""
        assert normalize_response_content("hello") == "hello"
        assert normalize_response_content([{"text": "foo"}, {"text": "bar"}]) == "foo bar"
        assert normalize_response_content(None) == ""

    def test_parse_validation_code_valid(self):
        """parse_validation_code must extract executable TARGET_SYSTEM_TEST_CASES and validator."""
        code = """```python
TARGET_SYSTEM_TEST_CASES = [{"x": 1}, {"x": 2}, {"x": 3}]
def validate_target_system_output(idx, state):
    return True, "Passed"
```"""
        block, errors = parse_validation_code(code)
        assert block is not None
        assert errors is None
        assert "TARGET_SYSTEM_TEST_CASES" in block

    def test_parse_validation_code_invalid(self):
        """parse_validation_code must return error when no valid validation block is found."""
        code = "No code blocks here."
        block, errors = parse_validation_code(code)
        assert block is None

    def test_ignored_nodes_message(self):
        """ignored_nodes_message generates readable warning notes for disallowed AST structures."""
        tree = ast.parse("x = 1\ny: int = 2\ndef f(): pass")
        msg = ignored_nodes_message(tree.body)
        assert "Variable assignment for 'x'" in msg
        assert "Typed variable assignment for 'y'" in msg
        assert "FunctionDef 'f'" in msg


class TestMetaSystemNodesAndRouting:
    def test_formatting_function(self):
        """formatting_function formats task statement with max_iterations and resets state."""
        state: MetaState = {
            "initial_task": "Build a math agent --- Specific Validation Instructions --- do not leak",
            "max_iterations": 25,
        }
        res = formatting_function(state)
        assert "messages" in res
        assert "Build a math agent" in str(res["messages"][0].content)
        assert "25 iterations" in str(res["messages"][0].content)
        assert res["system_passed"] is False
        assert res["hardening_steps"] == 0

    def test_hardening_condition_routes(self):
        """hardening_condition must correctly route based on optimize and hardening_passed flags."""
        assert hardening_condition({"optimize": False}) == "MetaAgent"
        assert hardening_condition({"optimize": True, "hardening_passed": False}) == "MetaAgent"
        assert hardening_condition({"optimize": True, "hardening_passed": True, "hardening_steps": 1}) == "Validation"
        assert hardening_condition({"optimize": True, "hardening_passed": True, "hardening_steps": 5}) == END

    def test_design_completed_condition_routes(self):
        """design_completed_condition routes to END when design is completed or iteration limit exceeded."""
        sys = VirtualAgenticSystem("Dummy")
        assert design_completed_condition({"design_completed": False, "messages": []}) == "MetaAgent"
        assert (
            design_completed_condition({"design_completed": True, "messages": [], "target_agentic_system": sys}) == END
        )
