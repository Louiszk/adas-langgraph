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
    workflow,
)
from meta_system.nodes import (
    formatting_function,
    initial_test_runner_function,
    normalize_response_content,
)
from meta_system.prompts import (
    agentic_system_documentation,
    build_meta_agent_prompt,
    decorator_reminder,
    decorator_tool_prompt,
    test_reminder,
    trimming_message,
)
from meta_system.state import MetaState
from meta_system.tools import function_signatures, ignored_nodes_message, tools


class TestMetaSystemWorkflow:
    def test_workflow_compiled_and_has_nodes(self):
        """Workflow must compile successfully and have all required nodes."""
        assert workflow is not None
        compiled_nodes = workflow.nodes
        assert "Formatting" in compiled_nodes
        assert "InitialTestRunner" in compiled_nodes
        assert "MetaAgent" in compiled_nodes
        assert "ToolExecution" in compiled_nodes
        assert "Validation" not in compiled_nodes

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
        assert "ADAS_INPUT_DIR" in agentic_system_documentation
        assert "@@decorator_name" in decorator_reminder
        assert "## Plan & Diagnosis" in decorator_reminder
        assert "Analyze these test result logs" in test_reminder
        assert "{trimmed_iterations}" in trimming_message
        assert "@@test_system()" in decorator_tool_prompt

    def test_build_meta_agent_prompt(self):
        """build_meta_agent_prompt must embed function signatures and documentation."""
        prompt = build_meta_agent_prompt(function_signatures)
        assert "You are an expert AI software engineer" in prompt
        assert "@@manage_node" in prompt
        assert "LangGraph + ADAS Core Reference" in prompt
        assert "## Plan & Diagnosis" in prompt
        assert "Understanding the TaskSpec Contract" in prompt


class TestMetaSystemHelpers:
    def test_normalize_response_content(self):
        """normalize_response_content must handle strings, lists of dicts, and None."""
        assert normalize_response_content("hello") == "hello"
        assert normalize_response_content([{"text": "foo"}, {"text": "bar"}]) == "foo bar"
        assert normalize_response_content(None) == ""

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

    def test_design_completed_condition_routes(self):
        """design_completed_condition routes to END when design is completed or iteration limit exceeded."""
        sys = VirtualAgenticSystem("Dummy")
        assert design_completed_condition({"design_completed": False, "messages": []}) == "MetaAgent"
        assert (
            design_completed_condition({"design_completed": True, "messages": [], "target_agentic_system": sys}) == END
        )

    def test_initial_test_runner_function_skips_when_not_optimize(self):
        """initial_test_runner_function returns empty dict when optimize is not enabled."""
        res = initial_test_runner_function({"optimize": False})
        assert res == {}

    def test_initial_test_runner_function_when_optimize_all_pass(self, monkeypatch):
        """When initial system already passes all tests, prompts to optimize robustness and efficiency."""

        class MockTool:
            def invoke(self, kwargs):
                state = kwargs["state"]
                state["test_metrics"] = {"passed": 3, "total": 3, "pass_rate": 1.0}
                return "The system passed 3/3 tests."

        monkeypatch.setitem(tools, "TestSystem", MockTool())
        state: MetaState = {"optimize": True}
        res = initial_test_runner_function(state)
        assert "verbose_initial_test_results" in res
        content = str(res["verbose_initial_test_results"].content)
        assert "already passes all 3/3" in content
        assert "optimize the system for greater robustness, token efficiency" in content
        assert res.get("test_metrics") == {"passed": 3, "total": 3, "pass_rate": 1.0}

    def test_initial_test_runner_function_when_optimize_partial_pass(self, monkeypatch):
        """When initial system fails some tests, prompts to pass all tests."""

        class MockTool:
            def invoke(self, kwargs):
                state = kwargs["state"]
                state["test_metrics"] = {"passed": 1, "total": 3, "pass_rate": 0.33}
                return "The system passed 1/3 tests."

        monkeypatch.setitem(tools, "TestSystem", MockTool())
        state: MetaState = {"optimize": True}
        res = initial_test_runner_function(state)
        assert "verbose_initial_test_results" in res
        content = str(res["verbose_initial_test_results"].content)
        assert "passed 1/3" in content
        assert "achieving passing tests for all 3" in content
