"""
Specification tests for meta-system tools that manipulate target agentic systems,
enforce security guardrails, and handle malformed AI-generated code safely.
"""

import textwrap
import pytest
from langgraph.graph import START
from adas_core.virtual_agentic_system import VirtualAgenticSystem
from meta_systems.compact_system.tools import (
    upsert_component,
    add_edge,
    delete_edge,
    set_state,
    set_imports,
    upsert_utilities,
    install_package,
)


@pytest.fixture
def meta_state(empty_system: VirtualAgenticSystem) -> dict:
    """Provides state dict wrapping target_agentic_system as expected by meta tools."""
    return {"target_agentic_system": empty_system, "messages": []}


class TestMetaToolsSpecification:
    def test_upsert_component_node_and_tool(
        self, meta_state: dict, sample_node_researcher: str, sample_tool_calculator: str
    ):
        """Contract: upsert_component must add valid nodes or tools to state['target_agentic_system']."""
        res_node = upsert_component(
            component_type="node",
            name="researcher_node",
            function_code=sample_node_researcher,
            description="Performs research",
            state=meta_state,
        )
        assert "was created successfully" in res_node
        assert "researcher_node" in meta_state["target_agentic_system"].nodes

        res_tool = upsert_component(
            component_type="tool",
            name="calculate_expression",
            function_code=sample_tool_calculator,
            description="Math calculator",
            state=meta_state,
        )
        assert "was created successfully" in res_tool
        assert "calculate_expression" in meta_state["target_agentic_system"].tools

    def test_add_and_delete_standard_edge(self, meta_state: dict, sample_node_researcher: str):
        """Contract: add_edge and delete_edge must mutate the graph edges list in state['target_agentic_system']."""
        upsert_component(
            component_type="node",
            name="researcher_node",
            function_code=sample_node_researcher,
            description="Research",
            state=meta_state,
        )

        add_res = add_edge(source=START, target="researcher_node", state=meta_state)
        assert "added successfully" in add_res
        assert (START, "researcher_node") in meta_state["target_agentic_system"].edges

        del_res = delete_edge(source=START, target="researcher_node", state=meta_state)
        assert "deleted successfully" in del_res
        assert (START, "researcher_node") not in meta_state["target_agentic_system"].edges

    def test_set_state_meta_tool(self, meta_state: dict, sample_state_class_code: str):
        """Contract: set_state must update the TypedDict AgentState schema of the target system."""
        res = set_state(state_code=sample_state_class_code, state=meta_state)
        assert "defined successfully" in res
        assert "context" in meta_state["target_agentic_system"].state_attributes
        assert meta_state["target_agentic_system"].state_attributes["context"] == "str"

    def test_upsert_utilities_meta_tool(self, meta_state: dict):
        """Contract: upsert_utilities must append helper functions into utility_code."""
        util_code = "def clean_token(s: str) -> str:\n    return s.lower()\n"
        res = upsert_utilities(utility_code=util_code, state=meta_state)
        assert "updated successfully" in res
        assert "clean_token" in meta_state["target_agentic_system"].utility_code


class TestSecurityAndMalformedInputRejection:
    @pytest.mark.parametrize(
        "malicious_pkg",
        [
            "os; rm -rf /",
            "requests && cat /etc/passwd",
            "numpy | sh",
            "`whoami`",
            "pandas $(reboot)",
            "scipy; id",
        ],
    )
    def test_install_package_rejects_command_injection_payloads(self, malicious_pkg: str, meta_state: dict):
        """
        Security Contract: install_package must validate package name regex and reject any
        command injection or shell metacharacters without executing pip.
        """
        res = install_package(package_name=malicious_pkg, state=meta_state)
        assert "ERROR" in res and "invalid" in res.lower(), f"Expected rejection for '{malicious_pkg}', got: {res}"

    def test_install_package_accepts_valid_package_formats(self, meta_state: dict):
        """Contract: install_package allows standard package names and version specifiers."""
        # Pre-excluded package to verify format check passes before installation check
        res = install_package(package_name="langgraph==1.2.9", state=meta_state)
        assert "is already installed" in res


class TestInvalidSyntaxAndErrorRecovery:
    def test_set_state_catches_syntax_error(self, meta_state: dict):
        """Contract: Malformed state definition code must return an ERROR string instead of throwing."""
        malformed_code = textwrap.dedent("""
            class AgentState(TypedDict)
                messages: List[str]
        """)
        res = set_state(state_code=malformed_code, state=meta_state)
        assert "ERROR" in res and "syntax" in res.lower()

    def test_set_imports_catches_syntax_error(self, meta_state: dict):
        """Contract: Malformed import statements must return an ERROR string instead of throwing."""
        malformed_imports = "from typing import ("  # unclosed parenthesis
        res = set_imports(import_code=malformed_imports, state=meta_state)
        assert "ERROR" in res and "syntax" in res.lower()

    def test_upsert_component_catches_syntax_error(self, meta_state: dict):
        """Contract: Malformed node or tool Python code must return an ERROR string instead of throwing."""
        malformed_node = textwrap.dedent("""
            def broken_node(state: dict) -> dict
                return {'x': [1, 2}
        """)
        res = upsert_component(
            component_type="node",
            name="broken_node",
            function_code=malformed_node,
            description="Broken syntax node",
            state=meta_state,
        )
        assert "ERROR" in res and "syntax" in res.lower()
