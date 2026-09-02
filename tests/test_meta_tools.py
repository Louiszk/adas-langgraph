"""
Specification tests for meta-system tools that manipulate target agentic systems,
enforce security guardrails, and handle malformed AI-generated code safely.
"""

import textwrap

import pytest
from langgraph.graph import START

from adas_core.virtual_agentic_system import VirtualAgenticSystem
from meta_system.tools import (
    install_package,
    manage_conditional_edge,
    manage_edge,
    manage_node,
    manage_tool,
    manage_utilities,
    set_imports,
    set_state,
)


@pytest.fixture
def meta_state(empty_system: VirtualAgenticSystem) -> dict:
    """Provides state dict wrapping target_agentic_system as expected by meta tools."""
    return {"target_agentic_system": empty_system, "messages": []}


class TestMetaToolsSpecification:
    def test_manage_node_and_tool(self, meta_state: dict, sample_node_researcher: str, sample_tool_calculator: str):
        """Contract: create actions add valid nodes and tools to state['target_agentic_system']."""
        res_node = manage_node(
            action="create",
            name="researcher_node",
            function_code=sample_node_researcher,
            description="Performs research",
            state=meta_state,
        )
        assert "was created successfully" in res_node
        assert "researcher_node" in meta_state["target_agentic_system"].nodes

        res_tool = manage_tool(
            action="create",
            name="calculate_expression",
            function_code=sample_tool_calculator,
            description="Math calculator",
            state=meta_state,
        )
        assert "was created successfully" in res_tool
        assert "calculate_expression" in meta_state["target_agentic_system"].tools

    def test_add_and_delete_standard_edge(self, meta_state: dict, sample_node_researcher: str):
        """Contract: manage_edge create and delete actions must mutate graph edges."""
        manage_node(
            action="create",
            name="researcher_node",
            function_code=sample_node_researcher,
            description="Research",
            state=meta_state,
        )

        add_res = manage_edge(action="create", source=START, target="researcher_node", state=meta_state)
        assert "created successfully" in add_res
        assert (START, "researcher_node") in meta_state["target_agentic_system"].edges

        del_res = manage_edge(action="delete", source=START, target="researcher_node", state=meta_state)
        assert "deleted successfully" in del_res
        assert (START, "researcher_node") not in meta_state["target_agentic_system"].edges

    def test_set_state_meta_tool(self, meta_state: dict, sample_state_class_code: str):
        """Contract: set_state must update the TypedDict AgentState schema of the target system."""
        res = set_state(state_code=sample_state_class_code, state=meta_state)
        assert "defined successfully" in res
        assert "context" in meta_state["target_agentic_system"].state_attributes
        assert meta_state["target_agentic_system"].state_attributes["context"] == "str"

    def test_manage_utilities_create_and_delete(self, meta_state: dict):
        """Utility management uses explicit lifecycle actions and typed deletion targets."""
        util_code = "def clean_token(s: str) -> str:\n    return s.lower()\n"
        res = manage_utilities(action="create", utility_code=util_code, state=meta_state)
        assert "updated successfully" in res
        assert "clean_token" in meta_state["target_agentic_system"].utility_code
        deleted = manage_utilities(
            action="delete",
            definitions=[{"name": "clean_token", "kind": "function"}],
            state=meta_state,
        )
        assert "deleted successfully" in deleted
        assert "clean_token" not in meta_state["target_agentic_system"].utility_code

    def test_manage_conditional_edge_requires_and_persists_explicit_path_map(
        self, meta_state: dict, sample_node_researcher: str
    ):
        """Conditional-edge path maps are authored by the meta-agent and preserved for materialization."""
        manage_node(
            action="create",
            name="researcher_node",
            function_code=sample_node_researcher,
            description="Research",
            state=meta_state,
        )
        condition_code = "def route_research(state: dict) -> str:\n    return 'done'"

        missing_path_map = manage_conditional_edge(
            action="create",
            source="researcher_node",
            path_map={},
            function_code=condition_code,
            state=meta_state,
        )
        assert "explicit path_map" in missing_path_map

        result = manage_conditional_edge(
            action="create",
            source="researcher_node",
            path_map={"done": "END"},
            function_code=condition_code,
            state=meta_state,
        )
        assert "was created successfully" in result
        assert meta_state["target_agentic_system"].conditional_edges["researcher_node"]["path_map"]["done"] == "__end__"

        updated = manage_conditional_edge(
            action="update",
            source="researcher_node",
            path_map={"done": "researcher_node"},
            function_code=condition_code,
            state=meta_state,
        )
        assert "was updated successfully" in updated
        assert meta_state["target_agentic_system"].conditional_edges["researcher_node"]["path_map"] == {
            "done": "researcher_node"
        }

    def test_manage_actions_enforce_lifecycle_contracts(self, meta_state: dict, sample_node_researcher: str):
        """Create and update are intentionally distinct, while delete tolerates a missing item."""
        assert "does not exist" in manage_node(
            action="update",
            name="researcher_node",
            function_code=sample_node_researcher,
            state=meta_state,
        )

        assert "created successfully" in manage_node(
            action="create",
            name="researcher_node",
            function_code=sample_node_researcher,
            description="Research",
            state=meta_state,
        )
        assert "already exists" in manage_node(
            action="create",
            name="researcher_node",
            function_code=sample_node_researcher,
            description="Research",
            state=meta_state,
        )
        assert "updated successfully" in manage_node(
            action="update",
            name="researcher_node",
            function_code=sample_node_researcher.replace("Research completed", "Research refreshed"),
            state=meta_state,
        )
        assert "deleted successfully" in manage_node(
            action="delete",
            name="researcher_node",
            function_code="ignored",
            description="ignored",
            state=meta_state,
        )
        assert "WARNING" in manage_node(action="delete", name="researcher_node", state=meta_state)


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

    def test_manage_node_catches_syntax_error(self, meta_state: dict):
        """Contract: Malformed node or tool Python code must return an ERROR string instead of throwing."""
        malformed_node = textwrap.dedent("""
            def broken_node(state: dict) -> dict
                return {'x': [1, 2}
        """)
        res = manage_node(
            action="create",
            name="broken_node",
            function_code=malformed_node,
            description="Broken syntax node",
            state=meta_state,
        )
        assert "ERROR" in res and "syntax" in res.lower()
