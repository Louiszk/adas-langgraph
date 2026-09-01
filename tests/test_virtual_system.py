"""
Specification and contract tests for VirtualAgenticSystem.
Tests verify behavior invariants rather than implementation details.
"""

import ast
import textwrap

from langgraph.graph import END, START

from adas_core.virtual_agentic_system import VirtualAgenticSystem
from tests.conftest import add_conditional_edge_to_system, add_node_to_system, add_tool_to_system


class TestVirtualAgenticSystemInitialization:
    def test_sanitizes_system_name(self):
        """System name with path separators must be sanitized for safe filesystem usage."""
        system = VirtualAgenticSystem(system_name="Folder/Subfolder\\MySystem:v1")
        assert system.escaped_name == "FolderSubfolderMySystemv1"
        assert system.system_name == "Folder/Subfolder\\MySystem:v1"

    def test_default_state_contains_messages_annotation(self, empty_system: VirtualAgenticSystem):
        """By default, systems must define an Annotated messages field for LangGraph reduction."""
        assert "messages" in empty_system.state_attributes
        assert "add_messages" in empty_system.state_attributes["messages"]


class TestStateManagement:
    def test_set_state_attributes_retains_messages_and_adds_custom_fields(self, empty_system: VirtualAgenticSystem):
        """Setting custom attributes must preserve required 'messages' reducer while adding new fields."""
        custom_attrs = {"query": "str", "iterations": "int"}
        success = empty_system.set_state_attributes(custom_attrs)
        assert success is True
        assert empty_system.state_attributes["query"] == "str"
        assert empty_system.state_attributes["iterations"] == "int"
        assert "messages" in empty_system.state_attributes

    def test_set_state_from_node_ast_parses_typed_dict(self, empty_system: VirtualAgenticSystem):
        """Should accurately extract field annotations from an AST ClassDef representing AgentState."""
        state_code = textwrap.dedent("""
            class CustomAgentState(TypedDict):
                messages: Annotated[List[AnyMessage], add_messages]
                user_id: str
                confidence_score: float
        """)
        module_ast = ast.parse(state_code)
        class_def = module_ast.body[0]
        assert isinstance(class_def, ast.ClassDef)

        empty_system.set_state_from_node(class_def)
        assert "user_id" in empty_system.state_attributes
        assert empty_system.state_attributes["user_id"] == "str"
        assert empty_system.state_attributes["confidence_score"] == "float"
        assert "messages" in empty_system.state_attributes


class TestImportManagement:
    def test_deduplicate_imports_filters_base_and_returns_unique(self, empty_system: VirtualAgenticSystem):
        """
        Contract: deduplicate_imports filters out imports already present in base_imports
        and returns unique sorted new import statements.
        """
        new_imports = [
            "import os",  # Already in base_imports
            "import math",
            "import math",
            "from dataclasses import dataclass",
        ]
        deduped = empty_system.deduplicate_imports(new_imports)
        assert "import os" not in deduped
        assert deduped.count("import math") == 1
        assert "from dataclasses import dataclass" in deduped


class TestComponentLifecycleAndCascadeDeletion:
    def test_create_node_registers_code_and_description(
        self, empty_system: VirtualAgenticSystem, sample_node_researcher: str
    ):
        """Creating a node must register top-level function name, description, and source code."""
        success = add_node_to_system(empty_system, "researcher_node", sample_node_researcher, "Research node")
        assert success is True
        assert "researcher_node" in empty_system.nodes
        assert empty_system.nodes["researcher_node"]["description"] == "Research node"
        assert "def researcher_node" in empty_system.nodes["researcher_node"]["source_code"]

    def test_delete_node_cascades_to_edges(
        self, empty_system: VirtualAgenticSystem, sample_node_researcher: str, sample_node_writer: str
    ):
        """
        Contract: Deleting a node must also remove all standard edges and conditional edges
        that reference or target the deleted node to avoid dangling references.
        """
        add_node_to_system(empty_system, "researcher_node", sample_node_researcher, "Researcher")
        add_node_to_system(empty_system, "writer_node", sample_node_writer, "Writer")

        empty_system.create_edge(START, "researcher_node")
        empty_system.create_edge("researcher_node", "writer_node")
        empty_system.create_edge("writer_node", END)
        condition_code = "def choose_next(state: dict) -> str:\n    return 'retry'"
        add_conditional_edge_to_system(
            empty_system,
            "writer_node",
            condition_code,
            {"retry": "researcher_node", "finish": END},
        )

        assert len(empty_system.edges) == 3

        empty_system.delete_node("researcher_node")

        assert "researcher_node" not in empty_system.nodes
        for src, tgt in empty_system.edges:
            assert src != "researcher_node"
            assert tgt != "researcher_node"
        assert "writer_node" not in empty_system.conditional_edges

        empty_system.delete_node("writer_node")
        assert empty_system.conditional_edges == {}

    def test_create_and_delete_tool(self, empty_system: VirtualAgenticSystem, sample_tool_calculator: str):
        """Creating a tool must register it by function name; deleting must remove it."""
        success = add_tool_to_system(empty_system, "calculate_expression", sample_tool_calculator, "Math tool")
        assert success is True
        assert "calculate_expression" in empty_system.tools

        deleted = empty_system.delete_tool("calculate_expression")
        assert deleted is True
        assert "calculate_expression" not in empty_system.tools


class TestUtilityCodeManagement:
    def test_upsert_utility_code_updates_module(self, empty_system: VirtualAgenticSystem):
        """Adding utility code should store helper functions accessible during materialization."""
        helper_code = textwrap.dedent("""
            def parse_json_helper(raw: str) -> dict:
                import json
                return json.loads(raw)
        """)
        empty_system.upsert_utility_code(helper_code)
        assert "parse_json_helper" in empty_system.utility_code

    def test_delete_utility_definitions_distinguishes_definition_kind(self, empty_system: VirtualAgenticSystem):
        """Typed utility deletion removes only the requested same-named top-level definition."""
        utility_code = (
            "shared_name = 'constant'\n\n"
            "def shared_name(value: str) -> str:\n"
            "    return value\n\n"
            "class shared_name:\n"
            "    pass\n"
        )
        assert empty_system.upsert_utility_code(utility_code) == "Utilities updated successfully."

        deleted_assignment = empty_system.delete_utility_definitions([{"name": "shared_name", "kind": "assignment"}])
        assert "assignment:shared_name" in deleted_assignment
        assert "shared_name = 'constant'" not in empty_system.utility_code
        assert "def shared_name" in empty_system.utility_code
        assert "class shared_name" in empty_system.utility_code

        deleted_function = empty_system.delete_utility_definitions([{"name": "shared_name", "kind": "function"}])
        assert "function:shared_name" in deleted_function
        assert "def shared_name" not in empty_system.utility_code
        assert "class shared_name" in empty_system.utility_code

        deleted_class = empty_system.delete_utility_definitions([{"name": "shared_name", "kind": "class"}])
        assert "class:shared_name" in deleted_class
        assert not empty_system.utility_code

    def test_delete_utility_definition_removes_annotated_assignment(self, empty_system: VirtualAgenticSystem):
        """Annotated top-level assignments use the same assignment deletion kind."""
        assert empty_system.upsert_utility_code("MAX_RETRIES: int = 3") == "Utilities updated successfully."

        result = empty_system.delete_utility_definitions([{"name": "MAX_RETRIES", "kind": "assignment"}])

        assert "assignment:MAX_RETRIES" in result
        assert not empty_system.utility_code
