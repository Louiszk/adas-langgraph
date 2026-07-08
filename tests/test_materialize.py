"""
Specification tests for code materialization and AST module generation.
Verifies that generated Python code is syntactically valid and contains complete graph construction.
"""

import ast
import tempfile
import textwrap
import os
import pytest
from langgraph.graph import START, END
from adas_core.virtual_agentic_system import VirtualAgenticSystem
from adas_core.materialize import materialize_system, get_function_name
from tests.conftest import add_node_to_system, add_tool_to_system, add_conditional_edge_to_system


class TestGetFunctionNameSpecification:
    def test_extracts_function_name_via_ast(self):
        """Must correctly extract function name from valid Python def statement."""
        code = "def my_custom_agent_node(state: dict) -> dict:\n    return state"
        name = get_function_name(code)
        assert name == "my_custom_agent_node"

    def test_raises_value_error_on_invalid_code(self):
        """Must raise ValueError when source string has no function definition."""
        with pytest.raises(ValueError, match="Could not find function definition"):
            get_function_name("not a function def")


class TestMaterializeSystemSpecification:
    @pytest.fixture
    def fully_configured_system(
        self, sample_node_researcher: str, sample_node_writer: str, sample_tool_calculator: str
    ) -> VirtualAgenticSystem:
        system = VirtualAgenticSystem("MaterializedSystem_v1")
        system.set_state_attributes({"messages": "Annotated[List[AnyMessage], add_messages]", "query": "str"})
        system.upsert_utility_code("def format_prompt(txt: str) -> str:\n    return txt.strip()\n")
        add_tool_to_system(system, "calculate_expression", sample_tool_calculator, "Calculator Tool")
        add_node_to_system(system, "researcher_node", sample_node_researcher, "Researcher Node")
        add_node_to_system(system, "writer_node", sample_node_writer, "Writer Node")

        system.create_edge(START, "researcher_node")
        system.create_edge("researcher_node", "writer_node")
        system.create_edge("writer_node", END)
        return system

    def test_materialized_output_is_syntactically_valid_python(self, fully_configured_system: VirtualAgenticSystem):
        """
        Contract invariant: Any code emitted by materialize_system MUST be syntactically valid Python.
        materialize_system returns the generated Python source string directly and writes to output_dir.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            code_content = materialize_system(fully_configured_system, output_dir=tmpdir)
            file_path = os.path.join(tmpdir, fully_configured_system.escaped_name + ".py")
            assert os.path.exists(file_path)

            parsed_ast = ast.parse(code_content)
            assert isinstance(parsed_ast, ast.Module)

    def test_materialized_code_contains_required_graph_components(self, fully_configured_system: VirtualAgenticSystem):
        """
        Contract verification: The generated script must define AgentState, StateGraph, nodes, edges, tools, and compilation.
        """
        code_content = materialize_system(fully_configured_system, output_dir=None)

        assert "class AgentState(TypedDict):" in code_content
        assert "messages: Annotated[List[AnyMessage], add_messages]" in code_content
        assert "query: str" in code_content

        assert "def format_prompt(" in code_content

        assert "def calculate_expression(" in code_content
        assert "def researcher_node(" in code_content
        assert "def writer_node(" in code_content

        assert "StateGraph(AgentState)" in code_content
        assert 'agentic_system_graph.add_node("researcher_node", researcher_node)' in code_content
        assert 'agentic_system_graph.add_node("writer_node", writer_node)' in code_content
        assert 'agentic_system_graph.add_edge(START, "researcher_node")' in code_content
        assert 'agentic_system_graph.add_edge("researcher_node", "writer_node")' in code_content
        assert 'agentic_system_graph.add_edge("writer_node", END)' in code_content

        assert "compile()" in code_content

    def test_materialize_conditional_edges_routing(self):
        """
        Contract verification: Conditional edges must emit add_conditional_edges with correct router symbol and path map.
        """
        system = VirtualAgenticSystem("ConditionalSystem")
        add_node_to_system(system, "node_a", "def node_a(state): return state", "Node A")
        add_node_to_system(system, "node_b", "def node_b(state): return state", "Node B")
        system.create_edge(START, "node_a")

        router_code = textwrap.dedent("""
            def route_decision(state: dict) -> str:
                return 'node_b'
        """)
        add_conditional_edge_to_system(system, "node_a", router_code, {"b": "node_b", "end": END})

        code_content = materialize_system(system, output_dir=None)

        assert "def route_decision(" in code_content
        assert "agentic_system_graph.add_conditional_edges" in code_content
        assert '"node_a"' in code_content
        assert "route_decision" in code_content
