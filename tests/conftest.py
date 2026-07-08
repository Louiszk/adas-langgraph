import textwrap
from typing import Dict, Any, Optional
import pytest
from adas_core.virtual_agentic_system import VirtualAgenticSystem


def add_node_to_system(system: VirtualAgenticSystem, name: str, code: str, description: str = "Test node") -> bool:
    """Helper to register a node function on VirtualAgenticSystem through strict get_function validation."""
    func, parsed = system.get_function(code, "node")
    assert func is not None, f"get_function failed to parse node '{name}': {parsed}"
    return system.create_node(name, description, func, parsed)


def add_tool_to_system(system: VirtualAgenticSystem, name: str, code: str, description: str = "Test tool") -> bool:
    """Helper to register a tool function on VirtualAgenticSystem through strict get_function validation."""
    func, parsed = system.get_function(code, "tool")
    assert func is not None, f"get_function failed to parse tool '{name}': {parsed}"
    return system.create_tool(name, description, func, parsed)


def add_conditional_edge_to_system(
    system: VirtualAgenticSystem, source: str, code: str, path_map: Optional[Dict[str, Any]] = None
) -> bool:
    """Helper to register a conditional edge on VirtualAgenticSystem through strict get_function validation."""
    func, parsed = system.get_function(code, "router")
    assert func is not None, f"get_function failed to parse router for source '{source}': {parsed}"
    return system.create_conditional_edge(source, func, parsed, path_map)


@pytest.fixture
def empty_system() -> VirtualAgenticSystem:
    """Returns a fresh, unpopulated VirtualAgenticSystem instance."""
    return VirtualAgenticSystem(system_name="TestSystem")


@pytest.fixture
def sample_node_researcher() -> str:
    """Returns source code for a valid researcher node."""
    return textwrap.dedent("""
        def researcher_node(state: dict) -> dict:
            \"\"\"Performs research and updates messages.\"\"\"
            messages = state.get("messages", [])
            messages.append({"role": "assistant", "content": "Research completed"})
            return {"messages": messages}
    """).strip()


@pytest.fixture
def sample_node_writer() -> str:
    """Returns source code for a valid writer node."""
    return textwrap.dedent("""
        def writer_node(state: dict) -> dict:
            \"\"\"Drafts response based on state.\"\"\"
            return {"messages": state.get("messages", [])}
    """).strip()


@pytest.fixture
def sample_tool_calculator() -> str:
    """Returns source code for a valid LangChain tool."""
    return textwrap.dedent("""
        @tool
        def calculate_expression(expression: str) -> str:
            \"\"\"Calculates a mathematical expression safely.\"\"\"
            return str(eval(expression, {"__builtins__": {}}))
    """).strip()


@pytest.fixture
def sample_state_class_code() -> str:
    """Returns source code for an AgentState TypedDict definition."""
    return textwrap.dedent("""
        class AgentState(TypedDict):
            messages: Annotated[List[AnyMessage], add_messages]
            context: str
            step_count: int
    """).strip()
