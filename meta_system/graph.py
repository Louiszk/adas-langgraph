from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from adas_core.candidate_selection import finalize_best_candidate
from config.logging import get_logger
from config.settings import (
    CANDIDATE_OPTIMIZATION_METRIC,
    CLEANUP_CHECKPOINTS_ON_FINALIZATION,
)
from meta_system.nodes import (
    formatting_function,
    initial_test_runner_function,
    meta_agent_function,
    tool_execution,
)
from meta_system.state import MetaState

logger = get_logger("meta_system.graph")


def design_completed_condition(state: MetaState | dict[str, Any]) -> str:
    """Routes to END if design is completed, otherwise to MetaAgent."""
    messages = state.get("messages", [])
    iteration = len([msg for msg in messages if isinstance(msg, AIMessage)])
    max_iterations = state.get("max_iterations", 30)
    exhausted = iteration > max_iterations
    if exhausted and any(
        candidate.get("dev_pass_rate", 0.0) == 1.0
        or ((candidate.get("total_count") or 0) > 0 and candidate.get("passed_count") == candidate.get("total_count"))
        for candidate in state.get("candidates", [])
        if isinstance(candidate, dict)
    ):
        state["design_completed"] = True

    if state.get("design_completed", False) or exhausted:
        try:
            finalize_best_candidate(
                state,
                preference=state.get("optimization_metric") or CANDIDATE_OPTIMIZATION_METRIC,
                cleanup_checkpoints=CLEANUP_CHECKPOINTS_ON_FINALIZATION,
            )
        except Exception as e:
            logger.error(f"Error during final system save: {e!r}")

        return END

    return "MetaAgent"


def create_meta_workflow():
    """Assembles and compiles the StateGraph workflow for the meta-system."""
    graph = StateGraph(MetaState)

    # Nodes
    graph.add_node("Formatting", formatting_function)
    graph.add_node("InitialTestRunner", initial_test_runner_function)
    graph.add_node("MetaAgent", meta_agent_function)
    graph.add_node("ToolExecution", tool_execution)

    # Edges
    graph.add_edge(START, "Formatting")
    graph.add_edge("Formatting", "InitialTestRunner")
    graph.add_edge("InitialTestRunner", "MetaAgent")
    graph.add_edge("MetaAgent", "ToolExecution")

    # Conditional Edges
    graph.add_conditional_edges(
        "ToolExecution",
        design_completed_condition,
        path_map={
            "MetaAgent": "MetaAgent",
            END: END,
        },
    )

    return graph.compile()


workflow = create_meta_workflow()
