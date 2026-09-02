import os

import dill as pickle
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from adas_core.logging_config import get_logger
from adas_core.materialize import materialize_system
from meta_system.config import MAX_HARDENING_STEPS
from meta_system.nodes import (
    formatting_function,
    initial_test_runner_function,
    meta_agent_function,
    tool_execution,
    validation_function,
)
from meta_system.state import MetaState

logger = get_logger("meta_system.graph")


def hardening_condition(state: MetaState) -> str:
    """Routes to Validation for test hardening or to MetaAgent to start design."""
    if not state.get("optimize"):
        return "MetaAgent"

    passed = state.get("hardening_passed", False)
    steps = state.get("hardening_steps", 0)

    if passed:
        if steps < MAX_HARDENING_STEPS:
            return "Validation"
        else:
            return END
    else:
        return "MetaAgent"


def design_completed_condition(state: MetaState) -> str:
    """Routes to EndDesign if design is completed, otherwise to MetaAgent."""
    messages = state.get("messages", [])
    iteration = len([msg for msg in messages if isinstance(msg, AIMessage)])
    if state.get("design_completed", False) or iteration > state.get("max_iterations", 30):
        try:
            target_agentic_system = state.get("target_agentic_system")
            if target_agentic_system is None:
                return END
            num_test_cases = len(state.get("validation_code_snippets", [])) * 3
            code_dir = "sandbox/workspace/generated_systems"
            escaped_name = target_agentic_system.escaped_name
            base_path = os.path.join(code_dir, escaped_name)
            best_checkpoint_path = None
            final_system_path = f"{base_path}.pkl"

            # Check for checkpoints in order of preference
            checkpoint_paths_to_check = [
                f"{base_path}_checkpoint_{j}.pkl" for j in reversed(range(1, num_test_cases + 1))
            ]

            for path in checkpoint_paths_to_check:
                if os.path.exists(path):
                    best_checkpoint_path = path
                    break

            if best_checkpoint_path:
                logger.info(f"Finalizing system from best checkpoint: {os.path.basename(best_checkpoint_path)}")
                os.rename(best_checkpoint_path, final_system_path)
                with open(final_system_path, "rb") as f:
                    final_system_object = pickle.load(f)

                materialize_system(final_system_object, output_dir=code_dir)

                # Clean up any other partial checkpoints that might remain
                for path in checkpoint_paths_to_check:
                    if os.path.exists(path):
                        os.remove(path)
            else:
                # Fallback: No checkpoints exist, save the current (likely broken) state
                logger.warning("No checkpoints found. Saving current system state as final version.")
                with open(final_system_path, "wb") as f:
                    pickle.dump(target_agentic_system, f)

                materialize_system(target_agentic_system, output_dir=code_dir)

        except Exception as e:
            logger.error(f"Error during final system save: {e!r}")

        return END

    return "MetaAgent"


def create_meta_workflow():
    """Assembles and compiles the StateGraph workflow for the meta-system."""
    graph = StateGraph(MetaState)

    # Nodes
    graph.add_node("Formatting", formatting_function)
    graph.add_node("Validation", validation_function)
    graph.add_node("InitialTestRunner", initial_test_runner_function)
    graph.add_node("MetaAgent", meta_agent_function)
    graph.add_node("ToolExecution", tool_execution)

    # Edges
    graph.add_edge(START, "Formatting")
    graph.add_edge("Formatting", "Validation")
    graph.add_edge("Validation", "InitialTestRunner")
    graph.add_edge("MetaAgent", "ToolExecution")

    # Conditional Edges
    graph.add_conditional_edges(
        "InitialTestRunner",
        hardening_condition,
        path_map={
            "MetaAgent": "MetaAgent",
            "Validation": "Validation",
            END: END,
        },
    )

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
