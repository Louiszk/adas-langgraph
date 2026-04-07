import os
import dill as pickle
import textwrap, inspect
from langgraph.graph import START, END
from typing import Dict, List, Any
from adas_core.virtual_agentic_system import VirtualAgenticSystem
from adas_core.materialize import materialize_system
from meta_systems.compact_system.utilities import *
from meta_systems.compact_system.tools import *
from meta_systems.compact_system.nodes import *
from meta_systems.compact_system.documentation import agentic_system_documentation
from meta_systems.compact_system.configurations import MAX_HARDENING_STEPS

def create_meta_system():
    print(f"\n----- Creating Meta System -----\n")
    
    # Create a virtual agentic system instance
    meta_system = VirtualAgenticSystem("MetaSystem")

    # Add imports
    imports = [
        "from adas_core.helpers import get_filtered_packages, truncate_state, TruncatingStringIO, remove_old_test_results",
        "from adas_core.decorator_logic import execute_decorator_tool_calls, find_code_blocks",
        "from adas_core.virtual_agentic_system import VirtualAgenticSystem",
        "from langgraph.managed.is_last_step import RemainingSteps",
        "from adas_core.materialize import materialize_system",
        "import dill as pickle",
        "import contextlib",
        "import subprocess",
        "import traceback",
        "import time",
        "import ast",
        "import sys",
        "import re"
    ]
    meta_system.set_imports(import_statements=imports)
    
    with open("meta_systems/compact_system/utilities.py", "r") as spf:
        utilities_content = spf.read()
    with open("meta_systems/compact_system/configurations.py", "r") as spf:
        configurations_content = spf.read()

    imported_variables = [
        f"function_signatures = {repr(function_signatures)}",
        f"agentic_system_documentation = {repr(agentic_system_documentation)}",
        f"code_related_tools = {repr(code_related_tools)}"
        ]
    self_contained_content = "\n".join(
        imported_variables +
        configurations_content.split("\n") +
        utilities_content.split("\n")[5:] +
        [textwrap.dedent(inspect.getsource(ignored_nodes_message))]
        )

    meta_system.upsert_utility_code(self_contained_content)
    meta_system.set_state_attributes({
        "target_agentic_system": "VirtualAgenticSystem",
        "verbose_initial_test_results": "Optional[HumanMessage]",
        "initial_test_results": "Optional[HumanMessage]",
        "initial_test_passes": "int",
        "validation_code_snippets": "List[str]",
        "system_passed": "bool",
        "design_completed": "bool",
        "initial_task": "str",
        "designer_task": "HumanMessage",
        "remaining_steps": "RemainingSteps",
        "max_iterations": "int",
        "optimize": "bool",
        "hardening_passed": "bool",
        "hardening_steps": "int"
    })
    
    # --- Tools ---
    
    meta_system.create_tool("InstallPackage", install_package.__name__, install_package)
    meta_system.create_tool("SetImports", set_imports.__name__, set_imports)
    meta_system.create_tool("SetState", set_state.__name__, set_state)
    meta_system.create_tool("UpsertComponent", upsert_component.__name__, upsert_component)
    meta_system.create_tool("DeleteComponent", delete_component.__name__, delete_component)
    meta_system.create_tool("AddEdge", add_edge.__name__, add_edge)
    meta_system.create_tool("DeleteEdge", delete_edge.__name__, delete_edge)
    meta_system.create_tool("UpsertUtilities", upsert_utilities.__name__, upsert_utilities)
    meta_system.create_tool("TestSystem", test_system.__name__, test_system)
    meta_system.create_tool("EndDesign", end_design.__name__, end_design)
    
    # --- Nodes ---
    meta_system.create_node("Formatting", "Formats the initial state.", formatting_function)
    meta_system.create_node("Validation", "Generates or hardens the validation code for the system.", validation_function)
    meta_system.create_node("InitialTestRunner", "Runs tests on the baseline system to find its performance frontier.", initial_test_runner_function)
    meta_system.create_node("MetaAgent", "Iteratively builds the target system.", meta_agent_function)
    meta_system.create_node("ToolExecution", "Executes the decorator tool calls.", tool_execution)

    # --- Graph Edges ---
    meta_system.create_edge(START, "Formatting")
    meta_system.create_edge("Formatting", "Validation")
    meta_system.create_edge("Validation", "InitialTestRunner")
    meta_system.create_edge("MetaAgent", "ToolExecution")
    
    def hardening_router(state: Dict[str, Any]) -> str:
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
            
    meta_system.create_conditional_edge("InitialTestRunner", hardening_router)

    def design_completed_router(state: Dict[str, Any]) -> str:
        """Routes to EndDesign if design is completed, otherwise to MetaAgent."""
        messages = state.get("messages", [])
        iteration = len([msg for msg in messages if isinstance(msg, AIMessage)])
        if state.get("design_completed", False) or iteration > state.get("max_iterations"):
            try:
                target_agentic_system: VirtualAgenticSystem = state.get("target_agentic_system")
                num_test_cases = len(state.get("validation_code_snippets", [])) * 3
                code_dir = "sandbox/workspace/generated_systems"
                escaped_name = target_agentic_system.escaped_name
                base_path = os.path.join(code_dir, escaped_name)
                best_checkpoint_path = None
                final_system_path = f"{base_path}.pkl"

                # Check for checkpoints in order of preference
                checkpoint_paths_to_check = [f"{base_path}_checkpoint_{j}.pkl" for j in reversed(range(1, num_test_cases + 1))]

                for path in checkpoint_paths_to_check:
                    if os.path.exists(path):
                        best_checkpoint_path = path
                        break

                if best_checkpoint_path:
                    print(f"Finalizing system from best checkpoint: {os.path.basename(best_checkpoint_path)}")
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
                    print("No checkpoints found. Saving current system state as final version.")
                    with open(final_system_path, "wb") as f:
                        pickle.dump(target_agentic_system, f)
                        
                    materialize_system(target_agentic_system, output_dir=code_dir)

            except Exception as e:
                print(f"Error during final system save: {repr(e)}")

            return END

        return "MetaAgent"
    
    meta_system.create_conditional_edge("ToolExecution", design_completed_router)
    
    
    # Materialize the MetaSystem itself
    materialize_system(meta_system, output_dir="materialized_meta_system")
    print("----- Materialized Meta System -----")
    return meta_system