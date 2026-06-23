import re
import os
import sys
import time
import ast
import dill as pickle
import subprocess
import traceback
import contextlib
from typing import Dict, List, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from adas_core.llm_wrapper import LargeLanguageModel
from adas_core.virtual_agentic_system import VirtualAgenticSystem
from adas_core.materialize import materialize_system
from adas_core.decorator_logic import build_decorator_signatures
from adas_core.helpers import get_filtered_packages, truncate_state, TruncatingStringIO
from meta_systems.compact_system.utilities import test_reminder
from meta_systems.compact_system.configurations import RECURSION_LIMIT


def ignored_nodes_message(ignored_nodes: List[ast.AST]) -> str:
    """Creates a formatted 'Note:' string for any AST nodes that were ignored by a tool."""
    if not ignored_nodes:
        return ""

    messages = []
    for node in ignored_nodes:
        readable_format = f"A code structure of type '{type(node).__name__}'"

        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            readable_format = f"{type(node).__name__} '{node.name}'"
        elif isinstance(node, ast.Assign):
            if node.targets and isinstance(node.targets[0], ast.Name):
                readable_format = f"Variable assignment for '{node.targets[0].id}'"
            else:
                readable_format = "A variable assignment"
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                readable_format = f"Typed variable assignment for '{node.target.id}'"
            else:
                readable_format = "A typed variable assignment"
        messages.append(readable_format)

    note = "\nNote: The following structure(s) were ignored as they are not allowed in this block: "
    limit = 4
    if len(messages) > limit:
        ignored_list_str = ", ".join(messages[:limit])
        remaining_count = len(messages) - limit
        note += f"[{ignored_list_str}, ... (and {remaining_count} more)]."
    else:
        ignored_list_str = ", ".join(messages)
        note += f"[{ignored_list_str}]"
    return note


def install_package(package_name: str, state: Dict[str, Any]) -> str:
    """
    Installs a Python package into the environment using pip.
    Args:
        package_name: The name of the package to install, optionally with a version specifier (e.g., "numpy", "pandas==2.0.3").
    """
    target_agentic_system: VirtualAgenticSystem = state["target_agentic_system"]
    exclude_packages = [
        "datasets",
        "docker",
        "grpcio-status",
        "langchain-openai",
        "wheel",
        "llm-sandbox",
        "pip",
        "dill",
        "podman",
        "python-dotenv",
        "setuptools",
    ]
    # Validate package name to prevent command injection
    valid_pattern = r"^[a-zA-Z0-9._-]+(\s*[=<>!]=\s*[0-9a-zA-Z.]+)?$"

    if not re.match(valid_pattern, package_name):
        return f"ERROR: Invalid package name format. Package name '{package_name}' contains invalid characters."
    if any((ep in package_name for ep in exclude_packages + ["langgraph", "langchain-core"])):
        return f"{package_name} is already installed."

    # Parse package name to get the canonical name for `pip show`
    name_only_match = re.match(r"^[a-zA-Z0-9._-]+", package_name.strip())
    if not name_only_match:
        return f"ERROR: Could not parse package name from '{package_name}'."
    name_only = name_only_match.group(0).lower()

    try:
        process = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False,
        )

        if process.returncode == 0:
            # After successful installation, get the exact version installed to ensure accuracy
            try:
                show_process = subprocess.run(
                    [sys.executable, "-m", "pip", "show", name_only],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                version = ""
                for line in show_process.stdout.splitlines():
                    if line.startswith("Version:"):
                        version = line.split(":", 1)[1].strip()
                        break

                if version:
                    target_agentic_system.installed_packages[name_only] = f"{name_only}=={version}"
                else:
                    target_agentic_system.installed_packages[name_only] = package_name.strip()
            except Exception:
                target_agentic_system.installed_packages[name_only] = package_name.strip()

            target_agentic_system.packages_info = get_filtered_packages(exclude_packages) + ["langchain-core 0.3.75"]
            return f"Successfully installed {package_name}"
        else:
            return f"ERROR: installing {package_name}:\n{process.stdout}"

    except Exception as e:
        return f"ERROR: installing {package_name}: {str(e)}"


def set_imports(import_code: str, state: Dict[str, Any]) -> str:
    """
    Sets the import statements for the target system. This replaces any existing custom imports.
    The Python code containing the import statements MUST be placed immediately after this decorator line.
    """
    if not import_code:
        return "ERROR: You must provide the import statements code block below the decorator."

    target_agentic_system: VirtualAgenticSystem = state["target_agentic_system"]
    imports_found = []
    ignored_nodes = []
    try:
        tree = ast.parse(import_code)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports_found.append(ast.unparse(node))
            else:
                ignored_nodes.append(node)

        main_message = target_agentic_system.set_imports(imports_found)
        note = ignored_nodes_message(ignored_nodes)
        return f"{main_message}{note}"

    except SyntaxError as e:
        return f"ERROR: Invalid Python syntax in imports block: {e}"
    except Exception as e:
        return f"ERROR: setting imports: {repr(e)}"


def set_state(state_code: str, state: Dict[str, Any]) -> str:
    """
    Defines the AgentState for the target system. This decorator should be used at the beginning of the design process.
    If called again, it will completely replace the previous AgentState definition.
    The Python code defining the AgentState class MUST be placed immediately after this decorator line.
    """
    if not state_code:
        return "ERROR: You must provide the AgentState class definition below the decorator."

    target_agentic_system: VirtualAgenticSystem = state["target_agentic_system"]
    ignored_nodes = []
    imports_found = []
    class_def_node = None

    try:
        tree = ast.parse(state_code)

        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "AgentState":
                class_def_node = node
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports_found.append(ast.unparse(node))
            elif isinstance(node, ast.FunctionDef):
                target_agentic_system.upsert_utility_code(ast.unparse(node))
            else:
                ignored_nodes.append(node)

        target_agentic_system.imports.extend(target_agentic_system.deduplicate_imports(imports_found))
        if class_def_node is None:
            return "ERROR: No TypedDict state definition class found."
        main_message = target_agentic_system.set_state_from_node(class_def_node)
        note = ignored_nodes_message(ignored_nodes)
        return f"{main_message}{note}"

    except SyntaxError as e:
        return f"ERROR: Invalid Python syntax in state definition block: {e}"
    except Exception as e:
        return f"ERROR: during state setup: {repr(e)}"


def upsert_component(
    component_type: str,
    name: str,
    function_code: str,
    description: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Creates a new component or updates an existing one in the target system.
    Args:
        component_type: The type of component. Must be one of: 'node', 'tool', or 'router'.
        name: The unique name for the component. For a router, this must be the name of the source node it attaches to.
        description: A description of the component's purpose. This is required when creating a new component.

    The Python code defining the component's function (e.g., `def my_node(state): ...`) MUST be placed immediately after this decorator line.
    """
    if state is None:
        return "ERROR: state is required"
    target_agentic_system: VirtualAgenticSystem = state["target_agentic_system"]
    try:
        if component_type.lower() not in ["node", "tool", "router"]:
            return f"ERROR: Invalid component type '{component_type}'. Must be 'node', 'tool', or 'router'."

        if not function_code:
            return "ERROR: You must provide the function implementation below the decorator."

        func, parsed_function_code = target_agentic_system.get_function(function_code, component_type)
        if func is None:
            return "ERROR: Could not parse a valid function from the provided code."
        if parsed_function_code.startswith("ERROR:"):
            return parsed_function_code

        # Check if component exists
        component_exists = False
        if component_type.lower() == "node":
            component_exists = name in target_agentic_system.nodes
        elif component_type.lower() == "tool":
            component_exists = name in target_agentic_system.tools
        elif component_type.lower() == "router":
            component_exists = name in target_agentic_system.conditional_edges

        if not component_exists and not description:
            return f"ERROR: Description required when creating a new {component_type}"

        if component_exists and not description:
            if component_type.lower() == "node" and name in target_agentic_system.nodes:
                description = target_agentic_system.nodes[name].get("description", "")
            elif component_type.lower() == "tool" and name in target_agentic_system.tools:
                description = target_agentic_system.tools[name].get("description", "")

        action = "updated" if component_exists else "created"

        action_taken = False
        safe_description = description or ""
        if component_type.lower() == "node":
            action_taken = target_agentic_system.create_node(name, safe_description, func, parsed_function_code)
        elif component_type.lower() == "tool":
            action_taken = target_agentic_system.create_tool(name, safe_description, func, parsed_function_code)
        elif component_type.lower() == "router":
            action_taken = target_agentic_system.create_conditional_edge(name, func, parsed_function_code)

        if action_taken:
            return f"{component_type.capitalize()} '{name}' was {action} successfully."
        else:
            return f"WARNING: Your submitted code for the {component_type.lower()} '{name}' is identical to the existing code. **No update was performed.**"

    except Exception as e:
        return f"ERROR: with {component_type} '{name}': {repr(e)}"


def delete_component(component_type: str, name: str, state: Dict[str, Any]) -> str:
    """
    Deletes a component from the target system.
    Args:
        component_type: The type of component to delete. Must be one of: 'node', 'tool', or 'router'.
        name: The name of the component to delete. For a router, this is the name of its source node.
    """
    target_agentic_system: VirtualAgenticSystem = state["target_agentic_system"]
    try:
        if component_type.lower() not in ["node", "tool", "router"]:
            return f"ERROR: Invalid component type '{component_type}'. Must be 'node', 'tool', or 'router'."

        deleted = False
        if component_type.lower() == "node":
            deleted = target_agentic_system.delete_node(name)
        elif component_type.lower() == "tool":
            deleted = target_agentic_system.delete_tool(name)
        elif component_type.lower() == "router":
            deleted = target_agentic_system.delete_conditional_edge(name)

        if deleted:
            return f"{component_type.capitalize()} '{name}' deleted successfully."
        else:
            return f"WARNING: No {component_type} named '{name}' found to delete. No change was made."
    except Exception as e:
        return f"ERROR: deleting {component_type} '{name}': {repr(e)}"


def add_edge(source: str, target: str, state: Dict[str, Any]) -> str:
    """
    Adds a standard (unconditional) edge between nodes in the target system.
    Args:
        source: Name of the source node (or START)
        target: Name of the target node (or END)
    """
    target_agentic_system: VirtualAgenticSystem = state["target_agentic_system"]
    try:
        if target_agentic_system.create_edge(source, target):
            return f"Edge from '{source}' to '{target}' added successfully."
        else:
            return f"WARNING: Edge from '{source}' to '{target}' already exists. No change was made."
    except Exception as e:
        return f"ERROR: adding edge: {repr(e)}"


def delete_edge(source: str, target: str, state: Dict[str, Any]) -> str:
    """
    Deletes a standard (unconditional) edge between nodes in the target system.
    Args:
        source: Name of the source node (or START)
        target: Name of the target node (or END)
    """
    target_agentic_system: VirtualAgenticSystem = state["target_agentic_system"]
    try:
        if target_agentic_system.delete_edge(source, target):
            return f"Edge from '{source}' to '{target}' deleted successfully."
        else:
            return f"WARNING: No edge from '{source}' to '{target}' found to delete. No change was made."
    except Exception as e:
        return f"ERROR: deleting edge: {repr(e)}"


def upsert_utilities(utility_code: str, state: Dict[str, Any]) -> str:
    """
    Adds or updates helper constants, functions, or classes at the top of the main system file.
    This is the ideal place for defining system prompts, custom reducer functions, or helper classes.
    If a constant, function, or class with the same name already exists, it will be replaced.
    The Python code defining the utility components MUST be placed immediately after this decorator line.
    """
    if not utility_code:
        return "ERROR: You must provide the utility code below the decorator."

    target_agentic_system: VirtualAgenticSystem = state["target_agentic_system"]
    imports_found = []
    ignored_nodes = []

    try:
        tree = ast.parse(utility_code)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports_found.append(ast.unparse(node))
            elif not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
                ignored_nodes.append(node)

        target_agentic_system.imports.extend(target_agentic_system.deduplicate_imports(imports_found))

        main_message = target_agentic_system.upsert_utility_code(utility_code)
        note = ignored_nodes_message(ignored_nodes)
        return f"{main_message}{note}"

    except SyntaxError as e:
        return f"ERROR: Invalid Python syntax in utility code block: {e}"
    except Exception as e:
        return f"ERROR: updating utilities: {repr(e)}"


def test_system(state: Dict[str, Any]) -> str:
    """
    Executes the current target system with predefined test input states to validate its functionality.
    This tool is essential for debugging. It provides a detailed report including the final state,
    any output printed to stdout/stderr, the execution path of the graph, and performance metrics.
    Analyze this report carefully to identify errors or confirm correct behavior.
    """
    target_agentic_system: VirtualAgenticSystem = state["target_agentic_system"]
    validation_code_snippets = state.get("validation_code_snippets", [])
    full_final_state = None
    final_test_case_id = ""
    error_message = ""
    stdout_capture = TruncatingStringIO()
    stderr_capture = TruncatingStringIO()
    final_captured_output = ""
    final_flow_chart = ""
    parallel_processing_note = ""
    start_time = time.time()
    total_iterations = 0
    validation_results_summary = []
    all_tests_passed_overall = True
    num_passed_tests = 0
    usage_before = LargeLanguageModel.usage_metrics["target_usage"]["overall"].copy()
    num_tests = 0

    if not validation_code_snippets:
        return "ERROR: validation_code_snippets list is empty."

    try:
        validation_errors = target_agentic_system.validate_graph()
        if validation_errors:
            return "ERROR: Validation failed before execution. The TargetSystem has structural flaws:\n" + "\n".join(
                validation_errors
            )

        # Aggregate test cases and validators from all snippets
        all_test_cases = []
        all_validator_funcs = []
        for snippet in validation_code_snippets:
            snippet_namespace = {
                "LargeLanguageModel": LargeLanguageModel,
                "HumanMessage": HumanMessage,
                "ToolMessage": ToolMessage,
                "SystemMessage": SystemMessage,
                "AIMessage": AIMessage,
            }
            try:
                exec(snippet, snippet_namespace)
                cases = snippet_namespace.get("TARGET_SYSTEM_TEST_CASES")
                validator = snippet_namespace.get("validate_target_system_output")
                if isinstance(cases, list) and len(cases) == 3 and callable(validator):
                    all_test_cases.extend(cases)
                    all_validator_funcs.append(validator)
            except Exception as e_snippet:
                print(f"ERROR: Failed to parse a validation code snippet: {repr(e_snippet)}")
        num_tests = len(all_test_cases)

        source_code = materialize_system(target_agentic_system, output_dir=None)
        main_namespace = {}
        exec(source_code, main_namespace)

        if "workflow" not in main_namespace:
            raise Exception("Could not find 'workflow' in generated code.")
        target_workflow = main_namespace["workflow"]

        for i, test_input_state in enumerate(all_test_cases):
            test_case_id = f"Test Case {i + 1}"
            final_test_case_id = test_case_id
            current_test_final_state = {}
            execution_flow: List[Any] = ["START"]

            purge_command = "rm -rf /sandbox/workspace/data/output && mkdir -p /sandbox/workspace/data/output"
            subprocess.run(purge_command, shell=True, check=False)

            stdout_capture.truncate(0)
            stdout_capture.seek(0)
            stderr_capture.truncate(0)
            stderr_capture.seek(0)

            with (
                contextlib.redirect_stdout(stdout_capture),
                contextlib.redirect_stderr(stderr_capture),
            ):
                try:
                    for stream_mode, update in target_workflow.stream(
                        test_input_state,
                        config={"recursion_limit": RECURSION_LIMIT},
                        stream_mode=["values", "debug"],
                    ):
                        if stream_mode == "values":
                            current_test_final_state = update
                        elif stream_mode == "debug" and update["type"] == "task_result":
                            step = update["step"]
                            if step >= len(execution_flow):
                                execution_flow.append([update["payload"]["name"]])
                            else:
                                execution_flow[step].append(update["payload"]["name"])
                            total_iterations = step + 1
                    execution_flow.append("END")

                    try:
                        # Determine which validator to use and the sub-index
                        validator_index = i // 3
                        sub_index = i % 3
                        validator_func = all_validator_funcs[validator_index]
                        is_pass, message = validator_func(sub_index, current_test_final_state)
                    except Exception as e_validation:
                        is_pass, message = (
                            False,
                            f"ERROR: executing validation function for {test_case_id}: {repr(e_validation)}",
                        )

                    if is_pass:
                        num_passed_tests += 1
                    else:
                        all_tests_passed_overall = False
                        if num_passed_tests > 0:
                            success_message = (
                                f"Test cases 1-{num_passed_tests} passed."
                                if num_passed_tests > 1
                                else "Test case 1 passed."
                            )
                            validation_results_summary.append(success_message)
                        validation_results_summary.append(f"{test_case_id}: FAIL - {message}")

                except Exception as e_test_case:
                    execution_flow.append("... -> FAILED_DURING_EXECUTION")
                    e_message = f"ERROR: during {test_case_id} execution: {repr(e_test_case)}"

                    if "GraphRecursionError" in repr(e_test_case):
                        e_message += " The TargetSystem hit the 20 iteration recursion limit during the test case."
                    else:
                        e_message += f"\n{traceback.format_exc(chain=False)}"

                    all_tests_passed_overall = False
                    if num_passed_tests > 0:
                        success_message = (
                            f"Test cases 1-{num_passed_tests} passed."
                            if num_passed_tests > 1
                            else "Test case 1 passed."
                        )
                        validation_results_summary.append(success_message)
                    validation_results_summary.append(f"{test_case_id}: FAIL - {e_message}")

            full_final_state = current_test_final_state
            final_captured_output = stdout_capture.getvalue() + stderr_capture.getvalue()
            final_flow_chart = " -> ".join([str(flow_step) for flow_step in execution_flow])
            parallel_index, paths = None, None
            for index, flow_step in enumerate(execution_flow):
                if isinstance(flow_step, list) and len(flow_step) > 1:
                    parallel_index, paths = max(index - 1, 0), len(flow_step)
                    parallel_processing_note = f"\nNote: Node {str(execution_flow[parallel_index])} introduced {paths} parallel execution paths."
                    break
            if not all_tests_passed_overall:
                break

        if all_tests_passed_overall and num_tests > 0:
            validation_results_summary.append(f"All {num_tests} test cases passed successfully.")

    except Exception:
        error_message += f"\n\nERROR: running the test_system tool:\n{traceback.format_exc(chain=False)}"
        all_tests_passed_overall = False

    end_time = time.time()
    duration = end_time - start_time
    usage_after = LargeLanguageModel.usage_metrics["target_usage"]["overall"].copy()
    metrics = {metric: usage_after[metric] - usage_before[metric] for metric in usage_before}

    captured_output_str = f"\n{final_test_case_id}:\n<STDOUT+STDERR>\n{final_captured_output}\n</STDOUT+STDERR>"
    flow_chart_str = (
        f"\n{final_test_case_id}:\n<ExecutionFlow>\n{final_flow_chart}{parallel_processing_note}\n</ExecutionFlow>"
    )

    if num_tests > 0:
        avg_duration = duration / num_tests
        avg_llm_calls = metrics["llm_calls"] / num_tests
        avg_total_tokens = metrics["total_tokens"] / num_tests
        avg_input_tokens = metrics["input_tokens"] / num_tests
        avg_output_tokens = metrics["output_tokens"] / num_tests
        avg_iterations = total_iterations / num_tests

        metrics_str = (
            f"\n\n<Metrics>\n"
            f"Avg. Graph Iterations: {round(avg_iterations, 2)}\n"
            f"Avg. Duration: {round(avg_duration, 3)} seconds\n"
            f"Avg. LLM Calls: {round(avg_llm_calls, 2)}\n"
            f"Avg. Tokens: {round(avg_total_tokens, 2)} (Input: {round(avg_input_tokens, 2)}, Output: {round(avg_output_tokens, 2)})\n"
            f"</Metrics>"
        )
    else:
        # Fallback
        metrics_str = (
            f"\n\n<Metrics>\n"
            f"Total Duration: {round(duration, 3)} seconds\n"
            f"Note: No tests were successfully loaded or run, so detailed metrics are unavailable.\n"
            f"</Metrics>"
        )

    validator_result_str = (
        f"\n\n<ValidatorResult>\n"
        f"Overall: {'PASSED' if all_tests_passed_overall else 'FAILED'}\n"
        f"Details:\n" + "\n".join(validation_results_summary) + "\n"
        "</ValidatorResult>"
    )

    final_report = (
        str(truncate_state(full_final_state))
        if full_final_state
        else "No final state captured (possibly due to an early error)."
    )
    final_report_str = f"\n{final_test_case_id}:\n<FinalState>\n{final_report}\n</FinalState>"

    test_result = (
        f"Test suite completed.{final_report_str}"
        f"{captured_output_str}"
        f"{flow_chart_str}"
        f"{metrics_str}"
        f"{validator_result_str}"
    )

    # Also checkpoint on initial tests so we do not accept a system with decreased performance
    if num_passed_tests > 0:
        try:
            code_dir = "sandbox/workspace/generated_systems"
            os.makedirs(code_dir, exist_ok=True)
            escaped_name = target_agentic_system.escaped_name
            base_path = os.path.join(code_dir, escaped_name)

            checkpoint_path = f"{base_path}_checkpoint_{num_passed_tests}.pkl"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(target_agentic_system, f)

            test_result += f"\n\nThe system passed {num_passed_tests}/{num_tests} tests. A snapshot of the current system has been saved."

        except Exception:
            print(f"ERROR: during system checkpoint saving: {traceback.format_exc(chain=False)}")

    is_initial_test = state.get("optimize") and state.get("initial_test_results") is None
    if is_initial_test:
        return test_result + error_message

    if num_passed_tests >= num_tests:
        state["design_completed"] = True
        return test_result + "\nThe design process will now end automatically."

    initial_test_passes = state.get("initial_test_passes")
    tests_to_pass = 2
    if initial_test_passes is not None:
        tests_to_pass = max(2, initial_test_passes + 1)

    if num_passed_tests >= tests_to_pass:
        state["system_passed"] = True
        test_result += (
            "\nYou can continue improving the system, or execute `@@end_design()` if it fulfills all task requirements."
        )

    final_test_output = test_result + error_message
    if num_passed_tests < tests_to_pass:
        final_test_output += test_reminder
    return final_test_output


def end_design(state: Dict[str, Any]) -> str:
    """
    Signals that the design process is complete and ends the session. Use this only when the system has been successfully tested.
    """
    messages = state.get("messages", [])
    max_iterations = state.get("max_iterations", 30)
    iteration = len([msg for msg in messages if isinstance(msg, AIMessage)]) - 1
    system_passed = state.get("system_passed")
    if system_passed or iteration >= (max_iterations - 2):
        state["design_completed"] = True
        return "Ending the design process..."
    else:
        return "ERROR: The design cannot be finalized yet. Please run fully successful tests using `@@test_system()` first."


# Define all code related tools and corresponding attributes
code_related_tools = {
    "set_imports": "import_code",
    "set_state": "state_code",
    "upsert_component": "function_code",
    "upsert_utilities": "utility_code",
}

# Build signatures
available_tools = [
    install_package,
    set_imports,
    set_state,
    upsert_component,
    delete_component,
    add_edge,
    delete_edge,
    upsert_utilities,
    test_system,
    end_design,
]
function_signatures = build_decorator_signatures(available_tools, code_related_tools)

# Rendered during materialization
tools = {}
