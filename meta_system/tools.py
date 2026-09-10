import ast
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from adas_core.candidate_selection import record_candidate_evaluation
from adas_core.decorator_logic import build_decorator_signatures
from adas_core.environment import (
    DEFAULT_EXCLUDED_PACKAGES,
    SANDBOX_FIXTURES_DIR,
    SANDBOX_GENERATED_SYSTEMS_DIR,
    SANDBOX_TASK_SETUP_DIR,
    SANDBOX_TASK_SPEC_PATH,
    is_package_excluded,
    normalize_package_name,
    validate_package_requirement,
)
from adas_core.helpers import (
    get_filtered_packages,
    sanitize_identifier,
    truncate_state,
)
from adas_core.logging_config import get_logger
from adas_core.task_spec import TaskSpec, TestCaseSpec
from adas_core.test_runner import execute_test_suite
from adas_core.virtual_agentic_system import VirtualAgenticSystem
from meta_system.config import RECURSION_LIMIT
from meta_system.helpers import ignored_nodes_message
from meta_system.prompts import test_reminder

logger = get_logger("meta_system.tools")


def install_package(package_name: str, state: dict[str, Any]) -> str:
    """
    Installs a Python package into the environment using pip.
    Args:
        package_name: The name of the package to install, optionally with a version specifier (e.g., "numpy", "pandas==2.0.3").
    """
    target_agentic_system: VirtualAgenticSystem = state["target_agentic_system"]
    # Validate package name to prevent command injection
    if not validate_package_requirement(package_name):
        return f"ERROR: Invalid package name format. Package name '{package_name}' contains invalid characters."
    if is_package_excluded(package_name, excluded_packages=DEFAULT_EXCLUDED_PACKAGES + ["langgraph", "langchain-core"]):
        return f"{package_name} is already installed."

    # Parse package name to get the canonical name for `pip show`
    name_only = normalize_package_name(package_name)
    if not name_only:
        return f"ERROR: Could not parse package name from '{package_name}'."

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

            target_agentic_system.packages_info = get_filtered_packages(DEFAULT_EXCLUDED_PACKAGES) + [
                "langchain-core 0.3.75"
            ]
            return f"Successfully installed {package_name}"
        else:
            return f"ERROR: installing {package_name}:\n{process.stdout}"

    except Exception as e:
        return f"ERROR: installing {package_name}: {e!s}"


def set_imports(import_code: str, state: dict[str, Any]) -> str:
    """
    Sets the import statements for the target system. This replaces any existing custom imports.
    The Python code containing the import statements must be placed immediately after this decorator line.
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
        return f"ERROR: setting imports: {e!r}"


def set_state(state_code: str, state: dict[str, Any]) -> str:
    """
    Defines the AgentState for the target system. This decorator should be used at the beginning of the design process.
    If called again, it will completely replace the previous AgentState definition.
    The Python code defining the AgentState class must be placed immediately after this decorator line.
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
        return f"ERROR: during state setup: {e!r}"


def _upsert_function(
    name: str,
    function_code: str,
    description: str | None,
    state: dict[str, Any] | None,
    component_type: str,
    path_map: dict[str, Any] | None = None,
) -> str:
    """Shared implementation for dedicated node, tool, and conditional-edge upserts."""
    if state is None:
        return "ERROR: state is required"
    target_agentic_system: VirtualAgenticSystem = state["target_agentic_system"]
    try:
        if not function_code:
            return "ERROR: You must provide the function implementation below the decorator."
        if component_type == "conditional edge" and not path_map:
            return "ERROR: Conditional edges require a non-empty explicit path_map."

        function_kind = "conditional_edge" if component_type == "conditional edge" else component_type
        func, parsed_function_code = target_agentic_system.get_function(function_code, function_kind)
        if parsed_function_code.startswith("ERROR:"):
            return parsed_function_code
        if func is None:
            return "ERROR: Could not parse a valid function from the provided code."

        if component_type == "conditional edge":
            component_exists = name in target_agentic_system.conditional_edges
        else:
            component_exists = name in getattr(target_agentic_system, f"{component_type}s")
        if component_type != "conditional edge" and not component_exists and not description:
            return f"ERROR: Description required when creating a new {component_type}"
        if component_exists and not description and component_type != "conditional edge":
            description = getattr(target_agentic_system, f"{component_type}s")[name].get("description", "")

        action = "updated" if component_exists else "created"
        if component_type == "node":
            action_taken = target_agentic_system.create_node(name, description or "", func, parsed_function_code)
        elif component_type == "tool":
            action_taken = target_agentic_system.create_tool(name, description or "", func, parsed_function_code)
        else:
            action_taken = target_agentic_system.create_conditional_edge(name, func, parsed_function_code, path_map)

        if action_taken:
            return f"{component_type.capitalize()} '{name}' was {action} successfully."
        return f"WARNING: Your submitted {component_type} '{name}' is identical to the existing one. **No update was performed.**"
    except Exception as e:
        return f"ERROR: with {component_type} '{name}': {e!r}"


def _validate_action(action: str) -> str | None:
    if action not in {"create", "update", "delete"}:
        return "ERROR: action must be one of: 'create', 'update', or 'delete'."
    return None


def _manage_function(
    action: str,
    name: str,
    function_code: str | None,
    description: str | None,
    state: dict[str, Any] | None,
    component_type: str,
    path_map: dict[str, str] | None = None,
) -> str:
    """Apply strict lifecycle semantics for one domain-specific system object."""
    if error := _validate_action(action):
        return error
    if state is None:
        return "ERROR: state is required"

    target_agentic_system: VirtualAgenticSystem = state["target_agentic_system"]
    if component_type == "conditional edge":
        exists = name in target_agentic_system.conditional_edges
    else:
        exists = name in getattr(target_agentic_system, f"{component_type}s")

    if action == "delete":
        return _delete_named_item(name, state, component_type)
    if action == "create" and exists:
        return f"ERROR: {component_type.capitalize()} '{name}' already exists. Use action='update' instead."
    if action == "update" and not exists:
        return f"ERROR: {component_type.capitalize()} '{name}' does not exist. Use action='create' instead."
    if not function_code:
        return f"ERROR: {component_type.capitalize()} creation and updates require function code."
    if component_type != "conditional edge" and action == "create" and not description:
        return f"ERROR: Description is required when creating a {component_type}."
    if component_type == "conditional edge" and not path_map:
        return "ERROR: Conditional-edge creation and updates require a non-empty explicit path_map."

    return _upsert_function(name, function_code, description, state, component_type, path_map)


def manage_node(
    action: Literal["create", "update", "delete"],
    name: str,
    function_code: str | None = None,
    description: str | None = None,
    state: dict[str, Any] | None = None,
) -> str:
    """Creates, updates, or deletes a node; function code follows create and update calls."""
    return _manage_function(action, name, function_code, description, state, "node")


def manage_tool(
    action: Literal["create", "update", "delete"],
    name: str,
    function_code: str | None = None,
    description: str | None = None,
    state: dict[str, Any] | None = None,
) -> str:
    """Creates, updates, or deletes a tool; function code follows create and update calls."""
    return _manage_function(action, name, function_code, description, state, "tool")


def manage_conditional_edge(
    action: Literal["create", "update", "delete"],
    source: str,
    path_map: dict[str, str] | None = None,
    function_code: str | None = None,
    state: dict[str, Any] | None = None,
) -> str:
    """Creates, updates, or deletes the conditional edge attached to a source node."""
    return _manage_function(action, source, function_code, None, state, "conditional edge", path_map)


def _delete_named_item(name: str, state: dict[str, Any], component_type: str) -> str:
    """
    Deletes a named node, tool, or conditional edge from the target system.
    """
    target_agentic_system: VirtualAgenticSystem = state["target_agentic_system"]
    try:
        delete_method = getattr(target_agentic_system, f"delete_{component_type.replace(' ', '_')}")
        if delete_method(name):
            return f"{component_type.capitalize()} '{name}' deleted successfully."
        return f"WARNING: No {component_type} named '{name}' found to delete. No change was made."
    except Exception as e:
        return f"ERROR: deleting {component_type} '{name}': {e!r}"


def manage_edge(action: Literal["create", "delete"], source: str, target: str, state: dict[str, Any]) -> str:
    """Creates or deletes a standard edge; changing endpoints requires delete then create."""
    target_agentic_system: VirtualAgenticSystem = state["target_agentic_system"]
    if action not in {"create", "delete"}:
        return "ERROR: Edge action must be either 'create' or 'delete'."
    try:
        if action == "create":
            if target_agentic_system.create_edge(source, target):
                return f"Edge from '{source}' to '{target}' created successfully."
            return f"WARNING: Edge from '{source}' to '{target}' already exists."
        if target_agentic_system.delete_edge(source, target):
            return f"Edge from '{source}' to '{target}' deleted successfully."
        return f"WARNING: Edge from '{source}' to '{target}' does not exist."
    except Exception as e:
        operation = "creating" if action == "create" else "deleting"
        return f"ERROR: {operation} edge from '{source}' to '{target}': {e!r}"


def manage_utilities(
    action: Literal["create", "update", "delete"],
    utility_code: str | None = None,
    definitions: list[dict[str, str]] | None = None,
    state: dict[str, Any] | None = None,
) -> str:
    """Creates, updates, or deletes typed utility definitions."""
    if state is None:
        return "ERROR: state is required"
    if action not in {"create", "update", "delete"}:
        return "ERROR: Utility action must be one of: 'create', 'update', or 'delete'."
    target_agentic_system: VirtualAgenticSystem = state["target_agentic_system"]
    if action == "delete":
        return target_agentic_system.delete_utility_definitions(definitions or [])
    if not utility_code:
        return f"ERROR: Utility {action} requires utility code below the decorator."

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
        return f"{main_message}{ignored_nodes_message(ignored_nodes)}"
    except SyntaxError as e:
        return f"ERROR: Invalid Python syntax in utility code block: {e}"
    except Exception as e:
        return f"ERROR: {action}ing utilities: {e!r}"


def _resolve_task_validation(
    state: dict[str, Any],
) -> tuple[TaskSpec | None, Any | None, list[TestCaseSpec]]:
    """Resolve TaskSpec, validation module, and dev suite test cases.

    Returns:
        (task_spec, validation_module, test_cases)
    """
    # Import lazily: automatic_validation depends on meta_system.config, whose
    # package initialization imports this module.
    from adas_core.automatic_validation import load_validation_module

    # 1. Check if task_spec is explicitly in state
    task_spec: TaskSpec | None = None
    raw_spec = state.get("task_spec")
    if isinstance(raw_spec, TaskSpec):
        task_spec = raw_spec
    elif isinstance(raw_spec, dict):
        task_spec = TaskSpec.model_validate(raw_spec)
    elif isinstance(raw_spec, (str, Path)):
        task_spec = TaskSpec.from_file(raw_spec)

    # 2. Check for task.json in task_dir or standard sandbox locations
    if task_spec is None:
        task_dir_env = os.environ.get("ADAS_TASK_DIR")
        candidate_paths: list[Path] = [
            Path(SANDBOX_TASK_SPEC_PATH),
            Path("task_setup/task.json"),
        ]
        if task_dir_env:
            candidate_paths.insert(0, Path(task_dir_env) / "task.json")
        for cp in candidate_paths:
            if cp.exists():
                try:
                    task_spec = TaskSpec.from_file(cp)
                    break
                except Exception as e:
                    logger.debug(f"Could not load TaskSpec from {cp}: {e}")

    # 3. If TaskSpec is resolved, load or generate the validation module
    if task_spec is not None:
        validation_module = state.get("validation_module")
        if validation_module is None:
            val_path = state.get("validation_module_path")
            if val_path and Path(val_path).exists():
                validation_module = load_validation_module(val_path)
            else:
                task_dir_candidates: list[Path] = []
                if state.get("task_dir"):
                    task_dir_candidates.append(Path(state["task_dir"]))
                if os.environ.get("ADAS_TASK_DIR"):
                    task_dir_candidates.append(Path(os.environ["ADAS_TASK_DIR"]))
                task_dir_candidates.extend(
                    [
                        Path(SANDBOX_TASK_SETUP_DIR),
                        Path("task_setup"),
                        Path("."),
                    ]
                )

                found_file = None
                safe_name = sanitize_identifier(task_spec.name)
                for td in task_dir_candidates:
                    for name in [f"{safe_name}.validation.py", f"{task_spec.name}.validation.py", "validation.py"]:
                        candidate = td / name
                        if candidate.exists():
                            found_file = candidate
                            break
                    if found_file:
                        break

                if found_file:
                    validation_module = load_validation_module(found_file)
                else:
                    searched = ", ".join(str(path) for path in task_dir_candidates)
                    raise FileNotFoundError(
                        f"No frozen validation module found for TaskSpec '{task_spec.name}'. "
                        f"Generate and review its LLM-authored validator before design. Searched: {searched}"
                    )

        return task_spec, validation_module, list(task_spec.dev_suite)

    return None, None, []


def test_system(state: dict[str, Any]) -> str:
    """
    Executes the current target system with predefined test input states to validate its functionality.
    This tool is essential for debugging. It provides a detailed report including the final state,
    any output printed to stdout/stderr, the execution path of the graph, and performance metrics.
    Analyze this report carefully to identify errors or confirm correct behavior.
    """
    target_agentic_system: VirtualAgenticSystem = state["target_agentic_system"]

    try:
        task_spec, validation_module, all_test_cases = _resolve_task_validation(state)
    except Exception as e_prep:
        eval_err = (
            f"EVALUATOR_ERROR: Failed to prepare validation environment: {e_prep!r}\n"
            f"{traceback.format_exc(chain=False)}"
        )
        logger.error(eval_err)
        return f"Test suite aborted.\n\n<ValidatorResult>\nOverall: FAILED\nDetails:\n{eval_err}\n</ValidatorResult>"

    num_tests = len(all_test_cases)
    if num_tests == 0:
        return "ERROR: No validation test cases found. No valid TaskSpec dev_suite was provided."
    if validation_module is None:
        return "ERROR: No validation module found to validate test execution."

    fixtures_dir_env = os.environ.get("ADAS_FIXTURES_DIR")
    fixtures_path = Path(fixtures_dir_env) if fixtures_dir_env else Path(SANDBOX_FIXTURES_DIR)
    active_fixtures_dir = fixtures_path if fixtures_path.exists() else None

    start_time = time.time()
    error_message = ""
    try:
        exec_result = execute_test_suite(
            system=target_agentic_system,
            test_cases=all_test_cases,
            validation_module=validation_module,
            recursion_limit=RECURSION_LIMIT,
            stop_on_first_failure=True,
            workspace_root=os.environ.get("ADAS_WORKSPACE_ROOT"),
            fixtures_dir=active_fixtures_dir,
            capture_debug_flow=True,
            system_role="target",
            task_spec=task_spec,
        )

        if exec_result.structural_errors:
            return "ERROR: Validation failed before execution. The TargetSystem has structural flaws:\n" + "\n".join(
                exec_result.structural_errors
            )

        if exec_result.materialization_error:
            eval_err = f"EVALUATOR_ERROR: Failed to prepare system: {exec_result.materialization_error}\n"
            logger.error(eval_err)
            return (
                f"Test suite aborted.\n\n<ValidatorResult>\nOverall: FAILED\nDetails:\n{eval_err}\n</ValidatorResult>"
            )

        final_test_case_id = exec_result.last_executed_case_id
        full_final_state = exec_result.last_final_state
        final_captured_output = exec_result.last_captured_output
        final_flow_chart = " -> ".join([str(flow_step) for flow_step in exec_result.last_execution_flow])
        parallel_processing_note = exec_result.parallel_processing_note
        all_tests_passed_overall = exec_result.all_passed
        num_passed_tests = exec_result.passed_count
        duration = exec_result.duration_seconds
        metrics = exec_result.token_usage
        total_iterations = exec_result.total_iterations
        validation_results_summary = exec_result.summary_lines

    except Exception:
        error_message += f"\n\nERROR: running the test_system tool:\n{traceback.format_exc(chain=False)}"
        all_tests_passed_overall = False
        num_passed_tests = 0
        final_test_case_id = all_test_cases[0].id if all_test_cases else ""
        full_final_state = None
        final_captured_output = ""
        final_flow_chart = ""
        parallel_processing_note = ""
        duration = time.time() - start_time
        metrics = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        total_iterations = 0
        validation_results_summary = [error_message.strip()]

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

    # Record structured test metrics in state
    state["test_metrics"] = {
        "passed": num_passed_tests,
        "total": num_tests,
        "pass_rate": (num_passed_tests / num_tests) if num_tests > 0 else 0.0,
    }

    test_result += f"\n\nThe system passed {num_passed_tests}/{num_tests} tests."

    messages = state.get("messages", [])
    iteration = len([msg for msg in messages if isinstance(msg, AIMessage)])
    candidate = record_candidate_evaluation(
        state=state,
        system=target_agentic_system,
        iteration=iteration,
        passed_count=num_passed_tests,
        total_count=num_tests,
        total_tokens=metrics.get("total_tokens", 0),
        duration_seconds=round(duration, 3),
        llm_calls=metrics.get("llm_calls", 0),
        code_dir=SANDBOX_GENERATED_SYSTEMS_DIR,
    )
    if candidate.get("checkpoint_path") and num_passed_tests > 0:
        test_result += " A snapshot of the current system has been saved."

    is_initial_test = state.get("optimize") and state.get("initial_test_results") is None
    if is_initial_test:
        return test_result + error_message

    all_passed = num_tests > 0 and num_passed_tests >= num_tests
    if all_passed:
        state["system_passed"] = True
        if not state.get("optimize"):
            state["design_completed"] = True
            return test_result + "\nAll tests passed successfully! The design process will now end automatically."
        else:
            test_result += (
                "\nAll tests passed successfully! You can continue optimizing the system's architecture, efficiency, or robustness, "
                "or execute `@@end_design()` if you are satisfied."
            )
    else:
        state["system_passed"] = False

    final_test_output = test_result + error_message
    if not all_passed:
        final_test_output += test_reminder
    return final_test_output


def end_design(state: dict[str, Any]) -> str:
    """
    Signals that the design process is complete and ends the session. Use this only when the system has been successfully tested.
    """
    messages = state.get("messages", [])
    max_iterations = state.get("max_iterations", 30)
    iteration = len([msg for msg in messages if isinstance(msg, AIMessage)]) - 1
    system_passed = state.get("system_passed")
    candidates = state.get("candidates", [])
    has_passing_candidate = any(
        c.get("dev_pass_rate", 0.0) == 1.0
        or (c.get("total_count", 0) > 0 and c.get("passed_count") == c.get("total_count"))
        for c in candidates
    )
    if system_passed or has_passing_candidate or iteration >= (max_iterations - 2):
        state["design_completed"] = True
        return "Ending the design process..."
    else:
        return "ERROR: The design cannot be finalized yet. Please run fully successful tests using `@@test_system()` first."


# Define all code related tools and corresponding attributes
code_related_tools = {
    "set_imports": "import_code",
    "set_state": "state_code",
    "manage_node": "function_code",
    "manage_tool": "function_code",
    "manage_conditional_edge": "function_code",
    "manage_utilities": "utility_code",
}

# Build signatures
available_tools = [
    install_package,
    set_imports,
    set_state,
    manage_node,
    manage_tool,
    manage_conditional_edge,
    manage_edge,
    manage_utilities,
    test_system,
    end_design,
]
function_signatures = build_decorator_signatures(available_tools, code_related_tools)

# Registered LangChain tools dictionary
tools = {
    "InstallPackage": tool("InstallPackage")(install_package),
    "SetImports": tool("SetImports")(set_imports),
    "SetState": tool("SetState")(set_state),
    "ManageNode": tool("ManageNode")(manage_node),
    "ManageTool": tool("ManageTool")(manage_tool),
    "ManageConditionalEdge": tool("ManageConditionalEdge")(manage_conditional_edge),
    "ManageEdge": tool("ManageEdge")(manage_edge),
    "ManageUtilities": tool("ManageUtilities")(manage_utilities),
    "TestSystem": tool("TestSystem")(test_system),
    "EndDesign": tool("EndDesign")(end_design),
}
