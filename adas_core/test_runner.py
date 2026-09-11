"""Unified test runner for executing VirtualAgenticSystem workflows against test suites."""

from __future__ import annotations

import contextlib
import os
import sys
import time
import traceback
import types
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adas_core.chat_model import UsageRecorder, usage_scope
from adas_core.environment import SANDBOX_FIXTURES_DIR, isolated_case_workspace
from adas_core.exceptions import (
    MissingWorkflowError,
    ValidatorContractError,
    ValidatorDispatchError,
)
from adas_core.fixture_lifecycle import process_fixture_lifecycle
from adas_core.helpers import TruncatingStringIO, sanitize_test_id
from adas_core.logging_config import get_logger
from adas_core.materialize import materialize_system
from adas_core.task_spec import TaskSpec, TestCaseSpec
from adas_core.virtual_agentic_system import VirtualAgenticSystem

logger = get_logger("adas_core.test_runner")


@dataclass
class SingleTestCaseResult:
    """Execution results for an individual test case."""

    test_case_id: str
    passed: bool
    message: str
    final_state: dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    execution_flow: list[Any] = field(default_factory=list)
    total_iterations: int = 0
    error_message: str = ""


@dataclass
class TestSuiteExecutionResult:
    """Aggregate result from executing a suite of test cases against a VirtualAgenticSystem."""

    __test__ = False

    system_name: str
    total_tests: int
    passed_count: int
    pass_rate: float
    all_passed: bool
    case_results: list[SingleTestCaseResult]
    duration_seconds: float
    token_usage: dict[str, int]
    summary_lines: list[str]
    structural_errors: list[str] = field(default_factory=list)
    materialization_error: str | None = None
    last_executed_case_id: str = ""
    last_final_state: dict[str, Any] = field(default_factory=dict)
    last_captured_output: str = ""
    last_execution_flow: list[Any] = field(default_factory=list)
    parallel_processing_note: str = ""
    total_iterations: int = 0
    max_iterations: int = 0
    executed_count: int = 0


def _dispatch_validator(
    validation_module: Any,
    test_case: TestCaseSpec,
    final_state: dict[str, Any],
    workspace_dirs: dict[str, Any],
) -> tuple[bool, str]:
    """Locate and invoke the appropriate validator function within validation_module."""
    clean_id = sanitize_test_id(test_case.id)
    fn_name = f"validate_{clean_id}"

    validator_fn: Any = getattr(validation_module, "VALIDATORS", {}).get(test_case.id)
    if not validator_fn:
        validator_fn = getattr(validation_module, fn_name, None)
    if not validator_fn and hasattr(validation_module, "validate"):
        cid = test_case.id
        active_module = validation_module

        def _dispatch_validate(s: dict[str, Any], w: dict[str, str]) -> Any:
            return active_module.validate(cid, s, w)

        validator_fn = _dispatch_validate

    if not validator_fn:
        raise ValidatorDispatchError(f"No validator function '{fn_name}' found in validation module.")

    workspace_dirs_str = {
        "workspace": str(workspace_dirs.get("workspace", "")),
        "input": str(workspace_dirs.get("input", "")),
        "output": str(workspace_dirs.get("output", "")),
    }

    is_pass, message = validator_fn(final_state, workspace_dirs_str)
    if not isinstance(is_pass, bool):
        raise ValidatorContractError(
            f"Validator for {test_case.id} must return tuple[bool, str], got {type(is_pass).__name__}"
        )
    return is_pass, str(message)


def execute_test_suite(
    system: VirtualAgenticSystem,
    test_cases: list[TestCaseSpec],
    validation_module: Any,
    recursion_limit: int = 20,
    stop_on_first_failure: bool = False,
    workspace_root: str | Path | None = None,
    fixtures_dir: str | Path | None = None,
    capture_debug_flow: bool = True,
    system_role: str = "target",
    task_spec: TaskSpec | None = None,
) -> TestSuiteExecutionResult:
    """Execute a test suite against a VirtualAgenticSystem.

    Handles:
    - Pre-execution structural graph validation
    - Dynamic module compilation & typing registration
    - Per-test workspace isolation and stdout/stderr capture
    - Token usage tracking via UsageRecorder
    - Graph streaming execution with recursion limit
    - Validator resolution and error reporting
    """
    sys_name = getattr(system, "system_name", "unknown")
    total_tests = len(test_cases)

    # 1. Structural graph validation
    try:
        structural_errors = system.validate_graph()
    except Exception as exc:
        err_msg = f"Unexpected exception during graph validation: {exc!r}\n{traceback.format_exc(chain=False)}"
        logger.error("Graph validation threw an exception for '%s': %s", sys_name, err_msg)
        return TestSuiteExecutionResult(
            system_name=sys_name,
            total_tests=total_tests,
            passed_count=0,
            pass_rate=0.0,
            all_passed=False,
            case_results=[],
            duration_seconds=0.0,
            token_usage={"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            summary_lines=[f"Structural validation exception: {exc!r}"],
            structural_errors=[err_msg],
        )

    if structural_errors:
        logger.error("Structural graph validation failed for '%s': %s", sys_name, structural_errors)
        return TestSuiteExecutionResult(
            system_name=sys_name,
            total_tests=total_tests,
            passed_count=0,
            pass_rate=0.0,
            all_passed=False,
            case_results=[],
            duration_seconds=0.0,
            token_usage={"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            summary_lines=[f"Structural validation errors: {structural_errors}"],
            structural_errors=structural_errors,
        )

    if total_tests == 0:
        return TestSuiteExecutionResult(
            system_name=sys_name,
            total_tests=0,
            passed_count=0,
            pass_rate=1.0,
            all_passed=True,
            case_results=[],
            duration_seconds=0.0,
            token_usage={"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            summary_lines=["No test cases provided."],
        )

    # 2. Materialization into a registered module
    # LangGraph typing forward refs require sys.modules[mod_name] to be present.
    mod_name = f"dynamic_system_{uuid.uuid4().hex}"
    mod = types.ModuleType(mod_name)
    sys.modules[mod_name] = mod
    main_namespace = mod.__dict__
    main_namespace["__name__"] = mod_name

    try:
        source_code = materialize_system(system, output_dir=None)
        exec(source_code, main_namespace)
        if "workflow" not in main_namespace:
            raise MissingWorkflowError("Compiled system code did not expose a 'workflow' object.")
        target_workflow = main_namespace["workflow"]
    except Exception as exc:
        sys.modules.pop(mod_name, None)
        err_msg = f"Failed to materialize candidate system '{sys_name}': {exc!r}"
        logger.error("%s\n%s", err_msg, traceback.format_exc())
        return TestSuiteExecutionResult(
            system_name=sys_name,
            total_tests=total_tests,
            passed_count=0,
            pass_rate=0.0,
            all_passed=False,
            case_results=[],
            duration_seconds=0.0,
            token_usage={"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            summary_lines=[err_msg],
            materialization_error=err_msg,
        )

    # 3. Resolve fixtures directory
    active_fixtures_dir: Path | None = None
    if fixtures_dir:
        fp = Path(fixtures_dir)
        if fp.exists():
            active_fixtures_dir = fp
    else:
        env_fixtures = os.environ.get("ADAS_FIXTURES_DIR")
        if env_fixtures:
            fp = Path(env_fixtures)
            if fp.exists():
                active_fixtures_dir = fp
        else:
            default_fp = Path(SANDBOX_FIXTURES_DIR)
            if default_fp.exists():
                active_fixtures_dir = default_fp

    active_workspace_root = workspace_root or os.environ.get("ADAS_WORKSPACE_ROOT")

    # 4. Execute test cases
    test_run_id = uuid.uuid4().hex
    start_time = time.time()
    usage_before = UsageRecorder.get_aggregate(system=system_role, run_id=test_run_id)

    passed_count = 0
    all_passed_overall = True
    case_results: list[SingleTestCaseResult] = []
    summary_lines: list[str] = []

    last_case_id = ""
    last_final_state: dict[str, Any] = {}
    last_captured_output = ""
    last_execution_flow: list[Any] = []
    parallel_processing_note = ""

    try:
        for test_case in test_cases:
            test_case_id = test_case.id
            last_case_id = test_case_id
            # TODO: multi-turn execution (iterating through test_case.turns with state checkpointer/thread_id)
            # is not implemented yet. Currently only single-turn execution (turns[0]) is supported.
            test_input_state = test_case.turns[0] if test_case.turns else {}
            case_slug = sanitize_test_id(test_case.id)

            current_final_state: dict[str, Any] = {}
            execution_flow: list[Any] = ["START"]
            case_iterations = 0
            is_pass = False
            case_msg = ""
            case_error_msg = ""
            stdout_captured = ""
            stderr_captured = ""

            allowed_files = None
            if task_spec is not None:
                if test_case.fixture_ids is not None:
                    allowed_files = task_spec.test_fixtures.get_file_paths_for_fixture_ids(test_case.fixture_ids)
                else:
                    allowed_files = task_spec.test_fixtures.get_all_file_paths()

            with isolated_case_workspace(
                base_dir=active_workspace_root,
                run_id=test_run_id,
                case_id=case_slug,
                fixtures_dir=active_fixtures_dir,
                allowed_files=allowed_files,
            ) as workspace_dirs:
                stdout_buf = TruncatingStringIO()
                stderr_buf = TruncatingStringIO()

                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    try:
                        fixture_context = (
                            process_fixture_lifecycle(
                                task_spec.test_fixtures,
                                test_case.fixture_ids,
                                active_fixtures_dir if active_fixtures_dir else Path.cwd(),
                                workspace_dirs,
                            )
                            if task_spec is not None
                            else contextlib.nullcontext()
                        )
                        with fixture_context:
                            with usage_scope(system=system_role, run_id=test_run_id, case_id=test_case_id):
                                stream_modes = ["values", "debug"] if capture_debug_flow else ["values"]
                                for stream_mode, update in target_workflow.stream(
                                    test_input_state,
                                    config={"recursion_limit": recursion_limit},
                                    stream_mode=stream_modes,
                                ):
                                    if stream_mode == "values":
                                        current_final_state = update
                                    elif stream_mode == "debug" and update.get("type") == "task_result":
                                        step = update.get("step", 0)
                                        node_name = update.get("payload", {}).get("name", "node")
                                        if step >= len(execution_flow):
                                            execution_flow.append([node_name])
                                        else:
                                            execution_flow[step].append(node_name)
                                        case_iterations = step + 1

                            execution_flow.append("END")

                            # Validate results while process fixtures remain available.
                            try:
                                is_pass, case_msg = _dispatch_validator(
                                    validation_module, test_case, current_final_state, workspace_dirs
                                )
                            except Exception as e_val:
                                is_pass = False
                                case_msg = (
                                    f"EVALUATOR_ERROR: Validator failed unexpectedly for {test_case_id}: {e_val!r}\n"
                                    f"{traceback.format_exc(chain=False)}"
                                )

                    except Exception as e_run:
                        execution_flow.append("... -> FAILED_DURING_EXECUTION")
                        is_pass = False
                        err_repr = repr(e_run)
                        if "GraphRecursionError" in err_repr:
                            case_error_msg = (
                                f"ERROR: during {test_case_id} execution: {e_run!r} "
                                f"The TargetSystem hit the {recursion_limit} iteration recursion limit during the test case."
                            )
                        else:
                            case_error_msg = (
                                f"ERROR: during {test_case_id} execution: {e_run!r}\n"
                                f"{traceback.format_exc(chain=False)}"
                            )
                        case_msg = case_error_msg

                stdout_captured = stdout_buf.getvalue()
                stderr_captured = stderr_buf.getvalue()

            last_final_state = current_final_state
            last_captured_output = stdout_captured + stderr_captured
            last_execution_flow = execution_flow

            # Detect parallel branches in execution flow
            for index, flow_step in enumerate(execution_flow):
                if isinstance(flow_step, list) and len(flow_step) > 1:
                    parallel_index, paths = max(index - 1, 0), len(flow_step)
                    parallel_processing_note = (
                        f"\nNote: Node {execution_flow[parallel_index]!s} introduced {paths} parallel execution paths."
                    )
                    break

            case_result = SingleTestCaseResult(
                test_case_id=test_case_id,
                passed=is_pass,
                message=case_msg,
                final_state=current_final_state,
                stdout=stdout_captured,
                stderr=stderr_captured,
                execution_flow=execution_flow,
                total_iterations=case_iterations,
                error_message=case_error_msg,
            )
            case_results.append(case_result)

            if is_pass:
                passed_count += 1
            else:
                all_passed_overall = False
                if passed_count > 0:
                    summary_lines.append(
                        f"Test cases 1-{passed_count} passed." if passed_count > 1 else "Test case 1 passed."
                    )
                summary_lines.append(f"{test_case_id}: FAIL - {case_msg}")
                if stop_on_first_failure:
                    break

        if all_passed_overall and total_tests > 0:
            summary_lines.append(f"All {total_tests} test cases passed successfully.")

    finally:
        sys.modules.pop(mod_name, None)

    end_time = time.time()
    duration = end_time - start_time
    usage_after = UsageRecorder.get_aggregate(system=system_role, run_id=test_run_id)
    incomplete_diff = usage_after.get("incomplete_usage_count", 0) - usage_before.get("incomplete_usage_count", 0)
    usage_is_incomplete = incomplete_diff > 0

    token_usage: dict[str, Any] = {
        metric: usage_after.get(metric, 0) - usage_before.get(metric, 0)
        for metric in ["llm_calls", "input_tokens", "output_tokens", "total_tokens"]
    }
    if usage_is_incomplete:
        token_usage["total_tokens"] = None
        token_usage["input_tokens"] = None
        token_usage["output_tokens"] = None

    executed_count = len(case_results)
    summed_iterations = sum(cr.total_iterations for cr in case_results)
    max_iterations = max((cr.total_iterations for cr in case_results), default=0)

    pass_rate = (passed_count / total_tests) if total_tests > 0 else 0.0

    return TestSuiteExecutionResult(
        system_name=sys_name,
        total_tests=total_tests,
        passed_count=passed_count,
        pass_rate=pass_rate,
        all_passed=all_passed_overall and (passed_count == total_tests),
        case_results=case_results,
        duration_seconds=duration,
        token_usage=token_usage,
        summary_lines=summary_lines,
        last_executed_case_id=last_case_id,
        last_final_state=last_final_state,
        last_captured_output=last_captured_output,
        last_execution_flow=last_execution_flow,
        parallel_processing_note=parallel_processing_note,
        total_iterations=summed_iterations,
        max_iterations=max_iterations,
        executed_count=executed_count,
    )
