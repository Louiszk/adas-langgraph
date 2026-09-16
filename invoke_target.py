"""Run a target agentic system in an isolated sandbox with TaskSpec environment support."""

from __future__ import annotations

import argparse
import datetime
import json
import shlex
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from adas_core.environment import (
    SANDBOX_TARGET_METRICS_DIR,
    SANDBOX_WORKSPACE_DIR,
    get_installed_packages_from_metrics,
    load_environment,
)
from adas_core.helpers import validate_identifier
from adas_core.runtime_resources import RuntimeResourceProfile
from adas_core.task_spec import TaskSpec
from config.logging import get_logger, setup_logging
from create_setup import run_setup_for_task, setup_manifest_is_current
from sandbox.sandbox import (
    StreamingSandboxSession,
    copy_task_setup_to_sandbox,
    run_sandbox_preflight,
    setup_sandbox_environment,
)

logger = get_logger("invoke_target")


def provision_target_dependencies(session: StreamingSandboxSession, system_name: str) -> bool:
    """Install design-time target dependencies inside the active sandbox."""
    try:
        metrics_path = Path("generated_systems") / "metrics" / f"{system_name}.json"
        packages = get_installed_packages_from_metrics(metrics_path)
    except ValueError as exc:
        logger.error(str(exc))
        return False

    if not packages:
        return True

    command = "python3 -m pip install --disable-pip-version-check " + " ".join(
        shlex.quote(package) for package in packages
    )
    logger.info("Provisioning target dependencies in sandbox: %s", ", ".join(packages))
    result = session.execute_command(command)
    exit_code = getattr(result, "exit_code", 0) if result is not None else 1
    if exit_code != 0:
        logger.error("Failed to provision target dependencies in sandbox: %s", result)
        return False
    return True


def run_target_system_in_sandbox(
    session: StreamingSandboxSession,
    system_name: str,
    state: dict[str, Any],
    run_id: str,
    task_dir: str | None = None,
    runtime_profile: RuntimeResourceProfile | None = None,
) -> bool:
    """Constructs and executes the command to run the target system inside the sandbox."""
    cmd_parts = [
        "python3 "
        f"{shlex.quote(SANDBOX_WORKSPACE_DIR + '/run_target.py')} "
        f"--system_name={shlex.quote(system_name)} --run-id={shlex.quote(run_id)}"
    ]
    if task_dir:
        cmd_parts.append(f"--task-dir={shlex.quote(task_dir)}")
    if runtime_profile:
        cmd_parts.append(f"--runtime-profile={shlex.quote(runtime_profile.model_dump_json())}")

    state_str = json.dumps(state)
    cmd_parts.append(f"--state={shlex.quote(state_str)}")

    # execute_command_streaming does not expose the container exit code, so emit
    # one unambiguous marker after the target process completes.
    command = (
        " ".join(cmd_parts)
        + '; target_exit=$?; printf \'\\n__ADAS_TARGET_EXIT__%s\\n\' "$target_exit"; exit "$target_exit"'
    )

    logger.info(f"Executing Target System: {system_name} (Run ID: {run_id})")
    logger.info(f"Initial State: {state_str}")

    output_chunks: list[str] = []
    for chunk in session.execute_command_streaming(command):
        output_chunks.append(chunk)
        print(chunk, end="", flush=True)

    succeeded = "__ADAS_TARGET_EXIT__0" in "".join(output_chunks)
    if succeeded:
        logger.info("Target system execution completed")
    else:
        logger.error("Target system execution failed")
    return succeeded


def stage_runtime_profile_sources(
    session: StreamingSandboxSession, profile: RuntimeResourceProfile
) -> RuntimeResourceProfile:
    """Copy user-selected local artifacts into the sandbox and rewrite their profile paths."""
    staged = profile.model_copy(deep=True)
    staging_root = f"{SANDBOX_WORKSPACE_DIR}/runtime_resources"
    for fixture_id, override in staged.overrides.items():
        if override.provider != "local_file":
            continue
        source = Path(override.source or "").resolve()
        destination = f"{staging_root}/{fixture_id}"
        session.execute_command(f"mkdir -p {shlex.quote(destination)}")
        if source.is_file():
            runtime_source = f"{destination}/{source.name}"
            session.copy_to_runtime(str(source), runtime_source)
        else:
            for item in source.rglob("*"):
                relative = item.relative_to(source).as_posix()
                runtime_file = f"{destination}/{relative}"
                runtime_parent = PurePosixPath(runtime_file).parent.as_posix()
                if item.is_dir():
                    session.execute_command(f"mkdir -p {shlex.quote(runtime_file)}")
                elif item.is_file():
                    session.execute_command(f"mkdir -p {shlex.quote(runtime_parent)}")
                    session.copy_to_runtime(str(item), runtime_file)
            runtime_source = destination
        override.source = runtime_source
    return staged


def main() -> int:
    """Main function to set up and run a target agentic system in a sandboxed environment."""
    load_environment()
    setup_logging()

    parser = argparse.ArgumentParser(description="Run a target agentic system in a sandboxed environment.")
    parser.add_argument(
        "--system_name",
        "--system-name",
        dest="system_name",
        required=True,
        help="Name of the target system to run (e.g., 'DataAnalystSystem_v0').",
    )
    parser.add_argument(
        "--task-spec",
        type=Path,
        default=None,
        help="Optional TaskSpec JSON file defining task fixtures, mock services, and dependencies.",
    )
    parser.add_argument(
        "--auto-setup",
        action="store_true",
        help="Generate frozen fixtures and preflight artifacts in a separate sandbox before running.",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=None,
        help="Optional runtime resource profile JSON overriding selected fixture providers.",
    )
    state_group = parser.add_mutually_exclusive_group(required=True)
    state_group.add_argument(
        "--state",
        help="JSON string defining the initial state for the system.",
    )
    state_group.add_argument(
        "--state-file",
        type=Path,
        help="Path to a JSON file defining the initial state for the system.",
    )
    state_group.add_argument(
        "--batch",
        type=Path,
        help=(
            "Path to a JSON file containing an array of test cases to run in a single sandbox session. "
            'Each entry must be an object with "id" (str) and "state" (object) fields, '
            'e.g. [{"id": "case_1", "state": {"query": "..."}}].'
        ),
    )
    parser.add_argument(
        "--reinstall",
        action="store_true",
        help="Force re-installation of dependencies in the sandbox.",
    )
    parser.add_argument(
        "--base-image",
        default=None,
        help="The base container image to use for the sandbox.",
    )
    parser.add_argument(
        "--container",
        choices=["auto", "docker", "podman"],
        default="auto",
        help="Container runtime to use (auto tries Docker first, then Podman).",
    )

    args: argparse.Namespace = parser.parse_args()

    try:
        validate_identifier(args.system_name, field_name="target system name")
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    cases: list[tuple[str, dict[str, Any]]] = []

    if args.batch:
        batch_path = args.batch.resolve()
        if not batch_path.is_file():
            logger.error("Batch file not found: %s", batch_path)
            return 1
        try:
            batch_data: Any = json.loads(batch_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in --batch file: %s", e)
            return 1
        if not isinstance(batch_data, list) or not batch_data:
            logger.error("--batch file must contain a non-empty JSON array.")
            return 1
        seen_ids: set[str] = set()
        for idx, entry in enumerate(batch_data):
            if not isinstance(entry, dict) or "id" not in entry or "state" not in entry:
                logger.error("Batch entry %d must be an object with 'id' and 'state' fields.", idx)
                return 1
            try:
                case_id = validate_identifier(entry["id"], field_name=f"batch case id at entry {idx}")
            except ValueError as exc:
                logger.error(str(exc))
                return 1
            if case_id in seen_ids:
                logger.error("Duplicate batch case id: '%s'", case_id)
                return 1
            seen_ids.add(case_id)
            if not isinstance(entry["state"], dict):
                logger.error("Batch entry '%s' state must be a JSON object.", case_id)
                return 1
            cases.append((case_id, entry["state"]))
    else:
        raw_state: Any
        if args.state_file:
            state_file_path = args.state_file.resolve()
            if not state_file_path.is_file():
                logger.error("State file not found: %s", state_file_path)
                return 1
            try:
                raw_state = json.loads(state_file_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON provided in --state-file: %s", e)
                return 1
        else:
            try:
                raw_state = json.loads(args.state)
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON provided for --state argument: %s", e)
                return 1

        if not isinstance(raw_state, dict):
            logger.error("Initial state must be a JSON object.")
            return 1
        cases.append((timestamp, raw_state))

    runtime_profile: RuntimeResourceProfile | None = None
    task_spec: TaskSpec | None = None

    if args.task_spec:
        task_spec_path = args.task_spec.resolve()
        task_spec = TaskSpec.from_file(task_spec_path)
        if args.auto_setup or not setup_manifest_is_current(task_spec_path):
            logger.info(
                "Task setup is missing, stale, or --auto-setup was requested; ensuring setup for %s...",
                task_spec_path,
            )
            run_setup_for_task(
                task_spec_path,
                force=args.auto_setup,
                reinstall=args.reinstall,
                container=args.container,
                base_image=args.base_image,
            )
    if args.runtime_config:
        if task_spec is None:
            logger.error("--runtime-config requires --task-spec.")
            return 1
        try:
            runtime_profile = RuntimeResourceProfile.from_file(args.runtime_config)
            runtime_profile.validate_for_task(task_spec, check_local_sources=True)
        except (OSError, ValueError) as exc:
            logger.error("Invalid runtime resource profile: %s", exc)
            return 1

    session = StreamingSandboxSession(
        image=args.base_image,
        verbose=True,
        container_type=args.container,
    )

    try:
        logger.info("Opening sandbox session")
        session.open()

        if setup_sandbox_environment(session, reinstall=args.reinstall):
            if not provision_target_dependencies(session, args.system_name):
                return 1
            runtime_task_dir: str | None = None
            if args.task_spec:
                task_dir = args.task_spec.resolve().parent
                runtime_task_dir = copy_task_setup_to_sandbox(
                    session,
                    task_dir,
                    args.task_spec.resolve(),
                    task_spec.additional_documentation if task_spec else None,
                )
                if runtime_profile:
                    runtime_profile = stage_runtime_profile_sources(session, runtime_profile)
                preflight_ok = (
                    run_sandbox_preflight(session, runtime_task_dir, runtime_profile.model_dump_json())
                    if runtime_profile
                    else run_sandbox_preflight(session, runtime_task_dir)
                )
                if not preflight_ok:
                    logger.error("Sandbox preflight verification failed.")
                    return 1

            logger.info("Purging sandbox output and metrics directories")
            session.execute_command(f"rm -rf {SANDBOX_TARGET_METRICS_DIR}")
            session.execute_command(f"mkdir -p {SANDBOX_TARGET_METRICS_DIR}")
            session.execute_command(f"rm -rf {SANDBOX_WORKSPACE_DIR}/target_runs")
            session.execute_command(f"mkdir -p {SANDBOX_WORKSPACE_DIR}/target_runs")

            succeeded_cases: list[str] = []
            failed_cases: list[str] = []

            for case_id, case_state in cases:
                run_id = f"{timestamp}_{case_id}" if args.batch else case_id
                logger.info("Running case '%s' (run_id=%s)", case_id, run_id)

                ok = run_target_system_in_sandbox(
                    session,
                    args.system_name,
                    case_state,
                    run_id=run_id,
                    task_dir=runtime_task_dir,
                    runtime_profile=runtime_profile,
                )

                if ok:
                    succeeded_cases.append(case_id)
                else:
                    failed_cases.append(case_id)

                host_output_folder = f"data/output/{args.system_name}_{run_id}"
                sandbox_invocation_output = f"{SANDBOX_WORKSPACE_DIR}/target_runs/{run_id}/invocation/output"
                session.copy_dir_from_runtime(
                    src_dir=sandbox_invocation_output,
                    dest_dir=host_output_folder,
                    pattern="*",
                )
                logger.info("Output data copied to: %s", host_output_folder)

            logger.info("Checking for metrics files to copy back")
            session.copy_dir_from_runtime(
                src_dir=SANDBOX_TARGET_METRICS_DIR,
                dest_dir="target_metrics",
                pattern="*",
            )

            total = len(cases)
            if total > 1:
                logger.info(
                    "Batch complete: %d/%d succeeded, %d failed",
                    len(succeeded_cases),
                    total,
                    len(failed_cases),
                )
                if failed_cases:
                    logger.error("Failed cases: %s", ", ".join(failed_cases))

            logger.info("File copy process finished")
            return 1 if failed_cases else 0

        else:
            logger.error("Failed to set up the sandbox environment.")
            return 1

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        return 1

    finally:
        logger.info("Closing sandbox session")
        session.close()
        logger.info("Session closed.")


if __name__ == "__main__":
    sys.exit(main())
