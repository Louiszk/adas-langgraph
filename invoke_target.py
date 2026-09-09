"""Run a target agentic system in an isolated sandbox with TaskSpec environment support."""

from __future__ import annotations

import argparse
import datetime
import json
import shlex
import sys
from pathlib import Path
from typing import Any

from adas_core.environment import (
    SANDBOX_TARGET_METRICS_DIR,
    SANDBOX_WORKSPACE_DIR,
)
from adas_core.logging_config import get_logger, setup_logging
from create_setup import run_setup_for_task, setup_manifest_is_current
from sandbox.sandbox import (
    StreamingSandboxSession,
    copy_task_setup_to_sandbox,
    run_sandbox_preflight,
    setup_sandbox_environment,
)

logger = get_logger("invoke_target")


def run_target_system_in_sandbox(
    session: StreamingSandboxSession,
    system_name: str,
    state: dict[str, Any],
    run_id: str,
    task_dir: str | None = None,
) -> bool:
    """Constructs and executes the command to run the target system inside the sandbox."""
    cmd_parts = [
        "python3 "
        f"{shlex.quote(SANDBOX_WORKSPACE_DIR + '/run_target.py')} "
        f"--system_name={shlex.quote(system_name)} --run-id={shlex.quote(run_id)}"
    ]
    if task_dir:
        cmd_parts.append(f"--task-dir={shlex.quote(task_dir)}")

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


def main() -> int:
    """Main function to set up and run a target agentic system in a sandboxed environment."""
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

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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

    initial_state: dict[str, Any] = raw_state

    if args.task_spec:
        task_spec_path = args.task_spec.resolve()
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

    session = StreamingSandboxSession(
        image=args.base_image,
        verbose=True,
        container_type=args.container,
    )

    try:
        logger.info("Opening sandbox session")
        session.open()

        if setup_sandbox_environment(session, reinstall=args.reinstall):
            runtime_task_dir: str | None = None
            if args.task_spec:
                task_dir = args.task_spec.resolve().parent
                runtime_task_dir = copy_task_setup_to_sandbox(session, task_dir, args.task_spec.resolve())
                if not run_sandbox_preflight(session, runtime_task_dir):
                    logger.error("Sandbox preflight verification failed.")
                    return 1

            logger.info("Purging sandbox output and metrics directories")
            session.execute_command(f"rm -rf {SANDBOX_TARGET_METRICS_DIR}")
            session.execute_command(f"mkdir -p {SANDBOX_TARGET_METRICS_DIR}")
            session.execute_command(f"rm -rf {SANDBOX_WORKSPACE_DIR}/target_runs")
            session.execute_command(f"mkdir -p {SANDBOX_WORKSPACE_DIR}/target_runs")

            if not run_target_system_in_sandbox(
                session,
                args.system_name,
                initial_state,
                run_id=timestamp,
                task_dir=runtime_task_dir,
            ):
                return 1

            logger.info("Checking for output data to copy back")
            host_output_folder = f"data/output/{args.system_name}_{timestamp}"

            sandbox_invocation_output = f"{SANDBOX_WORKSPACE_DIR}/target_runs/{timestamp}/invocation/output"
            session.copy_dir_from_runtime(
                src_dir=sandbox_invocation_output,
                dest_dir=host_output_folder,
                pattern="*",
            )
            logger.info(f"Output data copied to: {host_output_folder}")

            logger.info("Checking for metrics files to copy back")
            session.copy_dir_from_runtime(
                src_dir=SANDBOX_TARGET_METRICS_DIR,
                dest_dir="target_metrics",
                pattern="*",
            )

            logger.info("File copy process finished")
            return 0

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
