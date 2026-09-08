"""Run a target agentic system in an isolated sandbox with TaskSpec environment support."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

from adas_core.environment import (
    SANDBOX_DATA_OUTPUT_DIR,
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
) -> None:
    """Constructs and executes the command to run the target system inside the sandbox."""
    cmd_parts = [f'python3 {SANDBOX_WORKSPACE_DIR}/run_target.py --system_name="{system_name}" --run-id="{run_id}"']

    state_str = json.dumps(state)
    quoted_state = state_str.replace('"', '\\"')
    cmd_parts.append(f'--state="{quoted_state}"')

    command = " ".join(cmd_parts)

    logger.info(f"Executing Target System: {system_name} (Run ID: {run_id})")
    logger.info(f"Initial State: {state_str}")

    for chunk in session.execute_command_streaming(command):
        print(chunk, end="", flush=True)

    logger.info("Target system execution completed")


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
    parser.add_argument(
        "--state",
        default='{"messages": []}',
        help="JSON string defining the initial state for the system.",
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

    try:
        initial_state: dict[str, Any] = json.loads(args.state)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON provided for --state argument: {e}. Using empty state.")
        initial_state = {}

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
            if args.task_spec:
                task_dir = args.task_spec.resolve().parent
                runtime_task_dir = copy_task_setup_to_sandbox(session, task_dir, args.task_spec.resolve())
                if not run_sandbox_preflight(session, runtime_task_dir):
                    logger.error("Sandbox preflight verification failed.")
                    return 1

            logger.info("Purging sandbox output and metrics directories")
            session.execute_command(f"rm -rf {SANDBOX_DATA_OUTPUT_DIR} && mkdir -p {SANDBOX_DATA_OUTPUT_DIR}")
            session.execute_command(f"rm -rf {SANDBOX_TARGET_METRICS_DIR} && mkdir -p {SANDBOX_TARGET_METRICS_DIR}")

            run_target_system_in_sandbox(session, args.system_name, initial_state, run_id=timestamp)

            logger.info("Checking for output data to copy back")
            host_output_folder = f"data/output/{args.system_name}_{timestamp}"

            session.copy_dir_from_runtime(
                src_dir=SANDBOX_DATA_OUTPUT_DIR,
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
