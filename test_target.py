import argparse
import datetime
import json
from typing import Any

from adas_core.logging_config import get_logger, setup_logging
from sandbox.sandbox import StreamingSandboxSession, setup_sandbox_environment

logger = get_logger("test_target")


def run_target_system_in_sandbox(
    session: StreamingSandboxSession,
    system_name: str,
    state: dict[str, Any],
    run_id: str,
) -> None:
    """Constructs and executes the command to run the target system inside the sandbox."""

    # Construct command with the run_id passed down
    cmd_parts = [f'python3 /sandbox/workspace/run_target.py --system_name="{system_name}" --run-id="{run_id}"']

    # Safely serialize and quote the initial state for the command line
    state_str = json.dumps(state)
    quoted_state = state_str.replace('"', '\\"')
    cmd_parts.append(f'--state="{quoted_state}"')

    command = " ".join(cmd_parts)

    logger.info(f"Executing Target System: {system_name} (Run ID: {run_id})")
    logger.info(f"Initial State: {state_str}")

    # Stream the output from the sandbox command
    for chunk in session.execute_command_streaming(command):
        print(chunk, end="", flush=True)

    logger.info("Target system execution completed")


def main() -> None:
    """Main function to set up and run a target agentic system in a sandboxed environment."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Run a target agentic system in a sandboxed environment.")
    parser.add_argument(
        "--system_name",
        required=True,
        help="Name of the target system to run (e.g., 'DataAnalystSystem_v0').",
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
        "--no-keep-template",
        dest="keep_template",
        action="store_false",
        default=True,
        help="Delete the sandbox image template after the session is closed (default: keep image).",
    )
    parser.add_argument(
        "--base-image",
        default="python:3.11-slim",
        help="The base container image to use for the sandbox.",
    )
    parser.add_argument(
        "--container",
        choices=["auto", "docker", "podman"],
        default="auto",
        help="Container runtime to use (auto tries Docker first, then Podman).",
    )

    args: argparse.Namespace = parser.parse_args()

    # Generate a synchronization timestamp for this specific run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parse the initial state from the JSON string argument
    try:
        initial_state: dict[str, Any] = json.loads(args.state)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON provided for --state argument: {e}. Using empty state.")
        initial_state = {}

    # Initialize the sandbox session with the specified configuration
    session = StreamingSandboxSession(
        image=args.base_image,
        keep_template=args.keep_template,
        verbose=True,
        container_type=args.container,
    )

    try:
        logger.info("Opening sandbox session")
        session.open()

        # Set up the sandbox environment, reinstalling dependencies if requested
        if setup_sandbox_environment(session, reinstall=args.reinstall):
            # Purge output and metrics directories to ensure a clean run
            logger.info("Purging sandbox output and metrics directories")
            session.execute_command("rm -rf /sandbox/workspace/data/output && mkdir -p /sandbox/workspace/data/output")
            session.execute_command(
                "rm -rf /sandbox/workspace/target_metrics && mkdir -p /sandbox/workspace/target_metrics"
            )

            # Run the target system with the provided state AND the timestamp
            run_target_system_in_sandbox(session, args.system_name, initial_state, run_id=timestamp)

            logger.info("Checking for output data to copy back")

            # Define specific output folder for this run
            host_output_folder = f"data/output/{args.system_name}_{timestamp}"

            session.copy_dir_from_runtime(
                src_dir="/sandbox/workspace/data/output",
                dest_dir=host_output_folder,
                pattern="*",
            )
            logger.info(f"Output data copied to: {host_output_folder}")

            logger.info("Checking for metrics files to copy back")
            session.copy_dir_from_runtime(
                src_dir="/sandbox/workspace/target_metrics",
                dest_dir="target_metrics",
                pattern="*",
            )

            logger.info("File copy process finished")

        else:
            logger.error("Failed to set up the sandbox environment.")

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")

    finally:
        logger.info("Closing sandbox session")
        session.close()
        logger.info("Session closed.")


if __name__ == "__main__":
    main()
