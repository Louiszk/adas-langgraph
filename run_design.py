import argparse
import os
from pathlib import Path

from adas_core.environment import (
    SANDBOX_FIXTURES_DIR,
    SANDBOX_GENERATED_SYSTEMS_DIR,
    SANDBOX_TASK_SETUP_DIR,
    SANDBOX_WORKSPACE_DIR,
)
from adas_core.helpers import escape_system_name
from adas_core.logging_config import get_logger, setup_logging
from adas_core.task_spec import TaskSpec
from config import settings
from create_setup import run_setup_for_task, setup_manifest_is_current
from sandbox.sandbox import StreamingSandboxSession, setup_sandbox_environment

logger = get_logger("run_design")


def run_meta_system_in_sandbox(
    session: StreamingSandboxSession,
    problem_statement,
    target_name,
    optimize_system=None,
):
    quoted_problem = problem_statement.replace('"', '\\"')
    command = (
        f"ADAS_FIXTURES_DIR={SANDBOX_FIXTURES_DIR} "
        "ADAS_WORKSPACE_ROOT=/tmp/adas-runs "
        f'python3 {SANDBOX_WORKSPACE_DIR}/run_meta.py "{quoted_problem}" "{target_name}" "{settings.max_iterations}" '
    )
    command += f'"{optimize_system}"' if optimize_system else ""

    for chunk in session.execute_command_streaming(command):
        print(chunk, end="", flush=True)

    logger.info("Meta system execution completed!")

    if "generated_systems" in str(session.execute_command(f"ls -la {SANDBOX_WORKSPACE_DIR}")):
        logger.info("Copying generated systems and metrics back to host...")
        os.makedirs("generated_systems", exist_ok=True)
        escaped_target_name = escape_system_name(target_name)
        target_file_name = escaped_target_name + ".py"
        target_pickle_name = escaped_target_name + ".pkl"

        as_dir = str(session.execute_command(f"ls -la {SANDBOX_GENERATED_SYSTEMS_DIR}"))
        if target_file_name in as_dir:
            session.copy_from_runtime(
                f"{SANDBOX_GENERATED_SYSTEMS_DIR}/{target_file_name}",
                f"generated_systems/{target_file_name}",
            )
        if target_pickle_name in as_dir:
            session.copy_from_runtime(
                f"{SANDBOX_GENERATED_SYSTEMS_DIR}/{target_pickle_name}",
                f"generated_systems/{target_pickle_name}",
            )
        logger.info(f"Copied {target_file_name} and .pkl back to host")

        if "metrics" in str(session.execute_command(f"ls -la {SANDBOX_GENERATED_SYSTEMS_DIR}")):
            metrics_file = f"{escaped_target_name}.json"

            if metrics_file in str(session.execute_command(f"ls -la {SANDBOX_GENERATED_SYSTEMS_DIR}/metrics")):
                os.makedirs("generated_systems/metrics", exist_ok=True)
                session.copy_from_runtime(
                    f"{SANDBOX_GENERATED_SYSTEMS_DIR}/metrics/{metrics_file}",
                    f"generated_systems/metrics/{metrics_file}",
                )
                logger.info(f"Copied metrics file {metrics_file} back to host")

    return True


def copy_task_setup_to_sandbox(session: StreamingSandboxSession, task_dir: Path, task_spec_path: Path) -> str:
    """Copy the visible, frozen setup artifacts into a design sandbox."""
    runtime_task_dir = SANDBOX_TASK_SETUP_DIR
    session.execute_command(f"mkdir -p {runtime_task_dir}")
    for source_path in task_dir.rglob("*"):
        if source_path.is_file() and "__pycache__" not in source_path.parts:
            relative_path = source_path.relative_to(task_dir).as_posix()
            session.copy_to_runtime(str(source_path), f"{runtime_task_dir}/{relative_path}")
    # The sandbox entry point always loads the explicit, visible TaskSpec from
    # this stable path, regardless of the host file's chosen name.
    session.copy_to_runtime(str(task_spec_path), f"{runtime_task_dir}/task.json")
    return runtime_task_dir


def run_sandbox_preflight(session: StreamingSandboxSession, runtime_task_dir: str) -> bool:
    """Install frozen setup requirements and validate them inside the sandbox."""
    result = session.execute_command(f"python3 {SANDBOX_WORKSPACE_DIR}/run_preflight.py --task-dir {runtime_task_dir}")
    if getattr(result, "exit_code", 1) == 0:
        logger.info("Sandbox preflight verification passed.")
        return True
    output = getattr(result, "stdout", "") or getattr(result, "stderr", "") or result
    logger.error("Sandbox preflight verification failed: %s", output)
    return False


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Run agentic systems in a sandboxed environment")
    parser.add_argument("--reinstall", action="store_true", help="Reinstall dependencies.")
    parser.add_argument("--task-spec", type=Path, required=True, help="Validated TaskSpec JSON file")
    parser.add_argument(
        "--auto-setup",
        action="store_true",
        help="Generate frozen fixtures and preflight artifacts in a separate sandbox before design.",
    )
    parser.add_argument(
        "--optimize-system",
        default=None,
        help="Specify target system name to optimize or change",
    )
    parser.add_argument(
        "--container",
        choices=["auto", "docker", "podman"],
        default="auto",
        help="Container runtime to use (auto will try Docker first, then Podman)",
    )
    parser.add_argument(
        "--base-image",
        default=None,
        help="The base container image to use for the sandbox.",
    )
    args = parser.parse_args()
    task_spec = TaskSpec.from_file(args.task_spec)
    problem_statement = task_spec.system_goal
    target_name = task_spec.name
    logger.info(f"Running with arguments: {args}")

    if args.auto_setup or not setup_manifest_is_current(args.task_spec):
        logger.info(
            "Task setup is missing, stale, or --auto-setup was requested; ensuring setup for %s...",
            args.task_spec,
        )
        run_setup_for_task(
            args.task_spec,
            force=args.auto_setup,
            reinstall=args.reinstall,
            container=args.container,
            base_image=args.base_image,
        )

    task_dir = args.task_spec.resolve().parent

    session = StreamingSandboxSession(
        image=args.base_image,
        verbose=True,
        container_type=args.container,
    )

    try:
        session.open()
        if setup_sandbox_environment(session, args.reinstall):
            runtime_task_dir = copy_task_setup_to_sandbox(session, task_dir, args.task_spec.resolve())
            if run_sandbox_preflight(session, runtime_task_dir):
                run_meta_system_in_sandbox(session, problem_statement, target_name, args.optimize_system)
                logger.info("Finished successfully!")
            else:
                return 1
        else:
            logger.error("Failed to set up sandbox environment")
            return 1
    except Exception as e:
        logger.exception(f"Error during execution: {e}")
        return 1
    finally:
        logger.info("Session closed.")
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
