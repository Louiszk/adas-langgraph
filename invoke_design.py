"""Run the meta-system design optimization loop in an isolated container sandbox."""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path

from adas_core.environment import (
    SANDBOX_FIXTURES_DIR,
    SANDBOX_GENERATED_SYSTEMS_DIR,
    SANDBOX_TASK_SPEC_PATH,
    SANDBOX_WORKSPACE_DIR,
)
from adas_core.helpers import escape_system_name, sanitize_identifier, validate_identifier
from adas_core.logging_config import get_logger, setup_logging
from adas_core.task_spec import TaskSpec
from config import settings
from create_setup import run_setup_for_task, setup_manifest_is_current
from sandbox.sandbox import (
    StreamingSandboxSession,
    copy_task_setup_to_sandbox,
    run_sandbox_preflight,
    setup_sandbox_environment,
)

logger = get_logger("invoke_design")


def run_meta_system_in_sandbox(
    session: StreamingSandboxSession,
    target_name: str,
    optimize_system: str | None = None,
) -> bool:
    escaped_target_name = escape_system_name(target_name)
    target_file_name = escaped_target_name + ".py"
    target_pickle_name = escaped_target_name + ".pkl"
    metrics_file = f"{escaped_target_name}.json"

    cmd_parts = [
        f"ADAS_FIXTURES_DIR={shlex.quote(SANDBOX_FIXTURES_DIR)}",
        f"ADAS_WORKSPACE_ROOT={shlex.quote('/tmp/adas-runs')}",
        "python3",
        f"{SANDBOX_WORKSPACE_DIR}/run_meta.py",
        f"--task-spec={shlex.quote(SANDBOX_TASK_SPEC_PATH)}",
        f"--system-name={shlex.quote(target_name)}",
        f"--max-iterations={shlex.quote(str(settings.max_iterations))}",
    ]
    if optimize_system:
        cmd_parts.append(f"--optimize-system={shlex.quote(optimize_system)}")

    command = (
        " ".join(cmd_parts) + '; meta_exit=$?; printf \'\\n__ADAS_META_EXIT__%s\\n\' "$meta_exit"; exit "$meta_exit"'
    )

    logger.info(f"Executing Meta-System Design Optimization for: {target_name}")

    output_chunks: list[str] = []
    for chunk in session.execute_command_streaming(command):
        output_chunks.append(chunk)
        print(chunk, end="", flush=True)

    meta_succeeded = "__ADAS_META_EXIT__0" in "".join(output_chunks)
    if not meta_succeeded:
        logger.error("Meta system execution failed inside container")
        return False

    logger.info("Meta system execution completed in sandbox. Copying generated systems and metrics back to host...")
    os.makedirs("generated_systems", exist_ok=True)

    # Verify and copy target code
    as_dir = str(session.execute_command(f"ls -la {SANDBOX_GENERATED_SYSTEMS_DIR}"))
    if target_file_name not in as_dir or target_pickle_name not in as_dir:
        logger.error(
            f"Expected artifacts {target_file_name} and/or {target_pickle_name} not found in {SANDBOX_GENERATED_SYSTEMS_DIR}"
        )
        return False

    session.copy_from_runtime(
        f"{SANDBOX_GENERATED_SYSTEMS_DIR}/{target_file_name}",
        f"generated_systems/{target_file_name}",
    )
    session.copy_from_runtime(
        f"{SANDBOX_GENERATED_SYSTEMS_DIR}/{target_pickle_name}",
        f"generated_systems/{target_pickle_name}",
    )
    logger.info(f"Copied {target_file_name} and {target_pickle_name} back to host")

    # Verify and copy metrics artifact
    metrics_listing = str(session.execute_command(f"ls -la {SANDBOX_GENERATED_SYSTEMS_DIR}/metrics"))
    if metrics_file not in metrics_listing:
        logger.error(f"Expected metrics artifact {metrics_file} not found in {SANDBOX_GENERATED_SYSTEMS_DIR}/metrics")
        return False

    os.makedirs("generated_systems/metrics", exist_ok=True)
    session.copy_from_runtime(
        f"{SANDBOX_GENERATED_SYSTEMS_DIR}/metrics/{metrics_file}",
        f"generated_systems/metrics/{metrics_file}",
    )
    logger.info(f"Copied metrics file {metrics_file} back to host")

    return True


def main() -> int:
    setup_logging()

    parser = argparse.ArgumentParser(description="Run agentic systems in a sandboxed environment")
    parser.add_argument("--reinstall", action="store_true", help="Reinstall dependencies.")
    parser.add_argument("--task-spec", type=Path, required=True, help="Validated TaskSpec JSON file")
    parser.add_argument(
        "--system-name",
        default=None,
        help="Optional target system name override (defaults to TaskSpec name).",
    )
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
    target_name = args.system_name or task_spec.name
    logger.info(f"Running with arguments: {args}")

    # Validate identifiers strictly before any container or file operations
    try:
        validate_identifier(target_name, field_name="target system name")
        if args.optimize_system:
            validate_identifier(args.optimize_system, field_name="optimize system name")
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    task_dir = args.task_spec.resolve().parent

    # Host-side check: ensure frozen validation module exists BEFORE setup or opening sandbox
    safe_spec_name = sanitize_identifier(task_spec.name)
    validation_candidates = [
        task_dir / f"{safe_spec_name}.validation.py",
        task_dir / f"{task_spec.name}.validation.py",
        task_dir / "validation.py",
    ]
    validation_file = next((p for p in validation_candidates if p.exists()), None)
    if validation_file is None:
        logger.error(
            "Frozen validation module not found for TaskSpec '%s' in %s. "
            "Please generate it before running design optimization: "
            "python create_validation.py --task-spec %s",
            task_spec.name,
            task_dir,
            args.task_spec,
        )
        return 1

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
                success = run_meta_system_in_sandbox(
                    session=session,
                    target_name=target_name,
                    optimize_system=args.optimize_system,
                )
                if success:
                    logger.info("Finished successfully!")
                    return 0
                else:
                    logger.error("Meta system execution failed.")
                    return 1
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


if __name__ == "__main__":
    raise SystemExit(main())
