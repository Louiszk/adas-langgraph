"""Create frozen task fixtures and preflight code in an isolated sandbox."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from adas_core.logging_config import get_logger, setup_logging
from adas_core.task_spec import TaskSpec
from sandbox.sandbox import StreamingSandboxSession, setup_sandbox_environment

logger = get_logger("create_setup")
_RUNTIME_TASK_DIR = "/sandbox/workspace/task_setup"


def setup_manifest_is_current(task_spec_path: Path) -> bool:
    """Return whether the frozen setup manifest was built from this TaskSpec."""
    manifest_path = task_spec_path.resolve().parent / "setup_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_hash = manifest.get("files", {}).get("task.json")
        current_hash = hashlib.sha256(task_spec_path.read_bytes()).hexdigest()
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(expected_hash, str) and expected_hash == current_hash


def _copy_tree_from_runtime(session: StreamingSandboxSession, runtime_dir: str, destination: Path) -> None:
    """Copy a generated directory tree back while preserving relative paths."""
    listing = session.execute_command(f"find {runtime_dir} -type f -print")
    paths = getattr(listing, "stdout", None) if not isinstance(listing, str) else listing
    paths = str(paths or "")
    for runtime_path in str(paths).splitlines():
        relative = Path(runtime_path).relative_to(Path(runtime_dir))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        session.copy_from_runtime(runtime_path, str(target))


def run_setup_for_task(
    task_spec_path: Path,
    *,
    force: bool = False,
    reinstall: bool = False,
    container: str = "auto",
    base_image: str | None = None,
) -> Path:
    """Run setup in its own sandbox and return the host task directory."""
    task_spec_path = task_spec_path.resolve()
    TaskSpec.from_file(task_spec_path)  # Fail before starting a sandbox for an invalid specification.
    task_dir = task_spec_path.parent
    if setup_manifest_is_current(task_spec_path) and not force:
        logger.info("Frozen task setup is current; use --force to regenerate it.")
        return task_dir
    session = StreamingSandboxSession(image=base_image, verbose=True, container_type=container)
    try:
        session.open()
        if not setup_sandbox_environment(session, reinstall):
            raise RuntimeError("Failed to set up sandbox environment")
        session.execute_command(f"mkdir -p {_RUNTIME_TASK_DIR}")
        session.copy_to_runtime(str(task_spec_path), f"{_RUNTIME_TASK_DIR}/task.json")
        command = (
            "python3 /sandbox/workspace/run_setup.py "
            f"--task-spec {_RUNTIME_TASK_DIR}/task.json --task-dir {_RUNTIME_TASK_DIR}"
        )
        result = session.execute_command(command)
        if getattr(result, "exit_code", 1) != 0:
            output = getattr(result, "stdout", "") or getattr(result, "stderr", "")
            raise RuntimeError(f"Sandbox task setup failed: {output}")
        _copy_tree_from_runtime(session, _RUNTIME_TASK_DIR, task_dir)
    finally:
        session.close()
    logger.info("Frozen task setup written to %s", task_dir)
    return task_dir


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Generate a TaskSpec's fixtures and preflight script in a sandbox.")
    parser.add_argument("--task-spec", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="Regenerate an existing setup.")
    parser.add_argument("--reinstall", action="store_true", help="Reinstall sandbox base dependencies.")
    parser.add_argument("--container", choices=["auto", "docker", "podman"], default="auto")
    parser.add_argument("--base-image", default=None)
    args = parser.parse_args()
    run_setup_for_task(
        args.task_spec,
        force=args.force,
        reinstall=args.reinstall,
        container=args.container,
        base_image=args.base_image,
    )


if __name__ == "__main__":
    main()
