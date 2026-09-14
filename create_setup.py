"""Create frozen task fixtures and preflight code in an isolated sandbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from adas_core.environment import SANDBOX_TASK_SETUP_DIR, SANDBOX_WORKSPACE_DIR, load_environment
from adas_core.fixture_lifecycle import FIXTURE_LIFECYCLE_VERSION
from adas_core.logging_config import get_logger, setup_logging
from adas_core.task_spec import TaskSpec
from sandbox.sandbox import StreamingSandboxSession, setup_sandbox_environment

logger = get_logger("create_setup")
_RUNTIME_TASK_DIR = SANDBOX_TASK_SETUP_DIR
_TEXT_HASH_SUFFIXES = {".json", ".py", ".md", ".txt", ".yaml", ".yml"}


def _file_hash(path: Path) -> str:
    """Return the manifest hash for a setup artifact, normalizing text newlines."""
    contents = path.read_bytes()
    if path.suffix.lower() in _TEXT_HASH_SUFFIXES:
        contents = contents.replace(b"\r\n", b"\n")
    return hashlib.sha256(contents).hexdigest()


def verify_setup_manifest(task_spec_path: Path) -> tuple[bool, list[str]]:
    """Return whether every declared frozen setup artifact matches its manifest, plus diagnostic issues."""
    task_spec_path = task_spec_path.resolve()
    task_dir = task_spec_path.parent
    manifest_path = task_dir / "setup_manifest.json"
    if not manifest_path.is_file():
        return False, ["Missing setup_manifest.json"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, [f"Unreadable setup_manifest.json: {exc}"]

    issues: list[str] = []
    if manifest.get("fixture_lifecycle_version") != FIXTURE_LIFECYCLE_VERSION:
        issues.append(
            f"fixture_lifecycle_version mismatch (manifest={manifest.get('fixture_lifecycle_version')}, current={FIXTURE_LIFECYCLE_VERSION})"
        )

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        issues.append("Manifest 'files' mapping is missing or empty")
        return False, issues

    expected_task_hash = files.get("task.json")
    if not isinstance(expected_task_hash, str):
        issues.append("Manifest 'files' mapping missing 'task.json'")
    elif _file_hash(task_spec_path) != expected_task_hash:
        issues.append(
            f"Hash mismatch for task.json (expected: {expected_task_hash[:12]}..., actual: {_file_hash(task_spec_path)[:12]}...)"
        )

    resolved_root = task_dir.resolve()
    for relative_name, expected_hash in files.items():
        if relative_name == "task.json":
            continue
        if not isinstance(relative_name, str) or not isinstance(expected_hash, str):
            issues.append(f"Invalid manifest file entry: {relative_name!r}")
            continue
        artifact = (task_dir / relative_name).resolve()
        if not artifact.is_relative_to(resolved_root) or not artifact.is_file():
            issues.append(f"Missing artifact file: {relative_name}")
            continue
        actual_hash = _file_hash(artifact)
        if actual_hash != expected_hash:
            issues.append(
                f"Hash mismatch for {relative_name} (expected: {expected_hash[:12]}..., actual: {actual_hash[:12]}...)"
            )

    return len(issues) == 0, issues


def setup_manifest_is_current(task_spec_path: Path) -> bool:
    """Return whether every declared frozen setup artifact still matches its manifest."""
    is_current, _ = verify_setup_manifest(task_spec_path)
    return is_current


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


def _existing_validation_section(task_dir: Path) -> dict[str, Any] | None:
    """Read validation metadata before setup regeneration replaces the shared manifest."""
    try:
        manifest = json.loads((task_dir / "setup_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    validation = manifest.get("validation") if isinstance(manifest, dict) else None
    return validation if isinstance(validation, dict) else None


def _restore_validation_section(task_dir: Path, validation: Mapping[str, Any] | None) -> None:
    """Preserve validation metadata while allowing its hashes to report staleness after setup changes."""
    if validation is None:
        return
    manifest_path = task_dir / "setup_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Generated setup manifest is invalid: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"Generated setup manifest must contain an object: {manifest_path}")
    manifest["validation"] = validation
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    previous_validation = _existing_validation_section(task_dir)
    session = StreamingSandboxSession(image=base_image, verbose=True, container_type=container)
    try:
        session.open()
        if not setup_sandbox_environment(session, reinstall):
            raise RuntimeError("Failed to set up sandbox environment")
        session.execute_command(f"mkdir -p {_RUNTIME_TASK_DIR}")
        session.copy_to_runtime(str(task_spec_path), f"{_RUNTIME_TASK_DIR}/task.json")
        command = (
            f"python3 {SANDBOX_WORKSPACE_DIR}/run_setup.py "
            f"--task-spec {_RUNTIME_TASK_DIR}/task.json --task-dir {_RUNTIME_TASK_DIR}"
        )
        result = session.execute_command(command)
        if getattr(result, "exit_code", 1) != 0:
            output = getattr(result, "stdout", "") or getattr(result, "stderr", "")
            raise RuntimeError(f"Sandbox task setup failed: {output}")
        for generated_dir in ("fixtures", "setup_scripts"):
            target_dir_path = task_dir / generated_dir
            if target_dir_path.exists():
                shutil.rmtree(target_dir_path)
        _copy_tree_from_runtime(session, _RUNTIME_TASK_DIR, task_dir)
        _restore_validation_section(task_dir, previous_validation)
    finally:
        session.close()
    logger.info("Frozen task setup written to %s", task_dir)
    return task_dir


def main(argv: list[str] | None = None) -> int:
    load_environment()
    setup_logging()
    parser = argparse.ArgumentParser(description="Generate a TaskSpec's fixtures and preflight script in a sandbox.")
    parser.add_argument("--task-spec", required=True, type=Path)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify whether the setup manifest is current and exit without making changes.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate an existing setup.")
    parser.add_argument("--reinstall", action="store_true", help="Reinstall sandbox base dependencies.")
    parser.add_argument("--container", choices=["auto", "docker", "podman"], default="auto")
    parser.add_argument("--base-image", default=None)
    args = parser.parse_args(argv)

    if args.verify:
        try:
            TaskSpec.from_file(args.task_spec)
        except Exception as exc:
            logger.error("TaskSpec validation failed for '%s': %s", args.task_spec, exc)
            return 1

        is_current, issues = verify_setup_manifest(args.task_spec)
        if is_current:
            logger.info("Frozen task setup is current for '%s'.", args.task_spec)
            return 0
        logger.error(
            "Frozen task setup is stale or incomplete for '%s':\n  - %s\nRun 'python create_setup.py --task-spec %s --force' to regenerate.",
            args.task_spec,
            "\n  - ".join(issues),
            args.task_spec,
        )
        return 1

    run_setup_for_task(
        args.task_spec,
        force=args.force,
        reinstall=args.reinstall,
        container=args.container,
        base_image=args.base_image,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
