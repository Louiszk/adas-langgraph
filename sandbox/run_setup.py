"""Run task fixture synthesis and materialization inside the sandbox."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from adas_core.automatic_setup import AutomaticSetup
from adas_core.environment import _PACKAGE_PATTERN, ensure_packages_installed
from adas_core.task_spec import TaskSpec

__all__ = ["_PACKAGE_PATTERN", "install_packages", "write_manifest"]


def install_packages(packages: list[str]) -> None:
    """Install TaskSpec and generated setup dependencies in the sandbox."""
    ensure_packages_installed(packages)


def write_manifest(task_spec: TaskSpec, task_dir: Path, packages: list[str]) -> Path:
    """Record the frozen setup output used by future design runs."""
    files = {
        path.relative_to(task_dir).as_posix(): hashlib.sha256(
            path.read_bytes().replace(b"\r\n", b"\n")
            if path.suffix in {".json", ".py", ".md", ".txt", ".yaml", ".yml"}
            else path.read_bytes()
        ).hexdigest()
        for path in sorted(task_dir.rglob("*"))
        if path.is_file() and path.name != "setup_manifest.json"
    }
    manifest = {
        "schema_version": "1.0",
        "task_name": task_spec.name,
        "task_schema_version": task_spec.schema_version,
        "required_packages": packages,
        "files": files,
    }
    manifest_path = task_dir / "setup_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and materialize a TaskSpec's setup inside the sandbox.")
    parser.add_argument("--task-spec", required=True, type=Path)
    parser.add_argument("--task-dir", required=True, type=Path)
    args = parser.parse_args()

    task_spec = TaskSpec.from_file(args.task_spec)
    task_dir = args.task_dir
    task_dir.mkdir(parents=True, exist_ok=True)
    setup = AutomaticSetup()
    result = setup.generate_all(task_spec, task_dir, execute_generated_code=False)
    install_packages(result.discovered_packages)
    setup.execute_generated_artifacts(task_spec, task_dir, result.created_files)
    manifest_path = write_manifest(task_spec, task_dir, result.discovered_packages)
    print(f"Task setup completed: {manifest_path}")


if __name__ == "__main__":
    main()
