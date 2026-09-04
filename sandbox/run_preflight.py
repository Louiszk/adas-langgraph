"""Provision a frozen task setup and run its preflight inside the sandbox."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from adas_core.environment import ensure_packages_installed, isolated_case_workspace, run_preflight_check


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a task setup's preflight check inside the sandbox.")
    parser.add_argument("--task-dir", required=True, type=Path)
    args = parser.parse_args()

    manifest_path = args.task_dir / "setup_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        packages = manifest["required_packages"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"Invalid setup manifest {manifest_path}: {exc}", file=sys.stderr)
        return 1

    try:
        ensure_packages_installed([str(package) for package in packages])
    except (RuntimeError, ValueError) as exc:
        print(f"Package provisioning failed: {exc}", file=sys.stderr)
        return 1

    with isolated_case_workspace(
        base_dir="/tmp/adas-preflight",
        run_id="preflight",
        case_id="environment",
        fixtures_dir=args.task_dir / "fixtures",
        clean_up=True,
    ) as workspace_dirs:
        is_ok, message = run_preflight_check(args.task_dir / "preflight.py", workspace_dirs)
    if not is_ok:
        print(f"Preflight check failed: {message}", file=sys.stderr)
        return 1
    print(f"Preflight verification passed: {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
