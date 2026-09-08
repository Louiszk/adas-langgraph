from __future__ import annotations

SETUP_REQUIREMENTS = ["wikipedia"]


import importlib
import os
from pathlib import Path
from typing import Iterable


REQUIRED_ENV_VARS: tuple[str, ...] = ()
REQUIRED_API_KEYS: tuple[str, ...] = ()


def _check_required_environment_variables() -> str | None:
    missing = [
        name
        for name in (*REQUIRED_ENV_VARS, *REQUIRED_API_KEYS)
        if not os.environ.get(name)
    ]
    if missing:
        return "Missing required environment variables: " + ", ".join(missing)
    return None


def _verify_readable_file(path: Path) -> str | None:
    if not path.exists():
        return f"Declared resource does not exist: {path}"
    if not path.is_file():
        return f"Declared resource is not a file: {path}"
    if not os.access(path, os.R_OK):
        return f"Declared file is not readable: {path}"

    try:
        with path.open("rb") as resource:
            resource.read(1)
    except OSError:
        return f"Unable to read declared file: {path}"
    return None


def _verify_directory(path: Path) -> str | None:
    if not path.exists():
        return f"Declared directory does not exist: {path}"
    if not path.is_dir():
        return f"Declared resource is not a directory: {path}"
    if not os.access(path, os.R_OK | os.X_OK):
        return f"Declared directory is not accessible: {path}"

    try:
        entries: Iterable[Path] = path.rglob("*")
        for entry in entries:
            if entry.is_file():
                error = _verify_readable_file(entry)
                if error:
                    return error
            elif entry.is_dir() and not os.access(entry, os.R_OK | os.X_OK):
                return f"Nested directory is not accessible: {entry}"
    except OSError as exc:
        return f"Unable to search declared directory {path}: {exc.__class__.__name__}"
    return None


def _verify_declared_resources(workspace_dirs: dict[str, str]) -> str | None:
    if not isinstance(workspace_dirs, dict):
        return "workspace_dirs must be a dictionary"

    for name, raw_path in workspace_dirs.items():
        if not isinstance(name, str) or not name:
            return "Declared workspace resource has an invalid name"
        if not isinstance(raw_path, str) or not raw_path.strip():
            return f"Declared workspace resource has an invalid path: {name}"

        path = Path(raw_path).expanduser()
        try:
            if path.is_dir():
                error = _verify_directory(path)
            else:
                error = _verify_readable_file(path)
        except OSError as exc:
            return f"Unable to access declared resource {name}: {exc.__class__.__name__}"

        if error:
            return f"{name}: {error}"
    return None


def _verify_required_packages() -> str | None:
    for package in SETUP_REQUIREMENTS:
        try:
            importlib.import_module(package)
        except ImportError:
            return f"Required package is unavailable: {package}"
    return None


def check_environment(workspace_dirs: dict[str, str]) -> tuple[bool, str]:
    error = _check_required_environment_variables()
    if error:
        return False, error

    error = _verify_required_packages()
    if error:
        return False, error

    error = _verify_declared_resources(workspace_dirs)
    if error:
        return False, error

    return True, "Environment verified"


if __name__ == "__main__":
    ok, message = check_environment({})
    print(message)
    raise SystemExit(0 if ok else 1)