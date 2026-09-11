from __future__ import annotations

SETUP_REQUIREMENTS = []


import os
from pathlib import Path
from urllib.parse import unquote, urlparse


REQUIRED_ENV_VARS: tuple[str, ...] = ()
REQUIRED_API_KEYS: tuple[str, ...] = ()


def _missing_environment_variables() -> list[str]:
    required = tuple(dict.fromkeys((*REQUIRED_ENV_VARS, *REQUIRED_API_KEYS)))
    return [name for name in required if not os.environ.get(name)]


def _verify_path(label: str, value: str) -> str | None:
    expanded = os.path.expandvars(os.path.expanduser(value))
    parsed = urlparse(expanded)

    if parsed.scheme and parsed.scheme != "file":
        if parsed.scheme == "sqlite":
            database_path = parsed.path
            if parsed.netloc and not database_path:
                database_path = parsed.netloc
            if not database_path:
                return f"{label}: SQLite resource has no database path"
            expanded = unquote(database_path)
        else:
            return (
                f"{label}: unsupported resource scheme "
                f"'{parsed.scheme}'; no database driver is configured"
            )
    elif parsed.scheme == "file":
        expanded = unquote(parsed.path)

    path = Path(expanded)
    try:
        path = path.resolve(strict=True)
    except FileNotFoundError:
        return f"{label}: resource does not exist: {expanded}"
    except OSError as exc:
        return f"{label}: cannot resolve resource '{expanded}': {exc}"

    if not os.access(path, os.R_OK):
        return f"{label}: resource is not readable: {path}"

    if path.is_dir():
        try:
            for child in path.rglob("*"):
                if not os.access(child, os.R_OK):
                    return f"{label}: unreadable resource: {child}"
        except OSError as exc:
            return f"{label}: cannot recursively inspect '{path}': {exc}"
    elif not path.is_file():
        return f"{label}: resource is neither a file nor directory: {path}"

    return None


def check_environment(workspace_dirs: dict[str, str]) -> tuple[bool, str]:
    """Verify required variables and access to declared workspace resources."""
    missing = _missing_environment_variables()
    if missing:
        names = ", ".join(missing)
        return False, f"Missing required environment variable(s): {names}"

    if workspace_dirs is None:
        return False, "workspace_dirs must be a dictionary"

    if not isinstance(workspace_dirs, dict):
        return False, "workspace_dirs must be a dictionary"

    for label, resource in workspace_dirs.items():
        if not isinstance(label, str) or not label.strip():
            return False, "workspace_dirs contains an invalid resource name"
        if not isinstance(resource, str) or not resource.strip():
            return False, f"{label}: resource path must be a non-empty string"

        error = _verify_path(label, resource)
        if error is not None:
            return False, error

    return True, "Environment verified"