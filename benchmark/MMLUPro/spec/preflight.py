SETUP_REQUIREMENTS = []

import os
from pathlib import Path
from urllib.parse import unquote, urlparse


REQUIRED_ENV_VARS: tuple[str, ...] = ()
REQUIRED_API_KEYS: tuple[str, ...] = ()


def _check_required_environment() -> tuple[bool, str]:
    missing = [
        name
        for name in (*REQUIRED_ENV_VARS, *REQUIRED_API_KEYS)
        if not os.environ.get(name)
    ]
    if missing:
        return False, "Missing required environment variables: " + ", ".join(missing)
    return True, ""


def _resource_path(resource: str) -> Path | None:
    parsed = urlparse(resource)

    if parsed.scheme in {"", "file"}:
        if parsed.scheme == "file":
            if parsed.netloc not in {"", "localhost"}:
                return None
            return Path(unquote(parsed.path))
        return Path(resource).expanduser()

    return None


def _verify_resource(label: str, resource: str) -> tuple[bool, str]:
    if not isinstance(resource, str) or not resource.strip():
        return False, f"Invalid resource declaration for '{label}'"

    path = _resource_path(resource)
    if path is None:
        return False, f"Unsupported resource type for '{label}'"

    try:
        path = path.resolve()
    except OSError:
        return False, f"Unable to resolve resource '{label}'"

    if not path.exists():
        return False, f"Resource '{label}' does not exist"

    try:
        if path.is_dir():
            if not os.access(path, os.R_OK | os.X_OK):
                return False, f"Resource directory '{label}' is not accessible"

            # Recursively traverse declared input directories, including
            # fixtures stored in relative subdirectories.
            for child in path.rglob("*"):
                try:
                    child.stat()
                except OSError:
                    return False, f"Unable to access an item in resource '{label}'"
        else:
            if not os.access(path, os.R_OK):
                return False, f"File resource '{label}' is not readable"
            path.stat()
    except OSError:
        return False, f"Unable to access resource '{label}'"

    return True, ""


def check_environment(workspace_dirs: dict[str, str]) -> tuple[bool, str]:
    environment_ok, environment_error = _check_required_environment()
    if not environment_ok:
        return False, environment_error

    if not isinstance(workspace_dirs, dict):
        return False, "workspace_dirs must be a dictionary"

    for label, resource in workspace_dirs.items():
        if not isinstance(label, str) or not label.strip():
            return False, "Workspace resource names must be non-empty strings"

        resource_ok, resource_error = _verify_resource(label, resource)
        if not resource_ok:
            return False, resource_error

    return True, "Environment verified"