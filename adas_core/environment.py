from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from packaging.utils import canonicalize_name

from adas_core.helpers import normalize_fixture_path
from config.dependencies import DEFAULT_EXCLUDED_PACKAGES
from config.logging import get_logger

logger = get_logger("adas_core.environment")


def get_core_package_versions() -> list[str]:
    """Return installed versions of the core LangChain packages."""
    packages: list[str] = []
    for package_name in ("langchain-core", "langgraph"):
        try:
            packages.append(f"{package_name} {importlib.metadata.version(package_name)}")
        except importlib.metadata.PackageNotFoundError:
            continue
    return packages


def load_environment() -> None:
    """Load environment variables from project and sandbox .env files."""
    load_dotenv(find_dotenv(usecwd=True))
    sandbox_env = Path(SANDBOX_WORKSPACE_DIR) / ".env"
    if sandbox_env.is_file():
        load_dotenv(sandbox_env)


ADAS_WORKSPACE_DIR_ENV = "ADAS_WORKSPACE_DIR"
ADAS_INPUT_DIR_ENV = "ADAS_INPUT_DIR"
ADAS_OUTPUT_DIR_ENV = "ADAS_OUTPUT_DIR"

# Central sandbox and container filesystem paths
SANDBOX_WORKSPACE_DIR = "/sandbox/workspace"
SANDBOX_TASK_SETUP_DIR = f"{SANDBOX_WORKSPACE_DIR}/task_setup"
SANDBOX_TASK_SPEC_PATH = f"{SANDBOX_TASK_SETUP_DIR}/task.json"
SANDBOX_FIXTURES_DIR = f"{SANDBOX_TASK_SETUP_DIR}/fixtures"
SANDBOX_GENERATED_SYSTEMS_DIR = f"{SANDBOX_WORKSPACE_DIR}/generated_systems"
SANDBOX_TARGET_METRICS_DIR = f"{SANDBOX_WORKSPACE_DIR}/target_metrics"
SANDBOX_DATA_DIR = f"{SANDBOX_WORKSPACE_DIR}/data"
SANDBOX_DATA_OUTPUT_DIR = f"{SANDBOX_DATA_DIR}/output"

_PACKAGE_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[A-Za-z0-9._,-]+\])?(?:\s*(?:==|!=|<=|>=|<|>|~=)\s*[A-Za-z0-9.*+!._-]+(?:\s*,\s*(?:==|!=|<=|>=|<|>|~=)\s*[A-Za-z0-9.*+!._-]+)*)?$"
)


def validate_package_requirement(requirement: str, *, raise_on_error: bool = False) -> bool:
    """Validate that a package requirement string has a valid format and contains no unsafe characters."""
    if not requirement or not isinstance(requirement, str) or not _PACKAGE_PATTERN.fullmatch(requirement.strip()):
        if raise_on_error:
            raise ValueError(f"Invalid package requirement: '{requirement}'")
        return False
    return True


def extract_literal_package_requirements(code: str, declaration_name: str) -> list[str]:
    """Extract and validate a literal package declaration from generated source code."""
    try:
        parsed = ast.parse(code)
    except SyntaxError:
        return []

    for node in parsed.body:
        value_node: ast.expr | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == declaration_name for target in node.targets
        ):
            value_node = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == declaration_name
            and node.value is not None
        ):
            value_node = node.value

        if value_node is None:
            continue

        try:
            value = ast.literal_eval(value_node)
        except (ValueError, TypeError):
            return []
        if not isinstance(value, list):
            return []

        requirements = [str(item) for item in value]
        invalid = [requirement for requirement in requirements if not validate_package_requirement(requirement)]
        if invalid:
            raise ValueError(f"Invalid {declaration_name} package requirement(s): {invalid}")

        return [requirement for requirement in requirements if not is_package_excluded(requirement)]

    return []


def normalize_package_name(package_spec: str) -> str:
    """Extract canonical distribution name from requirement specifier using PEP 503 rules."""
    # Strip extras, versions, and markers: e.g. "uvicorn[standard]>=0.20.0" -> "uvicorn"
    name = re.split(r"[><=~!\[;,]", package_spec.strip())[0].strip()
    return canonicalize_name(name)


def is_package_excluded(
    package_spec: str,
    excluded_packages: Sequence[str] | None = None,
) -> bool:
    """Return True if the package matches any excluded distribution using exact canonical name comparison."""
    if not package_spec:
        return False
    canonical = normalize_package_name(package_spec)
    excluded = excluded_packages if excluded_packages is not None else DEFAULT_EXCLUDED_PACKAGES
    excluded_canonical = {normalize_package_name(p) for p in excluded}
    return canonical in excluded_canonical


def is_package_installed(package_name: str) -> bool:
    """Check if a package distribution is installed in the current environment and satisfies any version constraints."""
    try:
        from packaging.requirements import Requirement

        req = Requirement(package_name)
        canonical = canonicalize_name(req.name)
        try:
            installed_ver = importlib.metadata.version(canonical)
            if req.specifier and installed_ver not in req.specifier:
                return False
            return True
        except (importlib.metadata.PackageNotFoundError, ValueError):
            if req.specifier:
                return False
            try:
                importlib.import_module(canonical.replace("-", "_"))
                return True
            except (ImportError, ValueError):
                return False
    except Exception:
        canonical = normalize_package_name(package_name)
        try:
            importlib.metadata.version(canonical)
            return True
        except (importlib.metadata.PackageNotFoundError, ValueError):
            try:
                importlib.import_module(canonical.replace("-", "_"))
                return True
            except (ImportError, ValueError):
                return False


def ensure_packages_installed(
    packages: list[str],
    python_executable: str | None = None,
) -> list[str]:
    """Ensure required packages are installed, installing missing ones via pip.

    Args:
        packages: List of package requirements (e.g. ['neo4j>=5.0', 'fastapi']).
        python_executable: Optional python executable to run pip (defaults to sys.executable).

    Returns:
        List of packages that were installed during this call.

    Raises:
        ValueError: If any package specifier contains invalid characters.
        RuntimeError: If pip install fails.
    """
    if not packages:
        return []

    # Validate all requirements before running any commands
    invalid = [pkg for pkg in packages if not validate_package_requirement(pkg)]
    if invalid:
        raise ValueError(f"Invalid package requirement(s): {invalid}")

    installable = [pkg for pkg in packages if not is_package_excluded(pkg)]
    missing: list[str] = []
    for pkg in installable:
        pkg_clean = pkg.strip()
        if not is_package_installed(pkg_clean):
            missing.append(pkg_clean)

    if not missing:
        logger.debug("All required packages are already installed.")
        return []

    python_exe = python_executable or sys.executable
    logger.info(f"Installing missing package(s) via pip: {missing}")
    cmd = [python_exe, "-m", "pip", "install", "--disable-pip-version-check", *missing]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.debug(f"pip install output:\n{result.stdout}")
        return missing
    except subprocess.CalledProcessError as e:
        err_msg = f"Failed to install packages {missing}: {e.stderr or e.stdout}"
        logger.error(err_msg)
        raise RuntimeError(err_msg) from e


def get_installed_packages_from_metrics(metrics_file: str | Path) -> list[str]:
    """Extract dynamically installed package requirements from a metrics file."""
    path = Path(metrics_file)
    if not path.is_file():
        return []

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        packages_value = data.get("installed_packages")
        if isinstance(packages_value, list):
            packages = [str(package).strip() for package in packages_value if str(package).strip()]
        elif isinstance(packages_value, str):
            packages = [package.strip() for package in packages_value.split() if package.strip()]
        else:
            return []

        invalid = [package for package in packages if not validate_package_requirement(package)]
        if invalid:
            raise ValueError(f"Invalid package requirement(s) in {path}: {invalid}")
        return list(dict.fromkeys(packages))
    except (OSError, json.JSONDecodeError, TypeError, AttributeError) as exc:
        logger.warning("Failed to parse installed packages from %s: %s", path, exc)
    return []


PROVISIONING_SCRIPT_PREFIXES = ("generate_", "seed_", "mock_", "setup_")


def is_provisioning_script(path: Path | str) -> bool:
    """Return True if path is a Python fixture provisioning or setup script."""
    p = Path(path)
    return p.suffix == ".py" and p.name.startswith(PROVISIONING_SCRIPT_PREFIXES)


def copy_directory_contents(
    src: Path,
    dest: Path,
    *,
    exclude_provisioning_scripts: bool = True,
) -> int:
    """Copy all files and subdirectories from src to dest preserving structure."""
    if not src.exists() or not src.is_dir():
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in src.rglob("*"):
        if item.is_file():
            # Skip python bytecode and cache dirs
            if "__pycache__" in item.parts or item.suffix in (".pyc", ".pyo"):
                continue
            if exclude_provisioning_scripts and is_provisioning_script(item):
                continue
            relative = item.relative_to(src)
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            count += 1
    return count


@contextmanager
def isolated_case_workspace(
    base_dir: Path | str | None = None,
    run_id: str | None = None,
    case_id: str = "case_default",
    fixtures_dir: Path | str | None = None,
    allowed_files: list[str] | None = None,
    clean_up: bool = False,
) -> Iterator[dict[str, Path]]:
    """Context manager setting up isolated input, output, and workspace directories for a test case.

    Sets ADAS_WORKSPACE_DIR, ADAS_INPUT_DIR, and ADAS_OUTPUT_DIR in os.environ for the duration
    of the block, and restores previous values on exit.

    Args:
        base_dir: Base directory for runs (defaults to temp directory / 'adas_runs').
        run_id: Unique run identifier (defaults to random 8-char hex).
        case_id: Identifier for the test case.
        fixtures_dir: Optional path to directory containing frozen fixtures to seed into input/.
        allowed_files: Optional list of relative file paths to seed into input/. If None, seeds all fixtures.
        clean_up: If True, deletes the case directory upon exit.

    Yields:
        Dictionary mapping 'workspace', 'input', and 'output' to their respective Path objects.
    """
    root_base = Path(base_dir) if base_dir else Path(tempfile.gettempdir()) / "adas_runs"
    effective_run_id = run_id or uuid.uuid4().hex[:8]

    # Sanitize case_id for filesystem safety
    safe_case_id = re.sub(r"[^A-Za-z0-9_.-]", "_", case_id)
    case_dir = root_base / effective_run_id / safe_case_id
    input_dir = case_dir / "input"
    output_dir = case_dir / "output"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed fixtures into input directory if available
    if fixtures_dir:
        fixtures_path = Path(fixtures_dir)
        if fixtures_path.exists():
            if allowed_files is not None:
                copied_count = 0
                resolved_fixtures = fixtures_path.resolve()
                resolved_input = input_dir.resolve()
                for rel_path in allowed_files:
                    clean_rel = normalize_fixture_path(rel_path)
                    src = (fixtures_path / clean_rel).resolve()
                    target = (input_dir / clean_rel).resolve()

                    # Containment verification: ensure paths do not escape base directories
                    try:
                        src.relative_to(resolved_fixtures)
                    except ValueError:
                        raise ValueError(f"Invalid fixture path '{rel_path}': path escapes fixtures directory.")
                    try:
                        target.relative_to(resolved_input)
                    except ValueError:
                        raise ValueError(f"Invalid fixture path '{rel_path}': path escapes input directory.")

                    if not src.exists():
                        continue
                    if src.is_file():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, target)
                        copied_count += 1
                    elif src.is_dir():
                        # The TaskSpec explicitly declared this artifact. It may
                        # legitimately contain Python source such as setup.py.
                        copied_count += copy_directory_contents(src, target, exclude_provisioning_scripts=False)
            else:
                copied_count = copy_directory_contents(fixtures_path, input_dir, exclude_provisioning_scripts=True)
            logger.debug(f"Seeded {copied_count} fixture file(s) into {input_dir}")

    # Preserve previous environment variables
    old_env = {
        ADAS_WORKSPACE_DIR_ENV: os.environ.get(ADAS_WORKSPACE_DIR_ENV),
        ADAS_INPUT_DIR_ENV: os.environ.get(ADAS_INPUT_DIR_ENV),
        ADAS_OUTPUT_DIR_ENV: os.environ.get(ADAS_OUTPUT_DIR_ENV),
    }

    # Set environment variables for this case
    os.environ[ADAS_WORKSPACE_DIR_ENV] = str(case_dir)
    os.environ[ADAS_INPUT_DIR_ENV] = str(input_dir)
    os.environ[ADAS_OUTPUT_DIR_ENV] = str(output_dir)

    workspace_dirs = {
        "workspace": case_dir,
        "input": input_dir,
        "output": output_dir,
    }

    try:
        yield workspace_dirs
    finally:
        # Restore environment variables
        for key, val in old_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

        if clean_up:
            try:
                shutil.rmtree(case_dir, ignore_errors=True)
            except Exception as e_clean:
                logger.debug(f"Failed to clean up case directory {case_dir}: {e_clean}")


def run_preflight_check(
    preflight_path: Path | str,
    workspace_dirs: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Execute preflight.py to verify external environment access before running tests.

    Args:
        preflight_path: Path to the preflight.py script.
        workspace_dirs: Directory mapping passed to check_environment.

    Returns:
        Tuple of (is_pass: bool, message: str).
    """
    path = Path(preflight_path)
    if not path.exists():
        logger.info(f"No preflight script found at {path}; skipping preflight check.")
        return True, "No preflight check defined"

    dirs_dict = {k: str(v) for k, v in (workspace_dirs or {}).items()}

    try:
        spec = importlib.util.spec_from_file_location(f"preflight_{path.stem}", path)
        if not spec or not spec.loader:
            return False, f"Failed to load preflight spec from {path}"

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        check_func = getattr(module, "check_environment", None)
        if not check_func or not callable(check_func):
            return False, "preflight.py must define callable check_environment()."

        result = check_func(dirs_dict)
        if isinstance(result, tuple) and len(result) == 2:
            is_pass, msg = bool(result[0]), str(result[1])
            return is_pass, msg
        elif isinstance(result, bool):
            return result, "Preflight check passed" if result else "Preflight check failed"
        else:
            return False, f"Preflight returned unexpected type: {type(result)}; expected bool or (bool, str)."
    except Exception as e:
        err_msg = f"Preflight check exception: {e!r}"
        logger.error(err_msg)
        return False, err_msg
