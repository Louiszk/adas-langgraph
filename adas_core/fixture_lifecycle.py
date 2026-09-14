"""Lifecycle management for process-backed test fixtures."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

from adas_core.exceptions import FixtureExecutionError
from adas_core.task_spec import (
    ExternalDatabaseSeedSpec,
    MCPFixtureSpec,
    MockServiceFixtureSpec,
    TaskSpec,
    TestFixturesSpec,
)

FIXTURE_LIFECYCLE_VERSION = 1
_READY_TIMEOUT_SECONDS = 10.0
_TERMINATE_TIMEOUT_SECONDS = 2.0
_LOG_TAIL_BYTES = 4096


class FixtureStartupError(RuntimeError):
    """Raised when a declared process fixture cannot be started safely."""


def _script_path(setup_dir: Path, fixture: MCPFixtureSpec | MockServiceFixtureSpec) -> Path:
    """Locate the deterministic generated script, retaining a legacy fallback."""
    filename = f"mock_{fixture.name}.py"
    for directory in (setup_dir / "fixtures", setup_dir / "setup_scripts", setup_dir):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    raise FixtureStartupError(f"Fixture '{fixture.id}' script is missing: fixtures/{filename}")


def _tail(path: Path) -> str:
    try:
        data = path.read_bytes()[-_LOG_TAIL_BYTES:]
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace").strip()


def _failure_message(fixture_id: str, reason: str, stdout: Path, stderr: Path) -> str:
    sections = [f"Fixture '{fixture_id}' failed to start: {reason}."]
    if text := _tail(stdout):
        sections.append(f"stdout tail:\n{text}")
    if text := _tail(stderr):
        sections.append(f"stderr tail:\n{text}")
    return "\n".join(sections)


def _port_is_occupied(port: int) -> bool:
    """Return whether another process is already listening on the fixture port.

    This narrows the readiness race substantially, although it cannot reserve a
    TCP port between this check and the child process binding it.
    """
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def _raise_if_exited(process: subprocess.Popen[bytes], fixture_id: str, stdout: Path, stderr: Path) -> None:
    if (exit_code := process.poll()) is not None:
        raise FixtureStartupError(_failure_message(fixture_id, f"process exited with code {exit_code}", stdout, stderr))


def _wait_for_port(process: subprocess.Popen[bytes], port: int, fixture_id: str, stdout: Path, stderr: Path) -> None:
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        _raise_if_exited(process, fixture_id, stdout, stderr)
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                # A listener is not enough by itself: make sure it is still our
                # child rather than a process that died during the connect race.
                _raise_if_exited(process, fixture_id, stdout, stderr)
                return
        except OSError:
            time.sleep(0.05)
    raise FixtureStartupError(
        _failure_message(fixture_id, f"port 127.0.0.1:{port} did not become ready", stdout, stderr)
    )


def _wait_for_mcp_endpoint(
    process: subprocess.Popen[bytes], fixture: MCPFixtureSpec, stdout: Path, stderr: Path
) -> None:
    """Verify that the declared Streamable HTTP endpoint exists without using it.

    OPTIONS is intentionally non-destructive. MCP endpoints can be
    POST-only, so 405 is a valid proof that the configured path exists.
    A 404 is definitive evidence that the TaskSpec endpoint path is wrong.
    """
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        _raise_if_exited(process, fixture.id, stdout, stderr)
        connection = HTTPConnection("127.0.0.1", fixture.port, timeout=0.2)
        try:
            connection.request("OPTIONS", fixture.endpoint_path)
            status = connection.getresponse().status
        except OSError:
            time.sleep(0.05)
            continue
        finally:
            connection.close()

        if status == 404:
            raise FixtureStartupError(
                _failure_message(
                    fixture.id,
                    f"MCP endpoint {fixture.endpoint_path!r} returned HTTP 404",
                    stdout,
                    stderr,
                )
            )
        if 200 <= status < 500:
            _raise_if_exited(process, fixture.id, stdout, stderr)
            return
        time.sleep(0.05)
    raise FixtureStartupError(
        _failure_message(
            fixture.id,
            f"MCP endpoint {fixture.endpoint_path!r} did not become ready",
            stdout,
            stderr,
        )
    )


@contextmanager
def process_fixture_lifecycle(
    test_fixtures: TestFixturesSpec,
    fixture_ids: list[str] | None,
    setup_dir: Path | str,
    workspace_dirs: dict[str, Path],
) -> Iterator[None]:
    """Start selected process fixtures and expose their URLs for the context duration."""
    selected = test_fixtures.get_process_fixtures_for_fixture_ids(fixture_ids)
    setup_root = Path(setup_dir)
    log_dir = Path(workspace_dirs["workspace"]) / "fixture_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[bytes]] = []
    handles = []
    previous_env: dict[str, str | None] = {}
    try:
        for fixture in selected:
            stdout = log_dir / f"{fixture.id}.stdout.log"
            stderr = log_dir / f"{fixture.id}.stderr.log"
            script = _script_path(setup_root, fixture)
            if _port_is_occupied(fixture.port):
                raise FixtureStartupError(
                    _failure_message(
                        fixture.id,
                        f"port 127.0.0.1:{fixture.port} is already occupied before launch",
                        stdout,
                        stderr,
                    )
                )
            child_env = os.environ.copy()
            child_env.update(
                {
                    "ADAS_WORKSPACE_DIR": str(workspace_dirs["workspace"]),
                    "ADAS_INPUT_DIR": str(workspace_dirs["input"]),
                    "ADAS_OUTPUT_DIR": str(workspace_dirs["output"]),
                }
            )
            out_handle = stdout.open("wb")
            err_handle = stderr.open("wb")
            handles.extend((out_handle, err_handle))
            process = subprocess.Popen(
                [sys.executable, "-u", str(script)],
                cwd=str(script.parent),
                env=child_env,
                stdout=out_handle,
                stderr=err_handle,
            )
            processes.append(process)
            _wait_for_port(process, fixture.port, fixture.id, stdout, stderr)
            if isinstance(fixture, MCPFixtureSpec):
                _wait_for_mcp_endpoint(process, fixture, stdout, stderr)
                env_name, url = fixture.url_env, f"http://127.0.0.1:{fixture.port}{fixture.endpoint_path}"
            else:
                env_name, url = fixture.base_url_env, f"http://127.0.0.1:{fixture.port}"
            if env_name not in previous_env:
                previous_env[env_name] = os.environ.get(env_name)
            os.environ[env_name] = url
        yield
    finally:
        for process in reversed(processes):
            try:
                running = process.poll() is None
            except OSError:
                running = False
            if running:
                try:
                    process.terminate()
                    try:
                        process.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                except OSError:
                    pass
        for handle in handles:
            try:
                handle.close()
            except OSError:
                pass
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _seed_script_path(setup_dir: Path, seed: ExternalDatabaseSeedSpec) -> Path:
    """Find a generated external-seed script without treating fixture data as code."""
    filename = f"seed_external_{seed.name}.py"
    candidates = (
        setup_dir / "setup_scripts" / filename,
        setup_dir.parent / "setup_scripts" / filename,
        setup_dir / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FixtureExecutionError(f"External database seed '{seed.id}' is missing its generated seed script.")


def _connection_config(seed: ExternalDatabaseSeedSpec) -> dict[str, str]:
    """Read required connection settings while keeping secret values out of errors."""
    config: dict[str, str] = {}
    missing: list[str] = []
    for parameter, env_var in seed.connection_env.items():
        value = os.environ.get(env_var)
        if not value:
            missing.append(env_var)
        else:
            config[parameter] = value
    if missing:
        raise FixtureExecutionError(
            f"External database seed '{seed.id}' is missing required connection environment variable(s): "
            f"{', '.join(sorted(missing))}."
        )
    return config


def _load_seed_functions(script: Path, seed: ExternalDatabaseSeedSpec) -> tuple[Any, Any]:
    namespace: dict[str, Any] = {"__name__": f"adas_external_seed_{seed.id}", "__file__": str(script)}
    try:
        exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), namespace)
    except Exception as exc:
        raise FixtureExecutionError(f"External database seed '{seed.id}' could not load its generated script.") from exc

    seed_fn = namespace.get("seed_external_database")
    cleanup_fn = namespace.get("cleanup_external_database")
    if not callable(seed_fn) or not callable(cleanup_fn):
        raise FixtureExecutionError(
            f"External database seed '{seed.id}' must define both "
            "seed_external_database(connection_config, namespace) and "
            "cleanup_external_database(connection_config, namespace)."
        )
    return seed_fn, cleanup_fn


def _validate_selected_seed_environment_contract(
    task_spec: TaskSpec,
    fixture_ids: list[str] | None,
    selected_seeds: list[ExternalDatabaseSeedSpec],
) -> None:
    """Reject environment collisions before any seed can read or alter configuration."""
    connection_owners: dict[str, list[str]] = {}
    for seed in selected_seeds:
        for env_name in seed.connection_env.values():
            connection_owners.setdefault(env_name, []).append(seed.id)

    if len(selected_seeds) > 1:
        missing_namespace_env = [seed.id for seed in selected_seeds if not seed.namespace_env]
        if missing_namespace_env:
            raise FixtureExecutionError(
                "Multiple external database seeds require an explicit namespace_env for every selected seed. "
                f"Missing for: {', '.join(missing_namespace_env)}."
            )
        namespace_envs: dict[str, str] = {}
        for seed in selected_seeds:
            if existing_seed_id := namespace_envs.get(seed.namespace_env):
                raise FixtureExecutionError(
                    f"External database seeds '{existing_seed_id}' and '{seed.id}' cannot run together because both "
                    f"set namespace environment variable '{seed.namespace_env}'."
                )
            namespace_envs[seed.namespace_env] = seed.id

    if "ADAS_TEST_NAMESPACE" in connection_owners:
        owners = ", ".join(connection_owners["ADAS_TEST_NAMESPACE"])
        raise FixtureExecutionError(
            "ADAS_TEST_NAMESPACE is reserved for the active external database namespace and cannot be used "
            f"as a connection environment variable (seed(s): {owners})."
        )

    namespace_exports: dict[str, list[str]] = {}
    if len(selected_seeds) == 1:
        namespace_exports["ADAS_TEST_NAMESPACE"] = [selected_seeds[0].id]
    for seed in selected_seeds:
        if seed.namespace_env:
            namespace_exports.setdefault(seed.namespace_env, []).append(seed.id)

    selected_process_fixtures = task_spec.test_fixtures.get_process_fixtures_for_fixture_ids(fixture_ids)
    process_envs = {
        fixture.url_env if isinstance(fixture, MCPFixtureSpec) else fixture.base_url_env
        for fixture in selected_process_fixtures
    }
    for env_name, seed_ids in namespace_exports.items():
        if connection_seed_ids := connection_owners.get(env_name):
            raise FixtureExecutionError(
                f"Namespace environment variable '{env_name}' exported by seed(s) {', '.join(seed_ids)} "
                f"collides with connection configuration for seed(s) {', '.join(connection_seed_ids)}."
            )
        if env_name in process_envs:
            raise FixtureExecutionError(
                f"Namespace environment variable '{env_name}' exported by seed(s) {', '.join(seed_ids)} "
                "collides with a selected process-fixture URL environment variable."
            )


@contextmanager
def external_database_seed_lifecycle(
    task_spec: TaskSpec,
    fixture_ids: list[str] | None,
    setup_dir: str | Path,
) -> Iterator[None]:
    """Seed selected isolated external namespaces and reliably drop them afterwards.

    Credentials are passed only as function arguments to the generated script.
    Errors deliberately identify the seed and configuration names, never values.
    """
    prepared: list[tuple[ExternalDatabaseSeedSpec, dict[str, str], Any]] = []
    primary_error: BaseException | None = None
    previous_env: dict[str, str | None] = {}
    selected_seeds = task_spec.test_fixtures.get_external_database_seeds_for_fixture_ids(fixture_ids)
    _validate_selected_seed_environment_contract(task_spec, fixture_ids, selected_seeds)
    try:
        if len(selected_seeds) > 1:
            previous_env["ADAS_TEST_NAMESPACE"] = os.environ.get("ADAS_TEST_NAMESPACE")
            os.environ.pop("ADAS_TEST_NAMESPACE", None)
        for seed in selected_seeds:
            config = _connection_config(seed)
            seed_fn, cleanup_fn = _load_seed_functions(_seed_script_path(Path(setup_dir), seed), seed)
            # Register before mutation: a seed function may partially populate a namespace before failing.
            prepared.append((seed, config, cleanup_fn))
            try:
                seed_fn(config, seed.namespace)
            except Exception as exc:
                raise FixtureExecutionError(
                    f"External database seed '{seed.id}' failed during setup ({type(exc).__name__})."
                ) from exc
            if len(selected_seeds) == 1 and "ADAS_TEST_NAMESPACE" not in previous_env:
                previous_env["ADAS_TEST_NAMESPACE"] = os.environ.get("ADAS_TEST_NAMESPACE")
                os.environ["ADAS_TEST_NAMESPACE"] = seed.namespace
            if seed.namespace_env:
                if seed.namespace_env not in previous_env:
                    previous_env[seed.namespace_env] = os.environ.get(seed.namespace_env)
                os.environ[seed.namespace_env] = seed.namespace
        yield
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        cleanup_failures: list[str] = []
        for seed, config, cleanup_fn in reversed(prepared):
            try:
                cleanup_fn(config, seed.namespace)
            except Exception as cleanup_exc:
                cleanup_failures.append(f"{seed.id} ({type(cleanup_exc).__name__})")
        if cleanup_failures and primary_error is None:
            raise FixtureExecutionError(
                "External database cleanup failed for seed(s): " + ", ".join(cleanup_failures) + "."
            )
        if cleanup_failures and primary_error is not None:
            primary_error.add_note(
                "External database cleanup also failed for seed(s): " + ", ".join(cleanup_failures) + "."
            )
