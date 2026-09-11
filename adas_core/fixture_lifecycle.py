"""Lifecycle management for process-backed test fixtures."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from adas_core.task_spec import MCPFixtureSpec, MockServiceFixtureSpec, TestFixturesSpec

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


def _wait_for_port(process: subprocess.Popen[bytes], port: int, fixture_id: str, stdout: Path, stderr: Path) -> None:
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise FixtureStartupError(
                _failure_message(fixture_id, f"process exited with code {exit_code}", stdout, stderr)
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise FixtureStartupError(
        _failure_message(fixture_id, f"port 127.0.0.1:{port} did not become ready", stdout, stderr)
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
