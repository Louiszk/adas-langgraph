import os
import socket
import time
from pathlib import Path

import pytest

from adas_core.fixture_lifecycle import FixtureStartupError, _script_path, _wait_for_port, process_fixture_lifecycle
from adas_core.task_spec import MCPFixtureSpec, MockServiceFixtureSpec, TestFixturesSpec


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _workspace(tmp_path: Path) -> dict[str, Path]:
    workspace = tmp_path / "case"
    input_dir, output_dir = workspace / "input", workspace / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir()
    return {"workspace": workspace, "input": input_dir, "output": output_dir}


def _server_script(port: int) -> str:
    return (
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.end_headers()\n"
        "    def log_message(self, *args): pass\n"
        f"HTTPServer(('127.0.0.1', {port}), Handler).serve_forever()\n"
    )


def _mcp_endpoint_script(port: int) -> str:
    return (
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_OPTIONS(self):\n"
        "        self.send_response(405 if self.path == '/mcp' else 404); self.end_headers()\n"
        "    def do_POST(self):\n"
        "        self.send_response(200 if self.path == '/mcp' else 404); self.end_headers()\n"
        "    def log_message(self, *args): pass\n"
        f"HTTPServer(('127.0.0.1', {port}), Handler).serve_forever()\n"
    )


def test_process_fixture_lifecycle_scopes_url_and_restores_environment(tmp_path, monkeypatch):
    port = _free_port()
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "mock_weather.py").write_text(_server_script(port), encoding="utf-8")
    fixture = MockServiceFixtureSpec(name="weather", port=port, base_url_env="WEATHER_BASE_URL")
    spec = TestFixturesSpec(mock_services=[fixture])
    monkeypatch.setenv("WEATHER_BASE_URL", "previous-value")

    with process_fixture_lifecycle(spec, ["weather"], tmp_path, _workspace(tmp_path)):
        assert os.environ["WEATHER_BASE_URL"] == f"http://127.0.0.1:{port}"
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass

    assert os.environ["WEATHER_BASE_URL"] == "previous-value"
    deadline = time.monotonic() + 2
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                pass
        except OSError:
            break
        if time.monotonic() >= deadline:
            pytest.fail("fixture process still accepts connections after lifecycle cleanup")
        time.sleep(0.05)


def test_process_fixture_lifecycle_reports_crashing_script_output(tmp_path):
    port = _free_port()
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "mock_broken.py").write_text(
        "import sys\nprint('safe diagnostic')\nraise SystemExit(7)\n", encoding="utf-8"
    )
    spec = TestFixturesSpec(mock_services=[MockServiceFixtureSpec(name="broken", port=port)])

    with pytest.raises(FixtureStartupError, match="safe diagnostic"):
        with process_fixture_lifecycle(spec, None, tmp_path, _workspace(tmp_path)):
            pass


def test_process_fixture_lifecycle_rejects_an_already_occupied_port(tmp_path, monkeypatch):
    port = _free_port()
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "mock_weather.py").write_text("raise AssertionError('must not launch')\n", encoding="utf-8")
    fixture = MockServiceFixtureSpec(name="weather", port=port, base_url_env="WEATHER_BASE_URL")
    listener = socket.socket()
    listener.bind(("127.0.0.1", port))
    listener.listen()
    monkeypatch.delenv("WEATHER_BASE_URL", raising=False)
    try:
        with pytest.raises(FixtureStartupError, match="already occupied before launch"):
            with process_fixture_lifecycle(
                TestFixturesSpec(mock_services=[fixture]), ["weather"], tmp_path, _workspace(tmp_path)
            ):
                pass
    finally:
        listener.close()
    assert "WEATHER_BASE_URL" not in os.environ


def test_mcp_endpoint_readiness_accepts_post_only_route_and_rejects_wrong_path(tmp_path, monkeypatch):
    port = _free_port()
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "mock_tools.py").write_text(_mcp_endpoint_script(port), encoding="utf-8")
    fixture = MCPFixtureSpec(name="tools", port=port, endpoint_path="/mcp", url_env="TOOLS_MCP_URL")
    monkeypatch.delenv("TOOLS_MCP_URL", raising=False)

    with process_fixture_lifecycle(TestFixturesSpec(mcps=[fixture]), ["tools"], tmp_path, _workspace(tmp_path)):
        assert os.environ["TOOLS_MCP_URL"] == f"http://127.0.0.1:{port}/mcp"

    wrong_port = _free_port()
    (fixtures_dir / "mock_wrong_tools.py").write_text(_mcp_endpoint_script(wrong_port), encoding="utf-8")
    wrong_fixture = MCPFixtureSpec(name="wrong_tools", port=wrong_port, endpoint_path="/wrong", url_env="WRONG_MCP_URL")
    with pytest.raises(FixtureStartupError, match="returned HTTP 404"):
        with process_fixture_lifecycle(
            TestFixturesSpec(mcps=[wrong_fixture]), ["wrong_tools"], tmp_path, _workspace(tmp_path / "wrong")
        ):
            pass
    deadline = time.monotonic() + 2
    while True:
        try:
            with socket.create_connection(("127.0.0.1", wrong_port), timeout=0.1):
                pass
        except OSError:
            break
        if time.monotonic() >= deadline:
            pytest.fail("MCP fixture process still accepts connections after failed readiness")
        time.sleep(0.05)


def test_wait_for_port_detects_child_death_after_connection(tmp_path, monkeypatch):
    class Process:
        def __init__(self):
            self.calls = 0

        def poll(self):
            self.calls += 1
            return None if self.calls == 1 else 9

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: Connection())
    stdout, stderr = tmp_path / "stdout.log", tmp_path / "stderr.log"

    with pytest.raises(FixtureStartupError, match="exited with code 9"):
        _wait_for_port(Process(), 9999, "dying", stdout, stderr)  # type: ignore[arg-type]


def test_script_path_accepts_a_fixtures_directory_directly(tmp_path):
    script = tmp_path / "mock_weather.py"
    script.write_text("# fixture", encoding="utf-8")
    fixture = MockServiceFixtureSpec(name="weather", port=_free_port())
    assert _script_path(tmp_path, fixture) == script
