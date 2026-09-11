import os
import socket
import time
from pathlib import Path

import pytest

from adas_core.fixture_lifecycle import FixtureStartupError, _script_path, process_fixture_lifecycle
from adas_core.task_spec import MockServiceFixtureSpec, TestFixturesSpec


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


def test_script_path_accepts_a_fixtures_directory_directly(tmp_path):
    script = tmp_path / "mock_weather.py"
    script.write_text("# fixture", encoding="utf-8")
    fixture = MockServiceFixtureSpec(name="weather", port=_free_port())
    assert _script_path(tmp_path, fixture) == script
