"""Unit tests for sandbox/run_target.py verifying isolated workspace and ADAS env vars."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from sandbox import run_target


class DummyState(TypedDict):
    messages: list[str]
    result: str


def dummy_node(state: DummyState) -> dict[str, str]:
    input_dir = os.environ.get("ADAS_INPUT_DIR")
    output_dir = os.environ.get("ADAS_OUTPUT_DIR")
    workspace_dir = os.environ.get("ADAS_WORKSPACE_DIR")

    assert input_dir is not None and Path(input_dir).is_dir(), "ADAS_INPUT_DIR must be a valid directory"
    assert output_dir is not None and Path(output_dir).is_dir(), "ADAS_OUTPUT_DIR must be a valid directory"
    assert workspace_dir is not None and Path(workspace_dir).is_dir(), "ADAS_WORKSPACE_DIR must be a valid directory"

    # Check if a seeded fixture is visible
    fixture_file = Path(input_dir) / "test_fixture.txt"
    if fixture_file.is_file():
        content = fixture_file.read_text(encoding="utf-8")
        assert content == "fixture_data"

    # Write output artifact beneath ADAS_OUTPUT_DIR
    out_file = Path(output_dir) / "analysis_result.txt"
    out_file.write_text("analysis_complete", encoding="utf-8")

    return {"result": "ok"}


@pytest.fixture(autouse=True)
def register_dummy_target_system():
    builder = StateGraph(DummyState)
    builder.add_node("dummy", dummy_node)
    builder.add_edge(START, "dummy")
    builder.add_edge("dummy", END)
    compiled = builder.compile()

    mod_name = "generated_systems.DummyTargetSystem"
    dummy_mod = types.ModuleType(mod_name)
    dummy_mod.workflow = compiled  # type: ignore[attr-defined]

    sys.modules[mod_name] = dummy_mod
    yield
    sys.modules.pop(mod_name, None)


def test_run_target_executes_in_isolated_workspace_with_fixtures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    task_dir = tmp_path / "task_setup"
    fixtures_dir = task_dir / "fixtures"
    fixtures_dir.mkdir(parents=True)
    (fixtures_dir / "test_fixture.txt").write_text("fixture_data", encoding="utf-8")

    sandbox_root = tmp_path / "sandbox_workspace"
    metrics_dir = sandbox_root / "target_metrics"
    target_runs_dir = sandbox_root / "target_runs"

    metrics_dir.mkdir(parents=True)
    target_runs_dir.mkdir(parents=True)

    monkeypatch.setattr("sandbox.run_target.SANDBOX_WORKSPACE_DIR", str(sandbox_root))
    monkeypatch.setattr("sandbox.run_target.SANDBOX_TARGET_METRICS_DIR", str(metrics_dir))

    run_id = "test_run_42"
    argv = [
        "run_target.py",
        "--system_name",
        "DummyTargetSystem",
        "--state",
        json.dumps({"messages": ["start"]}),
        "--run-id",
        run_id,
        "--task-dir",
        str(task_dir),
    ]

    monkeypatch.setattr("sys.argv", argv)

    run_target.main()

    # Verify metrics were written and contain workspace info
    metrics_file = metrics_dir / f"DummyTargetSystem_{run_id}.json"
    assert metrics_file.is_file()
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert metrics["status"] == "completed"
    assert metrics["run_id"] == run_id
    assert "workspace" in metrics
    assert "input_dir" in metrics
    assert "output_dir" in metrics

    # Verify output was created in the isolated workspace
    isolated_output = Path(metrics["output_dir"]) / "analysis_result.txt"
    assert isolated_output.is_file()
    assert isolated_output.read_text(encoding="utf-8") == "analysis_complete"


def test_run_target_without_task_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sandbox_root = tmp_path / "sandbox_workspace"
    metrics_dir = sandbox_root / "target_metrics"

    metrics_dir.mkdir(parents=True)

    monkeypatch.setattr("sandbox.run_target.SANDBOX_WORKSPACE_DIR", str(sandbox_root))
    monkeypatch.setattr("sandbox.run_target.SANDBOX_TARGET_METRICS_DIR", str(metrics_dir))

    run_id = "test_run_no_task_dir"
    argv = [
        "run_target.py",
        "--system_name",
        "DummyTargetSystem",
        "--state",
        json.dumps({"messages": ["hi"]}),
        "--run-id",
        run_id,
    ]

    monkeypatch.setattr("sys.argv", argv)

    run_target.main()

    metrics_file = metrics_dir / f"DummyTargetSystem_{run_id}.json"
    assert metrics_file.is_file()
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert metrics["status"] == "completed"


def test_run_target_returns_nonzero_when_workflow_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fail_node(state: DummyState) -> dict[str, str]:
        raise RuntimeError("intentional failure")

    builder = StateGraph(DummyState)
    builder.add_node("fail", fail_node)
    builder.add_edge(START, "fail")
    builder.add_edge("fail", END)
    failing_mod = types.ModuleType("generated_systems.FailingTargetSystem")
    failing_mod.workflow = builder.compile()  # type: ignore[attr-defined]
    sys.modules["generated_systems.FailingTargetSystem"] = failing_mod

    sandbox_root = tmp_path / "sandbox_workspace"
    metrics_dir = sandbox_root / "target_metrics"
    metrics_dir.mkdir(parents=True)
    monkeypatch.setattr("sandbox.run_target.SANDBOX_WORKSPACE_DIR", str(sandbox_root))
    monkeypatch.setattr("sandbox.run_target.SANDBOX_TARGET_METRICS_DIR", str(metrics_dir))
    monkeypatch.setattr(
        "sys.argv",
        ["run_target.py", "--system_name", "FailingTargetSystem", "--state", '{"messages": []}', "--run-id", "failed"],
    )

    assert run_target.main() == 1
    metrics = json.loads((metrics_dir / "FailingTargetSystem_failed.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "error"
    sys.modules.pop("generated_systems.FailingTargetSystem", None)
