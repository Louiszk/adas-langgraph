"""Unit tests for sandbox/run_meta.py verifying CLI argument parsing, execution, and exit code propagation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from adas_core.task_spec import TaskSpec
from sandbox import run_meta


@pytest.fixture
def sample_task_spec_file(tmp_path: Path) -> Path:
    spec_path = tmp_path / "task.json"
    spec_data = {
        "name": "MathSolver",
        "system_goal": "Solve mathematical equations accurately.",
        "architecture_contract": {
            "execution_mode": "single_turn",
            "state_schema": {"problem": "str", "solution": "float"},
            "required_tools": [],
        },
        "resource_manifest": {"available_resources": [], "available_api_keys": []},
        "dev_suite": [
            {
                "id": "case_1",
                "description": "Basic addition",
                "turns": [{"problem": "1 + 1"}],
                "expected_outputs": ["solution"],
            }
        ],
    }
    spec = TaskSpec.model_validate(spec_data)
    spec.save(spec_path)
    return spec_path


def test_load_visible_task_spec_sets_allowed_models(sample_task_spec_file: Path):
    spec = run_meta.load_visible_task_spec(sample_task_spec_file)
    assert spec.name == "MathSolver"
    assert spec.system_goal == "Solve mathematical equations accurately."


def test_run_meta_success_with_design_completed(
    tmp_path: Path, sample_task_spec_file: Path, monkeypatch: pytest.MonkeyPatch
):
    sandbox_gen_dir = tmp_path / "generated_systems"
    metrics_dir = sandbox_gen_dir / "metrics"
    sandbox_gen_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    monkeypatch.setattr("sandbox.run_meta.SANDBOX_GENERATED_SYSTEMS_DIR", str(sandbox_gen_dir))

    import dill

    from adas_core.virtual_agentic_system import VirtualAgenticSystem

    mock_workflow = MagicMock()

    def stream_and_write_artifacts(*args, **kwargs):
        (sandbox_gen_dir / "MathSolverAgent.pkl").write_bytes(dill.dumps(VirtualAgenticSystem("MathSolverAgent")))
        (sandbox_gen_dir / "MathSolverAgent.py").write_text("# generated python", encoding="utf-8")
        yield {
            "Finalize": {
                "design_completed": True,
                "finalization_succeeded": True,
                "messages": [],
            }
        }

    mock_workflow.stream.side_effect = stream_and_write_artifacts
    monkeypatch.setattr("sandbox.run_meta.workflow", mock_workflow)

    argv = [
        "run_meta.py",
        "--task-spec",
        str(sample_task_spec_file),
        "--system-name",
        "MathSolverAgent",
        "--max-iterations",
        "5",
    ]
    monkeypatch.setattr("sys.argv", argv)

    # Run main; it should succeed and not raise SystemExit
    run_meta.main()

    metrics_file = metrics_dir / "MathSolverAgent.json"
    assert metrics_file.is_file(), "Metrics file must be written"
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert metrics["status"] == "completed"
    assert metrics["system_name"] == "MathSolverAgent"
    assert "Solve mathematical equations accurately." in metrics["problem_statement"]


def test_run_meta_exits_with_code_1_on_workflow_failure(
    tmp_path: Path, sample_task_spec_file: Path, monkeypatch: pytest.MonkeyPatch
):
    sandbox_gen_dir = tmp_path / "generated_systems"
    metrics_dir = sandbox_gen_dir / "metrics"
    sandbox_gen_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    monkeypatch.setattr("sandbox.run_meta.SANDBOX_GENERATED_SYSTEMS_DIR", str(sandbox_gen_dir))

    mock_workflow = MagicMock()
    mock_workflow.stream.side_effect = RuntimeError("Workflow graph crashed")
    monkeypatch.setattr("sandbox.run_meta.workflow", mock_workflow)

    argv = [
        "run_meta.py",
        "--task-spec",
        str(sample_task_spec_file),
        "--system-name",
        "CrashSystem",
    ]
    monkeypatch.setattr("sys.argv", argv)

    assert run_meta.main() == 1

    metrics_file = metrics_dir / "CrashSystem.json"
    assert metrics_file.is_file()
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert metrics["status"] == "error"
    assert "Workflow graph crashed" in str(metrics["error"])


def test_run_meta_exits_with_code_1_when_incomplete_and_no_artifact(
    tmp_path: Path, sample_task_spec_file: Path, monkeypatch: pytest.MonkeyPatch
):
    sandbox_gen_dir = tmp_path / "generated_systems"
    metrics_dir = sandbox_gen_dir / "metrics"
    sandbox_gen_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    monkeypatch.setattr("sandbox.run_meta.SANDBOX_GENERATED_SYSTEMS_DIR", str(sandbox_gen_dir))

    # Workflow finishes without design_completed and without creating final pickle
    mock_workflow = MagicMock()
    mock_workflow.stream.return_value = [{"ToolExecution": {"messages": []}}]
    monkeypatch.setattr("sandbox.run_meta.workflow", mock_workflow)

    argv = [
        "run_meta.py",
        "--task-spec",
        str(sample_task_spec_file),
        "--system-name",
        "IncompleteSystem",
    ]
    monkeypatch.setattr("sys.argv", argv)

    assert run_meta.main() == 1


def test_run_meta_exits_with_code_1_when_pickle_exists_but_design_incomplete(
    tmp_path: Path, sample_task_spec_file: Path, monkeypatch: pytest.MonkeyPatch
):
    sandbox_gen_dir = tmp_path / "generated_systems"
    metrics_dir = sandbox_gen_dir / "metrics"
    sandbox_gen_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    monkeypatch.setattr("sandbox.run_meta.SANDBOX_GENERATED_SYSTEMS_DIR", str(sandbox_gen_dir))

    # Existing pickle and py from a prior run (e.g. optimization baseline)
    import dill

    from adas_core.virtual_agentic_system import VirtualAgenticSystem

    old_pickle = sandbox_gen_dir / "OptimizedSystem.pkl"
    old_pickle.write_bytes(dill.dumps(VirtualAgenticSystem("OptimizedSystem")))
    old_py = sandbox_gen_dir / "OptimizedSystem.py"
    old_py.write_text("# prior py", encoding="utf-8")

    # Workflow finishes without design_completed
    mock_workflow = MagicMock()
    mock_workflow.stream.return_value = [{"ToolExecution": {"design_completed": False, "messages": []}}]
    monkeypatch.setattr("sandbox.run_meta.workflow", mock_workflow)

    argv = [
        "run_meta.py",
        "--task-spec",
        str(sample_task_spec_file),
        "--system-name",
        "OptimizedSystem",
        "--optimize-system",
        "OptimizedSystem",
    ]
    monkeypatch.setattr("sys.argv", argv)

    assert run_meta.main() == 1
    metrics_file = metrics_dir / "OptimizedSystem.json"
    assert metrics_file.is_file()
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert metrics["status"] == "error"
    assert "Design loop ended without design_completed flag" in metrics["error"]["message"]


def test_run_meta_does_not_accept_stale_same_name_artifacts(
    tmp_path: Path, sample_task_spec_file: Path, monkeypatch: pytest.MonkeyPatch
):
    sandbox_gen_dir = tmp_path / "generated_systems"
    metrics_dir = sandbox_gen_dir / "metrics"
    sandbox_gen_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    monkeypatch.setattr("sandbox.run_meta.SANDBOX_GENERATED_SYSTEMS_DIR", str(sandbox_gen_dir))

    import dill

    from adas_core.virtual_agentic_system import VirtualAgenticSystem

    (sandbox_gen_dir / "StaleSystem.pkl").write_bytes(dill.dumps(VirtualAgenticSystem("StaleSystem")))
    (sandbox_gen_dir / "StaleSystem.py").write_text("# stale artifact", encoding="utf-8")

    mock_workflow = MagicMock()
    mock_workflow.stream.return_value = [{"ToolExecution": {"design_completed": True, "messages": []}}]
    monkeypatch.setattr("sandbox.run_meta.workflow", mock_workflow)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_meta.py",
            "--task-spec",
            str(sample_task_spec_file),
            "--system-name",
            "StaleSystem",
        ],
    )

    assert run_meta.main() == 1
    metrics = json.loads((metrics_dir / "StaleSystem.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "error"


def test_run_meta_rejects_invalid_system_name(
    tmp_path: Path, sample_task_spec_file: Path, monkeypatch: pytest.MonkeyPatch
):
    sandbox_gen_dir = tmp_path / "generated_systems"
    (sandbox_gen_dir / "metrics").mkdir(parents=True)
    monkeypatch.setattr("sandbox.run_meta.SANDBOX_GENERATED_SYSTEMS_DIR", str(sandbox_gen_dir))

    argv = [
        "run_meta.py",
        "--task-spec",
        str(sample_task_spec_file),
        "--system-name",
        "Bad/System/Name",
    ]
    monkeypatch.setattr("sys.argv", argv)

    assert run_meta.main() == 1
    metrics_file = sandbox_gen_dir / "metrics" / "BadSystemName.json"
    assert metrics_file.is_file()
