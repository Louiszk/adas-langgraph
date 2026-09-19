"""Unit tests for the consolidated Python orchestrator engine (scripts/orchestrator.py)."""

import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from scripts.orchestrator import (
    ContainerManager,
    DependencyParser,
    ExecutionManager,
    Orchestrator,
    ResultAggregator,
    parse_args,
)


def test_dependency_parser_list_format(tmp_path: Path):
    metrics_file = tmp_path / "metrics_list.json"
    metrics_file.write_text(json.dumps({"installed_packages": ["pandas==2.0.0", "numpy"]}), encoding="utf-8")
    pkgs = DependencyParser.get_installed_packages(metrics_file)
    assert pkgs == ["pandas==2.0.0", "numpy"]


def test_dependency_parser_string_format(tmp_path: Path):
    metrics_file = tmp_path / "metrics_str.json"
    metrics_file.write_text(json.dumps({"installed_packages": "pandas==2.0.0   scipy "}), encoding="utf-8")
    pkgs = DependencyParser.get_installed_packages(metrics_file)
    assert pkgs == ["pandas==2.0.0", "scipy"]


def test_dependency_parser_missing_or_empty(tmp_path: Path):
    assert DependencyParser.get_installed_packages(tmp_path / "nonexistent.json") == []
    metrics_file = tmp_path / "metrics_empty.json"
    metrics_file.write_text(json.dumps({"other_field": 123}), encoding="utf-8")
    assert DependencyParser.get_installed_packages(metrics_file) == []


def test_orchestrator_parse_iterations():
    assert Orchestrator._parse_iterations("1-4") == ["1", "2", "3", "4"]
    assert Orchestrator._parse_iterations("1, 3 , 5") == ["1", "3", "5"]
    assert Orchestrator._parse_iterations("7") == ["7"]


def test_orchestrator_caches_the_persisted_core_image_name():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.container_manager = cast(ContainerManager, SimpleNamespace(client="container-client"))
    orchestrator._cached_sandbox_image_name = None

    with patch("scripts.orchestrator.ensure_cached_sandbox_image", return_value="adas-sandbox:core") as ensure_image:
        assert orchestrator._cached_sandbox_image() == "adas-sandbox:core"
        assert orchestrator._cached_sandbox_image() == "adas-sandbox:core"

    ensure_image.assert_called_once_with(client="container-client")


def test_result_aggregator_export(tmp_path: Path):
    aggregator = ResultAggregator(tmp_path / "results")
    records = [
        {"approach": "ablationC", "benchmark": "mmlu", "iteration": "1", "status": "WORKING"},
        {"approach": "ablationC", "benchmark": "mmlu", "iteration": "2", "status": "TIMEOUT"},
    ]
    csv_file = aggregator.export_summary_csv(records, "test.csv")
    txt_file = aggregator.export_summary_txt("job123", records, "test.txt")

    assert csv_file.is_file()
    assert txt_file.is_file()
    csv_content = csv_file.read_text(encoding="utf-8")
    assert "ablationC,mmlu,1,WORKING" in csv_content
    txt_content = txt_file.read_text(encoding="utf-8")
    assert "ablationC_mmlu1_gpt: WORKING" in txt_content


def test_execution_manager_run_command(tmp_path: Path):
    log_file = tmp_path / "cmd_log.txt"
    cmd = [sys.executable, "-c", "print('hello world')"]
    res = ExecutionManager.run_command(cmd, timeout=5, log_file=log_file)
    assert res["exit_code"] == 0
    assert res["status"] == "WORKING"
    assert "hello world" in res["stdout"]
    assert log_file.is_file()


def test_parse_args_task_flag():
    args = parse_args(["--task", "benchmark", "--benchmark", "mmlu", "--iterations", "1-16"])
    assert args.task == "benchmark"
    assert args.benchmark == "mmlu"
    assert args.iterations == "1-16"


def test_parse_args_positional_task():
    args = parse_args(["design", "--benchmark", "gsm", "--iterations", "1-3"])
    assert args.task == "design"
    assert args.benchmark == "gsm"
    assert args.iterations == "1-3"


def test_parse_args_task_spec():
    args = parse_args(
        [
            "--task",
            "design",
            "--task-spec",
            "specs/my_task/task.json",
        ]
    )
    assert args.task == "design"
    assert args.task_spec == "specs/my_task/task.json"


def test_container_manager_cleanup_is_scoped_to_tracked_job_resources():
    class FakeContainers:
        def __init__(self):
            self.removed = []

        def get(self, name):
            return SimpleNamespace(remove=lambda force: self.removed.append((name, force)))

    class FakeImages:
        def __init__(self):
            self.removed = []

        def remove(self, image, force):
            self.removed.append((image, force))

    containers = FakeContainers()
    images = FakeImages()
    manager = ContainerManager.__new__(ContainerManager)
    manager.unique_id = "job-123"
    manager.client = SimpleNamespace(containers=containers, images=images)
    manager._created_container_names = {"adas-temp-builder-job-123-1"}
    manager._created_image_names = {"adas-temp-image-job-123"}

    manager.cleanup_job_resources()

    assert containers.removed == [("adas-temp-builder-job-123-1", True)]
    assert images.removed == [("adas-temp-image-job-123", True)]
    assert manager._created_container_names == set()
    assert manager._created_image_names == set()


def test_container_manager_untracks_successfully_removed_image():
    images = SimpleNamespace(remove=lambda image, force: None)
    manager = ContainerManager.__new__(ContainerManager)
    manager.client = SimpleNamespace(images=images)
    manager.container_type = "docker"
    manager._created_image_names = {"adas-temp-image-job-123"}

    manager.remove_image("adas-temp-image-job-123")

    assert manager._created_image_names == set()


def test_orchestrator_marks_invalid_dependency_metadata_as_failed(tmp_path: Path, monkeypatch):
    work_dir = tmp_path / "work"
    metrics_dir = work_dir / "generated_systems" / "metrics"
    metrics_dir.mkdir(parents=True)
    (work_dir / "generated_systems" / "ablationC_gsm1_gpt.py").write_text("", encoding="utf-8")
    (metrics_dir / "ablationC_gsm1_gpt.json").write_text(
        json.dumps({"installed_packages": ["not a valid requirement;"]}), encoding="utf-8"
    )
    results_dir = tmp_path / "results"
    args = Namespace(
        container="docker",
        iterations="1",
        benchmark="gsm",
        type="ablationC",
        cleanup_frequency=8,
        timeout=1200,
        results_dir=str(results_dir),
    )
    monkeypatch.chdir(work_dir)
    orchestrator = Orchestrator(args)

    with patch.object(DependencyParser, "get_installed_packages", side_effect=ValueError("invalid requirement")):
        assert orchestrator.run_benchmark() == 1

    summary = (results_dir / "results_summary.csv").read_text(encoding="utf-8")
    assert "FAILED_INVALID_DEPENDENCIES" in summary


def test_orchestrator_run_design_command_construction(tmp_path: Path):
    dummy_spec = tmp_path / "task.json"
    dummy_spec.write_text("{}", encoding="utf-8")

    args = parse_args(
        [
            "--task",
            "design",
            "--task-spec",
            str(dummy_spec),
            "--benchmark",
            "gsm",
            "--type",
            "ablationC",
            "--iterations",
            "1",
        ]
    )
    orchestrator = Orchestrator(args)

    with patch.object(ExecutionManager, "run_command", return_value={"exit_code": 0}) as mock_run:
        exit_code = orchestrator.run_design()
        assert exit_code == 0
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[1] == "invoke_design.py"
        assert "--task-spec" in cmd
        assert str(dummy_spec.resolve()) in cmd
        assert "--system-name" in cmd
        assert "ablationC_gsm1_gpt" in cmd


def test_orchestrator_uses_task_directory_name_for_canonical_task_spec(tmp_path: Path):
    task_dir = tmp_path / "my_agent"
    task_dir.mkdir()
    dummy_spec = task_dir / "task.json"
    dummy_spec.write_text("{}", encoding="utf-8")

    args = parse_args(
        [
            "--task",
            "design",
            "--task-spec",
            str(dummy_spec),
            "--type",
            "ablationC",
            "--iterations",
            "1",
        ]
    )
    orchestrator = Orchestrator(args)

    with patch.object(ExecutionManager, "run_command", return_value={"exit_code": 0}) as mock_run:
        assert orchestrator.run_design() == 0
        cmd = mock_run.call_args[0][0]
        assert "ablationC_my_agent1_gpt" in cmd


def test_orchestrator_sanitizes_hyphenated_task_spec_directory(tmp_path: Path):
    task_dir = tmp_path / "my-agent"
    task_dir.mkdir()
    dummy_spec = task_dir / "task.json"
    dummy_spec.write_text("{}", encoding="utf-8")

    args = parse_args(
        [
            "--task",
            "design",
            "--task-spec",
            str(dummy_spec),
            "--type",
            "ablationC",
            "--iterations",
            "1",
        ]
    )
    orchestrator = Orchestrator(args)

    with patch.object(ExecutionManager, "run_command", return_value={"exit_code": 0}) as mock_run:
        assert orchestrator.run_design() == 0
        cmd = mock_run.call_args[0][0]
        assert "ablationC_my_agent1_gpt" in cmd


def test_orchestrator_run_target_command_construction(tmp_path: Path):
    dummy_spec = tmp_path / "task.json"
    dummy_spec.write_text("{}", encoding="utf-8")

    args = parse_args(
        [
            "--task",
            "target",
            "--task-spec",
            str(dummy_spec),
            "--system-names",
            "test_sys",
            "--state",
            '{"messages": ["hi"]}',
        ]
    )
    orchestrator = Orchestrator(args)

    with patch.object(ExecutionManager, "run_command", return_value={"exit_code": 0}) as mock_run:
        exit_code = orchestrator.run_target()
        assert exit_code == 0
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[1] == "invoke_target.py"
        assert "--system_name" in cmd
        assert "test_sys" in cmd
        assert "--task-spec" in cmd
        assert str(dummy_spec.resolve()) in cmd


def test_orchestrator_run_target_forwards_batch_file(tmp_path: Path):
    dummy_spec = tmp_path / "task.json"
    dummy_spec.write_text("{}", encoding="utf-8")
    batch_file = tmp_path / "batch.json"
    batch_file.write_text("[]", encoding="utf-8")

    args = parse_args(
        [
            "--task",
            "target",
            "--task-spec",
            str(dummy_spec),
            "--system-names",
            "test_sys",
            "--batch",
            str(batch_file),
            "--state",
            '{"ignored": true}',
        ]
    )
    orchestrator = Orchestrator(args)

    with patch.object(ExecutionManager, "run_command", return_value={"exit_code": 0}) as mock_run:
        assert orchestrator.run_target() == 0

    cmd = mock_run.call_args[0][0]
    assert "--batch" in cmd
    assert str(batch_file.resolve()) in cmd
    assert "--state" not in cmd


def test_orchestrator_run_target_with_state_file(tmp_path: Path):
    dummy_spec = tmp_path / "task.json"
    dummy_spec.write_text("{}", encoding="utf-8")
    dummy_state = tmp_path / "state.json"
    dummy_state.write_text('{"messages": ["hi"]}', encoding="utf-8")

    args = parse_args(
        [
            "--task",
            "target",
            "--task-spec",
            str(dummy_spec),
            "--system-names",
            "test_sys",
            "--state-file",
            str(dummy_state),
        ]
    )
    orchestrator = Orchestrator(args)

    with patch.object(ExecutionManager, "run_command", return_value={"exit_code": 0}) as mock_run:
        exit_code = orchestrator.run_target()
        assert exit_code == 0
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[1] == "invoke_target.py"
        assert "--system_name" in cmd
        assert "test_sys" in cmd
        assert "--state-file" in cmd
        assert str(dummy_state.resolve()) in cmd
