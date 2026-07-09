"""Unit tests for the consolidated Python orchestrator engine (scripts/orchestrator.py)."""

import json
import sys
from pathlib import Path
from scripts.orchestrator import (
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
