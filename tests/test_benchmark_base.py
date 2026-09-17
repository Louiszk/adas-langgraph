"""Unit tests for benchmark/benchmark_base.py verifying container execution and CLI contracts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from adas_core.environment import SANDBOX_WORKSPACE_DIR
from benchmark.benchmark_base import benchmark_cli_main, run_benchmark_in_sandbox


class TestRunBenchmarkInSandbox:
    def test_required_packages_use_exit_codes_and_install_missing_packages(self):
        mock_session = MagicMock()
        show_result = MagicMock(exit_code=1)
        install_result = MagicMock(exit_code=0)

        def execute_command(command):
            if command.startswith("pip show"):
                return show_result
            if command.startswith("pip install"):
                return install_result
            if "ls -la" in command:
                return "benchmark_results_TestSystem.json"
            return ""

        mock_session.execute_command.side_effect = execute_command
        mock_session.execute_command_streaming.return_value = ["\n__ADAS_BENCH_EXIT__0\n"]

        with patch("benchmark.benchmark_base.os.makedirs"):
            assert run_benchmark_in_sandbox(
                session=mock_session,
                benchmark_name="gsm8k",
                system_name="TestSystem",
                runner_script="benchmark/gsm8k/run_gsm8k_bench.py",
                required_packages=["scikit-learn==1.5.1"],
            )

        commands = [call.args[0] for call in mock_session.execute_command.call_args_list]
        assert "pip show scikit-learn" in commands
        assert "pip install scikit-learn==1.5.1" in commands

    def test_container_marker_failure_returns_false(self, tmp_path):
        mock_session = MagicMock()
        mock_session.execute_command_streaming.return_value = [
            "Running benchmark...\n",
            "Something failed\n",
            "\n__ADAS_BENCH_EXIT__1\n",
        ]
        mock_session.execute_command.return_value = ""

        with patch("benchmark.benchmark_base.os.makedirs"):
            result = run_benchmark_in_sandbox(
                session=mock_session,
                benchmark_name="gsm8k",
                system_name="TestSystem",
                runner_script="benchmark/gsm8k/run_gsm8k_bench.py",
            )

        assert result is False
        # Must not attempt to copy results if execution failed in container
        mock_session.copy_from_runtime.assert_not_called()

    def test_container_rejects_nonzero_final_marker(self):
        mock_session = MagicMock()
        mock_session.execute_command_streaming.return_value = ["\n__ADAS_BENCH_EXIT__1\n"]

        with patch("benchmark.benchmark_base.os.makedirs"):
            result = run_benchmark_in_sandbox(
                session=mock_session,
                benchmark_name="gsm8k",
                system_name="TestSystem",
                runner_script="benchmark/gsm8k/run_gsm8k_bench.py",
            )

        assert result is False
        mock_session.copy_from_runtime.assert_not_called()

    def test_missing_result_artifact_returns_false(self, tmp_path):
        mock_session = MagicMock()
        mock_session.execute_command_streaming.return_value = [
            "Running benchmark...\n",
            "\n__ADAS_BENCH_EXIT__0\n",
        ]
        # ls -la output lacks benchmark_results_TestSystem.json
        mock_session.execute_command.return_value = "total 0\n-rw-r--r-- other_file.txt"

        with patch("benchmark.benchmark_base.os.makedirs"):
            result = run_benchmark_in_sandbox(
                session=mock_session,
                benchmark_name="gsm8k",
                system_name="TestSystem",
                runner_script="benchmark/gsm8k/run_gsm8k_bench.py",
            )

        assert result is False
        mock_session.copy_from_runtime.assert_not_called()

    def test_successful_run_copies_result_and_returns_true(self, tmp_path):
        mock_session = MagicMock()
        mock_session.execute_command_streaming.return_value = [
            "Running benchmark...\n",
            "Finished 100 problems\n",
            "\n__ADAS_BENCH_EXIT__0\n",
        ]
        mock_session.execute_command.return_value = "total 4\n-rw-r--r-- benchmark_results_TestSystem.json"

        with patch("benchmark.benchmark_base.os.makedirs"):
            result = run_benchmark_in_sandbox(
                session=mock_session,
                benchmark_name="gsm8k",
                system_name="TestSystem",
                runner_script="benchmark/gsm8k/run_gsm8k_bench.py",
            )

        assert result is True
        mock_session.copy_from_runtime.assert_called_once_with(
            f"{SANDBOX_WORKSPACE_DIR}/benchmark/gsm8k/results/benchmark_results_TestSystem.json",
            "benchmark/gsm8k/results/benchmark_results_TestSystem.json",
        )

    def test_rejects_invalid_system_name_before_running_command(self):
        mock_session = MagicMock()
        assert not run_benchmark_in_sandbox(
            session=mock_session,
            benchmark_name="gsm8k",
            system_name="System with spaces",
            runner_script="benchmark/gsm8k/run_gsm8k_bench.py",
        )
        mock_session.execute_command_streaming.assert_not_called()

    def test_accepts_dotted_system_module_and_copies_its_file(self):
        mock_session = MagicMock()
        mock_session.execute_command_streaming.return_value = ["\n__ADAS_BENCH_EXIT__0\n"]
        mock_session.execute_command.return_value = "benchmark_results_generated_systems.TestSystem.json"

        with patch("benchmark.benchmark_base.os.makedirs"):
            assert run_benchmark_in_sandbox(
                session=mock_session,
                benchmark_name="gsm8k",
                system_name="generated_systems.TestSystem",
                runner_script="benchmark/gsm8k/run_gsm8k_bench.py",
            )

        from adas_core.environment import SANDBOX_WORKSPACE_DIR

        mock_session.copy_to_runtime.assert_any_call(
            "generated_systems/TestSystem.py",
            f"{SANDBOX_WORKSPACE_DIR}/generated_systems/TestSystem.py",
        )

    def test_stages_shared_benchmark_module_and_dataset(self, tmp_path):
        dataset = tmp_path / "problem_subset.json"
        dataset.write_text("[]", encoding="utf-8")
        mock_session = MagicMock()
        mock_session.execute_command_streaming.return_value = ["\n__ADAS_BENCH_EXIT__0\n"]
        mock_session.execute_command.return_value = "benchmark_results_TestSystem.json"

        with patch("benchmark.benchmark_base.os.makedirs"):
            assert run_benchmark_in_sandbox(
                session=mock_session,
                benchmark_name="gsm8k",
                system_name="TestSystem",
                runner_script="benchmark/gsm8k/run_gsm8k_bench.py",
                dataset_file=str(dataset),
            )

        mock_session.copy_to_runtime.assert_any_call(
            "benchmark/benchmark_base.py",
            f"{SANDBOX_WORKSPACE_DIR}/benchmark/benchmark_base.py",
        )
        mock_session.copy_to_runtime.assert_any_call(
            str(dataset),
            f"{SANDBOX_WORKSPACE_DIR}/{dataset}",
        )

    def test_missing_dataset_prevents_benchmark_execution(self):
        mock_session = MagicMock()

        assert not run_benchmark_in_sandbox(
            session=mock_session,
            benchmark_name="gsm8k",
            system_name="TestSystem",
            runner_script="benchmark/gsm8k/run_gsm8k_bench.py",
            dataset_file="missing/problem_subset.json",
        )
        mock_session.execute_command_streaming.assert_not_called()


class TestBenchmarkCliMain:
    def test_returns_0_on_success(self, monkeypatch):
        mock_session = MagicMock()
        mock_run_fn = MagicMock(return_value=True)

        with (
            patch("sandbox.sandbox.StreamingSandboxSession", return_value=mock_session),
            patch("sandbox.sandbox.setup_sandbox_environment", return_value=True),
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["main_gsm8k_bench.py", "--system", "GSM8KBaseline"],
            )
            exit_code = benchmark_cli_main(
                benchmark_name="gsm8k",
                run_in_sandbox_fn=mock_run_fn,
            )

        assert exit_code == 0
        mock_session.open.assert_called_once()
        mock_run_fn.assert_called_once_with(mock_session, "GSM8KBaseline")
        mock_session.close.assert_called_once()

    def test_returns_1_on_run_failure(self, monkeypatch):
        mock_session = MagicMock()
        mock_run_fn = MagicMock(return_value=False)

        with (
            patch("sandbox.sandbox.StreamingSandboxSession", return_value=mock_session),
            patch("sandbox.sandbox.setup_sandbox_environment", return_value=True),
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["main_gsm8k_bench.py", "--system", "GSM8KBaseline"],
            )
            exit_code = benchmark_cli_main(
                benchmark_name="gsm8k",
                run_in_sandbox_fn=mock_run_fn,
            )

        assert exit_code == 1
        mock_session.open.assert_called_once()
        mock_session.close.assert_called_once()

    def test_returns_1_on_setup_failure(self, monkeypatch):
        mock_session = MagicMock()
        mock_run_fn = MagicMock()

        with (
            patch("sandbox.sandbox.StreamingSandboxSession", return_value=mock_session),
            patch("sandbox.sandbox.setup_sandbox_environment", return_value=False),
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["main_gsm8k_bench.py", "--system", "GSM8KBaseline"],
            )
            exit_code = benchmark_cli_main(
                benchmark_name="gsm8k",
                run_in_sandbox_fn=mock_run_fn,
            )

        assert exit_code == 1
        mock_session.open.assert_called_once()
        mock_run_fn.assert_not_called()
        mock_session.close.assert_called_once()

    def test_returns_1_on_exception_and_closes_session(self, monkeypatch):
        mock_session = MagicMock()
        mock_session.open.side_effect = RuntimeError("Container daemon unavailable")

        with patch("sandbox.sandbox.StreamingSandboxSession", return_value=mock_session):
            monkeypatch.setattr(
                "sys.argv",
                ["main_gsm8k_bench.py", "--system", "GSM8KBaseline"],
            )
            exit_code = benchmark_cli_main(
                benchmark_name="gsm8k",
                run_in_sandbox_fn=MagicMock(),
            )

        assert exit_code == 1
        mock_session.close.assert_called_once()

    def test_rejects_invalid_system_name_before_opening_session(self, monkeypatch):
        with patch("sandbox.sandbox.StreamingSandboxSession") as mock_session_cls:
            monkeypatch.setattr("sys.argv", ["main_gsm8k_bench.py", "--system", "Bad/System"])
            assert benchmark_cli_main("gsm8k", MagicMock()) == 1
        mock_session_cls.assert_not_called()

    def test_accepts_dotted_system_module(self, monkeypatch):
        mock_session = MagicMock()
        mock_run_fn = MagicMock(return_value=True)
        with (
            patch("sandbox.sandbox.StreamingSandboxSession", return_value=mock_session),
            patch("sandbox.sandbox.setup_sandbox_environment", return_value=True),
        ):
            monkeypatch.setattr("sys.argv", ["main_gsm8k_bench.py", "--system", "generated_systems.TestSystem"])
            assert benchmark_cli_main("gsm8k", mock_run_fn) == 0
        mock_run_fn.assert_called_once_with(mock_session, "generated_systems.TestSystem")
