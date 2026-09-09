"""Unit tests for benchmark/benchmark_base.py verifying container execution and CLI contracts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from benchmark.benchmark_base import benchmark_cli_main, run_benchmark_in_sandbox


class TestRunBenchmarkInSandbox:
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
        from adas_core.environment import SANDBOX_WORKSPACE_DIR

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
