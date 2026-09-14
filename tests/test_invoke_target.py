"""Specification tests for invoke_target CLI."""

from unittest.mock import MagicMock, patch

import pytest

import invoke_target
from adas_core.task_spec import TaskSpec


class TestInvokeTargetCLI:
    def test_invoke_target_rejects_invalid_system_name(self, monkeypatch):
        with patch("invoke_target.StreamingSandboxSession") as mock_session_cls:
            monkeypatch.setattr(
                "sys.argv",
                ["invoke_target.py", "--system-name", "Bad/System", "--state", "{}"],
            )
            assert invoke_target.main() == 1
        mock_session_cls.assert_not_called()

    def test_invoke_target_without_task_spec(self, monkeypatch):
        mock_session = MagicMock()
        mock_session.execute_command.return_value = ""

        with (
            patch("invoke_target.StreamingSandboxSession", return_value=mock_session),
            patch("invoke_target.setup_sandbox_environment", return_value=True),
            patch("invoke_target.run_target_system_in_sandbox") as mock_run_target,
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["invoke_target.py", "--system-name", "TestSystem", "--state", '{"messages": ["hello"]}'],
            )
            exit_code = invoke_target.main()
            assert exit_code == 0
            mock_run_target.assert_called_once()
            assert mock_run_target.call_args[0][1] == "TestSystem"
            assert mock_run_target.call_args[0][2] == {"messages": ["hello"]}

    def test_invoke_target_with_task_spec_and_preflight(self, tmp_path, monkeypatch):
        spec_file = tmp_path / "task.json"
        spec_data = {
            "name": "TargetTask",
            "system_goal": "Goal",
            "architecture_contract": {
                "execution_mode": "single_turn",
                "state_schema": {"messages": "list[dict]"},
                "persistence": {},
                "required_tools": [],
            },
            "resource_manifest": {"available_resources": [], "available_api_keys": []},
            "dev_suite": [
                {
                    "id": "case_1",
                    "description": "Test case 1",
                    "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
                }
            ],
        }
        spec = TaskSpec.model_validate(spec_data)
        spec.save(spec_file)

        mock_session = MagicMock()
        mock_session.execute_command.return_value = ""

        with (
            patch("invoke_target.setup_manifest_is_current", return_value=True),
            patch("invoke_target.StreamingSandboxSession", return_value=mock_session),
            patch("invoke_target.setup_sandbox_environment", return_value=True),
            patch("invoke_target.copy_task_setup_to_sandbox", return_value="/sandbox/task_setup") as mock_copy,
            patch("invoke_target.run_sandbox_preflight", return_value=True) as mock_preflight,
            patch("invoke_target.run_target_system_in_sandbox") as mock_run_target,
        ):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "invoke_target.py",
                    "--system_name",
                    "TargetTask_v0",
                    "--task-spec",
                    str(spec_file),
                    "--state",
                    '{"messages": [{"role": "user", "content": "hi"}]}',
                ],
            )
            exit_code = invoke_target.main()
            assert exit_code == 0
            mock_copy.assert_called_once()
            mock_preflight.assert_called_once_with(mock_session, "/sandbox/task_setup")
            mock_run_target.assert_called_once()
            assert mock_run_target.call_args[0][1] == "TargetTask_v0"
            assert mock_run_target.call_args[0][2] == {"messages": [{"role": "user", "content": "hi"}]}
            assert mock_run_target.call_args[1]["task_dir"] == "/sandbox/task_setup"

    def test_invoke_target_with_state_file(self, tmp_path, monkeypatch):
        state_file = tmp_path / "custom_state.json"
        state_file.write_text('{"analysis_task": "sales"}', encoding="utf-8")

        mock_session = MagicMock()
        mock_session.execute_command.return_value = ""

        with (
            patch("invoke_target.StreamingSandboxSession", return_value=mock_session),
            patch("invoke_target.setup_sandbox_environment", return_value=True),
            patch("invoke_target.run_target_system_in_sandbox") as mock_run_target,
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["invoke_target.py", "--system-name", "TestSystem", "--state-file", str(state_file)],
            )
            exit_code = invoke_target.main()
            assert exit_code == 0
            mock_run_target.assert_called_once()
            assert mock_run_target.call_args[0][2] == {"analysis_task": "sales"}
            assert mock_run_target.call_args[1]["task_dir"] is None

    def test_invoke_target_with_missing_state_file_fails(self, tmp_path, monkeypatch):
        missing_state_file = tmp_path / "non_existent.json"

        mock_session = MagicMock()

        with (
            patch("invoke_target.StreamingSandboxSession", return_value=mock_session),
            patch("invoke_target.run_target_system_in_sandbox") as mock_run_target,
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["invoke_target.py", "--system-name", "TestSystem", "--state-file", str(missing_state_file)],
            )
            exit_code = invoke_target.main()
            assert exit_code == 1
            mock_run_target.assert_not_called()

    def test_invoke_target_requires_state(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["invoke_target.py", "--system-name", "TestSystem"])
        with pytest.raises(SystemExit, match="2"):
            invoke_target.main()

    def test_run_target_system_in_sandbox_command_formatting(self):
        mock_session = MagicMock()
        mock_session.execute_command_streaming.return_value = ["__ADAS_TARGET_EXIT__0\n"]

        assert invoke_target.run_target_system_in_sandbox(
            session=mock_session,
            system_name="MySystem",
            state={"query": "test"},
            run_id="run_123",
            task_dir="/sandbox/workspace/task_setup",
        )

        mock_session.execute_command_streaming.assert_called_once()
        cmd = mock_session.execute_command_streaming.call_args[0][0]
        assert "--system_name=MySystem" in cmd
        assert "--run-id=run_123" in cmd
        assert "--task-dir=/sandbox/workspace/task_setup" in cmd
        assert "--state='{" + '"query": "test"' + "}'" in cmd

    def test_invoke_target_aborts_on_preflight_failure(self, tmp_path, monkeypatch):
        spec_file = tmp_path / "task.json"
        spec_data = {
            "name": "TargetTask",
            "system_goal": "Goal",
            "architecture_contract": {
                "execution_mode": "single_turn",
                "state_schema": {"messages": "list[dict]"},
                "persistence": {},
                "required_tools": [],
            },
            "resource_manifest": {"available_resources": [], "available_api_keys": []},
            "dev_suite": [
                {
                    "id": "case_1",
                    "description": "Test case 1",
                    "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
                }
            ],
        }
        spec = TaskSpec.model_validate(spec_data)
        spec.save(spec_file)

        mock_session = MagicMock()
        mock_session.execute_command.return_value = ""

        with (
            patch("invoke_target.setup_manifest_is_current", return_value=True),
            patch("invoke_target.StreamingSandboxSession", return_value=mock_session),
            patch("invoke_target.setup_sandbox_environment", return_value=True),
            patch("invoke_target.copy_task_setup_to_sandbox", return_value="/sandbox/task_setup"),
            patch("invoke_target.run_sandbox_preflight", return_value=False),
            patch("invoke_target.run_target_system_in_sandbox") as mock_run_target,
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["invoke_target.py", "--system_name", "TargetTask_v0", "--task-spec", str(spec_file), "--state", "{}"],
            )
            exit_code = invoke_target.main()
            assert exit_code == 1
            mock_run_target.assert_not_called()
