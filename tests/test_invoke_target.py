"""Specification tests for invoke_target CLI."""

import json
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

    @pytest.mark.parametrize(
        "batch_data",
        [
            [{"state": {}}],
            [{"id": "case_1"}],
            [{"id": "case_1", "state": []}],
            [{"id": "../escape", "state": {}}],
            [{"id": "case/one", "state": {}}],
            [{"id": ".", "state": {}}],
            [{"id": "", "state": {}}],
            [{"id": 1, "state": {}}],
        ],
    )
    def test_batch_rejects_malformed_entries(self, batch_data, tmp_path, monkeypatch):
        batch_file = tmp_path / "batch.json"
        batch_file.write_text(json.dumps(batch_data), encoding="utf-8")

        with patch("invoke_target.StreamingSandboxSession") as mock_session_cls:
            monkeypatch.setattr(
                "sys.argv",
                ["invoke_target.py", "--system-name", "TestSystem", "--batch", str(batch_file)],
            )
            assert invoke_target.main() == 1

        mock_session_cls.assert_not_called()

    def test_batch_rejects_duplicate_ids(self, tmp_path, monkeypatch):
        batch_file = tmp_path / "batch.json"
        batch_file.write_text(
            '[{"id": "case_1", "state": {}}, {"id": "case_1", "state": {}}]',
            encoding="utf-8",
        )

        with patch("invoke_target.StreamingSandboxSession") as mock_session_cls:
            monkeypatch.setattr(
                "sys.argv",
                ["invoke_target.py", "--system-name", "TestSystem", "--batch", str(batch_file)],
            )
            assert invoke_target.main() == 1

        mock_session_cls.assert_not_called()

    def test_batch_runs_safe_ids_and_returns_failure_if_any_case_fails(self, tmp_path, monkeypatch):
        batch_file = tmp_path / "batch.json"
        batch_file.write_text(
            '[{"id": "case-1", "state": {"query": "one"}}, {"id": "case_2", "state": {"query": "two"}}]',
            encoding="utf-8",
        )
        mock_session = MagicMock()
        mock_session.execute_command.return_value = ""

        with (
            patch("invoke_target.StreamingSandboxSession", return_value=mock_session),
            patch("invoke_target.setup_sandbox_environment", return_value=True),
            patch("invoke_target.run_target_system_in_sandbox", side_effect=[True, False]) as mock_run_target,
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["invoke_target.py", "--system-name", "TestSystem", "--batch", str(batch_file)],
            )
            assert invoke_target.main() == 1

        assert mock_run_target.call_count == 2
        run_ids = [call.kwargs["run_id"] for call in mock_run_target.call_args_list]
        assert all(run_id.endswith(case_id) for run_id, case_id in zip(run_ids, ["case-1", "case_2"], strict=True))
        assert all("/" not in run_id and "\\" not in run_id for run_id in run_ids)
        copied_destinations = [
            call.kwargs["dest_dir"] for call in mock_session.copy_dir_from_runtime.call_args_list[:2]
        ]
        assert copied_destinations == [
            f"data/output/TestSystem_{run_ids[0]}",
            f"data/output/TestSystem_{run_ids[1]}",
        ]

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
