"""Specification tests for invoke_design CLI."""

from unittest.mock import MagicMock, patch

import invoke_design
from adas_core.task_spec import ArchitectureContract, TaskSpec


class TestInvokeDesignCLI:
    def test_rejects_empty_development_suite(self, tmp_path, monkeypatch):
        spec_file = tmp_path / "task.json"
        spec = TaskSpec(
            name="NoDevCasesTask",
            system_goal="Goal",
            architecture_contract=ArchitectureContract(
                execution_mode="single_turn",
                state_schema={"messages": "list[dict]"},
                required_tools=[],
            ),
            dev_suite=[],
        )
        spec.save(spec_file)

        with patch("invoke_design.StreamingSandboxSession") as mock_session_cls:
            monkeypatch.setattr("sys.argv", ["invoke_design.py", "--task-spec", str(spec_file)])
            assert invoke_design.main() == 1

        mock_session_cls.assert_not_called()

    def test_system_name_override(self, tmp_path, monkeypatch):
        spec_file = tmp_path / "task.json"
        spec_data = {
            "name": "OriginalTaskName",
            "system_goal": "Goal",
            "architecture_contract": {
                "execution_mode": "single_turn",
                "state_schema": {"messages": "list[dict]"},
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

        # Seed the validation file so host-side check passes
        (tmp_path / "OriginalTaskName.validation.py").touch()

        mock_session = MagicMock()
        mock_session.execute_command.return_value = ""
        mock_session.execute_command_streaming.return_value = ["chunk"]

        with (
            patch("invoke_design.setup_manifest_is_current", return_value=True),
            patch("invoke_design.is_validation_manifest_current", return_value=True),
            patch("invoke_design.StreamingSandboxSession", return_value=mock_session),
            patch("invoke_design.setup_sandbox_environment", return_value=True),
            patch("invoke_design.copy_task_setup_to_sandbox", return_value="/sandbox/task_setup"),
            patch("invoke_design.run_sandbox_preflight", return_value=True),
            patch("invoke_design.run_meta_system_in_sandbox", return_value=True) as mock_run_meta,
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["invoke_design.py", "--task-spec", str(spec_file), "--system-name", "CustomOverrideSystem"],
            )
            exit_code = invoke_design.main()
            assert exit_code == 0
            mock_run_meta.assert_called_once_with(
                session=mock_session,
                target_name="CustomOverrideSystem",
                optimize_system=None,
            )

    def test_fails_early_when_validation_missing_on_host(self, tmp_path, monkeypatch):
        spec_file = tmp_path / "task.json"
        spec_data = {
            "name": "MissingValidationTask",
            "system_goal": "Goal",
            "architecture_contract": {
                "execution_mode": "single_turn",
                "state_schema": {"messages": "list[dict]"},
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

        with (
            patch("invoke_design.run_setup_for_task") as mock_run_setup,
            patch("invoke_design.setup_manifest_is_current", return_value=False),
            patch("invoke_design.StreamingSandboxSession") as mock_session_cls,
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["invoke_design.py", "--task-spec", str(spec_file)],
            )
            exit_code = invoke_design.main()
            assert exit_code == 1
            # Must fail before running auto-setup or opening a sandbox session
            mock_run_setup.assert_not_called()
            mock_session_cls.assert_not_called()

    def test_runs_auto_validation_when_validation_stale_and_auto_setup_flag_present(self, tmp_path, monkeypatch):
        spec_file = tmp_path / "task.json"
        spec_data = {
            "name": "AutoValidationTask",
            "system_goal": "Goal",
            "architecture_contract": {
                "execution_mode": "single_turn",
                "state_schema": {"messages": "list[dict]"},
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
            patch("invoke_design.setup_manifest_is_current", return_value=True),
            patch("invoke_design.is_validation_manifest_current", return_value=False),
            patch("invoke_design.run_validation_for_task", return_value=tmp_path / "val.py") as mock_run_validation,
            patch("invoke_design.StreamingSandboxSession", return_value=mock_session),
            patch("invoke_design.setup_sandbox_environment", return_value=True),
            patch("invoke_design.copy_task_setup_to_sandbox", return_value="/sandbox/task_setup"),
            patch("invoke_design.run_sandbox_preflight", return_value=True),
            patch("invoke_design.run_meta_system_in_sandbox", return_value=True),
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["invoke_design.py", "--task-spec", str(spec_file), "--auto-setup"],
            )
            exit_code = invoke_design.main()
            assert exit_code == 0
            mock_run_validation.assert_called_once_with(spec_file, force=True)

    def test_runs_both_setup_and_validation_when_both_stale_and_auto_setup_flag_present(self, tmp_path, monkeypatch):
        spec_file = tmp_path / "task.json"
        spec_data = {
            "name": "AutoBothTask",
            "system_goal": "Goal",
            "architecture_contract": {
                "execution_mode": "single_turn",
                "state_schema": {"messages": "list[dict]"},
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
            patch("invoke_design.setup_manifest_is_current", return_value=False),
            patch("invoke_design.is_validation_manifest_current", return_value=False),
            patch("invoke_design.run_setup_for_task") as mock_run_setup,
            patch("invoke_design.run_validation_for_task", return_value=tmp_path / "val.py") as mock_run_validation,
            patch("invoke_design.StreamingSandboxSession", return_value=mock_session),
            patch("invoke_design.setup_sandbox_environment", return_value=True),
            patch("invoke_design.copy_task_setup_to_sandbox", return_value="/sandbox/task_setup"),
            patch("invoke_design.run_sandbox_preflight", return_value=True),
            patch("invoke_design.run_meta_system_in_sandbox", return_value=True),
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["invoke_design.py", "--task-spec", str(spec_file), "--auto-setup"],
            )
            exit_code = invoke_design.main()
            assert exit_code == 0
            mock_run_setup.assert_called_once()
            mock_run_validation.assert_called_once_with(spec_file, force=True)

    def test_fails_early_when_setup_stale_without_auto_setup(self, tmp_path, monkeypatch):
        spec_file = tmp_path / "task.json"
        spec_data = {
            "name": "StaleSetupTask",
            "system_goal": "Goal",
            "architecture_contract": {
                "execution_mode": "single_turn",
                "state_schema": {"messages": "list[dict]"},
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
        (tmp_path / "StaleSetupTask.validation.py").touch()

        with (
            patch("invoke_design.is_validation_manifest_current", return_value=True),
            patch("invoke_design.setup_manifest_is_current", return_value=False),
            patch("invoke_design.run_setup_for_task") as mock_run_setup,
            patch("invoke_design.StreamingSandboxSession") as mock_session_cls,
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["invoke_design.py", "--task-spec", str(spec_file)],
            )
            exit_code = invoke_design.main()
            assert exit_code == 1
            mock_run_setup.assert_not_called()
            mock_session_cls.assert_not_called()

    def test_runs_auto_setup_when_setup_stale_and_auto_setup_flag_present(self, tmp_path, monkeypatch):
        spec_file = tmp_path / "task.json"
        spec_data = {
            "name": "AutoSetupTask",
            "system_goal": "Goal",
            "architecture_contract": {
                "execution_mode": "single_turn",
                "state_schema": {"messages": "list[dict]"},
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
        (tmp_path / "AutoSetupTask.validation.py").touch()

        mock_session = MagicMock()
        mock_session.execute_command.return_value = ""

        with (
            patch("invoke_design.is_validation_manifest_current", return_value=True),
            patch("invoke_design.setup_manifest_is_current", return_value=False),
            patch("invoke_design.run_setup_for_task") as mock_run_setup,
            patch("invoke_design.StreamingSandboxSession", return_value=mock_session),
            patch("invoke_design.setup_sandbox_environment", return_value=True),
            patch("invoke_design.copy_task_setup_to_sandbox", return_value="/sandbox/task_setup"),
            patch("invoke_design.run_sandbox_preflight", return_value=True),
            patch("invoke_design.run_meta_system_in_sandbox", return_value=True),
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["invoke_design.py", "--task-spec", str(spec_file), "--auto-setup"],
            )
            exit_code = invoke_design.main()
            assert exit_code == 0
            mock_run_setup.assert_called_once()
