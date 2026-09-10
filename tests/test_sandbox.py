"""
Specification tests for sandbox runtime initialization and configuration.
"""

from unittest.mock import MagicMock, patch

import pytest

from sandbox.sandbox import StreamingSandboxSession


class TestSandboxSessionSpecification:
    def test_invalid_container_type_raises_value_error(self):
        """Contract: Must raise ValueError when passed an unknown container runtime type."""
        with pytest.raises(ValueError, match="Unknown container type: lxc"):
            StreamingSandboxSession(container_type="lxc")

    @patch("sandbox.sandbox.create_session")
    @patch("sandbox.sandbox.check_docker_running", return_value=True)
    def test_auto_selects_docker_when_running(self, mock_docker, mock_create):
        """Contract: When container_type='auto' and Docker is running, should select Docker backend."""
        session = StreamingSandboxSession(container_type="auto", verbose=False)
        assert session.verbose is False
        mock_docker.assert_called_once()
        mock_create.assert_called_once()
        assert mock_create.call_args.kwargs["skip_environment_setup"] is True

    @patch("sandbox.sandbox.create_session")
    @patch("sandbox.sandbox.check_docker_running", return_value=False)
    @patch("sandbox.sandbox.check_podman_running", return_value=True)
    def test_auto_fallback_to_podman(self, mock_podman, mock_docker, mock_create):
        """Contract: When Docker is unavailable, 'auto' must fall back to Podman."""
        session = StreamingSandboxSession(container_type="auto", verbose=False)
        assert session.verbose is False
        mock_docker.assert_called_once()
        mock_podman.assert_called_once()
        mock_create.assert_called_once()
        assert mock_create.call_args.kwargs["skip_environment_setup"] is True

    @patch("sandbox.sandbox.check_docker_running", return_value=False)
    @patch("sandbox.sandbox.check_podman_running", return_value=False)
    def test_auto_raises_when_neither_container_engine_available(self, mock_podman, mock_docker):
        """Contract: If neither Docker nor Podman are available, auto must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Neither Docker nor Podman are running"):
            StreamingSandboxSession(container_type="auto")

    @patch("docker.from_env")
    def test_reuses_existing_cached_default_image(self, mock_from_env):
        """The default runtime must not rebuild packages for each new session."""
        from sandbox.sandbox import ensure_cached_sandbox_image

        mock_client = MagicMock()
        mock_from_env.return_value = mock_client

        image = ensure_cached_sandbox_image()

        assert image.startswith("adas-sandbox:")
        mock_client.images.get.assert_called_once_with(image)
        mock_client.images.build.assert_not_called()

    @patch("docker.from_env")
    def test_builds_cached_image_with_configured_dependencies_when_absent(self, mock_from_env):
        """A new dependency set creates one new reusable image."""
        from docker.errors import ImageNotFound

        from config import settings
        from sandbox.sandbox import ensure_cached_sandbox_image

        mock_client = MagicMock()
        mock_client.images.get.side_effect = ImageNotFound("missing")
        mock_from_env.return_value = mock_client

        image = ensure_cached_sandbox_image()

        assert image.startswith("adas-sandbox:")
        assert mock_client.images.build.call_args.kwargs["tag"] == image
        assert mock_client.images.build.call_args.kwargs["buildargs"] == {
            "ADAS_SANDBOX_DEPENDENCIES": " ".join(settings.dependencies)
        }

    @patch("sandbox.sandbox.ensure_cached_sandbox_image", return_value="adas-sandbox:testtag")
    @patch("sandbox.sandbox.create_session")
    @patch("sandbox.sandbox.check_docker_running", return_value=True)
    def test_open_sets_cached_image_when_default(self, mock_docker, mock_create, mock_ensure_cached):
        mock_session = MagicMock()
        mock_create.return_value = mock_session
        session = StreamingSandboxSession(container_type="auto", verbose=False)
        session.open()
        assert mock_session.config.image == "adas-sandbox:testtag"
        mock_ensure_cached.assert_called_once_with(client=mock_session.client)
        mock_session.open.assert_called_once()

    @patch("sandbox.sandbox.ensure_cached_sandbox_image")
    @patch("sandbox.sandbox.create_session")
    @patch("sandbox.sandbox.check_docker_running", return_value=True)
    def test_open_preserves_custom_image(self, mock_docker, mock_create, mock_ensure_cached):
        mock_session = MagicMock()
        mock_create.return_value = mock_session
        session = StreamingSandboxSession(image="my-custom-image:1.0", container_type="auto", verbose=False)
        assert session._uses_cached_image is False
        session.open()
        mock_ensure_cached.assert_not_called()
        mock_session.open.assert_called_once()

    def test_reuses_cached_image_with_provided_client(self):
        """When an explicit client (e.g., Podman) is passed, it should be used directly."""
        from sandbox.sandbox import ensure_cached_sandbox_image

        mock_client = MagicMock()
        image = ensure_cached_sandbox_image(client=mock_client)

        assert image.startswith("adas-sandbox:")
        mock_client.images.get.assert_called_once_with(image)
        mock_client.images.build.assert_not_called()

    def test_builds_cached_image_with_provided_client_when_absent(self):
        """When the image is absent in the provided client, build using that client."""
        from config import settings
        from sandbox.sandbox import ensure_cached_sandbox_image

        mock_client = MagicMock()
        from docker.errors import ImageNotFound

        mock_client.images.get.side_effect = ImageNotFound("missing")
        image = ensure_cached_sandbox_image(client=mock_client)

        assert image.startswith("adas-sandbox:")
        assert mock_client.images.build.call_args.kwargs["tag"] == image
        assert mock_client.images.build.call_args.kwargs["buildargs"] == {
            "ADAS_SANDBOX_DEPENDENCIES": " ".join(settings.dependencies)
        }

    def test_propagates_cached_image_lookup_errors(self):
        """Daemon and permission failures must not be mistaken for a missing image."""
        from sandbox.sandbox import ensure_cached_sandbox_image

        mock_client = MagicMock()
        mock_client.images.get.side_effect = RuntimeError("daemon unavailable")

        with pytest.raises(RuntimeError, match="daemon unavailable"):
            ensure_cached_sandbox_image(client=mock_client)
        mock_client.images.build.assert_not_called()

    def test_setup_sandbox_environment_syncs_meta_system_and_core(self):
        """setup_sandbox_environment creates workspaces and syncs meta_system package."""
        from sandbox.sandbox import setup_sandbox_environment

        mock_session = MagicMock()
        mock_check = MagicMock()
        mock_check.exit_code = 0
        mock_check.stderr = ""
        mock_session.execute_command.return_value = mock_check

        success = setup_sandbox_environment(mock_session, reinstall=False)
        assert success is True

        # Check directories created
        created_dirs = [call.args[0] for call in mock_session.execute_command.call_args_list]
        assert any("mkdir -p /sandbox/workspace/meta_system" in cmd for cmd in created_dirs)
        assert any("mkdir -p /sandbox/workspace/adas_core" in cmd for cmd in created_dirs)

        # Check meta_system directory copy was invoked
        copied_dirs = [call.kwargs.get("dest_dir") for call in mock_session.copy_dir_to_runtime.call_args_list]
        assert "/sandbox/workspace/meta_system" in copied_dirs


class TestSetupSandboxUtilities:
    def test_setup_manifest_currentness_tracks_task_spec_contents(self, tmp_path):
        import hashlib
        import json

        from create_setup import setup_manifest_is_current

        task_spec = tmp_path / "task.json"
        task_spec.write_text('{"name": "first"}', encoding="utf-8")
        digest = hashlib.sha256(task_spec.read_bytes()).hexdigest()
        (tmp_path / "setup_manifest.json").write_text(json.dumps({"files": {"task.json": digest}}), encoding="utf-8")
        assert setup_manifest_is_current(task_spec)

        task_spec.write_text('{"name": "changed"}', encoding="utf-8")
        assert not setup_manifest_is_current(task_spec)

    def test_setup_manifest_currentness_handles_crlf_and_lf(self, tmp_path):
        import hashlib
        import json

        from create_setup import setup_manifest_is_current

        task_spec = tmp_path / "task.json"
        lf_content = b'{\n  "name": "cross_platform"\n}\n'
        crlf_content = b'{\r\n  "name": "cross_platform"\r\n}\r\n'
        lf_hash = hashlib.sha256(lf_content).hexdigest()

        manifest_path = tmp_path / "setup_manifest.json"
        manifest_path.write_text(json.dumps({"files": {"task.json": lf_hash}}), encoding="utf-8")

        # When file on disk has CRLF (typical Windows checkout)
        task_spec.write_bytes(crlf_content)
        assert setup_manifest_is_current(task_spec)

        # When file on disk has LF (typical Linux checkout)
        task_spec.write_bytes(lf_content)
        assert setup_manifest_is_current(task_spec)

    def test_setup_manifest_currentness_checks_every_declared_artifact(self, tmp_path):
        import hashlib
        import json

        from create_setup import setup_manifest_is_current

        task_spec = tmp_path / "task.json"
        task_spec.write_text('{"name": "complete"}\n', encoding="utf-8")
        fixture = tmp_path / "fixtures" / "input.csv"
        fixture.parent.mkdir()
        fixture.write_text("value\n1\n", encoding="utf-8")
        hashes = {
            "task.json": hashlib.sha256(task_spec.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "fixtures/input.csv": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        }
        (tmp_path / "setup_manifest.json").write_text(json.dumps({"files": hashes}), encoding="utf-8")
        assert setup_manifest_is_current(task_spec)

        fixture.unlink()
        assert not setup_manifest_is_current(task_spec)

        fixture.write_text("value\n2\n", encoding="utf-8")
        assert not setup_manifest_is_current(task_spec)

    def test_setup_regeneration_preserves_validation_section(self, tmp_path):
        import json

        from create_setup import _restore_validation_section

        manifest_path = tmp_path / "setup_manifest.json"
        manifest_path.write_text(json.dumps({"files": {"task.json": "new"}}), encoding="utf-8")
        validation = {"validator_hash": "old", "task_spec_hash": "old"}

        _restore_validation_section(tmp_path, validation)

        assert json.loads(manifest_path.read_text(encoding="utf-8"))["validation"] == validation

    def test_package_pattern_validation(self):
        from adas_core.environment import _PACKAGE_PATTERN, validate_package_requirement
        from sandbox.run_setup import install_packages

        valid_packages = [
            "neo4j",
            "neo4j>=5.0",
            "neo4j>=5.0,<6.0",
            "uvicorn[standard]",
            "uvicorn[standard]>=0.20.0",
            "psycopg2-binary",
            "duckdb~=0.9.0",
            "faker",
        ]
        for pkg in valid_packages:
            assert _PACKAGE_PATTERN.fullmatch(pkg), f"Expected valid: {pkg}"
            assert validate_package_requirement(pkg), f"Expected valid: {pkg}"

        invalid_packages = [
            "sh -c rm -rf /",
            "pkg; rm -rf /",
            "pkg && curl evil.com",
            "-r requirements.txt",
        ]
        for pkg in invalid_packages:
            assert not _PACKAGE_PATTERN.fullmatch(pkg), f"Expected invalid: {pkg}"
            assert not validate_package_requirement(pkg), f"Expected invalid: {pkg}"

        with pytest.raises(ValueError, match="Invalid package requirement"):
            install_packages(["safe-pkg", "bad; rm -rf"])

    def test_copy_tree_from_runtime_handles_object_and_string(self, tmp_path):
        from create_setup import _copy_tree_from_runtime

        mock_session = MagicMock()
        # Case 1: command returns object with .stdout
        mock_result = MagicMock()
        mock_result.stdout = "/sandbox/task/file1.txt\n/sandbox/task/subdir/file2.txt\n"
        mock_session.execute_command.return_value = mock_result

        dest = tmp_path / "out1"
        _copy_tree_from_runtime(mock_session, "/sandbox/task", dest)
        assert mock_session.copy_from_runtime.call_count == 2

        # Case 2: command returns raw string
        mock_session.reset_mock()
        mock_session.execute_command.return_value = "/sandbox/task/file3.txt\n"
        dest2 = tmp_path / "out2"
        _copy_tree_from_runtime(mock_session, "/sandbox/task", dest2)
        assert mock_session.copy_from_runtime.call_count == 1

    def test_copy_task_setup_to_sandbox(self, tmp_path):
        from adas_core.environment import SANDBOX_TASK_SETUP_DIR
        from sandbox.sandbox import copy_task_setup_to_sandbox

        task_dir = tmp_path / "my_task"
        task_dir.mkdir()
        (task_dir / "setup.py").write_text("# setup")
        sub = task_dir / "fixtures"
        sub.mkdir()
        (sub / "data.csv").write_text("1,2,3")
        pycache = task_dir / "__pycache__"
        pycache.mkdir()
        (pycache / "cached.pyc").write_text("bytecode")
        spec_path = tmp_path / "custom.task.json"
        spec_path.write_text('{"name": "test"}')

        mock_session = MagicMock()
        ret = copy_task_setup_to_sandbox(mock_session, task_dir, spec_path)

        assert ret == SANDBOX_TASK_SETUP_DIR
        # mkdir called for runtime_task_dir and subdirectories (excluding pycache)
        mkdir_commands = [
            call.args[0] for call in mock_session.execute_command.call_args_list if "mkdir -p" in call.args[0]
        ]
        assert any(f"mkdir -p '{SANDBOX_TASK_SETUP_DIR}'" in cmd for cmd in mkdir_commands)
        assert any(f"mkdir -p '{SANDBOX_TASK_SETUP_DIR}/fixtures'" in cmd for cmd in mkdir_commands)
        assert not any("__pycache__" in cmd for cmd in mkdir_commands)

        # copy_to_runtime called for regular files and task.json
        copied_destinations = [call.args[1] for call in mock_session.copy_to_runtime.call_args_list]
        assert f"{SANDBOX_TASK_SETUP_DIR}/setup.py" in copied_destinations
        assert f"{SANDBOX_TASK_SETUP_DIR}/fixtures/data.csv" in copied_destinations
        assert f"{SANDBOX_TASK_SETUP_DIR}/task.json" in copied_destinations
        assert not any("__pycache__" in dest for dest in copied_destinations)

    def test_run_sandbox_preflight_success(self):
        from adas_core.environment import SANDBOX_TASK_SETUP_DIR
        from sandbox.sandbox import run_sandbox_preflight

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.exit_code = 0
        mock_result.stdout = "Preflight verification passed"
        mock_result.stderr = ""
        mock_session.execute_command.return_value = mock_result

        assert run_sandbox_preflight(mock_session, SANDBOX_TASK_SETUP_DIR) is True
        mock_session.execute_command.assert_called_once_with(
            f"python3 /sandbox/workspace/run_preflight.py --task-dir {SANDBOX_TASK_SETUP_DIR}"
        )

    def test_run_sandbox_preflight_failure_text(self):
        from sandbox.sandbox import run_sandbox_preflight

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.exit_code = 0
        mock_result.stdout = "Preflight check failed: missing pkg"
        mock_result.stderr = ""
        mock_session.execute_command.return_value = mock_result

        assert run_sandbox_preflight(mock_session) is False

    def test_run_sandbox_preflight_failure_exit_code(self):
        from sandbox.sandbox import run_sandbox_preflight

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.exit_code = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error importing dependency"
        mock_session.execute_command.return_value = mock_result

        assert run_sandbox_preflight(mock_session) is False


class TestInvokeDesignCLI:
    def test_system_name_override(self, tmp_path, monkeypatch):
        import invoke_design
        from adas_core.task_spec import TaskSpec

        spec_file = tmp_path / "task.json"
        spec_data = {
            "name": "OriginalTaskName",
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
        import invoke_design
        from adas_core.task_spec import TaskSpec

        spec_file = tmp_path / "task.json"
        spec_data = {
            "name": "MissingValidationTask",
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

        with (
            patch("invoke_design.run_setup_for_task") as mock_run_setup,
            patch("invoke_design.setup_manifest_is_current", return_value=False),
            patch("invoke_design.StreamingSandboxSession") as mock_session_cls,
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["invoke_design.py", "--task-spec", str(spec_file), "--auto-setup"],
            )
            exit_code = invoke_design.main()
            assert exit_code == 1
            # Must fail before running auto-setup or opening a sandbox session
            mock_run_setup.assert_not_called()
            mock_session_cls.assert_not_called()


class TestInvokeTargetCLI:
    def test_invoke_target_rejects_invalid_system_name(self, monkeypatch):
        import invoke_target

        with patch("invoke_target.StreamingSandboxSession") as mock_session_cls:
            monkeypatch.setattr(
                "sys.argv",
                ["invoke_target.py", "--system-name", "Bad/System", "--state", "{}"],
            )
            assert invoke_target.main() == 1
        mock_session_cls.assert_not_called()

    def test_invoke_target_without_task_spec(self, monkeypatch):
        import invoke_target

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
        import invoke_target
        from adas_core.task_spec import TaskSpec

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
        import invoke_target

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
        import invoke_target

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
        import invoke_target

        monkeypatch.setattr("sys.argv", ["invoke_target.py", "--system-name", "TestSystem"])
        with pytest.raises(SystemExit, match="2"):
            invoke_target.main()

    def test_run_target_system_in_sandbox_command_formatting(self):
        import invoke_target

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
        import invoke_target
        from adas_core.task_spec import TaskSpec

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
