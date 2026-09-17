"""
Specification tests for sandbox runtime initialization and configuration.
"""

import contextlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from config.dependencies import SANDBOX_DEPENDENCIES
from sandbox.sandbox import StreamingSandboxSession


class TestSandboxSessionSpecification:
    def test_copy_dir_from_runtime_recursively_preserves_relative_paths(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from sandbox.sandbox import StreamingSandboxSession

        session = object.__new__(StreamingSandboxSession)
        session.verbose = False
        copied = []
        executed_commands = []
        monkeypatch.setattr(
            session,
            "execute_command",
            lambda command: (
                executed_commands.append(command),
                SimpleNamespace(
                    stdout="/sandbox/output/reports/monthly/summary.csv\n/sandbox/output/charts/revenue.png\n"
                ),
            )[1],
        )
        monkeypatch.setattr(session, "copy_from_runtime", lambda source, dest: copied.append((source, dest)))

        session.copy_dir_from_runtime("/sandbox/output", str(tmp_path))

        assert len(executed_commands) == 1
        assert executed_commands[0].startswith('sh -c "find ')
        assert copied == [
            ("/sandbox/output/reports/monthly/summary.csv", str(tmp_path / "reports" / "monthly" / "summary.csv")),
            ("/sandbox/output/charts/revenue.png", str(tmp_path / "charts" / "revenue.png")),
        ]

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

        from sandbox.sandbox import ensure_cached_sandbox_image

        mock_client = MagicMock()
        mock_client.images.get.side_effect = ImageNotFound("missing")
        mock_from_env.return_value = mock_client

        image = ensure_cached_sandbox_image()

        assert image.startswith("adas-sandbox:")
        assert mock_client.images.build.call_args.kwargs["tag"] == image
        assert mock_client.images.build.call_args.kwargs["buildargs"] == {
            "ADAS_SANDBOX_DEPENDENCIES": " ".join(SANDBOX_DEPENDENCIES)
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
        from sandbox.sandbox import ensure_cached_sandbox_image

        mock_client = MagicMock()
        from docker.errors import ImageNotFound

        mock_client.images.get.side_effect = ImageNotFound("missing")
        image = ensure_cached_sandbox_image(client=mock_client)

        assert image.startswith("adas-sandbox:")
        assert mock_client.images.build.call_args.kwargs["tag"] == image
        assert mock_client.images.build.call_args.kwargs["buildargs"] == {
            "ADAS_SANDBOX_DEPENDENCIES": " ".join(SANDBOX_DEPENDENCIES)
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
        assert "/sandbox/workspace/config" in copied_dirs
        assert "/sandbox/workspace/meta_system" in copied_dirs

        copied_files = {call.args[1] for call in mock_session.copy_to_runtime.call_args_list}
        assert "/sandbox/workspace/requirements.txt" in copied_files


class TestSetupSandboxUtilities:
    def test_setup_manifest_currentness_tracks_task_spec_contents(self, tmp_path):
        import hashlib
        import json

        from create_setup import setup_manifest_is_current

        task_spec = tmp_path / "task.json"
        task_spec.write_text('{"name": "first"}', encoding="utf-8")
        digest = hashlib.sha256(task_spec.read_bytes()).hexdigest()
        (tmp_path / "setup_manifest.json").write_text(
            json.dumps({"fixture_lifecycle_version": 1, "files": {"task.json": digest}}), encoding="utf-8"
        )
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
        manifest_path.write_text(
            json.dumps({"fixture_lifecycle_version": 1, "files": {"task.json": lf_hash}}), encoding="utf-8"
        )

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
        (tmp_path / "setup_manifest.json").write_text(
            json.dumps({"fixture_lifecycle_version": 1, "files": hashes}), encoding="utf-8"
        )
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

    def test_copy_task_setup_to_sandbox_stages_declared_documentation(self, tmp_path):
        from adas_core.environment import SANDBOX_WORKSPACE_DIR
        from sandbox.sandbox import copy_task_setup_to_sandbox

        task_dir = tmp_path / "task"
        task_dir.mkdir()
        docs_dir = task_dir / "docs"
        docs_dir.mkdir()
        doc_path = docs_dir / "reference.md"
        doc_path.write_text("# Reference", encoding="utf-8")
        spec_path = task_dir / "task.json"
        spec_path.write_text("{}", encoding="utf-8")
        mock_session = MagicMock()

        copy_task_setup_to_sandbox(mock_session, task_dir, spec_path, ["docs/reference.md"])

        assert (str(doc_path), f"{SANDBOX_WORKSPACE_DIR}/docs/reference.md") in [
            call.args for call in mock_session.copy_to_runtime.call_args_list
        ]

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

    def test_run_sandbox_preflight_passes_runtime_profile(self):
        from sandbox.sandbox import run_sandbox_preflight

        mock_session = MagicMock()
        mock_session.execute_command.return_value = MagicMock(
            exit_code=0, stdout="Preflight verification passed", stderr=""
        )
        profile_json = '{"overrides":{"api":{"provider":"external","url":"https://example.test"}}}'

        assert run_sandbox_preflight(mock_session, "/sandbox/task", profile_json)
        command = mock_session.execute_command.call_args.args[0]
        assert "--runtime-profile" in command
        assert "https://example.test" in command

    def test_stage_runtime_profile_sources_preserves_empty_directories(self, tmp_path):
        from adas_core.runtime_resources import RuntimeResourceProfile
        from invoke_target import stage_runtime_profile_sources

        source = tmp_path / "source"
        (source / "empty" / "nested").mkdir(parents=True)
        profile = RuntimeResourceProfile.model_validate(
            {"overrides": {"docs": {"provider": "local_file", "source": str(source)}}}
        )
        session = MagicMock()

        staged = stage_runtime_profile_sources(session, profile)

        assert staged.overrides["docs"].source == "/sandbox/workspace/runtime_resources/docs"
        commands = [call.args[0] for call in session.execute_command.call_args_list]
        assert any("runtime_resources/docs/empty/nested" in command for command in commands)

    def test_run_sandbox_preflight_failure_exit_code(self):
        from sandbox.sandbox import run_sandbox_preflight

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.exit_code = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error importing dependency"
        mock_session.execute_command.return_value = mock_result

        assert run_sandbox_preflight(mock_session) is False

    def test_default_preflight_does_not_install_validation_packages(self, monkeypatch, tmp_path):
        import sandbox.run_preflight as run_preflight

        manifest_path = tmp_path / "setup_manifest.json"
        manifest_path.write_text('{"required_packages": ["setup-package"]}', encoding="utf-8")
        task_spec = MagicMock()
        installed_packages = MagicMock()

        monkeypatch.setattr(sys, "argv", ["run_preflight.py", "--task-dir", str(tmp_path)])
        with (
            patch.object(run_preflight.TaskSpec, "from_file", return_value=task_spec),
            patch.object(run_preflight, "validation_requirements") as validation_requirements,
            patch.object(run_preflight, "ensure_packages_installed", installed_packages),
            patch.object(run_preflight, "fixture_paths_for_profile", return_value=[]),
            patch.object(run_preflight, "isolated_case_workspace", return_value=contextlib.nullcontext({})),
            patch.object(run_preflight, "process_fixture_lifecycle", return_value=contextlib.nullcontext()),
            patch.object(run_preflight, "external_url_overrides", return_value=contextlib.nullcontext()),
            patch.object(run_preflight, "run_preflight_check", return_value=(True, "ok")),
        ):
            assert run_preflight.main() == 0

        validation_requirements.assert_not_called()
        installed_packages.assert_called_once_with(["setup-package"])
