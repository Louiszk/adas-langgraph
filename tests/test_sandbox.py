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
