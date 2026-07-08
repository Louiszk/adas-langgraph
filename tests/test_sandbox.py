"""
Specification tests for sandbox runtime initialization and configuration.
"""

import pytest
from unittest.mock import patch
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

    @patch("sandbox.sandbox.check_docker_running", return_value=False)
    @patch("sandbox.sandbox.check_podman_running", return_value=False)
    def test_auto_raises_when_neither_container_engine_available(self, mock_podman, mock_docker):
        """Contract: If neither Docker nor Podman are available, auto must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Neither Docker nor Podman are running"):
            StreamingSandboxSession(container_type="auto")
