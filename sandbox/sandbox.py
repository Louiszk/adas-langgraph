import codecs
import glob
import hashlib
import os
from pathlib import Path

from llm_sandbox import SandboxBackend, create_session

from adas_core.logging_config import get_logger
from config import settings

logger = get_logger("sandbox")

_SANDBOX_DOCKERFILE = Path(__file__).with_name("Dockerfile")
_CACHED_IMAGE_REPOSITORY = "adas-sandbox"


def _cached_sandbox_image_tag() -> str:
    """Return an image tag that changes when the runtime or its dependencies do."""
    fingerprint = "\n".join(
        [
            _SANDBOX_DOCKERFILE.read_text(encoding="utf-8"),
            *settings.dependencies,
        ]
    )
    return f"{_CACHED_IMAGE_REPOSITORY}:{hashlib.sha256(fingerprint.encode()).hexdigest()[:16]}"


def ensure_cached_sandbox_image(client=None) -> str:
    """Build the default sandbox image once, then reuse it for future runs with Docker or Podman."""
    image_tag = _cached_sandbox_image_tag()

    if client is None:
        if check_docker_running():
            import docker

            client = docker.from_env()
        elif check_podman_running():
            socket_path = os.environ.get("ADAS_PODMAN_SOCKET")
            from podman import PodmanClient  # type: ignore

            client = PodmanClient(base_url=socket_path) if socket_path else PodmanClient()  # type: ignore
        else:
            import docker

            client = docker.from_env()

    try:
        client.images.get(image_tag)
        logger.info("Using cached sandbox image %s", image_tag)
    except _image_not_found_errors():
        logger.info("Building sandbox image %s (this happens once per dependency version)", image_tag)
        client.images.build(
            path=str(_SANDBOX_DOCKERFILE.parent),
            dockerfile=_SANDBOX_DOCKERFILE.name,
            tag=image_tag,
            buildargs={"ADAS_SANDBOX_DEPENDENCIES": " ".join(settings.dependencies)},
        )
    return image_tag


def _image_not_found_errors() -> tuple[type[Exception], ...]:
    """Return the Docker/Podman exceptions that specifically mean an image is absent."""
    from docker.errors import ImageNotFound as DockerImageNotFound

    errors: list[type[Exception]] = [DockerImageNotFound]
    try:
        from podman.errors.exceptions import ImageNotFound as PodmanImageNotFound  # type: ignore

        errors.append(PodmanImageNotFound)
    except ImportError:
        pass
    return tuple(errors)


class StreamingSandboxSession:
    def __init__(
        self,
        image=None,
        dockerfile=None,
        stream=True,
        verbose=True,
        runtime_configs=None,
        container_type="auto",
        **kwargs,
    ):
        self.verbose = verbose
        self.session = None
        self._uses_cached_image = image is None and dockerfile is None

        # The cached image already contains the complete Python environment.
        # Skipping llm-sandbox's per-container venv/pip bootstrap avoids a
        # second package-management step on every isolated run.
        skip_environment_setup = kwargs.pop("skip_environment_setup", self._uses_cached_image)

        # Determine which container technology backend to use
        backend = None
        if container_type == "docker":
            if not check_docker_running():
                raise RuntimeError("Docker is selected but not running or available.")
            backend = SandboxBackend.DOCKER
        elif container_type == "podman":
            if not check_podman_running():
                raise RuntimeError("Podman is selected but not running or available.")
            backend = SandboxBackend.PODMAN
        elif container_type == "auto":
            if check_docker_running():
                backend = SandboxBackend.DOCKER
            elif check_podman_running():
                backend = SandboxBackend.PODMAN
            else:
                raise RuntimeError("Neither Docker nor Podman are running or available. Please install and start one.")
        else:
            raise ValueError(f"Unknown container type: {container_type}")

        if self.verbose:
            logger.info(f"Using {backend.value} as container runtime")

        # Prepare the arguments for the create_session factory
        session_kwargs = {
            "image": image,
            "dockerfile": dockerfile,
            "verbose": verbose,
            "runtime_configs": runtime_configs,
            "stream": stream,
            "skip_environment_setup": skip_environment_setup,
            **kwargs,
        }

        # If using Podman, check for our custom isolated socket and add it to the arguments
        if backend == SandboxBackend.PODMAN:
            socket_path = os.environ.get("ADAS_PODMAN_SOCKET")
            if socket_path:
                logger.info(f"Connecting Podman client to isolated service socket: {socket_path}")
                # 'base_url' is the kwarg the internal PodmanClient uses for the socket
                session_kwargs["base_url"] = socket_path
            else:
                logger.warning("ADAS_PODMAN_SOCKET not set. Connecting to default Podman service.")

        # Use the library's factory to create the correct session instance
        self.session = create_session(backend=backend, **session_kwargs)

    def open(self):
        if not self.session:
            raise RuntimeError("Session was not initialized correctly.")
        if self._uses_cached_image:
            # Each run receives a fresh container, while this image (including
            # Python packages) persists in the container engine's local image cache.
            self.session.config.image = ensure_cached_sandbox_image(client=getattr(self.session, "client", None))
        return self.session.open()

    def close(self):
        if self.session:
            return self.session.close()

    def execute_command(self, command, workdir=None):
        if not self.session:
            raise RuntimeError("Session is not open.")
        return self.session.execute_command(command, workdir)

    def copy_to_runtime(self, src, dest):
        if not self.session:
            raise RuntimeError("Session is not open.")
        try:
            return self.session.copy_to_runtime(src, dest)
        except Exception as e:
            logger.error(f"Exception during copying to runtime: {e!r}")
            return None

    def copy_from_runtime(self, src, dest):
        if not self.session:
            raise RuntimeError("Session is not open.")
        return self.session.copy_from_runtime(src, dest)

    def execute_command_streaming(self, command, workdir=None):
        if not self.session or not self.session.container:
            raise RuntimeError("Session is not open or container is not running.")

        kwargs = {"stream": True, "tty": True}
        if workdir:
            kwargs["workdir"] = workdir

        _, output_stream = self.session.container.exec_run(command, **kwargs)

        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        for chunk in output_stream:
            # buffer incomplete bytes and yield valid strings
            yield decoder.decode(chunk, final=False)

        yield decoder.decode(b"", final=True)

    def copy_dir_to_runtime(self, src_dir: str, dest_dir: str, pattern: str = "*"):
        """
        Copies files matching a glob pattern from a local source directory
        to a destination directory inside the sandbox.
        """
        if not os.path.isdir(src_dir):
            if self.verbose:
                logger.warning(f"Source directory '{src_dir}' not found, skipping copy.")
            return

        self.execute_command(f"mkdir -p {dest_dir}")

        files_to_copy = glob.glob(os.path.join(src_dir, pattern))

        if not files_to_copy:
            if self.verbose:
                logger.info(f"No files found in '{src_dir}' matching pattern '{pattern}'.")
            return

        if self.verbose:
            logger.info(f"Copying {len(files_to_copy)} files from '{src_dir}' to sandbox '{dest_dir}'...")

        for src_path in files_to_copy:
            if os.path.isfile(src_path):
                filename = os.path.basename(src_path)
                dest_path = os.path.join(dest_dir, filename).replace("\\", "/")
                self.copy_to_runtime(src_path, dest_path)

    def copy_dir_from_runtime(self, src_dir: str, dest_dir: str, pattern: str = "*"):
        """
        Copies files matching a glob pattern from a source directory inside the sandbox
        to a local destination directory.
        """
        os.makedirs(dest_dir, exist_ok=True)

        full_path_pattern = os.path.join(src_dir, pattern).replace("\\", "/")
        command = f'sh -c "ls -d {full_path_pattern} 2>/dev/null"'
        command_output = self.execute_command(command)
        file_list_str = str(command_output.stdout) if command_output and command_output.stdout else ""

        if not file_list_str.strip():
            if self.verbose:
                logger.info(f"No files found in sandbox '{src_dir}' matching pattern '{pattern}'.")
            return

        sandbox_paths = [path for path in file_list_str.strip().split("\n") if path]

        if self.verbose:
            logger.info(f"Copying {len(sandbox_paths)} files from sandbox '{src_dir}' to '{dest_dir}'...")

        for src_path_in_sandbox in sandbox_paths:
            filename = os.path.basename(src_path_in_sandbox)
            dest_path_on_host = os.path.join(dest_dir, filename)
            self.copy_from_runtime(src_path_in_sandbox, dest_path_on_host)


def check_docker_running():
    """Check if Docker is running and available."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except (ImportError, Exception):
        return False


def check_podman_running():
    """Check if Podman is running and available."""
    if os.environ.get("ADAS_PODMAN_SOCKET"):
        return True

    try:
        from podman import PodmanClient  # type: ignore

        client = PodmanClient()  # type: ignore
        if client.info()["host"]["remoteSocket"] is None:
            return False
        return True
    except (ImportError, Exception):
        return False


def setup_sandbox_environment(session, reinstall=False):
    """Set up the sandbox environment with required files and dependencies."""
    logger.info("Setting up sandbox environment...")

    session.execute_command("mkdir -p /sandbox/workspace/materialized_meta_system")
    session.execute_command("mkdir -p /sandbox/workspace/adas_core")
    session.execute_command("mkdir -p /sandbox/workspace/generated_systems")
    session.execute_command("mkdir -p /sandbox/workspace/config")
    session.execute_command("rm -rf /sandbox/workspace/data/input")
    session.execute_command("rm -rf /sandbox/workspace/data/output")
    session.execute_command("rm -rf /sandbox/workspace/target_metrics")

    session.execute_command("mkdir -p /sandbox/workspace/data/output")
    session.copy_dir_to_runtime(src_dir="data/input", dest_dir="/sandbox/workspace/data/input", pattern="*")

    # Copy core framework files
    required_files = [
        "adas_core/ast_parser.py",
        "adas_core/virtual_agentic_system.py",
        "adas_core/decorator_logic.py",
        "adas_core/llm_wrapper.py",
        "adas_core/materialize.py",
        "adas_core/helpers.py",
        "adas_core/logging_config.py",
        "materialized_meta_system/MetaSystem.py",
        "config/settings.py",
        ".env",
    ]

    copy_paths = [(path, f"/sandbox/workspace/{path}") for path in required_files] + [
        ("sandbox/run_meta.py", "/sandbox/workspace/run_meta.py"),
        ("sandbox/run_target.py", "/sandbox/workspace/run_target.py"),
    ]

    for src_path, dest_path in copy_paths:
        if os.path.exists(src_path):
            session.copy_to_runtime(src_path, dest_path)
        else:
            logger.warning(f"Required file {src_path} not found")

    logger.info("Searching for existing agentic systems to copy to sandbox...")
    session.copy_dir_to_runtime(
        src_dir="generated_systems",
        dest_dir="/sandbox/workspace/generated_systems",
        pattern="*.py",
    )
    session.copy_dir_to_runtime(
        src_dir="generated_systems",
        dest_dir="/sandbox/workspace/generated_systems",
        pattern="*.pkl",
    )

    check_deps = session.execute_command("python -c 'import dill, langgraph'")
    deps_stderr = getattr(check_deps, "stderr", "") if check_deps else ""
    deps_exit_code = getattr(check_deps, "exit_code", 1) if check_deps else 1
    deps_missing = check_deps is None or deps_exit_code != 0 or "Error" in deps_stderr or "Traceback" in deps_stderr

    if reinstall or deps_missing:
        logger.info("Installing dependencies in sandbox...")
        session.execute_command(f"pip install {' '.join(settings.dependencies)}")

    logger.info("Sandbox environment set up successfully!")
    return True
