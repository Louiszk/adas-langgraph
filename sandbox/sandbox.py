import codecs
import glob
import hashlib
import os
import posixpath
import shlex
from pathlib import Path

from llm_sandbox import SandboxBackend, create_session

from adas_core.environment import (
    SANDBOX_GENERATED_SYSTEMS_DIR,
    SANDBOX_TARGET_METRICS_DIR,
    SANDBOX_TASK_SETUP_DIR,
    SANDBOX_WORKSPACE_DIR,
)
from adas_core.exceptions import (
    SandboxConfigurationError,
    SandboxRuntimeUnavailableError,
    SandboxSessionError,
)
from config.dependencies import SANDBOX_DEPENDENCIES
from config.logging import get_logger

logger = get_logger("sandbox")


_SANDBOX_DOCKERFILE = Path(__file__).with_name("Dockerfile")
_CACHED_IMAGE_REPOSITORY = "adas-sandbox"


def _cached_sandbox_image_tag() -> str:
    """Return an image tag that changes when the runtime or its dependencies do."""
    fingerprint = "\n".join(
        [
            _SANDBOX_DOCKERFILE.read_text(encoding="utf-8"),
            *SANDBOX_DEPENDENCIES,
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
            buildargs={"ADAS_SANDBOX_DEPENDENCIES": " ".join(SANDBOX_DEPENDENCIES)},
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
                raise SandboxRuntimeUnavailableError("Docker is selected but not running or available.")
            backend = SandboxBackend.DOCKER
        elif container_type == "podman":
            if not check_podman_running():
                raise SandboxRuntimeUnavailableError("Podman is selected but not running or available.")
            backend = SandboxBackend.PODMAN
        elif container_type == "auto":
            if check_docker_running():
                backend = SandboxBackend.DOCKER
            elif check_podman_running():
                backend = SandboxBackend.PODMAN
            else:
                raise SandboxRuntimeUnavailableError(
                    "Neither Docker nor Podman are running or available. Please install and start one."
                )
        else:
            raise SandboxConfigurationError(f"Unknown container type: {container_type}")

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
            raise SandboxSessionError("Session was not initialized correctly.")
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
            raise SandboxSessionError("Session is not open.")
        return self.session.execute_command(command, workdir)

    def copy_to_runtime(self, src, dest):
        if not self.session:
            raise SandboxSessionError("Session is not open.")
        dest_dir = os.path.dirname(str(dest).replace("\\", "/"))
        if dest_dir:
            result = self.session.execute_command(f"mkdir -p '{dest_dir}'")
            exit_code = getattr(result, "exit_code", None)
            if isinstance(exit_code, int) and exit_code != 0:
                raise SandboxSessionError(f"Could not create runtime directory '{dest_dir}' (exit code {exit_code}).")
        return self.session.copy_to_runtime(src, dest)

    def copy_from_runtime(self, src, dest):
        if not self.session:
            raise SandboxSessionError("Session is not open.")
        return self.session.copy_from_runtime(src, dest)

    def execute_command_streaming(self, command, workdir=None, environment=None):
        if not self.session or not self.session.container:
            raise SandboxSessionError("Session is not open or container is not running.")

        kwargs = {"stream": True, "tty": True}
        if workdir:
            kwargs["workdir"] = workdir
        if environment:
            kwargs["environment"] = environment

        cmd_to_run = ["/bin/sh", "-c", command] if isinstance(command, str) else command
        _, output_stream = self.session.container.exec_run(cmd_to_run, **kwargs)

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

        result = self.execute_command(f"mkdir -p {dest_dir}")
        exit_code = getattr(result, "exit_code", None)
        if isinstance(exit_code, int) and exit_code != 0:
            raise SandboxSessionError(f"Could not create runtime directory '{dest_dir}' (exit code {exit_code}).")

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
        Recursively copy matching files from a sandbox directory while preserving
        their paths relative to ``src_dir``.
        """
        os.makedirs(dest_dir, exist_ok=True)

        normalized_src_dir = src_dir.replace("\\", "/").rstrip("/")
        command = f'sh -c "find {shlex.quote(normalized_src_dir)} -type f -name {shlex.quote(pattern)} 2>/dev/null"'
        command_output = self.execute_command(command)
        file_list_str = str(command_output.stdout) if command_output and command_output.stdout else ""

        if not file_list_str.strip():
            if self.verbose:
                logger.info(f"No files found in sandbox '{src_dir}' matching pattern '{pattern}'.")
            return

        sandbox_paths = [path for path in file_list_str.splitlines() if path]

        if self.verbose:
            logger.info(f"Copying {len(sandbox_paths)} files from sandbox '{src_dir}' to '{dest_dir}'...")

        for src_path_in_sandbox in sandbox_paths:
            relative_path = posixpath.relpath(src_path_in_sandbox, normalized_src_dir)
            if relative_path == ".." or relative_path.startswith("../"):
                logger.warning("Skipping sandbox artifact outside requested directory: %s", src_path_in_sandbox)
                continue
            dest_path_on_host = os.path.join(dest_dir, *relative_path.split("/"))
            os.makedirs(os.path.dirname(dest_path_on_host), exist_ok=True)
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
        return client.info()["host"]["remoteSocket"] is not None
    except (ImportError, Exception):
        return False


def setup_sandbox_environment(session, reinstall=False):
    """Set up the sandbox environment with required files and dependencies."""
    logger.info("Setting up sandbox environment...")

    def run_checked(command: str) -> None:
        result = session.execute_command(command)
        exit_code = getattr(result, "exit_code", None)
        if isinstance(exit_code, int) and exit_code != 0:
            raise SandboxSessionError(f"Sandbox command failed (exit code {exit_code}): {command}")

    try:
        run_checked(f"mkdir -p {SANDBOX_WORKSPACE_DIR}/meta_system")
        run_checked(f"mkdir -p {SANDBOX_WORKSPACE_DIR}/adas_core")
        run_checked(f"mkdir -p {SANDBOX_GENERATED_SYSTEMS_DIR}")
        run_checked(f"mkdir -p {SANDBOX_WORKSPACE_DIR}/config")
        run_checked(f"rm -rf {SANDBOX_TARGET_METRICS_DIR}")

        # Copy config, meta-system, and core framework files
        session.copy_dir_to_runtime(src_dir="config", dest_dir=f"{SANDBOX_WORKSPACE_DIR}/config", pattern="*.py")
        session.copy_dir_to_runtime(
            src_dir="meta_system", dest_dir=f"{SANDBOX_WORKSPACE_DIR}/meta_system", pattern="*.py"
        )
        session.copy_dir_to_runtime(src_dir="adas_core", dest_dir=f"{SANDBOX_WORKSPACE_DIR}/adas_core", pattern="*.py")

        # Copy environment and runner files
        additional_files = [
            ("requirements.txt", f"{SANDBOX_WORKSPACE_DIR}/requirements.txt"),
            (".env", f"{SANDBOX_WORKSPACE_DIR}/.env"),
            ("sandbox/run_meta.py", f"{SANDBOX_WORKSPACE_DIR}/run_meta.py"),
            ("sandbox/run_target.py", f"{SANDBOX_WORKSPACE_DIR}/run_target.py"),
            ("sandbox/run_setup.py", f"{SANDBOX_WORKSPACE_DIR}/run_setup.py"),
            ("sandbox/run_preflight.py", f"{SANDBOX_WORKSPACE_DIR}/run_preflight.py"),
        ]

        for src_path, dest_path in additional_files:
            if not os.path.exists(src_path):
                raise SandboxSessionError(f"Required sandbox file not found: {src_path}")
            session.copy_to_runtime(src_path, dest_path)

        logger.info("Searching for existing agentic systems to copy to sandbox...")
        session.copy_dir_to_runtime(src_dir="generated_systems", dest_dir=SANDBOX_GENERATED_SYSTEMS_DIR, pattern="*.py")
        session.copy_dir_to_runtime(
            src_dir="generated_systems", dest_dir=SANDBOX_GENERATED_SYSTEMS_DIR, pattern="*.pkl"
        )

        check_deps = session.execute_command("python -c 'import dill, langgraph'")
        deps_stderr = getattr(check_deps, "stderr", "") if check_deps else ""
        deps_exit_code = getattr(check_deps, "exit_code", 1) if check_deps else 1
        deps_missing = check_deps is None or deps_exit_code != 0 or "Error" in deps_stderr or "Traceback" in deps_stderr

        if reinstall or deps_missing:
            logger.info("Installing dependencies in sandbox...")
            run_checked(f"pip install {' '.join(SANDBOX_DEPENDENCIES)}")

        logger.info("Sandbox environment set up successfully!")
        return True
    except Exception as exc:
        logger.error("Sandbox environment setup failed: %s", exc)
        return False


def copy_task_setup_to_sandbox(
    session: StreamingSandboxSession,
    task_dir: Path | str,
    task_spec_path: Path | str,
    additional_documentation: list[str] | None = None,
) -> str:
    """Copy the visible, frozen setup artifacts into a design sandbox."""
    task_dir_path = Path(task_dir)
    spec_path = Path(task_spec_path)
    runtime_task_dir = SANDBOX_TASK_SETUP_DIR
    session.execute_command(f"mkdir -p '{runtime_task_dir}'")

    # Pre-create all necessary subdirectories
    subdirs = {
        source_path.relative_to(task_dir_path).parent.as_posix()
        for source_path in task_dir_path.rglob("*")
        if source_path.is_file()
        and "__pycache__" not in source_path.parts
        and source_path.relative_to(task_dir_path).parent != Path(".")
    }
    for subdir in sorted(subdirs):
        session.execute_command(f"mkdir -p '{runtime_task_dir}/{subdir}'")

    for source_path in task_dir_path.rglob("*"):
        if source_path.is_file() and "__pycache__" not in source_path.parts:
            relative_path = source_path.relative_to(task_dir_path).as_posix()
            session.copy_to_runtime(str(source_path), f"{runtime_task_dir}/{relative_path}")
    # The sandbox entry point always loads the explicit, visible TaskSpec from
    # this stable path, regardless of the host file's chosen name.
    session.copy_to_runtime(str(spec_path), f"{runtime_task_dir}/task.json")

    for doc_path in additional_documentation or []:
        relative_path = Path(doc_path)
        source_candidates = (task_dir_path / relative_path, Path.cwd() / relative_path)
        source_path = next((path for path in source_candidates if path.is_file()), None)
        if source_path is None:
            logger.warning("Referenced documentation file '%s' was not found for sandbox staging.", doc_path)
            continue
        destination = f"{SANDBOX_WORKSPACE_DIR}/{relative_path.as_posix()}"
        session.copy_to_runtime(str(source_path), destination)
    return runtime_task_dir


def run_sandbox_preflight(
    session: StreamingSandboxSession,
    runtime_task_dir: str = SANDBOX_TASK_SETUP_DIR,
    runtime_profile_json: str | None = None,
    require_validation: bool = False,
) -> bool:
    """Install frozen setup requirements and validate them inside the sandbox."""
    command = f"python3 {SANDBOX_WORKSPACE_DIR}/run_preflight.py --task-dir {runtime_task_dir}"
    if require_validation:
        command += " --require-validation"
    if runtime_profile_json:
        command += f" --runtime-profile {shlex.quote(runtime_profile_json)}"
    result = session.execute_command(command)
    exit_code = getattr(result, "exit_code", None)
    output_str = str(getattr(result, "stdout", "") or "")
    stderr_str = str(getattr(result, "stderr", "") or "")
    combined = (output_str + "\n" + stderr_str).strip()

    if "Preflight check failed" in combined or "Preflight verification passed" not in combined:
        logger.error("Sandbox preflight verification failed: %s", combined or result)
        return False

    if exit_code is not None and exit_code != 0:
        logger.error("Sandbox preflight verification failed with exit code %s: %s", exit_code, combined or result)
        return False

    logger.info("Sandbox preflight verification passed.")
    return True
