import os
import codecs
import glob
from config import settings
from llm_sandbox import create_session, SandboxBackend

class StreamingSandboxSession:
    def __init__(self, image=None, dockerfile=None, keep_template=False, 
                 stream=True, verbose=True, runtime_configs=None, 
                 container_type='auto', **kwargs):
        self.verbose = verbose
        self.session = None
        
        # Determine which container technology backend to use
        backend = None
        if container_type == 'docker':
            if not check_docker_running():
                raise RuntimeError("Docker is selected but not running or available.")
            backend = SandboxBackend.DOCKER
        elif container_type == 'podman':
            if not check_podman_running():
                raise RuntimeError("Podman is selected but not running or available.")
            backend = SandboxBackend.PODMAN
        elif container_type == 'auto':
            if check_docker_running():
                backend = SandboxBackend.DOCKER
            elif check_podman_running():
                backend = SandboxBackend.PODMAN
            else:
                raise RuntimeError("Neither Docker nor Podman are running or available. Please install and start one.")
        else:
             raise ValueError(f"Unknown container type: {container_type}")

        if self.verbose:
            print(f"Using {backend.value} as container runtime")

        # Prepare the arguments for the create_session factory
        session_kwargs = {
            "image": image,
            "dockerfile": dockerfile,
            "keep_template": keep_template,
            "verbose": verbose,
            "runtime_configs": runtime_configs,
            "stream": stream,
            **kwargs
        }

        # If using Podman, check for our custom isolated socket and add it to the arguments
        if backend == SandboxBackend.PODMAN:
            socket_path = os.environ.get("ADAS_PODMAN_SOCKET")
            if socket_path:
                print(f"--> Connecting Podman client to isolated service socket: {socket_path}")
                # 'base_url' is the kwarg the internal PodmanClient uses for the socket
                session_kwargs['base_url'] = socket_path
            else:
                print("--> WARNING: ADAS_PODMAN_SOCKET not set. Connecting to default Podman service.")
        
        # Use the library's factory to create the correct session instance
        self.session = create_session(backend=backend, **session_kwargs)

    def open(self):
        if not self.session:
            raise RuntimeError("Session was not initialized correctly.")
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
            print(f"Exception during copying to runtime: {repr(e)}")
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
                print(f"Warning: Source directory '{src_dir}' not found, skipping copy.")
            return

        self.execute_command(f"mkdir -p {dest_dir}")

        files_to_copy = glob.glob(os.path.join(src_dir, pattern))
        
        if not files_to_copy:
            if self.verbose:
                print(f"No files found in '{src_dir}' matching pattern '{pattern}'.")
            return
            
        if self.verbose:
            print(f"Copying {len(files_to_copy)} files from '{src_dir}' to sandbox '{dest_dir}'...")

        for src_path in files_to_copy:
            if os.path.isfile(src_path):
                filename = os.path.basename(src_path)
                dest_path = os.path.join(dest_dir, filename).replace('\\', '/')
                self.copy_to_runtime(src_path, dest_path)

    def copy_dir_from_runtime(self, src_dir: str, dest_dir: str, pattern: str = "*"):
        """
        Copies files matching a glob pattern from a source directory inside the sandbox
        to a local destination directory.
        """
        os.makedirs(dest_dir, exist_ok=True)

        full_path_pattern = os.path.join(src_dir, pattern).replace('\\', '/')
        command = f'sh -c "ls -d {full_path_pattern} 2>/dev/null"'
        command_output = self.execute_command(command)
        file_list_str = str(command_output.stdout) if command_output and command_output.stdout else ""

        if not file_list_str.strip():
            if self.verbose:
                print(f"No files found in sandbox '{src_dir}' matching pattern '{pattern}'.")
            return

        sandbox_paths = [path for path in file_list_str.strip().split('\n') if path]
        
        if self.verbose:
            print(f"Copying {len(sandbox_paths)} files from sandbox '{src_dir}' to '{dest_dir}'...")

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
    except (ImportError, docker.errors.DockerException):
        return False

def check_podman_running():
    """Check if Podman is running and available."""
    if os.environ.get("ADAS_PODMAN_SOCKET"):
        return True
    
    try:
        from podman import PodmanClient
        client = PodmanClient()
        if client.info()["host"]["remoteSocket"] is None:
            return False
        return True
    except (ImportError, Exception):
        return False
    
def setup_sandbox_environment(session, reinstall=False):
    """Set up the sandbox environment with required files and dependencies."""
    print("Setting up sandbox environment...")
    
    session.execute_command("mkdir -p /sandbox/workspace/materialized_meta_system")
    session.execute_command("mkdir -p /sandbox/workspace/adas_core")
    session.execute_command("mkdir -p /sandbox/workspace/generated_systems")
    session.execute_command("mkdir -p /sandbox/workspace/config")
    session.execute_command("rm -rf /sandbox/workspace/data/input")
    session.execute_command("rm -rf /sandbox/workspace/data/output")
    session.execute_command("rm -rf /sandbox/workspace/target_metrics")
    
    session.execute_command("mkdir -p /sandbox/workspace/data/output")
    session.copy_dir_to_runtime(
        src_dir="data/input", 
        dest_dir="/sandbox/workspace/data/input", 
        pattern="*"
    )
    
    # Copy core framework files
    required_files = [
        "adas_core/virtual_agentic_system.py",
        "adas_core/decorator_logic.py",
        "adas_core/llm_wrapper.py",
        "adas_core/materialize.py",
        "adas_core/helpers.py",
        "materialized_meta_system/MetaSystem.py",
        "config/settings.py",
        ".env"
    ] 
    
    copy_paths = [(path, f"/sandbox/workspace/{path}") for path in required_files] + [
        ("sandbox/run_meta.py", "/sandbox/workspace/run_meta.py"),
        ("sandbox/run_target.py", "/sandbox/workspace/run_target.py")
    ]
    
    for src_path, dest_path in copy_paths:
        if os.path.exists(src_path):
            session.copy_to_runtime(src_path, dest_path)
        else:
            print(f"Warning: Required file {src_path} not found")

    print("Searching for existing agentic systems to copy to sandbox...")
    session.copy_dir_to_runtime(
        src_dir="generated_systems",
        dest_dir="/sandbox/workspace/generated_systems",
        pattern="*.py"
    )
    session.copy_dir_to_runtime(
        src_dir="generated_systems",
        dest_dir="/sandbox/workspace/generated_systems",
        pattern="*.pkl"
    )
    
    if reinstall:
        print("Installing dependencies in sandbox...")
        session.execute_command(f"pip install {' '.join(settings.dependencies)}")
    
    print("Sandbox environment set up successfully!")
    return True