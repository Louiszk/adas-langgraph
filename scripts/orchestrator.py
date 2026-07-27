from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import docker
    from docker.errors import DockerException
except ImportError:
    docker = None
    DockerException = Exception


class ContainerManager:
    """Manages Docker/Podman base images, builder containers, and temporary images."""

    def __init__(self, container_type: str = "auto", unique_id: Optional[str] = None):
        self.container_type = container_type
        self.unique_id = unique_id or str(random.randint(10000, 99999))
        self.client = self._init_client()

    def _init_client(self) -> Any:
        if docker is None:
            return None

        socket_path = os.environ.get("ADAS_PODMAN_SOCKET")

        if self.container_type == "podman":
            if not socket_path:
                return None
            try:
                client = docker.DockerClient(base_url=socket_path)
                client.ping()
                return client
            except Exception:
                return None

        if self.container_type == "auto" and socket_path:
            try:
                client = docker.DockerClient(base_url=socket_path)
                client.ping()
                return client
            except Exception:
                pass

        if self.container_type in ("auto", "docker"):
            try:
                client = docker.from_env()
                client.ping()
                return client
            except Exception:
                return None

        return None

    def create_base_image(
        self,
        base_image_name: str,
        builder_image: str = "python:3.11-slim",
        core_packages: Optional[List[str]] = None,
    ) -> bool:
        """Create a custom base image with core dependencies installed if it does not exist."""
        if core_packages is None:
            core_packages = [
                "langgraph==1.2.9",
                "langchain_openai==1.4.1",
                "python-dotenv==1.2.2",
                "dill==0.3.9",
            ]

        if self.client is not None:
            try:
                self.client.images.get(base_image_name)
                print(f"--> Custom base image '{base_image_name}' already exists. Skipping creation.")
                return True
            except Exception:
                pass

        print(f"--> Custom base image '{base_image_name}' not found. Creating it...")
        if self.client is not None:
            try:
                self.client.images.pull(builder_image)
                container = self.client.containers.run(
                    builder_image,
                    command="sleep 3600",
                    detach=True,
                    name=f"adas-builder-{self.unique_id}",
                )
                print("--> Installing core dependencies in builder container...")
                install_cmd = f"pip install {' '.join(core_packages)}"
                exit_code, output = container.exec_run(install_cmd)
                if exit_code != 0:
                    output_str = output.decode("utf-8", errors="ignore") if isinstance(output, bytes) else str(output)
                    print(f"!!ERROR: pip install failed inside builder container:\n{output_str}", file=sys.stderr)
                    container.stop()
                    container.remove()
                    return False

                print(f"--> Committing container to create image '{base_image_name}'...")
                container.commit(repository=base_image_name)
                container.stop()
                container.remove()
                print("--> Custom base image created successfully.")
                return True
            except Exception as e:
                print(f"--> SDK build failed ({e}), falling back to CLI subprocess...")

        return self._create_base_image_cli(base_image_name, builder_image, core_packages)

    def _create_base_image_cli(self, base_image_name: str, builder_image: str, core_packages: List[str]) -> bool:
        cli = self._get_cli_cmd()
        cid_name = f"adas-builder-{self.unique_id}"
        subprocess.run([cli, "pull", builder_image], stdout=subprocess.DEVNULL, check=False)
        res = subprocess.run(
            [cli, "run", "-d", "--name", cid_name, builder_image, "sleep", "3600"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            print(f"!!ERROR: Failed to start builder container: {res.stderr}", file=sys.stderr)
            return False

        install_res = subprocess.run(
            [cli, "exec", cid_name, "pip", "install"] + core_packages,
            capture_output=True,
            text=True,
            check=False,
        )
        if install_res.returncode != 0:
            print(f"!!ERROR: pip install failed in builder container: {install_res.stderr}", file=sys.stderr)
            subprocess.run([cli, "rm", "-f", cid_name], stdout=subprocess.DEVNULL, check=False)
            return False

        subprocess.run([cli, "commit", cid_name, base_image_name], check=False)
        subprocess.run([cli, "rm", "-f", cid_name], stdout=subprocess.DEVNULL, check=False)
        print("--> Custom base image created successfully via CLI.")
        return True

    def create_temp_image(self, base_image_name: str, temp_image_name: str, packages: List[str]) -> bool:
        """Create a temporary image with additional packages installed on top of the base image."""
        if not packages:
            return True

        print(f"--> Creating temporary image '{temp_image_name}' with custom dependencies: {packages}")
        if self.client is not None:
            try:
                container = self.client.containers.run(
                    base_image_name,
                    command="sleep 3600",
                    detach=True,
                    name=f"adas-temp-builder-{self.unique_id}-{random.randint(1000, 9999)}",
                )
                install_cmd = f"pip install {' '.join(packages)}"
                exit_code, output = container.exec_run(install_cmd)
                if exit_code != 0:
                    output_str = output.decode("utf-8", errors="ignore") if isinstance(output, bytes) else str(output)
                    print(f"!!ERROR: Failed to install custom dependencies:\n{output_str}", file=sys.stderr)
                    container.stop()
                    container.remove()
                    return False
                container.commit(repository=temp_image_name)
                container.stop()
                container.remove()
                print("--> Temporary image created.")
                return True
            except Exception as e:
                print(f"--> SDK temp build failed ({e}), falling back to CLI subprocess...")

        cli = self._get_cli_cmd()
        cid_name = f"adas-temp-builder-{self.unique_id}-{random.randint(1000, 9999)}"
        res = subprocess.run(
            [cli, "run", "-d", "--name", cid_name, base_image_name, "sleep", "3600"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            return False
        install_res = subprocess.run(
            [cli, "exec", cid_name, "pip", "install"] + packages,
            capture_output=True,
            text=True,
            check=False,
        )
        if install_res.returncode != 0:
            subprocess.run([cli, "rm", "-f", cid_name], stdout=subprocess.DEVNULL, check=False)
            return False
        subprocess.run([cli, "commit", cid_name, temp_image_name], check=False)
        subprocess.run([cli, "rm", "-f", cid_name], stdout=subprocess.DEVNULL, check=False)
        return True

    def remove_image(self, image_name: str, force: bool = True) -> None:
        """Remove a container image by repository/name."""
        if self.client is not None:
            try:
                self.client.images.remove(image=image_name, force=force)
                return
            except Exception:
                pass
        cli = self._get_cli_cmd()
        subprocess.run(
            [cli, "rmi", "-f", image_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        )

    def perform_maintenance(self) -> None:
        """Prune unused containers and networks."""
        print("--- Performing container maintenance/cleanup ---")
        if self.client is not None:
            try:
                self.client.containers.prune()
                self.client.networks.prune()
                return
            except Exception:
                pass
        cli = self._get_cli_cmd()
        subprocess.run([cli, "container", "prune", "-f"], stdout=subprocess.DEVNULL, check=False)
        subprocess.run([cli, "network", "prune", "-f"], stdout=subprocess.DEVNULL, check=False)

    def cleanup_job_resources(self, base_image_name: str) -> None:
        """Clean up all job-specific resources (containers and base/temp images)."""
        print(f"--- Cleaning up resources for Job {self.unique_id} ---")
        self.remove_image(base_image_name, force=True)
        cli = self._get_cli_cmd()
        subprocess.run([cli, "container", "prune", "-f"], stdout=subprocess.DEVNULL, check=False)
        subprocess.run([cli, "image", "prune", "-f"], stdout=subprocess.DEVNULL, check=False)

    def _get_cli_cmd(self) -> str:
        if self.container_type == "podman":
            return "podman"
        if self.container_type == "docker":
            return "docker"
        if shutil.which("docker"):
            return "docker"
        if shutil.which("podman"):
            return "podman"
        return "docker"


class DependencyParser:
    """Parses dependencies from generated system metrics files natively without jq."""

    @staticmethod
    def get_installed_packages(metrics_file: Union[str, Path]) -> List[str]:
        """Extract installed_packages from a JSON metrics file.

        Supports both list format (e.g. ['pkg1', 'pkg2']) and space-delimited string format.
        """
        path = Path(metrics_file)
        if not path.is_file():
            return []

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            packages_val = data.get("installed_packages")
            if not packages_val:
                return []
            if isinstance(packages_val, list):
                return [str(pkg).strip() for pkg in packages_val if str(pkg).strip()]
            if isinstance(packages_val, str):
                return [pkg.strip() for pkg in packages_val.split() if pkg.strip()]
        except Exception as e:
            print(f"Warning: Failed to parse installed_packages from {path}: {e}")
        return []


class ExecutionManager:
    """Handles subprocess execution with built-in timeout management."""

    @staticmethod
    def run_command(
        cmd: List[str],
        timeout: int = 1200,
        log_file: Optional[Union[str, Path]] = None,
        cwd: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Execute a Python module/command with native timeout handling."""
        log_path = Path(log_file) if log_file else None
        if log_path and log_path.parent:
            log_path.parent.mkdir(parents=True, exist_ok=True)

        cmd_str = " ".join(cmd)
        print(f"  Executing: {cmd_str}")

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                timeout=timeout,
                capture_output=True,
                text=True,
                cwd=cwd,
                check=False,
            )
            duration = time.time() - start_time
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            exit_code = result.returncode

            if log_path:
                with log_path.open("w", encoding="utf-8") as f:
                    f.write(f"Command: {cmd_str}\n")
                    f.write(stdout)
                    if stderr:
                        f.write("\nSTDERR:\n" + stderr)
                    f.write(f"\nExit code: {exit_code}\n")

            return {
                "exit_code": exit_code,
                "status": "WORKING" if exit_code == 0 else "FAILED",
                "stdout": stdout,
                "stderr": stderr,
                "duration": duration,
            }

        except subprocess.TimeoutExpired as e:
            duration = time.time() - start_time
            print(f"  ERROR: Execution timed out after {timeout} seconds.")
            stdout = (
                (e.stdout or b"").decode("utf-8", errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or "")
            )
            stderr = (
                (e.stderr or b"").decode("utf-8", errors="ignore") if isinstance(e.stderr, bytes) else (e.stderr or "")
            )

            if log_path:
                with log_path.open("w", encoding="utf-8") as f:
                    f.write(f"Command: {cmd_str}\n")
                    f.write(stdout)
                    if stderr:
                        f.write("\nSTDERR:\n" + stderr)
                    f.write("\nExit code: 124 (TIMEOUT)\n")

            return {
                "exit_code": 124,
                "status": "TIMEOUT",
                "stdout": stdout,
                "stderr": stderr,
                "duration": duration,
            }


class ResultAggregator:
    """Aggregates experiment results into CSV summaries and human-readable reports."""

    def __init__(self, results_dir: Union[str, Path]):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def export_summary_csv(
        self,
        records: List[Dict[str, Any]],
        filename: str = "results_summary.csv",
    ) -> Path:
        """Export experiment records to a standard CSV file."""
        csv_path = self.results_dir / filename
        fieldnames = ["approach", "benchmark", "iteration", "status"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for rec in records:
                writer.writerow(rec)
        print(f"CSV summary generated: {csv_path}")
        return csv_path

    def export_summary_txt(
        self,
        job_id: str,
        records: List[Dict[str, Any]],
        filename: str = "test_summary.txt",
    ) -> Path:
        """Export a human-readable text summary."""
        txt_path = self.results_dir / filename
        with txt_path.open("w", encoding="utf-8") as f:
            f.write(f"Benchmark Test Summary ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
            f.write(f"Job ID: {job_id}\n")
            f.write("=========================================\n")
            for rec in records:
                approach = rec.get("approach", "")
                bench = rec.get("benchmark", "")
                iter_num = rec.get("iteration", "")
                status = rec.get("status", "UNKNOWN")
                sys_name = (
                    f"{approach}_{bench}{iter_num}_gpt" if approach and bench else str(rec.get("system_name", ""))
                )
                f.write(f"  - {sys_name}: {status}\n")
        print(f"Text summary generated: {txt_path}")
        return txt_path


class Orchestrator:
    """Central engine orchestrating container operations and job workflows."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.unique_id = str(os.environ.get("SLURM_JOB_ID") or random.randint(10000, 99999))
        self.base_image_name = f"adas-base-job:{self.unique_id}"
        self.container_manager = ContainerManager(container_type=args.container, unique_id=self.unique_id)
        self.results_dir = Path(getattr(args, "results_dir", "benchmark_results"))

    def run(self) -> int:
        """Dispatch task based on CLI configuration."""
        task_name = self.args.task
        try:
            if task_name == "benchmark":
                return self.run_benchmark()
            if task_name == "design":
                return self.run_design()
            if task_name == "target":
                return self.run_target()
            print(f"Unknown task: {task_name}", file=sys.stderr)
            return 1
        finally:
            self.container_manager.cleanup_job_resources(self.base_image_name)

    @staticmethod
    def _parse_iterations(iterations_val: str) -> List[str]:
        if not iterations_val:
            return ["1"]
        if "," in iterations_val:
            return [x.strip() for x in iterations_val.split(",") if x.strip()]
        if "-" in iterations_val:
            parts = iterations_val.split("-")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                start, end = int(parts[0]), int(parts[1])
                return [str(x) for x in range(start, end + 1)]
        return [iterations_val.strip()]

    @staticmethod
    def _get_benchmark_info(benchmark: str) -> Tuple[str, str]:
        bench_lower = benchmark.lower()
        if bench_lower in ("gsm", "gsmhard"):
            return "GSMHard", "gsmhard"
        if bench_lower in ("mmlu", "mmlupro"):
            return "MMLUPro", "mmlupro"
        if bench_lower in ("fever",):
            return "FEVER", "fever"
        return benchmark, bench_lower

    def run_benchmark(self) -> int:
        """Execute benchmark workflow across iterations."""
        if not self.container_manager.create_base_image(self.base_image_name):
            print("ERROR: Base image creation failed.", file=sys.stderr)
            return 1

        aggregator = ResultAggregator(self.results_dir)
        records: List[Dict[str, Any]] = []
        overall_exit = 0

        iterations = self._parse_iterations(getattr(self.args, "iterations", "1"))
        bench_dir, bench_name = self._get_benchmark_info(getattr(self.args, "benchmark", "gsm"))
        approach = getattr(self.args, "type", "ablationC")

        initial_dir = Path.cwd()
        adas_dir = initial_dir / f"ADAS_{approach}"
        work_dir = adas_dir if adas_dir.is_dir() else initial_dir

        cleanup_freq = getattr(self.args, "cleanup_frequency", 8)
        timeout = getattr(self.args, "timeout", 1200)

        cleanup_counter = 0
        for iter_num in iterations:
            cleanup_counter += 1
            if cleanup_counter > cleanup_freq:
                self.container_manager.perform_maintenance()
                cleanup_counter = 1

            system_name_base = f"{approach}_{getattr(self.args, 'benchmark', 'gsm')}{iter_num}_gpt"
            system_module_path = f"generated_systems.{system_name_base}"

            print("-----------------------------------------")
            print(f"Testing System: {system_name_base} (in {work_dir})")
            print(f"  Benchmark: {getattr(self.args, 'benchmark', 'gsm')}, Iteration: {iter_num}")

            system_file = work_dir / "generated_systems" / f"{system_name_base}.py"
            if not system_file.is_file():
                print(f"  ERROR: System file not found: {system_file}")
                records.append(
                    {
                        "approach": approach,
                        "benchmark": getattr(self.args, "benchmark", "gsm"),
                        "iteration": iter_num,
                        "status": "FAILED_NOT_FOUND",
                    }
                )
                overall_exit = 1
                continue

            metrics_file = work_dir / "generated_systems" / "metrics" / f"{system_name_base}.json"
            packages = DependencyParser.get_installed_packages(metrics_file)
            image_to_use = self.base_image_name
            temp_image_name = ""

            if packages:
                temp_image_name = f"adas-temp-image-{system_name_base}-{self.unique_id}"
                success = self.container_manager.create_temp_image(self.base_image_name, temp_image_name, packages)
                if success:
                    image_to_use = temp_image_name
                else:
                    print("  WARNING: Failed to build temp image. Using base image.")

            bench_script = initial_dir / "benchmark" / bench_dir / f"main_{bench_name}_bench.py"
            cmd = [
                sys.executable,
                str(bench_script),
                f"--system={system_module_path}",
                f"--base-image={image_to_use}",
                f"--container={self.args.container}",
            ]

            log_file = self.results_dir / f"{system_name_base}_log.txt"
            res = ExecutionManager.run_command(cmd, timeout=timeout, log_file=log_file, cwd=work_dir)

            if temp_image_name:
                self.container_manager.remove_image(temp_image_name, force=True)

            status = res["status"]
            if res["exit_code"] == 0:
                result_src = (
                    work_dir / "benchmark" / bench_dir / "results" / f"benchmark_results_{system_module_path}.json"
                )
                if result_src.is_file():
                    result_dest = self.results_dir / f"{system_name_base}_results.json"
                    shutil.copy2(result_src, result_dest)
                else:
                    status = "FAILED_NO_RESULTS"
                    overall_exit = 1
            else:
                overall_exit = 1

            records.append(
                {
                    "approach": approach,
                    "benchmark": getattr(self.args, "benchmark", "gsm"),
                    "iteration": iter_num,
                    "status": status,
                }
            )

        aggregator.export_summary_csv(records)
        aggregator.export_summary_txt(self.unique_id, records)
        return overall_exit

    def run_design(self) -> int:
        """Execute iterative system generation workflow."""
        if not self.container_manager.create_base_image(self.base_image_name):
            print("ERROR: Base image creation failed.", file=sys.stderr)
            return 1

        overall_exit = 0
        iterations = self._parse_iterations(getattr(self.args, "iterations", "1-16"))
        approach = getattr(self.args, "type", "ablationC")
        benchmark = getattr(self.args, "benchmark", "gsm")
        bench_dir, _ = self._get_benchmark_info(benchmark)

        initial_dir = Path.cwd()
        target_dir = initial_dir / f"ADAS_{approach}"
        work_dir = target_dir if target_dir.is_dir() else initial_dir

        problem_path = getattr(self.args, "problem_path", None)
        if not problem_path:
            candidate = work_dir / "generated_systems" / bench_dir / "prompts.txt"
            problem_path = (
                candidate if candidate.is_file() else initial_dir / "generated_systems" / bench_dir / "prompts.txt"
            )

        problem_text = ""
        if Path(problem_path).is_file():
            problem_text = Path(problem_path).read_text(encoding="utf-8").strip()
        else:
            problem_text = getattr(self.args, "problem", "") or "Solve the benchmark tasks."

        for iter_num in iterations:
            sys_name = f"{approach}_{benchmark}{iter_num}_gpt"
            print("=========================================================")
            print(f"   Running Design Generation for {sys_name}")
            print("=========================================================")
            cmd = [
                sys.executable,
                "run_design.py",
                "--problem",
                problem_text,
                "--name",
                sys_name,
                "--base-image",
                self.base_image_name,
                "--container",
                self.args.container,
            ]
            res = ExecutionManager.run_command(cmd, timeout=getattr(self.args, "timeout", 3600), cwd=work_dir)
            if res["exit_code"] != 0:
                overall_exit = 1

        return overall_exit

    def run_target(self) -> int:
        """Execute generated target systems on specified inputs."""
        if not self.container_manager.create_base_image(self.base_image_name):
            print("ERROR: Base image creation failed.", file=sys.stderr)
            return 1

        overall_exit = 0
        system_names = getattr(self.args, "system_names", [])
        if isinstance(system_names, str):
            system_names = system_names.split()
        if not system_names:
            system_names = ["data_analyst_gpt5_v0"]

        state_json = getattr(self.args, "state", '{"messages": []}')
        data_gen_script = getattr(self.args, "data_gen_script", "")

        Path("data/input").mkdir(parents=True, exist_ok=True)
        Path("data/output").mkdir(parents=True, exist_ok=True)

        if data_gen_script and Path(data_gen_script).is_file():
            print(f"--- Running Data Generation Script: {data_gen_script} ---")
            res_gen = subprocess.run([sys.executable, data_gen_script], check=False)
            if res_gen.returncode != 0:
                print("ERROR: Data generation script failed.", file=sys.stderr)
                return 1

        for sys_name in system_names:
            print("-------------------------------------------------")
            print(f" STARTING RUN FOR: {sys_name}")
            print("-------------------------------------------------")

            metrics_file = Path("generated_systems/metrics") / f"{sys_name}.json"
            packages = DependencyParser.get_installed_packages(metrics_file)
            image_to_use = self.base_image_name
            temp_image_name = ""

            if packages:
                temp_image_name = f"adas-temp-image-{self.unique_id}-{sys_name}"
                if self.container_manager.create_temp_image(self.base_image_name, temp_image_name, packages):
                    image_to_use = temp_image_name

            cmd = [
                sys.executable,
                "test_target.py",
                "--system_name",
                sys_name,
                "--state",
                state_json,
                "--base-image",
                image_to_use,
                "--container",
                self.args.container,
                "--keep-template",
            ]

            res = ExecutionManager.run_command(cmd, timeout=getattr(self.args, "timeout", 1200))
            if res["exit_code"] != 0:
                overall_exit = 1

            if temp_image_name:
                self.container_manager.remove_image(temp_image_name, force=True)

        return overall_exit


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments supporting both positional subcommand and --task flag."""
    parser = argparse.ArgumentParser(description="Unified Python orchestrator for ADAS container and job management.")
    parser.add_argument(
        "task_positional",
        nargs="?",
        choices=["benchmark", "design", "target"],
        help="Task to execute (can also be passed via --task).",
    )
    parser.add_argument(
        "--task",
        choices=["benchmark", "design", "target"],
        default=None,
        help="Task to execute: benchmark, design, or target.",
    )
    parser.add_argument("--type", "--approach", dest="type", default="ablationC", help="System type/approach prefix.")
    parser.add_argument("--benchmark", default="gsm", help="Benchmark name (gsm, mmlu, fever).")
    parser.add_argument(
        "--iterations", "--range", dest="iterations", default="1", help="Iteration range (e.g. 1-16 or 1,2,3)."
    )
    parser.add_argument("--results-dir", default="benchmark_results", help="Directory for logs and summary output.")
    parser.add_argument(
        "--cleanup-frequency", type=int, default=8, help="Frequency of Docker/Podman cleanup operations."
    )
    parser.add_argument("--timeout", type=int, default=1200, help="Execution timeout in seconds.")
    parser.add_argument(
        "--container",
        choices=["auto", "docker", "podman"],
        default="auto",
        help="Container runtime engine.",
    )
    parser.add_argument("--problem-path", default=None, help="Path to prompt file for design runs.")
    parser.add_argument("--problem", default="", help="Problem statement for design runs.")
    parser.add_argument(
        "--system-names", nargs="+", default=["data_analyst_gpt5_v0"], help="System names for target execution."
    )
    parser.add_argument("--state", default='{"messages": []}', help="Initial JSON state string for target execution.")
    parser.add_argument(
        "--data-gen-script", default="", help="Optional script for generating input data before running target."
    )

    args = parser.parse_args(argv)
    if args.task is None and args.task_positional is not None:
        args.task = args.task_positional
    elif args.task is None:
        args.task = "benchmark"
    return args


def main(argv: Optional[List[str]] = None) -> int:
    """Entrypoint for the orchestrator."""
    args = parse_args(argv)
    orchestrator = Orchestrator(args)
    return orchestrator.run()


if __name__ == "__main__":
    sys.exit(main())
