import concurrent.futures
import json
import os
import shlex
from collections.abc import Callable
from typing import Any

from packaging.requirements import Requirement

from adas_core.environment import SANDBOX_GENERATED_SYSTEMS_DIR, SANDBOX_WORKSPACE_DIR
from adas_core.helpers import parse_streaming_exit_code, validate_python_module_path
from config.logging import get_logger

logger = get_logger("benchmark_base")


def run_benchmark_parallel(
    benchmark_name: str,
    dataset_path: str,
    system_path: str,
    execute_problem_fn: Callable[[dict, str], dict],
    max_workers: int,
    custom_results_init: Callable[[dict], None] | None = None,
    custom_results_update: Callable[[dict, dict], None] | None = None,
    custom_results_finalize: Callable[[dict], None] | None = None,
    custom_print_summary: Callable[[dict], None] | None = None,
):
    """
    Base function to run a benchmark in parallel with metric aggregation.
    """
    logger.info(f"Running benchmark for: {system_path}")

    # Handle absolute/relative pathing for sandbox
    if not os.path.exists(dataset_path):
        dataset_path = f"{SANDBOX_WORKSPACE_DIR}/{dataset_path}"

    try:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Could not find {dataset_path}.")

        with open(dataset_path, encoding="utf-8") as f:
            dataset = json.load(f)

        logger.info(f"Loaded static dataset with {len(dataset)} problems")
    except Exception as e:
        logger.error(f"Error loading dataset: {e!s}")
        return

    results = {
        "system": system_path,
        "total_problems": len(dataset),
        "correct": 0,
        "incorrect": 0,
        "problem_results": {},
        "aggregate_metrics": {
            "total_duration_seconds": 0,
            "total_llm_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
        },
    }

    if custom_results_init:
        custom_results_init(results)

    logger.info(f"Executing problems in parallel (max_workers={max_workers})...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_problem = {
            executor.submit(execute_problem_fn, problem_item, system_path): idx
            for idx, problem_item in enumerate(dataset)
        }

        for i, future in enumerate(concurrent.futures.as_completed(future_to_problem), 1):
            idx = future_to_problem[future]
            try:
                result_info = future.result()

                results["aggregate_metrics"]["total_duration_seconds"] += result_info["duration_seconds"]
                results["aggregate_metrics"]["total_llm_calls"] += result_info["llm_calls"]
                results["aggregate_metrics"]["total_input_tokens"] += result_info["input_tokens"]
                results["aggregate_metrics"]["total_output_tokens"] += result_info["output_tokens"]
                results["aggregate_metrics"]["total_tokens"] += result_info["total_tokens"]

                if custom_results_update:
                    custom_results_update(results, result_info)

                if result_info["is_correct"]:
                    logger.info(f"✓ Problem {idx + 1}: Correct")
                    results["correct"] += 1
                else:
                    logger.info(
                        f"✗ Problem {idx + 1}: Incorrect. Expected: {result_info['expected']}, Got: {result_info['predicted']}"
                    )
                    results["incorrect"] += 1

                results["problem_results"][idx] = result_info
                logger.info(f"Progress: {i}/{len(dataset)} problems processed")

            except Exception as exc:
                logger.error(f"Problem {idx + 1} generated an exception: {exc}")
                results["incorrect"] += 1

    total_attempted = results["correct"] + results["incorrect"]
    if total_attempted > 0:
        results["accuracy"] = results["correct"] / total_attempted
        results["aggregate_metrics"]["avg_duration_per_problem"] = (
            results["aggregate_metrics"]["total_duration_seconds"] / total_attempted
        )
        results["aggregate_metrics"]["avg_tokens_per_problem"] = (
            results["aggregate_metrics"]["total_tokens"] / total_attempted
        )

    if custom_results_finalize:
        custom_results_finalize(results)

    results_file = f"{SANDBOX_WORKSPACE_DIR}/benchmark/{benchmark_name}/results/benchmark_results_{system_path}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("--- Benchmark Summary ---")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Total problems: {len(dataset)}")
    logger.info(f"Correct: {results['correct']} | Incorrect: {results['incorrect']}")
    logger.info(f"Accuracy: {results.get('accuracy', 0) * 100:.2f}%")
    logger.info(f"Total LLM Calls: {results['aggregate_metrics']['total_llm_calls']}")
    logger.info(f"Total Tokens: {results['aggregate_metrics']['total_tokens']}")
    logger.info(f"Avg. Duration/Problem: {results['aggregate_metrics'].get('avg_duration_per_problem', 0):.2f}s")

    if custom_print_summary:
        custom_print_summary(results)

    return results


def reset_target_usage() -> None:
    """Reset target usage telemetry in ChatModel."""
    from adas_core.chat_model import ChatModel

    ChatModel.usage_metrics.setdefault("target_usage", {})["overall"] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "llm_calls": 0,
    }


def extract_target_usage(duration_seconds: float) -> dict[str, Any]:
    """Extract captured target token and call usage from ChatModel."""
    from adas_core.chat_model import ChatModel

    usage = ChatModel.usage_metrics.get("target_usage", {}).get("overall", {})
    return {
        "duration_seconds": duration_seconds,
        "llm_calls": usage.get("llm_calls", 0),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def run_benchmark_in_sandbox(
    session: Any,
    benchmark_name: str,
    system_name: str,
    runner_script: str,
    extra_files: list[str] | None = None,
    required_packages: list[str] | None = None,
    dataset_file: str | None = None,
) -> bool:
    """Shared implementation for executing benchmarks inside an isolated sandbox session."""
    try:
        validate_python_module_path(system_name, field_name="benchmark system module")
    except ValueError as exc:
        logger.error(str(exc))
        return False

    logger.info(f"Running {benchmark_name} benchmark for system: {system_name}")

    base_path = f"benchmark/{benchmark_name}"
    system_path = system_name.replace(".", "/") + ".py"
    os.makedirs(base_path, exist_ok=True)

    # Ensure benchmark and generated systems directories exist in sandbox
    session.execute_command(f"mkdir -p {SANDBOX_WORKSPACE_DIR}/{base_path}/results")
    session.execute_command(f"mkdir -p {SANDBOX_GENERATED_SYSTEMS_DIR}")
    if os.path.dirname(system_path):
        session.execute_command(f"mkdir -p {SANDBOX_WORKSPACE_DIR}/{os.path.dirname(system_path)}")

    # Copy the shared benchmark implementation, runner, and target system files.
    session.copy_to_runtime(
        "benchmark/benchmark_base.py",
        f"{SANDBOX_WORKSPACE_DIR}/benchmark/benchmark_base.py",
    )
    session.copy_to_runtime(
        runner_script,
        f"{SANDBOX_WORKSPACE_DIR}/{runner_script}",
    )
    session.copy_to_runtime(system_path, f"{SANDBOX_WORKSPACE_DIR}/{system_path}")

    if dataset_file:
        if not os.path.isfile(dataset_file):
            logger.error("Benchmark dataset was not found: %s", dataset_file)
            return False
        session.copy_to_runtime(dataset_file, f"{SANDBOX_WORKSPACE_DIR}/{dataset_file}")

    if extra_files:
        for fpath in extra_files:
            session.copy_to_runtime(fpath, f"{SANDBOX_WORKSPACE_DIR}/{fpath}")

    if required_packages:
        for pkg in required_packages:
            try:
                requirement = Requirement(pkg)
            except Exception as exc:
                logger.error("Invalid benchmark dependency %r: %s", pkg, exc)
                return False
            show_result = session.execute_command(f"pip show {shlex.quote(requirement.name)}")
            installed_version = None
            for line in str(getattr(show_result, "stdout", "") or "").splitlines():
                if line.startswith("Version:"):
                    installed_version = line.partition(":")[2].strip()
                    break
            installed = getattr(show_result, "exit_code", 1) == 0 and installed_version is not None
            if installed and requirement.specifier and installed_version is not None:
                installed = installed_version in requirement.specifier
            if not installed:
                install_result = session.execute_command(f"pip install {shlex.quote(pkg)}")
                if getattr(install_result, "exit_code", 1) != 0:
                    logger.error("Failed to install benchmark dependency: %s", pkg)
                    return False

    # Run the benchmark
    command = (
        f"python3 {SANDBOX_WORKSPACE_DIR}/{runner_script} --system={shlex.quote(system_name)}"
        + '; bench_exit=$?; printf \'\\n__ADAS_BENCH_EXIT__%s\\n\' "$bench_exit"; exit "$bench_exit"'
    )
    logger.info(f"Executing command: {command}")

    output_chunks: list[str] = []
    for chunk in session.execute_command_streaming(command):
        output_chunks.append(chunk)
        print(chunk, end="", flush=True)

    exit_code = parse_streaming_exit_code(output_chunks, "BENCH")
    bench_succeeded = exit_code == 0
    if not bench_succeeded:
        logger.error("Benchmark execution failed in container")
        return False

    logger.info("Benchmark execution completed!")

    # Copy the results back to the host
    os.makedirs(f"{base_path}/results", exist_ok=True)
    results_file = f"benchmark_results_{system_name}.json"
    if results_file in str(session.execute_command(f"ls -la {SANDBOX_WORKSPACE_DIR}/{base_path}/results")):
        session.copy_from_runtime(
            f"{SANDBOX_WORKSPACE_DIR}/{base_path}/results/{results_file}",
            f"{base_path}/results/{results_file}",
        )
        logger.info(f"Copied benchmark results back to host as {results_file}")
        return True

    logger.error(f"Expected benchmark results file {results_file} was not found")
    return False


def benchmark_cli_main(
    benchmark_name: str,
    run_in_sandbox_fn: Callable[[Any, str], bool],
) -> int:
    """Unified CLI entry point for benchmark sandbox runners."""
    import argparse

    from config.logging import setup_logging
    from sandbox.sandbox import StreamingSandboxSession, setup_sandbox_environment

    setup_logging()

    parser = argparse.ArgumentParser(description=f"Run {benchmark_name} benchmark in a sandboxed environment")
    parser.add_argument(
        "--system",
        required=True,
        help=f"Name of the system to benchmark (e.g., '{benchmark_name}Baseline')",
    )
    parser.add_argument("--reinstall", action="store_true", help="Reinstall dependencies")
    parser.add_argument(
        "--base-image",
        default=None,
        help="The base container image to use for the sandbox.",
    )
    parser.add_argument(
        "--container",
        choices=["auto", "docker", "podman"],
        default="auto",
        help="Container runtime to use (auto tries Docker first, then Podman).",
    )

    args = parser.parse_args()

    try:
        validate_python_module_path(args.system, field_name="benchmark system module")
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    session = StreamingSandboxSession(
        image=args.base_image,
        verbose=True,
        container_type=args.container,
    )

    try:
        session.open()
        logger.info("Sandbox session opened")

        if setup_sandbox_environment(session, args.reinstall):
            success = run_in_sandbox_fn(session, args.system)
            if success:
                logger.info("Benchmark finished successfully!")
                return 0
            else:
                logger.error("Benchmark execution failed in sandbox")
                return 1
        else:
            logger.error("Failed to set up sandbox environment")
            return 1

    except Exception as e:
        logger.exception(f"Error during benchmark execution: {e!s}")
        return 1
    finally:
        logger.info("Closing session...")
        session.close()
