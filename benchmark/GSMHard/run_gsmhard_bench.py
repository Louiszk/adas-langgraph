import argparse
import importlib
import sys
import time

from adas_core.environment import SANDBOX_WORKSPACE_DIR
from benchmark.benchmark_base import extract_target_usage, reset_target_usage, run_benchmark_parallel
from config.logging import setup_logging

sys.path.append(SANDBOX_WORKSPACE_DIR)


def execute_problem(problem_item: dict, system_path: str) -> dict:
    start_time = time.time()
    reset_target_usage()

    expected = "UNKNOWN"
    predicted = "ERROR"
    is_correct = False
    try:
        system_module = importlib.import_module(system_path)
        workflow = system_module.workflow

        expected = str(problem_item.get("target", ""))

        input_state = {"messages": [], "problem": problem_item.get("input", "")}
        output = workflow.invoke(input_state)
        predicted = output.get("solution", "")

        try:
            predicted_float = float(predicted)
            expected_float = float(expected)
            is_correct = abs(predicted_float - expected_float) < 1e-3
        except (ValueError, TypeError):
            is_correct = predicted == expected

    except Exception as e:
        predicted = f"Exception: {e!r}"
        is_correct = False

    finally:
        duration = time.time() - start_time

    return {
        "question": problem_item["input"],
        "predicted": predicted,
        "expected": expected,
        "is_correct": is_correct,
        **extract_target_usage(duration),
    }


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description="Run GSM-Hard benchmark in parallel with metric aggregation.")

    parser.add_argument(
        "--system",
        required=True,
        help="System path to benchmark (e.g., 'my_system.main')",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel processes to use.")
    args = parser.parse_args()

    run_benchmark_parallel(
        benchmark_name="GSMHard",
        dataset_path="benchmark/GSMHard/problem_subset.json",
        system_path=args.system,
        execute_problem_fn=execute_problem,
        max_workers=args.workers,
    )
