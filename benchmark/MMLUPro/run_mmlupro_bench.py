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

    expected = problem_item.get("answer", "")
    predicted = "ERROR"
    is_correct = False
    try:
        system_module = importlib.import_module(system_path)
        workflow = system_module.workflow

        input_state = {
            "messages": [],
            "question": problem_item.get("question", ""),
            "options": problem_item.get("options", []),
        }

        # Run the problem through the workflow
        output = workflow.invoke(input_state)
        predicted = output.get("solution", "")

        is_correct = predicted == expected

    except Exception as e:
        predicted = f"Exception: {e!r}"
        is_correct = False

    finally:
        duration = time.time() - start_time

    return {
        "question_id": problem_item.get("question_id", "unknown"),
        "question": problem_item.get("question", ""),
        "predicted": predicted,
        "expected": expected,
        "is_correct": is_correct,
        "category": problem_item.get("category", "unknown"),
        **extract_target_usage(duration),
    }


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description="Run MMLU-Pro benchmark in parallel with metric aggregation.")

    parser.add_argument(
        "--system",
        required=True,
        help="System path to benchmark (e.g., 'my_system.main')",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel processes to use.")
    args = parser.parse_args()

    run_benchmark_parallel(
        benchmark_name="MMLUPro",
        dataset_path="benchmark/MMLUPro/problem_subset.json",
        system_path=args.system,
        execute_problem_fn=execute_problem,
        max_workers=args.workers,
    )
