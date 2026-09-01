import importlib
import sys
import time

from adas_core.logging_config import get_logger, setup_logging
from adas_core.llm_wrapper import LargeLanguageModel
from benchmark.benchmark_base import run_benchmark_parallel

logger = get_logger("run_fever_bench")

# Set Wikipedia User-Agent if the library is installed
try:
    import wikipedia  # type: ignore

    wikipedia.set_user_agent("FEVER-Benchmark/1.0 (lf37cyti@studserv.uni-leipzig.de)")
    logger.info("Wikipedia User-Agent configured")
except ImportError:
    logger.warning("Wikipedia library not installed")

sys.path.append("/sandbox/workspace")


def execute_problem(problem_item: dict, system_path: str) -> dict:
    time.sleep(0.2)
    start_time = time.time()

    LargeLanguageModel.usage_metrics["target_usage"]["overall"] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "llm_calls": 0,
    }

    predicted = "NO ANSWER"
    is_correct = False
    try:
        expected = problem_item["label"]
        system_module = importlib.import_module(system_path)
        workflow = system_module.workflow

        # Run the problem through the workflow
        input_state = {"messages": [], "claim": problem_item["claim"]}
        output = workflow.invoke(input_state)
        predicted = output.get("prediction", "")

        is_correct = predicted == expected

    except Exception as e:
        predicted = f"Exception: {e!r}"
        is_correct = False

    finally:
        duration = time.time() - start_time
        usage = LargeLanguageModel.usage_metrics["target_usage"]["overall"]

        return {
            "id": problem_item["id"],
            "claim": problem_item["claim"],
            "predicted": predicted,
            "expected": problem_item["label"],
            "is_correct": is_correct,
            "duration_seconds": duration,
            "llm_calls": usage.get("llm_calls", 0),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }


def custom_results_init(results: dict):
    results["label_metrics"] = {
        "SUPPORTS": {"true": 0, "false": 0, "total": 0},
        "REFUTES": {"true": 0, "false": 0, "total": 0},
        "NOT ENOUGH INFO": {"true": 0, "false": 0, "total": 0},
    }


def custom_results_update(results: dict, result_info: dict):
    expected_label = result_info["expected"]
    is_correct = result_info["is_correct"]
    if expected_label in results["label_metrics"]:
        results["label_metrics"][expected_label]["total"] += 1
        if is_correct:
            results["label_metrics"][expected_label]["true"] += 1
        else:
            results["label_metrics"][expected_label]["false"] += 1


def custom_results_finalize(results: dict):
    for label in results["label_metrics"]:
        label_total = results["label_metrics"][label]["total"]
        results["label_metrics"][label]["accuracy"] = (
            results["label_metrics"][label]["true"] / label_total if label_total > 0 else 0
        )


def custom_print_summary(results: dict):
    logger.info("--- Per-label Performance ---")
    for label, metrics in results["label_metrics"].items():
        if metrics["total"] > 0:
            logger.info(f"{label}: {metrics['accuracy'] * 100:.2f}% accuracy ({metrics['true']}/{metrics['total']})")


if __name__ == "__main__":
    import argparse

    setup_logging()

    parser = argparse.ArgumentParser(description="Run FEVER benchmark in parallel with metric aggregation.")
    parser.add_argument(
        "--system",
        required=True,
        help="System path to benchmark (e.g., 'my_system.main')",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel processes to use (default: 1 to respect Wikipedia rate limits).",
    )
    args = parser.parse_args()

    run_benchmark_parallel(
        benchmark_name="FEVER",
        dataset_path="benchmark/FEVER/problem_subset.json",
        system_path=args.system,
        execute_problem_fn=execute_problem,
        max_workers=args.workers,
        custom_results_init=custom_results_init,
        custom_results_update=custom_results_update,
        custom_results_finalize=custom_results_finalize,
        custom_print_summary=custom_print_summary,
    )
