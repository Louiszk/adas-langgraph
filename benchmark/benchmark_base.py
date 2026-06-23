import os
import json
import concurrent.futures
from typing import Dict, Callable, Optional


def run_benchmark_parallel(
    benchmark_name: str,
    dataset_path: str,
    system_path: str,
    execute_problem_fn: Callable[[Dict, str], Dict],
    max_workers: int,
    custom_results_init: Optional[Callable[[Dict], None]] = None,
    custom_results_update: Optional[Callable[[Dict, Dict], None]] = None,
    custom_results_finalize: Optional[Callable[[Dict], None]] = None,
    custom_print_summary: Optional[Callable[[Dict], None]] = None,
):
    """
    Base function to run a benchmark in parallel with metric aggregation.
    """
    print(f"Running benchmark for: {system_path}")

    # Handle absolute/relative pathing for sandbox
    if not os.path.exists(dataset_path):
        dataset_path = f"/sandbox/workspace/{dataset_path}"

    try:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Could not find {dataset_path}.")

        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        print(f"Loaded static dataset with {len(dataset)} problems")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
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

    print(f"Executing problems in parallel (max_workers={max_workers})...")

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
                    print(f"✓ Problem {idx + 1}: Correct")
                    results["correct"] += 1
                else:
                    print(
                        f"✗ Problem {idx + 1}: Incorrect. Expected: {result_info['expected']}, Got: {result_info['predicted']}"
                    )
                    results["incorrect"] += 1

                results["problem_results"][idx] = result_info
                print(f"Progress: {i}/{len(dataset)} problems processed")

            except Exception as exc:
                print(f"Problem {idx + 1} generated an exception: {exc}")
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

    results_file = f"sandbox/workspace/benchmark/{benchmark_name}/results/benchmark_results_{system_path}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Benchmark Summary ---")
    print(f"Results saved to: {results_file}")
    print(f"Total problems: {len(dataset)}")
    print(f"Correct: {results['correct']} | Incorrect: {results['incorrect']}")
    print(f"Accuracy: {results.get('accuracy', 0) * 100:.2f}%")
    print(f"Total LLM Calls: {results['aggregate_metrics']['total_llm_calls']}")
    print(f"Total Tokens: {results['aggregate_metrics']['total_tokens']}")
    print(f"Avg. Duration/Problem: {results['aggregate_metrics'].get('avg_duration_per_problem', 0):.2f}s")

    if custom_print_summary:
        custom_print_summary(results)

    return results
