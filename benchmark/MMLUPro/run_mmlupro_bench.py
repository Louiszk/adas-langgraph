import sys
import time
import importlib
from typing import Dict
from benchmark.benchmark_base import run_benchmark_parallel

sys.path.append('/sandbox/workspace')
from adas_core.llm_wrapper import LargeLanguageModel

def execute_problem(problem_item: Dict, system_path: str) -> Dict:
    start_time = time.time()

    LargeLanguageModel.usage_metrics["target_usage"]["overall"] = {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0
    }

    try:
        expected = problem_item.get("answer", "unknown")
        system_module = importlib.import_module(system_path)
        workflow = system_module.workflow

        input_state = {
            "messages": [], 
            "question": problem_item.get("question", ""), 
            "options": problem_item.get("options", [])
        }
        output = workflow.invoke(input_state)
        predicted = output.get("solution", "")

        is_correct = predicted == expected

    except Exception as e:
        predicted = f"Exception: {repr(e)}"
        is_correct = False

    finally:
        duration = time.time() - start_time
        usage = LargeLanguageModel.usage_metrics["target_usage"]["overall"]

    return {
        "question_id": problem_item.get("question_id", "unknown"),
        "question": problem_item.get("question", ""),
        "predicted": predicted,
        "expected": expected,
        "is_correct": is_correct,
        "category": problem_item.get("category", "unknown"),
        "duration_seconds": duration,
        "llm_calls": usage.get("llm_calls", 0),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MMLU-Pro benchmark in parallel with metric aggregation.")
    parser.add_argument("--system", required=True, help="System path to benchmark (e.g., 'my_system.main')")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel processes to use.")
    args = parser.parse_args()

    run_benchmark_parallel(
        benchmark_name="MMLUPro",
        dataset_path="benchmark/MMLUPro/problem_subset.json",
        system_path=args.system,
        execute_problem_fn=execute_problem,
        max_workers=args.workers
    )