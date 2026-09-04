import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from benchmark.benchmark_base import benchmark_cli_main, run_benchmark_in_sandbox


def run_fever_benchmark_in_sandbox(session, system_name: str, dataset_name: str = "problem_subset.json") -> bool:
    return run_benchmark_in_sandbox(
        session=session,
        benchmark_name="FEVER",
        system_name=system_name,
        runner_script="benchmark/FEVER/run_fever_bench.py",
        extra_files=[f"benchmark/FEVER/{dataset_name}"],
    )


def main() -> None:
    benchmark_cli_main(
        benchmark_name="FEVER",
        run_in_sandbox_fn=run_fever_benchmark_in_sandbox,
    )


if __name__ == "__main__":
    main()
