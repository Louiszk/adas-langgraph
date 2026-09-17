import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from benchmark.benchmark_base import benchmark_cli_main, run_benchmark_in_sandbox


def run_mmlupro_benchmark_in_sandbox(session, system_name: str) -> bool:
    return run_benchmark_in_sandbox(
        session=session,
        benchmark_name="MMLUPro",
        system_name=system_name,
        runner_script="benchmark/MMLUPro/run_mmlupro_bench.py",
        dataset_file="benchmark/MMLUPro/problem_subset.json",
    )


def main() -> int:
    return benchmark_cli_main(
        benchmark_name="MMLUPro",
        run_in_sandbox_fn=run_mmlupro_benchmark_in_sandbox,
    )


if __name__ == "__main__":
    raise SystemExit(main())
