import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from benchmark.benchmark_base import benchmark_cli_main, run_benchmark_in_sandbox


def run_gsmhard_benchmark_in_sandbox(session, system_name: str) -> bool:
    return run_benchmark_in_sandbox(
        session=session,
        benchmark_name="GSMHard",
        system_name=system_name,
        runner_script="benchmark/GSMHard/run_gsmhard_bench.py",
        required_packages=["datasets"],
    )


def main() -> None:
    benchmark_cli_main(
        benchmark_name="GSMHard",
        run_in_sandbox_fn=run_gsmhard_benchmark_in_sandbox,
    )


if __name__ == "__main__":
    main()
