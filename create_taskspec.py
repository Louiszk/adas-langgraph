"""CLI entrypoint for interactive TaskSpec generation and synthesis.

Run interactively:
    python create_taskspec.py

Or non-interactively with arguments:
    python create_taskspec.py --name DataAnalyst --goal "Analyze CSV data and output summary" --non-interactive
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from adas_core.automatic_taskspec import run_interactive_wizard
from adas_core.environment import load_environment
from config.logging import get_logger, setup_logging

logger = get_logger("create_taskspec")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the TaskSpec CLI."""
    parser = argparse.ArgumentParser(description="Interactive CLI for synthesizing validated ADAS TaskSpecs.")
    parser.add_argument(
        "--name",
        default=None,
        help="System name for the task (e.g. 'DataAnalyst', 'ResearchAssistant').",
    )
    parser.add_argument(
        "--goal",
        default=None,
        help="System goal string or path to a file containing the task/problem description.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save the generated TaskSpec (defaults to specs/<task_name>/).",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without interactive prompts (requires --goal or input prompt).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main execution function for the create_taskspec CLI."""
    load_environment()
    setup_logging()
    args = parse_args(argv)

    goal = args.goal
    if goal and os.path.isfile(goal):
        try:
            goal = Path(goal).read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to read goal file %s: %r", args.goal, exc)
            return 1

    try:
        spec = run_interactive_wizard(
            initial_prompt=goal,
            output_dir=args.output_dir,
            non_interactive=args.non_interactive,
            task_name=args.name,
        )
        return 0 if spec is not None else 1
    except Exception:
        logger.exception("TaskSpec synthesis failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
