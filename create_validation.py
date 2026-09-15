"""Create frozen task validation module using the validation model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from adas_core.automatic_validation import ensure_automatic_validation, verify_validation_manifest
from adas_core.environment import load_environment
from adas_core.task_spec import TaskSpec
from config.logging import get_logger, setup_logging

logger = get_logger("create_validation")


def run_validation_for_task(
    task_spec_path: Path | str,
    *,
    output_dir: Path | str | None = None,
    force: bool = False,
) -> Path | None:
    """Generate the frozen validation module for a TaskSpec."""
    spec_path = Path(task_spec_path).resolve()
    task_spec = TaskSpec.from_file(spec_path)
    target_dir = Path(output_dir).resolve() if output_dir else spec_path.parent

    logger.info("Generating frozen validation module for task '%s' in %s...", task_spec.name, target_dir)
    result = ensure_automatic_validation(task_spec, target_dir, force=force)
    if result is None:
        logger.info("Frozen validation module is already up to date (use --force to regenerate).")
        safe_name = task_spec.name.replace("/", "_").replace("\\", "_").replace(":", "_")
        for candidate in [
            target_dir / f"{safe_name}.validation.py",
            target_dir / f"{task_spec.name}.validation.py",
            target_dir / "validation.py",
        ]:
            if candidate.exists():
                return candidate
        return None

    logger.info("Validation module created at %s", result.validation_file_path)
    if result.required_packages:
        logger.info("Discovered validation requirements: %s", result.required_packages)
    return result.validation_file_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize an LLM-authored, frozen validation module for a TaskSpec.")
    parser.add_argument(
        "--task-spec",
        required=True,
        type=Path,
        help="Path to the task specification JSON file (e.g. specs/my_task/task.json).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Target directory for the validation module (defaults to task_spec's parent directory).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify whether the validation module and manifest are current and exit without making changes.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the validation module even if one already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_environment()
    setup_logging()
    args = parse_args(argv)

    if args.verify:
        spec_path = Path(args.task_spec).resolve()
        try:
            task_spec = TaskSpec.from_file(spec_path)
        except Exception as exc:
            logger.error("TaskSpec validation failed for '%s': %s", args.task_spec, exc)
            return 1

        target_dir = Path(args.output_dir).resolve() if args.output_dir else spec_path.parent
        is_current, issues = verify_validation_manifest(task_spec, target_dir)
        if is_current:
            logger.info("Frozen validation module is current for '%s'.", args.task_spec)
            return 0
        logger.error(
            "Frozen validation module is stale or missing for '%s':\n  - %s\nRun 'python create_validation.py --task-spec %s --force' to regenerate.",
            args.task_spec,
            "\n  - ".join(issues),
            args.task_spec,
        )
        return 1

    try:
        val_path = run_validation_for_task(
            args.task_spec,
            output_dir=args.output_dir,
            force=args.force,
        )
        return 0 if val_path is not None else 1
    except Exception as exc:
        logger.error("Validation generation failed: %r", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
