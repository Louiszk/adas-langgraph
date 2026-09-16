import argparse
import contextlib
import datetime
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

from langgraph.graph.state import CompiledStateGraph

# Import ChatModel and UsageRecorder
from adas_core.chat_model import ChatModel, UsageRecorder, usage_scope
from adas_core.environment import (
    SANDBOX_TARGET_METRICS_DIR,
    SANDBOX_WORKSPACE_DIR,
    isolated_case_workspace,
    load_environment,
)
from adas_core.fixture_lifecycle import process_fixture_lifecycle
from adas_core.helpers import escape_system_name, validate_system_name
from adas_core.runtime_resources import (
    RuntimeResourceProfile,
    external_url_overrides,
    fixture_paths_for_profile,
    fixture_process_ids_for_profile,
    stage_local_overrides,
)
from adas_core.task_spec import TaskSpec
from config.logging import get_logger, setup_logging
from config.settings import TARGET_SYSTEM_RECURSION_LIMIT

logger = get_logger("run_target")


def main() -> int:
    """
    Main entry point for running a compiled agentic system inside the sandbox.
    Captures execution metrics and the full final state.
    """
    load_environment()
    setup_logging()

    parser = argparse.ArgumentParser(description="Run a compiled agentic system and record metrics.")
    parser.add_argument(
        "--system_name",
        required=True,
        help="Name of the target system module in 'generated_systems'.",
    )
    parser.add_argument(
        "--state",
        default="{}",
        help="JSON string defining the initial state for the workflow.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Unique identifier/timestamp for this run to sync output filenames.",
    )
    parser.add_argument(
        "--task-dir",
        default=None,
        help="Path to sandbox task setup directory containing fixtures and task.json.",
    )
    parser.add_argument(
        "--runtime-profile", default=None, help="Runtime resource profile JSON supplied by invoke_target."
    )
    args = parser.parse_args()

    # --- Metrics Initialization ---
    start_time = time.time()
    step_counter = 0
    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics: dict[str, Any] = {
        "system_name": args.system_name,
        "run_id": run_id,
        "status": "started",
        "initial_state": {},
        "error": None,
    }

    # Define the metrics directory
    metrics_dir = SANDBOX_TARGET_METRICS_DIR
    os.makedirs(metrics_dir, exist_ok=True)

    # Variable to hold the full final state snapshot
    final_state_snapshot = None
    exit_code = 0

    try:
        validate_system_name(args.system_name, field_name="target system name")
        try:
            raw_state: Any = json.loads(args.state)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON for --state argument: {e}") from e
        if not isinstance(raw_state, dict):
            raise ValueError("Initial state must be a JSON object.")
        initial_state: dict[str, Any] = raw_state
        metrics["initial_state"] = initial_state

        logger.info(f"Preparing to run target system: {args.system_name}")

        module_path = f"generated_systems.{args.system_name}"
        try:
            target_module = importlib.import_module(module_path)
            workflow: CompiledStateGraph = target_module.workflow
            logger.info(f"Successfully imported '{args.system_name}'.")
        except ModuleNotFoundError:
            raise RuntimeError(f"System module not found at '{module_path}'. Please ensure the file exists.")
        except Exception as e:
            raise RuntimeError(f"Failed to load the workflow from the module: {e}")

        try:
            viz_path = f"{metrics_dir}/{args.system_name}.png"
            workflow.get_graph().draw_mermaid_png(output_file_path=viz_path)
            logger.info(f"System graph visualization saved to '{viz_path}'")
        except Exception as e:
            logger.warning(f"Warning: Failed to generate graph visualization: {e}")

        logger.info("Starting system execution with initial state:")
        logger.info(json.dumps(initial_state, indent=2))

        fixtures_dir: Path | None = None
        task_spec: TaskSpec | None = None
        runtime_profile: RuntimeResourceProfile | None = None
        if args.runtime_profile:
            runtime_profile = RuntimeResourceProfile.model_validate_json(args.runtime_profile)
        if args.task_dir:
            task_dir = Path(args.task_dir)
            task_spec_path = task_dir / "task.json"
            if task_spec_path.is_file():
                task_spec = TaskSpec.from_file(task_spec_path)
                if runtime_profile:
                    runtime_profile.validate_for_task(task_spec)
            candidate = task_dir / "fixtures"
            if candidate.is_dir():
                fixtures_dir = candidate

        target_runs_dir = Path(SANDBOX_WORKSPACE_DIR) / "target_runs"
        allowed_files = fixture_paths_for_profile(task_spec, runtime_profile) if task_spec else None
        with isolated_case_workspace(
            base_dir=target_runs_dir,
            run_id=run_id,
            case_id="invocation",
            fixtures_dir=fixtures_dir,
            allowed_files=allowed_files,
            clean_up=False,
        ) as workspace_dirs:
            metrics["workspace"] = str(workspace_dirs["workspace"])
            metrics["input_dir"] = str(workspace_dirs["input"])
            metrics["output_dir"] = str(workspace_dirs["output"])

            logger.info("Isolated workspace initialized at: %s", workspace_dirs["workspace"])
            logger.info("ADAS_INPUT_DIR: %s", os.environ.get("ADAS_INPUT_DIR"))
            logger.info("ADAS_OUTPUT_DIR: %s", os.environ.get("ADAS_OUTPUT_DIR"))

            if task_spec and runtime_profile:
                stage_local_overrides(task_spec, runtime_profile, workspace_dirs)
            fixture_context = (
                process_fixture_lifecycle(
                    task_spec.test_fixtures,
                    fixture_process_ids_for_profile(task_spec, runtime_profile),
                    Path(args.task_dir),
                    workspace_dirs,
                )
                if task_spec is not None
                else contextlib.nullcontext()
            )
            url_context = external_url_overrides(task_spec, runtime_profile) if task_spec else contextlib.nullcontext()
            with fixture_context, url_context:
                with usage_scope(system="target", run_id=run_id):
                    for mode, payload in workflow.stream(
                        initial_state,
                        config={"recursion_limit": TARGET_SYSTEM_RECURSION_LIMIT},
                        stream_mode=["updates", "values"],
                    ):
                        if mode == "updates" and isinstance(payload, dict):
                            step_counter += 1
                            logger.info(f"[Step {step_counter}]")
                            for node_name, state_update in payload.items():
                                logger.info(f"Update from node '{node_name}': {json.dumps(state_update, default=str)}")

                        elif mode == "values":
                            final_state_snapshot = payload

        metrics["status"] = "completed"
        logger.info("System execution finished successfully")

    except Exception as e:
        exit_code = 1
        import traceback

        metrics["status"] = "error"
        error_info = {
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        metrics["error"] = error_info
        logger.error(f"An error occurred during system execution: {e}\n{traceback.format_exc()}")

    finally:
        # --- Finalize and Save Metrics ---
        end_time = time.time()
        metrics["duration_seconds"] = round(end_time - start_time, 2)
        metrics["iterations"] = step_counter
        metrics["usage_metrics"] = ChatModel.usage_metrics.get("target_usage", {})
        metrics["scoped_metrics"] = UsageRecorder.get_aggregate(system="target", run_id=run_id)

        metrics_filename = f"{escape_system_name(args.system_name)}_{run_id}.json"
        metrics_filepath = os.path.join(metrics_dir, metrics_filename)

        try:
            with open(metrics_filepath, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Execution metrics saved to: {metrics_filepath}")
        except Exception as e:
            logger.error(f"Could not save metrics file: {e}")

        if final_state_snapshot:
            state_filename = f"{args.system_name}_{run_id}_final_state.txt"
            state_filepath = os.path.join(metrics_dir, state_filename)
            try:
                state_str = json.dumps(final_state_snapshot, indent=2, default=str)
                with open(state_filepath, "w", encoding="utf-8") as f:
                    f.write(state_str)
                logger.info(f"Final state saved to: {state_filepath}")
            except Exception as e:
                logger.error(f"Could not save final state file: {e}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
