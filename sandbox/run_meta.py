import json
import os
import sys
import time
from typing import Any, cast

import dill as pickle

from adas_core.environment import (
    SANDBOX_GENERATED_SYSTEMS_DIR,
    SANDBOX_TASK_SETUP_DIR,
    SANDBOX_TASK_SPEC_PATH,
    SANDBOX_WORKSPACE_DIR,
)
from adas_core.helpers import escape_system_name

sys.path.append(SANDBOX_WORKSPACE_DIR)
from adas_core.chat_model import ChatModel, UsageRecorder, usage_scope
from adas_core.logging_config import get_logger, setup_logging
from adas_core.task_spec import TaskSpec
from adas_core.virtual_agentic_system import VirtualAgenticSystem
from meta_system.graph import workflow

logger = get_logger("run_meta")
_TASK_SPEC_PATH = SANDBOX_TASK_SPEC_PATH


def load_visible_task_spec() -> TaskSpec:
    """Load the visible TaskSpec and configure the target-model allow list."""
    task_spec = TaskSpec.from_file(_TASK_SPEC_PATH)
    ChatModel.allowed_target_models = [m.model_dump() for m in task_spec.available_models]
    return task_spec


def load_visible_task_context(task_spec: TaskSpec) -> str:
    """Load the visible development contract without any holdout data."""
    return (
        "\n\n--- TaskSpec Design Contract ---\n"
        "Use this contract for architecture, state, declared fixture paths, resources, and output requirements. "
        "Concrete development cases are intentionally withheld; generalize to the stated goal.\n"
        f"```json\n{task_spec.to_design_context()}\n```\n"
        "--- End TaskSpec Design Contract ---"
    )


def main():
    setup_logging()
    start_time = time.time()
    metrics = {
        "system_name": "",
        "duration_seconds": 0,
        "iterations": 0,
        "usage_metrics": {},
        "status": "started",
        "error": None,
        "stream_content": "",
        "installed_packages": "",
    }

    if len(sys.argv) < 3:
        raise ValueError(
            "run_meta.py requires at least 2 arguments: <problem_statement> <system_name> [max_iterations] [optimize_from_file]"
        )

    problem_statement = sys.argv[1]
    system_name = sys.argv[2]

    max_iterations = 30
    if len(sys.argv) >= 4:
        max_iterations = int(sys.argv[3])

    optimize_from_file = None
    if len(sys.argv) >= 5:
        optimize_from_file = sys.argv[4]
        metrics["optimize_from_file"] = optimize_from_file

    metrics["system_name"] = system_name
    try:
        task_spec = load_visible_task_spec()
        problem_statement += load_visible_task_context(task_spec)
    except Exception as exc:
        raise RuntimeError(f"Could not load visible TaskSpec context: {exc}") from exc
    metrics["problem_statement"] = problem_statement
    logger.info(f"Running meta system for '{system_name}'...")

    target_agentic_system: VirtualAgenticSystem | None = None

    try:
        if optimize_from_file:
            path = f"{SANDBOX_GENERATED_SYSTEMS_DIR}/" + escape_system_name(optimize_from_file)
            try:
                with open(path + ".pkl", "rb") as f:
                    target_agentic_system = cast(VirtualAgenticSystem, pickle.load(f))
                target_agentic_system.system_name = system_name
                target_agentic_system.escaped_name = escape_system_name(system_name)
                logger.info("System initialized from existing file.")
            except Exception as e:
                raise RuntimeError(f"Error initializing from file: {e}") from e
        else:
            target_agentic_system = VirtualAgenticSystem(system_name)

        inputs = {
            "messages": [],
            "initial_task": problem_statement,
            "target_agentic_system": target_agentic_system,
            "optimize": bool(optimize_from_file),
            "max_iterations": max_iterations,
            "task_spec": task_spec,
            "task_dir": SANDBOX_TASK_SETUP_DIR,
        }

        processed_msg_count = 0
        logger.info("Streaming meta system execution...")

        with usage_scope(system="meta"):
            for output in workflow.stream(cast(Any, inputs), config={"recursion_limit": 999}):
                metrics["iterations"] += 1

                for out in output.values():
                    if not isinstance(out, dict):
                        continue

                    if "messages" in out:
                        messages = out["messages"]
                        if messages:
                            new_messages = messages[processed_msg_count:]
                            for msg in new_messages:
                                msg_type = getattr(msg, "type", "Unknown")
                                content = getattr(msg, "content", "")
                                stream_content = f"\n[{msg_type}]: {content}\n"
                                metrics["stream_content"] += stream_content
                                logger.info(stream_content.strip())

                            processed_msg_count = len(messages)

                    if out.get("design_completed"):
                        logger.info("Design completed.")
                        metrics["status"] = "completed"

        metrics["status"] = "completed"

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        logger.error(f"Error running meta system: {e!s}\n{error_traceback}")

        metrics["status"] = "error"
        metrics["error"] = {"message": repr(e), "traceback": error_traceback}

    finally:
        # Finalize metrics
        end_time = time.time()
        metrics["duration_seconds"] = end_time - start_time
        metrics["usage_metrics"] = ChatModel.usage_metrics
        metrics["scoped_metrics"] = UsageRecorder.get_aggregate(system="meta")

        escaped_name = escape_system_name(system_name)
        metrics_dir = f"{SANDBOX_GENERATED_SYSTEMS_DIR}/metrics"
        os.makedirs(metrics_dir, exist_ok=True)

        # Load the final saved system to get the list of installed packages
        final_system_path = f"{SANDBOX_GENERATED_SYSTEMS_DIR}/{escaped_name}.pkl"
        if os.path.exists(final_system_path):
            try:
                with open(final_system_path, "rb") as f:
                    final_system_object = pickle.load(f)
                if hasattr(final_system_object, "installed_packages") and final_system_object.installed_packages:
                    metrics["installed_packages"] = " ".join(final_system_object.installed_packages.values())
            except Exception as e:
                logger.warning(f"Could not read installed packages from final system pickle: {e!r}")

        metrics_file = f"{metrics_dir}/{escaped_name}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()
