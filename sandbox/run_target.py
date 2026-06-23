import os
import json
import time
import argparse
import importlib
import datetime
from typing import Dict, Any
from langgraph.graph.state import CompiledStateGraph

# Import the LLM wrapper to access usage metrics
from adas_core.llm_wrapper import LargeLanguageModel


def main() -> None:
    """
    Main entry point for running a compiled agentic system inside the sandbox.
    Captures execution metrics and the full final state.
    """
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
    args = parser.parse_args()

    # --- Metrics Initialization ---
    start_time = time.time()
    step_counter = 0

    metrics: Dict[str, Any] = {
        "system_name": args.system_name,
        "run_id": args.run_id,
        "status": "started",
        "initial_state": {},
        "error": None,
    }

    # Define the metrics directory
    metrics_dir = "/sandbox/workspace/target_metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    # Variable to hold the full final state object
    final_state_snapshot = None

    try:
        try:
            initial_state: Dict[str, Any] = json.loads(args.state)
            metrics["initial_state"] = initial_state
        except json.JSONDecodeError:
            print("Warning: Invalid JSON for --state argument. Using an empty state {}.")
            initial_state = {}
            metrics["initial_state"] = {}

        print(f"--- Preparing to run target system: {args.system_name} ---")

        module_path = f"generated_systems.{args.system_name}"
        try:
            target_module = importlib.import_module(module_path)
            workflow: CompiledStateGraph = target_module.workflow
            print(f"Successfully imported '{args.system_name}'.")
        except ModuleNotFoundError:
            raise RuntimeError(f"System module not found at '{module_path}'. Please ensure the file exists.")
        except Exception as e:
            raise RuntimeError(f"Failed to load the workflow from the module: {e}")

        try:
            viz_path = f"{metrics_dir}/{args.system_name}.png"
            workflow.get_graph().draw_mermaid_png(output_file_path=viz_path)
            print(f"System graph visualization saved to '{viz_path}'")
        except Exception as e:
            print(f"Warning: Failed to generate graph visualization: {e}")

        print("\n--- Starting system execution with initial state ---")
        print(json.dumps(initial_state, indent=2))
        print("-" * 50)

        for mode, payload in workflow.stream(
            initial_state,
            config={"recursion_limit": 20},
            stream_mode=["updates", "values"],
        ):
            if mode == "updates" and isinstance(payload, dict):
                step_counter += 1
                print(f"\n[Step {step_counter}]")
                for node_name, state_update in payload.items():
                    print(f"--- Update from node: '{node_name}' ---")
                    print(json.dumps(state_update, indent=2, default=str))

            elif mode == "values":
                final_state_snapshot = payload

        metrics["status"] = "completed"
        print("\n" + "=" * 50)
        print("--- System execution finished successfully ---")
        print("=" * 50)

    except Exception as e:
        import traceback

        metrics["status"] = "error"
        error_info = {
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        metrics["error"] = error_info
        print("\n" + "!" * 50)
        print("--- An error occurred during system execution ---")
        traceback.print_exc()
        print("!" * 50)

    finally:
        # --- Finalize and Save Metrics ---
        end_time = time.time()
        metrics["duration_seconds"] = round(end_time - start_time, 2)
        metrics["iterations"] = step_counter
        metrics["usage_metrics"] = LargeLanguageModel.usage_metrics.get("target_usage", {})

        if args.run_id:
            run_id = args.run_id
        else:
            run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        metrics_filename = f"{args.system_name}_{run_id}.json"
        metrics_filepath = os.path.join(metrics_dir, metrics_filename)

        try:
            with open(metrics_filepath, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"\n--- Execution metrics saved to: {metrics_filepath} ---")
        except Exception as e:
            print(f"\nFATAL: Could not save metrics file: {e}")

        if final_state_snapshot:
            state_filename = f"{args.system_name}_{run_id}_final_state.txt"
            state_filepath = os.path.join(metrics_dir, state_filename)
            try:
                state_str = json.dumps(final_state_snapshot, indent=2, default=str)
                with open(state_filepath, "w", encoding="utf-8") as f:
                    f.write(state_str)
                print(f"--- Final state saved to: {state_filepath} ---")
            except Exception as e:
                print(f"\nFATAL: Could not save final state file: {e}")


if __name__ == "__main__":
    main()
