import json
import argparse
import datetime
import os
from typing import Dict, Any

from sandbox.sandbox import StreamingSandboxSession, setup_sandbox_environment

def run_target_system_in_sandbox(session: StreamingSandboxSession, system_name: str, state: Dict[str, Any], run_id: str) -> None:
    """Constructs and executes the command to run the target system inside the sandbox."""
    
    # Construct command with the run_id passed down
    cmd_parts = [f'python3 /sandbox/workspace/run_target.py --system_name="{system_name}" --run-id="{run_id}"']
    
    # Safely serialize and quote the initial state for the command line
    state_str = json.dumps(state)
    quoted_state = state_str.replace('"', '\\"')
    cmd_parts.append(f'--state="{quoted_state}"')
    
    command = " ".join(cmd_parts)
    
    print(f"\n--- Executing Target System: {system_name} (Run ID: {run_id}) ---")
    print(f"Initial State: {state_str}")
    print("-" * 20)
    
    # Stream the output from the sandbox command
    for chunk in session.execute_command_streaming(command):
        print(chunk, end="", flush=True)
    
    print("\n" + "-" * 20)
    print("--- Target system execution completed ---\n")

def main() -> None:
    """Main function to set up and run a target agentic system in a sandboxed environment."""
    parser = argparse.ArgumentParser(description="Run a target agentic system in a sandboxed environment.")
    parser.add_argument("--system_name", required=True, help="Name of the target system to run (e.g., 'DataAnalystSystem_v0').")
    parser.add_argument("--state", default='{"messages": []}', help="JSON string defining the initial state for the system.")
    parser.add_argument("--reinstall", action="store_true", help="Force re-installation of dependencies in the sandbox.")
    parser.add_argument("--keep-template", action="store_true", help="Keep the sandbox image template after the session is closed.")
    parser.add_argument("--base-image", default="python:3.11-slim", help="The base container image to use for the sandbox.")
    parser.add_argument("--container", choices=["auto", "docker", "podman"], default="auto", help="Container runtime to use (auto tries Docker first, then Podman).")
    
    args: argparse.Namespace = parser.parse_args()

    # Generate a synchronization timestamp for this specific run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parse the initial state from the JSON string argument
    try:
        initial_state: Dict[str, Any] = json.loads(args.state)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON provided for --state argument: {e}")
        print("Using a default empty state: {}")
        initial_state = {}

    # Initialize the sandbox session with the specified configuration
    session = StreamingSandboxSession(
        image=args.base_image,
        keep_template=args.keep_template,
        verbose=True,
        container_type=args.container
    )

    try:
        print("--- Opening sandbox session ---")
        session.open()
        
        # Set up the sandbox environment, reinstalling dependencies if requested
        if setup_sandbox_environment(session, reinstall=args.reinstall):
            
            # Purge output and metrics directories to ensure a clean run
            print("--- Purging sandbox output and metrics directories ---")
            session.execute_command("rm -rf /sandbox/workspace/data/output && mkdir -p /sandbox/workspace/data/output")
            session.execute_command("rm -rf /sandbox/workspace/target_metrics && mkdir -p /sandbox/workspace/target_metrics")

            # Run the target system with the provided state AND the timestamp
            run_target_system_in_sandbox(session, args.system_name, initial_state, run_id=timestamp)

            print("\n--- Checking for output data to copy back ---")
            
            # Define specific output folder for this run
            host_output_folder = f"data/output/{args.system_name}_{timestamp}"
            
            session.copy_dir_from_runtime(
                src_dir="/sandbox/workspace/data/output",
                dest_dir=host_output_folder, 
                pattern="*"
            )
            print(f"Output data copied to: {host_output_folder}")
            
            print("--- Checking for metrics files to copy back ---")
            session.copy_dir_from_runtime(
                src_dir="/sandbox/workspace/target_metrics",
                dest_dir="target_metrics",
                pattern="*"
            )
            
            print("--- File copy process finished ---")
            
        else:
            print("Error: Failed to set up the sandbox environment.")

    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()

    finally:
        print("\n--- Closing sandbox session ---")
        session.close()
        print("Session closed.")

if __name__ == "__main__":
    main()