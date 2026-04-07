import os
import sys
import json
import dill as pickle
import time
from langchain_core.messages import HumanMessage

sys.path.append('/sandbox/workspace')
from adas_core.virtual_agentic_system import VirtualAgenticSystem
from adas_core.llm_wrapper import LargeLanguageModel
from materialized_meta_system import MetaSystem

def main():
    start_time = time.time()
    metrics = {
        "system_name": "",
        "duration_seconds": 0,
        "iterations": 0,
        "usage_metrics": {},
        "status": "started",
        "error": None,
        "stream_content": "",
        "installed_packages": ""
    }
    
    problem_statement = "Create me a simple system that can solve math problems."
    if len(sys.argv) >= 2:
        problem_statement = sys.argv[1]
   
    system_name = "MathProblemSolver"
    if len(sys.argv) >= 3:
        system_name = sys.argv[2]
    
    if len(sys.argv) >= 4:
        max_iterations = int(sys.argv[3])

    optimize_from_file = None
    if len(sys.argv) >= 5:
        optimize_from_file = sys.argv[4]
        metrics["optimize_from_file"] = optimize_from_file
   
    metrics["system_name"] = system_name
    metrics["problem_statement"] = problem_statement
    print(f"Running meta system for '{system_name}'...")

    target_agentic_system = None
   
    try:
        if optimize_from_file:
            path = "/sandbox/workspace/generated_systems/" + optimize_from_file.replace("/", "").replace("\\", "").replace(":", "")
            try:
                with open(path + '.pkl', 'rb') as f:
                    target_agentic_system: VirtualAgenticSystem = pickle.load(f)
                target_agentic_system.system_name = system_name
                target_agentic_system.escaped_name = system_name.replace("/", "").replace("\\", "").replace(":", "")
                print("System initialized from existing file.")
            except Exception as e:
                raise RuntimeError(f"Error initializing from file: {e}") from e
        else:
            target_agentic_system = VirtualAgenticSystem(system_name)
       
        workflow = MetaSystem.workflow
        inputs = {
            "messages": [],
            "initial_task": problem_statement,
            "target_agentic_system": target_agentic_system,
            "optimize": bool(optimize_from_file),
            "max_iterations": max_iterations
            }
        
        processed_msg_count = 0
        print("Streaming meta system execution...")
        
        for output in workflow.stream(inputs, config={"recursion_limit": 999}):
            metrics["iterations"] += 1
            
            for out in output.values():
                if "messages" in out:
                    messages = out["messages"]
                    if messages:
                        new_messages = messages[processed_msg_count:]
                        for msg in new_messages:
                            msg_type = getattr(msg, 'type', 'Unknown')
                            content = getattr(msg, 'content', '')
                            stream_content = f"\n[{msg_type}]: {content}\n"
                            metrics["stream_content"] += stream_content
                            print(stream_content)
                        
                        processed_msg_count = len(messages)

                if "validation_code_snippets" in out and out["validation_code_snippets"]:
                    metrics["validation_code_snippets"] = out["validation_code_snippets"]
                
                if "design_completed" in out and out["design_completed"]:
                    print("Design completed.")
                    metrics["status"] = "completed"

        metrics["status"] = "completed"
       
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(error_traceback)
        print(f"Error running meta system: {str(e)}")
        
        metrics["status"] = "error"
        metrics["error"] = {
            "message": repr(e),
            "traceback": error_traceback
        }
    
    finally:
        # Finalize metrics
        end_time = time.time()
        metrics["duration_seconds"] = end_time - start_time
        metrics["usage_metrics"] = LargeLanguageModel.usage_metrics
        
        escaped_name = system_name.replace('/', '').replace('\\', '').replace(':', '')
        metrics_dir = "/sandbox/workspace/generated_systems/metrics"
        os.makedirs(metrics_dir, exist_ok=True)
            
        # Load the final saved system to get the list of installed packages
        final_system_path = f"/sandbox/workspace/generated_systems/{escaped_name}.pkl"
        if os.path.exists(final_system_path):
            try:
                with open(final_system_path, 'rb') as f:
                    final_system_object = pickle.load(f)
                if hasattr(final_system_object, 'installed_packages') and final_system_object.installed_packages:
                    metrics["installed_packages"] = " ".join(final_system_object.installed_packages.values())
            except Exception as e:
                print(f"Could not read installed packages from final system pickle: {repr(e)}")
        
        metrics_file = f"{metrics_dir}/{escaped_name}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    main()
