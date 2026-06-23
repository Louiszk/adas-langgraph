import os
import argparse
import importlib
from config import settings, task
from sandbox.sandbox import StreamingSandboxSession, setup_sandbox_environment


def run_meta_system_in_sandbox(
    session: StreamingSandboxSession,
    problem_statement,
    target_name,
    optimize_system=None,
):
    quoted_problem = problem_statement.replace('"', '\\"')
    command = f'python3 /sandbox/workspace/run_meta.py "{quoted_problem}" "{target_name}" "{settings.max_iterations}" '
    command += f'"{optimize_system}"' if optimize_system else ""

    for chunk in session.execute_command_streaming(command):
        print(chunk, end="", flush=True)

    print("\nMeta system execution completed!")

    if "generated_systems" in str(session.execute_command("ls -la /sandbox/workspace")):
        print("Copying generated systems and metrics back to host...")
        os.makedirs("generated_systems", exist_ok=True)
        escaped_target_name = target_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        target_file_name = escaped_target_name + ".py"
        target_pickle_name = escaped_target_name + ".pkl"

        as_dir = str(session.execute_command("ls -la /sandbox/workspace/generated_systems"))
        if target_file_name in as_dir:
            session.copy_from_runtime(
                f"/sandbox/workspace/generated_systems/{target_file_name}",
                f"generated_systems/{target_file_name}",
            )
        if target_pickle_name in as_dir:
            session.copy_from_runtime(
                f"/sandbox/workspace/generated_systems/{target_pickle_name}",
                f"generated_systems/{target_pickle_name}",
            )
        print(f"Copied {target_file_name} and .pkl back to host")

        if "metrics" in str(session.execute_command("ls -la /sandbox/workspace/generated_systems")):
            metrics_file = target_name.replace("/", "_").replace("\\", "_").replace(":", "_") + ".json"

            if metrics_file in str(session.execute_command("ls -la /sandbox/workspace/generated_systems/metrics")):
                os.makedirs("generated_systems/metrics", exist_ok=True)
                session.copy_from_runtime(
                    f"/sandbox/workspace/generated_systems/metrics/{metrics_file}",
                    f"generated_systems/metrics/{metrics_file}",
                )
                print(f"Copied metrics file {metrics_file} back to host")

    session.copy_dir_from_runtime(src_dir="/sandbox/workspace/data/output", dest_dir="data/output", pattern="*")

    return True


def main():
    parser = argparse.ArgumentParser(description="Run agentic systems in a sandboxed environment")
    parser.add_argument(
        "--keep-template",
        action="store_true",
        help="Keep the image after the session is closed",
    )
    parser.add_argument("--reinstall", action="store_true", help="Reinstall dependencies.")
    parser.add_argument("--problem", default=task.problem_statement, help="Problem statement to solve")
    parser.add_argument("--name", default="UnnamedSystem", help="Target system name")
    parser.add_argument(
        "--meta-system",
        default="compact_system",
        help="The name of the meta-system to use from the meta_systems folder.",
    )
    parser.add_argument(
        "--optimize-system",
        default=None,
        help="Specify target system name to optimize or change",
    )
    parser.add_argument(
        "--container",
        choices=["auto", "docker", "podman"],
        default="auto",
        help="Container runtime to use (auto will try Docker first, then Podman)",
    )
    parser.add_argument(
        "--base-image",
        default="python:3.11-slim",
        help="The base container image to use for the sandbox.",
    )

    args = parser.parse_args()
    print(args)

    try:
        module_path = f"meta_systems.{args.meta_system}.build"
        build_module = importlib.import_module(module_path)
        create_meta_system_func = getattr(build_module, "create_meta_system")

    except (ImportError, AttributeError) as e:
        print(f"ERROR: Could not load the build script for the meta-system '{args.meta_system}'.")
        print(f"Details: {e}")
        return
    create_meta_system_func()

    if not os.path.exists("materialized_meta_system/MetaSystem.py"):
        print("ERROR: The expected MetaSystem.py file was not found.")
        return
    print("Meta-system successfully built.")

    session = StreamingSandboxSession(
        image=args.base_image,
        keep_template=args.keep_template,
        verbose=True,
        container_type=args.container,
    )

    try:
        session.open()
        if setup_sandbox_environment(session, args.reinstall):
            run_meta_system_in_sandbox(session, args.problem, args.name, args.optimize_system)
            print("Finished successfully!")
        else:
            print("Failed to set up sandbox environment")
    except Exception as e:
        print(repr(e))
    finally:
        print("Session closed.")
        session.close()


if __name__ == "__main__":
    main()
