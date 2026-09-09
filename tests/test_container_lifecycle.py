"""Live-container smoke coverage for the design and target invocation lifecycles.

These tests deliberately replace only the API-backed meta-design loop with small
in-container scripts.  The surrounding host invocation, task copy, preflight,
exit-marker, artifact-copy, and isolated-workspace code all run unchanged against
a real Docker or Podman container.  This keeps the smoke test deterministic and
free of model credentials while covering the integration boundary that mocks miss.
"""

from __future__ import annotations

import json
import shutil
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

import invoke_design
import invoke_target
from sandbox.sandbox import (
    StreamingSandboxSession,
    check_docker_running,
    check_podman_running,
    copy_task_setup_to_sandbox,
    run_sandbox_preflight,
    setup_sandbox_environment,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def container_runtime() -> str:
    """Select a live runtime or skip cleanly when container testing is unavailable."""
    if check_docker_running():
        return "docker"
    if check_podman_running():
        return "podman"
    pytest.skip("container lifecycle smoke tests require a running Docker or Podman engine")


def _write_smoke_task(task_dir: Path, name: str) -> Path:
    fixtures_dir = task_dir / "fixtures"
    fixtures_dir.mkdir(parents=True)
    (fixtures_dir / "input.csv").write_text("value\n42\n", encoding="utf-8")

    task = {
        "schema_version": "1.0",
        "name": name,
        "system_goal": "Copy the fixture value to an output artifact.",
        "architecture_contract": {"execution_mode": "single_turn", "state_schema": {"result": "str"}},
        "test_fixtures": {
            "files": [{"id": "input_csv", "path": "input.csv", "description": "Smoke input", "content_type": "csv"}]
        },
        "dev_suite": [{"id": "smoke_case", "description": "Smoke case", "turns": [{}]}],
    }
    spec_path = task_dir / "task.json"
    spec_path.write_text(json.dumps(task), encoding="utf-8")
    (task_dir / f"{name}.validation.py").write_text(
        textwrap.dedent(
            """\
            def validate_smoke_case(final_state, workspace_dirs):
                from pathlib import Path

                return (Path(workspace_dirs['input']) / 'input.csv').is_file(), 'fixture must be present'

            VALIDATORS = {'smoke_case': validate_smoke_case}
            """
        ),
        encoding="utf-8",
    )
    (task_dir / "preflight.py").write_text(
        textwrap.dedent(
            """\
            from pathlib import Path

            def check_environment(workspace_dirs):
                return (Path(workspace_dirs['input']) / 'input.csv').is_file(), 'fixture must be present'
            """
        ),
        encoding="utf-8",
    )
    (task_dir / "setup_manifest.json").write_text(json.dumps({"required_packages": []}), encoding="utf-8")
    return spec_path


def _copy_runtime_script(session: StreamingSandboxSession, source: Path, destination: str) -> None:
    session.copy_to_runtime(str(source), destination)


@pytest.mark.container
def test_design_lifecycle_copies_artifacts_and_propagates_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], container_runtime: str
) -> None:
    """Exercise design setup/preflight, frozen validation, artifacts, and exit markers live."""
    system_name = f"ContainerDesignSmoke{uuid.uuid4().hex}"
    spec_path = _write_smoke_task(tmp_path / "design_task", system_name)
    success_runner = tmp_path / "successful_run_meta.py"
    success_runner.write_text(
        textwrap.dedent(
            """\
            import argparse
            import json
            from pathlib import Path
            import dill

            parser = argparse.ArgumentParser()
            parser.add_argument('--task-spec', type=Path, required=True)
            parser.add_argument('--system-name', required=True)
            parser.add_argument('--max-iterations')
            parser.add_argument('--optimize-system')
            args = parser.parse_args()

            task = json.loads(args.task_spec.read_text(encoding='utf-8'))
            task_dir = args.task_spec.parent
            validator = task_dir / f"{task['name']}.validation.py"
            assert validator.is_file(), 'frozen validator missing'

            namespace = {}
            exec(validator.read_text(encoding='utf-8'), namespace)
            passed, message = namespace['validate_smoke_case'](
                {},
                {'input': str(task_dir / 'fixtures'), 'output': '/tmp'},
            )
            assert passed, message

            generated = Path('/sandbox/workspace/generated_systems')
            generated.mkdir(parents=True, exist_ok=True)
            (generated / f"{args.system_name}.py").write_text('workflow = None\\n', encoding='utf-8')
            with (generated / f"{args.system_name}.pkl").open('wb') as stream:
                dill.dump({'validated': True}, stream)

            metrics = generated / 'metrics'
            metrics.mkdir(exist_ok=True)
            (metrics / f"{args.system_name}.json").write_text(
                json.dumps({'status': 'completed', 'validator_passed': passed}),
                encoding='utf-8',
            )
            """
        ),
        encoding="utf-8",
    )
    failing_runner = tmp_path / "failing_run_meta.py"
    failing_runner.write_text("raise SystemExit(17)\n", encoding="utf-8")

    original_setup = invoke_design.setup_sandbox_environment

    def setup_with_smoke_runner(session: StreamingSandboxSession, reinstall: bool = False) -> bool:
        assert original_setup(session, reinstall)
        _copy_runtime_script(session, success_runner, "/sandbox/workspace/run_meta.py")
        return True

    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(invoke_design, "setup_manifest_is_current", lambda _: True)
    monkeypatch.setattr(invoke_design, "setup_sandbox_environment", setup_with_smoke_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "invoke_design.py",
            "--task-spec",
            str(spec_path),
            "--system-name",
            system_name,
            "--container",
            container_runtime,
        ],
    )

    artifacts = [
        REPO_ROOT / "generated_systems" / f"{system_name}.py",
        REPO_ROOT / "generated_systems" / f"{system_name}.pkl",
        REPO_ROOT / "generated_systems" / "metrics" / f"{system_name}.json",
    ]
    try:
        assert invoke_design.main() == 0
        assert all(path.is_file() for path in artifacts)
        assert json.loads(artifacts[-1].read_text(encoding="utf-8"))["validator_passed"] is True

        session = StreamingSandboxSession(verbose=False, container_type=container_runtime)
        try:
            session.open()
            assert setup_sandbox_environment(session)
            runtime_task_dir = copy_task_setup_to_sandbox(session, spec_path.parent, spec_path)
            assert run_sandbox_preflight(session, runtime_task_dir)
            _copy_runtime_script(session, failing_runner, "/sandbox/workspace/run_meta.py")
            assert not invoke_design.run_meta_system_in_sandbox(session, system_name)
            assert "__ADAS_META_EXIT__17" in capsys.readouterr().out
        finally:
            session.close()
    finally:
        for artifact in artifacts:
            artifact.unlink(missing_ok=True)


@pytest.mark.container
def test_target_lifecycle_seeds_fixtures_collects_output_and_propagates_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], container_runtime: str
) -> None:
    """Exercise the target isolated workspace and target-exit marker against a live container."""
    suffix = uuid.uuid4().hex
    system_name = f"ContainerTargetSmoke{suffix}"
    failing_system_name = f"ContainerTargetFailure{suffix}"
    spec_path = _write_smoke_task(tmp_path / "target_task", system_name)
    generated_dir = REPO_ROOT / "generated_systems"
    generated_dir.mkdir(exist_ok=True)
    source_template = textwrap.dedent(
        """\
        import os
        from pathlib import Path
        from langgraph.graph import END, START, StateGraph

        def run(state):
            input_file = Path(os.environ['ADAS_INPUT_DIR']) / 'input.csv'
            output_file = Path(os.environ['ADAS_OUTPUT_DIR']) / 'result.txt'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(input_file.read_text(encoding='utf-8'), encoding='utf-8')
            return {'result': output_file.read_text(encoding='utf-8')}

        builder = StateGraph(dict)
        builder.add_node('run', run)
        builder.add_edge(START, 'run')
        builder.add_edge('run', END)
        workflow = builder.compile()
        """
    )
    (generated_dir / f"{system_name}.py").write_text(source_template, encoding="utf-8")
    (generated_dir / f"{failing_system_name}.py").write_text(
        "raise RuntimeError('intentional target smoke failure')\n", encoding="utf-8"
    )

    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(invoke_target, "setup_manifest_is_current", lambda _: True)
    output_prefix = REPO_ROOT / "data" / "output"
    try:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "invoke_target.py",
                "--system-name",
                system_name,
                "--task-spec",
                str(spec_path),
                "--state",
                "{}",
                "--container",
                container_runtime,
            ],
        )
        assert invoke_target.main() == 0
        output_dirs = sorted(output_prefix.glob(f"{system_name}_*"))
        assert output_dirs
        assert (output_dirs[-1] / "result.txt").read_text(encoding="utf-8") == "value\n42\n"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "invoke_target.py",
                "--system-name",
                failing_system_name,
                "--task-spec",
                str(spec_path),
                "--state",
                "{}",
                "--container",
                container_runtime,
            ],
        )
        assert invoke_target.main() == 1
        assert "__ADAS_TARGET_EXIT__1" in capsys.readouterr().out
    finally:
        for source in (generated_dir / f"{system_name}.py", generated_dir / f"{failing_system_name}.py"):
            source.unlink(missing_ok=True)
        for output_dir in output_prefix.glob(f"{system_name}_*"):
            shutil.rmtree(output_dir)
        metrics_dir = REPO_ROOT / "target_metrics"
        for metric in metrics_dir.glob(f"{system_name}*"):
            metric.unlink(missing_ok=True)
        for metric in metrics_dir.glob(f"{failing_system_name}*"):
            metric.unlink(missing_ok=True)
