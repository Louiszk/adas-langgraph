"""Tests for isolated external database seed setup and teardown."""

from __future__ import annotations

from pathlib import Path

import pytest

from adas_core.exceptions import FixtureExecutionError
from adas_core.fixture_lifecycle import external_database_seed_lifecycle
from adas_core.task_spec import (
    ArchitectureContract,
    ExternalDatabaseSeedSpec,
    ResourceEntry,
    ResourceManifest,
    TaskSpec,
    TestCaseSpec,
    TestFixturesSpec,
)


def _task_spec() -> TaskSpec:
    return TaskSpec(
        name="ExternalSeedTask",
        system_goal="Exercise an isolated external database.",
        architecture_contract=ArchitectureContract(execution_mode="single_turn", state_schema={"query": "str"}),
        resource_manifest=ResourceManifest(
            available_resources=[ResourceEntry(name="evaluation_db", type="database", description="Test database")]
        ),
        test_fixtures=TestFixturesSpec(
            external_database_seeds=[
                ExternalDatabaseSeedSpec(
                    name="orders_seed",
                    resource_name="evaluation_db",
                    db_type="postgres",
                    driver="psycopg",
                    connection_env={"uri": "EVALUATION_DB_URI"},
                    namespace_kind="schema",
                    namespace="adas_test_orders",
                    description="Create deterministic orders and customers.",
                )
            ]
        ),
        dev_suite=[
            TestCaseSpec(
                id="seeded", description="Uses seeded orders", fixture_ids=["orders_seed"], turns=[{"query": "q"}]
            )
        ],
    )


def _write_script(root: Path, body: str) -> Path:
    scripts = root / "setup_scripts"
    scripts.mkdir()
    script = scripts / "seed_external_orders_seed.py"
    script.write_text(body, encoding="utf-8")
    return script


def _script(event_file: Path, *, seed_body: str = "", cleanup_body: str = "") -> str:
    return (
        "from pathlib import Path\n"
        f"EVENT_FILE = Path({str(event_file)!r})\n"
        "def _event(value):\n"
        "    EVENT_FILE.write_text(EVENT_FILE.read_text() + value + '\\n' if EVENT_FILE.exists() else value + '\\n')\n"
        "def seed_external_database(connection_config, namespace):\n"
        "    _event('seed:' + namespace)\n"
        f"    {seed_body or 'pass'}\n"
        "def cleanup_external_database(connection_config, namespace):\n"
        "    _event('cleanup:' + namespace)\n"
        f"    {cleanup_body or 'pass'}\n"
    )


def test_lifecycle_seeds_and_cleans_on_success(tmp_path, monkeypatch):
    events = tmp_path / "events.txt"
    _write_script(tmp_path, _script(events))
    monkeypatch.setenv("EVALUATION_DB_URI", "postgres://secret@example/test")

    with external_database_seed_lifecycle(_task_spec(), ["orders_seed"], tmp_path):
        assert events.read_text(encoding="utf-8").splitlines() == ["seed:adas_test_orders"]

    assert events.read_text(encoding="utf-8").splitlines() == ["seed:adas_test_orders", "cleanup:adas_test_orders"]


def test_lifecycle_cleans_after_workflow_or_validator_failure(tmp_path, monkeypatch):
    events = tmp_path / "events.txt"
    _write_script(tmp_path, _script(events))
    monkeypatch.setenv("EVALUATION_DB_URI", "postgres://secret@example/test")

    with pytest.raises(RuntimeError, match="workflow failed"):
        with external_database_seed_lifecycle(_task_spec(), ["orders_seed"], tmp_path):
            raise RuntimeError("workflow failed")

    assert events.read_text(encoding="utf-8").splitlines()[-1] == "cleanup:adas_test_orders"


def test_lifecycle_attempts_cleanup_after_seed_failure_without_leaking_credential(tmp_path, monkeypatch):
    events = tmp_path / "events.txt"
    _write_script(tmp_path, _script(events, seed_body="raise RuntimeError(connection_config['uri'])"))
    secret = "postgres://super-secret@example/test"
    monkeypatch.setenv("EVALUATION_DB_URI", secret)

    with pytest.raises(FixtureExecutionError, match="orders_seed.*failed during setup") as exc_info:
        with external_database_seed_lifecycle(_task_spec(), ["orders_seed"], tmp_path):
            pass

    assert secret not in str(exc_info.value)
    assert events.read_text(encoding="utf-8").splitlines() == ["seed:adas_test_orders", "cleanup:adas_test_orders"]


def test_lifecycle_rejects_missing_connection_variable_by_name_only(tmp_path, monkeypatch):
    _write_script(tmp_path, _script(tmp_path / "events.txt"))
    monkeypatch.delenv("EVALUATION_DB_URI", raising=False)

    with pytest.raises(FixtureExecutionError, match="EVALUATION_DB_URI") as exc_info:
        with external_database_seed_lifecycle(_task_spec(), ["orders_seed"], tmp_path):
            pass

    assert "postgres://" not in str(exc_info.value)


def test_lifecycle_requires_cleanup_contract_before_seeding(tmp_path, monkeypatch):
    script_dir = tmp_path / "setup_scripts"
    script_dir.mkdir()
    (script_dir / "seed_external_orders_seed.py").write_text(
        "def seed_external_database(connection_config, namespace):\n    raise AssertionError('must not run')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EVALUATION_DB_URI", "postgres://secret@example/test")

    with pytest.raises(FixtureExecutionError, match="must define both"):
        with external_database_seed_lifecycle(_task_spec(), ["orders_seed"], tmp_path):
            pass
