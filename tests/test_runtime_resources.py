import os

import pytest
from pydantic import ValidationError

from adas_core.runtime_resources import (
    RuntimeResourceProfile,
    external_url_overrides,
    fixture_paths_for_profile,
    fixture_process_ids_for_profile,
    stage_local_overrides,
)
from adas_core.task_spec import (
    ArchitectureContract,
    CustomFixtureSpec,
    FileFixtureSpec,
    MockServiceFixtureSpec,
    TaskSpec,
    TestFixturesSpec,
)


def _spec() -> TaskSpec:
    return TaskSpec(
        name="RuntimeProfileTask",
        system_goal="Goal",
        architecture_contract=ArchitectureContract(execution_mode="single_turn", state_schema={"q": "str"}),
        test_fixtures=TestFixturesSpec(
            files=[FileFixtureSpec(id="stations", path="data/stations.csv")],
            mock_services=[
                MockServiceFixtureSpec(id="transit_api", name="transit", port=8110, base_url_env="TRANSIT_URL")
            ],
        ),
    )


def test_runtime_profile_stages_local_file_and_overrides_external_url(tmp_path, monkeypatch):
    source = tmp_path / "my_stations.csv"
    source.write_text("id,name\n1,Central\n", encoding="utf-8")
    profile = RuntimeResourceProfile.model_validate(
        {
            "overrides": {
                "stations": {"provider": "local_file", "source": str(source)},
                "transit_api": {"provider": "external", "url": "https://transit.example/api"},
            }
        }
    )
    spec = _spec()
    profile.validate_for_task(spec, check_local_sources=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace_dirs = {"workspace": workspace, "input": workspace / "input", "output": workspace / "output"}
    workspace_dirs["input"].mkdir()
    workspace_dirs["output"].mkdir()

    stage_local_overrides(spec, profile, workspace_dirs)
    assert (workspace_dirs["input"] / "data" / "stations.csv").read_text(encoding="utf-8") == source.read_text(
        encoding="utf-8"
    )
    assert fixture_paths_for_profile(spec, profile) == []
    assert fixture_process_ids_for_profile(spec, profile) == []

    monkeypatch.setenv("TRANSIT_URL", "previous")
    with external_url_overrides(spec, profile):
        assert os.environ["TRANSIT_URL"] == "https://transit.example/api"
    assert os.environ["TRANSIT_URL"] == "previous"


def test_runtime_profile_rejects_wrong_provider_and_unknown_fixture(tmp_path):
    spec = _spec()
    profile = RuntimeResourceProfile.model_validate(
        {"overrides": {"stations": {"provider": "external", "url": "https://example.test"}}}
    )
    with pytest.raises(ValueError, match="does not support an external"):
        profile.validate_for_task(spec)
    profile = RuntimeResourceProfile.model_validate(
        {"overrides": {"missing": {"provider": "local_file", "source": str(tmp_path)}}}
    )
    with pytest.raises(ValueError, match="unknown fixture"):
        profile.validate_for_task(spec)
    with pytest.raises(ValidationError, match="absolute http"):
        RuntimeResourceProfile.model_validate({"overrides": {"transit_api": {"provider": "external", "url": "x"}}})


def test_runtime_profile_validates_and_deduplicates_additional_packages():
    profile = RuntimeResourceProfile.model_validate(
        {"additional_packages": ["external-client>=2", "external-client>=2", "urllib3"]}
    )

    assert profile.additional_packages == ["external-client>=2", "urllib3"]
    with pytest.raises(ValidationError, match="Invalid runtime package"):
        RuntimeResourceProfile.model_validate({"additional_packages": ["bad package; rm -rf /"]})


def test_runtime_profile_validates_local_source_shape(tmp_path):
    spec = _spec()
    file_profile = RuntimeResourceProfile.model_validate(
        {"overrides": {"stations": {"provider": "local_file", "source": str(tmp_path)}}}
    )
    with pytest.raises(ValueError, match="must be a file"):
        file_profile.validate_for_task(spec, check_local_sources=True)
    spec.test_fixtures.files[0] = FileFixtureSpec(id="stations", path="data/stations", count=2)
    file_source = tmp_path / "stations.csv"
    file_source.write_text("id", encoding="utf-8")
    directory_profile = RuntimeResourceProfile.model_validate(
        {"overrides": {"stations": {"provider": "local_file", "source": str(file_source)}}}
    )
    with pytest.raises(ValueError, match="must be a directory"):
        directory_profile.validate_for_task(spec, check_local_sources=True)


def test_runtime_profile_keeps_custom_fixture_paths_and_replaces_conflicting_destination(tmp_path):
    source = tmp_path / "replacement.csv"
    source.write_text("replacement", encoding="utf-8")
    spec = _spec()
    spec.test_fixtures.custom_fixtures = []
    spec.test_fixtures.custom_fixtures.append(
        CustomFixtureSpec(id="custom", name="custom", path="tools/run.sh", description="d")
    )
    profile = RuntimeResourceProfile.model_validate(
        {"overrides": {"stations": {"provider": "local_file", "source": str(source)}}}
    )
    assert fixture_paths_for_profile(spec, profile) == ["tools/run.sh"]
    workspace = tmp_path / "workspace"
    input_dir = workspace / "input"
    destination = input_dir / "data" / "stations.csv"
    destination.mkdir(parents=True)
    workspace_dirs = {"workspace": workspace, "input": input_dir, "output": workspace / "output"}
    workspace_dirs["output"].mkdir()
    stage_local_overrides(spec, profile, workspace_dirs)
    assert destination.is_file()
    assert destination.read_text(encoding="utf-8") == "replacement"


def test_runtime_profile_directory_override_replaces_existing_directory(tmp_path):
    source = tmp_path / "replacement"
    source.mkdir()
    (source / "only-local.csv").write_text("local", encoding="utf-8")
    spec = _spec()
    spec.test_fixtures.files[0] = FileFixtureSpec(id="stations", path="data/stations")
    profile = RuntimeResourceProfile.model_validate(
        {"overrides": {"stations": {"provider": "local_file", "source": str(source)}}}
    )
    workspace = tmp_path / "workspace"
    destination = workspace / "input" / "data" / "stations"
    destination.mkdir(parents=True)
    (destination / "generated-only.csv").write_text("generated", encoding="utf-8")
    workspace_dirs = {"workspace": workspace, "input": workspace / "input", "output": workspace / "output"}
    workspace_dirs["output"].mkdir()

    stage_local_overrides(spec, profile, workspace_dirs)

    assert (destination / "only-local.csv").is_file()
    assert not (destination / "generated-only.csv").exists()
