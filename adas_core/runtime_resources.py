"""Runtime resource-profile contracts and workspace staging helpers."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from adas_core.task_spec import DatabaseFixtureSpec, FileFixtureSpec, MCPFixtureSpec, MockServiceFixtureSpec, TaskSpec


class RuntimeResourceOverride(BaseModel):
    """Select a fixture, external endpoint, or local artifact for one fixture ID."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["fixture", "external", "local_file"]
    url: str | None = Field(default=None, min_length=1)
    source: str | None = Field(default=None, min_length=1)

    @field_validator("url")
    @classmethod
    def validate_http_url(cls, value: str | None) -> str | None:
        if value is None:
            return value
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("External runtime resource url must be an absolute http:// or https:// URL.")
        return value

    @model_validator(mode="after")
    def validate_provider_fields(self) -> RuntimeResourceOverride:
        if self.provider == "external" and not self.url:
            raise ValueError("An external runtime resource override requires url.")
        if self.provider == "local_file" and not self.source:
            raise ValueError("A local_file runtime resource override requires source.")
        if self.provider != "external" and self.url:
            raise ValueError("url is only valid for an external runtime resource override.")
        if self.provider != "local_file" and self.source:
            raise ValueError("source is only valid for a local_file runtime resource override.")
        return self


class RuntimeResourceProfile(BaseModel):
    """Per-invocation provider choices without modifying a TaskSpec or target."""

    model_config = ConfigDict(extra="forbid")

    default_provider: Literal["fixture"] = "fixture"
    overrides: dict[str, RuntimeResourceOverride] = Field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path | str) -> RuntimeResourceProfile:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def validate_for_task(self, task_spec: TaskSpec, *, check_local_sources: bool = False) -> None:
        for fixture_id, override in self.overrides.items():
            fixture = task_spec.test_fixtures.get_fixture_by_id(fixture_id)
            if fixture is None:
                raise ValueError(f"Runtime profile references unknown fixture '{fixture_id}'.")
            if override.provider == "external" and not isinstance(fixture, (MCPFixtureSpec, MockServiceFixtureSpec)):
                raise ValueError(f"Fixture '{fixture_id}' does not support an external URL override.")
            if override.provider == "local_file" and not (
                isinstance(fixture, FileFixtureSpec)
                or isinstance(fixture, DatabaseFixtureSpec)
                and fixture.db_type in {"sqlite", "duckdb"}
            ):
                raise ValueError(f"Fixture '{fixture_id}' does not support a local file override.")
            if check_local_sources and override.provider == "local_file":
                source = Path(override.source or "")
                if not source.exists():
                    raise ValueError(f"Runtime profile source for '{fixture_id}' does not exist: {override.source}")
                expects_directory = isinstance(fixture, FileFixtureSpec) and fixture.count > 1
                expects_file = (
                    isinstance(fixture, DatabaseFixtureSpec)
                    or isinstance(fixture, FileFixtureSpec)
                    and fixture.count == 1
                )
                if expects_directory and not source.is_dir():
                    raise ValueError(
                        f"Runtime profile source for directory fixture '{fixture_id}' must be a directory."
                    )
                if expects_file and not source.is_file():
                    raise ValueError(f"Runtime profile source for file fixture '{fixture_id}' must be a file.")


def fixture_paths_for_profile(task_spec: TaskSpec, profile: RuntimeResourceProfile | None) -> list[str] | None:
    """Return generated fixture paths that remain active for this invocation."""
    if profile is None:
        return None
    active_ids = [fixture_id for fixture_id, override in profile.overrides.items() if override.provider == "fixture"]
    overridden_ids = set(profile.overrides)
    active_ids.extend(fixture.id for fixture in task_spec.test_fixtures.files if fixture.id not in overridden_ids)
    active_ids.extend(
        fixture.id
        for fixture in task_spec.test_fixtures.databases
        if fixture.id not in overridden_ids and fixture.file_path
    )
    active_ids.extend(
        fixture.id
        for fixture in task_spec.test_fixtures.custom_fixtures
        if fixture.id not in overridden_ids and fixture.path
    )
    return task_spec.test_fixtures.get_file_paths_for_fixture_ids(active_ids) or []


def fixture_process_ids_for_profile(task_spec: TaskSpec, profile: RuntimeResourceProfile | None) -> list[str] | None:
    """Return generated process fixture IDs that remain active for this invocation."""
    if profile is None:
        return None
    return [
        fixture.id
        for fixture in [*task_spec.test_fixtures.mcps, *task_spec.test_fixtures.mock_services]
        if profile.overrides.get(fixture.id, RuntimeResourceOverride(provider="fixture")).provider == "fixture"
    ]


def stage_local_overrides(
    task_spec: TaskSpec, profile: RuntimeResourceProfile, workspace_dirs: dict[str, Path]
) -> None:
    """Copy already-sandboxed local override artifacts into their declared input paths."""
    input_dir = Path(workspace_dirs["input"])
    for fixture_id, override in profile.overrides.items():
        if override.provider != "local_file":
            continue
        fixture = task_spec.test_fixtures.get_fixture_by_id(fixture_id)
        if not isinstance(fixture, (FileFixtureSpec, DatabaseFixtureSpec)):
            raise ValueError(f"Fixture '{fixture_id}' does not support a local file override.")
        destination_rel = fixture.path if isinstance(fixture, FileFixtureSpec) else fixture.file_path
        if not destination_rel:
            continue
        source, destination = Path(override.source or ""), input_dir / destination_rel
        if destination.exists():
            if source.is_dir() or destination.is_dir() != source.is_dir():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


@contextmanager
def external_url_overrides(task_spec: TaskSpec, profile: RuntimeResourceProfile | None) -> Iterator[None]:
    """Expose external HTTP/MCP URLs under the same env names used by fixtures."""
    previous: dict[str, str | None] = {}
    try:
        if profile:
            for fixture_id, override in profile.overrides.items():
                if override.provider != "external":
                    continue
                fixture = task_spec.test_fixtures.get_fixture_by_id(fixture_id)
                if not isinstance(fixture, (MCPFixtureSpec, MockServiceFixtureSpec)):
                    raise ValueError(f"Fixture '{fixture_id}' does not support an external URL override.")
                name = fixture.url_env if isinstance(fixture, MCPFixtureSpec) else fixture.base_url_env
                previous[name] = os.environ.get(name)
                os.environ[name] = override.url or ""
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
