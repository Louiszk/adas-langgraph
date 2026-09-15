from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from adas_core.automatic_setup import AutomaticSetup
from adas_core.automatic_taskspec import AutomaticTaskSpec
from adas_core.automatic_validation import AutomaticValidation
from config import settings


def test_sandbox_dependencies_match_requirements_file():
    sandbox_dependency_names = {"langgraph", "langchain-openai", "python-dotenv", "dill"}
    expected = [
        line.strip()
        for line in (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and line.split("==", maxsplit=1)[0].strip() in sandbox_dependency_names
    ]
    assert settings.dependencies == expected


def test_meta_agent_web_search_is_opt_in_by_default():
    assert settings.meta_agent_enable_web_search is False


def test_setup_model_can_opt_in_to_web_search(monkeypatch):
    monkeypatch.setattr("adas_core.automatic_setup.setup_enable_web_search", True)
    with patch("adas_core.automatic_setup.ChatModel") as chat_model:
        AutomaticSetup()
    chat_model.assert_called_once_with(
        provider=settings.setup_wrapper,
        model=settings.setup_model,
        reasoning_effort=settings.setup_reasoning_effort,
        name="AutomaticSetup",
        is_meta=True,
        default_tools=["web_search"],
    )


def test_taskspec_model_can_opt_in_to_web_search(monkeypatch):
    monkeypatch.setattr("adas_core.automatic_taskspec.taskspec_enable_web_search", True)
    with patch("adas_core.automatic_taskspec.ChatModel") as chat_model:
        AutomaticTaskSpec()
    chat_model.assert_called_once_with(
        provider=settings.taskspec_wrapper,
        model=settings.taskspec_model,
        reasoning_effort=settings.taskspec_reasoning_effort,
        name="AutomaticTaskSpec",
        is_meta=True,
        default_tools=["web_search"],
    )


def test_validation_model_can_opt_in_to_web_search(monkeypatch):
    monkeypatch.setattr("adas_core.automatic_validation.validation_enable_web_search", True)
    with patch("adas_core.automatic_validation.ChatModel") as chat_model:
        AutomaticValidation()
    chat_model.assert_called_once_with(
        provider=settings.validation_wrapper,
        model=settings.validation_model,
        reasoning_effort=settings.validation_reasoning_effort,
        name="AutomaticValidation",
        is_meta=True,
        default_tools=["web_search"],
    )
