"""Dependency sets and package-exclusion policy shared by the runtime and sandbox."""

from pathlib import Path

SANDBOX_DEPENDENCY_NAMES = frozenset({"langgraph", "langchain-openai", "python-dotenv", "dill"})

DEFAULT_EXCLUDED_PACKAGES: list[str] = [
    "datasets",
    "docker",
    "grpcio-status",
    "langchain-openai",
    "wheel",
    "llm-sandbox",
    "pip",
    "dill",
    "podman",
    "python-dotenv",
    "setuptools",
]


def load_pinned_sandbox_dependencies() -> list[str]:
    """Read only the minimal sandbox runtime packages from requirements.txt."""
    requirements_file = Path(__file__).resolve().parents[1] / "requirements.txt"
    return [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and line.split("==", maxsplit=1)[0].strip() in SANDBOX_DEPENDENCY_NAMES
    ]


SANDBOX_DEPENDENCIES = load_pinned_sandbox_dependencies()
