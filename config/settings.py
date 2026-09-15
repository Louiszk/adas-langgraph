from pathlib import Path

# Central runtime settings. Keep additional configuration modules in this package
# as the project grows (for example, provider-specific or deployment settings).


_SANDBOX_DEPENDENCY_NAMES = frozenset({"langgraph", "langchain-openai", "python-dotenv", "dill"})


def _load_pinned_dependencies() -> list[str]:
    """Read the minimal sandbox runtime set with versions pinned by requirements.txt."""
    requirements_file = Path(__file__).resolve().parents[1] / "requirements.txt"
    return [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and line.split("==", maxsplit=1)[0].strip() in _SANDBOX_DEPENDENCY_NAMES
    ]


# Sandbox bootstrap installs only the target runtime dependencies
dependencies = _load_pinned_dependencies()

# Meta-agent runtime
ACTION_CUTOFF = 4
TARGET_SYSTEM_RECURSION_LIMIT = 20
meta_agent_wrapper = "openai"
meta_agent_model = "gpt-5.6-luna"
meta_agent_reasoning_effort = "medium"
meta_agent_enable_web_search = False

# Independent generation roles. Enable web search only when the role needs live external information.
setup_wrapper = "openai"
setup_model = "gpt-5.6-luna"
setup_reasoning_effort = "medium"
setup_enable_web_search = False

taskspec_wrapper = "openai"
taskspec_model = "gpt-5.6-luna"
taskspec_reasoning_effort = "medium"
taskspec_enable_web_search = False

validation_wrapper = "openai"
validation_model = "gpt-5.6-luna"
validation_reasoning_effort = "medium"
validation_enable_web_search = False

# Candidate Selection & Checkpoint Management
CANDIDATE_OPTIMIZATION_METRIC: str = "tokens"
CLEANUP_CHECKPOINTS_ON_FINALIZATION: bool = True

max_iterations = 40  # Maximum number of steps the meta system should perform (e.g., LLM calls)
additional_documentation_max_tokens = 40_000
additional_documentation_token_encoding = "cl100k_base"
