"""Domain exceptions used by ADAS runtime and meta-system boundaries.

The leaf classes intentionally remain compatible with ``ValueError``,
``RuntimeError``, or ``TypeError`` where the existing public API exposed
those built-ins.
"""


class AdasError(Exception):
    """Base class for classified, actionable ADAS failures."""


class GraphError(AdasError):
    """Base class for graph construction and materialization failures."""


class ModelError(AdasError):
    """Base class for model configuration and output failures."""


class EvaluationError(AdasError):
    """Base class for evaluator failures."""


class RuntimeEnvironmentError(AdasError):
    """Base class for fixture and runtime environment failures."""


class SandboxError(RuntimeEnvironmentError):
    """Base class for container sandbox failures."""


class TaskSpecError(AdasError):
    """Base class for TaskSpec failures."""


class MetaSystemError(AdasError):
    """Base class for meta-system state and parsing failures."""


class _ValueErrorCompat(ValueError):
    """Mixin preserving the project's established ValueError API."""


class _RuntimeErrorCompat(RuntimeError):
    """Mixin preserving the project's established RuntimeError API."""


class _TypeErrorCompat(TypeError):
    """Mixin preserving the project's established TypeError API."""


class GraphTopologyError(GraphError, _ValueErrorCompat):
    """An invalid graph endpoint, node reference, route, or unconditional loop."""


class MaterializationError(GraphError, _ValueErrorCompat):
    """Generated graph source cannot be assembled into a Python module."""


class ToolProtocolError(AdasError, _ValueErrorCompat):
    """A tool-call/message-history protocol violation."""


class ModelConfigurationError(ModelError, _ValueErrorCompat):
    """A requested provider, model, credential, or capability is unavailable."""


class StructuredOutputError(ModelError, _RuntimeErrorCompat):
    """Raised when a model response cannot satisfy a structured-output contract."""


class DecoratorParseError(MetaSystemError, _ValueErrorCompat):
    """Raised when decorator-style meta-tool arguments cannot be parsed."""


class MetaResponseParseError(MetaSystemError, _ValueErrorCompat):
    """Raised when a meta-agent response does not contain a usable JSON object."""


class MissingWorkflowError(EvaluationError, _RuntimeErrorCompat):
    """Raised when materialized candidate code does not define ``workflow``."""


class ValidatorDispatchError(EvaluationError, _ValueErrorCompat):
    """Raised when a validator function is missing or cannot be dispatched."""


class ValidatorContractError(EvaluationError, _TypeErrorCompat):
    """Raised when a validator returns a malformed result violating the validator contract."""


class JudgePayloadError(EvaluationError, _ValueErrorCompat):
    """Raised when a judge image payload is absent, unreadable, or unsupported."""


class JudgeMaxRetriesExceededError(EvaluationError, _RuntimeErrorCompat):
    """Raised when the structured LLM judge exhausts its retry budget."""


class SandboxConfigurationError(SandboxError, _ValueErrorCompat):
    """Raised when sandbox options or container backends are invalid."""


class SandboxRuntimeUnavailableError(SandboxError, _RuntimeErrorCompat):
    """Raised when the selected container runtime is unavailable."""


class SandboxSessionError(SandboxError, _RuntimeErrorCompat):
    """Raised when a sandbox session is unavailable or no longer running."""


class FixtureExecutionError(RuntimeEnvironmentError, _RuntimeErrorCompat):
    """Raised when a generated fixture provisioning script fails."""


class FeatureNotImplementedError(TaskSpecError, NotImplementedError, _ValueErrorCompat):
    """Raised when a TaskSpec requests a runtime capability not yet implemented."""


class MetaStateError(MetaSystemError, _ValueErrorCompat):
    """Raised when required meta-system state or tools are unavailable."""
