"""Ahead-of-time interactive CLI and LLM synthesizer for validated TaskSpec specifications."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from adas_core.chat_model import ChatModel, ModelRegistry, usage_scope
from adas_core.helpers import sanitize_identifier
from adas_core.markdown_parser import (
    extract_json_block,
    extract_json_block_optional,
    find_markdown_fences,
)
from adas_core.task_spec import TaskSpec
from config.logging import get_logger
from config.settings import taskspec_enable_web_search, taskspec_model, taskspec_reasoning_effort, taskspec_wrapper

logger = get_logger("adas_core.automatic_taskspec")

DEFAULT_SPECS_DIR = Path("specs")


def format_assistant_message_for_display(content: str, saved_path: Path | None = None) -> str:
    """Format the assistant response for terminal display by replacing raw TaskSpec JSON blocks with a clean banner."""
    blocks = find_markdown_fences(content)
    lines = content.splitlines()

    target_block: dict[str, Any] | None = None
    for block in blocks:
        raw = str(block.get("content", "")).strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "schema_version" in parsed and "name" in parsed:
                target_block = block
                break
        except Exception:
            continue

    if not target_block:
        return content.strip()

    start_line = int(target_block["start_line"])
    end_line = int(target_block["end_line"])

    dest = f" to: {Path(saved_path).as_posix()}" if saved_path else ""
    banner = [
        "-" * 60,
        f"  [Draft TaskSpec persisted{dest}]",
        "  (Inspect file in your editor or continue discussing adjustments)",
        "-" * 60,
    ]

    before_lines = lines[: max(0, start_line - 1)]
    after_lines = lines[end_line:]

    return "\n".join(before_lines + banner + after_lines).strip()


def build_model_catalog_context() -> str:
    """Query ModelRegistry to build a formatted catalog of known models and capabilities."""
    with ModelRegistry._lock:
        if not ModelRegistry._capabilities:
            ModelRegistry._init_defaults()
        caps_dict = dict(ModelRegistry._capabilities)

    lines = [
        "### AVAILABLE TARGET MODELS & CAPABILITIES CATALOG:",
        "When declaring `available_models` or assigning a `judge_model` for a test case, choose from:",
        "| Provider | Model Name | Supports Vision | Supports Temp | Supports Reasoning Effort | Supports Web Search |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for (provider, model_name), caps in sorted(caps_dict.items()):
        vis = "YES" if caps.supports_vision else "NO (text-only)"
        temp = "YES" if caps.supports_temperature else "NO"
        reason = "YES (" + ", ".join(caps.supported_reasoning_efforts) + ")" if caps.supports_reasoning_effort else "NO"
        web = "YES" if caps.supports_web_search else "NO"
        lines.append(f"| `{provider}` | `{model_name}` | {vis} | {temp} | {reason} | {web} |")

    lines.extend(
        [
            "",
            "CRITICAL MODEL & MODALITY CONSTRAINTS:",
            "1. VISION EVALUATION: If a test case involves vision (e.g. modalities includes 'vision'):",
            "   - `llm_judge_needed` must be True.",
            "   - `judge_model` (if set) must support vision (e.g. 'gpt-4o', 'o3', 'gpt-5.6-luna').",
            "2. REASONING EFFORT: Models supporting reasoning effort (like o3, gpt-5.4, gpt-5.6) do not support temperature.",
            "3. NO UNREGISTERED MODELS: Only declare models that exist in this catalog. If the user requests a model not available in the catalog, suggest that the user may update the `ModelRegistry`.",
            "4. WEB SEARCH CAPABILITIES: Models supporting web search can be granted external browsing capabilities by setting `enable_web_search: true` on their `ModelSpec` in `available_models`. Test cases requiring web search verification by judges can declare `judge_web_search: true` (which requires a `judge_model` supporting web search).",
        ]
    )

    return "\n".join(lines)


def build_architect_system_prompt(task_dir_hint: str | None = None) -> str:
    """Build the conversational system prompt for the interactive architect model."""
    schema_json = json.dumps(TaskSpec.model_json_schema(), indent=2)
    catalog_context = build_model_catalog_context()
    target_dir = task_dir_hint or "specs/<task_name>/"

    return f"""You are an expert AI agentic system architect conducting an interactive requirements elicitation and specification interview for ADAS (Automated Design of Agentic Systems).
In ADAS, an autonomous meta-agent automatically designs, implements, and refines a LangGraph target system based on the synthesized task specification and development test cases.

Your mission is to collaborate with the user to design a robust, production-grade agentic system, explore architectural trade-offs, and produce an authoritative, validated `TaskSpec` JSON specification.

### INTERACTION GUIDELINES:
1. GRILL & CLARIFY AMBIGUITIES:
   - When the user describes what they want to build, critically analyze their idea.
   - Do NOT immediately jump to a premature or generic specification if key details are missing or ambiguous.
   - Ask 2 to 3 targeted, insightful questions to probe unstated requirements:
     * Execution Mode: Does this need a single-turn input -> output pipeline, or a multi-turn conversational loop with stateful memory and checkpointer?
     * LangGraph State Schema: What are the input keys, intermediate scratchpad keys, and output keys?
     * Tools & Dependencies: What external libraries (e.g. pandas, requests) and API keys (e.g. SERPER_API_KEY) will it require?
     * Resource ownership: Ask whether required MCP servers, HTTP APIs, or databases already run under the user's control, or whether the evaluation needs a controlled fixture.
     * Available Models: Which models from the catalog are needed? (Check vision/reasoning constraints).
     * Web Search & Live Retrieval: Does the target agent require server-side web search capabilities? If so, configure `enable_web_search: true` for authorized models in `available_models`. Do test cases require the evaluation judge to verify external facts against live web results (`judge_web_search: true`)?
   - Present trade-offs and options clearly (e.g. "Option A: single-turn pipeline vs Option B: multi-turn agent with memory") to help the user choose.
   - Do NOT quiz or consult internal ADAS schema plumbing to the user (e.g. how sandbox workspace directories map or relative fixture paths). Handle all such architectural plumbing silently and automatically according to the mandatory rules.

2. DRAFTING AND UPDATING THE SPECIFICATION:
   - When key dimensions are sufficiently clarified, OR when the user asks you to draft or update the spec:
     * Embed the complete, schema-compliant `TaskSpec` JSON inside a fenced code block ```json ... ```.
     * Accompany the JSON with conversational commentary:
       - Inform the user that the draft has been generated and will be saved to `{target_dir}<clean_name>.task.json`.
       - Summarize the key architectural choices made (state keys, tools, test cases).
       - Point out specific aspects you recommend the user inspect or consider adjusting.
       - Ask if they would like to add edge-case test scenarios or adjust any parameters.

3. CONTINUOUS COLLABORATIVE REFINEMENT:
   - The conversation does NOT end upon generating the first draft!
   - The user will inspect the generated file, answer questions, or ask for adjustments (e.g., "Change the judge model to gpt-4o", "Add an edge-case test for missing CSV columns").
   - Discuss their requested changes and provide an updated, complete `TaskSpec` JSON code block.
   - Continue refining until the user is completely satisfied and indicates they are done.

### AUTHORITATIVE TASK SPECIFICATION JSON SCHEMA:
Your output JSON must strictly conform to the following JSON Schema generated directly from the TaskSpec Pydantic model:
```json
{schema_json}
```

{catalog_context}

### MANDATORY ARCHITECTURAL RULES:
1. SINGLE-TURN VS. MULTI-TURN CONTRACT:
   - If execution_mode is 'single_turn', persistence must be null, and each test case turn must provide the required input state.
   - If execution_mode is 'multi_turn', state_schema must declare a message history key (e.g. 'messages'), and persistence must specify 'checkpointer': 'memory' and 'requires_thread_id': true.

2. VALID JSON OUTPUT:
   - When providing a specification, output a complete, valid JSON object in a ```json ... ``` block.

3. FIXTURE DEFINITIONS AND TEST CASE SCOPING:
   - Each fixture declared in `test_fixtures` (files, databases, mcps, mock_services, custom_fixtures, external_database_seeds) must have a clean unique `id` (e.g. 'sales_csv', 'customers_json', 'weather_api', 'analytics_db').
   - Keep public interface contracts, schemas, table/node definitions, and API routes in `description` (visible to the meta-agent). Put deterministic evaluation seed data, specific rows/records, planted secrets, or test-bench ground truth in `private_description` (withheld from the meta-agent to prevent overfitting).
   - When declaring file fixtures in `test_fixtures.files`, `path` must be relative to the input folder (e.g. "sales.csv", "customers.json"). Never use hardcoded sandbox prefixes or absolute paths.
   - In `dev_suite`, each test case SHOULD specify `fixture_ids: ["fixture_id_1", ...]` to declare the exact subset of fixtures provisioned into its environment.
     * Tier 1 tests should provision only clean baseline fixtures.
     * Tier 2 tests should provision the relational/multi-fixture subset needed for core logic.
     * Tier 3 tests should provision dirty or edge-case fixtures to ensure robustness.
   - In `resource_manifest.available_resources`, declare input and output directories generically (e.g. "data/input/" or "input/") and specify in descriptions that systems should inspect files dynamically or use standard environment variables (`ADAS_INPUT_DIR`, `ADAS_OUTPUT_DIR`) rather than assuming static host paths.
   - Keep user-owned resources in `resource_manifest`, including files, directories, running services, and their connection URL/URI. Put required credentials in `available_api_keys`; preflight verifies those resources rather than starting or mocking them.
   - Use `test_fixtures` only for harness-owned, deterministic evaluation resources. The harness can mock Streamable HTTP MCP servers and HTTP services, and can create and seed embedded SQLite or DuckDB database files.
   - Do NOT represent an external database service (for example Postgres, Neo4j, Redis, or Qdrant) as a mock fixture: it must already be running and be declared in `resource_manifest`.
   - To seed deterministic evaluation data, declare `test_fixtures.external_database_seeds` with a unique id, the matching database resource name, engine/driver, connection_env mapping containing ENVIRONMENT VARIABLE NAMES only (never credentials), an explicit namespace_kind, cleanup_policy `drop_namespace`, public schema/ontology in `description`, and evaluation-only seed records in `private_description`.
   - Use `adas_test_...` for PostgreSQL namespaces because PostgreSQL rejects unquoted hyphens; use `adas-test-...` for Neo4j database namespaces because Neo4j rejects underscores.

4. DEV SUITE 3-TIER PROGRESSION:
   - The test cases in `dev_suite` must follow a strictly progressive difficulty gradient:
     * Tier 1 (Baseline / Smoke Test): Minimal viable happy-path test on clean, standard input with 1-2 primary expected outputs.
     * Tier 2 (Core Functional Complexity): Primary domain logic exercising core analytical or multi-step capabilities.
     * Tier 3 (Resilience / Edge Cases): Defensive handling of realistic anomalies without crashing, reporting caveats or skipped records transparently.
   - If `execution_mode` is 'single_turn', ensure each test turn's prompt describes an objective achievable in a single graph invocation without multi-turn clarification.
   - Test case prompts should explicitly identify which input files or resources they target, matching their declared `fixture_ids`.
"""


def build_taskspec_system_prompt() -> str:
    """Convenience alias for build_architect_system_prompt."""
    return build_architect_system_prompt()


class AutomaticTaskSpec:
    """Synthesizer and validator for TaskSpec specifications using LLMs."""

    def __init__(self, llm: ChatModel | None = None) -> None:
        self.llm = llm or ChatModel(
            provider=taskspec_wrapper,
            model=taskspec_model,
            reasoning_effort=taskspec_reasoning_effort,
            name="AutomaticTaskSpec",
            is_meta=True,
            default_tools=["web_search"] if taskspec_enable_web_search else None,
        )

    def _invoke_model(self, messages: list[BaseMessage]) -> Any:
        with usage_scope(system="meta", node="automatic_taskspec"):
            return self.llm.invoke(messages)

    def chat(self, messages: list[BaseMessage]) -> str:
        """Invoke the chat model with conversation messages and return the response text."""
        response = self._invoke_model(messages)
        return str(response.content)

    def generate_task_spec(
        self,
        requirements: str | dict[str, Any],
        max_retries: int = 3,
    ) -> TaskSpec:
        """Synthesize and validate a TaskSpec from user requirements."""
        system_prompt = build_taskspec_system_prompt()

        if isinstance(requirements, dict) and "existing_spec" in requirements:
            existing = requirements["existing_spec"]
            instructions = requirements.get("system_goal", "")
            req_text = (
                f"Existing TaskSpec specification to refine:\n"
                f"```json\n{json.dumps(existing, indent=2)}\n```\n\n"
                f"Refinement instructions:\n{instructions}"
            )
        else:
            req_text = json.dumps(requirements, indent=2) if isinstance(requirements, dict) else str(requirements)

        user_prompt = (
            "Please synthesize a complete, validated TaskSpec JSON specification based on the following requirements:\n\n"
            f"{req_text}\n\n"
            "Ensure the output strictly conforms to the TaskSpec schema and respects all model/modality constraints. "
            "Output the full JSON inside a ```json ... ``` code block."
        )

        conversation: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        for attempt in range(max_retries):
            logger.info("Generating TaskSpec (attempt %d/%d)...", attempt + 1, max_retries)
            response = self._invoke_model(conversation)
            content = str(response.content)

            try:
                raw_json = extract_json_block(content)
                task_spec = TaskSpec.model_validate(raw_json)
                logger.info("Successfully synthesized and validated TaskSpec for '%s'", task_spec.name)
                return task_spec
            except (ValueError, ValidationError) as exc:
                logger.warning("Validation failed on attempt %d: %r", attempt + 1, exc)
                if attempt == max_retries - 1:
                    raise ValueError(
                        f"Failed to generate a valid TaskSpec after {max_retries} attempts: {exc!r}"
                    ) from exc

                # Feed the validation error back for correction
                conversation.append(
                    HumanMessage(
                        content=(
                            f"The generated JSON failed validation with the following error:\n{exc!r}\n\n"
                            "Please correct the JSON specification and return only the updated valid JSON inside a ```json ... ``` block."
                        )
                    )
                )

        raise RuntimeError("Unreachable TaskSpec generation loop termination")

    def save_task_spec(
        self,
        task_spec: TaskSpec,
        output_dir: Path | str | None = None,
        filename: str | None = None,
    ) -> Path:
        """Write the TaskSpec JSON to disk without calling AutomaticSetup."""
        clean_name = sanitize_identifier(task_spec.name.lower())
        if output_dir:
            out_p = Path(output_dir)
            if out_p.suffix == ".json":
                base_dir = out_p.parent
                target_file = out_p
            else:
                base_dir = out_p
                target_file = base_dir / (filename or f"{clean_name}.task.json")
        else:
            base_dir = DEFAULT_SPECS_DIR / clean_name
            target_file = base_dir / (filename or f"{clean_name}.task.json")

        base_dir.mkdir(parents=True, exist_ok=True)
        target_file.write_text(task_spec.to_json(indent=2), encoding="utf-8")
        logger.info("Saved TaskSpec to %s", target_file)
        return target_file


def find_existing_task_spec_file(
    output_dir: Path | str | None = None,
    task_name: str | None = None,
) -> Path | None:
    """Check if an existing TaskSpec JSON file exists for the given output_dir and/or task_name."""
    if output_dir:
        out_path = Path(output_dir)
        if out_path.is_file():
            return out_path
        if out_path.is_dir():
            if task_name:
                clean_name = sanitize_identifier(task_name.lower())
                named_file = out_path / f"{clean_name}.task.json"
                if named_file.is_file():
                    return named_file
            task_files = list(out_path.glob("*.task.json"))
            if len(task_files) == 1:
                return task_files[0]

    if task_name:
        clean_name = sanitize_identifier(task_name.lower())
        default_file = DEFAULT_SPECS_DIR / clean_name / f"{clean_name}.task.json"
        if default_file.is_file():
            return default_file

    return None


def run_interactive_wizard(
    initial_prompt: str | None = None,
    output_dir: Path | str | None = None,
    non_interactive: bool = False,
    task_name: str | None = None,
    generator: AutomaticTaskSpec | None = None,
) -> TaskSpec | None:
    """Run an interactive consultative interview to elicit requirements and synthesize a TaskSpec."""
    gen = generator or AutomaticTaskSpec()

    existing_file = find_existing_task_spec_file(output_dir=output_dir, task_name=task_name)
    existing_spec: TaskSpec | None = None
    existing_content: str | None = None

    if existing_file is not None:
        try:
            existing_content = existing_file.read_text(encoding="utf-8")
            existing_spec = TaskSpec.from_json(existing_content)
            logger.info("Found existing TaskSpec for '%s' at %s", existing_spec.name, existing_file)
        except Exception as exc:
            logger.warning("Failed to parse existing TaskSpec from %s: %r", existing_file, exc)
            existing_file = None
            existing_spec = None
            existing_content = None

    # Non-interactive mode (for automation, CLI flags, tests)
    if non_interactive:
        if not initial_prompt:
            raise ValueError("Non-interactive mode requires a prompt or problem statement.")
        requirements: dict[str, Any] = {"system_goal": initial_prompt}
        if existing_spec is not None:
            requirements["existing_spec"] = existing_spec.model_dump()
        if task_name:
            requirements["name"] = task_name
        task_spec = gen.generate_task_spec(requirements)
        saved_path = gen.save_task_spec(task_spec, output_dir=output_dir)
        print(f"TaskSpec saved successfully to: {saved_path}")
        return task_spec

    # Interactive consultative interview
    print("\n" + "=" * 65)
    print("  ADAS TaskSpec CLI")
    print("=" * 65)
    print("Welcome! I will help you design, clarify, and synthesize a validated TaskSpec.")
    print("Describe what you want to build. We'll discuss requirements, options, and trade-offs.")
    print("Whenever ready, I'll generate and persist draft schemas directly to disk.")
    if existing_spec is not None and existing_file is not None:
        print(f"\n[i] Found existing TaskSpec for '{existing_spec.name}' at:")
        print(f"    {existing_file}")
        print("    Loaded current specification. Describe any modifications, additions,")
        print("    or say 'start over' to reset.")
    print("\nCommands at any time:")
    print("  'done'   - Finalize and exit with the current TaskSpec")
    if existing_spec is not None:
        print("  'reset'  - Discard loaded draft and start over from scratch")
    print("  'exit'   - Cancel and exit")
    print("=" * 65 + "\n")

    target_dir_hint = str(output_dir) if output_dir else "specs/<task_name>/"
    system_prompt = build_architect_system_prompt(task_dir_hint=target_dir_hint)
    conversation: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    current_spec: TaskSpec | None = existing_spec
    saved_path: Path | None = existing_file
    pending_user_message: str | None = initial_prompt
    conversation_started = False

    if existing_spec is not None and existing_content is not None:
        conversation.append(
            HumanMessage(
                content=(
                    f"An existing TaskSpec specification was loaded from '{existing_file}':\n"
                    f"```json\n{existing_content}\n```\n"
                    "The user wants to inspect or refine this existing specification."
                )
            )
        )
        conversation_started = True

    if not pending_user_message and existing_spec is None:
        print("Describe the agentic system you would like to build:")
    elif not pending_user_message and existing_spec is not None:
        print(f"What changes or additions would you like to make to '{existing_spec.name}'?")

    while True:
        if pending_user_message:
            user_line = pending_user_message.strip()
            pending_user_message = None
            print(f"You: {user_line}")
        else:
            try:
                user_line = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                return current_spec

        if not user_line:
            continue

        cmd = user_line.lower()
        if cmd in ("done", "/done", "finish", "/finish"):
            if current_spec is not None:
                print("\n" + "=" * 65)
                print("  [OK] TaskSpec Finalized & Saved Successfully!")
                print("=" * 65)
                print(f"Path: {saved_path}")
                return current_spec
            else:
                print(
                    "\n[!] No TaskSpec has been drafted yet. Please describe your agent or continue the discussion first.\n"
                )
                continue

        if cmd in ("reset", "/reset", "start over", "/start over"):
            print("\nResetting to a clean slate.")
            current_spec = None
            saved_path = None
            conversation = [SystemMessage(content=system_prompt)]
            conversation_started = False
            print("Describe the agentic system you would like to build:")
            continue

        if cmd in ("exit", "/exit", "quit", "/quit"):
            if current_spec is not None:
                try:
                    confirm = input(f"Draft is saved at {saved_path}. Exit now? ([y]/n): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    confirm = "y"
                if confirm in ("", "y", "yes"):
                    print("Exiting.")
                    return current_spec
                continue
            print("Exiting.")
            return None

        if cmd in ("help", "/help"):
            print("\nAvailable commands:")
            print("  'done'   - Finalize and exit with the current TaskSpec")
            print("  'reset'  - Discard current draft and start over from scratch")
            print("  'exit'   - Exit")
            print("  Or describe your system or feedback to continue the discussion.\n")
            continue

        # Regular conversational input: append to conversation and invoke architect
        if not conversation_started and task_name and existing_spec is None:
            content_to_send = f"Proposed System Name: {task_name}\n\nSystem Goal / Description:\n{user_line}"
        else:
            content_to_send = user_line
        conversation_started = True

        conversation.append(HumanMessage(content=content_to_send))

        print("\n[Architect is thinking...]")
        response_text = gen.chat(conversation)
        conversation.append(AIMessage(content=response_text))

        candidate_json = extract_json_block_optional(response_text)
        if candidate_json is not None:
            try:
                current_spec = TaskSpec.model_validate(candidate_json)
                saved_path = gen.save_task_spec(current_spec, output_dir=output_dir)
                clean_display = format_assistant_message_for_display(response_text, saved_path=saved_path)
            except (ValueError, ValidationError) as exc:
                logger.warning("TaskSpec validation failed: %r", exc)
                clean_display = format_assistant_message_for_display(response_text)
                clean_display += (
                    f"\n\n[Notice: Draft schema contained a validation issue: {exc!r}. The architect can correct it.]"
                )
        else:
            clean_display = response_text

        print(f"\nArchitect:\n{clean_display}\n")
