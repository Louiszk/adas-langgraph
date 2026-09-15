# Web Research Agent Specification

This example task specification exercises ADAS's native OpenAI Responses API server-side web search capability (`enable_web_search: true`). The target system is a single-turn autonomous research agent that investigates real-time or recent scientific and technical queries, gathers verifiable factual findings with authoritative citations, and synthesizes structured reports.

## Web Search Architecture & Provider Integration

Unlike client-side function tools that require round-trip multi-turn execution (`tool_calls` -> `ToolMessage`), OpenAI's native `web_search` provider tool executes server-side within a single `llm.invoke(...)` call:

* **Model Authorization**: Declared in `task.json` under `available_models` via `enable_web_search: true`.
* **Selective Model Instantiation**: Target nodes or helper steps equip search specifically on models that perform live information retrieval via `ChatModel(default_tools=["web_search"])`. Nodes that only classify, route, or format run with standard `ChatModel()` instances to avoid unnecessary search overhead.
* **Single-Invocation Execution**: The model queries the web, inspects pages, and synthesizes its response directly into `response.content`. `response.tool_calls` remains empty for search operations.
* **Citations & Telemetry**:
  * `response.additional_kwargs["citations"]`: List of referenced URLs, page titles, and cited snippets.
  * `response.additional_kwargs["web_search_calls"]`: Log of search queries and actions executed by the provider.
* **Tool Composition**: Custom Python tools can be combined with server-side web search via `ChatModel(default_tools=["web_search"]).bind_tools([my_tool])`.

## Running the Lifecycle for this Task

1. **Setup Fixtures & Preflight (Sandboxed):**
   ```bash
   python create_setup.py --task-spec example_specs/research_agent/task.json
   ```

2. **Generate Frozen Programmatic Validation Logic:**
   ```bash
   python create_validation.py --task-spec example_specs/research_agent/task.json
   ```

3. **Run Design Optimization (Meta-Agent):**
   ```bash
   python invoke_design.py --task-spec example_specs/research_agent/task.json
   ```

4. **Execute Target System in Sandbox:**
   ```bash
   python invoke_target.py --system-name web_research_agent --task-spec example_specs/research_agent/task.json --state '{"query": "Summarize the execution and astronaut crew of NASA'\''s Artemis II lunar flyby mission."}'
   ```
