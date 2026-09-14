# Model Context Protocol (MCP) Agent Specification

This example task specification exercises ADAS's process-backed Streamable HTTP Model Context Protocol fixtures (`test_fixtures.mcps`). The target system is a general-purpose single-turn MCP agent that dynamically connects to active MCP servers declared via environment variables (e.g. `WORKSPACE_MCP_URL`, `OPERATIONS_MCP_URL`), discovers available tools via the MCP protocol (`tools/list`), binds them to its model, and executes tool calls (`tools/call`) to fulfill user requests without hardcoding domain-specific tools.

## MCP Fixture Architecture & Exposed Tools

Rather than hardcoding tool implementations directly inside the agent code, the test environment provisions two distinct Streamable HTTP MCP server fixtures across test cases:

### 1. `workspace_mcp` (Port 8810, `WORKSPACE_MCP_URL`)
Provides workspace knowledge retrieval and quantitative analytics:
* **`search_docs(query: str, category: str = "all") -> dict`**: Searches engineering policy documentation (e.g. database connection retry policy: `max_retries=5`, `backoff_seconds=2.0`; blue-green deployment guides).
* **`calculate_metric(metric_name: str, values: list[float]) -> dict`**: Computes statistical metrics (`mean`, `median`, `p95`, `sum`) over numeric measurements.
* **`GET /audit/calls`**: Custom audit route returning records of all executed tool calls.

### 2. `operations_mcp` (Port 8820, `OPERATIONS_MCP_URL`)
Provides stateful, multi-step incident diagnostics and remediation:
* **`get_service_health(service_name: str) -> dict`**: Returns service status and active incident details (e.g. degraded status with `incident_id: "INC-4091"` for `auth-service`).
* **`fetch_incident_logs(incident_id: str, max_lines: int = 5) -> dict`**: Returns recent error logs revealing root causes (e.g. database connection pool exhaustion).
* **`apply_mitigation(incident_id: str, action: str) -> dict`**: Executes approved remediation actions (`restart_service`, `resize_connection_pool`, `failover`) to resolve active incidents.
* **`GET /audit/remediations`**: Custom audit route returning records of all applied incident mitigations.

## Running the Lifecycle for this Task

1. **Setup Fixtures & Preflight (Sandboxed):**
   ```bash
   python create_setup.py --task-spec example_specs/mcp_agent/task.json
   ```

2. **Generate Frozen Programmatic Validation Logic:**
   ```bash
   python create_validation.py --task-spec example_specs/mcp_agent/task.json
   ```

3. **Run Design Optimization (Meta-Agent):**
   ```bash
   python invoke_design.py --task-spec example_specs/mcp_agent/task.json
   ```

4. **Execute Target System in Sandbox:**
   ```bash
   python invoke_target.py --system-name mcp_agent --task-spec example_specs/mcp_agent/task.json --state '{"user_request": "Search our engineering documentation for our database connection retry policy and state the max retries."}'
   ```
