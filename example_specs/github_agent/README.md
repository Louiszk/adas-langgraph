# GitHub Repository Research Agent Specification

This example task specification defines a single-turn multi-agent research system that investigates public GitHub repositories using the official read-only GitHub Streamable HTTP Model Context Protocol (MCP) server. A central orchestrator decomposes natural-language queries about repository architecture and source code, dynamically delegates focused research roles to up to three subagents, coordinates their findings, and synthesizes a grounded answer with file paths and line references.

## Resource Architecture & Remote MCP Integration

Rather than deploying local mock servers or synthetic fixtures, this specification exercises ADAS's `resource_manifest` to connect to the official GitHub-hosted read-only MCP service.

### Remote MCP Resource (`github_mcp`)
* **Endpoint**: `https://api.githubcopilot.com/mcp/readonly` (can be overridden via `GITHUB_MCP_URL`).
* **Capabilities**: Provides repository tree inspection, file content retrieval, symbol search, and commit/PR activity analysis for public repositories.
* **Safety Boundary**: The `/readonly` endpoint strictly guarantees that mutation-capable operations (such as creating branches, editing files, opening or merging PRs) are disabled.

### Required Tools
* **`discover_github_tools`**: Connects to the GitHub Streamable HTTP MCP server, queries `tools/list` for available read-only tool schemas, and exposes them to the orchestrator.
* **`call_github_tool`**: Dispatches requests (`tools/call`) to the discovered MCP tools with validated parameters, extracts structured evidence, and records an auditable trail in `tool_history`.

## Authentication & Environment Variables

Before running preflight or executing the target system, configure your GitHub Personal Access Token in your environment or `.env` file:

```dotenv
GITHUB_PAT=ghp_...
# Optional override for custom MCP gateways:
# GITHUB_MCP_URL=https://api.githubcopilot.com/mcp/readonly
```

The preflight check verifies that `GITHUB_PAT` is set and that the remote MCP endpoint responds to protocol discovery.

## Running the Lifecycle for this Task

1. **Setup Fixtures & Preflight (Sandboxed):**
   ```bash
   python create_setup.py --task-spec example_specs/github_agent/task.json
   ```

2. **Generate Frozen Programmatic Validation Logic:**
   ```bash
   python create_validation.py --task-spec example_specs/github_agent/task.json
   ```

3. **Run Design Optimization (Meta-Agent):**
   ```bash
   python invoke_design.py --task-spec example_specs/github_agent/task.json
   ```

4. **Execute Target System in Sandbox:**
   ```bash
   python invoke_target.py --system-name github_repository_research --task-spec example_specs/github_agent/task.json --state '{"user_request": "For the public repository encode/starlette, identify its primary purpose, core directory layout, and main application entry point."}'
   ```
