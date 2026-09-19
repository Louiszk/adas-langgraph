---
name: adas-task-lifecycle
description: Create, refine, or verify an ADAS TaskSpec and its frozen setup and validation artifacts in this repository.
---

# Automated Design of Agentic Systems (ADAS) task lifecycle

Help the user turn an agent requirement into a valid, testable ADAS task directory. The deliverables are a `task.json`, its deterministic fixture/setup artifacts, and a frozen validation module that reflect the user's actual contract.

## Required order and checks

The preferred workflow is:
1. Discuss the task with the user and create `specs/<spec-dir>/task.json`.
2. Once the user approves it, make sure Docker or Podman is active before running the creation scripts for setup and validation.
3. Verify the generated artifacts for correctness; if generation produces mistakes, fix them and regenerate the hash.
4. When the `--verify` flags pass, inform the user that the design process can now be invoked.

This workflow can vary depending on how the user wants to handle the task. Ask clarifying questions when needed.

Some notes:
  - Use web search if concepts, dependencies, resources, or other details are outside your current knowledge.
  - Use this repository's virtual environment unless the user has configured another supported environment.
  - Setup requires the configured Docker/Podman sandbox. Use `--force` only when regeneration is intended.
  - Never run `create_taskspec.py` in interactive mode. Use `--non-interactive` when necessary.
  - Report missing engine or model credentials as a blocker rather than fabricating generated artifacts.
  - Keep fixture seed details private to fixtures and validators, and do not reveal hidden answers to the meta-agent.
  - Additional documentation for smaller or unfamiliar libraries or resources may be provided to the meta-agent.

## File references

For context, please read:
- `README.md`
- `adas_core/task_spec.py` + `adas_core/chat_model.py`
- `create_taskspec.py` + `adas_core/automatic_taskspec.py`
- `create_setup.py` + `adas_core/automatic_setup.py`
- `create_validation.py` + `adas_core/automatic_validation.py`

Optionally, start from a similar example:

- File/data processing: `example_specs/data_analyst_agent/task.json`
- HTTP MCP fixtures: `example_specs/mcp_agent/task.json` and `example_docs/mcp-documentation.md`
- External database integration: `example_specs/neo4j_agent/task.json`
- General research or multi-step work: `example_specs/research_agent/task.json`
