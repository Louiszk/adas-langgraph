# Neo4j Graph Exploration Agent Specification

This example task specification exercises ADAS's external database fixture seeding capabilities (`test_fixtures.external_database_seeds`). The target system is a general-purpose, read-only Graph Intelligence and Text-to-Cypher agent that dynamically connects to any Neo4j instance declared via environment variables (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`), discovers the graph schema at runtime, and executes read-only Cypher queries to answer multi-hop graph traversal and relationship queries.

## External Database Seed Fixtures

Rather than hardcoding domain-specific knowledge, the test environment provisions two distinct isolated database namespaces across test cases using `ExternalDatabaseSeedSpec`:

### 1. `seed_company_network` (`adas-test-company-network`)
An organizational network graph modeling employees, management hierarchy, project allocations, and technical skills:
* **Nodes**: `Person` (name, role, department), `Project` (name, status, budget), `Skill` (name, category).
* **Relationships**: `(:Person)-[:MANAGES]->(:Person)`, `(:Person)-[:ASSIGNED_TO {role_in_project, hours_per_week}]->(:Project)`, `(:Person)-[:HAS_SKILL {proficiency}]->(:Skill)`.
* **Test Case**: Multi-hop path exploration to find an engineer reporting to a specific director assigned to a specific project with a required skill.

### 2. `seed_dependency_graph` (`adas-test-dependency-graph`)
A software microservice and package dependency graph modeling services and packages:
* **Nodes**: `Service` (name, tier, language), `Package` (name, version, license).
* **Relationships**: `(:Service)-[:CALLS]->(:Service)`, `(:Service)-[:USES]->(:Package)`, `(:Package)-[:DEPENDS_ON]->(:Package)`.
* **Test Case**: Transitive dependency traversal from a high-level service down through intermediate packages to find the complete dependency chain.

## Running Neo4j Locally

Before running the lifecycle or tests with this spec, ensure a running Neo4j instance is available.
Set the connection environment variables in the project `.env` file:

```dotenv
NEO4J_URI=bolt://host.docker.internal:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=adaspassword123
```


## Running the Lifecycle for this Task

1. **Setup Fixtures & Preflight:**
   ```bash
   python create_setup.py --task-spec example_specs/neo4j_agent/task.json
   ```

2. **Generate Programmatic Validation Logic:**
   ```bash
   python create_validation.py --task-spec example_specs/neo4j_agent/task.json
   ```

3. **Run Design Optimization (Meta-Agent):**
   ```bash
   python invoke_design.py --task-spec example_specs/neo4j_agent/task.json
   ```

4. **Execute Target System in Sandbox:**
   ```bash
   python invoke_target.py --system-name neo4j_agent --task-spec example_specs/neo4j_agent/task.json --state '{"user_request": "Inspect the graph schema and find which engineer who directly reports to Director Sarah is assigned to Project Titan."}'
   ```
