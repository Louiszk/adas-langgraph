# Social Media Agent Specification

This example task specification exercises ADAS's process-backed HTTP mock-service fixture (`test_fixtures.mock_services`). The target system is a single-turn social media agent that answers questions about a deterministic social network feed and can safely publish grounded updates only when explicitly authorized.

## Mock Service Architecture & Endpoints

Rather than calling live external social APIs, the test environment provisions a deterministic local JSON HTTP service (`mock_social_index_service.py`) bound to port 8765 and exposed via the `SOCIAL_INDEX_BASE_URL` environment variable:

* **`GET /trends`**: Returns ranked current trends (Harbor Bridge closure, Aurora Studios game-patch outage, Riverlight Festival opening) with topic summaries, post counts, and representative post IDs.
* **`GET /users/{handle}`**: Returns profile metadata (display name, verification status, bio, follower count) for known accounts (`@citydesk`, `@mara_chen`, `@aurora_status`, `@pixelpilot`, `@riverlightfest`, `@northstarnews`).
* **`GET /users/{handle}/posts?limit=N`**: Returns chronological post history for an account in descending order.
* **`GET /events/{event_id}`**: Returns authoritative civic/event context (e.g. `harbor_bridge` safety inspection status, `aurora_patch` degraded service status).
* **`POST /search/posts`**: Performs structured query searches over post records with optional filters for handles, event IDs, and time bounds.
* **`POST /posts`**: Gated publishing endpoint. Requires text between 1 and 280 characters and one or more valid seeded `evidence_post_ids`. Publishes under `@adas_social_agent` with HTTP 201.
* **`GET /audit/published-posts`**: Read-only audit log containing all published posts in creation order.

## Running the Lifecycle for this Task

1. **Setup Fixtures & Preflight (Sandboxed):**
   ```bash
   python create_setup.py --task-spec example_specs/social_agent/task.json
   ```

2. **Generate Frozen Programmatic Validation Logic:**
   ```bash
   python create_validation.py --task-spec example_specs/social_agent/task.json
   ```

3. **Run Design Optimization (Meta-Agent):**
   ```bash
   python invoke_design.py --task-spec example_specs/social_agent/task.json
   ```

4. **Execute Target System in Sandbox:**
   ```bash
   python invoke_target.py --system-name social_intelligence_agent --task-spec example_specs/social_agent/task.json --state '{"user_request": "What are the biggest stories on the network today? Do not post anything."}'
   ```
