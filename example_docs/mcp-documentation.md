# Model Context Protocol (MCP) Python Client SDK Reference

The Model Context Protocol (MCP) allows client applications to connect to external MCP servers to discover and invoke tools, read resources, and retrieve prompts.

In `mcp>=2`, client applications primarily interact through the unified high-level `Client` class, which handles transport lifecycle, protocol negotiation, response caching, and multi-round-trip interactions.

## 1. Client Architecture: High-Level `Client` vs. Low-Level `ClientSession`

The MCP Python SDK provides two levels of abstraction for client applications:

- **High-Level `Client` (`from mcp import Client`)**: The primary interface for applications. It wraps transport setup, automatic protocol negotiation (`mode="auto"`), response caching, and handles multi-round-trip interactions (such as tool elicitation prompts).
- **Low-Level `ClientSession` (`from mcp import ClientSession`)**: The raw JSON-RPC protocol session engine. Use this when you require manual control over stream pipes, raw JSON-RPC messages, or custom protocol dispatchers.

```python
from mcp import Client


async def connect_and_use():
    # Connects over Streamable HTTP by default when given a URL string
    async with Client("http://127.0.0.1:8000/mcp") as client:
        # Connected properties are available immediately on entry:
        print("Negotiated Version:", client.protocol_version)
        print("Server Info:", client.server_info)
        print("Server Capabilities:", client.server_capabilities)

        # Execute protocol operations
        tools = await client.list_tools()
        result = await client.call_tool("example_tool", {"param": "value"})
```

> **Lifecycle Note**: `Client` must be entered using `async with`. Construction (`Client(...)`) only configures the client; the connection is opened on `__aenter__` and torn down cleanly on `__aexit__`. A `Client` instance cannot be reused once exited.

---

## 2. Streamable HTTP Transport

Streamable HTTP is the standard HTTP transport for `mcp>=2`.

### Direct URL Connection

For endpoints without custom headers or authentication:
```python
from mcp import Client

async with Client("http://127.0.0.1:8000/mcp") as client:
    tools = await client.list_tools()
```

### Custom Headers, Timeouts, and Authentication

When custom headers, timeouts, or authentication are required, build an `httpx2.AsyncClient` and pass it to `streamable_http_client`:

```python
import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

# Note: The SDK depends on httpx2 (not httpx).
http_client = httpx2.AsyncClient(
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    timeout=httpx2.Timeout(30.0, read=300.0),
)

async with http_client:
    transport = streamable_http_client("http://127.0.0.1:8000/mcp", http_client=http_client)
    async with Client(transport) as client:
        tools = await client.list_tools()
```

> **Redirect Policy**: The transport only follows redirects within the same origin (e.g., `/mcp` -> `/mcp/`). Cross-origin redirects are intentionally not followed and raise `MCPError`.

---

## 3. Protocol Version Negotiation & Connection Modes

The `mode` parameter on `Client` controls how protocol versions are negotiated:

- `mode="auto"` (default): Probes the modern `server/discover` endpoint. If unsupported, automatically falls back to the legacy `initialize` handshake. Works seamlessly across both 2026-era and 2025-era servers.
- `mode="legacy"`: Forces the pre-2026 `initialize` handshake (spec versions `<=2025-11-25`).
- `mode="2026-07-28"` (or any modern version string): Adopts the specified protocol version directly without probing.

```python
# Force legacy handshake when targeting older servers or legacy test suites
async with Client("http://127.0.0.1:8000/mcp", mode="legacy") as client:
    ...
```

---

## 4. Tool Discovery (`client.list_tools`)

Servers advertise available tools via `client.list_tools()`. In `mcp>=2`, model fields use standard Python `snake_case`:

```python
response = await client.list_tools()

for tool in response.tools:
    name: str = tool.name
    title: str | None = tool.title  # Optional human-readable display title
    description: str | None = tool.description
    input_schema: dict = tool.input_schema  # Tool arguments JSON schema (snake_case)
```

### Pagination

Every `list_*` method accepts a `cursor` parameter and returns `next_cursor`:

```python
cursor = None
all_tools = []

while True:
    page = await client.list_tools(cursor=cursor)
    all_tools.extend(page.tools)
    cursor = page.next_cursor
    if cursor is None:
        break
```

*(Note: On the lower-level `ClientSession.list_tools()`, pass `params=PaginatedRequestParams(cursor=cursor)` instead).*

---

## 5. Tool Invocation (`client.call_tool`)

Tools are invoked by name with a dictionary of arguments matching the tool's schema:

```python
from mcp.types import TextContent, ImageContent

result = await client.call_tool(
    "query_database",
    arguments={"query": "SELECT * FROM users", "limit": 10},
    read_timeout_seconds=30.0,  # float seconds (not timedelta)
)
```

### Inspecting Execution Results

A tool encountering an error does **not** raise a Python exception; it returns an in-band result with `is_error=True`:

```python
if result.is_error:
    print("Tool reported an error:")
    for block in result.content:
        if isinstance(block, TextContent):
            print(block.text)
else:
    print("Tool executed successfully.")
```

### Structured Content vs. Content Blocks

In `mcp>=2`, a tool result provides both formats:

- `result.structured_content`: Machine-readable output (Pydantic model dump or dict) conforming to the tool's `output_schema`.
- `result.content`: A sequence of content blocks (`TextContent`, `ImageContent`, `AudioContent`, `ResourceLink`, `EmbeddedResource`) intended for LLM context.

```python
# 1. Read structured data directly in application code
if result.structured_content is not None:
    user_id = result.structured_content.get("id")

# 2. Extract content blocks for model context or UI rendering
for block in result.content:
    if isinstance(block, TextContent):
        print("Text:", block.text)
    elif isinstance(block, ImageContent):
        print(f"Image ({block.mime_type}): {len(block.data)} bytes")
```

### Timeouts & Cancellation

- Timeouts take `float` seconds (e.g., `read_timeout_seconds=60.0`).
- Exceeding the timeout raises `MCPError` with `code == REQUEST_TIMEOUT` (`-32001`, imported from `mcp.types`).
- Cancelling the awaiting coroutine automatically dispatches a cancellation notification (`notifications/cancelled`) to the server to terminate work.

---

## 6. Error Handling

- **Tool Failures**: Handled via `result.is_error = True` with details in `result.content`. Does not raise an exception.
- **Protocol & Network Errors**: Raise `MCPError` (`from mcp import MCPError`).
- **Request Timeouts**: Raise `MCPError` with `code == REQUEST_TIMEOUT` (`-32001`).

```python
from mcp import MCPError
from mcp.types import INVALID_PARAMS, REQUEST_TIMEOUT

try:
    result = await client.call_tool("generate_report", {"year": 2026})
except MCPError as exc:
    if exc.code == REQUEST_TIMEOUT:
        print("Request timed out. Server took too long to respond.")
    elif exc.code == INVALID_PARAMS:
        print(f"Invalid arguments: {exc.message}")
    else:
        print(f"MCP Protocol error {exc.code}: {exc.message}")
```

---

## 7. Running Asynchronous MCP Operations in Synchronous Workflows

MCP client operations are asynchronous. In synchronous applications (such as synchronous graph nodes or tools), execute them using `anyio.run`:

```python
import anyio
from mcp import Client


def fetch_available_tools(endpoint_url: str) -> list[str]:
    async def _async_fetch():
        async with Client(endpoint_url) as client:
            result = await client.list_tools()
            return [t.name for t in result.tools]

    return anyio.run(_async_fetch)


# Usage in synchronous code:
tool_names = fetch_available_tools("http://127.0.0.1:8000/mcp")
print("Available tools:", tool_names)
```

> **Warning on Active Loops**: If calling from an environment where an event loop is already running, use `await` directly or execute `anyio.run` in a separate worker thread (`concurrent.futures.ThreadPoolExecutor`).

---

## 8. Summary of Key Changes in `mcp>=2`

| Feature / Area | Earlier Versions (`mcp<2` / 2025-era) | Modern (`mcp>=2` / 2026-era) |
|---|---|---|
| **Primary Client API** | `ClientSession` with manual transport context | High-level `Client` wrapper |
| **HTTP Client Library** | `httpx` + `httpx-sse` | `httpx2` |
| **Streamable HTTP Client** | `streamablehttp_client` (yielded 3-tuple) | `streamable_http_client` (yields 2-tuple) |
| **HTTP Client Options** | `headers`/`timeout` on transport function | Configured on `httpx2.AsyncClient` |
| **Field Name Casing** | `camelCase` (`inputSchema`, `isError`) | `snake_case` (`input_schema`, `is_error`) |
| **Error Exception Class** | `McpError` (`exc.error.code`) | `MCPError` (`exc.code`, `exc.message`) |
| **Timeouts** | `datetime.timedelta` / HTTP status `408` | `float` seconds / `-32001` (`REQUEST_TIMEOUT`) |
| **Tool Results** | `result.content` | `result.content` + `result.structured_content` |
| **Content Types** | `Content` type alias | `ContentBlock` union |
| **Resource URIs** | Pydantic `AnyUrl` | Plain Python `str` |
| **Model Union Types** | Pydantic `RootModel` (`.root`) | Plain Python `Union` (`TypeAdapter`) |
