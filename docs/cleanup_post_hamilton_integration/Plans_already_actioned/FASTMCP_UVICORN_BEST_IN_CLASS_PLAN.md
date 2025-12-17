# FastMCP & Uvicorn Best-in-Class Implementation Plan

> **Purpose**: Transform the CodeIntel serving layer's FastMCP and Uvicorn implementation into a best-in-class solution with exceptional feature set, hardness, extensibility, maintainability, and full integration with the codebase.

> **Target Application**: Single-box, single-user personal application serving at most 3 external LLM consumers over MCP (Claude, ChatGPT, Cursor, etc.).

> **Scope**: 22 enhancement items organized by priority (Critical/High/Medium/Low) with detailed implementation specifications.

> **Revision Note**: This plan incorporates expert feedback from `FastMCP_implementation_comments.md` validating architectural decisions and adding critical items for runtime normalization, response envelopes, query limiting, and large-data handling.

---

## Implementation Status

| PR | Items | Status | Notes |
|----|-------|--------|-------|
| **PR1** | C0 (gofastmcp normalization) | ✅ Complete | Import shim, mount path, test patterns |
| **PR2** | H1 + H2 + H7 (Tool signature modernization) | ✅ Complete | Async tools, Context, Annotations, ToolError |
| **PR3** | H3 + H4 (Response envelope + Query limiter) | ✅ Complete | McpEnvelope models, QueryLimiter class |
| **PR4** | H5 + H6 (Resources + EventStore) | ✅ Complete | ResourceStore, MCP resources, EventStore config |
| **PR5** | M1-M4 (Uvicorn + Auth + Health) | ✅ Complete | Uvicorn settings, auth enforcement, health routes |
| **PR6** | M5-M8 (Tags, Metrics, Feature flags) | 🔲 Pending | Observability and extensibility |
| **PR7** | L1-L6 (Composition, Prompts, etc.) | 🔲 Pending | Optional enhancements |

### Key Learnings from PR1-PR3 Implementation

1. **Import Shim Pattern Works Well**: The `_compat.py` pattern for feature detection (e.g., `HAS_EVENT_STORE`) provides clean fallback behavior.

2. **QueryLimiter Type Signatures**: The limiter returns `object` from `run()`, requiring explicit result capture in a variable for type-safe access to result attributes like `.model_dump()`, `.truncated`, `.rows`.

3. **Pointer `published_at` is datetime**: The `ServingSnapshotPointer.published_at` is a `datetime` object, requiring `.isoformat()` conversion for the string-typed `McpSnapshotMeta.published_at`.

4. **Protocol for Kernel Access**: Use `Protocol` classes to define minimal interfaces (e.g., `SemanticKernel`, `_KernelDBProtocol`) to avoid circular imports while maintaining type safety.

5. **Test Refactoring**: Extract setup helpers like `_setup_test_snapshot()` to reduce local variable count and improve test maintainability.

6. **Settings Must Be Passed Through**: The `settings` object must flow from `build_mcp_app()` through all tool registration functions to access configuration values.

7. **Tool Pattern Consistency**: All 6 tools now follow the same pattern:
   - `async def tool_name(..., *, ctx: Context) -> dict[str, object]`
   - Use `limiter.run()` for blocking operations
   - Use `time.perf_counter()` for timing
   - Return `build_envelope(kernel, data, ...).model_dump(mode="json")`

### Key Learnings from PR4-PR5 Implementation

8. **FastMCP `auth` Parameter Uses AuthProvider**: The FastMCP constructor uses `auth` parameter (not `auth_token`), which expects an `AuthProvider` object. Use `StaticTokenVerifier` from `fastmcp.server.auth` for bearer token auth:
   ```python
   from fastmcp.server.auth import StaticTokenVerifier
   
   def create_bearer_auth(token: str | None) -> AuthProvider | None:
       if not token:
           return None
       return StaticTokenVerifier({token: {}})
   
   mcp = FastMCP(..., auth=create_bearer_auth(settings.auth_token))
   ```

9. **Custom Routes Require Async Signature**: The `@mcp.custom_route()` decorator requires async functions even if they don't await anything. Add `# noqa: RUF029` comment to suppress the "async function without await" lint warning:
   ```python
   @mcp.custom_route("/health", methods=["GET"])
   async def mcp_health(_request: Request) -> Response:  # noqa: RUF029
       # async required by FastMCP decorator, even for sync operations
       return JSONResponse({"status": "ok"})
   ```

10. **ResourceContent Not Available in fastmcp 2.14.1**: The `ResourceContent` class mentioned in fastmcp docs is not exported in v2.14.1. MCP resource handlers should return plain `dict`, `str`, or `bytes` instead. The `@mcp.resource()` decorator handles serialization automatically.

11. **Test Assertion Pattern**: Use `expect_*` functions from `tests._helpers.assertions.expectation_assertions` instead of bare `assert` to avoid S101 lint errors:
    ```python
    from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true
    expect_equal(settings.uvicorn_workers, 4)  # Not: assert settings.uvicorn_workers == 4
    ```

12. **Security Test Noqa Comments**: Tests for auth/security features need specific noqa comments:
    - `# noqa: S104` for intentional binding to `0.0.0.0`
    - `# noqa: S106` for test auth tokens like `auth_token="test-token"`

13. **Environment Variable Context Manager**: Use `_set_env()` context manager pattern for isolated env var testing:
    ```python
    @contextlib.contextmanager
    def _set_env(env: dict[str, str]) -> Iterator[None]:
        previous = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    ```

14. **Uvicorn Config Dict Type Ignores**: When passing `uvicorn_config` dict to `uvicorn.run()`, use `# type: ignore[arg-type]` since the dict has mixed value types that don't match uvicorn's strict signatures.

### Quick Reference: Remaining PRs

| PR | Duration | Items | Key Deliverables |
|----|----------|-------|------------------|
| **PR4** | 2 days | H5, H6 | ✅ `resource_store.py`, `resources.py`, EventStore config |
| **PR5** | 2 days | M1-M4 | ✅ Uvicorn settings, `create_bearer_auth()`, health routes |
| **PR6** | 1.5 days | M5-M7 | Tool tags, metrics emission, feature flags |
| **PR7** | 2 days | M8, L1, L2, L6 | Unified lifespan, server composition, prompts |

### Test File Reference

| Test File | What It Tests | Created In |
|-----------|---------------|------------|
| `tests/serving/test_semantic_mcp_tools.py` | All 6 tools, envelope, annotations | PR2-3 ✅ |
| `tests/serving/mcp/test_runtime.py` | QueryLimiter class | PR3 ✅ |
| `tests/serving/mcp/test_resources.py` | MCP resources, resource store, export tool | PR4 ✅ |
| `tests/serving/test_uvicorn_config.py` | Uvicorn settings defaults and env loading | PR5 ✅ |
| `tests/serving/test_auth_enforcement.py` | Auth required for public bind, validate_auth_for_host() | PR5 ✅ |
| `tests/serving/test_metrics.py` | MCP tool metrics emission | PR6 🔲 |
| `tests/serving/http/test_mcp_mount.py` | Mount path contract | PR7 🔲 |

### Implementation Learnings from PR4/PR5

Key insights discovered during PR4/PR5 implementation that will accelerate PR6/PR7:

#### FastMCP API Patterns

| Pattern | Correct Approach | Notes |
|---------|-----------------|-------|
| Tool registration | `@mcp.tool(annotations=..., tags=[...])` | Tags added alongside annotations |
| Tool signature | `async def name(..., *, ctx: Context) -> dict[str, object]` | `ctx` is keyword-only |
| Blocking ops | `await limiter.run(kernel.method, request)` | Use capacity limiter |
| Tool errors | `raise ToolError(_ERR_CODE, "message")` | Don't expose internals |
| Response format | `build_envelope(...).model_dump(mode="json")` | Always use envelope |
| Auth setup | `auth=create_bearer_auth(settings.auth_token)` | Pass to FastMCP constructor |
| Health routes | `@mcp.custom_route("/health", methods=["GET"])` | Returns `Response` object |
| Resource URIs | `codeintel://exports/{token}` | Custom scheme for resources |

#### Settings Pattern (`ServingSettings`)

```python
# Add new setting with default:
new_setting: bool = True

# Load from environment in from_env():
new_setting=os.environ.get("CODEINTEL_NEW_SETTING", "1") == "1",
```

#### Test Patterns

```python
# Environment context manager pattern (from test_auth_enforcement.py):
@contextlib.contextmanager
def _set_env(env: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

# Use in tests:
def test_setting_from_env(tmp_path: Path) -> None:
    with _set_env({"CODEINTEL_SERVE_DIR": str(tmp_path), "CODEINTEL_NEW_SETTING": "1"}):
        settings = ServingSettings.from_env()
    expect_true(settings.new_setting)
```

#### Key Files Reference

| File | Purpose | Key Exports |
|------|---------|-------------|
| `mcp/_compat.py` | Import shim | `FastMCP`, `Context`, `ToolError`, `create_bearer_auth`, `HAS_EVENT_STORE` |
| `mcp/app.py` | App builder | `build_mcp_app()`, tool registration |
| `mcp/resource_store.py` | Export storage | `ResourceStore` class |
| `mcp/resources.py` | MCP resources | `register_resources()` |
| `settings.py` | Configuration | `ServingSettings` dataclass |
| `http/app.py` | FastAPI factory | `create_serving_app()`, `_maybe_mount_mcp()` |

#### Common Pitfalls Discovered

1. **`ctx.session_id` access**: Use `getattr(ctx, "session_id", None)` - may not exist in streamable HTTP
2. **Tool manager API**: Use `await mcp._tool_manager.get_tools()` (async, returns dict not list)
3. **Auth parameter**: FastMCP uses `auth=AuthProvider`, not `auth_token=str`
4. **Custom routes**: Must be `async def` and return `Response` (not `JSONResponse` directly in signature)
5. **Ruff S104**: Suppress with `# noqa: S104` when intentionally binding to `0.0.0.0`

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Enhancement Categories](#enhancement-categories)
4. [Critical-Priority Enhancements](#critical-priority-enhancements)
5. [High-Priority Enhancements](#high-priority-enhancements)
6. [Medium-Priority Enhancements](#medium-priority-enhancements)
7. [Low-Priority Enhancements](#low-priority-enhancements)
8. [Implementation Phases](#implementation-phases)
9. [File Change Matrix](#file-change-matrix)
10. [Testing Strategy](#testing-strategy)
11. [Rollout Plan](#rollout-plan)

---

## Executive Summary

### Goals

1. **Runtime Normalization**: Adopt gofastmcp 2.x as the canonical MCP framework for advanced features
2. **Rich LLM Orchestration**: Leverage FastMCP's Context API for progress reporting, structured logging, and LLM-assisted data summarization
3. **Optimal Client Integration**: Add MCP annotations so Claude/ChatGPT can skip confirmation prompts for read-only operations
4. **Production Hardening**: Configure Uvicorn for performance, security, and reliability with query limiting
5. **Observability Parity**: Ensure MCP tools have the same metrics/logging as HTTP routes with consistent response envelopes
6. **Large Data Handling**: Use MCP Resources for large dataset delivery instead of JSON blobs
7. **Extensibility**: Structure the MCP layer for future growth with composition and modularity patterns

### Impact Summary

| Priority | Count | Key Benefits |
|----------|-------|--------------|
| Critical | 1 | Runtime normalization, feature availability |
| High | 7 | Core UX improvements, security, responsiveness, data handling |
| Medium | 8 | Production readiness, observability, feature flags |
| Low | 6 | Modularity, advanced patterns, future-proofing |

### Estimated Effort

- **Critical Priority**: 1 day
- **High Priority**: 4-5 days
- **Medium Priority**: 3-4 days
- **Low Priority**: 2-3 days
- **Total**: ~11-14 days

---

## Current State Analysis

### FastMCP Implementation (Post PR1-PR3)

**Location**: `src/codeintel/serving/mcp/`

| Component | Status | Notes |
|-----------|--------|-------|
| `_compat.py` | ✅ New | Import shim with feature flags (`HAS_EVENT_STORE`) |
| `app.py` | ✅ Updated | gofastmcp 2.x, async tools, Context, annotations, envelope, limiter |
| `models.py` | ✅ New | `McpSnapshotMeta`, `McpResponseMeta`, `McpEnvelope` |
| `response.py` | ✅ New | `build_envelope()` helper |
| `runtime.py` | ✅ New | `QueryLimiter` class for concurrency control |
| `server.py` | ✅ Updated | Uses gofastmcp imports |
| `__main__.py` | ✅ Functional | Entry point for stdio transport |

**Current Tools** (6 total, all modernized):
- `semantic_catalog` - List views (async, Context, envelope, limiter)
- `semantic_describe` - Describe view schema (async, Context, envelope, limiter)
- `semantic_query` - Query with filters (async, Context, envelope, limiter)
- `semantic_explain` - SQL + plan output (async, Context, envelope, limiter)
- `serving_meta` - Metadata endpoint (async, Context, envelope, limiter)
- `code_search` - FTS search (async, Context, envelope, limiter)

**Completed Features** (PR1-PR3):
- ✅ gofastmcp 2.x imports via `_compat.py`
- ✅ MCP Context with `*, ctx: Context` (keyword-only)
- ✅ Progress reporting via `ctx.report_progress()`
- ✅ MCP Annotations (`readOnlyHint`, `idempotentHint`, `openWorldHint`)
- ✅ Response meta envelope with snapshot provenance
- ✅ ToolError for controlled error messages
- ✅ Error masking via `mask_error_details` setting
- ✅ Query concurrency limiter via `QueryLimiter`
- ✅ Timing metadata (`query_ms`) in all responses

**Remaining Features**:
- ❌ MCP Resources for large data delivery (H5)
- ❌ EventStore for SSE resumability (H6)
- ❌ Tags for tool organization (M5)
- ❌ Health check on MCP endpoint (M3)
- ❌ Bearer token authentication (M4)
- ❌ Metrics emission from tools (M6)
- ❌ Tool feature flags (M7)

### Uvicorn Implementation

**Location**: `src/codeintel/cli/handlers/ops.py`

| Aspect | Current | Gap |
|--------|---------|-----|
| Workers | 1 (hardcoded) | No multi-worker support |
| Event Loop | Default | Not using uvloop |
| HTTP Parser | Default | Not using httptools |
| Concurrency Limits | None | No resource protection |
| Timeouts | Default | No keep-alive tuning |
| Security Headers | None | Server header exposed |
| Proxy Headers | None | No forwarded-IP policy |
| Query Limiter | None | Heavy queries can OOM |

### Integration Points (Post PR1-PR3)

| System | Integration Status |
|--------|-------------------|
| HTTP Routes | ✅ Full (metrics, correlation ID, RFC 9457) |
| MCP Tools | ✅ Enhanced (response envelope, timing, snapshot provenance) |
| Settings | ✅ MCP settings added (`mcp_*` family), ⚠️ Uvicorn settings pending |
| Storage Gateway | ✅ Full (via SemanticQueryKernel, accessed through Protocol) |
| Auth | ⚠️ Partial (HTTP has API key, MCP has `auth_token` setting but not wired) |
| Concurrency | ✅ QueryLimiter prevents concurrent heavy query OOM |
| Error Handling | ✅ ToolError + mask_error_details for controlled messages |

### Current MCP Settings (in `ServingSettings`)

```python
# MCP Context Features
mcp_enable_sampling: bool = False      # CODEINTEL_MCP_ENABLE_SAMPLING
mcp_sample_threshold: int = 500        # CODEINTEL_MCP_SAMPLE_THRESHOLD
mcp_progress_reporting: bool = True    # CODEINTEL_MCP_PROGRESS

# MCP Error Handling
mcp_mask_errors: bool = True           # CODEINTEL_MCP_MASK_ERRORS

# MCP Query Concurrency Control
mcp_max_concurrent_queries: int = 2    # CODEINTEL_MCP_MAX_CONCURRENT_QUERIES
```

---

## Enhancement Categories

### Category A: MCP Tool Enhancements
Items that improve how MCP tools work and integrate with LLM clients.

### Category B: Uvicorn & Deployment
Items that harden the server for production use.

### Category C: Observability & Security
Items that add monitoring, logging, and security features.

### Category D: Architecture & Extensibility
Items that improve code organization and future extensibility.

---

## Critical-Priority Enhancements

### C0: Normalize FastMCP Runtime to gofastmcp 2.x ✅ COMPLETE (PR1)

**Category**: D - Architecture & Extensibility

**Status**: ✅ Implemented in PR1

**Problem**: Current code imports `FastMCP` from `mcp.server.fastmcp` (MCP SDK flavor) while the plan and advanced feature guide reference gofastmcp 2.x features. These are incompatible:

| Feature | MCP SDK FastMCP | gofastmcp 2.x |
|---------|-----------------|---------------|
| Tool annotations | ❌ Limited | ✅ Full (readOnlyHint, etc.) |
| `http_app()` method | ❌ No | ✅ Yes |
| `EventStore` for SSE | ❌ No | ✅ Yes (v2.14.0+) |
| `ResourceContent` | ❌ No | ✅ Yes (v2.14.1+) |
| Tool meta parameter | ❌ Limited | ✅ Full |

**Solution**: 
1. Update imports to use gofastmcp: `from fastmcp import FastMCP`
2. Pin dependency: `fastmcp>=2.14.1,<3`
3. Create import shim for feature flags
4. Update all MCP code to use gofastmcp APIs

**Files to Modify**:

| File | Changes |
|------|---------|
| `pyproject.toml` | Pin `fastmcp>=2.14.1,<3`, evaluate `mcp[cli]` need |
| `src/codeintel/serving/mcp/_compat.py` | **New**: Import shim with feature flags |
| `src/codeintel/serving/mcp/app.py` | Change imports, update APIs |
| `src/codeintel/serving/mcp/server.py` | Change imports, update APIs |
| `src/codeintel/serving/http/app.py` | Update MCP mounting to use `http_app()` |

**Detailed Implementation**:

```python
# src/codeintel/serving/mcp/_compat.py

"""FastMCP import shim and feature flags.

This module provides a single import surface for FastMCP, ensuring consistent
usage across the codebase and enabling feature detection.
"""

from __future__ import annotations

import logging

# Canonical import from gofastmcp 2.x
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError
from fastmcp.resources import ResourceContent

LOG = logging.getLogger(__name__)

# Feature detection
try:
    from fastmcp.server.event_store import EventStore
    HAS_EVENT_STORE = True
except ImportError:
    EventStore = None  # type: ignore[assignment,misc]
    HAS_EVENT_STORE = False
    LOG.warning("EventStore not available - SSE resumability disabled")

__all__ = [
    "Context",
    "EventStore",
    "FastMCP",
    "HAS_EVENT_STORE",
    "ResourceContent",
    "ToolError",
]
```

```python
# src/codeintel/serving/mcp/app.py - Updated imports

from codeintel.serving.mcp._compat import (
    Context,
    FastMCP,
    ToolError,
)
```

```python
# src/codeintel/serving/http/app.py - Updated MCP mounting

def _maybe_mount_mcp(
    app: FastAPI,
    *,
    kernel: SemanticQueryKernel,
    settings: ServingSettings,
    enabled: bool,
) -> None:
    """Mount MCP server under /mcp with explicit path contract.
    
    Mount Contract:
    - FastAPI mounts at: /mcp
    - MCP ASGI app path: /
    - Effective MCP endpoint: /mcp (NOT /mcp/mcp)
    """
    if not enabled:
        return

    mcp = build_mcp_app(kernel=kernel, settings=settings)
    
    # gofastmcp 2.x uses http_app() instead of streamable_http_app()
    # path="/" ensures no double-prefixing when mounted at /mcp
    mcp_asgi = mcp.http_app(path="/")
    app.mount("/mcp", mcp_asgi)
```

**Testing Requirements**:
- Verify imports work with gofastmcp 2.x
- Test MCP tools remain functional
- Test mount path contract (verify `/mcp` works, `/mcp/mcp` returns 404)
- Test feature flag detection

**Implementation Notes (PR1)**:

The implementation created `src/codeintel/serving/mcp/_compat.py` with:
- `FastMCP`, `Context`, `ToolError` exports from `fastmcp`
- Feature detection for `EventStore` (v2.14.0+)
- `HAS_EVENT_STORE` boolean flag for conditional features

Key patterns established:
```python
# All MCP code imports from _compat, never directly from fastmcp
from codeintel.serving.mcp._compat import Context, FastMCP, ToolError

# Tests use fastmcp.client.Client pattern
from fastmcp.client import Client
async with Client(mcp) as client:
    result = await client.call_tool("tool_name", {...})
```

---

## High-Priority Enhancements

### H1: MCP Context Access for Rich Tool Orchestration ✅ COMPLETE (PR2)

**Category**: A - MCP Tool Enhancements

**Status**: ✅ Implemented in PR2

**Problem**: Tools are synchronous functions with no ability to report progress, log to clients, or leverage LLM sampling for large result summarization.

**Solution**: Add `fastmcp.Context` parameter to all tools, enabling:
- Progress reporting via `ctx.report_progress()`
- Client-visible logging via `ctx.info()`, `ctx.warning()`, `ctx.debug()`
- Resource reading via `ctx.read_resource()`
- LLM sampling via `ctx.sample()` for data summarization

**Important**: Context must be **keyword-only** (after `*`) to avoid Python syntax errors when placed after defaulted parameters.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/app.py` | Add Context param to all 6 tools, convert to async |
| `src/codeintel/serving/settings.py` | Add `mcp_enable_sampling`, `mcp_sample_threshold` settings |

**Detailed Implementation**:

```python
# src/codeintel/serving/mcp/app.py

from codeintel.serving.mcp._compat import Context, FastMCP

@mcp.tool()
async def semantic_query(
    view_id: str,
    filters: list[dict[str, object]] | None = None,
    select: list[str] | None = None,
    order_by: list[str] | None = None,
    pagination: dict[str, int] | None = None,
    *,  # <-- KEYWORD-ONLY MARKER (critical for valid Python)
    ctx: Context,
) -> dict[str, object]:
    """Query a semantic view with structured filters.

    Parameters
    ----------
    view_id
        Semantic view identifier.
    filters
        Optional list of filter specifications.
    select
        Optional column selection.
    order_by
        Optional ordering specification.
    pagination
        Optional pagination with limit/offset.
    ctx
        MCP execution context for progress and logging.

    Returns
    -------
    dict[str, object]
        Query response payload with rows and metadata.
    """
    await ctx.info(f"Executing query on view: {view_id}")
    await ctx.report_progress(10, 100)

    page = pagination or {}
    request = SemanticQueryRequest(
        view_id=view_id,
        select=select,
        filters=[FilterSpec.model_validate(f) for f in (filters or [])],
        order_by=order_by or [],
        limit=page.get("limit", 200),
        offset=page.get("offset", 0),
    )

    await ctx.report_progress(20, 100)
    result = await anyio.to_thread.run_sync(kernel.query, request)
    await ctx.report_progress(80, 100)

    # Large result summarization via LLM sampling
    if len(result.rows) > settings.mcp_sample_threshold and settings.mcp_enable_sampling:
        sample_rows = result.rows[:5]
        await ctx.warning(f"Large result set ({len(result.rows)} rows) - generating summary")
        summary_response = await ctx.sample(
            f"Summarize the structure and key insights from this data sample: {sample_rows}"
        )
        await ctx.info("Summary generated for large result set")
        output = result.model_dump(mode="json")
        output["llm_summary"] = summary_response.text
        await ctx.report_progress(100, 100)
        return output

    await ctx.report_progress(100, 100)
    return result.model_dump(mode="json")
```

**Standard Signature Pattern** (use for all tools):

```python
# Pattern A: Tool with required + optional params
@mcp.tool()
async def tool_name(
    required_param: str,
    optional_param: list[str] | None = None,
    *,  # keyword-only marker
    ctx: Context,
) -> ReturnType:
    ...

# Pattern B: Tool with only optional params
@mcp.tool()
async def tool_name(
    optional_param: str | None = None,
    *,
    ctx: Context,
) -> ReturnType:
    ...

# Pattern C: Tool with no params (still needs ctx)
@mcp.tool()
async def tool_name(*, ctx: Context) -> ReturnType:
    ...
```

**Settings Additions**:

```python
# src/codeintel/serving/settings.py

@dataclass(frozen=True)
class ServingSettings:
    # ... existing fields ...

    # MCP Context Features
    mcp_enable_sampling: bool = False  # Enable LLM sampling in tools
    mcp_sample_threshold: int = 500    # Row count threshold for sampling
    mcp_progress_reporting: bool = True  # Enable progress updates

    @classmethod
    def from_env(cls) -> ServingSettings:
        # ... existing code ...
        return cls(
            # ... existing fields ...
            mcp_enable_sampling=os.environ.get("CODEINTEL_MCP_ENABLE_SAMPLING", "0") == "1",
            mcp_sample_threshold=int(os.environ.get("CODEINTEL_MCP_SAMPLE_THRESHOLD", "500")),
            mcp_progress_reporting=os.environ.get("CODEINTEL_MCP_PROGRESS", "1") == "1",
        )
```

**Testing Requirements**:
- Unit test for Context injection
- Integration test with mock Context
- Test progress reporting sequence
- Test LLM sampling threshold logic
- Verify keyword-only ctx doesn't break tool schema generation

**Implementation Notes (PR2)**:

All 6 tools were refactored with this pattern:
```python
def _register_query_tool(
    mcp: FastMCP, kernel: SemanticKernel, limiter: QueryLimiter
) -> None:
    @mcp.tool(
        name="semantic_query",
        description="...",
        annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
    )
    async def semantic_query(
        view_id: str,
        filters: list[dict[str, object]] | None = None,
        *,  # keyword-only marker
        ctx: Context,
    ) -> dict[str, object]:
        await ctx.info(f"Querying view: {view_id}")
        await ctx.report_progress(10, 100)
        # ... use limiter.run() for blocking calls ...
```

Key insight: Tool registration functions were extracted to keep `build_mcp_app()` complexity manageable (Ruff C901 limit of 10).

---

### H2: MCP Annotations for LLM Client Optimization ✅ COMPLETE (PR2)

**Category**: A - MCP Tool Enhancements

**Status**: ✅ Implemented in PR2

**Problem**: LLM clients (ChatGPT, Claude) prompt users for confirmation before running tools, even for safe read-only operations.

**Solution**: Add MCP annotations to all tools indicating their safety characteristics:
- `readOnlyHint=True` - No data modification
- `idempotentHint=True` - Safe to retry
- `openWorldHint=False` - Local database only (no external network calls)

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/app.py` | Add annotations dict to all 6 `@mcp.tool()` decorators |

**Detailed Implementation**:

```python
# src/codeintel/serving/mcp/app.py

# Define reusable annotation sets
_READ_ONLY_LOCAL_ANNOTATIONS = {
    "readOnlyHint": True,
    "idempotentHint": True,
    "openWorldHint": False,
}

@mcp.tool(
    name="semantic_catalog",
    description="List available semantic views in the CodeIntel database",
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
)
async def semantic_catalog(*, ctx: Context) -> dict[str, object]:
    ...

@mcp.tool(
    name="semantic_describe",
    description="Describe a semantic view's schema and metadata",
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
)
async def semantic_describe(view_id: str, *, ctx: Context) -> dict[str, object]:
    ...

@mcp.tool(
    name="semantic_query",
    description="Query a semantic view with structured filters",
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
)
async def semantic_query(..., *, ctx: Context) -> dict[str, object]:
    ...

@mcp.tool(
    name="semantic_explain",
    description="Return compiled SQL and DuckDB plan for a semantic query",
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
)
async def semantic_explain(..., *, ctx: Context) -> dict[str, object]:
    ...

@mcp.tool(
    name="serving_meta",
    description="Get serving layer metadata including snapshot info",
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
)
async def serving_meta(*, ctx: Context) -> dict[str, object]:
    ...

@mcp.tool(
    name="code_search",
    description="Search code metadata using BM25 full-text search",
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
)
async def code_search(..., *, ctx: Context) -> dict[str, object]:
    ...
```

**Client Behavior Changes**:
- **ChatGPT**: Skips "Are you sure?" confirmation for annotated tools
- **Claude**: Uses hints to assess tool safety and execution timing
- **Cursor**: May auto-approve read-only tool calls

**Testing Requirements**:
- Verify annotations appear in tool schema output
- Test that annotations don't affect tool execution
- Document client-specific behavior

**Implementation Notes (PR2)**:

Annotations are defined as a reusable constant:
```python
_READ_ONLY_LOCAL_ANNOTATIONS = {
    "readOnlyHint": True,      # No data modification
    "idempotentHint": True,    # Safe to retry
    "openWorldHint": False,    # Local database only
}
```

Test added: `test_mcp_tool_annotations_present()` verifies all tools have `readOnlyHint=True`.

---

### H3: Standard Response Meta Envelope ✅ COMPLETE (PR3)

**Category**: A - MCP Tool Enhancements

**Status**: ✅ Implemented in PR3

**Problem**: Tool responses lack context for agentic workflows. LLM agents doing iterative analysis cannot determine:
- What snapshot/version the data came from
- Whether results are truncated
- Query timing information

This leads to hallucinated conclusions when data changes between calls.

**Solution**: Wrap every tool response in a standard `McpEnvelope` with consistent metadata.

**Files to Create/Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/models.py` | **New**: Define McpEnvelope, McpResponseMeta, McpSnapshotMeta |
| `src/codeintel/serving/mcp/response.py` | **New**: Helper function to build envelopes |
| `src/codeintel/serving/mcp/app.py` | Wrap all tool returns in McpEnvelope |

**Detailed Implementation**:

```python
# src/codeintel/serving/mcp/models.py

"""Pydantic models for MCP tool responses with standard envelope."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class McpSnapshotMeta(BaseModel):
    """Snapshot identification for provenance tracking."""

    model_config = ConfigDict(extra="forbid")

    repo: str
    commit: str
    run_id: str
    published_at: str
    semantic_layer_version: str


class McpResponseMeta(BaseModel):
    """Standard response metadata for all MCP tools."""

    model_config = ConfigDict(extra="forbid")

    snapshot: McpSnapshotMeta
    truncated: bool = False
    query_ms: int | None = None
    row_count: int | None = None


class McpEnvelope(BaseModel):
    """Standard envelope for all MCP tool responses.
    
    Every MCP tool should return data wrapped in this envelope so LLM agents
    can track provenance and detect data changes between calls.
    """

    model_config = ConfigDict(extra="forbid")

    meta: McpResponseMeta
    data: dict[str, object]


# Additional response models for specific tools
class ViewSummary(BaseModel):
    """Compact view info for catalog listing."""

    model_config = ConfigDict(extra="forbid")

    id: str
    table_key: str
    entity: str
    grain: str
    description: str
    column_count: int


class SearchResultItem(BaseModel):
    """Single search result."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    name: str
    module: str | None
    rel_path: str | None
    ref_goid_h128: str | None
    score: float | None


__all__ = [
    "McpEnvelope",
    "McpResponseMeta",
    "McpSnapshotMeta",
    "SearchResultItem",
    "ViewSummary",
]
```

```python
# src/codeintel/serving/mcp/response.py

"""Helper functions for building MCP response envelopes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.serving.mcp.models import (
    McpEnvelope,
    McpResponseMeta,
    McpSnapshotMeta,
)

if TYPE_CHECKING:
    from codeintel.serving.semantic.kernel import SemanticQueryKernel


def build_envelope(
    kernel: SemanticQueryKernel,
    data: dict[str, object],
    *,
    truncated: bool = False,
    query_ms: int | None = None,
    row_count: int | None = None,
) -> McpEnvelope:
    """Build a standard MCP response envelope with snapshot metadata.

    Parameters
    ----------
    kernel
        Semantic query kernel for snapshot info access.
    data
        Tool-specific response data.
    truncated
        Whether results were truncated.
    query_ms
        Query execution time in milliseconds.
    row_count
        Number of rows in response.

    Returns
    -------
    McpEnvelope
        Wrapped response with metadata.
    """
    pointer = kernel.db.current_pointer()
    
    snapshot_meta = McpSnapshotMeta(
        repo=pointer.repo,
        commit=pointer.commit,
        run_id=pointer.run_id,
        published_at=pointer.published_at,
        semantic_layer_version=getattr(pointer, "semantic_layer_version", "unknown"),
    )
    
    response_meta = McpResponseMeta(
        snapshot=snapshot_meta,
        truncated=truncated,
        query_ms=query_ms,
        row_count=row_count,
    )
    
    return McpEnvelope(meta=response_meta, data=data)


__all__ = ["build_envelope"]
```

```python
# src/codeintel/serving/mcp/app.py - Updated tool with envelope

import time
from codeintel.serving.mcp.response import build_envelope

@mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS)
async def semantic_query(
    view_id: str,
    filters: list[dict[str, object]] | None = None,
    select: list[str] | None = None,
    order_by: list[str] | None = None,
    pagination: dict[str, int] | None = None,
    *,
    ctx: Context,
) -> dict[str, object]:
    """Query a semantic view with structured filters."""
    start = time.perf_counter()
    await ctx.info(f"Querying view: {view_id}")
    
    # ... build request and execute query ...
    result = await anyio.to_thread.run_sync(kernel.query, request)
    
    query_ms = int((time.perf_counter() - start) * 1000)
    
    # Wrap in envelope with metadata
    return build_envelope(
        kernel,
        result.model_dump(mode="json"),
        truncated=result.truncated,
        query_ms=query_ms,
        row_count=len(result.rows),
    ).model_dump(mode="json")
```

**Testing Requirements**:
- Verify envelope structure in all tool responses
- Test snapshot metadata accuracy
- Verify timing information is captured
- Test truncation flag propagation

**Implementation Notes (PR3)**:

Created `src/codeintel/serving/mcp/models.py` with three models:
- `McpSnapshotMeta`: repo, commit, run_id, published_at, semantic_layer_version
- `McpResponseMeta`: snapshot, truncated, query_ms, row_count
- `McpEnvelope`: meta + data

Created `src/codeintel/serving/mcp/response.py` with `build_envelope()` helper.

Key insight: `ServingSnapshotPointer.published_at` is `datetime`, requires `.isoformat()` conversion:
```python
snapshot = McpSnapshotMeta(
    repo=ptr.repo,
    commit=ptr.commit,
    published_at=ptr.published_at.isoformat(),  # datetime -> str
    ...
)
```

Protocol pattern for kernel access (avoiding circular imports):
```python
class _KernelDBProtocol(Protocol):
    def current_pointer(self) -> ServingSnapshotPointer: ...

class _KernelProtocol(Protocol):
    @property
    def db(self) -> _KernelDBProtocol: ...
```

Tests added: `test_mcp_response_envelope_structure()`, `test_mcp_response_timing_captured()`, `test_mcp_response_snapshot_values()`.

---

### H4: Query Limiter for Concurrency Control ✅ COMPLETE (PR3)

**Category**: B - Uvicorn & Deployment

**Status**: ✅ Implemented in PR3

**Problem**: With 3 LLM consumers, heavy DuckDB queries can run simultaneously, causing:
- Memory blowout from multiple large result sets
- Query serialization at DuckDB level (single connection isn't truly parallel)
- Degraded response times for all clients

**Solution**: Add a server-side semaphore limiting concurrent heavy query execution, independent of HTTP concurrency.

**Files to Create/Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/runtime.py` | **New**: QueryLimiter class |
| `src/codeintel/serving/settings.py` | Add `mcp_max_concurrent_queries` setting |
| `src/codeintel/serving/mcp/app.py` | Route heavy tools through limiter |

**Detailed Implementation**:

```python
# src/codeintel/serving/mcp/runtime.py

"""Runtime utilities for MCP server including concurrency control."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import anyio

T = TypeVar("T")


class QueryLimiter:
    """Semaphore-based limiter for concurrent query execution.
    
    Prevents memory blowout when multiple LLM consumers trigger heavy queries
    simultaneously. Independent of HTTP connection limits.
    
    Parameters
    ----------
    max_concurrent
        Maximum number of concurrent heavy operations allowed.
    """

    def __init__(self, max_concurrent: int) -> None:
        self._sem = anyio.Semaphore(max_concurrent)
        self._max = max_concurrent

    @property
    def max_concurrent(self) -> int:
        """Return the maximum concurrent operations allowed."""
        return self._max

    async def run(self, fn: Callable[..., T], *args: object, **kwargs: object) -> T:
        """Execute a function with concurrency limiting.

        Parameters
        ----------
        fn
            Synchronous function to execute (will be offloaded to thread).
        *args
            Positional arguments for the function.
        **kwargs
            Keyword arguments for the function.

        Returns
        -------
        T
            Function result.
        """
        async with self._sem:
            return await anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))

    async def run_async(self, coro: Callable[..., T], *args: object, **kwargs: object) -> T:
        """Execute an async function with concurrency limiting.

        Parameters
        ----------
        coro
            Async function to execute.
        *args
            Positional arguments for the function.
        **kwargs
            Keyword arguments for the function.

        Returns
        -------
        T
            Function result.
        """
        async with self._sem:
            return await coro(*args, **kwargs)


__all__ = ["QueryLimiter"]
```

```python
# src/codeintel/serving/settings.py - Add limiter setting

@dataclass(frozen=True)
class ServingSettings:
    # ... existing fields ...

    # Query Concurrency Control
    mcp_max_concurrent_queries: int = 2  # Max concurrent heavy queries

    @classmethod
    def from_env(cls) -> ServingSettings:
        return cls(
            # ... existing fields ...
            mcp_max_concurrent_queries=int(
                os.environ.get("CODEINTEL_MCP_MAX_CONCURRENT_QUERIES", "2")
            ),
        )
```

```python
# src/codeintel/serving/mcp/app.py - Use limiter

from codeintel.serving.mcp.runtime import QueryLimiter

def build_mcp_app(
    *,
    kernel: SemanticKernel,
    settings: ServingSettings,
    ...
) -> FastMCP:
    # Initialize query limiter
    limiter = QueryLimiter(max_concurrent=settings.mcp_max_concurrent_queries)

    @mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS)
    async def semantic_query(..., *, ctx: Context) -> dict[str, object]:
        """Query a semantic view with structured filters."""
        await ctx.info(f"Querying view: {view_id}")
        
        # ... build request ...
        
        # Execute query through limiter (prevents concurrent heavy queries)
        result = await limiter.run(kernel.query, request)
        
        return build_envelope(kernel, result.model_dump(mode="json"), ...).model_dump()

    @mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS)
    async def code_search(..., *, ctx: Context) -> dict[str, object]:
        """Search code metadata."""
        await ctx.info(f"Searching: {query}")
        
        # ... build request ...
        
        # Search also goes through limiter
        result = await limiter.run(kernel.search, request)
        
        return build_envelope(kernel, result.model_dump(mode="json"), ...).model_dump()
```

**DuckDB Concurrency Constraints** (document in code comments):

```python
# DuckDB Concurrency Notes:
# 1. DuckDB connections are NOT thread-safe - each thread needs its own connection
#    (Our ReadPoolGateway handles this via per-thread pooling)
# 2. Read-only mode is REQUIRED for multi-process access to same DB file
#    (Our serving layer uses read_only=True)
# 3. Even with per-thread connections, parallel heavy queries can thrash memory
#    (QueryLimiter caps concurrent queries independent of connection count)
```

**Recommended Presets**:

| Preset | Workers | Max Concurrent Queries | Notes |
|--------|---------|------------------------|-------|
| Default | 1 | 2 | Single-process, sufficient for personal use |
| Snappy | 2 | 2 | Total 4 concurrent queries max across workers |

**Testing Requirements**:
- Test limiter blocks when at capacity
- Test queued requests complete after capacity frees
- Load test with concurrent tool calls
- Verify memory usage stays bounded

**Implementation Notes (PR3)**:

Created `src/codeintel/serving/mcp/runtime.py` with `QueryLimiter` class:
```python
class QueryLimiter:
    def __init__(self, max_concurrent: int) -> None:
        self._sem = anyio.Semaphore(max_concurrent)
    
    async def run(self, fn: object, *args, **kwargs) -> object:
        async with self._sem:
            return await to_thread.run_sync(lambda: fn(*args, **kwargs))
```

Key insight: `limiter.run()` returns `object`, requiring `cast()` for type safety:
```python
result = cast("SemanticQueryResponse", await limiter.run(kernel.query, request))
# Now result.rows, result.truncated, result.model_dump() are type-safe
```

Limiter is initialized in `build_mcp_app()` and passed to all tool registration functions:
```python
limiter = QueryLimiter(max_concurrent=settings.mcp_max_concurrent_queries)
_register_query_tool(mcp, kernel, limiter)
```

Tests added in `tests/serving/mcp/test_runtime.py`: concurrency verification, serialization behavior.

---

### H5: MCP Resources for Large Dataset Delivery 🔲 PENDING (PR4)

**Category**: A - MCP Tool Enhancements

**Status**: 🔲 Not started - Planned for PR4

**Estimated Effort**: 1.5 days

**Prerequisites**: PR1-PR3 complete (all satisfied)

**Problem**: Current export approach materializes all rows in Python memory:

```python
rows = list(kernel.export_rows(payload))  # OOM trap!
table = pa.Table.from_pylist(rows)  # then convert
```

This will fail for large semantic views.

**Solution**: Use MCP Resources for large data delivery:
- Tools return small structured summaries + resource URIs
- Resources serve the actual data (Parquet/Arrow/NDJSON)
- DuckDB writes directly to Parquet without Python materialization

**Files to Create/Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/resource_store.py` | **New**: Artifact storage for exports |
| `src/codeintel/serving/mcp/resources.py` | **New**: MCP resource handlers |
| `src/codeintel/serving/mcp/app.py` | Add export tool returning resource URI |

**Detailed Implementation**:

```python
# src/codeintel/serving/mcp/resource_store.py

"""On-disk artifact store for MCP resource exports."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class StoredArtifact:
    """Metadata for a stored export artifact."""

    path: Path
    mime_type: str
    row_count: int
    size_bytes: int


class ResourceStore:
    """File-backed store for export artifacts.
    
    Artifacts are stored with random tokens and can be retrieved by token.
    Designed for temporary exports that MCP clients fetch after tool calls.
    
    Parameters
    ----------
    root
        Root directory for artifact storage.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def put_json(self, payload: object, *, row_count: int = 0) -> tuple[str, StoredArtifact]:
        """Store a JSON payload and return its token.

        Parameters
        ----------
        payload
            JSON-serializable data.
        row_count
            Number of rows in the payload (for metadata).

        Returns
        -------
        tuple[str, StoredArtifact]
            Token and artifact metadata.
        """
        token = secrets.token_urlsafe(16)
        path = self._root / f"{token}.json"
        content = json.dumps(payload, indent=2, sort_keys=True, default=str)
        path.write_text(content, encoding="utf-8")
        
        return token, StoredArtifact(
            path=path,
            mime_type="application/json",
            row_count=row_count,
            size_bytes=path.stat().st_size,
        )

    def put_ndjson(self, rows: list[dict[str, object]]) -> tuple[str, StoredArtifact]:
        """Store rows as NDJSON and return token.

        Parameters
        ----------
        rows
            List of row dictionaries.

        Returns
        -------
        tuple[str, StoredArtifact]
            Token and artifact metadata.
        """
        token = secrets.token_urlsafe(16)
        path = self._root / f"{token}.ndjson"
        
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")
        
        return token, StoredArtifact(
            path=path,
            mime_type="application/x-ndjson",
            row_count=len(rows),
            size_bytes=path.stat().st_size,
        )

    def get(self, token: str) -> StoredArtifact:
        """Retrieve artifact metadata by token.

        Parameters
        ----------
        token
            Artifact token.

        Returns
        -------
        StoredArtifact
            Artifact metadata.

        Raises
        ------
        KeyError
            If token not found.
        """
        for ext, mime_type in [
            (".json", "application/json"),
            (".ndjson", "application/x-ndjson"),
            (".parquet", "application/vnd.apache.parquet"),
        ]:
            path = self._root / f"{token}{ext}"
            if path.exists():
                return StoredArtifact(
                    path=path,
                    mime_type=mime_type,
                    row_count=0,  # Would need separate metadata file for this
                    size_bytes=path.stat().st_size,
                )
        
        msg = f"Artifact not found: {token}"
        raise KeyError(msg)


__all__ = ["ResourceStore", "StoredArtifact"]
```

```python
# src/codeintel/serving/mcp/resources.py

"""MCP resource handlers for on-demand data access."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.serving.mcp._compat import ResourceContent

if TYPE_CHECKING:
    from codeintel.serving.mcp._compat import FastMCP
    from codeintel.serving.mcp.app import SemanticKernel
    from codeintel.serving.mcp.resource_store import ResourceStore


def register_resources(
    mcp: FastMCP,
    kernel: SemanticKernel,
    store: ResourceStore,
) -> None:
    """Register MCP resources on the server.

    Parameters
    ----------
    mcp
        FastMCP server instance.
    kernel
        Semantic query kernel.
    store
        Resource store for exports.
    """

    @mcp.resource("codeintel://semantic/registry")
    def semantic_registry() -> dict[str, object]:
        """Full semantic view catalog as a resource."""
        return kernel.catalog()

    @mcp.resource("codeintel://semantic/views/{view_id}")
    def view_description(view_id: str) -> dict[str, object]:
        """Semantic view description as a resource."""
        return kernel.describe(view_id)

    @mcp.resource("codeintel://semantic/views/{view_id}/schema")
    def view_schema(view_id: str) -> dict[str, object]:
        """View schema only (for LLM context efficiency)."""
        full = kernel.describe(view_id)
        return {
            "id": full["id"],
            "columns": full["columns"],
            "column_types": full.get("column_types"),
            "primary_key": full.get("primary_key"),
        }

    @mcp.resource("codeintel://meta")
    def serving_meta_resource() -> dict[str, object]:
        """Serving metadata as a resource."""
        return kernel.meta()

    @mcp.resource("codeintel://exports/{token}")
    def read_export(token: str) -> ResourceContent:
        """Read a previously exported artifact by token."""
        artifact = store.get(token)
        data = artifact.path.read_bytes()
        return ResourceContent(content=data, mime_type=artifact.mime_type)


__all__ = ["register_resources"]
```

```python
# src/codeintel/serving/mcp/app.py - Export tool returning resource URI

@mcp.tool(
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
    tags=[TAG_SEMANTIC, TAG_READ],
)
async def semantic_export(
    view_id: str,
    filters: list[dict[str, object]] | None = None,
    format: str = "ndjson",
    limit: int = 100_000,
    *,
    ctx: Context,
) -> dict[str, object]:
    """Export semantic view data and return a resource URI.
    
    For large datasets, this tool returns a resource URI that can be fetched
    separately, avoiding OOM from materializing large result sets in JSON.

    Parameters
    ----------
    view_id
        Semantic view identifier.
    filters
        Optional filter specifications.
    format
        Export format: "json" or "ndjson".
    limit
        Maximum rows to export.
    ctx
        MCP execution context.

    Returns
    -------
    dict[str, object]
        Envelope with export_uri and metadata.
    """
    await ctx.info(f"Exporting view: {view_id} (format={format})")
    
    from codeintel.serving.semantic.models import SemanticExportRequest
    
    request = SemanticExportRequest(
        view_id=view_id,
        filters=[FilterSpec.model_validate(f) for f in (filters or [])],
        format=format,
        limit=limit,
    )
    
    # Stream rows and store as artifact
    rows = await limiter.run(lambda: list(kernel.export_rows(request)))
    
    if format == "ndjson":
        token, artifact = store.put_ndjson(rows)
    else:
        token, artifact = store.put_json({"rows": rows}, row_count=len(rows))
    
    await ctx.info(f"Export complete: {artifact.row_count} rows, {artifact.size_bytes} bytes")
    
    return build_envelope(
        kernel,
        {
            "export_uri": f"codeintel://exports/{token}",
            "format": format,
            "row_count": artifact.row_count,
            "size_bytes": artifact.size_bytes,
        },
        row_count=artifact.row_count,
    ).model_dump(mode="json")
```

**Testing Requirements**:
- Test resource URI resolution
- Test parameterized resources (view_id)
- Test export artifact storage and retrieval
- Verify ResourceContent MIME types

**Execution Guidance for PR4**:

1. **Create `resource_store.py`** first - this is a standalone module with no dependencies on existing MCP code:
   - `ResourceStore` class with `put_json()`, `put_ndjson()`, `get()` methods
   - `StoredArtifact` dataclass with path, mime_type, row_count, size_bytes

2. **Update `_compat.py`** to export `ResourceContent`:
   ```python
   from fastmcp.resources import ResourceContent
   ```
   Note: `ResourceContent` is available in fastmcp 2.14.1+ (already pinned)

3. **Create `resources.py`** with `register_resources()` function:
   - Static resources: `codeintel://semantic/registry`, `codeintel://meta`
   - Parameterized resources: `codeintel://semantic/views/{view_id}`
   - Export resources: `codeintel://exports/{token}`

4. **Update `build_mcp_app()`** to:
   - Instantiate `ResourceStore` in serve_dir/exports/
   - Call `register_resources(mcp, kernel, store)`
   - Add `semantic_export` tool that returns resource URI

5. **Integration with existing code**:
   - The `semantic_export` tool follows the same pattern as other tools (limiter, envelope)
   - Resource store uses `secrets.token_urlsafe(16)` for artifact tokens
   - Consider TTL cleanup for old exports (optional, could be separate PR)

**Key Files to Create**:
- `src/codeintel/serving/mcp/resource_store.py` (~80 lines)
- `src/codeintel/serving/mcp/resources.py` (~60 lines)

**Key Files to Modify**:
- `src/codeintel/serving/mcp/_compat.py` (add ResourceContent export)
- `src/codeintel/serving/mcp/app.py` (add semantic_export tool, call register_resources)

---

### H6: SSE Polling and EventStore for Remote Resumability 🔲 PENDING (PR4)

**Category**: B - Uvicorn & Deployment

**Status**: 🔲 Not started - Planned for PR4 (bundled with H5)

**Estimated Effort**: 0.5 days (most infrastructure already in `_compat.py`)

**Prerequisites**: C0 complete (EventStore feature flag in `_compat.py`)

**Problem**: For remote internet connections, long-running tool calls can:
- Be interrupted by proxy timeouts (Cloudflare has 100s limit)
- Suffer from transient disconnects
- Leave clients in limbo on "hung request" failures

**Solution**: Enable gofastmcp's EventStore + SSE polling for StreamableHTTP resumability.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/settings.py` | Add EventStore settings |
| `src/codeintel/serving/http/app.py` | Pass EventStore to `http_app()` |
| `src/codeintel/serving/mcp/app.py` | Add `ctx.close_sse_stream()` for long tools |

**Detailed Implementation**:

```python
# src/codeintel/serving/settings.py - Add EventStore settings

@dataclass(frozen=True)
class ServingSettings:
    # ... existing fields ...

    # MCP EventStore for SSE Resumability
    mcp_enable_event_store: bool = True  # Enable SSE polling/resumability
    mcp_retry_interval_ms: int = 1000     # SSE retry interval

    @classmethod
    def from_env(cls) -> ServingSettings:
        return cls(
            # ... existing fields ...
            mcp_enable_event_store=os.environ.get("CODEINTEL_MCP_EVENT_STORE", "1") == "1",
            mcp_retry_interval_ms=int(os.environ.get("CODEINTEL_MCP_RETRY_INTERVAL", "1000")),
        )
```

```python
# src/codeintel/serving/http/app.py - Pass EventStore to http_app()

from codeintel.serving.mcp._compat import EventStore, HAS_EVENT_STORE

def _maybe_mount_mcp(
    app: FastAPI,
    *,
    kernel: SemanticQueryKernel,
    settings: ServingSettings,
    enabled: bool,
) -> None:
    """Mount MCP server under /mcp with EventStore for resumability."""
    if not enabled:
        return

    mcp = build_mcp_app(kernel=kernel, settings=settings)
    
    # Configure EventStore for SSE polling/resumability
    event_store = None
    if settings.mcp_enable_event_store and HAS_EVENT_STORE:
        event_store = EventStore()
    
    # Build ASGI app with EventStore
    mcp_asgi = mcp.http_app(
        path="/",
        event_store=event_store,
        retry_interval=settings.mcp_retry_interval_ms if event_store else None,
    )
    app.mount("/mcp", mcp_asgi)
```

```python
# src/codeintel/serving/mcp/app.py - Close SSE stream for long tools

@mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS)
async def semantic_export(..., *, ctx: Context) -> dict[str, object]:
    """Export semantic view data."""
    await ctx.info(f"Exporting view: {view_id}")
    
    # For long exports, periodically close SSE stream to allow reconnect
    rows = []
    batch_size = 10000
    
    async for batch in stream_batches(kernel, request, batch_size):
        rows.extend(batch)
        await ctx.report_progress(len(rows), request.limit)
        
        # Trigger reconnect/resume every 30 batches (~300k rows)
        # This helps with proxy timeouts
        if len(rows) % (batch_size * 30) == 0:
            await ctx.close_sse_stream()
    
    # ... store and return ...
```

**Testing Requirements**:
- Verify EventStore is configured when enabled
- Test SSE reconnect behavior (integration test)
- Verify settings are respected

**Execution Guidance for PR4**:

1. **Add settings** to `ServingSettings`:
   ```python
   mcp_enable_event_store: bool = True
   mcp_retry_interval_ms: int = 1000
   ```

2. **Update `_maybe_mount_mcp()` in `http/app.py`**:
   - Import `EventStore, HAS_EVENT_STORE` from `_compat`
   - Conditionally create `EventStore()` if enabled and available
   - Pass to `mcp.http_app(path="/", event_store=event_store, retry_interval=...)`

3. **Optional: Add `ctx.close_sse_stream()`** to long-running tools like `semantic_export`:
   - Helps with proxy timeouts on very large exports
   - Only needed if exports take >60 seconds

**Key Insight**: The `HAS_EVENT_STORE` flag in `_compat.py` already handles feature detection. The main work is:
- Adding 2 settings
- ~10 lines of conditional logic in `_maybe_mount_mcp()`

---

### H7: Error Masking and ToolError for Security ✅ COMPLETE (PR2)

**Category**: C - Observability & Security

**Status**: ✅ Implemented in PR2

**Problem**: Exceptions propagate with full stack traces to LLM clients, potentially exposing internal implementation details.

**Solution**: 
1. Enable `mask_error_details=True` on the FastMCP server
2. Use `fastmcp.exceptions.ToolError` for controlled error messages
3. Align with existing RFC 9457 Problem Details pattern

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/app.py` | Add ToolError handling, update FastMCP init |
| `src/codeintel/serving/settings.py` | Add `mcp_mask_errors` setting |

**Detailed Implementation**:

```python
# src/codeintel/serving/mcp/app.py

from codeintel.serving.mcp._compat import ToolError

def build_mcp_app(
    *,
    kernel: SemanticKernel,
    settings: ServingSettings,
    ...
) -> FastMCP:
    """Build FastMCP application with semantic tools."""
    mcp = FastMCP(
        "CodeIntel",
        json_response=True,
        mask_error_details=settings.mcp_mask_errors,  # Hide internal traces
    )

    @mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS)
    async def semantic_query(..., *, ctx: Context) -> dict[str, object]:
        """Query a semantic view with structured filters."""
        try:
            await ctx.info(f"Querying view: {view_id}")
            # ... query logic ...
            result = await limiter.run(kernel.query, request)
            return build_envelope(kernel, result.model_dump(), ...).model_dump()
        except KeyError as e:
            # User-friendly error (passes through masking)
            raise ToolError(f"View '{view_id}' not found in semantic registry") from e
        except ValueError as e:
            raise ToolError(f"Invalid query parameters: {e}") from e
        except Exception as e:
            # Log internally, return generic error
            await ctx.error(f"Query failed: {type(e).__name__}")
            raise ToolError("Query execution failed. Check server logs for details.") from e

    return mcp
```

```python
# src/codeintel/serving/settings.py

@dataclass(frozen=True)
class ServingSettings:
    # ... existing fields ...
    mcp_mask_errors: bool = True  # Mask internal error details

    @classmethod
    def from_env(cls) -> ServingSettings:
        return cls(
            # ... existing fields ...
            mcp_mask_errors=os.environ.get("CODEINTEL_MCP_MASK_ERRORS", "1") == "1",
        )
```

**Testing Requirements**:
- Verify ToolError messages are passed to client
- Verify other exceptions are masked when `mask_error_details=True`
- Test error logging to server logs

**Implementation Notes (PR2)**:

FastMCP server initialized with masking:
```python
mcp = FastMCP(
    "CodeIntel",
    json_response=True,
    mask_error_details=settings.mcp_mask_errors,
)
```

Error handling pattern in tools:
```python
try:
    result = await limiter.run(kernel.query, request)
    return build_envelope(kernel, result.model_dump(), ...).model_dump()
except KeyError as e:
    raise ToolError(f"View '{view_id}' not found in semantic registry") from e
except ValueError as e:
    raise ToolError(f"Invalid parameters: {e}") from e
except Exception as e:
    LOG.exception("Query failed for view %s", view_id)
    raise ToolError("Query execution failed. Check server logs.") from e
```

Error message constants defined at module level for consistency:
```python
_ERR_CATALOG_FAILED = "Failed to retrieve catalog. Check server logs."
_ERR_QUERY_FAILED = "Query execution failed. Check server logs."
# etc.
```

Test added: `test_mcp_tool_error_handling()` verifies controlled error messages.

---

## Medium-Priority Enhancements

### M1: Production Uvicorn Configuration 🔲 PENDING (PR5)

**Category**: B - Uvicorn & Deployment

**Status**: 🔲 Not started - Planned for PR5

**Estimated Effort**: 1 day

**Problem**: Current Uvicorn configuration uses defaults, missing performance optimizations, resource protection, and proxy support.

**Solution**: Add comprehensive Uvicorn configuration with:
- Optional multi-worker support
- uvloop/httptools for performance
- Concurrency and request limits
- Keep-alive tuning
- Security headers
- **Proxy headers policy** (for Nginx/Cloudflare)

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/settings.py` | Add Uvicorn configuration settings |
| `src/codeintel/cli/handlers/ops.py` | Update serve_http_handler with new config |

**Detailed Implementation**:

```python
# src/codeintel/serving/settings.py

@dataclass(frozen=True)
class ServingSettings:
    # ... existing fields ...

    # Uvicorn Configuration
    uvicorn_workers: int = 1
    uvicorn_loop: str = "auto"  # "auto", "asyncio", "uvloop"
    uvicorn_http: str = "auto"  # "auto", "h11", "httptools"
    uvicorn_limit_concurrency: int | None = None  # None = unlimited
    uvicorn_limit_max_requests: int | None = None  # Graceful worker restart
    uvicorn_timeout_keep_alive: int = 30
    uvicorn_backlog: int = 2048
    uvicorn_access_log: bool = True
    uvicorn_server_header: bool = False  # Hide "uvicorn" server header
    uvicorn_proxy_headers: bool = False  # Trust proxy headers
    uvicorn_forwarded_allow_ips: str = "127.0.0.1"  # IPs allowed to set X-Forwarded-*

    @classmethod
    def from_env(cls) -> ServingSettings:
        limit_concurrency_raw = os.environ.get("CODEINTEL_UVICORN_LIMIT_CONCURRENCY")
        limit_max_requests_raw = os.environ.get("CODEINTEL_UVICORN_LIMIT_MAX_REQUESTS")

        return cls(
            # ... existing fields ...
            uvicorn_workers=int(os.environ.get("CODEINTEL_UVICORN_WORKERS", "1")),
            uvicorn_loop=os.environ.get("CODEINTEL_UVICORN_LOOP", "auto"),
            uvicorn_http=os.environ.get("CODEINTEL_UVICORN_HTTP", "auto"),
            uvicorn_limit_concurrency=(
                int(limit_concurrency_raw) if limit_concurrency_raw else None
            ),
            uvicorn_limit_max_requests=(
                int(limit_max_requests_raw) if limit_max_requests_raw else None
            ),
            uvicorn_timeout_keep_alive=int(
                os.environ.get("CODEINTEL_UVICORN_TIMEOUT_KEEP_ALIVE", "30")
            ),
            uvicorn_backlog=int(os.environ.get("CODEINTEL_UVICORN_BACKLOG", "2048")),
            uvicorn_access_log=os.environ.get("CODEINTEL_UVICORN_ACCESS_LOG", "1") == "1",
            uvicorn_server_header=os.environ.get("CODEINTEL_UVICORN_SERVER_HEADER", "0") == "1",
            uvicorn_proxy_headers=os.environ.get("CODEINTEL_UVICORN_PROXY_HEADERS", "0") == "1",
            uvicorn_forwarded_allow_ips=os.environ.get(
                "CODEINTEL_UVICORN_FORWARDED_ALLOW_IPS", "127.0.0.1"
            ),
        )
```

```python
# src/codeintel/cli/handlers/ops.py

def serve_http_handler(ctx: CommandContext) -> CliResult[ServeStartResult]:
    """Start the HTTP serving server with production-grade Uvicorn configuration."""
    settings = ServingSettings.from_env()
    host = ctx.params.get_str("host") or settings.host
    port = ctx.params.get_int("port", settings.port)
    reload = ctx.params.get_bool("reload", default=False)
    workers = ctx.params.get_int("workers", settings.uvicorn_workers)

    pointer = ServingSnapshotPointer.load(settings.serve_dir / "current.json")
    LOG.info("Starting HTTP server at http://%s:%d (workers=%d)", host, port, workers)

    # Build Uvicorn configuration
    uvicorn_config: dict[str, object] = {
        "host": host,
        "port": port,
        "loop": settings.uvicorn_loop,
        "http": settings.uvicorn_http,
        "timeout_keep_alive": settings.uvicorn_timeout_keep_alive,
        "backlog": settings.uvicorn_backlog,
        "access_log": settings.uvicorn_access_log,
        "log_level": "info",
    }

    # Add optional limits
    if settings.uvicorn_limit_concurrency is not None:
        uvicorn_config["limit_concurrency"] = settings.uvicorn_limit_concurrency
    if settings.uvicorn_limit_max_requests is not None:
        uvicorn_config["limit_max_requests"] = settings.uvicorn_limit_max_requests

    # Security: hide server header in production
    if not settings.uvicorn_server_header:
        uvicorn_config["server_header"] = False

    # Proxy support for Nginx/Cloudflare
    if settings.uvicorn_proxy_headers:
        uvicorn_config["proxy_headers"] = True
        uvicorn_config["forwarded_allow_ips"] = settings.uvicorn_forwarded_allow_ips

    # Multi-worker mode guidance
    if workers > 1:
        LOG.info(
            "Multi-worker mode: Recommended CLI command for production:\n"
            "  uvicorn codeintel.serving.http.app:create_serving_app "
            "--factory --workers %d --host %s --port %d",
            workers,
            host,
            port,
        )
        uvicorn.run(
            "codeintel.serving.http.app:create_serving_app",
            factory=True,
            workers=workers,
            **uvicorn_config,
        )
    elif reload:
        uvicorn.run(
            "codeintel.serving.http.app:create_serving_app",
            factory=True,
            reload=True,
            **uvicorn_config,
        )
    else:
        app = create_serving_app(settings)
        uvicorn.run(app, **uvicorn_config)

    return CliResult.ok(...)
```

**Testing Requirements**:
- Test single-worker mode
- Test multi-worker mode (if workers > 1)
- Verify settings are applied correctly
- Test proxy headers when enabled

**Execution Guidance for PR5**:

1. **Add settings to `ServingSettings`** (~15 new fields):
   - `uvicorn_workers`, `uvicorn_loop`, `uvicorn_http`
   - `uvicorn_limit_concurrency`, `uvicorn_limit_max_requests`
   - `uvicorn_timeout_keep_alive`, `uvicorn_backlog`
   - `uvicorn_access_log`, `uvicorn_server_header`
   - `uvicorn_proxy_headers`, `uvicorn_forwarded_allow_ips`

2. **Update `serve_http_handler()` in `cli/handlers/ops.py`**:
   - Build `uvicorn_config` dict from settings
   - Handle multi-worker mode with factory pattern
   - Log recommended CLI command for production

3. **Key consideration**: With `workers > 1`, each worker gets its own `QueryLimiter`. This is actually correct behavior - the limit is per-process. Document this.

---

### M2: Authentication Enforcement for Non-Localhost 🔲 PENDING (PR5)

**Category**: C - Observability & Security

**Status**: 🔲 Not started - Planned for PR5 (bundled with M1)

**Estimated Effort**: 0.5 days

**Problem**: Auth is optional everywhere, but remote serving without auth is a security risk. Some MCP clients may refuse to connect to unauthenticated remote servers.

**Solution**: Enforce auth requirement when bound to non-localhost:
- `host in {"0.0.0.0", "::"}` ⇒ auth required (fail-fast at startup)
- `host == "127.0.0.1"` or `host == "localhost"` ⇒ auth optional

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/settings.py` | Add `auth_required_for_remote` setting |
| `src/codeintel/serving/http/app.py` | Add auth enforcement check at startup |

**Detailed Implementation**:

```python
# src/codeintel/serving/settings.py

@dataclass(frozen=True)
class ServingSettings:
    # ... existing fields ...
    auth_required_for_remote: bool = True  # Require auth for non-localhost

    def validate_auth_for_host(self) -> None:
        """Validate that auth is configured when binding to non-localhost.
        
        Raises
        ------
        ValueError
            If bound to public interface without auth configured.
        """
        if not self.auth_required_for_remote:
            return
        
        public_hosts = {"0.0.0.0", "::", ""}
        if self.host in public_hosts:
            if not self.auth_token and not self.api_key:
                msg = (
                    f"Security error: Binding to {self.host!r} requires authentication. "
                    f"Set CODEINTEL_AUTH_TOKEN or CODEINTEL_SERVE_API_KEY, "
                    f"or set CODEINTEL_AUTH_REQUIRED_FOR_REMOTE=0 to disable this check."
                )
                raise ValueError(msg)

    @classmethod
    def from_env(cls) -> ServingSettings:
        return cls(
            # ... existing fields ...
            auth_required_for_remote=os.environ.get(
                "CODEINTEL_AUTH_REQUIRED_FOR_REMOTE", "1"
            ) == "1",
        )
```

```python
# src/codeintel/serving/http/app.py

def create_serving_app(
    settings: ServingSettings | None = None,
    *,
    mount_mcp: bool = True,
) -> FastAPI:
    """Create FastAPI serving application."""
    cfg = settings or ServingSettings.from_env()
    
    # Fail-fast: require auth for public interfaces
    cfg.validate_auth_for_host()
    
    # ... rest of function ...
```

**Testing Requirements**:
- Test startup fails without auth when bound to 0.0.0.0
- Test startup succeeds with auth when bound to 0.0.0.0
- Test startup succeeds without auth when bound to 127.0.0.1

---

### M3: Custom MCP Route for Health Checks 🔲 PENDING (PR5)

**Category**: B - Uvicorn & Deployment

**Status**: 🔲 Not started - Planned for PR5 (bundled with M1, M2, M4)

**Estimated Effort**: 0.25 days

**Problem**: Health check exists on FastAPI (`/health`) but not on the MCP endpoint, preventing load balancers from checking MCP server health.

**Solution**: Add `@mcp.custom_route` for `/health` on the MCP server.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/app.py` | Add custom health route |

**Detailed Implementation**:

```python
# src/codeintel/serving/mcp/app.py

from starlette.responses import JSONResponse, PlainTextResponse

def build_mcp_app(...) -> FastMCP:
    mcp = FastMCP(...)

    @mcp.custom_route("/health", methods=["GET"])
    async def mcp_health() -> JSONResponse:
        """Health check for load balancers targeting MCP endpoint."""
        try:
            pointer = kernel.db.current_pointer()
            return JSONResponse({
                "status": "ok",
                "repo": pointer.repo,
                "commit": pointer.commit[:12],
                "run_id": pointer.run_id,
            })
        except RuntimeError:
            return JSONResponse(
                {"status": "error", "detail": "No active snapshot"},
                status_code=503,
            )

    @mcp.custom_route("/ready", methods=["GET"])
    async def mcp_ready() -> PlainTextResponse:
        """Readiness probe for Kubernetes/orchestrators."""
        try:
            kernel.db.current_pointer()
            return PlainTextResponse("ready")
        except RuntimeError:
            return PlainTextResponse("not ready", status_code=503)

    return mcp
```

**Testing Requirements**:
- Test `/health` returns 200 when healthy
- Test `/health` returns 503 when no snapshot
- Test `/ready` probe behavior

**Key Callouts**:
- The `@mcp.custom_route()` decorator creates standard Starlette routes on the MCP HTTP app
- These routes are accessible at `/mcp/health` and `/mcp/ready` when MCP is mounted
- Health check returns pointer info for debugging; readiness probe returns minimal text
- The `RuntimeError` catch handles case where no snapshot is loaded yet

---

### M4: Bearer Token Authentication for MCP 🔲 PENDING (PR5)

**Category**: C - Observability & Security

**Status**: 🔲 Not started - Planned for PR5 (bundled with M1-M3)

**Estimated Effort**: 0.25 days

**Problem**: HTTP routes have API key protection, but MCP server has no authentication.

**Solution**: Enable FastMCP's `auth_token` configuration using the existing `auth_token` setting.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/app.py` | Add auth_token to FastMCP init |

**Detailed Implementation**:

```python
# src/codeintel/serving/mcp/app.py

def build_mcp_app(
    *,
    kernel: SemanticKernel,
    settings: ServingSettings,
    ...
) -> FastMCP:
    """Build FastMCP application with semantic tools."""
    mcp = FastMCP(
        "CodeIntel",
        json_response=True,
        mask_error_details=settings.mcp_mask_errors,
        auth_token=settings.auth_token,  # Use existing setting for MCP auth
    )
    # ... rest of function ...
```

**Client Configuration**:
- ChatGPT: Enter token in Connector setup UI
- Claude: Configure in MCP settings
- Cursor: Add to `mcp.json` configuration

**Testing Requirements**:
- Test request without token returns 401
- Test request with valid token succeeds
- Test request with invalid token returns 401

**Key Callouts**:
- The `auth_token` parameter in FastMCP constructor enables Bearer token auth
- Reuse the existing `settings.auth_token` field - no new setting needed
- FastMCP handles the 401 response automatically for invalid/missing tokens
- Works with both StreamableHTTP and SSE transports
- STDIO transport (for Claude Desktop) doesn't use HTTP auth - it relies on local trust

---

### M5: Tags for Tool Organization 🔲 PENDING (PR6)

**Category**: D - Architecture & Extensibility

**Status**: 🔲 Not started - Planned for PR6

**Estimated Effort**: 0.25 days

**Problem**: All 6 tools are flat with no categorization, making it harder to manage as the tool set grows.

**Solution**: Add tags to tools for logical grouping and potential filtering.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/app.py` | Add tags to all tool decorators |

**Detailed Implementation**:

```python
# src/codeintel/serving/mcp/app.py

# Tag constants for consistency (add near top of file after imports)
TAG_SEMANTIC = "semantic"
TAG_SEARCH = "search"
TAG_META = "meta"
TAG_READ = "read"
TAG_EXPORT = "export"

# Apply to existing tools - example for semantic_catalog:
@mcp.tool(
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
    tags=[TAG_SEMANTIC, TAG_READ],
)
async def semantic_catalog(*, ctx: Context) -> dict[str, object]:
    ...

# Tool tag mapping (all 7 tools):
# - semantic_catalog:  [TAG_SEMANTIC, TAG_READ]
# - semantic_describe: [TAG_SEMANTIC, TAG_READ]
# - semantic_query:    [TAG_SEMANTIC, TAG_READ]
# - semantic_explain:  [TAG_SEMANTIC, TAG_READ]
# - semantic_meta:     [TAG_META, TAG_READ]
# - code_search:       [TAG_SEARCH, TAG_READ]
# - semantic_export:   [TAG_SEMANTIC, TAG_EXPORT]
```

**Implementation Pattern** (from PR4/PR5 learnings):

The `@mcp.tool()` decorator accepts a `tags` parameter as a list of strings. Based on our PR4 implementation, the correct pattern is:

```python
@mcp.tool(
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,  # Existing annotation
    tags=["semantic", "read"],                  # Add tags parameter
)
async def tool_name(..., *, ctx: Context) -> dict[str, object]:
    ...
```

**Future Use**:
- `include_tags`/`exclude_tags` on FastMCP server for filtering
- Client-side tool discovery by tag
- Admin tools with `internal` tag that can be excluded

**Key Callouts**:
- Tags are metadata only - they don't change tool behavior
- Define tag constants at module level for consistency and typo prevention
- Current tags scheme: `semantic`, `search`, `meta` (functional) + `read`, `export` (operation type)
- Tags appear in tool discovery response, helping LLMs choose appropriate tools
- Very low effort - just add `tags=[...]` parameter to existing `@mcp.tool()` decorators
- The `annotations` parameter already exists on all tools - just add `tags` alongside it

**PR4/PR5 Insights Applied**:
- Tool registration happens via inner functions inside `build_mcp_app()` or via helper registration functions (like `_register_export_tool`)
- Tags should be added consistently to all 7 tools (6 original + 1 export tool added in PR4)
- Verify tags are present by checking `mcp._tool_manager.get_tools()` returns tools with correct metadata

---

### M6: Metrics Emission from MCP Tools 🔲 PENDING (PR6)

**Category**: C - Observability & Security

**Status**: 🔲 Not started - Planned for PR6 (bundled with M5, M7)

**Estimated Effort**: 0.5 days

**Problem**: HTTP routes have background metrics logging, but MCP tools have no metrics emission.

**Solution**: Reuse the existing `QueryMetrics` infrastructure for MCP tools.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/app.py` | Add metrics logging to all tools |

**Detailed Implementation**:

```python
# src/codeintel/serving/mcp/app.py

import time
from codeintel.serving.http.metrics import QueryMetrics, log_query_metrics

@mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS, tags=[TAG_SEMANTIC, TAG_READ])
async def semantic_query(..., *, ctx: Context) -> dict[str, object]:
    """Query a semantic view with structured filters."""
    start = time.perf_counter()
    result: dict[str, object] | None = None
    row_count = 0
    truncated = False

    try:
        await ctx.info(f"Querying view: {view_id}")
        # ... build request and execute ...
        result_raw = await limiter.run(kernel.query, request)
        row_count = len(result_raw.rows)
        truncated = result_raw.truncated
        result = build_envelope(kernel, result_raw.model_dump(), ...).model_dump()
        return result

    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        log_query_metrics(QueryMetrics(
            endpoint="mcp:semantic_query",
            view_id=view_id,
            query=None,
            row_count=row_count,
            truncated=truncated,
            duration_ms=duration_ms,
            correlation_id=ctx.session_id or "mcp-unknown",
        ))
```

**Implementation Pattern** (from PR4/PR5 learnings):

Based on existing tool structure in `app.py`, the metrics wrapper should follow this pattern:

```python
@mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS, tags=[TAG_SEMANTIC, TAG_READ])
async def semantic_query(
    view_id: str,
    filters: list[dict[str, object]] | None = None,
    limit: int = 100,
    *,
    ctx: Context,
) -> dict[str, object]:
    """Query a semantic view with structured filters."""
    start = time.perf_counter()
    row_count = 0
    truncated = False

    try:
        await ctx.info(f"Querying view: {view_id}")
        request = SemanticQueryRequest(view_id=view_id, filters=filters, limit=limit)
        result = await limiter.run(kernel.query, request)
        row_count = len(result.rows)
        truncated = result.truncated
        return build_envelope(kernel, result.model_dump(mode="json")).model_dump(mode="json")

    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        log_query_metrics(QueryMetrics(
            endpoint="mcp:semantic_query",
            view_id=view_id,
            query=None,
            row_count=row_count,
            truncated=truncated,
            duration_ms=duration_ms,
            correlation_id=getattr(ctx, "session_id", None) or "mcp-unknown",
        ))
```

**Key Insight from PR4**: The `ctx: Context` parameter is from `fastmcp.Context`, and `session_id` may not be available in all contexts (e.g., streamable HTTP). Use `getattr()` for safe access.

**Testing Requirements**:
- Verify metrics are logged for MCP tool calls
- Test correlation ID from ctx.session_id
- Verify metrics format matches HTTP route metrics
- Test metrics are logged even when tool raises `ToolError`

**Key Callouts**:
- Must import and reuse existing `QueryMetrics` / `log_query_metrics` from HTTP layer
- The `ctx.session_id` may be `None` for streamable HTTP - handle gracefully with `getattr(ctx, "session_id", None)`
- Metrics logging should be in `finally` block to capture even on errors
- Consider whether to use `ctx.info()` in addition to structured metrics logging
- This item pairs naturally with M5 (tags) for filtering metrics by tag
- All 7 tools (including `semantic_export` added in PR4) need the try/finally wrapper pattern

**Dependencies**:
- Requires no new files - reuses existing HTTP metrics infrastructure
- All 7 tools in `app.py` need the try/finally wrapper pattern

**PR4/PR5 Insights Applied**:
- The `semantic_export` tool added in PR4 via `_register_export_tool()` also needs metrics
- The tool pattern uses `await limiter.run(kernel.method, request)` for blocking operations
- Error cases raise `ToolError` which should still trigger metrics logging via `finally`

---

### M7: Tool Enable/Disable for Feature Flags 🔲 PENDING (PR6)

**Category**: D - Architecture & Extensibility

**Status**: 🔲 Not started - Planned for PR6 (bundled with M5, M6)

**Estimated Effort**: 0.5 days

**Problem**: All tools are always enabled; no way to conditionally disable features.

**Solution**: Use FastMCP's `enabled` parameter with settings-driven configuration.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/app.py` | Add enabled parameter based on settings |
| `src/codeintel/serving/settings.py` | Add feature flags for MCP tools |

**Detailed Implementation**:

```python
# src/codeintel/serving/settings.py

@dataclass(frozen=True)
class ServingSettings:
    # ... existing fields ...

    # MCP Tool Feature Flags (add after existing mcp_* fields)
    mcp_enable_search: bool = True
    mcp_enable_explain: bool = True
    mcp_enable_meta: bool = True
    mcp_enable_export: bool = True

    @classmethod
    def from_env(cls) -> ServingSettings:
        return cls(
            # ... existing fields (see PR5 implementation for full list) ...
            mcp_enable_search=os.environ.get("CODEINTEL_MCP_ENABLE_SEARCH", "1") == "1",
            mcp_enable_explain=os.environ.get("CODEINTEL_MCP_ENABLE_EXPLAIN", "1") == "1",
            mcp_enable_meta=os.environ.get("CODEINTEL_MCP_ENABLE_META", "1") == "1",
            mcp_enable_export=os.environ.get("CODEINTEL_MCP_ENABLE_EXPORT", "1") == "1",
        )
```

**Implementation Pattern** (from PR4/PR5 learnings):

Based on the `ServingSettings` pattern established in PR5:

```python
# src/codeintel/serving/settings.py - Add to existing fields:

# MCP Tool Feature Flags
mcp_enable_search: bool = True
mcp_enable_explain: bool = True
mcp_enable_meta: bool = True
mcp_enable_export: bool = True

# In from_env() method, add alongside existing loads:
mcp_enable_search=os.environ.get("CODEINTEL_MCP_ENABLE_SEARCH", "1") == "1",
mcp_enable_explain=os.environ.get("CODEINTEL_MCP_ENABLE_EXPLAIN", "1") == "1",
mcp_enable_meta=os.environ.get("CODEINTEL_MCP_ENABLE_META", "1") == "1",
mcp_enable_export=os.environ.get("CODEINTEL_MCP_ENABLE_EXPORT", "1") == "1",
```

```python
# src/codeintel/serving/mcp/app.py - Modify tool registration:

# For helper-registered tools like semantic_export (PR4 pattern):
def _register_export_tool(
    mcp: FastMCP,
    kernel: SemanticKernel,
    limiter: anyio.CapacityLimiter,
    store: ResourceStore,
    settings: ServingSettings,
) -> None:
    """Register semantic_export tool."""
    @mcp.tool(
        annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
        tags=[TAG_SEMANTIC, TAG_EXPORT],
        enabled=settings.mcp_enable_export,  # Add this
    )
    async def semantic_export(..., *, ctx: Context) -> dict[str, object]:
        ...

# For inline-registered tools (wrap with conditional):
if settings.mcp_enable_search:
    @mcp.tool(
        annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
        tags=[TAG_SEARCH, TAG_READ],
    )
    async def code_search(..., *, ctx: Context) -> dict[str, object]:
        ...
```

**Important**: The `enabled` parameter may not work as expected with the decorator pattern. An alternative is to use conditional registration:

```python
# Alternative: Conditional registration (more reliable)
def _register_search_tool(mcp: FastMCP, kernel: SemanticKernel, settings: ServingSettings) -> None:
    if not settings.mcp_enable_search:
        return  # Skip registration entirely

    @mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS, tags=[TAG_SEARCH, TAG_READ])
    async def code_search(..., *, ctx: Context) -> dict[str, object]:
        ...
```

**Testing Requirements**:
- Test disabled tool returns "Unknown tool" error
- Test enabled tools work normally
- Verify tool list excludes disabled tools
- Test environment variable loading for feature flags

**Key Callouts**:
- The `enabled` parameter on `@mcp.tool()` is evaluated at registration time, not at runtime
- Core tools (`semantic_catalog`, `semantic_query`, `semantic_describe`) should NOT be feature-flagged
- Only advanced/optional tools should have enable/disable flags
- Environment variable naming convention: `CODEINTEL_MCP_ENABLE_<FEATURE>`
- This is useful for gradual rollouts or disabling experimental features in production

**PR4/PR5 Insights Applied**:
- Follow the established `ServingSettings.from_env()` pattern with `os.environ.get(..., "1") == "1"` for boolean flags
- The `semantic_export` tool uses a helper registration function `_register_export_tool()` - use conditional within that helper
- Verify tool exclusion via `mcp._tool_manager.get_tools()` which returns a dict of tool names
- Test pattern similar to `tests/serving/test_auth_enforcement.py` using `_set_env()` context manager

**Implementation Note**:
The `enabled=settings.mcp_enable_X` must be evaluated when the tool is registered:
```python
# This works because settings is captured at registration time
@mcp.tool(enabled=settings.mcp_enable_search)
async def code_search(...): ...

# NOT this - lambda would be evaluated each call but enabled is registration-time
# @mcp.tool(enabled=lambda: settings.mcp_enable_search)  # WRONG
```

---

### M8: Unified Lifespan Management 🔲 PENDING (PR7)

**Category**: D - Architecture & Extensibility

**Status**: 🔲 Not started - Planned for PR7 (bundled with L1, L2)

**Estimated Effort**: 1 day

**Problem**: Separate lifespan contexts for FastAPI and standalone MCP, causing code duplication.

**Solution**: Allow dependency injection of ServingDBManager to share lifecycle.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/server.py` | Accept optional db_manager injection |

**Current Architecture** (from PR4/PR5 implementation):

The current flow in `http/app.py`:
```python
# create_serving_app() creates:
db_manager = ServingDBManager(...)  # Owns the lifecycle
kernel = SemanticQueryKernel(db=db_manager, settings=cfg)
# ...
app = FastAPI(lifespan=_build_lifespan(db_manager))  # FastAPI manages db_manager

# _maybe_mount_mcp() receives:
def _maybe_mount_mcp(app: FastAPI, *, kernel: SemanticKernel, settings: ServingSettings, enabled: bool) -> None:
    mcp = build_mcp_app(kernel=kernel, settings=settings)  # Kernel already connected to db_manager
    # EventStore integration if available
    app.mount("/mcp", mcp.http_app(...))
```

**Key Insight**: The `kernel` parameter already contains the `db` reference, so `build_mcp_app()` doesn't need to manage db lifecycle.

**Detailed Implementation**:

```python
# src/codeintel/serving/mcp/server.py

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from codeintel.serving.db_manager import ServingDBManager

def create_mcp_server(
    settings: ServingSettings | None = None,
    *,
    db_manager: ServingDBManager | None = None,
) -> FastMCP:
    """Create an MCP server bound to the current serving snapshot.

    Parameters
    ----------
    settings
        Serving settings (defaults to environment).
    db_manager
        Optional pre-configured database manager. If provided, the MCP server
        will not manage its lifecycle (caller is responsible for start/stop).
        If None, creates and manages its own db_manager.

    Returns
    -------
    FastMCP
        Configured MCP server.
    """
    cfg = settings or ServingSettings.from_env()
    
    # Fail-fast security check (from PR5)
    cfg.validate_auth_for_host()

    # Use injected or create new
    if db_manager is None:
        db_manager = ServingDBManager(
            pointer_path=cfg.serve_dir / "current.json",
            pool_cfg=PoolConfig(size=cfg.pool_size),
            poll_interval_s=cfg.poll_interval_s,
            hot_swap=cfg.hot_swap,
        )
        owns_db_manager = True
    else:
        owns_db_manager = False

    kernel = SemanticQueryKernel(db=db_manager, settings=cfg)

    @asynccontextmanager
    async def lifespan(_mcp: FastMCP) -> AsyncGenerator[object]:
        if owns_db_manager:
            await db_manager.start()
        try:
            yield object()
        finally:
            if owns_db_manager:
                await db_manager.stop()

    return build_mcp_app(
        kernel=kernel,
        settings=cfg,
        lifespan=lifespan,
    )
```

**Usage Patterns**:

```python
# Standalone MCP server (STDIO/SSE) - creates own db_manager
mcp = create_mcp_server()  # Owns and manages db_manager lifecycle

# Embedded in FastAPI (HTTP app owns db_manager)
# In http/app.py - keep using build_mcp_app() directly:
mcp = build_mcp_app(kernel=kernel, settings=cfg)  # kernel already has db reference
```

**Key Callouts**:
- The `owns_db_manager` pattern is critical - prevents double-stop on shared resources
- When HTTP app mounts MCP, the HTTP app owns `ServingDBManager` and MCP uses it
- When MCP runs standalone (STDIO/SSE), MCP owns and manages the lifecycle
- This enables clean embedding in FastAPI via `http/app.py` routing
- Include `cfg.validate_auth_for_host()` call from PR5 for security

**Testing Requirements**:
- Test standalone MCP server starts and stops cleanly
- Test MCP embedded in FastAPI shares db_manager (no double lifecycle management)
- Verify no double-close errors when shutting down composed app
- Test with injected mock db_manager for unit testing

**Dependencies**:
- Pairs naturally with L1 (Server Composition) for full modularity
- Must coordinate with existing `http/app.py` lifespan

**PR4/PR5 Insights Applied**:
- The `kernel` in `build_mcp_app()` already has `db` reference via `kernel.db`
- The EventStore integration in `_maybe_mount_mcp()` passes `event_store` to `mcp.http_app()`
- Bearer auth (PR5) should be applied in `create_mcp_server()` as well via `settings.auth_token`
- The `_compat.py` module provides consistent imports for `FastMCP`, `Context`, `ToolError`, etc.

---

## Low-Priority Enhancements

### L1: Server Composition for Modularity 🔲 PENDING (PR7)

**Category**: D - Architecture & Extensibility

**Status**: 🔲 Not started - Planned for PR7 (bundled with M8, L2)

**Estimated Effort**: 1 day

**Problem**: Single monolithic MCP app; as tool count grows, maintenance becomes harder.

**Solution**: Split tools into composable sub-servers using FastMCP's `mount()`.

**Files to Create/Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/servers/__init__.py` | New: Package init |
| `src/codeintel/serving/mcp/servers/semantic.py` | New: Semantic tools sub-server |
| `src/codeintel/serving/mcp/servers/search.py` | New: Search tools sub-server |
| `src/codeintel/serving/mcp/servers/meta.py` | New: Meta tools sub-server |
| `src/codeintel/serving/mcp/app.py` | Compose sub-servers |

**Current Tool Inventory** (from PR4 implementation):

| Tool | Current Location | Target Sub-Server |
|------|------------------|-------------------|
| `semantic_catalog` | `app.py` inline | `servers/semantic.py` |
| `semantic_describe` | `app.py` inline | `servers/semantic.py` |
| `semantic_query` | `app.py` inline | `servers/semantic.py` |
| `semantic_explain` | `app.py` inline | `servers/semantic.py` |
| `semantic_meta` | `app.py` inline | `servers/meta.py` |
| `code_search` | `app.py` inline | `servers/search.py` |
| `semantic_export` | `_register_export_tool()` | `servers/semantic.py` |

**Implementation Sketch**:

```python
# src/codeintel/serving/mcp/servers/semantic.py

from __future__ import annotations
from typing import TYPE_CHECKING
from codeintel.serving.mcp._compat import Context, FastMCP, ToolError

if TYPE_CHECKING:
    import anyio
    from codeintel.serving.mcp.app import SemanticKernel
    from codeintel.serving.mcp.resource_store import ResourceStore
    from codeintel.serving.settings import ServingSettings

def build_semantic_server(
    kernel: SemanticKernel,
    settings: ServingSettings,
    limiter: anyio.CapacityLimiter,
    store: ResourceStore,
) -> FastMCP:
    """Build semantic-focused MCP sub-server."""
    mcp = FastMCP("CodeIntel-Semantic")

    @mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS, tags=["semantic", "read"])
    async def catalog(*, ctx: Context) -> dict[str, object]:
        """List available semantic views."""
        await ctx.info("Listing semantic catalog")
        views = await limiter.run(kernel.catalog)
        return build_envelope(kernel, views).model_dump(mode="json")

    @mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS, tags=["semantic", "read"])
    async def query(
        view_id: str,
        filters: list[dict[str, object]] | None = None,
        limit: int = 100,
        *,
        ctx: Context,
    ) -> dict[str, object]:
        """Query a semantic view."""
        # ... implementation ...

    # ... other semantic tools ...

    return mcp
```

```python
# src/codeintel/serving/mcp/app.py

def build_mcp_app(
    *,
    kernel: SemanticKernel,
    settings: ServingSettings,
    lifespan: Callable[[FastMCP], AbstractAsyncContextManager[object]] | None = None,
) -> FastMCP:
    """Build composed MCP application."""
    main = FastMCP(
        "CodeIntel",
        json_response=True,
        mask_error_details=settings.mcp_mask_errors,
        lifespan=lifespan,
        auth=create_bearer_auth(settings.auth_token),  # From PR5
    )

    limiter = anyio.CapacityLimiter(settings.mcp_max_concurrent)
    store = ResourceStore(settings.serve_dir / "exports")

    # Compose sub-servers
    semantic = build_semantic_server(kernel, settings, limiter, store)
    search = build_search_server(kernel, settings, limiter)
    meta = build_meta_server(kernel, settings, limiter)

    main.mount(semantic, prefix="semantic")
    main.mount(search, prefix="search")
    main.mount(meta, prefix="meta")

    # Register resources on main server (from PR4)
    register_resources(main, kernel, store)

    return main
```

**Key Callouts**:
- The `mount(sub_server, prefix="semantic")` makes tools available as `semantic/catalog`, `semantic/query`, etc.
- Prefix naming should be intuitive for LLM agents discovering tools
- Sub-servers can have their own lifespan contexts if needed
- This is a refactor - all existing tool functionality must be preserved
- Resources stay on main server - they use `codeintel://` URI scheme

**Testing Requirements**:
- Verify all existing tool tests pass after composition
- Test tool discovery returns prefixed names (e.g., `semantic/catalog`)
- Test calling prefixed tools works correctly
- Verify `mcp._tool_manager.get_tools()` returns prefixed names

**PR4/PR5 Insights Applied**:
- The `limiter` and `store` need to be shared across sub-servers (created in `build_mcp_app()`)
- Resources registered via `register_resources()` stay on main server
- The `auth=create_bearer_auth(...)` from PR5 goes on main server only
- Health routes via `@mcp.custom_route()` stay on main server
- Consider whether sub-servers need the same `json_response=True` setting

**Implementation Considerations**:
- Tool names will change from `semantic_catalog` to `semantic/catalog` - this is a BREAKING CHANGE for clients
- Consider providing aliases during migration period
- Update any documentation/prompts that reference tool names
- Test with Claude Desktop, Cursor, and other MCP clients

**Note**: This is a structural refactor that doesn't change functionality. Consider implementing after M1-M8 are stable.

---

### L2: MCP Prompts for Guided Interactions 🔲 PENDING (PR7)

**Category**: A - MCP Tool Enhancements

**Status**: 🔲 Not started - Planned for PR7 (bundled with M8, L1)

**Estimated Effort**: 0.5 days

**Problem**: No guided prompts for common workflows; LLMs must discover tool usage patterns.

**Solution**: Add FastMCP prompts for common interaction patterns.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/prompts.py` | New file with prompt definitions |
| `src/codeintel/serving/mcp/app.py` | Register prompts |

**Implementation Sketch**:

```python
# src/codeintel/serving/mcp/prompts.py

from __future__ import annotations
from codeintel.serving.mcp._compat import FastMCP

def register_prompts(mcp: FastMCP) -> None:
    """Register guided prompts for common workflows.

    Parameters
    ----------
    mcp
        FastMCP application to register prompts on.

    Notes
    -----
    Prompts are discoverable via MCP protocol's `list_prompts()` method.
    LLM clients can request them to get guided workflows.
    """
    @mcp.prompt()
    def explore_codebase() -> str:
        """Guided workflow for exploring an unfamiliar codebase."""
        return """
        To explore this codebase:
        1. First call semantic_catalog() to see available data views
        2. Pick a view that interests you and call semantic_describe(view_id=...)
        3. Use semantic_query(view_id=...) to fetch data
        4. Use code_search(query=...) to find specific code patterns
        """

    @mcp.prompt()
    def find_function(name: str) -> str:
        """Guided workflow for finding and understanding a function.

        Parameters
        ----------
        name
            Name of the function to find.
        """
        return f"""
        To find function '{name}':
        1. Use code_search(query="{name}") to locate it
        2. Get details via semantic_describe(view_id="functions")
        3. Check its callers via semantic_query(view_id="call_graph", filters=[...])
        """

    @mcp.prompt()
    def export_data(view_id: str) -> str:
        """Guided workflow for exporting large datasets.

        Parameters
        ----------
        view_id
            The semantic view to export.
        """
        return f"""
        To export data from '{view_id}':
        1. First preview with semantic_query(view_id="{view_id}", limit=10)
        2. For full export, call semantic_export(view_id="{view_id}", format="ndjson")
        3. The response includes a resource URI - use it to download the data
        4. NDJSON format is recommended for large datasets (streaming-friendly)
        """
```

```python
# src/codeintel/serving/mcp/app.py - Add to build_mcp_app():

from codeintel.serving.mcp.prompts import register_prompts

def build_mcp_app(...) -> FastMCP:
    mcp = FastMCP(...)
    # ... existing tool registration ...

    # Register guided prompts
    register_prompts(mcp)

    return mcp
```

**Key Callouts**:
- Prompts are discoverable via MCP protocol - LLMs can request them
- Prompts should encode "best practices" for using the tool suite
- Keep prompts concise and actionable
- Use templating (f-strings) for dynamic prompts with parameters
- If L1 (Server Composition) is implemented, update tool names to prefixed versions (e.g., `semantic/catalog`)

**Testing Requirements**:
- Verify prompts are listed via `list_prompts()` or `mcp._prompt_manager.get_prompts()`
- Test prompt rendering with arguments
- Verify prompts are valid text (no rendering errors)
- Test prompt discovery in integration test

**PR4/PR5 Insights Applied**:
- Follow the same registration pattern as resources: create `register_prompts(mcp)` helper
- Call `register_prompts(mcp)` in `build_mcp_app()` after tool registration
- The `@mcp.prompt()` decorator is simpler than `@mcp.tool()` - no annotations needed
- Prompts with parameters use function arguments (e.g., `def find_function(name: str)`)
- Add NumPy-style docstrings to prompt functions for clarity

**Suggested Prompts** (based on tool inventory):
1. `explore_codebase()` - Getting started workflow
2. `find_function(name)` - Locate and understand a function
3. `export_data(view_id)` - Export large datasets (uses PR4's `semantic_export`)
4. `analyze_dependencies()` - Explore import/call graphs
5. `review_metrics()` - Examine code quality metrics

---

### L3: Correlation ID Propagation to MCP 🔲 PENDING (Optional)

**Category**: C - Observability & Security

**Status**: 🔲 Not started - Optional enhancement

**Estimated Effort**: 0.5 days

**Problem**: Correlation IDs are HTTP-middleware only; MCP tools use session_id but not integrated.

**Solution**: Use contextvars to propagate correlation ID across tool execution.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/context.py` | New file for MCP context management |
| `src/codeintel/serving/mcp/app.py` | Set correlation_id from ctx.session_id |

**Current State** (from PR4/PR5):

The `ctx: Context` parameter in MCP tools provides:
- `ctx.session_id` - Session identifier (may be None for streamable HTTP)
- `ctx.info()`, `ctx.warning()`, `ctx.error()` - Logging methods
- Other context methods

**Implementation Sketch**:
```python
# src/codeintel/serving/mcp/context.py

from __future__ import annotations
from contextvars import ContextVar
from uuid import uuid4

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

def get_correlation_id() -> str:
    """Get current correlation ID or generate one.

    Returns
    -------
    str
        Current correlation ID or newly generated UUID.
    """
    cid = correlation_id_var.get()
    if not cid:
        cid = str(uuid4())
        correlation_id_var.set(cid)
    return cid

def set_correlation_id(cid: str) -> None:
    """Set correlation ID for current context.

    Parameters
    ----------
    cid
        Correlation ID to set.
    """
    correlation_id_var.set(cid)

def correlation_id_from_ctx(ctx: Context) -> str:
    """Extract or generate correlation ID from MCP context.

    Parameters
    ----------
    ctx
        FastMCP Context object.

    Returns
    -------
    str
        Correlation ID (session_id if available, else generated).
    """
    session_id = getattr(ctx, "session_id", None)
    if session_id:
        return session_id
    return get_correlation_id()
```

**Integration with M6 (Metrics)**:
```python
# In tool implementation (from M6):
finally:
    duration_ms = (time.perf_counter() - start) * 1000
    log_query_metrics(QueryMetrics(
        endpoint="mcp:semantic_query",
        # ... other fields ...
        correlation_id=correlation_id_from_ctx(ctx),  # Use helper
    ))
```

**Key Callouts**:
- `ctx.session_id` from FastMCP is the natural correlation ID source for MCP tools
- For HTTP-embedded MCP, the HTTP correlation ID middleware should set the contextvar
- Logs and metrics should include the correlation ID for distributed tracing
- This is optional because M6 (metrics) can use `ctx.session_id` directly
- Use `getattr(ctx, "session_id", None)` for safe access (session_id may not exist)

**Testing Requirements**:
- Verify correlation ID flows from HTTP middleware to MCP tool logs
- Test correlation ID is consistent across a single request
- Test fallback to generated UUID when session_id is None

**PR4/PR5 Insights Applied**:
- The `ctx: Context` is from `fastmcp.Context` imported via `_compat.py`
- Access `session_id` safely with `getattr()` - it may not be present in all contexts
- The contextvar pattern allows propagation across async boundaries
- Consider using `structlog` if structured logging is desired

---

### L4: OpenAPI/FastAPI Integration Documentation 🔲 PENDING (Optional)

**Category**: D - Architecture & Extensibility

**Status**: 🔲 Not started - Documentation item

**Estimated Effort**: 0.5 days

**Problem**: HTTP routes and MCP tools are defined separately; potential for drift.

**Solution**: Document pattern for `FastMCP.from_fastapi()` integration.

**Key Callouts**:
- `FastMCP.from_fastapi(app)` auto-generates MCP tools from FastAPI routes
- Our current design has MCP tools as the primary API, with HTTP routes as secondary
- This documentation should explain WHY we chose manual tool definitions
- Useful for future consideration if HTTP API expands significantly

**Deliverable**: Add section to `docs/architecture/serving.md` explaining the design choice.

---

### L5: Proxy Pattern for Remote Services 🔲 PENDING (Optional)

**Category**: D - Architecture & Extensibility

**Status**: 🔲 Not started - Future extensibility

**Estimated Effort**: 0.5 days

**Problem**: No pattern for integrating external MCP services.

**Solution**: Document proxy pattern for future extensibility using FastMCP's `Client` and proxy helpers.

**Key Callouts**:
- FastMCP supports creating proxy servers that forward to upstream MCP servers
- Pattern: `FastMCP().proxy(client=remote_client)` or `register_all()` from upstream
- Useful for aggregating multiple code analysis services (e.g., SCIP indexer, test runner)
- This is a documentation/design item - no implementation required yet

**Deliverable**: Add section to `docs/architecture/serving.md` documenting the proxy pattern.

---

### L6: Mount Path Contract Test 🔲 PENDING (PR7)

**Category**: B - Uvicorn & Deployment

**Status**: 🔲 Not started - Planned for PR7 (bundled with M8, L1, L2)

**Estimated Effort**: 0.5 days

**Problem**: No test validates the mount path contract (effective endpoint is `/mcp`, not `/mcp/mcp`).

**Solution**: Add explicit test for mount path behavior.

**Files to Create**:

| File | Changes |
|------|---------|
| `tests/serving/http/test_mcp_mount.py` | New: Mount path contract tests |

**Implementation Sketch**:

```python
# tests/serving/http/test_mcp_mount.py

"""Tests for MCP mount path contract."""

import pytest
from fastapi.testclient import TestClient

from codeintel.serving.http.app import create_serving_app


def test_mcp_mount_path_contract(serving_app: FastAPI) -> None:
    """Verify MCP is accessible at /mcp, not /mcp/mcp."""
    client = TestClient(serving_app)
    
    # /mcp should respond (200 or redirect to /mcp/)
    response = client.get("/mcp/health")
    assert response.status_code in {200, 307}, "MCP should be mounted at /mcp"
    
    # /mcp/mcp should NOT exist (404)
    response = client.get("/mcp/mcp/health")
    assert response.status_code == 404, "Double prefix /mcp/mcp should not exist"
```

**Key Callouts**:
- This test catches a common FastMCP mounting error
- The mount happens in `http/app.py` via `app.mount("/mcp", mcp.http_app())`
- The FastMCP `root_path` setting must be coordinated with the mount path
- Regression test prevents accidental double-prefix bugs

**Testing Requirements**:
- Verify /mcp/* routes work
- Verify /mcp/mcp/* returns 404
- Test with different mount paths to ensure contract is understood

---

## Implementation Phases

### Phase 0: Critical Foundation ✅ COMPLETE (PR1)
**Duration**: 1 day
**Status**: ✅ Completed

| Day | Items | Focus | Status |
|-----|-------|-------|--------|
| 1 | C0 | Normalize to gofastmcp 2.x | ✅ Done |

**Gate**: ✅ All imports use gofastmcp, MCP tools functional

### Phase 1: Core Tool Enhancements (High Priority) ✅ COMPLETE (PR2 + PR3)
**Duration**: 4-5 days
**Status**: ✅ Completed

| Day | Items | Focus | Status |
|-----|-------|-------|--------|
| 1 | H1, H2, H7 | Context API + Annotations + ToolError | ✅ Done (PR2) |
| 2 | H3, H4 | Response envelope + Query limiter | ✅ Done (PR3) |
| 3-4 | H5, H6 | Resources + EventStore | 🔲 Pending (PR4) |

**Gate**: H1-H4, H7 complete ✅. H5, H6 pending in PR4.

### Remaining High-Priority Work (PR4)
**Duration**: 2 days
**Status**: 🔲 Not started

| Item | Focus | Estimated |
|------|-------|-----------|
| H5 | MCP Resources for Export Artifacts | 1 day |
| H6 | EventStore for Resumable Exports | 1 day |

**Gate**: Export workflow complete with SSE resumability

### Phase 2: Production Hardening (Medium Priority) 🔲 PENDING (PR5 + PR6)
**Duration**: 3-4 days
**Status**: 🔲 Not started

| PR | Items | Focus | Estimated |
|----|-------|-------|-----------|
| PR5 | M1, M2, M3, M4 | Uvicorn config + Auth + Health | 2 days |
| PR6 | M5, M6, M7 | Tags + Metrics + Feature flags | 1.5 days |

**Gate**: All medium-priority items complete, integration tests passing

### Phase 3: Extensibility (Low Priority) 🔲 PENDING (PR7)
**Duration**: 2-3 days
**Status**: 🔲 Not started

| PR | Items | Focus | Estimated |
|----|-------|-------|-----------|
| PR7 | M8, L1, L2, L6 | Lifespan + Composition + Prompts + Mount test | 2 days |

**Optional Items** (can be done ad-hoc):
- L3: Correlation ID propagation
- L4: OpenAPI integration docs
- L5: Proxy pattern docs

**Gate**: All items complete, documentation updated

---

## File Change Matrix

### New Files

| File | Purpose | Phase | Status |
|------|---------|-------|--------|
| `src/codeintel/serving/mcp/_compat.py` | FastMCP import shim | P0 | ✅ Created |
| `src/codeintel/serving/mcp/models.py` | Pydantic response models + envelope | P1 | ✅ Created |
| `src/codeintel/serving/mcp/response.py` | Envelope builder helper | P1 | ✅ Created |
| `src/codeintel/serving/mcp/runtime.py` | QueryLimiter | P1 | ✅ Created |
| `src/codeintel/serving/mcp/resource_store.py` | Export artifact storage | P1 (H5) | 🔲 Pending |
| `src/codeintel/serving/mcp/resources.py` | MCP resource handlers | P1 (H5) | 🔲 Pending |
| `src/codeintel/serving/mcp/prompts.py` | MCP prompt templates | P3 | 🔲 Pending |
| `src/codeintel/serving/mcp/context.py` | Correlation ID context | P3 | 🔲 Optional |
| `src/codeintel/serving/mcp/servers/` | Sub-server modules | P3 | 🔲 Pending |
| `tests/serving/mcp/__init__.py` | Test package | P1 | ✅ Created |
| `tests/serving/mcp/test_runtime.py` | QueryLimiter tests | P1 | ✅ Created |
| `tests/serving/mcp/test_resources.py` | MCP resource tests | P1 (H5) | 🔲 Pending |
| `tests/serving/http/test_mcp_mount.py` | Mount path contract test | P3 | 🔲 Pending |

### Modified Files

| File | Items Affecting | Phase | Status |
|------|-----------------|-------|--------|
| `pyproject.toml` | C0 | P0 | ✅ Updated |
| `src/codeintel/serving/mcp/app.py` | C0, H1-H4, H7 | P0-P1 | ✅ Updated (PR1-3) |
| `src/codeintel/serving/mcp/server.py` | C0 | P0 | ✅ Updated (PR1) |
| `src/codeintel/serving/settings.py` | H1, H4, H7 | P1 | ✅ Updated (PR2-3) |
| `src/codeintel/serving/http/app.py` | C0 | P0 | ✅ Updated (PR1) |
| `src/codeintel/serving/db/pointer.py` | H3 | P1 | ✅ Updated (PR3) |
| `tests/serving/test_semantic_mcp_tools.py` | H1-H4, H7 | P1 | ✅ Updated (PR2-3) |

### Files To Be Modified (Remaining Items)

| File | Items Affecting | Phase |
|------|-----------------|-------|
| `src/codeintel/serving/mcp/app.py` | H5, H6, M3-M7 | P1-P2 |
| `src/codeintel/serving/mcp/server.py` | M4, M8 | P2 |
| `src/codeintel/serving/settings.py` | H6, M1, M2, M7 | P1-P2 |
| `src/codeintel/cli/handlers/ops.py` | M1 | P2 |
| `src/codeintel/serving/http/app.py` | H6, M2 | P1-P2 |

### Dependencies to Update

| Package | Version | Purpose | Items |
|---------|---------|---------|-------|
| `fastmcp` | `>=2.14.1,<3` | Canonical MCP framework | C0 |
| `anyio` | (existing) | Async thread offloading | H1, H4 |
| `uvloop` | (optional) | High-performance event loop | M1 |
| `httptools` | (optional) | Fast HTTP parser | M1 |

---

## Testing Strategy

### Unit Tests

| Test File | Coverage |
|-----------|----------|
| `tests/serving/mcp/test_tools.py` | Tool behavior, error handling, envelope |
| `tests/serving/mcp/test_resources.py` | Resource resolution, artifact storage |
| `tests/serving/mcp/test_models.py` | Response model validation |
| `tests/serving/mcp/test_runtime.py` | QueryLimiter behavior |
| `tests/serving/mcp/test_context.py` | Context propagation |

### Integration Tests

| Test File | Coverage |
|-----------|----------|
| `tests/serving/mcp/test_mcp_server.py` | End-to-end MCP server |
| `tests/serving/http/test_mcp_mount.py` | MCP mounted in FastAPI, path contract |
| `tests/serving/test_uvicorn_config.py` | Uvicorn settings application |
| `tests/serving/test_auth_enforcement.py` | Auth required for public bind |

### Performance Tests

| Test | Metric |
|------|--------|
| Concurrent tool invocation | Latency under load |
| QueryLimiter saturation | Queuing behavior |
| Large result handling | Memory usage |

---

## Rollout Plan

### Stage 1: Development Environment
- Implement Phase 0 (Critical)
- Implement Phase 1 (High Priority)
- Run full test suite
- Manual testing with Claude Desktop / Cursor

### Stage 2: Local Validation
- Implement Phase 2 (Medium Priority)
- Load testing with 3 concurrent clients
- Verify metrics emission
- Test auth enforcement

### Stage 3: Production Deployment
- Implement Phase 3 (Low Priority)
- Update documentation
- Monitor logs for 24 hours

### Rollback Plan
- Feature flags allow disabling new functionality
- `mcp_mask_errors=False` for debugging
- Previous settings remain functional

---

## Success Metrics

### Completed (PR1 + PR2 + PR3)

| Metric | Status | Notes |
|--------|--------|-------|
| gofastmcp 2.x migration | ✅ Done | All imports via `_compat.py` |
| All 6 tools use `async def` + `Context` | ✅ Done | Blocking ops offloaded via `anyio.to_thread.run_sync` |
| Tool annotations (`readOnlyHint`, etc.) | ✅ Done | All tools have consistent annotations |
| `ToolError` for user-friendly errors | ✅ Done | Invalid inputs return clean error messages |
| Response envelope with provenance | ✅ Done | All tools return `{meta: {...}, data: {...}}` |
| Query concurrency limiter | ✅ Done | Default limit: 2 concurrent heavy queries |
| Error masking in production | ✅ Done | Internal errors hidden, configurable via setting |

### Remaining (PR4-PR7)

| Metric | Target | Item |
|--------|--------|------|
| MCP resources for exports | Export artifacts via `codeintel://` URIs | H5 |
| SSE resumability | Long exports survivel network reconnects | H6 |
| Environment-driven Uvicorn config | All settings via env vars | M1 |
| Auth required for public bind | Auto-reject if no auth on 0.0.0.0 | M2 |
| MCP health endpoint | `/mcp/health` responds correctly | M3 |
| Bearer auth support | Token-based auth for remote clients | M4 |
| Tool tagging | Semantic grouping for discovery | M5 |
| Metrics emission | Structured logs for all tool calls | M6 |
| Feature flags | Enable/disable tools via env vars | M7 |
| Unified lifespan | DB manager shared between HTTP and MCP | M8 |

---

## Appendix: Environment Variables Reference

### New Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CODEINTEL_MCP_ENABLE_SAMPLING` | `0` | Enable LLM sampling in tools |
| `CODEINTEL_MCP_SAMPLE_THRESHOLD` | `500` | Row count threshold for sampling |
| `CODEINTEL_MCP_PROGRESS` | `1` | Enable progress reporting |
| `CODEINTEL_MCP_MASK_ERRORS` | `1` | Mask internal error details |
| `CODEINTEL_MCP_MAX_CONCURRENT_QUERIES` | `2` | Max concurrent heavy queries |
| `CODEINTEL_MCP_EVENT_STORE` | `1` | Enable SSE resumability |
| `CODEINTEL_MCP_RETRY_INTERVAL` | `1000` | SSE retry interval (ms) |
| `CODEINTEL_MCP_ENABLE_SEARCH` | `1` | Enable code_search tool |
| `CODEINTEL_MCP_ENABLE_EXPLAIN` | `1` | Enable semantic_explain tool |
| `CODEINTEL_MCP_ENABLE_META` | `1` | Enable serving_meta tool |
| `CODEINTEL_MCP_ENABLE_EXPORT` | `1` | Enable semantic_export tool |
| `CODEINTEL_AUTH_REQUIRED_FOR_REMOTE` | `1` | Require auth for non-localhost |
| `CODEINTEL_UVICORN_WORKERS` | `1` | Number of Uvicorn workers |
| `CODEINTEL_UVICORN_LOOP` | `auto` | Event loop implementation |
| `CODEINTEL_UVICORN_HTTP` | `auto` | HTTP parser implementation |
| `CODEINTEL_UVICORN_LIMIT_CONCURRENCY` | (none) | Max concurrent connections |
| `CODEINTEL_UVICORN_LIMIT_MAX_REQUESTS` | (none) | Requests before worker restart |
| `CODEINTEL_UVICORN_TIMEOUT_KEEP_ALIVE` | `30` | Keep-alive timeout seconds |
| `CODEINTEL_UVICORN_BACKLOG` | `2048` | Connection backlog size |
| `CODEINTEL_UVICORN_ACCESS_LOG` | `1` | Enable access logging |
| `CODEINTEL_UVICORN_SERVER_HEADER` | `0` | Include server header |
| `CODEINTEL_UVICORN_PROXY_HEADERS` | `0` | Trust proxy headers |
| `CODEINTEL_UVICORN_FORWARDED_ALLOW_IPS` | `127.0.0.1` | IPs allowed to set X-Forwarded-* |

---

## Appendix: MCP Client Configuration Examples

### ChatGPT Connector

```json
{
  "name": "CodeIntel",
  "url": "https://your-domain.com/mcp/",
  "auth": {
    "type": "bearer",
    "token": "${CODEINTEL_AUTH_TOKEN}"
  }
}
```

### Claude Desktop

```json
{
  "mcpServers": {
    "codeintel": {
      "command": "python",
      "args": ["-m", "codeintel.serving.mcp"],
      "env": {
        "CODEINTEL_SERVE_DIR": "/path/to/serve"
      }
    }
  }
}
```

### Cursor

```json
{
  "mcpServers": {
    "codeintel": {
      "url": "http://localhost:8000/mcp/"
    }
  }
}
```

---

## Appendix: Recommended Uvicorn Presets

| Use Case | Workers | Query Limit | Command |
|----------|---------|-------------|---------|
| **Personal (default)** | 1 | 2 | `codeintel serve http` |
| **Multi-client** | 2 | 2 | `CODEINTEL_UVICORN_WORKERS=2 codeintel serve http` |
| **Behind proxy** | 1 | 2 | `CODEINTEL_UVICORN_PROXY_HEADERS=1 codeintel serve http` |

**Important**: With `workers > 1`, DuckDB connections **must** be `read_only=True` (our serving layer handles this automatically).

---

*Document created: 2025-12-16*
*Last updated: 2025-12-16 (Post PR3 - Response Envelope & Query Limiter complete)*
*Status: PR1-PR3 complete (C0, H1-H4, H7). PR4-PR7 pending.*
