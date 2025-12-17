# FastMCP & Uvicorn Best-in-Class Implementation Plan

> **Purpose**: Transform the CodeIntel serving layer's FastMCP and Uvicorn implementation into a best-in-class solution with exceptional feature set, hardness, extensibility, maintainability, and full integration with the codebase.

> **Target Application**: Single-box, single-user personal application serving at most 3 external LLM consumers over MCP (Claude, ChatGPT, Cursor, etc.).

> **Scope**: 22 enhancement items organized by priority (Critical/High/Medium/Low) with detailed implementation specifications.

> **Revision Note**: This plan incorporates expert feedback from `FastMCP_implementation_comments.md` validating architectural decisions and adding critical items for runtime normalization, response envelopes, query limiting, and large-data handling.

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

### FastMCP Implementation

**Location**: `src/codeintel/serving/mcp/`

| Component | Status | Gaps |
|-----------|--------|------|
| `app.py` | ✅ Functional | **Uses MCP SDK FastMCP, not gofastmcp 2.x**; No Context usage, no annotations, sync-only |
| `server.py` | ✅ Functional | Basic lifespan, no auth integration |
| `__main__.py` | ✅ Functional | Minimal entry point |

**Critical Issue Identified**: Current code imports from `mcp.server.fastmcp` (MCP SDK flavor) while plan references gofastmcp 2.x features (tool annotations, `http_app()`, `EventStore`, `ResourceContent`). These are **incompatible ecosystems**.

**Current Tools** (6 total):
- `semantic_catalog` - List views
- `semantic_describe` - Describe view schema
- `semantic_query` - Query with filters
- `semantic_explain` - SQL + plan output
- `serving_meta` - Metadata endpoint
- `code_search` - FTS search

**Missing Features**:
- No MCP Context access (progress, logging, sampling)
- No MCP Annotations (readOnlyHint, idempotentHint)
- No MCP Resources for large data delivery
- No structured return types (Pydantic models)
- No response meta envelope (snapshot info, version, truncation)
- No error masking or ToolError usage
- No query concurrency limiter
- No EventStore for SSE resumability
- No tags for tool organization
- No health check on MCP endpoint

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

### Integration Points

| System | Integration Status |
|--------|-------------------|
| HTTP Routes | ✅ Full (metrics, correlation ID, RFC 9457) |
| MCP Tools | ⚠️ Basic (no metrics, no correlation ID, no response envelope) |
| Settings | ⚠️ Partial (missing Uvicorn settings, MCP settings) |
| Storage Gateway | ✅ Full (via SemanticQueryKernel) |
| Auth | ⚠️ Partial (HTTP has API key, MCP has auth_token but no enforcement) |

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

### C0: Normalize FastMCP Runtime to gofastmcp 2.x

**Category**: D - Architecture & Extensibility

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

---

## High-Priority Enhancements

### H1: MCP Context Access for Rich Tool Orchestration

**Category**: A - MCP Tool Enhancements

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

---

### H2: MCP Annotations for LLM Client Optimization

**Category**: A - MCP Tool Enhancements

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

---

### H3: Standard Response Meta Envelope

**Category**: A - MCP Tool Enhancements

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

---

### H4: Query Limiter for Concurrency Control

**Category**: B - Uvicorn & Deployment

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

---

### H5: MCP Resources for Large Dataset Delivery

**Category**: A - MCP Tool Enhancements

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

---

### H6: SSE Polling and EventStore for Remote Resumability

**Category**: B - Uvicorn & Deployment

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

---

### H7: Error Masking and ToolError for Security

**Category**: C - Observability & Security

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

---

## Medium-Priority Enhancements

### M1: Production Uvicorn Configuration

**Category**: B - Uvicorn & Deployment

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

---

### M2: Authentication Enforcement for Non-Localhost

**Category**: C - Observability & Security

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

### M3: Custom MCP Route for Health Checks

**Category**: B - Uvicorn & Deployment

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

---

### M4: Bearer Token Authentication for MCP

**Category**: C - Observability & Security

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

---

### M5: Tags for Tool Organization

**Category**: D - Architecture & Extensibility

**Problem**: All 6 tools are flat with no categorization, making it harder to manage as the tool set grows.

**Solution**: Add tags to tools for logical grouping and potential filtering.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/app.py` | Add tags to all tool decorators |

**Detailed Implementation**:

```python
# src/codeintel/serving/mcp/app.py

# Tag constants for consistency
TAG_SEMANTIC = "semantic"
TAG_SEARCH = "search"
TAG_META = "meta"
TAG_READ = "read"
TAG_EXPORT = "export"

@mcp.tool(
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
    tags=[TAG_SEMANTIC, TAG_READ],
)
async def semantic_catalog(*, ctx: Context) -> dict[str, object]:
    ...

@mcp.tool(
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
    tags=[TAG_SEMANTIC, TAG_READ],
)
async def semantic_query(..., *, ctx: Context) -> dict[str, object]:
    ...

@mcp.tool(
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
    tags=[TAG_SEARCH, TAG_READ],
)
async def code_search(..., *, ctx: Context) -> dict[str, object]:
    ...

@mcp.tool(
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
    tags=[TAG_SEMANTIC, TAG_EXPORT],
)
async def semantic_export(..., *, ctx: Context) -> dict[str, object]:
    ...
```

**Future Use**:
- `include_tags`/`exclude_tags` on FastMCP server for filtering
- Client-side tool discovery by tag
- Admin tools with `internal` tag that can be excluded

---

### M6: Metrics Emission from MCP Tools

**Category**: C - Observability & Security

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

**Testing Requirements**:
- Verify metrics are logged for MCP tool calls
- Test correlation ID from ctx.session_id
- Verify metrics format matches HTTP route metrics

---

### M7: Tool Enable/Disable for Feature Flags

**Category**: D - Architecture & Extensibility

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

    # MCP Tool Feature Flags
    mcp_enable_search: bool = True
    mcp_enable_explain: bool = True
    mcp_enable_meta: bool = True
    mcp_enable_export: bool = True

    @classmethod
    def from_env(cls) -> ServingSettings:
        return cls(
            # ... existing fields ...
            mcp_enable_search=os.environ.get("CODEINTEL_MCP_ENABLE_SEARCH", "1") == "1",
            mcp_enable_explain=os.environ.get("CODEINTEL_MCP_ENABLE_EXPLAIN", "1") == "1",
            mcp_enable_meta=os.environ.get("CODEINTEL_MCP_ENABLE_META", "1") == "1",
            mcp_enable_export=os.environ.get("CODEINTEL_MCP_ENABLE_EXPORT", "1") == "1",
        )
```

```python
# src/codeintel/serving/mcp/app.py

@mcp.tool(
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
    tags=[TAG_SEMANTIC, TAG_READ],
    enabled=settings.mcp_enable_explain,
)
async def semantic_explain(..., *, ctx: Context) -> dict[str, object]:
    ...

@mcp.tool(
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
    tags=[TAG_SEARCH, TAG_READ],
    enabled=settings.mcp_enable_search,
)
async def code_search(..., *, ctx: Context) -> dict[str, object]:
    ...

@mcp.tool(
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
    tags=[TAG_SEMANTIC, TAG_EXPORT],
    enabled=settings.mcp_enable_export,
)
async def semantic_export(..., *, ctx: Context) -> dict[str, object]:
    ...
```

**Testing Requirements**:
- Test disabled tool returns "Unknown tool" error
- Test enabled tools work normally
- Verify tool list excludes disabled tools

---

### M8: Unified Lifespan Management

**Category**: D - Architecture & Extensibility

**Problem**: Separate lifespan contexts for FastAPI and standalone MCP, causing code duplication.

**Solution**: Allow dependency injection of ServingDBManager to share lifecycle.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/server.py` | Accept optional db_manager injection |

**Detailed Implementation**:

```python
# src/codeintel/serving/mcp/server.py

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

    Returns
    -------
    FastMCP
        Configured MCP server.
    """
    cfg = settings or ServingSettings.from_env()

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

---

## Low-Priority Enhancements

### L1: Server Composition for Modularity

**Category**: D - Architecture & Extensibility

**Problem**: Single monolithic MCP app; as tool count grows, maintenance becomes harder.

**Solution**: Split tools into composable sub-servers using FastMCP's `mount()`.

**Files to Create/Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/servers/semantic.py` | New: Semantic tools sub-server |
| `src/codeintel/serving/mcp/servers/search.py` | New: Search tools sub-server |
| `src/codeintel/serving/mcp/servers/meta.py` | New: Meta tools sub-server |
| `src/codeintel/serving/mcp/app.py` | Compose sub-servers |

**Implementation Sketch**:

```python
# src/codeintel/serving/mcp/servers/semantic.py

from codeintel.serving.mcp._compat import FastMCP

def build_semantic_server(kernel: SemanticKernel, settings: ServingSettings) -> FastMCP:
    """Build semantic-focused MCP sub-server."""
    mcp = FastMCP("CodeIntel-Semantic")

    @mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS)
    async def catalog(*, ctx: Context) -> dict[str, object]:
        ...

    @mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS)
    async def query(..., *, ctx: Context) -> dict[str, object]:
        ...

    return mcp
```

```python
# src/codeintel/serving/mcp/app.py

def build_mcp_app(...) -> FastMCP:
    main = FastMCP("CodeIntel", ...)

    # Compose sub-servers
    semantic = build_semantic_server(kernel, settings)
    search = build_search_server(kernel, settings)
    meta = build_meta_server(kernel, settings)

    main.mount(semantic, prefix="semantic")
    main.mount(search, prefix="search")
    main.mount(meta, prefix="meta")

    return main
```

**Note**: This is a structural refactor that doesn't change functionality. Consider implementing after M1-M8 are stable.

---

### L2: MCP Prompts for Guided Interactions

**Category**: A - MCP Tool Enhancements

**Problem**: No guided prompts for common workflows; LLMs must discover tool usage patterns.

**Solution**: Add FastMCP prompts for common interaction patterns.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/prompts.py` | New file with prompt definitions |
| `src/codeintel/serving/mcp/app.py` | Register prompts |

(Implementation details same as original plan)

---

### L3: Correlation ID Propagation to MCP

**Category**: C - Observability & Security

**Problem**: Correlation IDs are HTTP-middleware only; MCP tools use session_id but not integrated.

**Solution**: Use contextvars to propagate correlation ID across tool execution.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/context.py` | New file for MCP context management |
| `src/codeintel/serving/mcp/app.py` | Set correlation_id from ctx.session_id |

(Implementation details same as original plan)

---

### L4: OpenAPI/FastAPI Integration Documentation

**Category**: D - Architecture & Extensibility

**Problem**: HTTP routes and MCP tools are defined separately; potential for drift.

**Solution**: Document pattern for `FastMCP.from_fastapi()` integration.

(Documentation item, same as original plan)

---

### L5: Proxy Pattern for Remote Services

**Category**: D - Architecture & Extensibility

**Problem**: No pattern for integrating external MCP services.

**Solution**: Document proxy pattern for future extensibility.

(Documentation item, same as original plan)

---

### L6: Mount Path Contract Test

**Category**: B - Uvicorn & Deployment

**Problem**: No test validates the mount path contract (effective endpoint is `/mcp`, not `/mcp/mcp`).

**Solution**: Add explicit test for mount path behavior.

**Files to Create**:

| File | Changes |
|------|---------|
| `tests/serving/http/test_mcp_mount.py` | New: Mount path contract tests |

**Implementation**:

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

---

## Implementation Phases

### Phase 0: Critical Foundation
**Duration**: 1 day

| Day | Items | Focus |
|-----|-------|-------|
| 1 | C0 | Normalize to gofastmcp 2.x |

**Gate**: All imports use gofastmcp, MCP tools functional

### Phase 1: Core Tool Enhancements (High Priority)
**Duration**: 4-5 days

| Day | Items | Focus |
|-----|-------|-------|
| 1 | H1, H2 | Context API (keyword-only) + Annotations |
| 2 | H3 | Response meta envelope |
| 3 | H4 | Query limiter |
| 4 | H5, H6 | Resources + EventStore |
| 5 | H7 | Error handling |

**Gate**: All high-priority items complete, tests passing

### Phase 2: Production Hardening (Medium Priority)
**Duration**: 3-4 days

| Day | Items | Focus |
|-----|-------|-------|
| 1 | M1, M2 | Uvicorn config + Auth enforcement |
| 2 | M3, M4 | Health checks + Bearer auth |
| 3 | M5, M6, M7 | Tags + Metrics + Feature flags |
| 4 | M8 | Unified lifespan |

**Gate**: All medium-priority items complete, integration tests passing

### Phase 3: Extensibility (Low Priority)
**Duration**: 2-3 days

| Day | Items | Focus |
|-----|-------|-------|
| 1 | L1 | Server composition |
| 2 | L2, L3 | Prompts + Correlation ID |
| 3 | L4, L5, L6 | Documentation + Mount test |

**Gate**: All items complete, documentation updated

---

## File Change Matrix

### New Files

| File | Purpose | Phase |
|------|---------|-------|
| `src/codeintel/serving/mcp/_compat.py` | FastMCP import shim | P0 |
| `src/codeintel/serving/mcp/models.py` | Pydantic response models + envelope | P1 |
| `src/codeintel/serving/mcp/response.py` | Envelope builder helper | P1 |
| `src/codeintel/serving/mcp/runtime.py` | QueryLimiter | P1 |
| `src/codeintel/serving/mcp/resource_store.py` | Export artifact storage | P1 |
| `src/codeintel/serving/mcp/resources.py` | MCP resource handlers | P1 |
| `src/codeintel/serving/mcp/prompts.py` | MCP prompt templates | P3 |
| `src/codeintel/serving/mcp/context.py` | Correlation ID context | P3 |
| `src/codeintel/serving/mcp/servers/` | Sub-server modules | P3 |
| `tests/serving/mcp/test_tools.py` | MCP tool tests | P1 |
| `tests/serving/mcp/test_resources.py` | MCP resource tests | P1 |
| `tests/serving/http/test_mcp_mount.py` | Mount path contract test | P3 |

### Modified Files

| File | Items Affecting | Phase |
|------|-----------------|-------|
| `pyproject.toml` | C0 | P0 |
| `src/codeintel/serving/mcp/app.py` | C0, H1-H7, M3-M7 | P0-P2 |
| `src/codeintel/serving/mcp/server.py` | C0, M4, M8 | P0, P2 |
| `src/codeintel/serving/settings.py` | H1, H4, H6, H7, M1, M2, M7 | P1-P2 |
| `src/codeintel/cli/handlers/ops.py` | M1 | P2 |
| `src/codeintel/serving/http/app.py` | C0, H6, M2 | P0, P1, P2 |

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
*Last updated: 2025-12-16 (integrated feedback from FastMCP_implementation_comments.md)*
*Status: Ready for implementation*
