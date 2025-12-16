# FastMCP & Uvicorn Best-in-Class Implementation Plan

> **Purpose**: Transform the CodeIntel serving layer's FastMCP and Uvicorn implementation into a best-in-class solution with exceptional feature set, hardness, extensibility, maintainability, and full integration with the codebase.

> **Target Application**: Single-box, single-user personal application serving at most 3 external LLM consumers over MCP (Claude, ChatGPT, Cursor, etc.).

> **Scope**: 20 enhancement items organized by priority (High/Medium/Low) with detailed implementation specifications.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Enhancement Categories](#enhancement-categories)
4. [High-Priority Enhancements](#high-priority-enhancements)
5. [Medium-Priority Enhancements](#medium-priority-enhancements)
6. [Low-Priority Enhancements](#low-priority-enhancements)
7. [Implementation Phases](#implementation-phases)
8. [File Change Matrix](#file-change-matrix)
9. [Testing Strategy](#testing-strategy)
10. [Rollout Plan](#rollout-plan)

---

## Executive Summary

### Goals

1. **Rich LLM Orchestration**: Leverage FastMCP's Context API for progress reporting, structured logging, and LLM-assisted data summarization
2. **Optimal Client Integration**: Add MCP annotations so Claude/ChatGPT can skip confirmation prompts for read-only operations
3. **Production Hardening**: Configure Uvicorn for performance, security, and reliability
4. **Observability Parity**: Ensure MCP tools have the same metrics/logging as HTTP routes
5. **Extensibility**: Structure the MCP layer for future growth with composition and modularity patterns

### Impact Summary

| Priority | Count | Key Benefits |
|----------|-------|--------------|
| High | 5 | Core UX improvements, security, responsiveness |
| Medium | 9 | Production readiness, data handling, observability |
| Low | 6 | Modularity, advanced patterns, future-proofing |

### Estimated Effort

- **High Priority**: 3-4 days
- **Medium Priority**: 4-5 days
- **Low Priority**: 2-3 days
- **Total**: ~10-12 days

---

## Current State Analysis

### FastMCP Implementation

**Location**: `src/codeintel/serving/mcp/`

| Component | Status | Gaps |
|-----------|--------|------|
| `app.py` | ✅ Functional | No Context usage, no annotations, sync-only |
| `server.py` | ✅ Functional | Basic lifespan, no auth integration |
| `__main__.py` | ✅ Functional | Minimal entry point |

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
- No error masking or ToolError usage
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

### Integration Points

| System | Integration Status |
|--------|-------------------|
| HTTP Routes | ✅ Full (metrics, correlation ID, RFC 9457) |
| MCP Tools | ⚠️ Basic (no metrics, no correlation ID) |
| Settings | ⚠️ Partial (missing Uvicorn settings) |
| Storage Gateway | ✅ Full (via SemanticQueryKernel) |

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

## High-Priority Enhancements

### H1: MCP Context Access for Rich Tool Orchestration

**Category**: A - MCP Tool Enhancements

**Problem**: Tools are synchronous functions with no ability to report progress, log to clients, or leverage LLM sampling for large result summarization.

**Solution**: Add `fastmcp.Context` parameter to all tools, enabling:
- Progress reporting via `ctx.report_progress()`
- Client-visible logging via `ctx.info()`, `ctx.warning()`, `ctx.debug()`
- Resource reading via `ctx.read_resource()`
- LLM sampling via `ctx.sample()` for data summarization

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/app.py` | Add Context param to all 6 tools, convert to async |
| `src/codeintel/serving/settings.py` | Add `mcp_enable_sampling`, `mcp_sample_threshold` settings |

**Detailed Implementation**:

```python
# src/codeintel/serving/mcp/app.py

from fastmcp import Context

@mcp.tool()
async def semantic_query(
    view_id: str,
    filters: list[dict[str, object]] | None = None,
    select: list[str] | None = None,
    order_by: list[str] | None = None,
    pagination: dict[str, int] | None = None,
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
    if len(result.rows) > 500 and settings.mcp_enable_sampling:
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
async def semantic_catalog(ctx: Context) -> dict[str, object]:
    ...

@mcp.tool(
    name="semantic_describe",
    description="Describe a semantic view's schema and metadata",
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
)
async def semantic_describe(view_id: str, ctx: Context) -> dict[str, object]:
    ...

@mcp.tool(
    name="semantic_query",
    description="Query a semantic view with structured filters",
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
)
async def semantic_query(..., ctx: Context) -> dict[str, object]:
    ...

@mcp.tool(
    name="semantic_explain",
    description="Return compiled SQL and DuckDB plan for a semantic query",
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
)
async def semantic_explain(..., ctx: Context) -> dict[str, object]:
    ...

@mcp.tool(
    name="serving_meta",
    description="Get serving layer metadata including snapshot info",
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
)
async def serving_meta(ctx: Context) -> dict[str, object]:
    ...

@mcp.tool(
    name="code_search",
    description="Search code metadata using BM25 full-text search",
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
)
async def code_search(..., ctx: Context) -> dict[str, object]:
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

### H3: Async-First Tool Implementation

**Category**: A - MCP Tool Enhancements

**Problem**: Current tools are synchronous, blocking the event loop during CPU-bound DuckDB queries. This reduces responsiveness when serving multiple LLM clients.

**Solution**: Convert all tools to `async def` with explicit thread offloading via `anyio.to_thread.run_sync()` for kernel methods.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/app.py` | Convert all 6 tools to async, add anyio offloading |

**Detailed Implementation**:

```python
# src/codeintel/serving/mcp/app.py

import anyio

@mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS)
async def semantic_catalog(ctx: Context) -> dict[str, object]:
    """List available semantic views in the CodeIntel database."""
    await ctx.info("Fetching semantic view catalog")
    result = await anyio.to_thread.run_sync(kernel.catalog)
    return result

@mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS)
async def semantic_describe(view_id: str, ctx: Context) -> dict[str, object]:
    """Describe a semantic view's schema and metadata."""
    await ctx.info(f"Describing view: {view_id}")
    result = await anyio.to_thread.run_sync(kernel.describe, view_id)
    return result

@mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS)
async def semantic_query(
    view_id: str,
    filters: list[dict[str, object]] | None = None,
    select: list[str] | None = None,
    order_by: list[str] | None = None,
    pagination: dict[str, int] | None = None,
    ctx: Context,
) -> dict[str, object]:
    """Query a semantic view with structured filters."""
    await ctx.info(f"Querying view: {view_id}")
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
    await ctx.report_progress(100, 100)

    return result.model_dump(mode="json")

@mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS)
async def semantic_explain(
    view_id: str,
    filters: list[dict[str, object]] | None = None,
    select: list[str] | None = None,
    order_by: list[str] | None = None,
    pagination: dict[str, int] | None = None,
    ctx: Context,
) -> dict[str, object]:
    """Return compiled SQL and DuckDB plan for a semantic query."""
    await ctx.info(f"Explaining query for view: {view_id}")

    page = pagination or {}
    request = SemanticQueryRequest(
        view_id=view_id,
        select=select,
        filters=[FilterSpec.model_validate(f) for f in (filters or [])],
        order_by=order_by or [],
        limit=page.get("limit", 200),
        offset=page.get("offset", 0),
    )

    result = await anyio.to_thread.run_sync(kernel.explain, request)
    return result.model_dump(mode="json")

@mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS)
async def serving_meta(ctx: Context) -> dict[str, object]:
    """Get serving layer metadata."""
    await ctx.info("Fetching serving metadata")
    return await anyio.to_thread.run_sync(kernel.meta)

@mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS)
async def code_search(
    query: str,
    kinds: list[str] | None = None,
    limit: int = 20,
    offset: int = 0,
    ctx: Context,
) -> dict[str, object]:
    """Search code metadata using the serving snapshot search index."""
    await ctx.info(f"Searching: {query}")

    request = SearchQueryRequest(
        query=query,
        kinds=kinds,
        limit=limit,
        offset=offset,
    )

    result = await anyio.to_thread.run_sync(kernel.search, request)
    return result.model_dump(mode="json")
```

**Dependencies**:
- `anyio` (already in project dependencies)

**Testing Requirements**:
- Verify tools remain functional after async conversion
- Test concurrent tool invocation
- Measure latency improvement under load

---

### H4: Error Masking and ToolError for Security

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

from fastmcp.exceptions import ToolError

def build_mcp_app(
    *,
    kernel: SemanticKernel,
    settings: ServingSettings,
    host: str = "127.0.0.1",
    port: int = 8000,
    streamable_http_path: str = "/mcp",
    lifespan: Callable[[FastMCP], AbstractAsyncContextManager[object]] | None = None,
) -> FastMCP:
    """Build FastMCP application with semantic tools."""
    mcp = FastMCP(
        "CodeIntel",
        json_response=True,
        host=host,
        port=port,
        streamable_http_path=streamable_http_path,
        lifespan=lifespan,
        mask_error_details=settings.mcp_mask_errors,  # Hide internal traces
    )

    @mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS)
    async def semantic_query(..., ctx: Context) -> dict[str, object]:
        """Query a semantic view with structured filters."""
        try:
            await ctx.info(f"Querying view: {view_id}")
            # ... query logic ...
            result = await anyio.to_thread.run_sync(kernel.query, request)
            return result.model_dump(mode="json")
        except KeyError as e:
            # User-friendly error (passes through masking)
            raise ToolError(f"View '{view_id}' not found in semantic registry") from e
        except ValueError as e:
            raise ToolError(f"Invalid query parameters: {e}") from e
        except Exception as e:
            # Log internally, return generic error
            await ctx.error(f"Query failed: {type(e).__name__}")
            raise ToolError("Query execution failed. Check server logs for details.") from e

    @mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS)
    async def code_search(..., ctx: Context) -> dict[str, object]:
        """Search code metadata."""
        try:
            # ... search logic ...
        except ValueError as e:
            raise ToolError(f"Invalid search query: {e}") from e
        except Exception as e:
            await ctx.error(f"Search failed: {type(e).__name__}")
            raise ToolError("Search execution failed. Check server logs for details.") from e

    return mcp
```

**Settings Addition**:

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

### H5: Production Uvicorn Configuration

**Category**: B - Uvicorn & Deployment

**Problem**: Current Uvicorn configuration uses defaults, missing performance optimizations and resource protection.

**Solution**: Add comprehensive Uvicorn configuration with:
- Optional multi-worker support
- uvloop/httptools for performance
- Concurrency and request limits
- Keep-alive tuning
- Security headers

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

    if reload:
        uvicorn.run(
            "codeintel.serving.http.app:create_serving_app",
            factory=True,
            reload=True,
            **uvicorn_config,
        )
    elif workers > 1:
        # Multi-worker mode requires string import path
        uvicorn.run(
            "codeintel.serving.http.app:create_serving_app",
            factory=True,
            workers=workers,
            **uvicorn_config,
        )
    else:
        # Single worker - can use app instance directly
        app = create_serving_app(settings)
        uvicorn.run(app, **uvicorn_config)

    return CliResult.ok(
        ServeStartResult(
            server_type="http",
            host=host,
            port=port,
            auto_pipeline=False,
            repo=pointer.repo,
            commit=pointer.commit,
            db_path=str(pointer.db_path),
        )
    )
```

**Optional Dependencies**:
- `uvloop` - High-performance event loop (Linux/macOS)
- `httptools` - Fast HTTP parser

**Testing Requirements**:
- Test single-worker mode
- Test multi-worker mode (if workers > 1)
- Verify settings are applied correctly
- Load test with concurrency limits

---

## Medium-Priority Enhancements

### M1: MCP Resources for Large Dataset Delivery

**Category**: A - MCP Tool Enhancements

**Problem**: All data is delivered through tool return values, which can overwhelm LLM context windows for large datasets.

**Solution**: Expose semantic view data and schemas as MCP resources that LLMs can fetch on-demand.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/app.py` | Add resource decorators |
| `src/codeintel/serving/mcp/resources.py` | New file for resource handlers |

**Detailed Implementation**:

```python
# src/codeintel/serving/mcp/resources.py

"""MCP resource handlers for on-demand data access."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.serving.mcp.app import SemanticKernel


def register_resources(mcp: FastMCP, kernel: SemanticKernel) -> None:
    """Register MCP resources on the server.

    Parameters
    ----------
    mcp
        FastMCP server instance.
    kernel
        Semantic query kernel.
    """

    @mcp.resource("data://catalog")
    def catalog_resource() -> dict[str, object]:
        """Full semantic view catalog as a resource."""
        return kernel.catalog()

    @mcp.resource("data://view/{view_id}")
    def view_resource(view_id: str) -> dict[str, object]:
        """Semantic view description as a resource."""
        return kernel.describe(view_id)

    @mcp.resource("data://view/{view_id}/schema")
    def view_schema_resource(view_id: str) -> dict[str, object]:
        """View schema only (for LLM context efficiency)."""
        full = kernel.describe(view_id)
        return {
            "id": full["id"],
            "columns": full["columns"],
            "column_types": full["column_types"],
            "primary_key": full["primary_key"],
        }

    @mcp.resource("data://meta")
    def meta_resource() -> dict[str, object]:
        """Serving metadata as a resource."""
        return kernel.meta()

    @mcp.resource("resource://buildspec")
    def buildspec_resource() -> dict[str, object]:
        """BuildSpec summary for agent context."""
        meta = kernel.meta()
        return {
            "buildspec_hash": meta.get("buildspec_hash"),
            "buildspec_version": meta.get("buildspec_version"),
            "targets": meta.get("targets"),
            "datasets": meta.get("datasets"),
        }
```

```python
# src/codeintel/serving/mcp/app.py

from codeintel.serving.mcp.resources import register_resources

def build_mcp_app(...) -> FastMCP:
    mcp = FastMCP(...)

    # Register tools (existing)
    # ...

    # Register resources (new)
    register_resources(mcp, kernel)

    return mcp
```

**Testing Requirements**:
- Test resource URI resolution
- Test parameterized resources (view_id)
- Verify JSON serialization of resources

---

### M2: Custom MCP Route for Health Checks

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

### M3: ASGI App Extraction for Flexible Mounting

**Category**: B - Uvicorn & Deployment

**Problem**: MCP is mounted using `mcp.streamable_http_app()` which has limited middleware control.

**Solution**: Use `mcp.http_app()` for full ASGI control with custom middleware support.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/http/app.py` | Update `_maybe_mount_mcp` to use `http_app()` |

**Detailed Implementation**:

```python
# src/codeintel/serving/http/app.py

from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

def _maybe_mount_mcp(
    app: FastAPI,
    *,
    kernel: SemanticQueryKernel,
    settings: ServingSettings,
    enabled: bool,
) -> None:
    """Mount MCP server under /mcp with appropriate middleware."""
    if not enabled:
        return

    mcp = build_mcp_app(
        kernel=kernel,
        settings=settings,
        streamable_http_path="/",
    )

    # Build middleware stack for MCP
    mcp_middleware: list[Middleware] = []

    # Add CORS if configured (with MCP-specific headers)
    if settings.cors_origins:
        mcp_middleware.append(
            Middleware(
                CORSMiddleware,
                allow_origins=list(settings.cors_origins),
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=[
                    "*",
                    "mcp-protocol-version",
                    "mcp-session-id",
                ],
                expose_headers=[
                    "mcp-session-id",
                    "X-Correlation-ID",
                ],
            )
        )

    # Get full ASGI app with middleware
    mcp_asgi = mcp.http_app(path="/", middleware=mcp_middleware)
    app.mount("/mcp", mcp_asgi)
```

**Testing Requirements**:
- Verify MCP endpoints remain functional after change
- Test CORS headers on MCP responses
- Verify middleware is applied

---

### M4: Structured Output with Return Type Annotations

**Category**: A - MCP Tool Enhancements

**Problem**: Tools return `dict[str, object]` which loses schema information for LLM clients.

**Solution**: Use Pydantic models as return types to generate JSON schemas.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/models.py` | New file with MCP response models |
| `src/codeintel/serving/mcp/app.py` | Update return type annotations |

**Detailed Implementation**:

```python
# src/codeintel/serving/mcp/models.py

"""Pydantic models for MCP tool responses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SnapshotInfo(BaseModel):
    """Snapshot identification."""

    model_config = ConfigDict(extra="forbid")

    repo: str
    commit: str
    run_id: str


class ViewSummary(BaseModel):
    """Compact view info for catalog listing."""

    model_config = ConfigDict(extra="forbid")

    id: str
    table_key: str
    entity: str
    grain: str
    description: str
    column_count: int


class SemanticCatalogResult(BaseModel):
    """Response from semantic_catalog tool."""

    model_config = ConfigDict(extra="forbid")

    version: str
    snapshot: SnapshotInfo
    views: list[ViewSummary]


class SemanticQueryResult(BaseModel):
    """Response from semantic_query tool."""

    model_config = ConfigDict(extra="forbid")

    view_id: str
    columns: list[str]
    rows: list[dict[str, object]]
    truncated: bool
    snapshot: SnapshotInfo
    llm_summary: str | None = None


class SemanticExplainResult(BaseModel):
    """Response from semantic_explain tool."""

    model_config = ConfigDict(extra="forbid")

    view_id: str
    sql: str
    plan: str
    snapshot: SnapshotInfo


class SearchResultItem(BaseModel):
    """Single search result."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    name: str
    module: str | None
    rel_path: str | None
    ref_goid_h128: str | None
    score: float | None


class CodeSearchResult(BaseModel):
    """Response from code_search tool."""

    model_config = ConfigDict(extra="forbid")

    query: str
    results: list[SearchResultItem]
    truncated: bool
    snapshot: SnapshotInfo
    engine: str


__all__ = [
    "CodeSearchResult",
    "SearchResultItem",
    "SemanticCatalogResult",
    "SemanticExplainResult",
    "SemanticQueryResult",
    "SnapshotInfo",
    "ViewSummary",
]
```

```python
# src/codeintel/serving/mcp/app.py

from codeintel.serving.mcp.models import (
    SemanticCatalogResult,
    SemanticQueryResult,
    SemanticExplainResult,
    CodeSearchResult,
)

@mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS)
async def semantic_catalog(ctx: Context) -> SemanticCatalogResult:
    """List available semantic views."""
    result = await anyio.to_thread.run_sync(kernel.catalog)
    return SemanticCatalogResult.model_validate(result)

@mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS)
async def semantic_query(..., ctx: Context) -> SemanticQueryResult:
    """Query a semantic view."""
    # ... query logic ...
    return SemanticQueryResult.model_validate(result.model_dump())
```

**Testing Requirements**:
- Verify JSON schema is generated in tool definitions
- Test model validation on tool responses
- Verify backwards compatibility with existing clients

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

@mcp.tool(
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
    tags=[TAG_SEMANTIC, TAG_READ],
)
async def semantic_catalog(ctx: Context) -> SemanticCatalogResult:
    ...

@mcp.tool(
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
    tags=[TAG_SEMANTIC, TAG_READ],
)
async def semantic_describe(view_id: str, ctx: Context) -> dict[str, object]:
    ...

@mcp.tool(
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
    tags=[TAG_SEMANTIC, TAG_READ],
)
async def semantic_query(..., ctx: Context) -> SemanticQueryResult:
    ...

@mcp.tool(
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
    tags=[TAG_SEMANTIC, TAG_READ],
)
async def semantic_explain(..., ctx: Context) -> SemanticExplainResult:
    ...

@mcp.tool(
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
    tags=[TAG_META, TAG_READ],
)
async def serving_meta(ctx: Context) -> dict[str, object]:
    ...

@mcp.tool(
    annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
    tags=[TAG_SEARCH, TAG_READ],
)
async def code_search(..., ctx: Context) -> CodeSearchResult:
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
| `src/codeintel/serving/http/metrics.py` | Add MCP-specific endpoint names |

**Detailed Implementation**:

```python
# src/codeintel/serving/mcp/app.py

import time
from codeintel.serving.http.metrics import QueryMetrics, log_query_metrics

@mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS, tags=[TAG_SEMANTIC, TAG_READ])
async def semantic_query(..., ctx: Context) -> SemanticQueryResult:
    """Query a semantic view with structured filters."""
    start = time.perf_counter()
    result: SemanticQueryResult | None = None

    try:
        await ctx.info(f"Querying view: {view_id}")
        await ctx.report_progress(10, 100)

        # ... build request ...

        result_raw = await anyio.to_thread.run_sync(kernel.query, request)
        result = SemanticQueryResult.model_validate(result_raw.model_dump())

        await ctx.report_progress(100, 100)
        return result

    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        log_query_metrics(QueryMetrics(
            endpoint="mcp:semantic_query",
            view_id=view_id,
            query=None,
            row_count=len(result.rows) if result else 0,
            truncated=result.truncated if result else False,
            duration_ms=duration_ms,
            correlation_id=ctx.session_id or "mcp-unknown",
        ))


@mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS, tags=[TAG_SEARCH, TAG_READ])
async def code_search(..., ctx: Context) -> CodeSearchResult:
    """Search code metadata."""
    start = time.perf_counter()
    result: CodeSearchResult | None = None

    try:
        await ctx.info(f"Searching: {query}")
        # ... search logic ...
        result = CodeSearchResult.model_validate(result_raw.model_dump())
        return result

    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        log_query_metrics(QueryMetrics(
            endpoint="mcp:code_search",
            view_id=None,
            query=query,
            row_count=len(result.results) if result else 0,
            truncated=result.truncated if result else False,
            duration_ms=duration_ms,
            correlation_id=ctx.session_id or "mcp-unknown",
            engine=result.engine if result else None,
        ))
```

**Testing Requirements**:
- Verify metrics are logged for MCP tool calls
- Test correlation ID from ctx.session_id
- Verify metrics format matches HTTP route metrics

---

### M7: Bearer Token Authentication for MCP

**Category**: C - Observability & Security

**Problem**: HTTP routes have API key protection, but MCP server has no authentication.

**Solution**: Enable FastMCP's `auth_token` configuration using the existing `auth_token` setting.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/app.py` | Add auth_token to FastMCP init |
| `src/codeintel/serving/mcp/server.py` | Pass auth_token from settings |

**Detailed Implementation**:

```python
# src/codeintel/serving/mcp/app.py

def build_mcp_app(
    *,
    kernel: SemanticKernel,
    settings: ServingSettings,
    host: str = "127.0.0.1",
    port: int = 8000,
    streamable_http_path: str = "/mcp",
    lifespan: Callable[[FastMCP], AbstractAsyncContextManager[object]] | None = None,
) -> FastMCP:
    """Build FastMCP application with semantic tools."""
    mcp = FastMCP(
        "CodeIntel",
        json_response=True,
        host=host,
        port=port,
        streamable_http_path=streamable_http_path,
        lifespan=lifespan,
        mask_error_details=settings.mcp_mask_errors,
        auth_token=settings.auth_token,  # Use existing setting for MCP auth
    )
    # ... rest of function ...
```

```python
# src/codeintel/serving/mcp/server.py

def create_mcp_server(settings: ServingSettings | None = None) -> FastMCP:
    cfg = settings or ServingSettings.from_env()
    # ... existing setup ...

    return build_mcp_app(
        kernel=kernel,
        settings=cfg,  # Pass full settings (includes auth_token)
        host=cfg.host,
        port=cfg.port,
        streamable_http_path="/mcp",
        lifespan=lifespan,
    )
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

### M8: Tool Enable/Disable for Feature Flags

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

    @classmethod
    def from_env(cls) -> ServingSettings:
        return cls(
            # ... existing fields ...
            mcp_enable_search=os.environ.get("CODEINTEL_MCP_ENABLE_SEARCH", "1") == "1",
            mcp_enable_explain=os.environ.get("CODEINTEL_MCP_ENABLE_EXPLAIN", "1") == "1",
            mcp_enable_meta=os.environ.get("CODEINTEL_MCP_ENABLE_META", "1") == "1",
        )
```

```python
# src/codeintel/serving/mcp/app.py

def build_mcp_app(*, kernel: SemanticKernel, settings: ServingSettings, ...) -> FastMCP:
    mcp = FastMCP(...)

    # Always-enabled core tools
    @mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS, tags=[TAG_SEMANTIC, TAG_READ])
    async def semantic_catalog(ctx: Context) -> SemanticCatalogResult:
        ...

    @mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS, tags=[TAG_SEMANTIC, TAG_READ])
    async def semantic_query(...) -> SemanticQueryResult:
        ...

    # Conditionally-enabled tools
    @mcp.tool(
        annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
        tags=[TAG_SEMANTIC, TAG_READ],
        enabled=settings.mcp_enable_explain,
    )
    async def semantic_explain(...) -> SemanticExplainResult:
        ...

    @mcp.tool(
        annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
        tags=[TAG_META, TAG_READ],
        enabled=settings.mcp_enable_meta,
    )
    async def serving_meta(ctx: Context) -> dict[str, object]:
        ...

    @mcp.tool(
        annotations=_READ_ONLY_LOCAL_ANNOTATIONS,
        tags=[TAG_SEARCH, TAG_READ],
        enabled=settings.mcp_enable_search,
    )
    async def code_search(...) -> CodeSearchResult:
        ...

    return mcp
```

**Testing Requirements**:
- Test disabled tool returns "Unknown tool" error
- Test enabled tools work normally
- Verify tool list excludes disabled tools

---

### M9: Unified Lifespan Management

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
        host=cfg.host,
        port=cfg.port,
        streamable_http_path="/mcp",
        lifespan=lifespan,
    )
```

**Use Case**: When mounting MCP within FastAPI app, the FastAPI lifespan manages the db_manager, so we can inject it:

```python
# Potential future use in http/app.py
def _maybe_mount_mcp(...) -> None:
    if not enabled:
        return
    # Share the db_manager with MCP
    mcp = create_mcp_server(settings=settings, db_manager=state.db)
    app.mount("/mcp", mcp.http_app())
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

from fastmcp import FastMCP

def build_semantic_server(kernel: SemanticKernel, settings: ServingSettings) -> FastMCP:
    """Build semantic-focused MCP sub-server."""
    mcp = FastMCP("CodeIntel-Semantic")

    @mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS)
    async def catalog(ctx: Context) -> SemanticCatalogResult:
        ...

    @mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS)
    async def describe(view_id: str, ctx: Context) -> dict[str, object]:
        ...

    @mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS)
    async def query(..., ctx: Context) -> SemanticQueryResult:
        ...

    @mcp.tool(annotations=_READ_ONLY_LOCAL_ANNOTATIONS, enabled=settings.mcp_enable_explain)
    async def explain(..., ctx: Context) -> SemanticExplainResult:
        ...

    return mcp
```

```python
# src/codeintel/serving/mcp/app.py

from codeintel.serving.mcp.servers.semantic import build_semantic_server
from codeintel.serving.mcp.servers.search import build_search_server
from codeintel.serving.mcp.servers.meta import build_meta_server

def build_mcp_app(...) -> FastMCP:
    main = FastMCP("CodeIntel", ...)

    # Compose sub-servers
    semantic = build_semantic_server(kernel, settings)
    search = build_search_server(kernel, settings)
    meta = build_meta_server(kernel, settings)

    main.mount(semantic, prefix="semantic")
    main.mount(search, prefix="search")
    main.mount(meta, prefix="meta")

    # Register resources on main
    register_resources(main, kernel)

    return main
```

**Note**: This is a structural refactor that doesn't change functionality. Consider implementing after M1-M9 are stable.

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

**Implementation**:

```python
# src/codeintel/serving/mcp/prompts.py

"""MCP prompts for guided interactions."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP


def register_prompts(mcp: FastMCP) -> None:
    """Register MCP prompts on the server.

    Parameters
    ----------
    mcp
        FastMCP server instance.
    """

    @mcp.prompt()
    def explore_codebase() -> str:
        """Guide the user through codebase exploration."""
        return """
To explore the CodeIntel codebase:

1. **Discover available views**: Call `semantic_catalog()` to list all semantic views
2. **Understand a view**: Call `semantic_describe(view_id)` to see columns and schema
3. **Query data**: Call `semantic_query(view_id, filters=[...])` to retrieve rows
4. **Search symbols**: Call `code_search(query)` to find functions, classes, modules

Example workflow:
```
# Step 1: See what's available
catalog = semantic_catalog()

# Step 2: Understand function metrics
schema = semantic_describe("function_metrics")

# Step 3: Find complex functions
results = semantic_query(
    view_id="function_metrics",
    filters=[{"column": "cyclomatic_complexity", "op": "gt", "value": 10}],
    order_by=["-cyclomatic_complexity"],
    pagination={"limit": 20}
)
```
        """

    @mcp.prompt()
    def analyze_function(goid: str) -> list[dict[str, str]]:
        """Guide analysis of a specific function by GOID."""
        return [
            {
                "role": "user",
                "content": f"Analyze function with GOID: {goid}",
            },
            {
                "role": "assistant",
                "content": f"""I'll analyze the function with GOID `{goid}`.

First, let me search for it and gather metrics...

1. I'll use `code_search` to find the function
2. I'll use `semantic_query` on `function_metrics` to get complexity data
3. I'll check `coverage_edges` for test coverage information

Let me start by searching for this function.""",
            },
        ]

    @mcp.prompt()
    def find_risky_code() -> str:
        """Guide discovery of high-risk code areas."""
        return """
To find high-risk code areas in the codebase:

1. **High complexity functions**:
   ```
   semantic_query(
       view_id="function_metrics",
       filters=[{"column": "cyclomatic_complexity", "op": "gt", "value": 15}],
       order_by=["-cyclomatic_complexity"]
   )
   ```

2. **Large functions**:
   ```
   semantic_query(
       view_id="function_metrics",
       filters=[{"column": "loc", "op": "gt", "value": 100}],
       order_by=["-loc"]
   )
   ```

3. **Uncovered code** (if coverage data available):
   ```
   semantic_query(
       view_id="coverage_summary",
       filters=[{"column": "coverage_pct", "op": "lt", "value": 50}]
   )
   ```

4. **Recently changed hot spots**:
   Use `code_search` with function names from above to find related modules.
        """
```

---

### L3: ResourceContent for Binary/Streaming Exports

**Category**: A - MCP Tool Enhancements

**Problem**: Export endpoints are HTTP-only; MCP clients can't access Parquet/Arrow exports.

**Solution**: Expose binary exports as MCP resources with proper MIME types.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/resources.py` | Add binary export resources |

**Implementation Sketch**:

```python
# src/codeintel/serving/mcp/resources.py

from fastmcp.resources import ResourceContent

def register_resources(mcp: FastMCP, kernel: SemanticKernel) -> None:
    # ... existing resources ...

    @mcp.resource("export://view/{view_id}/ndjson")
    def export_ndjson(view_id: str) -> ResourceContent:
        """Export view as newline-delimited JSON."""
        from codeintel.serving.semantic.models import SemanticExportRequest

        request = SemanticExportRequest(view_id=view_id, format="ndjson")
        rows = list(kernel.export_rows(request))
        ndjson = "\n".join(json.dumps(row, default=str) for row in rows)

        return ResourceContent(
            content=ndjson,
            mime_type="application/x-ndjson",
        )

    @mcp.resource("export://view/{view_id}/parquet")
    def export_parquet(view_id: str) -> ResourceContent:
        """Export view as Parquet (requires pyarrow)."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            return ResourceContent(
                content="Parquet export requires pyarrow",
                mime_type="text/plain",
            )

        from codeintel.serving.semantic.models import SemanticExportRequest
        import io

        request = SemanticExportRequest(view_id=view_id, format="parquet")
        rows = list(kernel.export_rows(request))
        table = pa.Table.from_pylist(rows)

        buffer = io.BytesIO()
        pq.write_table(table, buffer)

        return ResourceContent(
            content=buffer.getvalue(),
            mime_type="application/vnd.apache.parquet",
        )
```

---

### L4: Correlation ID Propagation to MCP

**Category**: C - Observability & Security

**Problem**: Correlation IDs are HTTP-middleware only; MCP tools use session_id but not integrated.

**Solution**: Use contextvars to propagate correlation ID across tool execution.

**Files to Modify**:

| File | Changes |
|------|---------|
| `src/codeintel/serving/mcp/context.py` | New file for MCP context management |
| `src/codeintel/serving/mcp/app.py` | Set correlation_id from ctx.session_id |

**Implementation**:

```python
# src/codeintel/serving/mcp/context.py

"""MCP context variable management."""

from __future__ import annotations

import contextvars

mcp_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "mcp_correlation_id", default=""
)


def get_mcp_correlation_id() -> str:
    """Return the current MCP correlation ID."""
    return mcp_correlation_id.get()


def set_mcp_correlation_id(value: str) -> contextvars.Token[str]:
    """Set the MCP correlation ID for the current context."""
    return mcp_correlation_id.set(value)
```

```python
# src/codeintel/serving/mcp/app.py

from codeintel.serving.mcp.context import set_mcp_correlation_id

@mcp.tool(...)
async def semantic_query(..., ctx: Context) -> SemanticQueryResult:
    # Set correlation ID from session
    correlation_id = ctx.session_id or f"mcp-{uuid.uuid4().hex[:8]}"
    set_mcp_correlation_id(correlation_id)

    await ctx.info(f"[{correlation_id}] Querying view: {view_id}")
    # ... rest of function ...
```

---

### L5: OpenAPI/FastAPI Integration

**Category**: D - Architecture & Extensibility

**Problem**: HTTP routes and MCP tools are defined separately; potential for drift.

**Solution**: Document pattern for `FastMCP.from_fastapi()` integration.

**This is primarily a documentation/pattern item, not immediate implementation.**

**Documentation to Add**:

```markdown
## FastAPI to MCP Bridge (Future Pattern)

If we need to expose HTTP endpoints via MCP without code duplication:

```python
from fastmcp import FastMCP
from codeintel.serving.http.app import create_serving_app

# Create FastAPI app
fastapi_app = create_serving_app()

# Wrap as MCP server
mcp_from_api = FastMCP.from_fastapi(fastapi_app)

# Now LLMs can call HTTP endpoints via MCP tools
```

This pattern is useful when:
- You have many HTTP endpoints to expose
- You want automatic schema generation
- You need to bridge existing REST APIs to MCP
```

---

### L6: Proxy Pattern for Remote Services

**Category**: D - Architecture & Extensibility

**Problem**: No pattern for integrating external MCP services.

**Solution**: Document proxy pattern for future extensibility.

**This is primarily a documentation/pattern item.**

**Documentation to Add**:

```markdown
## MCP Proxy Pattern (Future Pattern)

If we need to proxy external MCP servers:

```python
from fastmcp import FastMCP, Client

async def create_proxy_server():
    # Connect to external MCP server
    async with Client("wss://external-mcp-server.com") as client:
        # Create proxy server
        proxy = FastMCP.as_proxy(client, name="ExternalTools")

        # Mount into our server
        main = create_mcp_server()
        main.mount(proxy, prefix="external")

        return main
```

Use cases:
- Aggregating multiple MCP servers
- Adding auth/caching layer in front of external services
- Bridging transports (STDIO ↔ HTTP)
```

---

## Implementation Phases

### Phase 1: Core Tool Enhancements (High Priority)
**Duration**: 3-4 days

| Day | Items | Focus |
|-----|-------|-------|
| 1 | H1, H2 | Context API + Annotations |
| 2 | H3, H4 | Async conversion + Error handling |
| 3-4 | H5 | Uvicorn configuration + testing |

**Gate**: All high-priority items complete, tests passing

### Phase 2: Production Hardening (Medium Priority)
**Duration**: 4-5 days

| Day | Items | Focus |
|-----|-------|-------|
| 1 | M1, M2 | Resources + Health checks |
| 2 | M3, M4 | ASGI mounting + Structured output |
| 3 | M5, M6 | Tags + Metrics |
| 4 | M7, M8 | Auth + Feature flags |
| 5 | M9 | Unified lifespan |

**Gate**: All medium-priority items complete, integration tests passing

### Phase 3: Extensibility (Low Priority)
**Duration**: 2-3 days

| Day | Items | Focus |
|-----|-------|-------|
| 1 | L1 | Server composition |
| 2 | L2, L3 | Prompts + Binary resources |
| 3 | L4, L5, L6 | Correlation ID + Documentation |

**Gate**: All items complete, documentation updated

---

## File Change Matrix

### New Files

| File | Purpose |
|------|---------|
| `src/codeintel/serving/mcp/models.py` | Pydantic response models |
| `src/codeintel/serving/mcp/resources.py` | MCP resource handlers |
| `src/codeintel/serving/mcp/prompts.py` | MCP prompt templates |
| `src/codeintel/serving/mcp/context.py` | Correlation ID context |
| `src/codeintel/serving/mcp/servers/` | Sub-server modules (L1) |
| `tests/serving/mcp/test_tools.py` | MCP tool tests |
| `tests/serving/mcp/test_resources.py` | MCP resource tests |

### Modified Files

| File | Items Affecting |
|------|-----------------|
| `src/codeintel/serving/mcp/app.py` | H1, H2, H3, H4, M4, M5, M6, M8, L1 |
| `src/codeintel/serving/mcp/server.py` | M7, M9 |
| `src/codeintel/serving/settings.py` | H1, H4, H5, M8 |
| `src/codeintel/cli/handlers/ops.py` | H5 |
| `src/codeintel/serving/http/app.py` | M3 |

### Dependencies to Add

| Package | Purpose | Items |
|---------|---------|-------|
| `anyio` | Async thread offloading | H3 (already present) |
| `uvloop` | High-performance event loop | H5 (optional) |
| `httptools` | Fast HTTP parser | H5 (optional) |

---

## Testing Strategy

### Unit Tests

| Test File | Coverage |
|-----------|----------|
| `tests/serving/mcp/test_tools.py` | Tool behavior, error handling |
| `tests/serving/mcp/test_resources.py` | Resource resolution |
| `tests/serving/mcp/test_models.py` | Response model validation |
| `tests/serving/mcp/test_context.py` | Context propagation |

### Integration Tests

| Test File | Coverage |
|-----------|----------|
| `tests/serving/mcp/test_mcp_server.py` | End-to-end MCP server |
| `tests/serving/test_http_mcp_mount.py` | MCP mounted in FastAPI |
| `tests/serving/test_uvicorn_config.py` | Uvicorn settings application |

### Performance Tests

| Test | Metric |
|------|--------|
| Concurrent tool invocation | Latency under load |
| Large result handling | Memory usage |
| Async vs sync comparison | Throughput |

---

## Rollout Plan

### Stage 1: Development Environment
- Implement Phase 1 (High Priority)
- Run full test suite
- Manual testing with Claude Desktop / Cursor

### Stage 2: Local Validation
- Implement Phase 2 (Medium Priority)
- Load testing with 3 concurrent clients
- Verify metrics emission

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
| `CODEINTEL_MCP_ENABLE_SEARCH` | `1` | Enable code_search tool |
| `CODEINTEL_MCP_ENABLE_EXPLAIN` | `1` | Enable semantic_explain tool |
| `CODEINTEL_MCP_ENABLE_META` | `1` | Enable serving_meta tool |
| `CODEINTEL_UVICORN_WORKERS` | `1` | Number of Uvicorn workers |
| `CODEINTEL_UVICORN_LOOP` | `auto` | Event loop implementation |
| `CODEINTEL_UVICORN_HTTP` | `auto` | HTTP parser implementation |
| `CODEINTEL_UVICORN_LIMIT_CONCURRENCY` | (none) | Max concurrent connections |
| `CODEINTEL_UVICORN_LIMIT_MAX_REQUESTS` | (none) | Requests before worker restart |
| `CODEINTEL_UVICORN_TIMEOUT_KEEP_ALIVE` | `30` | Keep-alive timeout seconds |
| `CODEINTEL_UVICORN_BACKLOG` | `2048` | Connection backlog size |
| `CODEINTEL_UVICORN_ACCESS_LOG` | `1` | Enable access logging |
| `CODEINTEL_UVICORN_SERVER_HEADER` | `0` | Include server header |

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

*Document created: 2025-12-16*
*Last updated: 2025-12-16*
*Status: Ready for implementation*

