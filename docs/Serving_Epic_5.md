
# Serving detailed implementation plan Epic 5 #

Nice, we’re into the “observability backbone” phase now 😄

Let’s turn Refactor 5 into a **concrete, patchable plan**, using your *current* layout:

* `serving/services/observability.py`
* `serving/services/query_service.py`
* `serving/http/fastapi.py` + `http/dependencies.py`
* `serving/mcp/tool_utils.py` + `serving/mcp/*_tools.py`
* `serving/services/errors.py`

We’ll do this in five steps:

1. Add a **RequestContext** + contextvars helper (`serving/context.py`), and hook it into `generate_correlation_id`.
2. Upgrade **ServiceCallMetrics / ServiceObservability / _observe_call** to use RequestContext.
3. Wire RequestContext from the **HTTP** entrypoint.
4. Wire RequestContext from **MCP** tools (spec-driven).
5. Add a couple of **tests** patterns to lock it in.

---

## 1. RequestContext + context management

### 1.1. New module: `serving/context.py`

**New file**: `src/codeintel/serving/context.py`

This will hold:

* The `RequestContext` dataclass.
* A `ContextVar` to store the current context.
* Helper functions to set/get/reset it.

```python
# src/codeintel/serving/context.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any
from contextvars import ContextVar, Token


@dataclass
class RequestContext:
    """
    Per-request context propagated across layers.

    This is transport-agnostic; it captures external (http/mcp/cli)
    and repo-level info. Operation/dataset remain in call metrics.
    """

    correlation_id: str
    transport: Literal["http", "mcp", "cli"]
    operation: str | None = None        # e.g. "datasets.rows"
    dataset: str | None = None
    repo: str | None = None
    commit: str | None = None
    snapshot: Any | None = None
    graph_scope: Any | None = None      # you can later type this with GraphScope
    client_id: str | None = None        # MCP connection ID / HTTP client hint
    user_agent: str | None = None       # HTTP User-Agent or MCP client name


_current_request_context: ContextVar[RequestContext | None] = ContextVar(
    "codeintel_current_request_context",
    default=None,
)


def set_current_request_context(ctx: RequestContext) -> Token:
    """Set the current RequestContext and return a token for reset."""
    return _current_request_context.set(ctx)


def get_current_request_context() -> RequestContext | None:
    """Return the current RequestContext, if any."""
    return _current_request_context.get()


def reset_current_request_context(token: Token) -> None:
    """Reset the current RequestContext to a previous value."""
    _current_request_context.reset(token)
```

### 1.2. Hook `generate_correlation_id` into RequestContext

Edit: `src/codeintel/serving/services/errors.py`.

Right now `generate_correlation_id()` just returns `str(uuid4())`. We want it to reuse the **current RequestContext** if present, so ProblemDetails share the same `instance` as the current request.

**Before:**

```python
from uuid import uuid4

def generate_correlation_id() -> str:
    """
    Return a new correlation identifier for tracing errors.
    """
    return str(uuid4())
```

**After:**

```python
from uuid import uuid4
from codeintel.serving.context import get_current_request_context

def generate_correlation_id() -> str:
    """
    Return a correlation identifier for tracing errors.

    If a RequestContext is active, reuse its correlation_id; otherwise
    generate a new one.
    """
    ctx = get_current_request_context()
    if ctx is not None and ctx.correlation_id:
        return ctx.correlation_id
    return str(uuid4())
```

This means:

* If an HTTP/MCP request has set `RequestContext`, all `ProblemDetail.instance` values created under that context share the same correlation id.
* If you call code locally without a request, you still get a fresh UUID.

---

## 2. Upgrade Observability to use RequestContext

Now we make observability aware of RequestContext, but we keep call sites minimal.

### 2.1. Extend ServiceCallMetrics

Edit: `src/codeintel/serving/services/observability.py`

Find:

```python
@dataclass
class ServiceCallMetrics:
    """Structured metrics describing a service invocation."""

    name: str
    transport: str
    duration_ms: float
    rows: int | None = None
    dataset: str | None = None
    messages: int | None = None
    error: str | None = None
    truncated: bool | None = None
    schema_version: str | None = None
    retries: int | None = None
```

**After** (add a few fields that will be filled from RequestContext):

```python
from codeintel.serving.context import RequestContext, get_current_request_context

@dataclass
class ServiceCallMetrics:
    """Structured metrics describing a service invocation."""

    # Call-level fields
    name: str                 # operation name, e.g. "datasets.rows"
    transport: str            # "local" / "http" (backend transport)
    duration_ms: float

    rows: int | None = None
    dataset: str | None = None
    messages: int | None = None
    error: str | None = None
    truncated: bool | None = None
    schema_version: str | None = None
    retries: int | None = None

    # RequestContext-projected fields
    correlation_id: str | None = None
    external_transport: str | None = None   # "http" / "mcp" / "cli"
    operation: str | None = None            # high-level op id (optional)
    repo: str | None = None
    commit: str | None = None
    client_id: str | None = None
    user_agent: str | None = None
```

We’ll let `_observe_call` fill these from `RequestContext`.

### 2.2. Change `ServiceObservability.record` to accept RequestContext

Still in `observability.py`, find `class ServiceObservability`:

**Before:**

```python
class ServiceObservability:
    """Configuration for service-level observability."""

    enabled: bool = False
    logger: logging.Logger = field(default_factory=lambda: LOG)

    def record(self, metrics: ServiceCallMetrics) -> None:
        ...
        payload: dict[str, object] = {
            "name": metrics.name,
            "transport": metrics.transport,
            "duration_ms": round(metrics.duration_ms, 2),
        }
        if metrics.rows is not None:
            payload["rows"] = metrics.rows
        ...
        self.logger.info("service_call %s", payload)
```

**After:**

```python
class ServiceObservability:
    """Configuration for service-level observability."""

    enabled: bool = False
    logger: logging.Logger = field(default_factory=lambda: LOG)

    def record(
        self,
        metrics: ServiceCallMetrics,
        context: RequestContext | None = None,
    ) -> None:
        """
        Emit a structured log line for a service call.

        Parameters
        ----------
        metrics:
            Call metrics describing the invocation outcome.
        context:
            Optional RequestContext to enrich the payload.
        """
        if not self.enabled or not self.logger.isEnabledFor(logging.INFO):
            return

        payload: dict[str, object] = {
            "name": metrics.name,
            "transport": metrics.transport,
            "duration_ms": round(metrics.duration_ms, 2),
        }
        if metrics.rows is not None:
            payload["rows"] = metrics.rows
        if metrics.dataset is not None:
            payload["dataset"] = metrics.dataset
        if metrics.messages is not None:
            payload["messages"] = metrics.messages
        if metrics.error is not None:
            payload["error"] = metrics.error
        if metrics.truncated is not None:
            payload["truncated"] = metrics.truncated
        if metrics.schema_version is not None:
            payload["schema_version"] = metrics.schema_version
        if metrics.retries is not None:
            payload["retries"] = metrics.retries

        # RequestContext extras
        if metrics.correlation_id is not None:
            payload["correlation_id"] = metrics.correlation_id
        if metrics.external_transport is not None:
            payload["external_transport"] = metrics.external_transport
        if metrics.operation is not None:
            payload["operation"] = metrics.operation
        if metrics.repo is not None:
            payload["repo"] = metrics.repo
        if metrics.commit is not None:
            payload["commit"] = metrics.commit
        if metrics.client_id is not None:
            payload["client_id"] = metrics.client_id
        if metrics.user_agent is not None:
            payload["user_agent"] = metrics.user_agent

        self.logger.info("service_call %s", payload)
```

### 2.3. Enrich metrics inside `_observe_call`

Finally, still in `observability.py`, update `_observe_call`.

**Before** (simplified):

```python
def _observe_call[T](
    observability: ServiceObservability | None,
    *,
    transport: str,
    name: str,
    context: ServiceCallContext | None,
    func: Callable[[], T],
) -> T:
    start = time.perf_counter()
    try:
        result = func()
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        if observability is not None:
            observability.record(
                ServiceCallMetrics(
                    name=name,
                    transport=transport,
                    duration_ms=duration_ms,
                    error=exc.__class__.__name__,
                    dataset=context.dataset if context is not None else None,
                    schema_version=context.schema_version if context is not None else None,
                    retries=context.retries if context is not None else None,
                )
            )
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    if observability is not None:
        observability.record(
            ServiceCallMetrics(
                name=name,
                transport=transport,
                duration_ms=duration_ms,
                rows=_infer_row_count(result),
                dataset=context.dataset if context is not None else None,
                messages=_extract_message_count(result),
                truncated=_extract_truncated(result),
                schema_version=context.schema_version if context is not None else None,
                retries=context.retries if context is not None else None,
            )
        )
    return result
```

**After**: pull current `RequestContext` once, use it to enrich `ServiceCallMetrics`, and pass it into `record`.

```python
def _observe_call[T](
    observability: ServiceObservability | None,
    *,
    transport: str,
    name: str,
    context: ServiceCallContext | None,
    func: Callable[[], T],
) -> T:
    """
    Execute a callable while capturing observability signals.

    Uses the current RequestContext (if any) to enrich metrics.
    """
    req_ctx = get_current_request_context()
    start = time.perf_counter()

    try:
        result = func()
    except Exception as exc:  # noqa: BLE001
        duration_ms = (time.perf_counter() - start) * 1000
        if observability is not None:
            metrics = ServiceCallMetrics(
                name=name,
                transport=transport,
                duration_ms=duration_ms,
                error=exc.__class__.__name__,
                dataset=context.dataset if context is not None else None,
                schema_version=context.schema_version if context is not None else None,
                retries=context.retries if context is not None else None,
            )

            if req_ctx is not None:
                metrics.correlation_id = req_ctx.correlation_id
                metrics.external_transport = req_ctx.transport
                metrics.operation = req_ctx.operation or name
                metrics.repo = req_ctx.repo
                metrics.commit = req_ctx.commit
                metrics.client_id = req_ctx.client_id
                metrics.user_agent = req_ctx.user_agent

            observability.record(metrics, context=req_ctx)

        raise

    duration_ms = (time.perf_counter() - start) * 1000
    if observability is not None:
        metrics = ServiceCallMetrics(
            name=name,
            transport=transport,
            duration_ms=duration_ms,
            rows=_infer_row_count(result),
            dataset=context.dataset if context is not None else None,
            messages=_extract_message_count(result),
            truncated=_extract_truncated(result),
            schema_version=context.schema_version if context is not None else None,
            retries=context.retries if context is not None else None,
        )

        if req_ctx is not None:
            metrics.correlation_id = req_ctx.correlation_id
            metrics.external_transport = req_ctx.transport
            metrics.operation = req_ctx.operation or name
            metrics.repo = req_ctx.repo
            metrics.commit = req_ctx.commit
            metrics.client_id = req_ctx.client_id
            metrics.user_agent = req_ctx.user_agent

        observability.record(metrics, context=req_ctx)

    return result
```

> Call sites (`LocalQueryService._call`, `_HttpTransportMixin._http_call`) do **not** need to change: they still call `_observe_call(self.observability, transport=..., name=..., context=..., func=...)`.

---

## 3. HTTP: create a RequestContext per incoming request

We’ll add a **middleware** in `fastapi.py` that:

* Extracts / generates a `correlation_id`.
* Builds a `RequestContext(transport="http", ...)`.
* Sets the contextvar for the duration of the request.

### 3.1. Middleware in `http/fastapi.py`

Edit: `src/codeintel/serving/http/fastapi.py`.

At top-level imports, add:

```python
from codeintel.serving.context import RequestContext, set_current_request_context, reset_current_request_context
from codeintel.serving.services.errors import generate_correlation_id
```

Inside `create_app`, **after** you create `app = FastAPI(...)` but **before** including routers, add:

```python
    @app.middleware("http")
    async def _inject_request_context(request: Request, call_next: Callable) -> JSONResponse:
        """
        Attach a RequestContext for each incoming HTTP request.

        Correlation ID is taken from X-Request-ID / X-Correlation-ID if provided,
        otherwise generated via generate_correlation_id().
        """
        # Determine correlation ID
        correlation_id = (
            request.headers.get("X-Request-ID")
            or request.headers.get("X-Correlation-ID")
            or generate_correlation_id()
        )

        # Try to fetch repo/commit from app state config if present
        cfg: ServingConfig | None = getattr(request.app.state, "config", None)
        repo = getattr(cfg, "repo", None) if cfg is not None else None
        commit = getattr(cfg, "commit", None) if cfg is not None else None

        ctx = RequestContext(
            correlation_id=correlation_id,
            transport="http",
            operation=None,  # filled in per call via metrics.name
            dataset=None,
            repo=repo,
            commit=commit,
            snapshot=None,
            graph_scope=None,
            client_id=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
        )
        token = set_current_request_context(ctx)
        try:
            response = await call_next(request)
        finally:
            reset_current_request_context(token)

        # Optionally echo correlation id to the client
        if hasattr(response, "headers"):
            response.headers.setdefault("X-Request-ID", correlation_id)
        return response
```

> Now, any call into `LocalQueryService` / `HttpQueryService` under HTTP will see this context in `_observe_call`, and your ProblemDetails will reuse this `correlation_id` via `generate_correlation_id`.

---

## 4. MCP: create a RequestContext per tool invocation

For MCP, the natural place to hook is the **spec-driven tool registration** we did in Epic 3: function/dataset/architecture tools.

We’ll update the spec-driven registration helpers (e.g. in `function_tools.py`, `dataset_tools.py`, `architecture_tools.py`) so that each tool:

* Builds a RequestContext with `transport="mcp"`.
* Sets it in a contextvar for the duration of the tool.
* Then calls `backend.<method>(**kwargs)` → service → observability.

We also want to mark the **operation id** from `OperationSpec.id`.

### 4.1. Example: `serving/mcp/function_tools.py`

You already refactored this to something like:

```python
from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.models import ProblemDetail
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap
from codeintel.serving.registry import iter_operation_specs, get_operation_spec


def register_function_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    def _register_tool_for_spec(spec_id: str) -> None:
        spec = get_operation_spec(spec_id)
        ...
        backend_method = getattr(backend, spec.backend_method)

        @_wrap
        def _tool(**kwargs: Any) -> dict[str, object] | dict[str, ProblemDetail]:
            response = backend_method(**kwargs)
            return response.model_dump()

        _tool.__name__ = spec.tool_name
        _tool.__doc__ = spec.description or spec.summary

        mcp.tool(name=spec.tool_name, description=spec.summary)(_tool)

    for spec in iter_operation_specs():
        if spec.category != "functions" or spec.tool_name is None:
            continue
        _register_tool_for_spec(spec.id)
```

We’ll wrap the `_tool` body with RequestContext.

Add imports:

```python
from codeintel.serving.context import RequestContext, set_current_request_context, reset_current_request_context
from codeintel.serving.services.errors import generate_correlation_id
```

Then change `_tool`:

```python
        @_wrap
        def _tool(**kwargs: Any) -> dict[str, object] | dict[str, ProblemDetail]:
            """
            Dynamically generated tool that forwards to backend.<method>(**kwargs).
            """
            # Build a RequestContext for this MCP invocation
            correlation_id = generate_correlation_id()
            dataset = kwargs.get("dataset_name") or kwargs.get("dataset")
            ctx = RequestContext(
                correlation_id=correlation_id,
                transport="mcp",
                operation=spec.id,
                dataset=str(dataset) if dataset is not None else None,
                repo=getattr(backend, "repo", None),
                commit=getattr(backend, "commit", None),
                snapshot=None,
                graph_scope=kwargs.get("scope"),
                client_id=None,     # you can thread FastMCP connection info here later
                user_agent=None,    # e.g. MCP client name if available
            )
            token = set_current_request_context(ctx)
            try:
                response = backend_method(**kwargs)
                return response.model_dump()
            finally:
                reset_current_request_context(token)
```

> Now, when a MCP tool runs, any observability call inside the `QueryService` path sees `RequestContext(transport="mcp", operation=spec.id, correlation_id=...)`.

### 4.2. Apply the same pattern to dataset + architecture tools

In `serving/mcp/dataset_tools.py` and `serving/mcp/architecture_tools.py`, you can do the same:

* Import `RequestContext`, `set_current_request_context`, `reset_current_request_context`, `generate_correlation_id`.
* In your spec-driven `_register_tool_for_spec`, wrap each `_tool(**kwargs)` with the same pattern, but tweak how you derive `dataset`/`graph_scope`:

#### Dataset tools (example):

```python
dataset = kwargs.get("dataset_name")
ctx = RequestContext(
    correlation_id=correlation_id,
    transport="mcp",
    operation=spec.id,
    dataset=str(dataset) if dataset is not None else None,
    repo=getattr(backend, "repo", None),
    commit=getattr(backend, "commit", None),
    snapshot=None,
    graph_scope=None,
    client_id=None,
    user_agent=None,
)
```

#### Subsystem / architecture tools (example):

```python
# operation-specific hints
dataset = None
graph_scope = kwargs.get("scope")

ctx = RequestContext(
    correlation_id=correlation_id,
    transport="mcp",
    operation=spec.id,
    dataset=dataset,
    repo=getattr(backend, "repo", None),
    commit=getattr(backend, "commit", None),
    snapshot=None,
    graph_scope=graph_scope,
    client_id=None,
    user_agent=None,
)
```

The important thing is: **every MCP tool call sets a RequestContext** before hitting the backend/service, and resets it after.

---

## 5. Tests to lock it in

You don’t have to add these immediately, but here are two focused patterns that will prove the backbone is working:

### 5.1. Service-level observability test

Create a stub observability to capture calls.

```python
# tests/services/test_request_context_observability.py

from dataclasses import dataclass
from typing import Any, List

from codeintel.serving.context import RequestContext, set_current_request_context, reset_current_request_context
from codeintel.serving.services.observability import ServiceObservability, ServiceCallMetrics
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.serving.backend import DuckDBQueryService, BackendLimits
# plus any helpers to build a test gateway/analytics context


@dataclass
class CapturingObservability(ServiceObservability):
    events: List[tuple[ServiceCallMetrics, RequestContext | None]] = None

    def __post_init__(self) -> None:
        if self.events is None:
            self.events = []

    def record(self, metrics: ServiceCallMetrics, context: RequestContext | None = None) -> None:
        self.events.append((metrics, context))


def test_local_query_observability_uses_request_context(dummy_local_service: LocalQueryService) -> None:
    obs = CapturingObservability(enabled=True)
    dummy_local_service.observability = obs

    ctx = RequestContext(
        correlation_id="test-cid-123",
        transport="http",
        operation=None,
        dataset=None,
        repo="my/repo",
        commit="abc123",
        snapshot=None,
        graph_scope=None,
        client_id="test-client",
        user_agent="pytest",
    )
    token = set_current_request_context(ctx)
    try:
        # Call some operation that triggers _observe_call, e.g. list_datasets
        dummy_local_service.list_datasets()
    finally:
        reset_current_request_context(token)

    assert len(obs.events) == 1
    metrics, context = obs.events[0]
    assert metrics.correlation_id == "test-cid-123"
    assert context is not None
    assert context.repo == "my/repo"
    assert metrics.name  # e.g. "datasets.list"
```

### 5.2. HTTP correlation id test

Using FastAPI’s `TestClient`:

```python
# tests/http/test_request_context_http.py

from fastapi.testclient import TestClient

from codeintel.serving.http.fastapi import create_app
from codeintel.config.serving_models import ServingConfig


def test_correlation_id_plumbed_into_problem_detail(tmp_path) -> None:
    cfg = ServingConfig(
        # minimal config that yields a repo with no datasets or some known 404
    )
    app = create_app(config_loader=lambda: cfg)
    client = TestClient(app)

    headers = {"X-Request-ID": "cid-from-header"}
    resp = client.get("/datasets/no_such_dataset", headers=headers)

    assert resp.status_code == 404
    payload = resp.json()
    assert payload["instance"] == "cid-from-header"
    assert payload["code"] == "dataset-not-found"
    assert resp.headers["X-Request-ID"] == "cid-from-header"
```

You can add a similar MCP test by:

* Spinning up a `FastMCP` instance + dummy backend.
* Registering tools.
* Invoking a dataset/tool and making sure `ServiceObservability` sees `RequestContext.transport == "mcp"` and a correlation id.

---

## TL;DR

Refactor 5 boils down to:

* **Single RequestContext backbone** in `serving/context.py` with a `ContextVar`.
* **Observability** enriched with correlation id, repo, external transport and operation id, using that RequestContext.
* **HTTP middleware** that sets `RequestContext(transport="http")` per incoming request and echoes `X-Request-ID`.
* **MCP tools** that set `RequestContext(transport="mcp", operation=spec.id, dataset=…)` per tool invocation.
* **Errors** reuse `RequestContext.correlation_id` via `generate_correlation_id`, so ProblemDetails and logs share the same correlation id everywhere.

Once this is in, your logs and metrics will be able to answer:

> “For correlation id X, what operations ran, against which dataset, via which transport, and what errors or truncation happened?”

And all of that without changing your core service call sites (`LocalQueryService._call`, `_HttpTransportMixin._http_call`) at all.
