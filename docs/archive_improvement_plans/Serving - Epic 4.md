Gotcha, let’s level this up into “real” domain vs transport separation + a unified error story.

I’ll treat Epic 4 as two tightly-related chunks:

1. **4a – Domain models vs transport models**

   * New `serving/domain_models.py` dataclasses.
   * QueryService protocols return **domain** types.
   * Local/HTTP services convert Pydantic ⇄ domain at the edges.
   * QueryBackend stays Pydantic for MCP, wrapping domain.

2. **4b – Single error & ProblemDetail hierarchy**

   * Make `ProblemDetail` dataclass the canonical internal representation.
   * HTTP + MCP each have thin wrappers that convert to/from it.
   * Introduce explicit error subclasses + cross-transport tests.

I’ll give concrete snippets and “where to paste” for each step.

---

## 4a. Domain models vs transport models

### 4a.0 Target state

* **Domain-layer** (`serving/domain_models.py`) knows nothing about Pydantic or transport.
* `QueryService` (in `serving/services/query_service.py`) returns **domain dataclasses**.
* `LocalQueryService` and `HttpQueryService`:

  * Call lower-level backends (`DuckDBQueryService` or HTTP) that still return Pydantic for now.
  * Convert those Pydantic responses `.to_domain()` before returning.
* **HTTP** and **MCP**:

  * Keep using **Pydantic** models for `response_model` and for MCP payloads.
  * But they **reconstruct Pydantic models from domain** via `.from_domain()` right before serialization.

So the layering becomes:

> DuckDB / HTTP JSON → Pydantic (mcp.models) → **domain dataclasses** → Pydantic transport models → wire.

---

### 4a.1 Create `serving/domain_models.py`

**New file**: `src/codeintel/serving/domain_models.py`

We don’t need to mirror *every* field right away, but we want:

* A canonical `ResponseMeta` for truncation/limit messages.
* Core domain models for:

  * Function summary.
  * High-risk functions.
  * Dataset rows/schema.
  * Subsystem summaries, etc.

Here’s a starter set that gives you the pattern (you/the agent can extend it across the rest of your Pydantic responses):

```python
# src/codeintel/serving/domain_models.py
"""Transport-agnostic domain models for serving."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """Domain-level diagnostic message attached to responses."""

    code: str
    severity: str  # "info" | "warning" | "error"
    detail: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseMeta:
    """Transport-agnostic metadata for paginated / limited responses."""

    requested_limit: int | None = None
    applied_limit: int | None = None
    requested_offset: int | None = None
    applied_offset: int | None = None
    truncated: bool = False
    messages: list[Message] = field(default_factory=list)


# ---------------------------------------------------------------------------
# FUNCTION / FILE / MODULE DOMAIN TYPES
# ---------------------------------------------------------------------------

@dataclass
class FunctionSummary:
    """Core function summary information used across transports."""

    urn: str
    goid_h128: int
    rel_path: str
    qualname: str
    short_summary: str | None
    long_summary: str | None
    is_test: bool
    meta: ResponseMeta


@dataclass
class HighRiskFunction:
    """Single row in a high-risk function listing."""

    goid_h128: int
    qualname: str
    rel_path: str
    risk_score: float
    is_tested: bool


@dataclass
class HighRiskFunctions:
    """Domain representation of high-risk functions listing."""

    functions: list[HighRiskFunction]
    meta: ResponseMeta


@dataclass
class FileSummary:
    """Summary of a file and its contained functions."""

    rel_path: str
    module: str | None
    functions: list[FunctionSummary]
    meta: ResponseMeta


# ---------------------------------------------------------------------------
# DATASET DOMAIN TYPES
# ---------------------------------------------------------------------------

@dataclass
class DatasetDescriptorDomain:
    """Domain-level description of a dataset."""

    name: str
    table: str
    description: str
    family: str | None = None
    owner: str | None = None
    schema_version: str | None = None
    stable_id: str | None = None
    is_docs_view: bool = False
    is_read_only: bool = False


@dataclass
class DatasetRows:
    """Domain representation of dataset rows plus meta."""

    dataset_name: str
    limit: int
    offset: int
    rows: list[dict[str, Any]]
    meta: ResponseMeta


@dataclass
class DatasetSchema:
    """Domain representation of a dataset schema + samples."""

    dataset_name: str
    table_key: str
    duckdb_schema: list[dict[str, Any]]
    json_schema: dict[str, Any] | None
    sample_rows: list[dict[str, Any]]
    schema_version: str | None
    owner: str | None
    retention_policy: str | None
    freshness_sla: str | None
    stable_id: str | None
    meta: ResponseMeta | None = None
```

You can (and should) extend this file with:

* `SubsystemSummary`, `SubsystemProfile`, `SubsystemCoverage`, `ModuleSubsystem`, etc.
* `FunctionProfile`, `FileProfile`, `ModuleProfile`.
* `FunctionArchitecture`, `ModuleArchitecture`, `FileHints`.

…but you only need a handful to start the refactor; the pattern is identical.

---

### 4a.2 Add `to_domain` / `from_domain` on key Pydantic response models

Now we teach your Pydantic models how to convert to/from these domain dataclasses.

Edit `src/codeintel/serving/mcp/models.py`:

At the top, after imports:

```python
from codeintel.serving import domain_models as dm
```

Then, for `ResponseMeta`:

```python
class ResponseMeta(BaseModel):
    requested_limit: int | None = None
    applied_limit: int | None = None
    requested_offset: int | None = None
    applied_offset: int | None = None
    truncated: bool = False
    messages: list[Message] | None = None

    def to_domain(self) -> dm.ResponseMeta:
        return dm.ResponseMeta(
            requested_limit=self.requested_limit,
            applied_limit=self.applied_limit,
            requested_offset=self.requested_offset,
            applied_offset=self.applied_offset,
            truncated=self.truncated,
            messages=[
                m.to_domain() for m in (self.messages or [])
            ],
        )

    @classmethod
    def from_domain(cls, meta: dm.ResponseMeta) -> "ResponseMeta":
        return cls(
            requested_limit=meta.requested_limit,
            applied_limit=meta.applied_limit,
            requested_offset=meta.requested_offset,
            applied_offset=meta.applied_offset,
            truncated=meta.truncated,
            messages=[
                Message.from_domain(m) for m in meta.messages
            ],
        )
```

And for `Message`:

```python
class Message(BaseModel):
    code: str
    severity: Literal["info", "warning", "error"] = "info"
    detail: str | None = None
    context: dict[str, object] | None = None

    def to_domain(self) -> dm.Message:
        return dm.Message(
            code=self.code,
            severity=self.severity,
            detail=self.detail,
            context=dict(self.context or {}),
        )

    @classmethod
    def from_domain(cls, msg: dm.Message) -> "Message":
        return cls(
            code=msg.code,
            severity=msg.severity,
            detail=msg.detail,
            context=msg.context or {},
        )
```

Then for **FunctionSummaryResponse** (as an example):

```python
class FunctionSummaryResponse(BaseModel):
    # existing fields...
    meta: ResponseMeta

    def to_domain(self) -> dm.FunctionSummary:
        return dm.FunctionSummary(
            urn=self.urn,
            goid_h128=self.goid_h128,
            rel_path=self.rel_path,
            qualname=self.qualname,
            short_summary=self.short_summary,
            long_summary=self.long_summary,
            is_test=self.is_test,
            meta=self.meta.to_domain(),
        )

    @classmethod
    def from_domain(cls, summary: dm.FunctionSummary) -> "FunctionSummaryResponse":
        return cls(
            urn=summary.urn,
            goid_h128=summary.goid_h128,
            rel_path=summary.rel_path,
            qualname=summary.qualname,
            short_summary=summary.short_summary,
            long_summary=summary.long_summary,
            is_test=summary.is_test,
            meta=ResponseMeta.from_domain(summary.meta),
        )
```

And for **DatasetRowsResponse**:

```python
class DatasetRowsResponse(BaseModel):
    dataset_name: str
    limit: int
    offset: int
    rows: list[ViewRow]
    meta: ResponseMeta | None = None

    def to_domain(self) -> dm.DatasetRows:
        return dm.DatasetRows(
            dataset_name=self.dataset_name,
            limit=self.limit,
            offset=self.offset,
            rows=[row.model_dump() for row in self.rows],
            meta=self.meta.to_domain() if self.meta is not None else dm.ResponseMeta(),
        )

    @classmethod
    def from_domain(cls, rows: dm.DatasetRows) -> "DatasetRowsResponse":
        return cls(
            dataset_name=rows.dataset_name,
            limit=rows.limit,
            offset=rows.offset,
            rows=[ViewRow.model_validate(r) for r in rows.rows],
            meta=ResponseMeta.from_domain(rows.meta),
        )
```

Same pattern for `DatasetSchemaResponse`, `HighRiskFunctionsResponse`, etc. — but you can roll those in incrementally.

---

### 4a.3 Change `QueryService` Protocol to return domain types

Edit `src/codeintel/serving/services/query_service.py`.

At the top, change the imports:

**Before**:

```python
from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    DatasetDescriptor,
    DatasetRowsResponse,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
    FileHintsResponse,
    FileProfileResponse,
    FileSummaryResponse,
    FunctionArchitectureResponse,
    FunctionProfileResponse,
    FunctionSummaryResponse,
    GraphNeighborhoodResponse,
    HighRiskFunctionsResponse,
    ImportBoundaryResponse,
    ModuleArchitectureResponse,
    ModuleProfileResponse,
    ModuleSubsystemResponse,
    ResponseMeta,
    SubsystemCoverageResponse,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
    TestsForFunctionResponse,
)
```

**After**:

```python
from codeintel.serving import domain_models as dm
from codeintel.serving.mcp.models import (
    DatasetDescriptor,
    DatasetSpecDescriptor,
    GraphScopePayload,
)
```

Then update the protocol signatures. For example:

```python
class FunctionQueryApi(Protocol):
    def get_function_summary(... ) -> FunctionSummaryResponse: ...
```

becomes:

```python
class FunctionQueryApi(Protocol):
    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.FunctionSummary:
        ...
```

And:

```python
    def list_high_risk_functions(...) -> HighRiskFunctionsResponse: ...
```

→

```python
    def list_high_risk_functions(...) -> dm.HighRiskFunctions: ...
```

Similarly for:

* `get_file_summary` → `dm.FileSummary`
* `read_dataset_rows` → `dm.DatasetRows`
* `dataset_schema` → `dm.DatasetSchema`
* subsystem-related APIs → their future domain equivalents (you can add them to `domain_models.py` as needed).

You can leave `DatasetDescriptor` / `DatasetSpecDescriptor` as Pydantic for now if you’d like; or add domain versions later.

---

### 4a.4 Convert Pydantic → domain inside service delegates

Now change the service delegate mixins to call `.to_domain()` before returning.

#### Example: local function delegates (`serving/services/functions.py`)

**Before** (simplified):

```python
class _FunctionQueryDelegates:
    query: DuckDBQueryService
    _call: Callable[..., Any]

    def get_function_summary(...) -> FunctionSummaryResponse:
        return self._call(
            "get_function_summary",
            lambda: self.query.get_function_summary(...),
            dataset="docs.v_function_summary",
        )
```

**After**:

```python
from codeintel.serving import domain_models as dm
from codeintel.serving.mcp.models import FunctionSummaryResponse

class _FunctionQueryDelegates:
    query: DuckDBQueryService
    _call: Callable[..., Any]

    def get_function_summary(... ) -> dm.FunctionSummary:
        pydantic_resp: FunctionSummaryResponse = self._call(
            "get_function_summary",
            lambda: self.query.get_function_summary(...),
            dataset="docs.v_function_summary",
        )
        return pydantic_resp.to_domain()
```

Do the same for `list_high_risk_functions`, `get_callgraph_neighbors`, etc., once you add appropriate `to_domain`() methods to their Pydantic models.

#### Example: dataset delegate (`serving/services/datasets.py`)

For `LocalQueryService.read_dataset_rows` via `_LocalDatasetMixin`:

**Before**:

```python
def read_dataset_rows(...) -> DatasetRowsResponse:
    return self._call(
        "read_dataset_rows",
        lambda: self.query.read_dataset_rows(...),
        dataset=dataset_name,
        schema_version=schema_version,
    )
```

**After**:

```python
from codeintel.serving import domain_models as dm
from codeintel.serving.mcp.models import DatasetRowsResponse

def read_dataset_rows(...) -> dm.DatasetRows:
    pydantic_resp: DatasetRowsResponse = self._call(
        "read_dataset_rows",
        lambda: self.query.read_dataset_rows(...),
        dataset=dataset_name,
        schema_version=schema_version,
    )
    return pydantic_resp.to_domain()
```

For `HttpQueryService` mixins (e.g. `_HttpDatasetQueryMixin`), you do the same: HTTP → Pydantic → `.to_domain()` → return domain.

---

### 4a.5 Wrap domain → Pydantic in QueryBackend

Now we keep MCP-specific `QueryBackend` returning **Pydantic** (because MCP tools still expect `.model_dump()`), but inside the backend we convert from domain.

Edit `src/codeintel/serving/mcp/backend.py`.

At the top, import the domain models:

```python
from codeintel.serving import domain_models as dm
from codeintel.serving.mcp.models import (
    FunctionSummaryResponse,
    HighRiskFunctionsResponse,
    DatasetRowsResponse,
    DatasetSchemaResponse,
    # ...
)
```

In `DuckDBBackend` and `HttpBackend`, methods look something like:

**Before**:

```python
class DatasetBackendMixin:
    def read_dataset_rows(...) -> DatasetRowsResponse:
        return self.service.read_dataset_rows(...)
```

**After**:

```python
class DatasetBackendMixin:
    def read_dataset_rows(...) -> DatasetRowsResponse:
        domain_rows: dm.DatasetRows = self.service.read_dataset_rows(
            dataset_name=dataset_name,
            limit=limit,
            offset=offset,
        )
        return DatasetRowsResponse.from_domain(domain_rows)

    def dataset_schema(...) -> DatasetSchemaResponse:
        domain_schema: dm.DatasetSchema = self.service.dataset_schema(
            dataset_name=dataset_name,
            sample_limit=sample_limit,
        )
        return DatasetSchemaResponse.from_domain(domain_schema)
```

Similarly for function/architecture methods:

```python
class DuckDBBackend(QueryBackend):
    service: QueryService

    def get_function_summary(...) -> FunctionSummaryResponse:
        domain_summary: dm.FunctionSummary = self.service.get_function_summary(...)
        return FunctionSummaryResponse.from_domain(domain_summary)
```

This keeps the **MCP-facing interface unchanged** (still Pydantic), while the rest of the app now talks in terms of domain dataclasses.

---

### 4a.6 HTTP handlers wrap domain → Pydantic

HTTP routes currently call `service` and return Pydantic directly. After changing `QueryService` to return domain types, you adapt routes:

#### Example: `serving/http/routes/functions.py`

**Before**:

```python
from codeintel.serving.mcp.models import FunctionSummaryResponse

@router.get("/function/summary", response_model=FunctionSummaryResponse)
def function_summary(..., service: ServiceDep, ...) -> FunctionSummaryResponse:
    return service.get_function_summary(...)
```

**After**:

```python
from codeintel.serving.mcp.models import FunctionSummaryResponse
from codeintel.serving import domain_models as dm

@router.get(spec_summary.http_path, response_model=FunctionSummaryResponse, summary=spec_summary.summary)
def function_summary(..., service: ServiceDep, ...) -> FunctionSummaryResponse:
    domain_summary: dm.FunctionSummary = service.get_function_summary(
        urn=urn,
        goid_h128=goid_h128,
        rel_path=rel_path,
        qualname=qualname,
        scope=scope,
    )
    return FunctionSummaryResponse.from_domain(domain_summary)
```

And similarly for dataset and subsystem routes.

---

## 4b. Single error & ProblemDetail hierarchy

Now let’s unify error handling so **all** code shares a common ProblemDetail dataclass and consistent codes.

### 4b.1 Canonical ProblemDetail dataclass

You already have `serving/services/errors.py` with a dataclass ProblemDetail + ProblemError subclasses. Let’s treat this as canonical and slightly tweak it to match the planned shape.

Edit `src/codeintel/serving/services/errors.py` so the `ProblemDetail` looks like:

```python
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

def generate_correlation_id() -> str:
    return str(uuid4())


@dataclass(frozen=True)
class ProblemDetail:
    """
    Canonical domain-level Problem Details representation.

    Mirrors RFC 9457/RFC 7807 plus:
    - code: short machine code ("dataset-not-found")
    - extras: arbitrary diagnostic payload.
    """

    type: str = "about:blank"
    title: str = ""
    detail: str | None = None
    status: int | None = None
    instance: str = field(default_factory=generate_correlation_id)
    code: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)
```

Then keep your existing `ProblemError`, `SchemaDriftError`, `ValidationError`, etc., but extend taxonomy with a few more domain-specific subclasses:

```python
class ProblemError(Exception):
    def __init__(self, detail: ProblemDetail) -> None:
        self.detail = detail
        super().__init__(detail.detail or detail.title)


class DatasetNotFoundError(ProblemError):
    """Requested dataset could not be located."""

    @classmethod
    def for_name(cls, dataset_name: str) -> "DatasetNotFoundError":
        return cls(
            ProblemDetail(
                type="https://codeintel/problems/dataset-not-found",
                title="Dataset not found",
                detail=f"Dataset {dataset_name!r} is not registered in the catalog.",
                status=404,
                code="dataset-not-found",
                extras={"dataset": dataset_name},
            )
        )


class DatasetSchemaDriftError(ProblemError):
    """Schema drift detected between expected and actual datasets."""


class GraphScopeError(ProblemError):
    """Invalid or unsupported graph scope."""


class GraphFeatureDisabledError(ProblemError):
    """Graph-related feature is disabled in current configuration."""


class BackendTimeoutError(ProblemError):
    """Backend operation exceeded its allowed time budget."""


class ValidationError(ProblemError):
    """Input or configuration validation failure."""
```

(You can wire `DatasetSchemaDriftError` to your existing schema drift logic later.)

---

### 4b.2 Pydantic ProblemDetail wrapper for MCP

Edit `src/codeintel/serving/mcp/models.py` and update the Pydantic `ProblemDetail` to act as a **wrapper** around the domain dataclass.

At the top, import:

```python
from codeintel.serving.services.errors import ProblemDetail as DomainProblemDetail
```

Then redefine Pydantic `ProblemDetail` as:

```python
class ProblemDetail(BaseModel):
    """Problem Details payload for MCP error responses."""

    type: str = Field(default="about:blank")
    title: str
    detail: str | None = None
    status: int | None = None
    instance: str | None = None
    code: str | None = None
    extras: dict[str, object] | None = None

    @classmethod
    def from_domain(cls, detail: DomainProblemDetail) -> "ProblemDetail":
        return cls(
            type=detail.type,
            title=detail.title,
            detail=detail.detail,
            status=detail.status,
            instance=detail.instance,
            code=detail.code,
            extras=detail.extras or {},
        )

    def to_domain(self) -> DomainProblemDetail:
        return DomainProblemDetail(
            type=self.type,
            title=self.title,
            detail=self.detail,
            status=self.status,
            instance=self.instance or "",
            code=self.code,
            extras=dict(self.extras or {}),
        )
```

You can keep your existing MCP `Message`/`ResponseMeta` unchanged; they are orthogonal.

---

### 4b.3 Make `McpError` carry domain ProblemDetail, not Pydantic

Edit `src/codeintel/serving/mcp/errors.py`:

**Before**:

```python
from dataclasses import dataclass

from codeintel.serving.mcp.models import ProblemDetail


@dataclass
class McpError(Exception):
    detail: ProblemDetail
```

**After**:

```python
from dataclasses import dataclass

from codeintel.serving.services.errors import ProblemDetail as DomainProblemDetail
from codeintel.serving.mcp.models import ProblemDetail as ProblemDetailModel


@dataclass
class McpError(Exception):
    """Base MCP error carrying a domain ProblemDetail payload."""

    detail: DomainProblemDetail

    def __str__(self) -> str:
        return self.detail.detail or self.detail.title
```

Update your helper constructors:

```python
def not_found(message: str) -> McpError:
    return McpError(
        detail=DomainProblemDetail(
            type="https://codeintel/problems/not-found",
            title="Resource not found",
            detail=message,
            status=404,
            code="not-found",
        )
    )


def backend_failure(message: str) -> McpError:
    return McpError(
        detail=DomainProblemDetail(
            type="https://codeintel/problems/backend-failure",
            title="Backend failure",
            detail=message,
            status=500,
            code="backend-failure",
        )
    )
```

(And similarly for any other MCP-specific constructors.)

---

### 4b.4 Update `_wrap` to serialize domain ProblemDetail → Pydantic → dict

Edit `src/codeintel/serving/mcp/tool_utils.py`.

**Before**:

```python
from codeintel.serving.mcp import errors

def _wrap(func: Callable[..., object]) -> Callable[..., object]:
    def _inner(*args: object, **kwargs: object) -> object:
        try:
            return func(*args, **kwargs)
        except errors.McpError as exc:
            return {"error": exc.detail.model_dump()}
    return _inner
```

**After**:

```python
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import ProblemDetail as ProblemDetailModel

def _wrap(func: Callable[..., object]) -> Callable[..., object]:
    def _inner(*args: object, **kwargs: object) -> object:
        try:
            return func(*args, **kwargs)
        except errors.McpError as exc:
            # exc.detail is a domain ProblemDetail; convert to Pydantic then dump
            model = ProblemDetailModel.from_domain(exc.detail)
            return {"error": model.model_dump()}
    return _inner
```

---

### 4b.5 HTTP error handlers use domain ProblemDetail and a single JSON conversion

Edit `src/codeintel/serving/http/fastapi.py`.

At the top, import the domain ProblemDetail and optionally the Pydantic wrapper:

```python
from codeintel.serving.services.errors import ProblemDetail as DomainProblemDetail
from codeintel.serving.mcp.models import ProblemDetail as ProblemDetailModel
from codeintel.serving.mcp import errors as mcp_errors
```

Replace `problem_response` to accept **domain** ProblemDetail and convert:

**Before** (simplified):

```python
from codeintel.serving.mcp.models import ProblemDetail

def problem_response(detail: ProblemDetail) -> JSONResponse:
    status_code = detail.status or status.HTTP_500_INTERNAL_SERVER_ERROR
    payload = detail.model_dump()
    payload.setdefault("status", status_code)
    return JSONResponse(status_code=status_code, content=payload)
```

**After**:

```python
def problem_response(detail: DomainProblemDetail) -> JSONResponse:
    """
    Convert a domain ProblemDetail payload into a JSON HTTP response.
    """
    status_code = detail.status or status.HTTP_500_INTERNAL_SERVER_ERROR
    model = ProblemDetailModel.from_domain(detail)
    payload = model.model_dump()
    payload.setdefault("status", status_code)
    return JSONResponse(status_code=status_code, content=payload)
```

Update exception handlers:

**MCP error handler**:

```python
@app.exception_handler(mcp_errors.McpError)
def _handle_mcp_error(
    _request: Request,
    exc: mcp_errors.McpError,
) -> JSONResponse:
    return problem_response(exc.detail)
```

**Validation error handler**:

```python
@app.exception_handler(RequestValidationError)
def _handle_validation_error(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    problem = DomainProblemDetail(
        type="https://codeintel/problems/invalid-request",
        title="Invalid request",
        detail=str(exc),
        status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        code="invalid-request",
        extras={"errors": exc.errors()},
    )
    return problem_response(problem)
```

**Catch-all handler**:

```python
@app.exception_handler(Exception)
def _handle_unexpected(
    _request: Request,
    exc: Exception,
) -> JSONResponse:
    problem = DomainProblemDetail(
        type="https://codeintel/problems/backend-failure",
        title="Backend failure",
        detail=str(exc),
        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="backend-failure",
    )
    return problem_response(problem)
```

Anywhere else you were manually constructing the Pydantic `ProblemDetail` can now use the domain dataclass instead.

---

### 4b.6 Cross-transport tests

Finally, add a few tests to assert that the error codes/types are consistent across:

* LocalQueryService → raises `DatasetNotFoundError` with `.detail.code == "dataset-not-found"`.
* HTTP `/datasets/{name}` returns JSON with `error.code == "dataset-not-found"`.
* MCP `read_dataset_rows(dataset_name="bogus")` returns `{"error": {"code": "dataset-not-found", ...}}`.

I’ll sketch one example for HTTP:

```python
# tests/serving/test_dataset_not_found_errors.py

import pytest
from fastapi.testclient import TestClient

from codeintel.serving.http.fastapi import create_app
from codeintel.config.serving_models import ServingConfig

def test_http_dataset_not_found_returns_problem_detail(tmp_path) -> None:
    cfg = ServingConfig(
        # minimal config for a test repo with no datasets
    )
    app = create_app(cfg)
    client = TestClient(app)

    resp = client.get("/datasets/no_such_dataset")
    assert resp.status_code == 404
    payload = resp.json()
    assert payload["code"] == "dataset-not-found"
    assert payload["type"].endswith("/problems/dataset-not-found")
```

And you can mirror the same pattern in a `tests/mcp/test_error_codes_dataset_not_found.py` using the MCP server.

---

### Net effect of Epic 4

After implementing these steps:

* **Domain vs transport:**

  * `codeintel.serving.domain_models` is the single internal representation.
  * `QueryService` speaks entirely in domain types.
  * HTTP and MCP surfaces convert domain ⇄ Pydantic only at the “edge.”

* **Errors:**

  * A single canonical `ProblemDetail` dataclass with stable `code` and `type`.
  * MCP and HTTP both serialize this via `ProblemDetailModel.from_domain(...)`.
  * Error codes like `"dataset-not-found"`, `"backend-failure"`, `"invalid-request"` are consistent everywhere.

If you’d like, we can next zoom in on **one vertical** (e.g. just the dataset path) and turn this into a precise patch plan with “before → after” code for `LocalQueryService.read_dataset_rows`, `DatasetRowsResponse`, MCP dataset tools, and HTTP dataset routes, so an implementation agent can run straight at it.


# vertical zoom in #

Great, let’s really nail the dataset vertical as a concrete example of Epic 4.

I’ll assume you **haven’t yet** wired domain models/errors for anything else, and walk you through a clean, self-contained patch plan for:

* `DatasetRows` domain model ↔ `DatasetRowsResponse` Pydantic
* `QueryService.read_dataset_rows`
* `_LocalDatasetMixin.read_dataset_rows` (LocalQueryService)
* `_HttpDatasetQueryMixin.read_dataset_rows` (HttpQueryService)
* `DatasetBackendMixin.read_dataset_rows` (MCP backend)
* HTTP `GET /datasets/{dataset_name}` route
* Dataset-not-found error behavior across all three surfaces

I’ll show each piece as:

* **Before** – current Pydantic-centric or error shape
* **After** – domain-centric + unified error

You can apply this as a template to other operations later.

---

## 0. Preconditions (what this plan assumes)

Before you start, add/ensure:

1. `serving/domain_models.py` exists and is importable as `codeintel.serving.domain_models`.
2. `serving/services/errors.py` contains the unified `ProblemDetail` dataclass and `DatasetNotFoundError` subclass (from the previous Epic 4 plan).

For this dataset vertical we only *need*:

### `serving/domain_models.py` (minimal bits)

```python
# src/codeintel/serving/domain_models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    code: str
    severity: str  # "info" | "warning" | "error"
    detail: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseMeta:
    requested_limit: int | None = None
    applied_limit: int | None = None
    requested_offset: int | None = None
    applied_offset: int | None = None
    truncated: bool = False
    messages: list[Message] = field(default_factory=list)


@dataclass
class DatasetRows:
    """Domain representation of dataset rows plus meta."""

    dataset_name: str
    limit: int
    offset: int
    rows: list[dict[str, Any]]
    meta: ResponseMeta
```

If you already added more domain types, keep them; just make sure this is present.

### `serving/services/errors.py` – DatasetNotFoundError

```python
# src/codeintel/serving/services/errors.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


def generate_correlation_id() -> str:
    return str(uuid4())


@dataclass(frozen=True)
class ProblemDetail:
    type: str = "about:blank"
    title: str = ""
    detail: str | None = None
    status: int | None = None
    instance: str = field(default_factory=generate_correlation_id)
    code: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class ProblemError(Exception):
    def __init__(self, detail: ProblemDetail) -> None:
        self.detail = detail
        super().__init__(detail.detail or detail.title)


class DatasetNotFoundError(ProblemError):
    """Requested dataset could not be located."""

    @classmethod
    def for_name(cls, dataset_name: str) -> "DatasetNotFoundError":
        return cls(
            ProblemDetail(
                type="https://codeintel/problems/dataset-not-found",
                title="Dataset not found",
                detail=f"Dataset {dataset_name!r} is not registered in the catalog.",
                status=404,
                code="dataset-not-found",
                extras={"dataset": dataset_name},
            )
        )
```

We’ll use that error from the service layer.

---

## 1. DatasetRowsResponse ⇄ domain conversions

First, teach `DatasetRowsResponse` how to convert to/from the new domain `DatasetRows`.

### 1.1. Pydantic Message / ResponseMeta (if not already wired)

In `src/codeintel/serving/mcp/models.py`:

```python
from codeintel.serving import domain_models as dm
```

**Before** (simplified):

```python
class Message(BaseModel):
    code: str
    severity: Literal["info", "warning", "error"] = "info"
    detail: str | None = None
    context: dict[str, object] | None = None


class ResponseMeta(BaseModel):
    requested_limit: int | None = None
    applied_limit: int | None = None
    requested_offset: int | None = None
    applied_offset: int | None = None
    truncated: bool = False
    messages: list[Message] | None = None
```

**After** (add conversions):

```python
class Message(BaseModel):
    code: str
    severity: Literal["info", "warning", "error"] = "info"
    detail: str | None = None
    context: dict[str, object] | None = None

    def to_domain(self) -> dm.Message:
        return dm.Message(
            code=self.code,
            severity=self.severity,
            detail=self.detail,
            context=dict(self.context or {}),
        )

    @classmethod
    def from_domain(cls, msg: dm.Message) -> "Message":
        return cls(
            code=msg.code,
            severity=msg.severity,
            detail=msg.detail,
            context=msg.context or {},
        )


class ResponseMeta(BaseModel):
    requested_limit: int | None = None
    applied_limit: int | None = None
    requested_offset: int | None = None
    applied_offset: int | None = None
    truncated: bool = False
    messages: list[Message] | None = None

    def to_domain(self) -> dm.ResponseMeta:
        return dm.ResponseMeta(
            requested_limit=self.requested_limit,
            applied_limit=self.applied_limit,
            requested_offset=self.requested_offset,
            applied_offset=self.applied_offset,
            truncated=self.truncated,
            messages=[m.to_domain() for m in (self.messages or [])],
        )

    @classmethod
    def from_domain(cls, meta: dm.ResponseMeta) -> "ResponseMeta":
        return cls(
            requested_limit=meta.requested_limit,
            applied_limit=meta.applied_limit,
            requested_offset=meta.requested_offset,
            applied_offset=meta.applied_offset,
            truncated=meta.truncated,
            messages=[Message.from_domain(m) for m in meta.messages],
        )
```

### 1.2. DatasetRowsResponse.to_domain / from_domain

Find `DatasetRowsResponse` in `mcp/models.py`.

**Before** (conceptual):

```python
class DatasetRowsResponse(BaseModel):
    dataset_name: str
    limit: int
    offset: int
    rows: list[ViewRow]
    meta: ResponseMeta | None = None
```

**After**:

```python
class DatasetRowsResponse(BaseModel):
    dataset_name: str
    limit: int
    offset: int
    rows: list[ViewRow]
    meta: ResponseMeta | None = None

    def to_domain(self) -> dm.DatasetRows:
        return dm.DatasetRows(
            dataset_name=self.dataset_name,
            limit=self.limit,
            offset=self.offset,
            rows=[row.model_dump() for row in self.rows],
            meta=self.meta.to_domain() if self.meta is not None else dm.ResponseMeta(),
        )

    @classmethod
    def from_domain(cls, rows: dm.DatasetRows) -> "DatasetRowsResponse":
        return cls(
            dataset_name=rows.dataset_name,
            limit=rows.limit,
            offset=rows.offset,
            rows=[ViewRow.model_validate(r) for r in rows.rows],
            meta=ResponseMeta.from_domain(rows.meta),
        )
```

---

## 2. QueryService.read_dataset_rows → domain type

In `src/codeintel/serving/services/query_service.py`, the dataset API currently returns Pydantic.

### 2.1. Change the protocol return type

**Before (DatasetQueryApi):**

```python
from codeintel.serving.mcp.models import (
    DatasetDescriptor,
    DatasetRowsResponse,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
    ...
)

class DatasetQueryApi(Protocol):
    def list_datasets(self) -> list[DatasetDescriptor]: ...
    def dataset_specs(self) -> list[DatasetSpecDescriptor]: ...
    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetRowsResponse: ...
    def dataset_schema(...) -> DatasetSchemaResponse: ...
```

**After:**

```python
from codeintel.serving import domain_models as dm
from codeintel.serving.mcp.models import (
    DatasetDescriptor,
    DatasetSpecDescriptor,
    GraphScopePayload,
)

class DatasetQueryApi(Protocol):
    def list_datasets(self) -> list[DatasetDescriptor]: ...
    def dataset_specs(self) -> list[DatasetSpecDescriptor]: ...

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> dm.DatasetRows:
        ...

    # (You can similarly change dataset_schema → dm.DatasetSchema later)
```

`QueryService` (the composite protocol) automatically updates because it inherits `DatasetQueryApi`.

---

## 3. LocalQueryService: Pydantic → domain in _LocalDatasetMixin

Now update `_LocalDatasetMixin.read_dataset_rows` to return `dm.DatasetRows`.

### 3.1. Before (conceptual)

In `src/codeintel/serving/services/datasets.py`:

```python
from codeintel.serving.mcp.models import (
    DatasetDescriptor,
    DatasetRowsResponse,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
    ResponseMeta,
    ViewRow,
)

class _LocalDatasetMixin:
    query: DuckDBQueryService
    dataset_tables: dict[str, str] | None
    describe_dataset_fn: Callable[[str, str], str]
    limits: BackendLimits
    _call: Callable[..., Any]

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        applied_limit = self.query.limits.default_limit if limit is None else limit
        registry = load_dataset_registry(self.query.gateway.con)
        schema_version = None
        if dataset_name in registry.by_name:
            schema_version = registry.by_name[dataset_name].schema_version
        return self._call(
            "read_dataset_rows",
            lambda: self.query.read_dataset_rows(
                dataset_name=dataset_name,
                limit=applied_limit,
                offset=offset,
            ),
            dataset=dataset_name,
            schema_version=schema_version,
        )
```

### 3.2. After: convert to domain

Add the domain import at the top of `services/datasets.py`:

```python
from codeintel.serving import domain_models as dm
from codeintel.serving.services.errors import DatasetNotFoundError
from codeintel.storage.datasets import load_dataset_registry
```

Then change `read_dataset_rows`:

```python
    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> dm.DatasetRows:
        """
        Read dataset rows with clamping and messaging, returning a domain model.
        """
        applied_limit = self.query.limits.default_limit if limit is None else limit
        registry = load_dataset_registry(self.query.gateway.con)

        # Resolve schema_version and raise a domain error if dataset is unknown.
        schema_version = None
        if dataset_name in registry.by_name:
            schema_version = registry.by_name[dataset_name].schema_version
        else:
            # This propagates up through QueryService, HTTP, and MCP
            raise DatasetNotFoundError.for_name(dataset_name)

        # Call the DuckDB-backed Pydantic-layer, then convert to domain.
        pydantic_resp: DatasetRowsResponse = self._call(
            "read_dataset_rows",
            lambda: self.query.read_dataset_rows(
                dataset_name=dataset_name,
                limit=applied_limit,
                offset=offset,
            ),
            dataset=dataset_name,
            schema_version=schema_version,
        )
        return pydantic_resp.to_domain()
```

Now **LocalQueryService.read_dataset_rows** (via this mixin) returns a `dm.DatasetRows`.

---

## 4. HttpQueryService: Pydantic → domain in _HttpDatasetQueryMixin

Similarly, `_HttpDatasetQueryMixin` currently returns Pydantic. We’ll make it return `dm.DatasetRows`.

### 4.1. Before (conceptual)

```python
class _HttpDatasetQueryMixin(_HttpTransportMixin):
    limits: BackendLimits

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        def _run() -> DatasetRowsResponse:
            clamp = clamp_limit_value(...)
            offset_clamp = clamp_offset_value(...)
            if clamp.has_error or offset_clamp.has_error:
                meta = ResponseMeta(...)
                return DatasetRowsResponse(
                    dataset_name=dataset_name,
                    limit=clamp.applied,
                    offset=offset_clamp.applied,
                    rows=[],
                    meta=meta,
                )
            data = self.request_json(
                f"/datasets/{dataset_name}",
                {"limit": clamp.applied, "offset": offset_clamp.applied},
            )
            response = DatasetRowsResponse.model_validate(data)
            # merge messages into meta and return DatasetRowsResponse(...)
        return self._http_call("read_dataset_rows", _run, dataset=dataset_name)
```

### 4.2. After: still build Pydantic, but return domain

In `services/datasets.py`, adjust imports as above, then change `_HttpDatasetQueryMixin.read_dataset_rows`:

```python
class _HttpDatasetQueryMixin(_HttpTransportMixin):
    """HTTP-based dataset query APIs used by HttpQueryService."""

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> dm.DatasetRows:
        def _run() -> DatasetRowsResponse:
            clamp = clamp_limit_value(
                limit,
                default=self.limits.default_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            offset_clamp = clamp_offset_value(offset)
            messages = [*clamp.messages, *offset_clamp.messages]
            if clamp.has_error or offset_clamp.has_error:
                meta = ResponseMeta(
                    requested_limit=limit,
                    applied_limit=clamp.applied,
                    requested_offset=offset,
                    applied_offset=offset_clamp.applied,
                    messages=messages,
                    truncated=False,
                )
                return DatasetRowsResponse(
                    dataset_name=dataset_name,
                    limit=clamp.applied,
                    offset=offset_clamp.applied,
                    rows=[],
                    meta=meta,
                )
            data = self.request_json(
                f"/datasets/{dataset_name}",
                {"limit": clamp.applied, "offset": offset_clamp.applied},
            )
            response = DatasetRowsResponse.model_validate(data)
            existing_meta = response.meta or ResponseMeta()
            merged_meta = ResponseMeta(
                requested_limit=limit,
                applied_limit=clamp.applied,
                requested_offset=offset,
                applied_offset=offset_clamp.applied,
                truncated=existing_meta.truncated,
                messages=[*messages, *(existing_meta.messages or [])],
            )
            return response.model_copy(update={"meta": merged_meta})

        pydantic_resp: DatasetRowsResponse = self._http_call(
            "read_dataset_rows",
            _run,
            dataset=dataset_name,
        )
        return pydantic_resp.to_domain()
```

So for HTTP services too, the **service** layer returns `dm.DatasetRows`.

---

## 5. MCP backend: domain → Pydantic at the edge

`QueryBackend` methods must still return Pydantic for MCP tools; we’ll make them wrap the domain.

### 5.1. DatasetBackendMixin.read_dataset_rows

In `src/codeintel/serving/mcp/backend.py`, find the dataset mixin; conceptually it looks like:

**Before:**

```python
from codeintel.serving.mcp.models import DatasetRowsResponse

class DatasetBackendMixin:
    service: QueryService

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        return self.service.read_dataset_rows(
            dataset_name=dataset_name,
            limit=limit,
            offset=offset,
        )
```

**After:**

```python
from codeintel.serving import domain_models as dm
from codeintel.serving.mcp.models import DatasetRowsResponse

class DatasetBackendMixin:
    service: QueryService

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        domain_rows: dm.DatasetRows = self.service.read_dataset_rows(
            dataset_name=dataset_name,
            limit=limit,
            offset=offset,
        )
        return DatasetRowsResponse.from_domain(domain_rows)
```

MCP dataset tools (which call `backend.read_dataset_rows(...)` and `.model_dump()`) don’t need to change for the domain split; they’re already spec-driven and Pydantic-facing from Epic 3.

---

## 6. HTTP dataset route: domain → Pydantic in the router

Now wire this into the HTTP route that exposes `/datasets/{dataset_name}`.

### 6.1. Before

In `src/codeintel/serving/http/routes/datasets.py`:

```python
from codeintel.serving.http.dependencies import ServiceDep
from codeintel.serving.mcp.models import DatasetDescriptor, DatasetRowsResponse, DatasetSchemaResponse

@router.get(
    "/datasets/{dataset_name}",
    response_model=DatasetRowsResponse,
    summary="Read dataset rows.",
)
def read_dataset_rows(
    dataset_name: str,
    limit: int | None = None,
    offset: int = 0,
    service: ServiceDep,
) -> DatasetRowsResponse:
    detail = service.read_dataset_rows(
        dataset_name=dataset_name,
        limit=limit,
        offset=offset,
    )
    LOG.info("Returned rows for dataset=%s", dataset_name)
    return detail
```

### 6.2. After: wrap domain → Pydantic

Add domain import:

```python
from codeintel.serving import domain_models as dm
from codeintel.serving.mcp.models import DatasetRowsResponse
```

Then change the handler:

```python
    @router.get(
        "/datasets/{dataset_name}",
        response_model=DatasetRowsResponse,
        summary="Read dataset rows.",
    )
    def read_dataset_rows(
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
        service: ServiceDep = ServiceDep,
    ) -> DatasetRowsResponse:
        """
        Read rows and metadata for a dataset, applying serving-layer limits.
        """
        domain_rows: dm.DatasetRows = service.read_dataset_rows(
            dataset_name=dataset_name,
            limit=limit,
            offset=offset,
        )
        detail = DatasetRowsResponse.from_domain(domain_rows)
        LOG.info("Returned rows for dataset=%s", dataset_name)
        return detail
```

Combine this with your `/meta/datasets` introspection and the behavior stays consistent.

---

## 7. Dataset not found: consistent error across Local / HTTP / MCP

With `DatasetNotFoundError.for_name(dataset_name)` raised in `_LocalDatasetMixin.read_dataset_rows` (and eventually in the HTTP variant if you choose), you get:

* **Local QueryService** callers: directly see `DatasetNotFoundError` with `detail.code == "dataset-not-found"`.
* **HTTP**:

  * Your FastAPI exception handler for `ProblemError` (see previous Epic 4 plan) converts `exc.detail` → Pydantic `ProblemDetail` → JSON via `problem_response`.
  * Resulting JSON has `{"code": "dataset-not-found", "type": "https://codeintel/problems/dataset-not-found", ...}`.
* **MCP**:

  * If backend/service raises `DatasetNotFoundError`, you can catch it in MCP-specific code and wrap it in `McpError(detail=exc.detail)`; `_wrap` then uses `ProblemDetailModel.from_domain` to send `{"error": {...}}` with the same `code` and `type`.

If you want to be very explicit in MCP, you can update the dataset tool generator to translate `DatasetNotFoundError` to `McpError`:

```python
from codeintel.serving.services.errors import DatasetNotFoundError
from codeintel.serving.mcp import errors as mcp_errors

@_wrap
def _tool(**kwargs: Any) -> dict[str, object] | dict[str, ProblemDetail]:
    try:
        response = backend_method(**kwargs)
    except DatasetNotFoundError as exc:
        # Wrap in MCP-specific error for consistency
        raise mcp_errors.McpError(exc.detail) from exc
    return response.model_dump()
```

…but if the service layer already throws `McpError` for dataset not found in MCP context, you can centralize that there instead.

---

## 8. Quick checklist for the agent

1. **Add / confirm** `DatasetRows` in `domain_models.py`.
2. Add `to_domain` / `from_domain` to `Message`, `ResponseMeta`, and `DatasetRowsResponse` in `mcp/models.py`.
3. Change `DatasetQueryApi.read_dataset_rows` to return `dm.DatasetRows`.
4. In `_LocalDatasetMixin.read_dataset_rows`:

   * Import `domain_models` + `DatasetNotFoundError`.
   * Raise `DatasetNotFoundError.for_name` if dataset not in registry.
   * Wrap `DuckDBQueryService.read_dataset_rows` result, call `.to_domain()`.
5. In `_HttpDatasetQueryMixin.read_dataset_rows`:

   * Keep limit/offset logic identical.
   * After `self._http_call`, call `.to_domain()` and return domain.
6. In `DatasetBackendMixin.read_dataset_rows`:

   * Call `service.read_dataset_rows(...)` to get domain.
   * Return `DatasetRowsResponse.from_domain(domain_rows)`.
7. In HTTP route `/datasets/{dataset_name}`:

   * Call `service.read_dataset_rows(...)` → domain.
   * Wrap with `DatasetRowsResponse.from_domain(domain_rows)` and return.
8. Ensure unified error handling is in place so `DatasetNotFoundError` yields:

   * Python exception with `code="dataset-not-found"` locally.
   * HTTP ProblemDetail JSON with the same code & type.
   * MCP `{"error": {...}}` with the same code & type.

Once this vertical is in place and tests are green, you can replicate the exact same pattern across:

* `dataset_schema` (DatasetSchema domain ↔ DatasetSchemaResponse).
* Subsystems, functions, profiles, etc.

If you’d like, next we can pick **dataset_schema** specifically and walk through the same “before → after” plan for its service, backend, MCP tools, and HTTP route.


# dataset schema zoom in #

Perfect, let’s finish the dataset story by doing for **`dataset_schema`** what we just did for `read_dataset_rows`: make it **domain-first** end-to-end, with consistent errors across Local, HTTP, and MCP.

I’ll walk through concrete “before → after” patches for:

1. `DatasetSchema` domain model + Pydantic `DatasetSchemaResponse` conversions
2. `QueryService.dataset_schema` signature
3. `_LocalDatasetMixin.dataset_schema` (LocalQueryService)
4. `_HttpDatasetQueryMixin.dataset_schema` (HttpQueryService)
5. `DatasetBackendMixin.dataset_schema` (MCP backend)
6. HTTP route `GET /datasets/{dataset_name}/schema`
7. Dataset-not-found behavior consistency

You can apply these mechanically.

---

## 0. Preconditions

I’ll assume you already have from earlier steps:

* `codeintel.serving.domain_models` with:

  * `ResponseMeta`
  * `Message`
* `codeintel.serving.services.errors.DatasetNotFoundError` with `code="dataset-not-found"`.

If not, you can treat the snippets below as authoritative for those parts too.

---

## 1. Domain model for dataset schema + Pydantic conversions

### 1.1. Add `DatasetSchema` to `domain_models.py`

Open `src/codeintel/serving/domain_models.py` and add:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    code: str
    severity: str
    detail: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseMeta:
    requested_limit: int | None = None
    applied_limit: int | None = None
    requested_offset: int | None = None
    applied_offset: int | None = None
    truncated: bool = False
    messages: list[Message] = field(default_factory=list)


@dataclass
class DatasetSchema:
    """Domain representation of a dataset schema + samples."""

    dataset_name: str
    table_key: str
    duckdb_schema: list[dict[str, Any]]
    json_schema: dict[str, Any] | None
    sample_rows: list[dict[str, Any]]
    capabilities: dict[str, bool]
    owner: str | None
    freshness_sla: str | None
    retention_policy: str | None
    schema_version: str | None
    stable_id: str | None
    validation_profile: str | None
    meta: ResponseMeta | None = None
```

(You may already have `DatasetRows` there; just add `DatasetSchema` alongside.)

### 1.2. Teach `DatasetSchemaResponse` how to convert

Open `src/codeintel/serving/mcp/models.py` and ensure you import domain models:

```python
from codeintel.serving import domain_models as dm
```

You likely already have `DatasetSchemaColumn`, `ViewRow`, and `DatasetSchemaResponse` defined.

**Before** (conceptual):

```python
class DatasetSchemaColumn(BaseModel):
    name: str
    type: str
    nullable: bool


class DatasetSchemaResponse(BaseModel):
    dataset: str
    table_key: str
    duckdb_schema: list[DatasetSchemaColumn]
    json_schema: dict[str, object] | None = None
    sample_rows: list[ViewRow] = []
    capabilities: dict[str, bool] = {}
    owner: str | None = None
    freshness_sla: str | None = None
    retention_policy: str | None = None
    schema_version: str | None = None
    stable_id: str | None = None
    validation_profile: str | None = None
    meta: ResponseMeta | None = None
```

**After**: add `to_domain` / `from_domain`:

```python
class DatasetSchemaResponse(BaseModel):
    dataset: str
    table_key: str
    duckdb_schema: list[DatasetSchemaColumn]
    json_schema: dict[str, object] | None = None
    sample_rows: list[ViewRow] = []
    capabilities: dict[str, bool] = {}
    owner: str | None = None
    freshness_sla: str | None = None
    retention_policy: str | None = None
    schema_version: str | None = None
    stable_id: str | None = None
    validation_profile: str | None = None
    meta: ResponseMeta | None = None

    def to_domain(self) -> dm.DatasetSchema:
        return dm.DatasetSchema(
            dataset_name=self.dataset,
            table_key=self.table_key,
            duckdb_schema=[col.model_dump() for col in self.duckdb_schema],
            json_schema=self.json_schema,
            sample_rows=[row.model_dump() for row in self.sample_rows],
            capabilities=dict(self.capabilities or {}),
            owner=self.owner,
            freshness_sla=self.freshness_sla,
            retention_policy=self.retention_policy,
            schema_version=self.schema_version,
            stable_id=self.stable_id,
            validation_profile=self.validation_profile,
            meta=self.meta.to_domain() if self.meta is not None else None,
        )

    @classmethod
    def from_domain(cls, schema: dm.DatasetSchema) -> "DatasetSchemaResponse":
        return cls(
            dataset=schema.dataset_name,
            table_key=schema.table_key,
            duckdb_schema=[
                DatasetSchemaColumn.model_validate(c) for c in schema.duckdb_schema
            ],
            json_schema=schema.json_schema,
            sample_rows=[ViewRow.model_validate(r) for r in schema.sample_rows],
            capabilities=schema.capabilities,
            owner=schema.owner,
            freshness_sla=schema.freshness_sla,
            retention_policy=schema.retention_policy,
            schema_version=schema.schema_version,
            stable_id=schema.stable_id,
            validation_profile=schema.validation_profile,
            meta=(
                ResponseMeta.from_domain(schema.meta)
                if schema.meta is not None
                else None
            ),
        )
```

We’re using `dataset` in the Pydantic model to carry the domain’s `dataset_name`.

---

## 2. QueryService.dataset_schema → domain

Open `src/codeintel/serving/services/query_service.py`.

### 2.1. Import domain models

At the top:

```python
from codeintel.serving import domain_models as dm
```

You already changed `DatasetQueryApi.read_dataset_rows`; now change `dataset_schema`.

**Before (protocol):**

```python
from codeintel.serving.mcp.models import (
    DatasetDescriptor,
    DatasetRowsResponse,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
    ...
)

class DatasetQueryApi(Protocol):
    def dataset_schema(
        self,
        *,
        dataset_name: str,
        sample_limit: int = 5,
    ) -> DatasetSchemaResponse:
        ...
```

**After:**

```python
from codeintel.serving.mcp.models import (
    DatasetDescriptor,
    DatasetSpecDescriptor,
    GraphScopePayload,
)

class DatasetQueryApi(Protocol):
    # ...

    def dataset_schema(
        self,
        *,
        dataset_name: str,
        sample_limit: int = 5,
    ) -> dm.DatasetSchema:
        ...
```

`QueryService` automatically picks that up.

---

## 3. LocalQueryService: `_LocalDatasetMixin.dataset_schema`

Open `src/codeintel/serving/services/datasets.py`.

You should already have `_LocalDatasetMixin` with a `dataset_schema` method that returns Pydantic; we’ll switch it to domain.

### 3.1. Imports

At the top of `services/datasets.py`, ensure:

```python
from codeintel.serving import domain_models as dm
from codeintel.serving.mcp.models import (
    DatasetDescriptor,
    DatasetRowsResponse,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
    ResponseMeta,
    ViewRow,
)
from codeintel.serving.services.errors import DatasetNotFoundError
from codeintel.storage.datasets import Dataset, load_dataset_registry
```

### 3.2. Before: Pydantic-returning `dataset_schema`

Current implementation probably looks like:

```python
class _LocalDatasetMixin:
    # ...

    def dataset_schema(
        self,
        *,
        dataset_name: str,
        sample_limit: int = 5,
    ) -> DatasetSchemaResponse:
        """
        Return DuckDB + JSON Schema details and sample rows for a dataset.
        """
        def _schema() -> DatasetSchemaResponse:
            return self.query.dataset_schema(
                dataset_name=dataset_name,
                sample_limit=sample_limit,
            )

        registry = load_dataset_registry(self.query.gateway.con)
        schema_version = None
        if dataset_name in registry.by_name:
            schema_version = registry.by_name[dataset_name].schema_version

        return self._call(
            "dataset_schema",
            _schema,
            dataset=dataset_name,
            schema_version=schema_version,
        )
```

### 3.3. After: domain + DatasetNotFoundError

Change it to:

```python
    def dataset_schema(
        self,
        *,
        dataset_name: str,
        sample_limit: int = 5,
    ) -> dm.DatasetSchema:
        """
        Return DuckDB + JSON Schema details and sample rows for a dataset, as a domain model.

        Raises
        ------
        DatasetNotFoundError
            If the dataset is not present in the registry.
        """
        registry = load_dataset_registry(self.query.gateway.con)
        schema_version = None
        if dataset_name in registry.by_name:
            schema_version = registry.by_name[dataset_name].schema_version
        else:
            # unify "dataset not found" behavior at the domain layer
            raise DatasetNotFoundError.for_name(dataset_name)

        def _schema() -> DatasetSchemaResponse:
            return self.query.dataset_schema(
                dataset_name=dataset_name,
                sample_limit=sample_limit,
            )

        pydantic_resp: DatasetSchemaResponse = self._call(
            "dataset_schema",
            _schema,
            dataset=dataset_name,
            schema_version=schema_version,
        )
        return pydantic_resp.to_domain()
```

So:

* Local service now returns `dm.DatasetSchema`.
* It raises `DatasetNotFoundError` if the registry has no such dataset, giving you the same error across surfaces.

---

## 4. HttpQueryService: `_HttpDatasetQueryMixin.dataset_schema`

Still in `services/datasets.py`, `_HttpDatasetQueryMixin` currently returns Pydantic. We’ll make it return domain.

### 4.1. Before

Something like:

```python
class _HttpDatasetQueryMixin(_HttpTransportMixin):
    def dataset_schema(
        self,
        *,
        dataset_name: str,
        sample_limit: int = 5,
    ) -> DatasetSchemaResponse:
        def _run() -> DatasetSchemaResponse:
            data = self.request_json(
                f"/datasets/{dataset_name}/schema",
                {"limit": sample_limit},
            )
            return DatasetSchemaResponse.model_validate(data)

        return self._http_call("dataset_schema", _run, dataset=dataset_name)
```

### 4.2. After: domain-returning

```python
class _HttpDatasetQueryMixin(_HttpTransportMixin):
    """HTTP-based dataset query APIs used by HttpQueryService."""

    def dataset_schema(
        self,
        *,
        dataset_name: str,
        sample_limit: int = 5,
    ) -> dm.DatasetSchema:
        def _run() -> DatasetSchemaResponse:
            data = self.request_json(
                f"/datasets/{dataset_name}/schema",
                {"limit": sample_limit},
            )
            return DatasetSchemaResponse.model_validate(data)

        pydantic_resp: DatasetSchemaResponse = self._http_call(
            "dataset_schema",
            _run,
            dataset=dataset_name,
        )
        return pydantic_resp.to_domain()
```

Here, HTTP errors (404 ProblemDetail, etc.) are still handled centrally in `_HttpTransportMixin` / FastAPI; when the call succeeds, the service layer works in terms of `dm.DatasetSchema`.

---

## 5. MCP backend: `DatasetBackendMixin.dataset_schema`

Now we keep MCP-facing backend returning **Pydantic** by wrapping the domain type from `QueryService`.

Open `src/codeintel/serving/mcp/backend.py`.

### 5.1. Imports

At the top:

```python
from codeintel.serving import domain_models as dm
from codeintel.serving.mcp.models import DatasetSchemaResponse, DatasetRowsResponse, ...
```

### 5.2. Before

`DatasetBackendMixin` probably looks like:

```python
class DatasetBackendMixin:
    service: QueryService

    def dataset_schema(
        self,
        *,
        dataset_name: str,
        sample_limit: int = 5,
    ) -> DatasetSchemaResponse:
        return self.service.dataset_schema(
            dataset_name=dataset_name,
            sample_limit=sample_limit,
        )
```

### 5.3. After: domain → Pydantic

```python
class DatasetBackendMixin:
    service: QueryService

    def dataset_schema(
        self,
        *,
        dataset_name: str,
        sample_limit: int = 5,
    ) -> DatasetSchemaResponse:
        domain_schema: dm.DatasetSchema = self.service.dataset_schema(
            dataset_name=dataset_name,
            sample_limit=sample_limit,
        )
        return DatasetSchemaResponse.from_domain(domain_schema)
```

MCP dataset tools (now spec-driven) call `backend.dataset_schema(...)` and get Pydantic as before; nothing else changes on the MCP surface.

---

## 6. HTTP route: `/datasets/{dataset_name}/schema`

Now wire domain → Pydantic at the HTTP edge.

Open `src/codeintel/serving/http/routes/datasets.py`.

### 6.1. Imports

Make sure you have:

```python
from fastapi import APIRouter

from codeintel.serving import domain_models as dm
from codeintel.serving.http.dependencies import ServiceDep
from codeintel.serving.mcp.models import (
    DatasetDescriptor,
    DatasetRowsResponse,
    DatasetSchemaResponse,
)
```

### 6.2. Before

Your route probably looks like:

```python
@router.get(
    "/datasets/{dataset_name}/schema",
    response_model=DatasetSchemaResponse,
    summary="Describe dataset schema and sample rows.",
)
def dataset_schema(
    dataset_name: str,
    limit: int = 5,
    service: ServiceDep,
) -> DatasetSchemaResponse:
    return service.dataset_schema(dataset_name=dataset_name, sample_limit=limit)
```

### 6.3. After: service returns domain, router converts to Pydantic

```python
@router.get(
    "/datasets/{dataset_name}/schema",
    response_model=DatasetSchemaResponse,
    summary="Describe dataset schema and sample rows.",
)
def dataset_schema(
    dataset_name: str,
    limit: int = 5,
    service: ServiceDep,
) -> DatasetSchemaResponse:
    """
    Describe a dataset by combining DuckDB schema, JSON Schema, and sample rows.
    """
    domain_schema: dm.DatasetSchema = service.dataset_schema(
        dataset_name=dataset_name,
        sample_limit=limit,
    )
    return DatasetSchemaResponse.from_domain(domain_schema)
```

Now the HTTP surface stays identical (same JSON shape, same OpenAPI), while the service layer is fully domain-based.

---

## 7. Dataset-not-found behavior across surfaces

With the changes above:

* `_LocalDatasetMixin.dataset_schema` raises `DatasetNotFoundError.for_name(dataset_name)` if the dataset isn’t registered.
* `HttpQueryService.dataset_schema` will see whatever HTTP error your FastAPI route emits; once you wire HTTP to raise domain `ProblemError`/`DatasetNotFoundError` as well, the behavior is fully uniform.
* MCP backend wraps `service.dataset_schema` and thus will also see `DatasetNotFoundError`.

To complete the cross-transport consistency, make sure:

1. Your **FastAPI exception handlers** in `http/fastapi.py` convert `ProblemError` (including `DatasetNotFoundError`) to a JSON ProblemDetail using the unified `ProblemDetail` dataclass and `ProblemDetailModel.from_domain(...)`, as described earlier.

2. Your **MCP `_wrap`** in `mcp/tool_utils.py` converts `McpError(detail=<domain ProblemDetail>)` to:

   ```json
   {"error": { "code": "dataset-not-found", "type": "https://codeintel/problems/dataset-not-found", ... }}
   ```

   by using the Pydantic `ProblemDetailModel.from_domain(...)`.

If you want MCP to be explicit about wrapping `DatasetNotFoundError` as `McpError`, you can augment the dataset tool generator:

```python
from codeintel.serving.services.errors import DatasetNotFoundError
from codeintel.serving.mcp import errors as mcp_errors

@_wrap
def _tool(**kwargs: Any) -> dict[str, object] | dict[str, ProblemDetail]:
    try:
        response = backend_method(**kwargs)
    except DatasetNotFoundError as exc:
        raise mcp_errors.McpError(exc.detail) from exc
    return response.model_dump()
```

…but often it’s cleaner to have the `QueryBackend` or service layer decide when to raise `McpError` vs `DatasetNotFoundError`.

---

## 8. Quick “agent checklist” for `dataset_schema` vertical

1. Add `DatasetSchema` to `domain_models.py`.
2. In `mcp/models.py`, implement `DatasetSchemaResponse.to_domain` / `.from_domain`.
3. Change `DatasetQueryApi.dataset_schema` to return `dm.DatasetSchema`.
4. In `_LocalDatasetMixin.dataset_schema`:

   * Check registry; raise `DatasetNotFoundError.for_name(dataset_name)` if missing.
   * Call `query.dataset_schema`, wrap result via `.to_domain()`.
5. In `_HttpDatasetQueryMixin.dataset_schema`:

   * Fetch Pydantic response via HTTP, `_http_call`.
   * Return `pydantic_resp.to_domain()`.
6. In `DatasetBackendMixin.dataset_schema`:

   * Call `service.dataset_schema` to get domain.
   * Return `DatasetSchemaResponse.from_domain(domain_schema)`.
7. In HTTP `/datasets/{dataset_name}/schema` route:

   * Call `service.dataset_schema` (domain).
   * Convert to `DatasetSchemaResponse.from_domain(domain_schema)` and return.

Once that’s in, your **dataset** vertical is genuinely domain-first and transport-agnostic, with unified errors and Pydantic just at the edges. From there, you can rinse-and-repeat the pattern for functions, subsystems, and profiles as aggressively as you like.
