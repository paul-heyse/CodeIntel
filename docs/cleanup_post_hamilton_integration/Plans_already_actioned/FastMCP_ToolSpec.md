

Below is a concrete, “drop‑in spec” for **(1) your MCP tool surface** (with **compat + typed** variants) and **(2) a clean, consistent resource URI taxonomy** that LLM agents can discover and use predictably.

I’m basing the tool list on the **6 tools currently in `src/codeintel/serving/mcp/app.py`**, which your FastMCP/Uvicorn plan already treats as the canonical MCP surface (and calls out converting “all 6 tools” to async + ctx). 

---

## 1) Tool API surface: compat + typed (recommended canonical)

### Design goals

1. **Keep existing callers working** (compat tools preserve the “flat args” shape).
2. Provide a **typed, schema-rich surface** for best LLM UX (Pydantic request/response models).
3. Make “big results” **resource-first** (tools return handles/URIs; the client reads resources lazily).

Also: every tool should be annotated as read-only so clients behave correctly; FastMCP supports MCP annotations like `readOnlyHint` that affect client behavior. 

---

## 2) Tool naming conventions

### Canonical naming

Keep the existing names as **compat** and introduce `*_typed` as the **canonical** (best) variants:

* `semantic_catalog` + `semantic_catalog_typed`
* `semantic_describe` + `semantic_describe_typed`
* `semantic_query` + `semantic_query_typed`
* `semantic_explain` + `semantic_explain_typed`
* `serving_meta` + `serving_meta_typed`
* `code_search` + `code_search_typed`

Optional but strongly recommended if you adopt exports/resources:

* `semantic_export` + `semantic_export_typed` (returns export handle(s) + resource URIs)

This keeps the “obvious” names stable while giving LLMs a more structured option.

---

## 3) Exact tool signatures

### Shared hints/tags (use consistently)

**Annotations** (read-only, deterministic, local):

```python
_READ_ONLY_ANNOTATIONS = {
    "readOnlyHint": True,
    "idempotentHint": True,
    "openWorldHint": False,
}
```

FastMCP explicitly supports these MCP annotations for guiding client-side behavior. 

**Tags** (simple, discoverable buckets):

* `semantic`, `search`, `meta`, `export`
* `read`
* `typed`, `compat`

(So you can later include/exclude tool subsets cleanly.)

---

### A) `semantic_catalog`

#### Compat

```python
@mcp.tool(name="semantic_catalog", annotations=_READ_ONLY_ANNOTATIONS, tags=["semantic", "read", "compat"])
async def semantic_catalog(ctx: Context) -> dict[str, object]:
    ...
```

#### Typed (canonical)

```python
@mcp.tool(name="semantic_catalog_typed", annotations=_READ_ONLY_ANNOTATIONS, tags=["semantic", "read", "typed"])
async def semantic_catalog_typed(ctx: Context) -> SemanticCatalogResponse:
    ...
```

**Behavior**: Returns a catalog of semantic views. Each view entry should include its canonical **resource URI** (see taxonomy below), so agents can jump directly to `codeintel://semantic/views/{view_id}` without guessing.

---

### B) `semantic_describe`

#### Compat

```python
@mcp.tool(name="semantic_describe", annotations=_READ_ONLY_ANNOTATIONS, tags=["semantic", "read", "compat"])
async def semantic_describe(view_id: str, ctx: Context) -> dict[str, object]:
    ...
```

#### Typed (canonical)

```python
@mcp.tool(name="semantic_describe_typed", annotations=_READ_ONLY_ANNOTATIONS, tags=["semantic", "read", "typed"])
async def semantic_describe_typed(view_id: str, ctx: Context) -> SemanticViewDescriptionResponse:
    ...
```

**Behavior**: Describe schema + metadata, and include:

* `resource_uri`: `codeintel://semantic/views/{view_id}`
* `schema_uri`: `codeintel://semantic/views/{view_id}/schema`

---

### C) `semantic_query`

#### Compat (keep your current flat shape)

Keep your existing flat signature shape for maximal compatibility (your plan already shows this signature form). 

```python
@mcp.tool(name="semantic_query", annotations=_READ_ONLY_ANNOTATIONS, tags=["semantic", "read", "compat"])
async def semantic_query(
    view_id: str,
    filters: list[dict[str, object]] | None = None,
    select: list[str] | None = None,
    order_by: list[str] | None = None,
    pagination: dict[str, int] | None = None,
    ctx: Context,
) -> dict[str, object]:
    ...
```

#### Typed (canonical)

```python
@mcp.tool(name="semantic_query_typed", annotations=_READ_ONLY_ANNOTATIONS, tags=["semantic", "read", "typed"])
async def semantic_query_typed(request: SemanticQueryRequest, ctx: Context) -> SemanticQueryResponse:
    ...
```

**Important best-in-class tweak**: add a *resource-first escape hatch* for big answers:

* If the result is large, return **a small preview** in `rows` (or none), and include:

  * `export_uri` (resource) pointing to the full result.
  * `export_meta_uri` for row count, schema, etc.

That prevents “LLM context blowups” and aligns with FastMCP’s intended pattern of using resources for large payloads. 

---

### D) `semantic_explain`

#### Compat

```python
@mcp.tool(name="semantic_explain", annotations=_READ_ONLY_ANNOTATIONS, tags=["semantic", "read", "compat"])
async def semantic_explain(
    view_id: str,
    filters: list[dict[str, object]] | None = None,
    select: list[str] | None = None,
    order_by: list[str] | None = None,
    pagination: dict[str, int] | None = None,
    ctx: Context,
) -> dict[str, object]:
    ...
```

#### Typed (canonical)

```python
@mcp.tool(name="semantic_explain_typed", annotations=_READ_ONLY_ANNOTATIONS, tags=["semantic", "read", "typed"])
async def semantic_explain_typed(request: SemanticQueryRequest, ctx: Context) -> SemanticExplainResponse:
    ...
```

Optional: include `compiled_sql_uri` resource: `codeintel://exports/{token}/sql` for easy retrieval.

---

### E) `serving_meta`

#### Compat

```python
@mcp.tool(name="serving_meta", annotations=_READ_ONLY_ANNOTATIONS, tags=["meta", "read", "compat"])
async def serving_meta(ctx: Context) -> dict[str, object]:
    ...
```

#### Typed (canonical)

Create a small Pydantic model (recommended) e.g.:

```python
class ServingMetaResponse(BaseModel):
    repo: str
    commit: str
    run_id: str
    published_at: str
    semantic_layer_version: str
    buildspec_hash: str
    buildspec_version: str
    schema_inventory: dict[str, int]
    resource_templates: list[dict[str, str]]  # see below
```

Then:

```python
@mcp.tool(name="serving_meta_typed", annotations=_READ_ONLY_ANNOTATIONS, tags=["meta", "read", "typed"])
async def serving_meta_typed(ctx: Context) -> ServingMetaResponse:
    ...
```

**Best-in-class addition**: `resource_templates` (a list of the URI templates below) so the LLM can discover everything without reading docs.

---

### F) `code_search`

#### Compat

```python
@mcp.tool(name="code_search", annotations=_READ_ONLY_ANNOTATIONS, tags=["search", "read", "compat"])
async def code_search(
    query: str,
    kinds: list[str] | None = None,
    limit: int = 20,
    offset: int = 0,
    ctx: Context,
) -> dict[str, object]:
    ...
```

#### Typed (canonical)

```python
@mcp.tool(name="code_search_typed", annotations=_READ_ONLY_ANNOTATIONS, tags=["search", "read", "typed"])
async def code_search_typed(request: SearchQueryRequest, ctx: Context) -> SearchQueryResponse:
    ...
```

---

### G) (Recommended) `semantic_export` tool for resource-based large results

This is the missing piece that makes the `codeintel://exports/{token}` resource namespace real.

#### Typed (canonical)

```python
class ExportHandleResponse(BaseModel):
    export_id: str
    uri: str              # codeintel://exports/{export_id}
    meta_uri: str         # codeintel://exports/{export_id}/meta
    mime_type: str
    filename: str
    snapshot: dict[str, str]
    expires_at: str | None = None

@mcp.tool(name="semantic_export_typed", annotations=_READ_ONLY_ANNOTATIONS, tags=["export", "read", "typed"])
async def semantic_export_typed(request: SemanticExportRequest, ctx: Context) -> ExportHandleResponse:
    ...
```

#### Compat

```python
@mcp.tool(name="semantic_export", annotations=_READ_ONLY_ANNOTATIONS, tags=["export", "read", "compat"])
async def semantic_export(
    view_id: str,
    filters: list[dict[str, object]] | None = None,
    select: list[str] | None = None,
    order_by: list[str] | None = None,
    format: str = "ndjson",
    limit: int = 100_000,
    offset: int = 0,
    ctx: Context,
) -> dict[str, object]:
    ...
```

---

## 4) Resource URI taxonomy (exact, consistent, “discoverable”)

FastMCP supports both **static resources** and **parameterized URI templates** (RFC 6570-style), e.g. `@mcp.resource("data://user/{user_id}")`. 
It also supports returning JSON, text, bytes, or a `ResourceContent` object with explicit MIME type + metadata. 

### Top-level rules

* Use one scheme: **`codeintel://`**
* Use the “authority” (host) segment as the **namespace**:

  * `codeintel://semantic/...`
  * `codeintel://exports/...`
  * `codeintel://meta/...`
  * `codeintel://build/...`
  * `codeintel://schema/...`

This keeps URIs short, readable, and clusterable.

---

### A) Semantic layer resources

#### 1) Catalog

* **URI**: `codeintel://semantic/views`
* **Returns**: `SemanticCatalogResponse` (JSON)

#### 2) View description (schema + metadata)

* **URI template**: `codeintel://semantic/views/{view_id}`
* **Returns**: `SemanticViewDescriptionResponse` (JSON)

#### 3) View schema only (small, easy to cache)

* **URI template**: `codeintel://semantic/views/{view_id}/schema`
* **Returns**:

  ```json
  { "view_id": "...", "table_key": "...", "column_types": { "col": "TYPE", ... }, "snapshot": {...} }
  ```

#### 4) View spec (the semantic registry “source of truth” object)

* **URI template**: `codeintel://semantic/views/{view_id}/spec`
* **Returns**: `SemanticViewSpec` (JSON)

---

### B) Export/result resources

#### 1) Export payload (bytes or JSON depending on format)

* **URI template**: `codeintel://exports/{export_id}`
* **Returns**:

  * `ResourceContent(content=<bytes>, mime_type="application/vnd.apache.parquet", meta={...})`
  * or JSON if you choose JSON format

#### 2) Export metadata

* **URI template**: `codeintel://exports/{export_id}/meta`
* **Returns**:

  ```json
  {
    "export_id": "...",
    "view_id": "...",
    "format": "ndjson|parquet|arrow|json",
    "mime_type": "...",
    "filename": "...",
    "row_count": 12345,
    "bytes": 999999,
    "created_at": "...",
    "expires_at": "...",
    "snapshot": { "repo": "...", "commit": "...", "run_id": "..." }
  }
  ```

#### 3) Export preview (safe for LLM context)

* **URI template**: `codeintel://exports/{export_id}/preview`
* **Returns**:

  ```json
  { "export_id": "...", "columns": [...], "rows": [... up to N ...], "truncated": true }
  ```

#### 4) Export compiled SQL (for debugging / citations)

* **URI template**: `codeintel://exports/{export_id}/sql`
* **Returns**: text/plain (compiled SQL)

---

### C) Meta / inventory resources

These make the server self-documenting.

#### 1) Serving meta (same as `serving_meta_typed`)

* **URI**: `codeintel://meta/serving`

#### 2) Resource templates catalog (discoverability)

* **URI**: `codeintel://meta/resources`
* **Returns** a JSON list of templates + descriptions, e.g.:

  ```json
  [
    {"uri": "codeintel://semantic/views", "description": "List semantic views"},
    {"uri": "codeintel://semantic/views/{view_id}", "description": "Describe semantic view"},
    {"uri": "codeintel://exports/{export_id}", "description": "Read export payload"},
    ...
  ]
  ```

---

### D) Build/spec/schema resources (optional but extremely useful)

If you already have BuildSpec + schema manifest in your serving snapshot, these are “free wins” for agent workflows.

* `codeintel://build/spec` → BuildSpec JSON (or a redacted version)
* `codeintel://schema/manifest` → schema manifest JSON (table_key → column_types/hash)
* `codeintel://schema/table/{table_key}` → per-table schema (if you want direct addressing)

---

## 5) How tools + resources should work together (LLM-friendly)

### Recommended pattern

* Tools return **small structured outputs** plus **URIs** for anything large.
* LLM then calls **resource reads** only when needed.

This is explicitly aligned with FastMCP’s guidance that resources are a better fit than returning huge datasets directly in tools. 

### Example: `semantic_query_typed` response augmentation

Without changing your existing `SemanticQueryResponse`, you can wrap it:

```python
class SemanticQueryToolResponse(BaseModel):
    result: SemanticQueryResponse
    export_uri: str | None = None
    export_meta_uri: str | None = None
    note: str | None = None
```

So:

* “Normal” query: `export_uri=None`
* “Large” query: `result.rows` small preview, `export_uri` points to the full NDJSON/Parquet

---

## 6) Representative implementation snippet: resource templates + export resource

This shows how the taxonomy becomes real in code (and uses templated URIs, which FastMCP supports). 

```python
from fastmcp import FastMCP, Context
from fastmcp.resources import ResourceContent

def register_resources(mcp: FastMCP, kernel: SemanticKernel, export_store: ExportStore) -> None:
    @mcp.resource("codeintel://semantic/views", mime_type="application/json", tags=["semantic"])
    async def semantic_views_resource(ctx: Context):
        return await anyio.to_thread.run_sync(kernel.catalog)

    @mcp.resource("codeintel://semantic/views/{view_id}", mime_type="application/json", tags=["semantic"])
    async def semantic_view_resource(view_id: str, ctx: Context):
        return await anyio.to_thread.run_sync(kernel.describe, view_id)

    @mcp.resource("codeintel://exports/{export_id}", tags=["export"])
    async def export_payload_resource(export_id: str, ctx: Context) -> ResourceContent:
        meta = export_store.meta(export_id)
        blob = export_store.read_bytes(export_id)
        return ResourceContent(
            content=blob,
            mime_type=meta.mime_type,
            meta={"filename": meta.filename, "snapshot": meta.snapshot},
        )

    @mcp.resource("codeintel://exports/{export_id}/meta", mime_type="application/json", tags=["export"])
    async def export_meta_resource(export_id: str, ctx: Context):
        return export_store.meta(export_id).model_dump(mode="json")
```

(You’d implement `ExportStore` as a thin “write on tool call, read on resource call” wrapper, usually file-backed under your serve directory, with TTL cleanup.)

---

## 7) Minimal “resource templates” list to embed in `serving_meta_typed`

Include **exactly these** at first (keeps it simple but complete):

```json
[
  {"uri": "codeintel://meta/serving", "description": "Serving snapshot + versions"},
  {"uri": "codeintel://meta/resources", "description": "List resource templates"},
  {"uri": "codeintel://semantic/views", "description": "Semantic view catalog"},
  {"uri": "codeintel://semantic/views/{view_id}", "description": "Semantic view description"},
  {"uri": "codeintel://semantic/views/{view_id}/schema", "description": "Semantic view schema"},
  {"uri": "codeintel://exports/{export_id}", "description": "Export payload"},
  {"uri": "codeintel://exports/{export_id}/meta", "description": "Export metadata"},
  {"uri": "codeintel://exports/{export_id}/preview", "description": "Export preview"},
  {"uri": "codeintel://exports/{export_id}/sql", "description": "Export compiled SQL"}
]
```

---

## 8) My strong recommendation on what to do next

1. Add the **typed tools** (keep compat tools for now).
2. Add the **resource taxonomy** above + `codeintel://meta/resources`.
3. Add **semantic_export_typed** (so exports become first-class and big result flows are clean).
4. Update `semantic_catalog(_typed)` to include:

   * `resource_uri` per view, pointing to `codeintel://semantic/views/{view_id}`

If you want, I can also propose the *exact return models* (Pydantic) for:

* `ServingMetaResponse`
* `ExportHandleResponse`
* `SemanticQueryToolResponse` (preview + export uris)

…so the typed surface becomes extremely crisp and self-documenting.

# return models specification #

Below are **drop‑in Pydantic (v2) return models** for the three responses you called out:

* `ServingMetaResponse`
* `ExportHandleResponse`
* `SemanticQueryToolResponse` (wraps your existing `SemanticQueryResult`/`SemanticQueryResponse`, adds **preview + export URIs**)

These are designed specifically to give FastMCP a **crisp JSON schema** and consistent structured outputs (FastMCP will emit `structuredContent` automatically for Pydantic returns). 
They also fit your plan direction of keeping the MCP layer clean/typed while still supporting the existing tool set (the 6 tools in `src/codeintel/serving/mcp/app.py`). 

---

## Recommended file location

Put these in something like:

* `src/codeintel/serving/mcp/models/returns.py`

and import them from your tool implementations.

---

## Code: common URI types + shared metadata models

```python
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from typing_extensions import Annotated

from pydantic import BaseModel, ConfigDict, Field


# ----------------------------
# URI / ID primitives
# ----------------------------

CodeIntelURI = Annotated[
    str,
    Field(
        pattern=r"^codeintel://.+",
        description="CodeIntel resource URI (codeintel://...).",
        examples=["codeintel://meta/serving", "codeintel://semantic/views/function_metrics"],
    ),
]

RFC6570TemplateURI = Annotated[
    str,
    Field(
        description="RFC 6570 URI template (may include {placeholders}).",
        examples=[
            "codeintel://semantic/views/{view_id}",
            "codeintel://exports/{export_id}/meta",
        ],
    ),
]

ViewId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_.-]+$",
        description="Semantic view identifier (stable).",
        examples=["function_metrics", "risk_factors", "module_profile"],
    ),
]

ExportId = Annotated[
    str,
    Field(
        min_length=8,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Export identifier (opaque token).",
        examples=["01HZY9E1K8ZQ6N9J3W2K9M3A8B"],
    ),
]


# ----------------------------
# Basic nested models
# ----------------------------

class SnapshotRef(BaseModel):
    """Identifies the immutable serving snapshot currently mounted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: str = Field(..., description="Repo identifier (usually path or logical name).")
    commit: str = Field(..., description="Git commit SHA (or equivalent).")
    run_id: str = Field(..., description="Build run identifier (stable for the snapshot).")
    published_at: datetime = Field(..., description="When the serving snapshot was published.")


class ResourceTemplate(BaseModel):
    """Self-documenting resource discovery for LLM agents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: RFC6570TemplateURI = Field(..., description="Resource URI (or template).")
    description: str = Field(..., description="Human/LLM friendly description.")
    mime_type: Optional[str] = Field(None, description="MIME type if fixed/known.")
    tags: tuple[str, ...] = Field(default_factory=tuple, description="Categorization tags.")


class SemanticLayerInfo(BaseModel):
    """Semantic layer identity + inventory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(..., description="Semantic layer version string.")
    hash: str = Field(..., description="Stable hash of semantic registry content.")
    view_count: int = Field(..., ge=0, description="Number of semantic views available.")
    schema_manifest_hash: Optional[str] = Field(
        None,
        description="Hash of schema manifest backing semantic view schemas (if applicable).",
    )


class BuildSpecInfo(BaseModel):
    """BuildSpec identity exposed by serving for parity/debugging."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(..., description="BuildSpec schema/version.")
    hash: str = Field(..., description="Stable deterministic hash of BuildSpec JSON.")
    compiled_at: datetime = Field(..., description="When BuildSpec was compiled.")


class QueryLimits(BaseModel):
    """Server-enforced limits (so agents don't guess)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_limit: int = Field(200, ge=1, description="Default row limit for query tools.")
    max_limit: int = Field(5_000, ge=1, description="Maximum allowed limit for query tools.")
    export_max_rows: int = Field(100_000, ge=1, description="Maximum rows allowed for exports.")
    export_ttl_seconds: Optional[int] = Field(
        None,
        ge=1,
        description="Optional TTL for exports; if None, exports are session-scoped or manual cleanup.",
    )
```

---

## 1) `ServingMetaResponse`

This is what your `serving_meta_typed` tool (and `codeintel://meta/serving`) should return.

Key best-in-class principle: **make the server self-describing**, so agents can learn inventories and URI conventions without external docs.

```python
class ServingMetaResponse(BaseModel):
    """High-level server + snapshot metadata for LLM discovery and debugging."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    service: Literal["codeintel"] = Field("codeintel", description="Service identifier.")
    server_version: str = Field(..., description="CodeIntel package/version string.")
    protocol: Literal["mcp"] = Field("mcp", description="Protocol identifier.")
    started_at: datetime = Field(..., description="When this server process started.")

    snapshot: SnapshotRef = Field(..., description="Currently mounted serving snapshot.")
    semantic_layer: SemanticLayerInfo = Field(..., description="Semantic layer identity + counts.")
    buildspec: BuildSpecInfo = Field(..., description="Compiled BuildSpec identity.")

    # Optional but very useful for hardening
    read_only: bool = Field(True, description="Whether serving DB connections are read-only.")
    features: dict[str, bool] = Field(
        default_factory=dict,
        description=(
            "Feature flags/capabilities. Example keys: "
            "supports_explain, supports_export, supports_sampling, supports_resources."
        ),
    )

    limits: QueryLimits = Field(..., description="Server limits for queries and exports.")

    # Discovery: tell agents what they can read without guessing
    resource_templates: tuple[ResourceTemplate, ...] = Field(
        default_factory=tuple,
        description="Resource URI taxonomy templates exposed by the server.",
    )

    # Optional inventories to reduce tool calls (keep lightweight)
    inventories: dict[str, int] = Field(
        default_factory=dict,
        description="Counts of available datasets/tables/exports/etc for quick triage.",
    )
```

**Usage note:** FastMCP will use the Pydantic schema for structured output. 

---

## 2) `ExportHandleResponse`

This is what `semantic_export_typed(...)` should return, and what `semantic_query_typed` can embed when it chooses to spill large results to an export.

```python
ExportFormat = Literal["ndjson", "json", "parquet", "arrow"]

class ExportSnapshot(BaseModel):
    """Snapshot identity captured at export time (proves results are stable)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot: SnapshotRef = Field(..., description="Serving snapshot in effect when export was created.")
    semantic_layer_hash: str = Field(..., description="Semantic layer hash at export time.")
    buildspec_hash: str = Field(..., description="BuildSpec hash at export time.")


class ExportHandleResponse(BaseModel):
    """Handle returned by export tool; payload is fetched via resources."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    export_id: ExportId = Field(..., description="Opaque export token.")
    format: ExportFormat = Field(..., description="Export serialization format.")
    mime_type: str = Field(..., description="MIME type for export payload.")
    filename: str = Field(..., description="Suggested filename for clients.")

    # Primary payload + common attachments
    uri: CodeIntelURI = Field(..., description="Resource URI to fetch the export payload.")
    meta_uri: CodeIntelURI = Field(..., description="Resource URI to fetch export metadata.")
    preview_uri: Optional[CodeIntelURI] = Field(None, description="Resource URI to fetch export preview.")
    sql_uri: Optional[CodeIntelURI] = Field(None, description="Resource URI to fetch compiled SQL (if available).")

    created_at: datetime = Field(..., description="When this export was generated.")
    expires_at: Optional[datetime] = Field(None, description="When this export expires (if TTL is used).")

    # Lightweight metadata so agents can decide whether to download the payload
    row_count: Optional[int] = Field(None, ge=0, description="Row count if known without reading payload.")
    byte_size: Optional[int] = Field(None, ge=0, description="Byte size if known without reading payload.")

    snapshot: ExportSnapshot = Field(..., description="Hashes/identities captured at export time.")

    # Nice for LLM UX/debugging
    note: Optional[str] = Field(None, description="Optional note for the client/LLM (e.g. 'result spilled to export').")
```

Why this structure works well with FastMCP:

* Tools return a **small structured handle**, and the LLM/agent fetches big payloads via resources (which is a core FastMCP pattern). 

---

## 3) `SemanticQueryToolResponse` (preview + export URIs)

This is the “typed” wrapper that makes `semantic_query_typed` perfect for LLMs:

* Always returns a normal `SemanticQueryResult`/`SemanticQueryResponse` (whatever your kernel currently uses)
* Optionally returns an `export` handle (and URIs) when results are large

Your plan already shows `SemanticQueryResult` as the return type in the typed direction. 

```python
class QueryPreview(BaseModel):
    """Small, safe preview that fits in LLM context windows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    columns: tuple[str, ...] = Field(default_factory=tuple, description="Column names in display order.")
    rows: tuple[dict[str, object], ...] = Field(default_factory=tuple, description="Preview rows (truncated).")
    truncated: bool = Field(True, description="Whether preview is truncated.")


class SemanticQueryToolResponse(BaseModel):
    """Return type for semantic_query_typed: result + optional export spillover."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Import your real model here:
    # from codeintel.serving.semantic.models import SemanticQueryResult
    result: "SemanticQueryResult" = Field(..., description="Primary query result (may be truncated).")

    # Optional preview to keep responses small & readable
    preview: Optional[QueryPreview] = Field(
        None,
        description="Optional small preview for LLM-friendly output (even if export exists).",
    )

    # If the query is large, return an export handle to fetch full payload via resources
    export: Optional[ExportHandleResponse] = Field(
        None,
        description="If present, full results are available via export resources.",
    )

    # Convenience URIs (redundant with export.*, but nice for directness)
    export_uri: Optional[CodeIntelURI] = Field(None, description="Shortcut to export payload URI.")
    export_meta_uri: Optional[CodeIntelURI] = Field(None, description="Shortcut to export meta URI.")

    # Agent-facing guidance (helps without being verbose)
    note: Optional[str] = Field(
        None,
        description=(
            "Short, user/LLM-friendly note. Example: "
            "'Result truncated to 200 rows; use export_uri for full dataset.'"
        ),
    )
```

> If your project uses forward refs, add:

```python
from pydantic import TypeAdapter
SemanticQueryToolResponse.model_rebuild()
```

---

## A tiny usage example inside `semantic_query_typed`

```python
# PSEUDOCODE - illustrates shape, not your exact kernel API.

async def semantic_query_typed(request: SemanticQueryRequest, ctx: Context) -> SemanticQueryToolResponse:
    result = await anyio.to_thread.run_sync(kernel.query, request)
    typed_result = SemanticQueryResult.model_validate(result)

    if typed_result.truncated:
        export = await anyio.to_thread.run_sync(kernel.export, request)  # returns ExportHandleResponse
        return SemanticQueryToolResponse(
            result=typed_result,
            preview=QueryPreview(columns=tuple(typed_result.columns), rows=tuple(typed_result.rows[:5]), truncated=True),
            export=export,
            export_uri=export.uri,
            export_meta_uri=export.meta_uri,
            note="Result truncated; use export_uri to fetch full payload.",
        )

    return SemanticQueryToolResponse(result=typed_result)
```

---

## Optional refinement (strongly recommended): strict “extra=forbid” everywhere

All models above set `extra="forbid"`. This is a big part of “best-in-class hardness” because:

* Agents can’t accidentally depend on undocumented fields
* You can safely evolve models without silent schema drift

---

If you want, next I can also propose the **matching `ExportMetaResponse`** (returned by `codeintel://exports/{id}/meta`) plus a **single `ResourceTemplatesResponse`** for `codeintel://meta/resources` so discovery is completely standardized and machine-readable.


# export meta response #


Absolutely — here are **matching, typed, discovery-first** return models for:

* `ExportMetaResponse` (served at `codeintel://exports/{export_id}/meta`)
* `ResourceTemplatesResponse` (served at `codeintel://meta/resources`)

They’re designed to fit your “tools return small handles; resources serve large payloads” pattern, which is exactly what FastMCP resources + URI templates are good at. 
And because FastMCP emits `structuredContent` automatically for Pydantic/dicts, you get a crisp machine-readable surface for LLM agents “for free.” 

---

## 1) ExportMetaResponse

### Intended semantics

* `semantic_export_typed(...)` returns an `ExportHandleResponse` (you already have this).
* The client reads:

  * `codeintel://exports/{export_id}` for the payload
  * `codeintel://exports/{export_id}/meta` for metadata (this model)
  * optionally `.../preview` and `.../sql`

`ExportMetaResponse` should be:

* **small** (safe to inline in LLM context)
* **complete enough** that an agent can decide whether to fetch the actual payload
* **provenanced** (snapshot + semantic/build hashes captured)

### Code

```python
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from typing_extensions import Annotated

from pydantic import BaseModel, ConfigDict, Field


# Reuse from your earlier models:
# - CodeIntelURI
# - ExportId
# - ViewId
# - ExportFormat
# - ExportSnapshot (includes SnapshotRef + semantic_layer_hash + buildspec_hash)
# - QueryLimits (optional)
#
# If you want this file to be standalone, you can import those primitives here.


Sha256Hex = Annotated[
    str,
    Field(
        pattern=r"^[a-f0-9]{64}$",
        description="SHA-256 hex digest.",
        examples=["e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"],
    ),
]


ExportStatus = Literal["ready", "expired", "missing", "error"]


class ExportQuerySpec(BaseModel):
    """
    A sanitized echo of what was exported.

    Keep this intentionally generic to avoid tight coupling to your internal
    SemanticQueryRequest model and to keep forward-compat.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    view_id: Optional[ViewId] = Field(
        None,
        description="Semantic view exported (if export came from semantic layer).",
    )

    # "select/order_by/filters" are stored as neutral shapes, not internal types.
    select: Optional[tuple[str, ...]] = Field(
        None,
        description="Selected columns (if specified).",
    )
    order_by: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Ordering spec (server conventions).",
    )
    filters: tuple[dict[str, object], ...] = Field(
        default_factory=tuple,
        description="Filter specs (sanitized, JSON-serializable).",
    )

    limit: Optional[int] = Field(
        None,
        ge=0,
        description="Limit used for export generation (if applicable).",
    )
    offset: Optional[int] = Field(
        None,
        ge=0,
        description="Offset used for export generation (if applicable).",
    )

    # Extremely useful for agents: they can check if two exports are “the same query”.
    query_hash: Optional[str] = Field(
        None,
        description="Stable fingerprint of query inputs (filters/select/order/limit/offset).",
        examples=["q_7c9a2c2b0f0d6a31"],
    )


class ExportSchemaSummary(BaseModel):
    """Lightweight schema summary for the exported payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    columns: tuple[str, ...] = Field(default_factory=tuple, description="Column names in payload order.")

    # Keep it a mapping of column -> type string (DuckDB / Arrow-ish strings)
    # so agents can reason about joins & predicates.
    types: dict[str, str] = Field(
        default_factory=dict,
        description="Column types keyed by column name.",
        examples=[{"repo": "VARCHAR", "commit": "VARCHAR", "cyclomatic_complexity": "INTEGER"}],
    )

    schema_hash: Optional[str] = Field(
        None,
        description="Stable fingerprint of the schema (e.g., hash(columns+types)).",
    )


class ExportURIs(BaseModel):
    """All resource URIs associated with an export."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    payload_uri: CodeIntelURI = Field(..., description="URI to fetch the export payload.")
    meta_uri: CodeIntelURI = Field(..., description="URI to fetch this metadata.")
    preview_uri: Optional[CodeIntelURI] = Field(None, description="URI to fetch a small preview (JSON).")
    sql_uri: Optional[CodeIntelURI] = Field(None, description="URI to fetch compiled SQL (if stored).")


class ExportMetaResponse(BaseModel):
    """Metadata returned by `codeintel://exports/{export_id}/meta`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    export_id: ExportId = Field(..., description="Export identifier.")
    status: ExportStatus = Field(..., description="Current export status.")
    created_at: datetime = Field(..., description="When this export was created.")
    expires_at: Optional[datetime] = Field(None, description="When this export expires (if TTL configured).")

    format: ExportFormat = Field(..., description="Serialization format.")
    mime_type: str = Field(..., description="MIME type for payload.")
    filename: str = Field(..., description="Suggested filename for the payload.")

    # Payload metrics (if cheap to compute; otherwise None)
    row_count: Optional[int] = Field(None, ge=0, description="Row count (if known).")
    byte_size: Optional[int] = Field(None, ge=0, description="Byte size (if known).")
    sha256: Optional[Sha256Hex] = Field(None, description="Hash of payload bytes (if computed).")

    # Provenance
    snapshot: ExportSnapshot = Field(
        ...,
        description="Snapshot/build/semantic hashes captured at export time.",
    )

    # What produced this export
    query: Optional[ExportQuerySpec] = Field(
        None,
        description="Sanitized query spec used to generate the export (if applicable).",
    )
    schema: Optional[ExportSchemaSummary] = Field(
        None,
        description="Schema summary for the exported payload (if applicable).",
    )

    uris: ExportURIs = Field(..., description="Resource URIs for payload + helpers.")

    warnings: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Non-fatal warnings (e.g., 'sql unavailable', 'row_count unknown').",
    )
    note: Optional[str] = Field(None, description="Human/LLM-friendly note.")
```

### Why these fields are “right-sized”

* Agents can decide: “Do I need the full export payload or just the preview?”
* Agents can reason about joins/types without downloading payload.
* You get stability + provenance via `snapshot` hashes.
* You avoid coupling the resource schema to internal request models (keeps things maintainable).

---

## 2) ResourceTemplatesResponse

### Intended semantics

This should be the one canonical “discovery” resource that an agent can call to understand:

* which `codeintel://...` resources exist
* which are templated
* what their MIME types and intended usage are

FastMCP explicitly supports resources and **RFC 6570 URI templates**, which is exactly what we’re leveraging. 

### Code

```python
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# Reuse:
# - SnapshotRef
# - ResourceTemplate
# - SemanticLayerInfo
# - BuildSpecInfo


class ResourceTemplatesResponse(BaseModel):
    """Returned by `codeintel://meta/resources` for standardized discovery."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: CodeIntelURI = Field(
        "codeintel://meta/resources",
        description="Canonical URI for this response.",
    )
    generated_at: datetime = Field(..., description="When this listing was generated.")
    snapshot: SnapshotRef = Field(..., description="Currently mounted serving snapshot.")

    # Optional but extremely helpful: discovery objects that correlate resources to the current spec.
    semantic_layer: Optional[SemanticLayerInfo] = Field(
        None,
        description="Semantic layer identity (if semantic layer enabled).",
    )
    buildspec: Optional[BuildSpecInfo] = Field(
        None,
        description="BuildSpec identity (if BuildSpec available at serving time).",
    )

    templates: tuple[ResourceTemplate, ...] = Field(
        default_factory=tuple,
        description="All supported resource templates, stable and machine-readable.",
    )

    # A place to document policy / constraints without requiring external docs.
    notes: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Short usage notes for agents (limits, auth, TTLs, etc.).",
    )
```

---

## 3) Suggested “canonical templates” to include in ResourceTemplatesResponse

You asked specifically for **standardized & machine-readable discovery** — so I’d explicitly include the most important resources (and keep it stable over time).

Here’s a recommended baseline list (you can adjust names without changing the response model):

```python
DEFAULT_RESOURCE_TEMPLATES: tuple[ResourceTemplate, ...] = (
    ResourceTemplate(
        uri="codeintel://meta/serving",
        description="Serving metadata: snapshot, semantic layer hash, BuildSpec hash, limits.",
        mime_type="application/json",
        tags=("meta",),
    ),
    ResourceTemplate(
        uri="codeintel://meta/resources",
        description="Machine-readable inventory of all CodeIntel resources and URI templates.",
        mime_type="application/json",
        tags=("meta", "discovery"),
    ),
    ResourceTemplate(
        uri="codeintel://semantic/views",
        description="JSON list of semantic views (catalog).",
        mime_type="application/json",
        tags=("semantic", "catalog"),
    ),
    ResourceTemplate(
        uri="codeintel://semantic/views/{view_id}",
        description="Semantic view descriptor (columns, entity, grain, defaults).",
        mime_type="application/json",
        tags=("semantic", "describe"),
    ),
    ResourceTemplate(
        uri="codeintel://exports/{export_id}",
        description="Export payload (format depends on export).",
        mime_type=None,  # dynamic
        tags=("exports", "payload"),
    ),
    ResourceTemplate(
        uri="codeintel://exports/{export_id}/meta",
        description="Export metadata (schema, counts, hashes, provenance).",
        mime_type="application/json",
        tags=("exports", "meta"),
    ),
    ResourceTemplate(
        uri="codeintel://exports/{export_id}/preview",
        description="Small JSON preview of export payload (LLM-friendly).",
        mime_type="application/json",
        tags=("exports", "preview"),
    ),
    ResourceTemplate(
        uri="codeintel://exports/{export_id}/sql",
        description="Compiled SQL used to generate the export (if recorded).",
        mime_type="text/plain",
        tags=("exports", "sql"),
    ),
)
```

This aligns with FastMCP’s resource approach (lazy, read-only, templated URIs). 

---

## 4) Minimal FastMCP resource implementations (wiring sketch)

FastMCP resources are defined with `@mcp.resource(...)` and are lazily executed. 
They can also accept a `Context` parameter if you want logging/progress/etc. 

```python
from fastmcp import Context

# codeintel://meta/resources
@mcp.resource("codeintel://meta/resources")
def meta_resources() -> ResourceTemplatesResponse:
    return ResourceTemplatesResponse(
        generated_at=datetime.now(tz=timezone.utc),
        snapshot=current_snapshot_ref(),
        semantic_layer=current_semantic_layer_info(),
        buildspec=current_buildspec_info(),
        templates=DEFAULT_RESOURCE_TEMPLATES,
        notes=(
            "Exports are immutable and tied to the mounted snapshot.",
            "Use export preview for LLM-friendly reads; payload may be large.",
        ),
    )


# codeintel://exports/{export_id}/meta
@mcp.resource("codeintel://exports/{export_id}/meta")
async def export_meta(export_id: str, ctx: Context) -> ExportMetaResponse:
    # Get metadata from your export registry/store (in-memory or persisted).
    record = await anyio.to_thread.run_sync(export_store.get, export_id)
    if record is None:
        # You can raise an exception; FastMCP will surface an MCP resource error.
        # Or return status="missing" — your choice.
        return ExportMetaResponse(
            export_id=export_id,
            status="missing",
            created_at=datetime.now(tz=timezone.utc),
            expires_at=None,
            format="json",
            mime_type="application/json",
            filename=f"{export_id}.json",
            row_count=None,
            byte_size=None,
            sha256=None,
            snapshot=recorded_snapshot_fallback(),
            query=None,
            schema=None,
            uris=ExportURIs(
                payload_uri=f"codeintel://exports/{export_id}",
                meta_uri=f"codeintel://exports/{export_id}/meta",
                preview_uri=f"codeintel://exports/{export_id}/preview",
                sql_uri=f"codeintel://exports/{export_id}/sql",
            ),
            warnings=("export_id not found",),
            note="Export metadata not available (unknown export_id).",
        )

    # Normal “ready” path: translate your internal export record -> typed meta response
    return ExportMetaResponse(
        export_id=record.export_id,
        status="ready",
        created_at=record.created_at,
        expires_at=record.expires_at,
        format=record.format,
        mime_type=record.mime_type,
        filename=record.filename,
        row_count=record.row_count,
        byte_size=record.byte_size,
        sha256=record.sha256,
        snapshot=record.snapshot,
        query=record.query_spec,
        schema=record.schema_summary,
        uris=record.uris,
        warnings=tuple(record.warnings),
        note=record.note,
    )
```

> If you want per-resource cache headers or richer response metadata, FastMCP supports returning a `ResourceContent` object that can carry content + MIME type + metadata.  (For JSON responses, most teams just return Pydantic models.)

---

If you want, the next “polish layer” is to define a **single, canonical `ErrorResponse` model** (for both tools and resources) and make `export_meta` return `status="error"` with a stable error shape when something is corrupt/expired — so LLM agents don’t have to parse exception strings.


# Error response model #


Below is a **drop‑in “canonical error shape”** that works identically for **FastMCP tools** *and* **FastMCP resources**, and specifically makes your `codeintel://exports/{id}/meta` resource return `status="error"` with a **stable, typed payload** for **expired/corrupt/missing** export handles—so LLM agents never have to parse exception strings.

This design complements (not replaces) `mask_error_details=True` + `ToolError`: FastMCP already supports masking internal traces and controlled error messages , and your plan already calls this out . The key upgrade here is: **expected/semantic failures return structured `ErrorResponse`**, while **unexpected internal failures still raise** (and get masked).

---

## 1) Canonical error model

### Goals

* **Same model** usable everywhere: tools, resources, HTTP adapters, internal kernel APIs.

* **Stable programmatic fields** for LLM agents:

  * `code` (stable string)
  * `kind` (category)
  * `message` (safe, short)
  * `retryable`
  * `hint` (what to do next)
  * `details` (safe key/value; never stack traces)

* “Envelope” pattern: **responses return either `status="ok"` or `status="error"`**, never “raise for normal cases”.

FastMCP already supports returning structured JSON from tools/resources (dicts/Pydantic models become machine‑readable `structuredContent`) , and resources can return JSON-serializable dict/model outputs directly .

---

## 2) Proposed types (Pydantic v2)

Put this in something like:

* `src/codeintel/serving/models/errors.py` (serving-only), **or**
* `src/codeintel/core/models/errors.py` (if you want it shared by build/serving/cli).

```python
# src/codeintel/serving/models/errors.py
from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Mapping

from pydantic import BaseModel, Field, ConfigDict


class ErrorKind(StrEnum):
    invalid_request = "invalid_request"
    not_found = "not_found"
    expired = "expired"
    corrupt = "corrupt"
    conflict = "conflict"
    unavailable = "unavailable"
    timeout = "timeout"
    internal = "internal"


class ErrorInfo(BaseModel):
    """Canonical, stable error payload for tools + resources."""
    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        ...,
        description="Stable machine code (e.g. CODEINTEL_EXPORT_EXPIRED). Never change once published.",
        examples=["CODEINTEL_EXPORT_EXPIRED"],
    )
    kind: ErrorKind = Field(..., description="Coarse error category.")
    message: str = Field(..., description="Short, safe human-readable description.")
    retryable: bool = Field(False, description="Whether client can retry safely.")
    hint: str | None = Field(
        None,
        description="What the client/agent should do next (safe guidance).",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Safe structured details (no stack traces, no internal file paths unless you want them).",
    )


class ErrorResponse(BaseModel):
    """Canonical top-level error response; usable for tools and resources."""
    model_config = ConfigDict(extra="forbid")

    status: Literal["error"] = "error"
    error: ErrorInfo


def err(
    *,
    code: str,
    kind: ErrorKind,
    message: str,
    retryable: bool = False,
    hint: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> ErrorResponse:
    """Small helper to construct ErrorResponse consistently."""
    return ErrorResponse(
        error=ErrorInfo(
            code=code,
            kind=kind,
            message=message,
            retryable=retryable,
            hint=hint,
            details=dict(details or {}),
        )
    )
```

### Recommended “error code” conventions

Keep it boring and stable:

* Prefix: `CODEINTEL_...`
* Middle: subsystem (`EXPORT`, `SEMANTIC`, `SERVING`, `SCHEMA`, `SPEC`)
* Suffix: condition (`EXPIRED`, `NOT_FOUND`, `INVALID_FILTER`, `CORRUPT`, etc.)

Example codes you’ll likely need immediately:

* `CODEINTEL_EXPORT_NOT_FOUND`
* `CODEINTEL_EXPORT_EXPIRED`
* `CODEINTEL_EXPORT_CORRUPT`
* `CODEINTEL_EXPORT_UNAVAILABLE`
* `CODEINTEL_SEMANTIC_VIEW_NOT_FOUND`
* `CODEINTEL_SEMANTIC_INVALID_QUERY`

---

## 3) Apply to ExportMetaResponse (resource)

You previously wanted:

* `codeintel://exports/{id}/meta` → returns typed metadata
* and now: **when corrupt/expired** return `status="error"` with stable `ErrorResponse`.

### Proposed ExportMetaResponse

Put in `src/codeintel/serving/models/exports.py`:

```python
# src/codeintel/serving/models/exports.py
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict

from codeintel.serving.models.errors import ErrorResponse


class ExportMetaResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "error"] = "ready"

    export_id: str
    created_at: datetime | None = None
    expires_at: datetime | None = None

    # What the export represents (helpful for agents)
    view_id: str | None = None
    row_count: int | None = None
    format: Literal["jsonl", "parquet"] | None = None

    # Resource URIs
    meta_uri: str = Field(..., description="Resource URI for this meta object.")
    data_uri: str | None = Field(None, description="Resource URI for export data, if available.")

    # If status == error
    error: ErrorResponse | None = None

    @classmethod
    def ready(
        cls,
        *,
        export_id: str,
        created_at: datetime,
        expires_at: datetime,
        view_id: str | None,
        row_count: int | None,
        format: Literal["jsonl", "parquet"] | None,
        meta_uri: str,
        data_uri: str,
    ) -> "ExportMetaResponse":
        return cls(
            status="ready",
            export_id=export_id,
            created_at=created_at,
            expires_at=expires_at,
            view_id=view_id,
            row_count=row_count,
            format=format,
            meta_uri=meta_uri,
            data_uri=data_uri,
            error=None,
        )

    @classmethod
    def error(
        cls,
        *,
        export_id: str,
        meta_uri: str,
        created_at: datetime | None = None,
        expires_at: datetime | None = None,
        error: ErrorResponse,
        view_id: str | None = None,
        format: Literal["jsonl", "parquet"] | None = None,
    ) -> "ExportMetaResponse":
        return cls(
            status="error",
            export_id=export_id,
            created_at=created_at,
            expires_at=expires_at,
            view_id=view_id,
            format=format,
            meta_uri=meta_uri,
            data_uri=None,
            error=error,
        )
```

---

## 4) Implement export meta resource with stable error returns

FastMCP resources can return JSON model outputs, and they support URI templates like `data://user/{user_id}` . So you can do the same for `codeintel://exports/{export_id}/meta`.

Example (adapt filenames to your current `src/codeintel/serving/mcp/` layout):

```python
# src/codeintel/serving/mcp/resources.py
from __future__ import annotations

from datetime import datetime, timezone

from fastmcp import Context, FastMCP

from codeintel.serving.models.errors import err, ErrorKind
from codeintel.serving.models.exports import ExportMetaResponse


def register_export_resources(mcp: FastMCP, *, kernel) -> None:
    # meta endpoint
    @mcp.resource("codeintel://exports/{export_id}/meta", mime_type="application/json")
    async def export_meta(export_id: str, ctx: Context) -> ExportMetaResponse:
        meta_uri = f"codeintel://exports/{export_id}/meta"

        try:
            # Whatever your kernel API is (examples):
            #   handle = kernel.exports.get_handle(export_id)
            #   meta = kernel.exports.get_meta(export_id)
            meta = await kernel.exports.meta(export_id)  # often offloaded to thread in your codebase
        except kernel.exports.ExportNotFound:
            return ExportMetaResponse.error(
                export_id=export_id,
                meta_uri=meta_uri,
                error=err(
                    code="CODEINTEL_EXPORT_NOT_FOUND",
                    kind=ErrorKind.not_found,
                    message="Export handle not found.",
                    retryable=False,
                    hint="Re-run semantic_query to generate a new export handle.",
                    details={"export_id": export_id},
                ),
            )
        except kernel.exports.ExportExpired as e:
            now = datetime.now(timezone.utc)
            return ExportMetaResponse.error(
                export_id=export_id,
                meta_uri=meta_uri,
                created_at=getattr(e, "created_at", None),
                expires_at=getattr(e, "expires_at", None),
                error=err(
                    code="CODEINTEL_EXPORT_EXPIRED",
                    kind=ErrorKind.expired,
                    message="Export handle expired.",
                    retryable=True,
                    hint="Re-run semantic_query with the same parameters to generate a fresh export.",
                    details={
                        "export_id": export_id,
                        "now": now.isoformat(),
                        "expires_at": getattr(e, "expires_at", None).isoformat()
                        if getattr(e, "expires_at", None)
                        else None,
                    },
                ),
            )
        except kernel.exports.ExportCorrupt as e:
            # IMPORTANT: do not leak raw exceptions/paths; just stable details
            return ExportMetaResponse.error(
                export_id=export_id,
                meta_uri=meta_uri,
                created_at=getattr(e, "created_at", None),
                expires_at=getattr(e, "expires_at", None),
                error=err(
                    code="CODEINTEL_EXPORT_CORRUPT",
                    kind=ErrorKind.corrupt,
                    message="Export metadata is corrupt or unreadable.",
                    retryable=True,
                    hint="Re-run semantic_query to regenerate the export; if this repeats, rebuild the serving snapshot.",
                    details={"export_id": export_id},
                ),
            )
        except Exception as e:
            # Unexpected => keep your masked errors posture
            await ctx.error(f"export_meta unexpected failure: {type(e).__name__}")
            return ExportMetaResponse.error(
                export_id=export_id,
                meta_uri=meta_uri,
                error=err(
                    code="CODEINTEL_EXPORT_INTERNAL_ERROR",
                    kind=ErrorKind.internal,
                    message="Internal error while reading export metadata.",
                    retryable=True,
                    hint="Retry. If it persists, check server logs.",
                    details={"export_id": export_id},
                ),
            )

        # Happy path
        data_uri = f"codeintel://exports/{export_id}/data"
        return ExportMetaResponse.ready(
            export_id=export_id,
            created_at=meta.created_at,
            expires_at=meta.expires_at,
            view_id=meta.view_id,
            row_count=meta.row_count,
            format=meta.format,
            meta_uri=meta_uri,
            data_uri=data_uri,
        )
```

### Key design choices

1. **No exceptions for normal “agent-visible” failure modes** (not found/expired/corrupt).
2. For truly unexpected failures:

   * you *still* return a stable shape here (so resource reads are predictable)
   * and you *also* log internally with `ctx.error`.
3. If you want stricter semantics: you can raise for unexpected failures, but then you lose “stable shape” (agents must parse protocol errors). For your stated goal, the above is better.

---

## 5) Apply to tools too (same ErrorResponse)

FastMCP supports `mask_error_details=True` and `ToolError`  (and you already planned to use it ). Keep that for **unexpected** problems.

But for **expected** problems (invalid view_id, invalid filter, schema mismatch), return a typed error result.

Example: update the tool return type to a union:

```python
# src/codeintel/serving/mcp/app.py
from typing import Union

from codeintel.serving.models.errors import ErrorResponse, err, ErrorKind
from codeintel.serving.models.semantic import SemanticQueryToolResponse

SemanticQueryResult = Union[SemanticQueryToolResponse, ErrorResponse]

@mcp.tool(...)
async def semantic_query(..., ctx: Context) -> SemanticQueryResult:
    try:
        ...
    except kernel.semantic.ViewNotFound:
        return err(
            code="CODEINTEL_SEMANTIC_VIEW_NOT_FOUND",
            kind=ErrorKind.not_found,
            message=f"Semantic view '{view_id}' not found.",
            retryable=False,
            hint="Call semantic_catalog to list available views, then retry.",
            details={"view_id": view_id},
        )
    except kernel.semantic.InvalidQuery as e:
        return err(
            code="CODEINTEL_SEMANTIC_INVALID_QUERY",
            kind=ErrorKind.invalid_request,
            message="Invalid semantic query parameters.",
            retryable=False,
            hint="Check filters/select/order_by against semantic_describe(view_id).",
            details={"view_id": view_id, "reason": e.safe_reason()},
        )
```

This gives agents a deterministic branch:

* if `status=="error"` → inspect `error.code`, apply hint
* else → proceed

And because FastMCP will serialize models to JSON structured outputs , the agent will see the structure clearly.

---

## 6) One canonical error surface across tools + resources

To make this truly “best-in-class” and effortless to keep consistent:

### A) Add an “error catalog” (optional but powerful)

Expose a resource:

* `codeintel://meta/errors`

Return:

* list of known error codes
* their meanings
* whether retryable
* suggested next action

This makes your system **self-documenting** for LLM consumers.

### B) Add `ErrorResponse` to every typed response *or* use unions everywhere

Two common approaches:

1. **Union style** (cleanest for agents):

   * tool returns `Union[OkModel, ErrorResponse]`
   * resource returns `Union[OkModel, ErrorResponse]`

2. **Envelope style** (cleanest for humans):

   * always return `{status, meta, data?, error?}`

Given you explicitly asked for `status="error"` on export_meta, you’re already leaning toward envelope/unified-status semantics.

---

## 7) Minimal tests that pay off hugely

Add a focused unit/integration test set:

### `tests/serving/test_error_models.py`

* `ErrorResponse.model_json_schema()` is stable (snapshot it if you want).
* `err(...)` creates correct shape.

### `tests/serving/mcp/test_export_meta_resource_errors.py`

* request meta for:

  * missing export → `status="error"`, code `CODEINTEL_EXPORT_NOT_FOUND`
  * expired export → `status="error"`, code `CODEINTEL_EXPORT_EXPIRED`
  * corrupt export → `status="error"`, code `CODEINTEL_EXPORT_CORRUPT`

If you already have the CLI snapshot manifest runner, you can also add:

* a snapshot for the JSON output of a local “read resource” call, but that depends on whether you’ve built a CLI entrypoint for resource reads.

---

## 8) Practical guidance: when to return ErrorResponse vs raise ToolError

**Return `ErrorResponse`** when:

* view/export not found
* expired
* invalid filter/operator/column
* schema mismatch
* agent can correct input and retry

**Raise ToolError** (and rely on masking) when:

* invariants violated (should never happen)
* unexpected bug / programming error
* partial failure that you cannot safely represent without leaking internals

This blends your existing security posture (mask internals)  with a **stable machine-facing API**.

---

If you want, I can also propose the **exact error code list** you should “lock in” now (about ~20 codes), and the **single mapping function** `exception_to_error_response(e, *, context)` that you call everywhere (kernel/tools/resources) so the shape can’t drift.


# Error code list #

Below is a concrete, **“lock-in now” error code set (~20)** plus a single **`exception_to_error_response(e, *, context)`** mapper you can call everywhere (kernel/tools/resources/HTTP routes) so the shape **cannot drift**.

I’m keeping this aligned with the canonical models we already established:

* `ErrorResponse(status="error", error=ErrorInfo(...))`
* `ErrorKind` enum (invalid_request / not_found / expired / corrupt / conflict / unavailable / timeout / internal)

---

## 1) The “lock-in now” error code list (~20)

### Naming conventions

* Prefix: `CODEINTEL_`
* Subsystem: `SEMANTIC` | `EXPORT` | `SERVING` | `SCHEMA` | `AUTH` | `RATE`
* Condition: `NOT_FOUND` | `INVALID_QUERY` | `EXPIRED` | `CORRUPT` | etc.

### The codes

I recommend locking in exactly these 20:

#### Semantic/query layer (8)

1. **`CODEINTEL_SEMANTIC_VIEW_NOT_FOUND`**

   * kind: `not_found`, retryable: `False`
   * when: `view_id` doesn’t exist in registry/spec

2. **`CODEINTEL_SEMANTIC_INVALID_QUERY`**

   * kind: `invalid_request`, retryable: `False`
   * when: request fails validation (missing required fields, wrong types, etc.)

3. **`CODEINTEL_SEMANTIC_INVALID_FILTER`**

   * kind: `invalid_request`, retryable: `False`
   * when: filter spec malformed or unsupported operator/value

4. **`CODEINTEL_SEMANTIC_COLUMN_NOT_FOUND`**

   * kind: `invalid_request`, retryable: `False`
   * when: request references a column not in schema

5. **`CODEINTEL_SEMANTIC_LIMIT_EXCEEDED`**

   * kind: `invalid_request`, retryable: `False`
   * when: requested limit > server max

6. **`CODEINTEL_SEMANTIC_QUERY_TIMEOUT`**

   * kind: `timeout`, retryable: `True`
   * when: query execution hit time budget

7. **`CODEINTEL_SEMANTIC_QUERY_UNAVAILABLE`**

   * kind: `unavailable`, retryable: `True`
   * when: DB is busy/unavailable, snapshot not ready, etc.

8. **`CODEINTEL_SEMANTIC_INTERNAL_ERROR`**

   * kind: `internal`, retryable: `True`
   * when: unexpected exception in semantic query path

#### Export subsystem (6)

9. **`CODEINTEL_EXPORT_NOT_FOUND`**

   * kind: `not_found`, retryable: `False`

10. **`CODEINTEL_EXPORT_EXPIRED`**

* kind: `expired`, retryable: `True`
* hint: re-run query/export to generate a fresh handle

11. **`CODEINTEL_EXPORT_CORRUPT`**

* kind: `corrupt`, retryable: `True`
* hint: regenerate; if repeats, rebuild snapshot

12. **`CODEINTEL_EXPORT_TOO_LARGE`**

* kind: `invalid_request`, retryable: `False`
* when: export would exceed max rows/bytes

13. **`CODEINTEL_EXPORT_UNAVAILABLE`**

* kind: `unavailable`, retryable: `True`
* when: export store temporarily unavailable

14. **`CODEINTEL_EXPORT_INTERNAL_ERROR`**

* kind: `internal`, retryable: `True`

#### Serving snapshot / DB (4)

15. **`CODEINTEL_SERVING_SNAPSHOT_NOT_MOUNTED`**

* kind: `unavailable`, retryable: `True`
* when: ServingDBManager has no current snapshot pointer

16. **`CODEINTEL_SERVING_SNAPSHOT_MISMATCH`**

* kind: `conflict`, retryable: `True`
* when: request references snapshot/run_id that differs from mounted snapshot

17. **`CODEINTEL_SERVING_DB_LOCKED`**

* kind: `unavailable`, retryable: `True`
* when: DuckDB locked/busy, cannot open read transaction

18. **`CODEINTEL_SERVING_DB_INTERNAL_ERROR`**

* kind: `internal`, retryable: `True`

#### Schema/spec/auth/rate (2)

19. **`CODEINTEL_SCHEMA_MANIFEST_MISSING`**

* kind: `unavailable`, retryable: `True`
* when: schema manifest not found for snapshot (or cannot be loaded)

20. **`CODEINTEL_AUTH_FORBIDDEN`**

* kind: `forbidden` (if you keep this kind) or `invalid_request`/`unavailable` if not
* retryable: `False`
* when: request not allowed (e.g. remote access disabled, tool/tag disallowed)

> If you don’t want `forbidden` as a kind, map it to `invalid_request` and keep the code name. The *code* is the real contract.

---

## 2) A single mapping function + context model

### Why a single mapper

You want:

* one “truth” for how exceptions become errors
* identical shapes across tools/resources
* stable codes/hints (LLM agents can build reliable strategies)

### Recommended structure

* `ErrorContext`: the safe, structured “who/what/where”
* `exception_to_error_response(e, *, context)`:

  * returns `ErrorResponse`
  * never raises
  * never leaks stack traces or file paths
  * includes a `debug_id` for logs correlation (optional but highly recommended)

---

## 3) Code: `ErrorContext` + mapper

Put this in something like:

* `src/codeintel/serving/models/error_mapping.py` (or core/shared)

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from pydantic import ValidationError

# Import your canonical models
# from codeintel.serving.models.errors import ErrorResponse, ErrorKind, err

# If you already have ErrorKind + err() from the earlier snippet, reuse them.
# Below I assume:
#   - ErrorKind is an Enum with: invalid_request/not_found/expired/corrupt/conflict/unavailable/timeout/internal
#   - err(...) builds ErrorResponse(status="error", error=ErrorInfo(...))


@dataclass(frozen=True)
class ErrorContext:
    """
    Safe, structured context for error mapping.

    Keep this intentionally small and non-sensitive — no stack traces, no file paths.
    """
    operation: str  # e.g. "semantic_query", "export_meta", "read_resource"
    tool_name: str | None = None
    resource_uri: str | None = None

    view_id: str | None = None
    export_id: str | None = None

    # Snapshot identity (safe)
    repo: str | None = None
    commit: str | None = None
    run_id: str | None = None
    semantic_layer_hash: str | None = None
    buildspec_hash: str | None = None

    # Request controls (safe)
    limit: int | None = None
    offset: int | None = None

    # Observability correlation IDs (safe)
    request_id: str | None = None  # e.g. HTTP correlation id / MCP session id
    debug_id: str | None = None    # unique id you log server-side


def _base_details(ctx: ErrorContext) -> dict[str, Any]:
    """Safe details attached to every error response."""
    d: dict[str, Any] = {
        "operation": ctx.operation,
        "tool_name": ctx.tool_name,
        "resource_uri": ctx.resource_uri,
        "view_id": ctx.view_id,
        "export_id": ctx.export_id,
        "repo": ctx.repo,
        "commit": ctx.commit,
        "run_id": ctx.run_id,
        "semantic_layer_hash": ctx.semantic_layer_hash,
        "buildspec_hash": ctx.buildspec_hash,
        "limit": ctx.limit,
        "offset": ctx.offset,
        "request_id": ctx.request_id,
        "debug_id": ctx.debug_id,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    # drop None values to keep payload small
    return {k: v for k, v in d.items() if v is not None}


# ---- Optional: define a small protocol for "safe" exceptions ----

class SafeCodeIntelError(Exception):
    """
    Optional base exception for domain errors.
    If you adopt this, mapping becomes trivial and uniform.

    You can define subclasses for ExportNotFound, ViewNotFound, etc.
    """
    code: str
    kind: "ErrorKind"
    message: str
    retryable: bool
    hint: str | None
    details: Mapping[str, Any] | None

    def __init__(
        self,
        *,
        code: str,
        kind: "ErrorKind",
        message: str,
        retryable: bool = False,
        hint: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.kind = kind
        self.message = message
        self.retryable = retryable
        self.hint = hint
        self.details = details


def exception_to_error_response(e: Exception, *, context: ErrorContext) -> "ErrorResponse":
    """
    Convert any exception into a canonical ErrorResponse.

    Rule of thumb:
      - Expected domain errors => stable, specific codes
      - Validation errors => invalid_request
      - Timeout/unavailable => retryable True
      - Everything else => internal error, retryable True (with debug_id in details)
    """
    details = _base_details(context)

    # 1) Domain exceptions that already carry safe metadata
    if isinstance(e, SafeCodeIntelError):
        merged = dict(details)
        if e.details:
            merged.update({k: v for k, v in dict(e.details).items() if v is not None})
        return err(
            code=e.code,
            kind=e.kind,
            message=e.message,
            retryable=e.retryable,
            hint=e.hint,
            details=merged,
        )

    # 2) Pydantic validation errors for typed tool inputs
    if isinstance(e, ValidationError):
        return err(
            code="CODEINTEL_SEMANTIC_INVALID_QUERY",
            kind=ErrorKind.invalid_request,
            message="Invalid request payload.",
            retryable=False,
            hint="Check required fields/types and retry.",
            details={
                **details,
                "validation_errors": e.errors()[:10],  # cap to avoid giant payloads
            },
        )

    # 3) Common built-ins you’ll hit
    if isinstance(e, TimeoutError):
        return err(
            code="CODEINTEL_SEMANTIC_QUERY_TIMEOUT",
            kind=ErrorKind.timeout,
            message="Query timed out.",
            retryable=True,
            hint="Retry with narrower filters, smaller limit, or call semantic_explain to inspect the plan.",
            details=details,
        )

    # 4) Map “not found” patterns
    # (Only use these for truly expected cases; otherwise prefer SafeCodeIntelError subclasses.)
    if isinstance(e, KeyError) and context.view_id is not None:
        return err(
            code="CODEINTEL_SEMANTIC_VIEW_NOT_FOUND",
            kind=ErrorKind.not_found,
            message=f"Semantic view '{context.view_id}' not found.",
            retryable=False,
            hint="Call semantic_catalog to list available views, then retry.",
            details=details,
        )

    # 5) Fallback: internal
    return err(
        code="CODEINTEL_SEMANTIC_INTERNAL_ERROR",
        kind=ErrorKind.internal,
        message="Internal error.",
        retryable=True,
        hint="Retry. If this persists, inspect server logs using debug_id.",
        details={
            **details,
            "exception_type": type(e).__name__,
        },
    )
```

### Notes on the mapper

* It’s intentionally **safe** (no stack traces).
* It supports both:

  * **explicit domain exceptions** (`SafeCodeIntelError`) → best practice
  * and “best-effort” fallback mapping for generic exceptions

---

## 4) The domain exception set (minimal) that makes mapping clean

To actually use your 20 locked-in codes reliably, define a small set of domain exceptions in the kernel/export store layers.

Example (these are thin wrappers; they’re worth it):

```python
# src/codeintel/serving/models/domain_errors.py
from __future__ import annotations
from codeintel.serving.models.errors import ErrorKind
from codeintel.serving.models.error_mapping import SafeCodeIntelError

class SemanticViewNotFound(SafeCodeIntelError):
    def __init__(self, *, view_id: str) -> None:
        super().__init__(
            code="CODEINTEL_SEMANTIC_VIEW_NOT_FOUND",
            kind=ErrorKind.not_found,
            message=f"Semantic view '{view_id}' not found.",
            retryable=False,
            hint="Call semantic_catalog to list available views, then retry.",
            details={"view_id": view_id},
        )

class ExportExpired(SafeCodeIntelError):
    def __init__(self, *, export_id: str, expires_at: str | None = None) -> None:
        super().__init__(
            code="CODEINTEL_EXPORT_EXPIRED",
            kind=ErrorKind.expired,
            message="Export handle expired.",
            retryable=True,
            hint="Re-run semantic_export to generate a fresh handle.",
            details={"export_id": export_id, "expires_at": expires_at},
        )

# ...and so on for ExportNotFound, ExportCorrupt, SchemaManifestMissing, etc.
```

This buys you two huge wins:

1. The mapper doesn’t need to know about your internal modules.
2. You can keep error code + hint stable even if exception class names move.

---

## 5) How to call it everywhere (kernel/tools/resources)

### Resources (example: export meta)

Return `ExportMetaResponse(status="error", error=<ErrorResponse>)` rather than raising:

```python
try:
    meta = export_store.meta(export_id)
    return ExportMetaResponse.ready(...)
except Exception as e:
    ctx = ErrorContext(operation="export_meta", export_id=export_id, resource_uri=f"codeintel://exports/{export_id}/meta")
    return ExportMetaResponse.error(export_id=export_id, meta_uri=ctx.resource_uri, error=exception_to_error_response(e, context=ctx))
```

### Tools

Either:

* return `Union[OkModel, ErrorResponse]`, or
* wrap into your `McpEnvelope` approach

Example union:

```python
try:
    out = await kernel.query(...)
    return SemanticQueryToolResponse(...)
except Exception as e:
    ctx = ErrorContext(operation="semantic_query", tool_name="semantic_query_typed", view_id=request.view_id)
    return exception_to_error_response(e, context=ctx)
```

### HTTP routes

If you have RFC 9457 Problem Details on HTTP, you can map `ErrorResponse` into that shape too — still using the same mapper as the source of truth.

---

## 6) Suggested tests (fast and high-value)

Create one parametrized test that ensures the code list **never changes by accident**:

* `tests/serving/test_error_code_contract.py`:

  * construct each domain exception
  * run through mapper
  * assert `error.code`, `kind`, `retryable`, and presence of hint
  * snapshot the JSON schema for `ErrorResponse` if you want “hard” contracts

This is the “make drift impossible” gate.

---

If you want, I can also provide a ready-to-drop `ERROR_CODE_CATALOG: dict[str, ErrorInfoTemplate]` (single dict that defines *all* 20 codes in one place) and have each domain exception simply reference it — so the *text* of messages/hints is also centralized and consistent.


# ERROR_CODE_CATALOG #


Below is a **ready-to-drop error code catalog** pattern that centralizes:

* the **canonical text** (message + hint),
* the **canonical kind/retryability**,
* optional **HTTP status** (useful if you also expose HTTP endpoints),
* and provides a single helper `error_from_code(...)` that everything (tools/resources/kernel) can call.

It also shows how to implement **domain exceptions that simply reference the catalog** (no duplicated strings).

---

## 1) `ErrorInfoTemplate` + `ERROR_CODE_CATALOG`

**Suggested location**

* `src/codeintel/serving/models/error_catalog.py`
  (or move to `src/codeintel/core/models/` if you want build+serving reuse)

```python
# src/codeintel/serving/models/error_catalog.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from codeintel.serving.models.errors import ErrorKind, ErrorResponse, err


class _SafeFormatDict(dict[str, Any]):
    """Format helper: missing keys keep '{key}' rather than raising KeyError."""
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _fmt(template: str, params: Mapping[str, Any] | None) -> str:
    if not params:
        return template
    return template.format_map(_SafeFormatDict(params))


@dataclass(frozen=True, slots=True)
class ErrorInfoTemplate:
    """
    Central template for an error code.

    message/hint can be format() templates:
      "Semantic view '{view_id}' not found."
    """
    code: str
    kind: ErrorKind
    message: str
    hint: str | None = None
    retryable: bool = False

    # Optional: makes it trivial to reuse the same error contract for HTTP routes.
    http_status: int | None = None

    def render_message(self, params: Mapping[str, Any] | None = None) -> str:
        return _fmt(self.message, params)

    def render_hint(self, params: Mapping[str, Any] | None = None) -> str | None:
        return _fmt(self.hint, params) if self.hint else None


# ---- Canonical catalog (LOCK THIS IN) ----

ERROR_CODE_CATALOG: dict[str, ErrorInfoTemplate] = {
    # -------- Semantic/query layer (8) --------
    "CODEINTEL_SEMANTIC_VIEW_NOT_FOUND": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_VIEW_NOT_FOUND",
        kind=ErrorKind.not_found,
        http_status=404,
        retryable=False,
        message="Semantic view '{view_id}' not found.",
        hint="Call semantic_catalog to list available views, then retry.",
    ),
    "CODEINTEL_SEMANTIC_INVALID_QUERY": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_INVALID_QUERY",
        kind=ErrorKind.invalid_request,
        http_status=400,
        retryable=False,
        message="Invalid semantic query request.",
        hint="Validate request fields/types and compare against semantic_describe(view_id).",
    ),
    "CODEINTEL_SEMANTIC_INVALID_FILTER": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_INVALID_FILTER",
        kind=ErrorKind.invalid_request,
        http_status=400,
        retryable=False,
        message="Invalid filter specification.",
        hint="Check filter operators and values against the view schema (semantic_describe).",
    ),
    "CODEINTEL_SEMANTIC_COLUMN_NOT_FOUND": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_COLUMN_NOT_FOUND",
        kind=ErrorKind.invalid_request,
        http_status=400,
        retryable=False,
        message="Column '{column}' not found in semantic view '{view_id}'.",
        hint="Use semantic_describe(view_id) to list available columns.",
    ),
    "CODEINTEL_SEMANTIC_LIMIT_EXCEEDED": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_LIMIT_EXCEEDED",
        kind=ErrorKind.invalid_request,
        http_status=400,
        retryable=False,
        message="Requested limit {limit} exceeds maximum {max_limit}.",
        hint="Lower the limit or use export/preview flows for large results.",
    ),
    "CODEINTEL_SEMANTIC_QUERY_TIMEOUT": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_QUERY_TIMEOUT",
        kind=ErrorKind.timeout,
        http_status=504,
        retryable=True,
        message="Semantic query timed out.",
        hint="Retry with narrower filters and/or a smaller limit; use semantic_explain for plan inspection.",
    ),
    "CODEINTEL_SEMANTIC_QUERY_UNAVAILABLE": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_QUERY_UNAVAILABLE",
        kind=ErrorKind.unavailable,
        http_status=503,
        retryable=True,
        message="Semantic query backend is temporarily unavailable.",
        hint="Retry shortly. If this persists, check serving snapshot state (serving_meta).",
    ),
    "CODEINTEL_SEMANTIC_INTERNAL_ERROR": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_INTERNAL_ERROR",
        kind=ErrorKind.internal,
        http_status=500,
        retryable=True,
        message="Internal error while processing semantic request.",
        hint="Retry. If it persists, inspect server logs using debug_id.",
    ),

    # -------- Export subsystem (6) --------
    "CODEINTEL_EXPORT_NOT_FOUND": ErrorInfoTemplate(
        code="CODEINTEL_EXPORT_NOT_FOUND",
        kind=ErrorKind.not_found,
        http_status=404,
        retryable=False,
        message="Export handle '{export_id}' not found.",
        hint="Re-run semantic_query/export to generate a fresh export handle.",
    ),
    "CODEINTEL_EXPORT_EXPIRED": ErrorInfoTemplate(
        code="CODEINTEL_EXPORT_EXPIRED",
        kind=ErrorKind.expired,
        http_status=410,
        retryable=True,
        message="Export handle '{export_id}' expired.",
        hint="Re-run semantic_query/export with the same parameters to generate a fresh handle.",
    ),
    "CODEINTEL_EXPORT_CORRUPT": ErrorInfoTemplate(
        code="CODEINTEL_EXPORT_CORRUPT",
        kind=ErrorKind.corrupt,
        http_status=500,
        retryable=True,
        message="Export '{export_id}' is corrupt or unreadable.",
        hint="Re-run semantic_export to regenerate; if it repeats, rebuild the serving snapshot.",
    ),
    "CODEINTEL_EXPORT_TOO_LARGE": ErrorInfoTemplate(
        code="CODEINTEL_EXPORT_TOO_LARGE",
        kind=ErrorKind.invalid_request,
        http_status=400,
        retryable=False,
        message="Export request is too large (rows/bytes exceed server limits).",
        hint="Add filters and/or reduce limit; consider exporting narrower slices.",
    ),
    "CODEINTEL_EXPORT_UNAVAILABLE": ErrorInfoTemplate(
        code="CODEINTEL_EXPORT_UNAVAILABLE",
        kind=ErrorKind.unavailable,
        http_status=503,
        retryable=True,
        message="Export store is temporarily unavailable.",
        hint="Retry shortly. If it persists, check disk space and export cleanup/TTL settings.",
    ),
    "CODEINTEL_EXPORT_INTERNAL_ERROR": ErrorInfoTemplate(
        code="CODEINTEL_EXPORT_INTERNAL_ERROR",
        kind=ErrorKind.internal,
        http_status=500,
        retryable=True,
        message="Internal error while retrieving export.",
        hint="Retry. If it persists, inspect server logs using debug_id.",
    ),

    # -------- Serving snapshot / DB (4) --------
    "CODEINTEL_SERVING_SNAPSHOT_NOT_MOUNTED": ErrorInfoTemplate(
        code="CODEINTEL_SERVING_SNAPSHOT_NOT_MOUNTED",
        kind=ErrorKind.unavailable,
        http_status=503,
        retryable=True,
        message="Serving snapshot is not mounted.",
        hint="Wait for snapshot publication or restart the server; check serving_meta.",
    ),
    "CODEINTEL_SERVING_SNAPSHOT_MISMATCH": ErrorInfoTemplate(
        code="CODEINTEL_SERVING_SNAPSHOT_MISMATCH",
        kind=ErrorKind.conflict,
        http_status=409,
        retryable=True,
        message="Requested snapshot does not match the mounted serving snapshot.",
        hint="Refresh snapshot metadata (serving_meta) and retry with the current snapshot identifiers.",
    ),
    "CODEINTEL_SERVING_DB_LOCKED": ErrorInfoTemplate(
        code="CODEINTEL_SERVING_DB_LOCKED",
        kind=ErrorKind.unavailable,
        http_status=503,
        retryable=True,
        message="Serving database is busy/locked.",
        hint="Retry shortly (backoff). Consider lowering concurrency.",
    ),
    "CODEINTEL_SERVING_DB_INTERNAL_ERROR": ErrorInfoTemplate(
        code="CODEINTEL_SERVING_DB_INTERNAL_ERROR",
        kind=ErrorKind.internal,
        http_status=500,
        retryable=True,
        message="Internal serving database error.",
        hint="Retry. If it persists, inspect server logs using debug_id.",
    ),

    # -------- Schema/spec/auth (2) --------
    "CODEINTEL_SCHEMA_MANIFEST_MISSING": ErrorInfoTemplate(
        code="CODEINTEL_SCHEMA_MANIFEST_MISSING",
        kind=ErrorKind.unavailable,
        http_status=503,
        retryable=True,
        message="Schema manifest missing for the mounted snapshot.",
        hint="Rebuild/publish the snapshot or rerun schema compilation.",
    ),
    # Note: if you don't want a new ErrorKind for forbidden, keep kind=invalid_request but http_status=403.
    "CODEINTEL_AUTH_FORBIDDEN": ErrorInfoTemplate(
        code="CODEINTEL_AUTH_FORBIDDEN",
        kind=ErrorKind.invalid_request,
        http_status=403,
        retryable=False,
        message="Forbidden.",
        hint="This operation is not permitted by server policy/settings.",
    ),
}
```

---

## 2) Single helper: `error_from_code(...)`

This is the function everything should call (tools/resources/kernel). It guarantees consistent message/hint text and consistent retryability.

```python
# src/codeintel/serving/models/error_catalog.py (continued)

from codeintel.serving.models.error_mapping import ErrorContext, _base_details


def error_from_code(
    code: str,
    *,
    context: ErrorContext,
    # used to format message/hint templates
    params: Mapping[str, Any] | None = None,
    # extra safe details to attach (merged with context details)
    details: Mapping[str, Any] | None = None,
    # optional overrides (rare; keep to a minimum)
    override_message: str | None = None,
    override_hint: str | None = None,
    override_retryable: bool | None = None,
    override_kind: ErrorKind | None = None,
) -> ErrorResponse:
    tmpl = ERROR_CODE_CATALOG.get(code) or ERROR_CODE_CATALOG["CODEINTEL_SEMANTIC_INTERNAL_ERROR"]

    base = _base_details(context)
    if details:
        base.update({k: v for k, v in dict(details).items() if v is not None})

    message = override_message or tmpl.render_message(params)
    hint = override_hint or tmpl.render_hint(params)

    return err(
        code=tmpl.code,
        kind=override_kind or tmpl.kind,
        message=message,
        retryable=override_retryable if override_retryable is not None else tmpl.retryable,
        hint=hint,
        details=base,
    )
```

---

## 3) Domain exceptions that “simply reference the catalog”

### Base exception

```python
# src/codeintel/serving/models/domain_errors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from codeintel.serving.models.error_mapping import ErrorContext
from codeintel.serving.models.errors import ErrorResponse
from codeintel.serving.models.error_catalog import error_from_code


@dataclass(frozen=True, slots=True)
class CodeIntelDomainError(Exception):
    """
    Domain error that maps to a stable error code.

    No duplicated strings — message/hint/kind live in ERROR_CODE_CATALOG.
    """
    code: str
    params: Mapping[str, Any] | None = None
    details: Mapping[str, Any] | None = None

    def to_error_response(self, *, context: ErrorContext) -> ErrorResponse:
        return error_from_code(
            self.code,
            context=context,
            params=self.params,
            details=self.details,
        )
```

### A few concrete domain exceptions (thin wrappers)

```python
class SemanticViewNotFound(CodeIntelDomainError):
    def __init__(self, view_id: str) -> None:
        super().__init__(
            code="CODEINTEL_SEMANTIC_VIEW_NOT_FOUND",
            params={"view_id": view_id},
        )


class SemanticColumnNotFound(CodeIntelDomainError):
    def __init__(self, view_id: str, column: str) -> None:
        super().__init__(
            code="CODEINTEL_SEMANTIC_COLUMN_NOT_FOUND",
            params={"view_id": view_id, "column": column},
        )


class ExportExpired(CodeIntelDomainError):
    def __init__(self, export_id: str, *, expires_at: str | None = None) -> None:
        super().__init__(
            code="CODEINTEL_EXPORT_EXPIRED",
            params={"export_id": export_id},
            details={"expires_at": expires_at},
        )
```

You don’t need 20 classes — typically you’ll have ~8–12 common ones, and for everything else you can raise the base class with just `code=...`.

---

## 4) Update `exception_to_error_response` to use catalog + domain errors

In your mapper (kernel/tools/resources), add:

```python
from codeintel.serving.models.domain_errors import CodeIntelDomainError

def exception_to_error_response(e: Exception, *, context: ErrorContext) -> ErrorResponse:
    if isinstance(e, CodeIntelDomainError):
        return e.to_error_response(context=context)

    # ValidationError, TimeoutError, etc...
    # Use error_from_code(...) for consistency even in these branches:
    if isinstance(e, TimeoutError):
        return error_from_code("CODEINTEL_SEMANTIC_QUERY_TIMEOUT", context=context)

    # Fallback:
    return error_from_code(
        "CODEINTEL_SEMANTIC_INTERNAL_ERROR",
        context=context,
        details={"exception_type": type(e).__name__},
    )
```

This ensures **all paths** go through the centralized catalog.

---

## 5) Minimal “contract gate” test (prevents drift)

Add a test that asserts the catalog is complete and codes are stable:

```python
# tests/serving/test_error_catalog_contract.py
from codeintel.serving.models.error_catalog import ERROR_CODE_CATALOG

EXPECTED_CODES = {
    "CODEINTEL_SEMANTIC_VIEW_NOT_FOUND",
    "CODEINTEL_SEMANTIC_INVALID_QUERY",
    "CODEINTEL_SEMANTIC_INVALID_FILTER",
    "CODEINTEL_SEMANTIC_COLUMN_NOT_FOUND",
    "CODEINTEL_SEMANTIC_LIMIT_EXCEEDED",
    "CODEINTEL_SEMANTIC_QUERY_TIMEOUT",
    "CODEINTEL_SEMANTIC_QUERY_UNAVAILABLE",
    "CODEINTEL_SEMANTIC_INTERNAL_ERROR",
    "CODEINTEL_EXPORT_NOT_FOUND",
    "CODEINTEL_EXPORT_EXPIRED",
    "CODEINTEL_EXPORT_CORRUPT",
    "CODEINTEL_EXPORT_TOO_LARGE",
    "CODEINTEL_EXPORT_UNAVAILABLE",
    "CODEINTEL_EXPORT_INTERNAL_ERROR",
    "CODEINTEL_SERVING_SNAPSHOT_NOT_MOUNTED",
    "CODEINTEL_SERVING_SNAPSHOT_MISMATCH",
    "CODEINTEL_SERVING_DB_LOCKED",
    "CODEINTEL_SERVING_DB_INTERNAL_ERROR",
    "CODEINTEL_SCHEMA_MANIFEST_MISSING",
    "CODEINTEL_AUTH_FORBIDDEN",
}

def test_error_catalog_codes_are_locked_in():
    assert set(ERROR_CODE_CATALOG.keys()) == EXPECTED_CODES
    for code, tmpl in ERROR_CODE_CATALOG.items():
        assert tmpl.code == code
        assert tmpl.message
```

This makes “oops we renamed a code” impossible to do accidentally.

---

If you want, next I can show the tiny refactor to your `ExportMetaResponse` / `semantic_query_typed` paths so that **every error** they emit uses `error_from_code(...)` (including the “status=error” resource envelope), and you never have mismatched messages or hint wording across tools vs resources.


# refactor to ExportMetaResponse` / `semantic_query_typed` paths so that **every error** they emit uses `error_from_code(...)`#

Below is the **surgical “tiny refactor”** I’d do to guarantee that **both**:

1. `semantic_query_typed` (tool)
2. `ExportMetaResponse` / `codeintel://exports/{id}/meta` (resource)

…**only ever emit errors via `error_from_code(...)`**, so **message + hint wording can’t drift** across tools vs resources.

This refactor also plays very nicely with `mask_error_details=True` (you keep masking ON, and still deliver stable, user-friendly error payloads). FastMCP supports masking + passing controlled messages via ToolError if you ever need that fallback. 

---

## The invariants we’re enforcing

### Invariant 1: Centralized wording

* **No path** constructs `"message"` or `"hint"` ad-hoc.
* The *only place* message/hint strings live is `ERROR_CODE_CATALOG`.

### Invariant 2: One constructor for error payloads

* All errors in these surfaces are created via:

  * `error_from_code(code, *, context=..., params=..., details=...)`
  * or `exception_to_error_response(e, *, context=...)` which **internally calls** `error_from_code(...)`.

### Invariant 3: Tools + resources use the same `ErrorResponse`

FastMCP tools returning dicts/Pydantic models produce machine-readable structured JSON (“structuredContent”), so a stable error model is ideal for LLM agents. 
FastMCP resources can also return dict/Pydantic payloads and will be JSON-serialized as `application/json`, which makes the same error envelope strategy work for resources too. 

---

## Minimal models (the two envelopes you’re refactoring)

### 1) `ExportMetaResponse` (resource envelope)

**Goal:** `ExportMetaResponse(status="error", error=<ErrorResponse>)` — and the `error` must come from `error_from_code(...)`.

```python
# src/codeintel/serving/mcp/models.py (or wherever your MCP Pydantic models live)

from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["error"] = "error"
    code: str
    title: str
    message: str
    hint: str | None = None
    retryable: bool = False

    # debugging / correlation
    debug_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] | None = None


class ExportMetaResponse(BaseModel):
    """Returned by resource: codeintel://exports/{id}/meta"""
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "error"]
    export_id: str

    # ok fields
    format: str | None = None
    row_count: int | None = None
    created_at: str | None = None
    expires_at: str | None = None
    uris: dict[str, str] | None = None

    # error fields
    error: ErrorResponse | None = None

    @classmethod
    def ok(cls, *, export_id: str, **meta: Any) -> "ExportMetaResponse":
        return cls(status="ok", export_id=export_id, error=None, **meta)

    @classmethod
    def err(cls, *, export_id: str, error: ErrorResponse) -> "ExportMetaResponse":
        return cls(status="error", export_id=export_id, error=error)
```

### 2) `SemanticQueryToolResponse` (typed tool envelope)

Same principle: **never** embed ad-hoc strings.

```python
class SemanticQueryToolResponse(BaseModel):
    """Returned by tool: semantic_query_typed"""
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "error"]

    # ok fields
    view_id: str | None = None
    preview: list[dict[str, Any]] | None = None
    preview_columns: list[str] | None = None
    truncated: bool | None = None

    # export handles/uris
    exports: list[dict[str, Any]] | None = None  # or list[ExportHandleResponse]

    # error fields
    error: ErrorResponse | None = None

    @classmethod
    def ok(
        cls,
        *,
        view_id: str,
        preview: list[dict[str, Any]],
        preview_columns: list[str],
        truncated: bool,
        exports: list[dict[str, Any]] | None = None,
    ) -> "SemanticQueryToolResponse":
        return cls(
            status="ok",
            view_id=view_id,
            preview=preview,
            preview_columns=preview_columns,
            truncated=truncated,
            exports=exports or [],
            error=None,
        )

    @classmethod
    def err(cls, *, error: ErrorResponse) -> "SemanticQueryToolResponse":
        return cls(status="error", error=error)
```

---

## The tiny refactor: route all errors through `error_from_code`

### Step A — ensure you have a single centralized constructor

Put this in `src/codeintel/serving/mcp/errors.py` (or `core/errors.py` if you’re already centralizing it beyond serving).

```python
# src/codeintel/serving/mcp/errors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .models import ErrorResponse

@dataclass(frozen=True, slots=True)
class ErrorInfoTemplate:
    title: str
    message: str
    hint: str | None = None
    retryable: bool = False

def _render(template: str, params: dict[str, Any] | None) -> str:
    if not template:
        return template
    params = params or {}
    try:
        return template.format(**params)
    except Exception:
        # Never explode trying to format error strings.
        return template

ERROR_CODE_CATALOG: dict[str, ErrorInfoTemplate] = {
    # --- semantic query ---
    "CI_SEMANTIC_VIEW_NOT_FOUND": ErrorInfoTemplate(
        title="Semantic view not found",
        message="No semantic view named '{view_id}' exists.",
        hint="Call semantic_catalog to list valid view IDs, then retry.",
        retryable=False,
    ),
    "CI_SEMANTIC_INVALID_REQUEST": ErrorInfoTemplate(
        title="Invalid semantic query request",
        message="The semantic query request is invalid.",
        hint="Fix the request fields and retry (check filter columns and operators).",
        retryable=False,
    ),
    "CI_SEMANTIC_LIMIT_EXCEEDED": ErrorInfoTemplate(
        title="Query limit exceeded",
        message="Requested limit {limit} exceeds max {max_limit}.",
        hint="Lower limit or use export flow for large results.",
        retryable=False,
    ),

    # --- exports ---
    "CI_EXPORT_NOT_FOUND": ErrorInfoTemplate(
        title="Export not found",
        message="No export handle '{export_id}' exists.",
        hint="Re-run the semantic query to create a new export handle.",
        retryable=False,
    ),
    "CI_EXPORT_EXPIRED": ErrorInfoTemplate(
        title="Export expired",
        message="Export handle '{export_id}' has expired.",
        hint="Re-run the query to create a fresh export handle.",
        retryable=False,
    ),
    "CI_EXPORT_CORRUPT": ErrorInfoTemplate(
        title="Export corrupt",
        message="Export handle '{export_id}' is corrupt or unreadable.",
        hint="Re-run the query to regenerate the export; if it persists, rebuild the snapshot.",
        retryable=False,
    ),

    # --- generic ---
    "CI_INTERNAL_ERROR": ErrorInfoTemplate(
        title="Internal error",
        message="An unexpected error occurred.",
        hint="Retry; if it persists, inspect server logs using debug_id.",
        retryable=True,
    ),
}

def error_from_code(
    code: str,
    *,
    context: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
    debug_id: str | None = None,
) -> ErrorResponse:
    tmpl = ERROR_CODE_CATALOG.get(code) or ERROR_CODE_CATALOG["CI_INTERNAL_ERROR"]
    did = debug_id or str(uuid4())

    return ErrorResponse(
        code=code,
        title=tmpl.title,
        message=_render(tmpl.message, params),
        hint=_render(tmpl.hint or "", params) or None,
        retryable=tmpl.retryable,
        debug_id=did,
        context=context or {},
        details=details,
    )

def exception_to_error_response(e: Exception, *, context: dict[str, Any]) -> ErrorResponse:
    # Map known domain exceptions to codes; everything else becomes CI_INTERNAL_ERROR.
    # (You likely already have domain exception classes; if so, map them here.)
    return error_from_code(
        "CI_INTERNAL_ERROR",
        context=context,
        details={"exception_type": type(e).__name__},
    )
```

> Note: if you already have domain exceptions that carry `.code` + `.params`, then `exception_to_error_response` should detect that and call `error_from_code(e.code, params=e.params, ...)`. That’s the cleanest approach.

---

## Step B — refactor `semantic_query_typed` to only emit `SemanticQueryToolResponse.err(error_from_code(...))`

The key is: **no `ToolError("...")`** for this typed tool path, and **no ad-hoc strings**.

```python
# src/codeintel/serving/mcp/app.py (or wherever semantic_query_typed is defined)

import anyio
from pydantic import ValidationError
from mcp.server.fastmcp import Context

from codeintel.serving.mcp.errors import error_from_code, exception_to_error_response
from codeintel.serving.mcp.models import SemanticQueryToolResponse
from codeintel.serving.semantic.models import SemanticQueryRequest, FilterSpec

# Example only: use your real request model if you already have SemanticQueryToolRequest
async def semantic_query_typed(
    view_id: str,
    filters: list[dict[str, object]] | None = None,
    select: list[str] | None = None,
    order_by: list[str] | None = None,
    limit: int = 200,
    offset: int = 0,
    ctx: Context | None = None,
) -> SemanticQueryToolResponse:
    err_ctx = {
        "op": "semantic_query_typed",
        "view_id": view_id,
        "tool": "semantic_query_typed",
    }

    try:
        # Optional: enforce serving-layer max limit (distinct from model validators)
        max_limit = 10_000  # pull from settings
        if limit > max_limit:
            return SemanticQueryToolResponse.err(
                error=error_from_code(
                    "CI_SEMANTIC_LIMIT_EXCEEDED",
                    context=err_ctx,
                    params={"limit": limit, "max_limit": max_limit},
                )
            )

        request = SemanticQueryRequest(
            view_id=view_id,
            select=select,
            filters=[FilterSpec.model_validate(f) for f in (filters or [])],
            order_by=order_by or [],
            limit=limit,
            offset=offset,
        )

        # Offload kernel query to threadpool (DuckDB is CPU-bound)
        result = await anyio.to_thread.run_sync(kernel.query, request)

        # Build preview + optional exports
        preview_rows = result.rows[: min(len(result.rows), 50)]
        preview_cols = result.columns

        return SemanticQueryToolResponse.ok(
            view_id=result.view_id,
            preview=preview_rows,
            preview_columns=preview_cols,
            truncated=result.truncated,
            exports=[],  # populate if you support export creation on query
        )

    except KeyError:
        # Map “view not found” to cataloged error
        return SemanticQueryToolResponse.err(
            error=error_from_code(
                "CI_SEMANTIC_VIEW_NOT_FOUND",
                context=err_ctx,
                params={"view_id": view_id},
            )
        )
    except ValidationError as e:
        # ValidationError wording can drift; don’t expose it as “message”
        return SemanticQueryToolResponse.err(
            error=error_from_code(
                "CI_SEMANTIC_INVALID_REQUEST",
                context=err_ctx,
                details={"validation_errors": e.errors()},
            )
        )
    except Exception as e:
        return SemanticQueryToolResponse.err(
            error=exception_to_error_response(e, context=err_ctx)
        )
```

Why this is “best-in-class” for agents:

* Agents don’t have to parse exception strings.
* The shape is stable and typed.
* You can keep `mask_error_details=True` as a backstop while still returning consistent error payloads (FastMCP masking behavior is well-supported). 

---

## Step C — refactor `export_meta` resource to return `ExportMetaResponse.err(error_from_code(...))`

FastMCP resources can return dict/Pydantic and it’ll emit JSON content (`application/json`). 
So don’t raise exceptions; return the envelope.

```python
# src/codeintel/serving/mcp/resources.py (or wherever you define resources)

from codeintel.serving.mcp.errors import error_from_code, exception_to_error_response
from codeintel.serving.mcp.models import ExportMetaResponse

@mcp.resource("codeintel://exports/{export_id}/meta")
def export_meta(export_id: str) -> dict[str, object]:
    err_ctx = {
        "op": "export_meta",
        "export_id": export_id,
        "resource": f"codeintel://exports/{export_id}/meta",
    }

    try:
        meta = kernel.export_store.get_meta(export_id)  # your actual store
        payload = ExportMetaResponse.ok(
            export_id=export_id,
            format=meta.format,
            row_count=meta.row_count,
            created_at=meta.created_at,
            expires_at=meta.expires_at,
            uris=meta.uris,
        )
        return payload.model_dump(mode="json", exclude_none=True)

    except ExportNotFound:
        return ExportMetaResponse.err(
            export_id=export_id,
            error=error_from_code(
                "CI_EXPORT_NOT_FOUND",
                context=err_ctx,
                params={"export_id": export_id},
            ),
        ).model_dump(mode="json", exclude_none=True)

    except ExportExpired:
        return ExportMetaResponse.err(
            export_id=export_id,
            error=error_from_code(
                "CI_EXPORT_EXPIRED",
                context=err_ctx,
                params={"export_id": export_id},
            ),
        ).model_dump(mode="json", exclude_none=True)

    except ExportCorrupt:
        return ExportMetaResponse.err(
            export_id=export_id,
            error=error_from_code(
                "CI_EXPORT_CORRUPT",
                context=err_ctx,
                params={"export_id": export_id},
            ),
        ).model_dump(mode="json", exclude_none=True)

    except Exception as e:
        return ExportMetaResponse.err(
            export_id=export_id,
            error=exception_to_error_response(e, context=err_ctx),
        ).model_dump(mode="json", exclude_none=True)
```

This guarantees:

* `status="error"` always exists for errors.
* `error.message`/`error.hint` are always from the catalog.
* No mismatch between tool and resource phrasing.

---

## The legacy code to delete as part of this refactor

Search and remove any of the following patterns **in these two paths**:

1. **Manual strings**:

* `message=f"..."`, `hint="..."`, `detail=str(e)` being returned to clients as primary message text.

2. **Ad-hoc dict error shapes**:

* `{"status": "error", "message": "...", ...}` (replace with `ErrorResponse` from `error_from_code`).

3. **`ToolError(...)` in typed endpoints**:

* For typed tools/resources, prefer returning the typed envelope.
* Keep `ToolError` only for *compat* tools where you intentionally want exception-style semantics (and even then: build the message from the catalog to avoid drift). FastMCP explicitly supports ToolError as the “controlled message” escape hatch when masking is enabled. 

---

## Tests to lock this in (small but very high value)

Add one focused test module like:

### `tests/serving/mcp/test_error_unification.py`

```python
def test_semantic_query_typed_view_not_found_uses_catalog():
    resp = semantic_query_typed(view_id="does.not.exist", ctx=None)
    assert resp.status == "error"
    assert resp.error is not None
    assert resp.error.code == "CI_SEMANTIC_VIEW_NOT_FOUND"
    # Ensure message/hint are from catalog, not ad-hoc:
    assert "does.not.exist" in resp.error.message
    assert resp.error.hint is not None

def test_export_meta_not_found_uses_catalog():
    payload = export_meta(export_id="missing123")
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "CI_EXPORT_NOT_FOUND"
    assert "missing123" in payload["error"]["message"]
```

Optional “guardrail” test:

* iterate every domain exception class that claims an `error_code`, and assert it exists in `ERROR_CODE_CATALOG` (prevents dead codes).

---

## One small recommendation while you’re touching these functions

Add MCP annotations to these read-only tools so LLM clients behave better (skip confirmations, treat as safe). This is directly supported by FastMCP’s “annotations” metadata. 

Even if you already do this globally, it’s worth ensuring `semantic_query_typed` is marked read-only and idempotent.

---



