

Below is what I would change/add to your **FASTMCP_UVICORN_BEST_IN_CLASS_PLAN.md** after reviewing:

* your current `src/codeintel/serving/` implementation (FastAPI app + `/mcp` mount, `ServingDBManager`, `SemanticKernel`, schema/registry artifacts, etc.), and
* your FastMCP plan + advanced guide (Context, annotations, structured outputs, prompts/resources, composition), and
* current upstream FastMCP/MCP deployment guidance + Uvicorn + DuckDB concurrency constraints.   ([FastMCP][1])

I’m going to keep this **practical and plan-shaped**: concrete “add this” / “change that” items, why they matter, and representative code sketches.

---

## 1) Add a “FastMCP version normalization” decision up-front

### Why this matters

Your current server code imports `FastMCP` from `mcp.server.fastmcp` (the MCP Python SDK flavor). Your plan + advanced guide discuss **FastMCP 2.x** features (tool annotations, structured outputs/output schemas, `ToolResult` meta, dependency injection, etc.) that are most cleanly supported in the actively maintained FastMCP 2.x ecosystem. FastMCP’s own docs explicitly say moving from the MCP SDK FastMCP to FastMCP 2.0 is often just an import change. ([FastMCP][2])

### Plan change

Add a **Phase 0 / H0** item:

**H0: Normalize FastMCP runtime to FastMCP 2.x (or explicitly pin to MCP SDK FastMCP and accept feature limits).**

If you go FastMCP 2.x:

* update imports (`from fastmcp import FastMCP`) as recommended ([FastMCP][2])
* ensure your dependency pins satisfy FastMCP’s protocol target requirements (FastMCP 2.14 targeting the 2025‑11‑25 MCP spec implies `mcp` min versions in practice; issues/notes indicate this direction) ([GitHub][3])
* re-validate your HTTP mounting semantics (`http_app()` defaults vs your `/mcp` mount; see §2 below) ([FastMCP][1])

If you stay on MCP SDK FastMCP:

* you can still do Context, stateless HTTP mode, multi-mount, event stores, etc.
* but you should adjust expectations on tool annotations + output schemas + some of the “rich semantics” described in your guide.

**My recommendation:** adopt FastMCP 2.x as the “best-in-class” path, because your plan explicitly wants those newer semantics (annotations, output schemas) and FastMCP positions 2.0 as the actively maintained standard framework. ([PyPI][4])

---

## 2) Fix mount-path semantics now, before you encode more assumptions

Right now you mount MCP under `/mcp` inside your FastAPI app, and you pass `streamable_http_path="/"` to your builder (so the MCP ASGI app’s root is `/`). That’s good.

But as soon as you switch to FastMCP 2.x’s `http_app()` model (or even use MCP SDK multi-mount patterns), you need to be deliberate because some servers default to exposing the MCP endpoint at `/mcp`. ([FastMCP][1])

### Plan change

Add a small explicit “mount contract” section:

* **FastAPI exposes** `/mcp/*`
* **MCP ASGI app path inside the mount** is `/`
* Therefore: **effective MCP endpoint** is `/mcp`

### Representative code sketch (what you want long-term)

```python
# src/codeintel/serving/http/app.py
def _maybe_mount_mcp(app: FastAPI, kernel: SemanticKernel, settings: ServingSettings) -> None:
    if not settings.enable_mcp:
        return

    mcp = build_mcp_app(kernel=kernel)
    # Important: the MCP app should mount at "/" inside the "/mcp" prefix.
    app.mount("/mcp", mcp.http_app(path="/"))  # if FastMCP 2.x
    # or app.mount("/mcp", mcp.streamable_http_app("/"))  # MCP SDK style
```

This prevents “/mcp/mcp” surprises when changing frameworks. (Your plan already talks about mount controls; make the invariant explicit.)

---

## 3) Add “SSE polling + event store” as a best-in-class reliability feature (especially for remote)

Your target includes remote internet connections. That’s exactly where long-running calls, transient disconnects, and proxies become painful.

FastMCP’s HTTP deployment docs call out **SSE polling for long-running operations** (StreamableHTTP) and it is explicitly “new in 2.14.0”. ([FastMCP][1])
The MCP SDK also describes Streamable HTTP resumability with event stores.

### Plan addition

Add **H6: Enable resumability for StreamableHTTP with an event store** (even if you keep it file-backed and tiny).

What you get:

* tool calls can survive disconnects/reconnects
* fewer “hung request” failures
* much nicer remote UX for LLM clients

### Representative code sketch

```python
# src/codeintel/serving/mcp/app.py (builder)
def build_mcp_app(...):
    mcp = FastMCP(
        "CodeIntel",
        # FastMCP 2.x: configure transport-level event store if supported,
        # or wire via http_app()/server options depending on API.
    )
    return mcp
```

I’m keeping that snippet abstract because the exact constructor wiring differs between MCP SDK FastMCP vs FastMCP 2.x, but the *plan* should explicitly include “resumability via event store” as a first-class requirement for remote serving.

---

## 4) Tighten your concurrency story: DuckDB thread-safety + multi-process read-only

You’re doing the right architectural thing by using a read-only snapshot DB for serving and by pooling.

But for “best-in-class hardness,” the plan should explicitly codify two constraints:

1. **DuckDB connections are not thread-safe; each thread needs its own connection**
2. **Read-only mode is required if multiple Python processes access the same DuckDB file simultaneously** (e.g., Uvicorn workers > 1)

### Plan change

Extend **H5 (Uvicorn worker config)** to include a DuckDB correctness clause:

* If `workers > 1`, serving must open DuckDB connections with `read_only=True` (or equivalent via your gateway/pool policy).
* The serving pool must guarantee “one connection per executing thread.”

### Add one operational guardrail (very worth it)

Add a **server-side semaphore** limiting concurrent query execution independent of HTTP concurrency. This prevents 3 LLM consumers from accidentally running 3 huge graph queries simultaneously and blowing memory.

Representative snippet (tool wrapper level):

```python
# src/codeintel/serving/mcp/runtime.py
class QueryLimiter:
    def __init__(self, max_concurrent: int) -> None:
        self._sem = anyio.Semaphore(max_concurrent)

    async def run(self, fn, *args, **kwargs):
        async with self._sem:
            return await anyio.to_thread.run_sync(fn, *args, **kwargs)
```

Then every heavy tool call goes through `limiter.run(...)`.

This is “small scale” friendly and dramatically increases stability.

---

## 5) Fix a correctness bug in the PLAN’s example signatures (ctx placement)

In your plan’s H1 example you show `ctx: Context` after defaulted params, which is invalid Python unless you make it keyword-only (or give it a default). 

### Plan correction

Standardize on one of these signatures:

**Option A (recommended): keyword-only ctx**

```python
@mcp.tool()
async def semantic_query(
    view_id: str,
    filters: list[FilterSpec] | None = None,
    *,
    ctx: Context,
) -> SemanticQueryResponse:
    ...
```

**Option B: ctx early**

```python
@mcp.tool()
async def semantic_query(ctx: Context, view_id: str, filters: ... = None) -> ...
```

I strongly recommend Option A for readability.

Also: if you decide to stay with MCP SDK FastMCP, use `from mcp.server.fastmcp import Context` (their docs demonstrate this pattern).
If you go FastMCP 2.x, align with its Context import conventions (your guide uses `fastmcp.Context`), but normalize it in one place so developers don’t guess.

---

## 6) Upgrade your MCP tool layer to *structured input models* (not just “dicts”)

Your current MCP tools accept `filters: list[dict]` and manually validate into `FilterSpec`. That works, but it’s not “best-in-class” for LLM consumers, because you lose the benefit of input JSON schema being explicit.

Your plan focuses on **structured outputs** (great) , but I’d add: **structured inputs** via your existing Pydantic request models.

### Plan addition

Add **H2b: Tool inputs use Pydantic request objects where possible**.

Representative snippet:

```python
# src/codeintel/serving/mcp/app.py
@mcp.tool()
async def semantic_query(
    request: SemanticQueryRequest,
    *,
    ctx: Context,
) -> SemanticQueryResponse:
    return await limiter.run(kernel.query, request)
```

This makes the tool schema vastly clearer to LLMs and reduces your validation boilerplate.

---

## 7) Make “ToolResult meta + schema version + snapshot id” a first-class design goal

For LLM consumers, the single most useful “hardness” improvement is: every response should carry enough metadata to reason about:

* what snapshot/version it came from
* what semantic registry version it used
* whether results were truncated
* how long it took

FastMCP tools docs highlight structured output and (in newer versions) extra metadata support patterns. ([FastMCP][5])
Your plan already wants “meta endpoints” and version visibility. 

### Plan addition

Add **H2c: Every tool response includes a standard meta envelope**.

Even if you return a Pydantic model, make it include:

```python
class ResponseMeta(BaseModel):
    snapshot_id: str
    semantic_registry_version: str
    schema_manifest_hash: str
    truncated: bool = False
    query_ms: int | None = None
```

This becomes *hugely* valuable for agents doing iterative analysis (“did the dataset change under me?”).

---

## 8) Treat “large data” as resources/files, not JSON rows

This is one of the biggest “best-in-class” deltas.

Right now, your export endpoints build `list(kernel.export_rows(...))` in memory, then convert to Arrow/Parquet. That will fall over as soon as a semantic view is big.

DuckDB’s docs explicitly warn that pulling huge relations fully into pandas can OOM, and generally you want streaming/columnar approaches. 
DuckDB also natively supports Parquet/Arrow/CSV I/O patterns. 

### Plan change

Upgrade **M1 (Resources)** into **H7 (Large results as resources/files)**:

* Tools return **small structured data** (counts, top-N, summary)
* For large payloads: tools return a **resource URI** (or file artifact) that the client can fetch explicitly

Representative “export as parquet and return resource” sketch:

```python
@mcp.tool()
async def semantic_export_parquet(
    request: SemanticExportRequest,
    *,
    ctx: Context,
) -> ExportResult:
    # 1) materialize parquet into a temp artifact dir
    # 2) return a resource URI or file handle
    return ExportResult(
        resource_uri=f"codeintel://exports/{export_id}.parquet",
        row_count=...,
    )
```

This is both faster and safer than shoving rows into JSON.

---

## 9) Authentication: keep it simple, but enforce it for remote

Your plan already includes an `auth_token` idea and API key gating on FastAPI routes. 

Two “best-in-class” additions:

1. **Explicitly require auth when bound to non-localhost**
2. Use the underlying framework’s supported auth pathway, so clients behave correctly

FastMCP’s HTTP deployment docs explicitly recommend authentication and note clients may refuse to connect without it. ([FastMCP][1])
The MCP SDK shows how auth can be configured (OAuth provider pattern).

### Plan change

Add a “security posture” rule:

* `host in {"0.0.0.0", "::"}` ⇒ `auth required`
* localhost-only can allow `auth optional`

For your single-user scale, a bearer token is fine; just ensure it applies to `/mcp` as well as `/v1/*`.

---

## 10) Uvicorn plan: add two small but high-impact knobs

Your plan’s H5 has the right idea. 
I’d add two details from Uvicorn’s own deployment/config guidance:

* prefer CLI/config-based startup for multi-worker rather than calling `uvicorn.run()` in library code (Uvicorn documents the configuration methods and caveats for reload/workers).
* if you ever run behind a proxy, include `proxy_headers` / forwarded IP policy explicitly in the plan (otherwise logs + client IPs get weird)

I’d also keep a simple recommended preset table:

**Recommended presets (single machine, 3 consumers)**

* default: `workers=1`, `max_concurrent_queries=2`
* “snappy UI”: `workers=2`, `max_concurrent_queries=2` (total 4 concurrent queries max)
* do **not** set workers high unless you need CPU isolation (DuckDB is fast, but parallel heavy queries can thrash)

And reiterate the DuckDB read-only + per-thread connection requirements here.

---

## The “delta summary” — what I’d literally change in your plan doc

If I were editing your plan file, I’d make these concrete edits:

### Add

* **H0** Normalize on FastMCP 2.x (or explicitly pin to MCP SDK FastMCP) ([FastMCP][2])
* **H6** StreamableHTTP resiliency via event store / SSE polling (for remote)
* **H7** Large results via resources/files, not JSON lists (DuckDB-native Parquet/Arrow) 
* **H2b** Structured tool inputs using Pydantic request models
* **H2c** Standard meta envelope on all tool responses

### Modify

* **H1** Fix ctx signature + normalize Context import style (and decide whether sampling is real/needed) 
* **H5** Explicitly encode DuckDB multi-process read-only + per-thread connection constraints
* **M1** Promote to H7 (large payload strategy), and make resources a central UX concept, not an “optional nice to have”

### Keep (these are already good)

* annotations plan (readOnlyHint, idempotentHint, openWorldHint) because it improves client UX 
* async-first + threadpool offloading for CPU/IO-bound DB work 
* auth token concept + metrics + /health + modular organization 

---

## One final “best-in-class” framing recommendation

Your serving layer will be easiest to maintain if you treat it as three layers:

1. **Runtime**: `ServingDBManager`, snapshot/version metadata, concurrency limiter
2. **SemanticKernel**: “business logic” (semantic registry + Ibis/DuckDB execution + schema gating)
3. **Transports**:

   * FastAPI routes call kernel
   * MCP tools call kernel (same request/response models)
   * MCP resources/prompt templates are “UX sugar” over the same primitives

That layering makes it much harder for HTTP and MCP to drift apart over time.

---

If you want, I can take *your current* `src/codeintel/serving/` tree and produce a **surgical “plan diff”** that says, file-by-file, what needs to change to incorporate the above additions (H0/H6/H7/H2b/H2c), without rewriting your already-good ServingDBManager + kernel structure.

[1]: https://gofastmcp.com/deployment/http "HTTP Deployment - FastMCP"
[2]: https://gofastmcp.com/getting-started/installation?utm_source=chatgpt.com "Installation"
[3]: https://github.com/jlowin/fastmcp/issues/2537?utm_source=chatgpt.com "Ensure 11/25/25 MCP protocol is advertised in 2.14 #2537"
[4]: https://pypi.org/project/fastmcp/?utm_source=chatgpt.com "fastmcp"
[5]: https://gofastmcp.com/servers/tools "Tools - FastMCP"


# Followup surgical plan diff #

Below is the “surgical plan diff” in **mergeable PR style**, mapped **file-by-file**, to incorporate **H0 / H6 / H7 / H2b / H2c** into your current `src/codeintel/serving/` without rewriting your already-good `ServingDBManager` + `SemanticQueryKernel` structure.

I’m assuming the current baseline is exactly what’s in `CodeIntel_Serving_aligned_with_Storage.zip` (notably: you currently import `FastMCP` from the **official** `mcp` SDK, while `pyproject.toml` also includes `fastmcp`).

---

## Executive summary: what changes & why

### The core pivot (H0)

Right now, your code *looks* like it’s “FastMCP”, but it’s actually the **MCP Python SDK**’s `FastMCP` (`from mcp.server.fastmcp import FastMCP`). That creates constant confusion and blocks you from the “best-in-class” feature set you referenced (resources/templates, event store + SSE polling/resumability, strong tool metadata, etc.).

**H0 = pick one MCP framework** and go all-in. The “advanced FastMCP features” you’ve been planning against are from **jlowin/fastmcp** (“gofastmcp”). That library’s API includes:

* `mcp.http_app(...)` for mounting into FastAPI and supporting StreamableHTTP
* `event_store=EventStore()` to enable **SSE polling/resumability** for StreamableHTTP ([FastMCP][1])
* `@mcp.resource("...")` with templates + strong return-type handling (dict/list/BaseModel → JSON, bytes → blob, etc.) ([FastMCP][2])
* tool annotations + tool-level meta for client UX and versioning ([FastMCP][3])

So: **PR-1 makes `fastmcp` the only “FastMCP” you use** (and stops importing the MCP SDK’s `FastMCP`).

### What you get from the rest

* **H2b**: strongly-typed tool inputs (or “typed inputs with compatibility fallbacks”)
* **H2c**: consistent response envelope (`meta` + `data`) so clients always know snapshot/version/context
* **H7**: resources + templates for “big payloads” (export handles) instead of shoving huge rowsets into tool responses
* **H6**: EventStore + SSE polling/resumability hook-up for long-running tools over StreamableHTTP ([FastMCP][1])

---

# PR-by-PR “plan diff” (H0/H2b/H2c/H7/H6)

## PR-S01 — H0: Unify MCP runtime on gofastmcp `fastmcp` (remove MCP SDK FastMCP usage)

### Goal

* Replace `from mcp.server.fastmcp import FastMCP` with `from fastmcp import FastMCP`
* Mount the MCP ASGI app into your FastAPI app correctly (including lifespan handling)
* Reduce ambiguity: one MCP framework in code

### Files changed

#### 1) `pyproject.toml`

**Change**

* Ensure you’re pinned to a FastMCP version that supports EventStore + ResourceContent.

  * SSE polling/EventStore is called out as “New in version 2.14.0” in the FastMCP HTTP docs ([FastMCP][1])
  * `ResourceContent` is documented as “New in version 2.14.1” ([FastMCP][2])

**Recommended**

* Pin: `fastmcp>=2.14.1,<3`
* Remove `mcp[cli]` **if you no longer need the MCP SDK** (you’ll avoid naming collisions/confusion).

  * If you *do* want the MCP SDK CLI/inspector, keep it—but you should stop importing its `FastMCP` anywhere.

---

#### 2) `src/codeintel/serving/mcp/app.py`

**Change**

* Rewrite to build a **gofastmcp server instance** (but keep tool semantics the same).

**Representative snippet (core pivot)**

```py
# src/codeintel/serving/mcp/app.py
from __future__ import annotations

from fastmcp import FastMCP
from codeintel.serving.semantic.kernel import SemanticQueryKernel

def build_mcp_server(*, kernel: SemanticQueryKernel) -> FastMCP:
    mcp = FastMCP(
        name="CodeIntel",
        # optionally: instructions="Semantic query interface for CodeIntel",
        # optionally: version="..." (pull from package version)
    )

    @mcp.tool  # (decorator form shown in docs) :contentReference[oaicite:6]{index=6}
    async def semantic_catalog() -> dict:
        return kernel.catalog().model_dump()

    # ...keep adding tools...
    return mcp
```

> Note: gofastmcp’s docs show `@mcp.tool` and `mcp.http_app()`/`mcp.run(...)` patterns. ([FastMCP][1])

**Delete**

* All imports from `mcp.server.fastmcp` in this module.

---

#### 3) `src/codeintel/serving/mcp/server.py`

**Change**

* Replace your current `.run(transport="streamable-http")` logic with gofastmcp equivalents.
* gofastmcp supports `transport` values including `http`, `streamable-http`, and `sse` in its server methods/docs. ([FastMCP][4])

**Representative snippet**

```py
# src/codeintel/serving/mcp/server.py
from __future__ import annotations

from codeintel.serving.mcp.app import build_mcp_server
from codeintel.serving.semantic.kernel import SemanticQueryKernel

def run_mcp_server(*, kernel: SemanticQueryKernel, transport: str) -> None:
    mcp = build_mcp_server(kernel=kernel)

    if transport == "stdio":
        mcp.run()  # stdio default
        return

    if transport in {"http", "streamable-http", "sse"}:
        # minimal - PR-S04 will introduce EventStore path
        mcp.run(transport="http")  # gofastmcp HTTP mode :contentReference[oaicite:9]{index=9}
        return

    raise ValueError(f"Unknown transport: {transport}")
```

---

#### 4) `src/codeintel/serving/http/app.py`

**Change**

* Replace `mcp.streamable_http_app()` with `mcp.http_app(...)`.
* **Critical:** gofastmcp docs explicitly note lifespan concerns when mounting MCP servers into FastAPI. The FastAPI integration guide shows passing lifespan, and warns “nested lifespans are not recognized” for StreamableHTTP in some contexts. ([FastMCP][5])

**Recommended approach**

* Build MCP server → build MCP ASGI app → combine lifespans → mount.

**Representative snippet (mount + lifespan composition)**

```py
# src/codeintel/serving/http/app.py
from contextlib import asynccontextmanager
from fastapi import FastAPI

from codeintel.serving.mcp.app import build_mcp_server

def create_serving_app(...):
    ...

    mcp_server = build_mcp_server(kernel=state.kernel)
    mcp_asgi = mcp_server.http_app(path="/")  # path is relative to the mount :contentReference[oaicite:11]{index=11}

    @asynccontextmanager
    async def combined_lifespan(app: FastAPI):
        # Start FastMCP session manager (required when mounting) :contentReference[oaicite:12]{index=12}
        async with mcp_asgi.lifespan(app):
            async with _lifespan(app):  # your existing lifespan that starts ServingState
                yield

    app = FastAPI(lifespan=combined_lifespan)
    ...
    app.mount("/mcp", mcp_asgi)
```

That aligns with the FastMCP FastAPI integration guidance about mounting via `http_app` and passing lifespan. ([FastMCP][5])

---

### Tests

#### Update: `tests/serving/test_semantic_mcp_tools.py`

* It currently uses `await mcp.call_tool(...)` (MCP SDK-style).
* gofastmcp’s recommended testing approach uses `fastmcp.client.Client(mcp)` in-memory transport. ([FastMCP][6])

**Representative snippet**

```py
from fastmcp.client import Client

@pytest.mark.anyio
async def test_mcp_tools(...):
    mcp = build_mcp_server(kernel=kernel)

    async with Client(mcp) as client:
        catalog = await client.call_tool("semantic_catalog", {})
        ...
```

---

### Legacy code to delete (as part of PR-S01)

* Remove usage of the MCP SDK’s FastMCP entirely:

  * `from mcp.server.fastmcp import FastMCP`
  * `mcp.streamable_http_app()`
* If nothing else in your repo uses `mcp[cli]`, remove it from dependencies to avoid confusion.

---

## PR-S02 — H2b + H2c: Typed tool I/O + standard response envelope (`meta` + `data`)

### Goal

* Tools should be introspectable and stable:

  * Inputs: no ad-hoc `dict[str, object]` parsing when avoidable
  * Outputs: consistent `meta` envelope for snapshot/version/build info
* Add tool annotations/meta for client UX (readOnly/idempotent hints) ([FastMCP][3])

### Files changed

#### 1) Add `src/codeintel/serving/mcp/models.py`

Define the consistent envelope.

**Representative snippet**

```py
# src/codeintel/serving/mcp/models.py
from __future__ import annotations
from pydantic import BaseModel

class McpSnapshotMeta(BaseModel):
    repo: str
    commit: str
    run_id: str
    published_at: str
    semantic_layer_version: str

class McpResponseMeta(BaseModel):
    snapshot: McpSnapshotMeta
    # optionally:
    # server_version: str
    # warnings: list[str] = []

class McpEnvelope(BaseModel):
    meta: McpResponseMeta
    data: dict  # keep simple; or GenericModel if you want
```

> Why this shape: your system’s “truth” is the snapshot pointer. Returning snapshot meta on every response eliminates “what data am I looking at?” ambiguity.

---

#### 2) Add `src/codeintel/serving/mcp/response.py`

A tiny helper to build meta once, consistently.

**Representative snippet**

```py
# src/codeintel/serving/mcp/response.py
from __future__ import annotations
from codeintel.serving.mcp.models import McpEnvelope, McpResponseMeta, McpSnapshotMeta
from codeintel.serving.semantic.kernel import SemanticQueryKernel

def envelope(kernel: SemanticQueryKernel, data: dict) -> McpEnvelope:
    ptr = kernel.db.current_pointer()
    meta = McpResponseMeta(
        snapshot=McpSnapshotMeta(
            repo=ptr.repo,
            commit=ptr.commit,
            run_id=ptr.run_id,
            published_at=ptr.published_at,
            semantic_layer_version=ptr.semantic_layer_version,
        )
    )
    return McpEnvelope(meta=meta, data=data)
```

---

#### 3) `src/codeintel/serving/semantic/kernel.py`

**Change**

* Add a trivial property exposing `db` (if it isn’t already public in your current branch).

Right now, the dataclass *already* has a `db: ServingDBManager` field, so the helper above works as-is.

---

#### 4) `src/codeintel/serving/mcp/app.py`

**Changes**

* Wrap every tool result in `McpEnvelope`
* Add FastMCP tool annotations to be explicit and LLM-friendly:

  * `readOnlyHint`, `idempotentHint`, etc. ([FastMCP][3])
* Start moving inputs away from “raw dicts” where practical

**Representative snippet**

```py
from fastmcp import FastMCP
from codeintel.serving.mcp.response import envelope

@mcp.tool(
    annotations={"readOnlyHint": True, "idempotentHint": True},
    meta={"domain": "semantic", "kind": "catalog"},
)
async def semantic_catalog() -> dict:
    payload = kernel.catalog().model_dump()
    return envelope(kernel, payload).model_dump()
```

---

### About “typed inputs” (practical best-in-class recommendation)

Even though H2b says “use Pydantic request objects”, be aware: some MCP clients have had issues with complex object parameters (nested schemas). If you want maximum compatibility, a good pattern is:

* keep tool signatures “flat-ish” (primitives + lists)
* parse into your Pydantic models internally (like you do today), **but** document the dict schema carefully in the tool docstring.

If you want to go fully typed anyway, you can:

* either accept `SemanticQueryRequest` directly,
* or ship **two entrypoints**:

  * `semantic_query` (compat)
  * `semantic_query_typed` (strict)

This is a product decision; technically both work.

---

### Tests

Update existing tests that assume raw dict payloads:

* `tests/serving/test_semantic_mcp_tools.py`

  * adapt to `result.data` shape if you use `Client.call_tool` returning a structured result (depends on client API)
  * or simply assert your returned JSON has keys `meta` and `data`

---

### Legacy code to delete

* Any per-tool ad-hoc `{"snapshot": ...}` scattered fields in responses; replace with the single envelope.

---

## PR-S03 — H7: Add semantic resources + export resources (large results by handle, not tool payload)

### Goal

* Stop returning huge result sets directly via tools.
* Use MCP Resources & Templates to deliver:

  * semantic registry
  * schema manifest
  * view schema
  * “export handles” (NDJSON/JSON/Parquet)

FastMCP’s resources model is designed exactly for “read-only data/files”, using `@mcp.resource(...)` and template URIs. ([FastMCP][2])

### Files changed

#### 1) Add `src/codeintel/serving/mcp/resource_store.py`

A simple on-disk artifact store (works great with your `serve_dir`).

**Representative snippet**

```py
# src/codeintel/serving/mcp/resource_store.py
from __future__ import annotations
import json
import secrets
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class StoredArtifact:
    path: Path
    mime_type: str

class ResourceStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def put_json(self, payload: object) -> tuple[str, StoredArtifact]:
        token = secrets.token_urlsafe(16)
        path = self._root / f"{token}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return token, StoredArtifact(path=path, mime_type="application/json")

    def get(self, token: str) -> StoredArtifact:
        # simplest: infer by scanning known extensions
        json_path = self._root / f"{token}.json"
        if json_path.exists():
            return StoredArtifact(path=json_path, mime_type="application/json")
        raise KeyError(token)
```

---

#### 2) Add `src/codeintel/serving/mcp/resources.py`

Register resource templates.

**Representative snippet**

```py
# src/codeintel/serving/mcp/resources.py
from __future__ import annotations
from fastmcp.resources import ResourceContent
from codeintel.serving.mcp.resource_store import ResourceStore

def register_resources(mcp, *, kernel, store: ResourceStore) -> None:
    @mcp.resource("codeintel://semantic/registry")
    def semantic_registry() -> dict:
        # reuse kernel catalog, or load registry file via snapshot context
        return kernel.catalog().model_dump()

    @mcp.resource("codeintel://exports/{token}")
    def read_export(token: str) -> ResourceContent:
        artifact = store.get(token)
        data = artifact.path.read_bytes()
        return ResourceContent(content=data, mime_type=artifact.mime_type)
```

This is straight out of the FastMCP Resources model (string/dict/bytes/BaseModel supported; `ResourceContent` gives explicit MIME + meta). ([FastMCP][2])

---

#### 3) `src/codeintel/serving/mcp/app.py`

* Instantiate a `ResourceStore`
* Call `register_resources(...)`
* Add a tool that creates export handles instead of returning huge row arrays:

  * `semantic_export(view_id, ...) -> { export_uri, … }`

**Representative tool snippet**

```py
@mcp.tool(annotations={"readOnlyHint": True})
async def semantic_export(view_id: str, filters: list[dict] | None = None) -> dict:
    # 1) generate rows (or a parquet file) - keep your existing kernel.export_rows
    rows = list(kernel.export_rows(SemanticExportRequest(view_id=view_id, filters=...)))

    # 2) store as JSON and return a resource URI handle
    token, _artifact = store.put_json({"rows": rows})
    return envelope(kernel, {"export_uri": f"codeintel://exports/{token}"}).model_dump()
```

---

### Tests

Add/adjust:

* `tests/serving/test_semantic_mcp_tools.py`

  * call `semantic_export`
  * assert it returns `export_uri`
  * then `client.read_resource(export_uri)` and assert contents

FastMCP client supports reading resources. ([FastMCP][7])

---

### Legacy code to delete / consolidate

* If you previously planned any “bespoke download endpoint wiring” for MCP payloads, you can drop it and standardize on resources.

---

## PR-S04 — H6: EventStore + SSE polling/resumability for StreamableHTTP (long-running tools)

### Goal

Enable “SSE polling for long-running operations” for StreamableHTTP by configuring an `EventStore` on the HTTP app. FastMCP explicitly documents that **SSE polling is enabled by providing an EventStore** ([FastMCP][1]).

This gives you:

* better reliability behind proxies / timeouts
* the ability for long tools to periodically close and have clients resume from stored events (progress notifications)

### Files changed

#### 1) `src/codeintel/serving/settings.py`

Add:

* `mcp_enable_event_store: bool = True`
* `mcp_retry_interval_ms: int = 1000` (optional)
* (optionally) `mcp_transport: Literal["http","streamable-http","sse"] = "http"` if you want explicit control

---

#### 2) `src/codeintel/serving/http/app.py`

When building the MCP ASGI app, pass the event store:

**Representative snippet**

```py
from fastmcp.server.event_store import EventStore

event_store = EventStore() if settings.mcp_enable_event_store else None

mcp_asgi = mcp_server.http_app(
    path="/",
    transport="http",  # streamable http default in FastMCP docs :contentReference[oaicite:21]{index=21}
    event_store=event_store,
    retry_interval=settings.mcp_retry_interval_ms if event_store else None,
)
```

FastMCP’s server API explicitly lists `event_store` and `retry_interval` on `http_app(...)` and explains their purpose (SSE polling/resumability). ([FastMCP][4])

---

#### 3) `src/codeintel/serving/mcp/app.py`

Where you have potentially long-running tools (export, big schema compile, etc.), accept `ctx` and optionally close the stream periodically:

FastMCP’s HTTP deployment guide shows `ctx.close_sse_stream()` as the mechanism to trigger reconnect/resume when EventStore is configured. ([FastMCP][1])

**Representative snippet**

```py
from fastmcp import Context

@mcp.tool
async def semantic_export_long(ctx: Context, view_id: str, ...) -> dict:
    for i in range(0, 100):
        await ctx.report_progress(i, 100)
        if i > 0 and i % 30 == 0:
            await ctx.close_sse_stream()  # triggers reconnect/resume w/ EventStore :contentReference[oaicite:24]{index=24}
        ...
```

---

### Tests

This is the only one that’s not “pure unit test friendly” because resumability is transport behavior.

Best pragmatic test approach:

* unit test: verify `create_serving_app` builds MCP ASGI app with an EventStore when enabled (structure/config test)
* integration test (optional): run `mcp_asgi` with TestClient and verify you can connect + call tool (if your harness supports it)

---

### Legacy code to delete

* Any homegrown “polling keepalive” logic in your MCP layer (if you were considering adding it) becomes unnecessary once EventStore is wired the canonical way.

---

# Final “file-by-file” quick map (what will change where)

### Modified

* `pyproject.toml`

  * pin/clean MCP deps (H0)
* `src/codeintel/serving/mcp/app.py`

  * swap MCP SDK FastMCP → gofastmcp FastMCP (H0)
  * tool envelope + tool annotations/meta (H2c)
  * typed inputs strategy (H2b)
  * export tool returns resource handle (H7)
  * add ctx hooks for SSE polling (H6)
* `src/codeintel/serving/mcp/server.py`

  * gofastmcp run/mount patterns (H0)
* `src/codeintel/serving/http/app.py`

  * mount `mcp.http_app(...)` + combine lifespan (H0)
  * pass `event_store`/`retry_interval` (H6)
* `src/codeintel/serving/settings.py`

  * add MCP serving toggles (H6/H7/H2c)

### Added

* `src/codeintel/serving/mcp/models.py` (H2c)
* `src/codeintel/serving/mcp/response.py` (H2c)
* `src/codeintel/serving/mcp/resource_store.py` (H7)
* `src/codeintel/serving/mcp/resources.py` (H7)

### Updated tests

* `tests/serving/test_semantic_mcp_tools.py`

  * switch to `fastmcp.client.Client(mcp)` testing style ([FastMCP][6])
  * add resource read assertions (H7) ([FastMCP][2])

---

# One extra “best-in-class” consolidation you should consider (optional, but very clean)

Once you’re on gofastmcp, you can split your MCP surface into subservers and mount them:

* `semantic_server` (semantic tools/resources)
* `search_server` (search tools/resources)
* `meta_server` (meta/version/schema inventory)

Then mount/import into a top-level `CodeIntel` server using FastMCP’s `mount`/`import_server` composition facilities. The SDK explicitly supports mounting/importing other servers. ([FastMCP][4])

This keeps `serving/mcp/app.py` small and makes the MCP surface easier to evolve as you add more semantic views.

---

If you want, I can now take **your current `src/codeintel/serving/mcp/app.py` tool list** and propose the **exact new tool signatures** (compat vs typed), plus the **exact resource URI taxonomy** (`codeintel://semantic/...`, `codeintel://exports/{token}`, etc.) so it’s consistent and discoverable for LLM agents.

[1]: https://gofastmcp.com/deployment/http "HTTP Deployment - FastMCP"
[2]: https://gofastmcp.com/servers/resources "Resources & Templates - FastMCP"
[3]: https://gofastmcp.com/servers/tools "Tools - FastMCP"
[4]: https://gofastmcp.com/python-sdk/fastmcp-server-server "server - FastMCP"
[5]: https://gofastmcp.com/integrations/fastapi "FastAPI  FastMCP - FastMCP"
[6]: https://gofastmcp.com/patterns/testing?utm_source=chatgpt.com "Testing your FastMCP Server"
[7]: https://gofastmcp.com/clients/client?utm_source=chatgpt.com "The FastMCP Client"
