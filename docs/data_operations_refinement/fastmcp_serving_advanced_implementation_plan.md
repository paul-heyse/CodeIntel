# FastMCP Serving Advanced Capabilities Implementation Plan (uvicorn_workers=1)

> **Purpose**: Implement a best-in-class, production-grade FastMCP (v2.14+) serving surface for CodeIntel,
> emphasizing streamlining, extensibility, hardening, and maintainability while explicitly targeting a
> **single-process** deployment (`uvicorn_workers=1`) to keep **sessionful** MCP features reliable.

**Generated**: 2025-12-18  
**Status**: Implementation Plan (ready to execute)  
**Primary scope**: `src/codeintel/serving` (MCP + HTTP mount integration + settings + tests)  
**Out of scope** (unless required by integration): build/ingestion pipelines, storage schema design, semantic registry design

---

## Executive Summary

We will upgrade the MCP serving layer to fully leverage FastMCP v2.14+ advanced capabilities:

1. **MCP middleware backbone** for consistent logging, timing, error shaping, and rate limiting (removes per-tool boilerplate).
2. **Structured, stable error contracts** at the protocol level (`McpError(ErrorData(..., data=...))`) using
   `codeintel.serving.mcp.errors` as the canonical source of truth.
3. **Background tasks (SEP-1686)** for long-running operations (export + potentially heavy queries), with progress and
   cancellation best-effort semantics.
4. **Sampling (ctx.sample)** for optional LLM summaries of large results, gated by settings and client capability.
5. **Elicitation (ctx.elicit)** to support interactive “wizard” flows (implemented via prompts to avoid changing tool semantics).
6. **Prompts upgraded** to `PromptResult` + multi-message + tags/meta, enabling rich guided workflows.
7. **Export resource hardening**: chunked reads for large artifacts + TTL + cleanup, avoiding memory blowups in MCP clients.
8. **Operational guardrails**: explicitly enforce the `uvicorn_workers=1` + sessionful model; document and validate it.

The result is a server that is simpler to evolve (middleware + DI), safer (rate limiting + structured errors),
more ergonomic for agents (prompts + resources + optional sampling), and more reliable for large operations (tasks).

---

## Decisions & Guardrails (locked-in for this plan)

### Deployment model

- **`uvicorn_workers=1`** when MCP is mounted under FastAPI.
- **Sessionful Streamable HTTP** (FastMCP default; `stateless_http=False`) because we want:
  - user elicitation (`ctx.elicit`)
  - sampling (`ctx.sample`) where applicable
  - consistent session/request scoping
- Keep **stdio transport working** (for local agents/inspector), but the “production-grade” target is HTTP mount.

### Security posture

- Keep the existing fail-fast rule: public binding requires auth (`ServingSettings.validate_auth_for_host()`).
- Prefer **FastMCP auth providers** (bearer token) over ad-hoc header middleware so stdio and HTTP behave consistently.
- Elicitation must not request secrets (explicitly prohibited by the MCP spec).

### Compatibility posture

- Do not break existing tool names or required arguments.
- Additive changes (new optional fields, new prompts/resources) are allowed.
- Maintain contract checks in `tools.quality_report` (notably `src/codeintel/serving/contracts/check_operation_contracts.py`).

---

## Definition of Done (target-state checklist)

### Reliability & operations

- [ ] MCP server mounts under `/mcp` with `uvicorn_workers=1` explicitly validated and documented.
- [ ] Middleware provides consistent logging + timing + rate limiting + error conversion (no per-tool custom scaffolding required).
- [ ] Long-running exports support background tasks and progress; cancellation cleans up partial artifacts best-effort.
- [ ] EventStore usage is consistent with the chosen model (either used with `close_sse_stream()` where needed or documented as optional).

### Developer experience & maintainability

- [ ] Tool/resource/prompt registration is modular (clear separation between builder, tools, resources, prompts, middleware).
- [ ] FastMCP DI (`CurrentContext()` + `Depends(...)`) is used where it clearly improves clarity and testability.
- [ ] Shared helpers exist for: “maybe progress”, “build error context”, “raise MCP error with structured data”.

### Agent experience (LLM clients)

- [ ] Prompts are tagged, meta-rich, and return multi-message templates (`PromptResult` / `Message`).
- [ ] Export flows return small handles and provide chunked resources so clients can safely retrieve large payloads.
- [ ] Optional sampling provides summaries above configurable thresholds (and is safe when unsupported).

### Quality gates

- [ ] `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- [ ] `uv run pytest -q`
- [ ] `uv run python -m codeintel.serving.contracts.check_operation_contracts` (or via quality report)
- [ ] `uv run fastmcp inspect ...` (manifest sanity; see “Validation Commands”)

---

## Minimal PR Sequence (recommended ordering)

This is written as a PR checklist. Each PR section contains:

- **Files to change**
- **Implementation notes**
- **Acceptance criteria**
- **Tests to add**

### PR0 — Enforce the single-worker + sessionful MCP contract
### PR1 — Add MCP middleware backbone (logging/timing/rate limiting baseline)
### PR2 — Canonical structured MCP errors (wire `codeintel.serving.mcp.errors`)
### PR3 — Background tasks for exports (+ progress + cancellation cleanup)
### PR4 — Sampling support for large outputs (opt-in; safe fallback)
### PR5 — Prompts upgrade: tags/meta/multi-message + elicitation-powered “wizards”
### PR6 — Export resources: chunked reads + TTL + cleanup policy
### PR7 — Integration + regression tests (in-memory FastMCP client) + operational docs

---

# PR0 — Enforce the single-worker + sessionful MCP contract

## Why

FastMCP’s sessionful Streamable HTTP features (elicitation/sampling) are not robust under multi-worker deployments.
We explicitly choose `uvicorn_workers=1`, so we should:

- fail fast if misconfigured
- document the constraint and the “what if we later want multi-worker” path

## Files

- `src/codeintel/serving/settings.py`
- `src/codeintel/serving/http/app.py`
- `src/codeintel/serving/mcp/server.py` (if it has an HTTP runner path)
- `docs/` (add a short runtime runbook note; can be in this doc or a dedicated serving doc)

## Implementation notes

1. Add a dedicated validator (or extend `validate_auth_for_host`) that enforces:
   - if MCP is mounted under FastAPI and `uvicorn_workers != 1`, raise `ValueError` with a clear message
   - if `stateless_http=True`, explicitly document which features are disabled (elicitation/sampling) and why

2. Make the configuration contract explicit in docs:
   - “MCP is sessionful; do not run with `--workers > 1`”
   - “If you need multi-worker later: enable stateless HTTP and disable elicitation/sampling; move tasks backend to Redis”

## Acceptance criteria

- Misconfiguration (`uvicorn_workers > 1`) fails before serving starts, with a clear error.
- Documentation includes a crisp “supported deployment modes” matrix.

## Tests to add

- `tests/serving/test_settings_mcp_worker_guardrails.py`
  - validates that `ServingSettings.validate_*` raises when `uvicorn_workers > 1` and MCP is enabled/mounted.

---

# PR1 — Add MCP middleware backbone (logging/timing/rate limiting baseline)

## Why

Right now, each tool manually implements:

- timing measurement
- log formatting
- error wrapping
- (and sometimes policy checks)

FastMCP middleware provides a standardized pipeline for this. Centralizing cross-cutting behavior in middleware:

- reduces repeated code and drift risk
- ensures uniform behavior across tools/resources/prompts
- makes future features (caching, request context enrichment) straightforward

## Files

- `src/codeintel/serving/mcp/app.py`
- `src/codeintel/serving/mcp/_compat.py` (only if we add more feature detection)
- `src/codeintel/serving/settings.py` (add MCP middleware knobs)
- `src/codeintel/serving/mcp/` (new module for middleware assembly)

## Implementation notes

### 1) Choose a baseline middleware stack and ordering

Per FastMCP ordering semantics: “first added runs first inbound and last outbound”.

Recommended default ordering:

1. `StructuredLoggingMiddleware` (outermost; sees everything)
2. `DetailedTimingMiddleware`
3. `RateLimitingMiddleware`
4. (optional in PR1) `ResponseCachingMiddleware` — **list-only** caching (safe); avoid tool/resource caching until snapshot-aware keys exist

Notes:

- Keep middleware transport-agnostic; do not depend on HTTP headers in core middleware so stdio remains supported.
- Rate limiting can be global initially; if needed, later extract a stable client ID from context/session meta.
- If adding caching: configure it to cache **only** `tools/list`, `resources/list`, `prompts/list` in PR1.
  (FastMCP’s `ResponseCachingMiddleware` cache keys do not include snapshot identity.)

### 2) Implement middleware assembly as a small pure function

Create a module like:

- `src/codeintel/serving/mcp/middleware_stack.py`
  - `def build_mcp_middleware(settings: ServingSettings) -> list[Middleware]: ...`

Then pass `middleware=...` to `FastMCP(...)` in `build_mcp_app`.

### 3) Add settings knobs (minimal)

Add (example) fields to `ServingSettings`:

- `mcp_enable_structured_logging: bool = True`
- `mcp_rate_limit_rps: float = 20.0`
- `mcp_rate_limit_burst: int = 40`
- `mcp_cache_listings: bool = True`

All should default to safe values and be controllable via env vars.

## Acceptance criteria

- Tool/resource/prompt calls produce consistent structured logs and timing logs.
- Rate limiting rejects excessive request volume with a proper MCP error (without crashing the server).
- `check_operation_contracts` still passes.

## Tests to add

- `tests/serving/test_mcp_middleware_rate_limit.py`
  - create an in-memory server + client; make N rapid requests; assert rate limit error appears.
- `tests/serving/test_mcp_middleware_logging_smoke.py`
  - smoke test that middleware does not break tool execution and does not require HTTP transport.

---

# PR2 — Canonical structured MCP errors (wire `codeintel.serving.mcp.errors`)

## Why

You already have a strong canonical error model:

- `src/codeintel/serving/mcp/errors.py`

But it is not used by the actual tool/resource behavior. Today tools raise `ToolError` with generic messages, which:

- loses stable machine codes and hints
- makes agent behavior less reliable
- duplicates error mapping logic across tools

The target is: **protocol-level errors that include structured data**, so clients can do robust handling:

- raise `mcp.McpError(mcp.types.ErrorData(code=<int>, message=<safe>, data=<ErrorResponse dict>))`

## Files

- `src/codeintel/serving/mcp/errors.py` (already exists; we will use it)
- `src/codeintel/serving/mcp/app.py`
- `src/codeintel/serving/mcp/resources.py`
- `src/codeintel/serving/mcp/resource_store.py` (if adding TTL/expires-at fields earlier)
- `src/codeintel/serving/mcp/` (new: error-mapping middleware)

## Implementation notes

### 1) Add a CodeIntel error-mapping middleware (single place)

Create `src/codeintel/serving/mcp/middleware_errors.py`:

- catches exceptions in `on_message` (or `on_request`)
- builds `ErrorContext` (operation/tool/resource identifiers + snapshot info if available)
- calls `exception_to_error_response(exc, context=...)`
- converts `ErrorResponse` → `McpError(ErrorData(..., data=...))`

Mapping `ErrorKind` → JSON-RPC error codes:

- `invalid_request` → `-32602` (Invalid params)
- `not_found` / `expired` / `corrupt` → `-32001` (Resource not found)
- `conflict` → `-32000` (Server error / conflict)
- `unavailable` / `timeout` → `-32000` (Server error; message clarifies retry)
- `internal` → `-32603` (Internal error)

### 2) Remove per-tool error wrapping where possible

Refactor tools/resources to raise domain errors (or allow natural exceptions) and rely on middleware for shaping.

Examples:

- When view is missing: raise `SemanticViewNotFoundError(view_id)`
- When export is missing: raise `ExportNotFoundError(export_id)`

### 3) Preserve security guarantees

- `ErrorResponse.error.details` must remain “safe” (no stack traces, no file paths).
- Use `ServingSettings.mcp_mask_errors` to decide whether internal errors should expose any details beyond the stable code/hint.

## Acceptance criteria

- For known failure modes, clients receive an MCP error with:
  - stable error code/hint in `ErrorData.data`
  - appropriate `ErrorData.code` (int) for generic classification
  - safe message
- Tool bodies shrink materially (less repeated try/except scaffolding).
- `check_operation_contracts` still passes.

## Tests to add

- `tests/serving/test_mcp_errors_structured.py`
  - call `semantic_describe(view_id="does_not_exist")`
  - assert MCP error `data` contains `{"status":"error","error":{"code":"CODEINTEL_SEMANTIC_VIEW_NOT_FOUND",...}}`

---

# PR3 — Background tasks for exports (+ progress + cancellation cleanup)

## Why

Exports can be slow and large. Background tasks provide:

- immediate response with a task handle
- progress reporting and cancellation at the protocol level
- resilience against long-held HTTP requests/timeouts

With `uvicorn_workers=1`, the in-memory task backend is acceptable for the initial implementation.

## Files

- `src/codeintel/serving/mcp/app.py` (mark heavy tools `task=...`, add progress and cancellation semantics)
- `src/codeintel/serving/mcp/resource_store.py` (support partial-write cleanup + TTL metadata)
- `src/codeintel/serving/settings.py` (task config knobs)
- `src/codeintel/serving/http/app.py` (document interaction with EventStore + keep-alives)

## Implementation notes

### 1) Select which operations become task-capable

Initial recommendation:

- `semantic_export`: `task=True` (optional) so clients that support tasks can avoid long requests.

Keep `semantic_query` synchronous initially (unless you’ve observed it running long enough to justify tasks).

### 2) Add progress reporting that respects settings

Your settings already include:

- `mcp_progress_reporting`

Implement a helper (`maybe_report_progress`) and ensure tools only emit progress when enabled.

### 3) Cancellation semantics

Cancellation can be best-effort in the first pass:

- detect cancellation via `anyio.get_cancelled_exc_class()` / `CancelledError`
- stop iteration/writing
- delete partial artifact files and sidecars
- return a stable cancellation error code in the task result (or surface as task-cancelled)

### 4) Concurrency isolation (export vs interactive)

Today `QueryLimiter` is shared. Consider splitting:

- `mcp_max_concurrent_queries`
- `mcp_max_concurrent_exports`

Exports already use `connect_export()` for pool isolation; add limiter isolation to match.

## Acceptance criteria

- Clients can call `semantic_export(..., task=True)` and receive a task handle.
- Progress updates are emitted during export when enabled.
- Cancelling an export removes partial artifacts best-effort.
- Interactive queries are not starved by a single long export (pool + limiter isolation).

## Tests to add

- `tests/serving/test_mcp_export_task_mode.py`
  - call export with `task=True`, wait for completion, verify result contains export handle.
- `tests/serving/test_mcp_export_cancellation_cleanup.py`
  - start export task, cancel, verify export artifact not present.

---

# PR4 — Sampling support for large outputs (opt-in; safe fallback)

## Why

For large query results, agents frequently need a compact “what is this?” summary without downloading everything.
FastMCP’s sampling enables the server to ask the client’s LLM handler to summarize structured data.

This must be:

- explicitly opt-in (`mcp_enable_sampling`)
- safe when client has no sampling capability
- bounded in token/size to avoid runaway costs

## Files

- `src/codeintel/serving/mcp/app.py` (wire into `semantic_query` and/or `serving_meta`)
- `src/codeintel/serving/settings.py` (already has sampling knobs)
- `src/codeintel/serving/mcp/response_models.py` (optional: add a `summary` field)

## Implementation notes

### 1) Sampling policy

Inputs to sampling should be:

- a small preview (already exists: `QueryPreview` in `semantic_query`)
- schema/context (view_id, columns, types, filters)
- server guidance (“Summarize in 5 bullets; call semantic_export if needed”)

### 2) When to sample

Trigger only when both:

- `settings.mcp_enable_sampling` is true
- `row_count >= settings.mcp_sample_threshold` or `result.truncated is True`

### 3) Capability detection / fallback

If `ctx.sample(...)` raises due to client not supporting sampling:

- do not fail the tool call
- add a note indicating sampling is unavailable

## Acceptance criteria

- When enabled, large query responses include an LLM-generated summary (bounded and stable).
- When unsupported, tool still returns normally with a clear note.

## Tests to add

- `tests/serving/test_mcp_sampling_opt_in.py`
  - use in-memory client with a fake sampling handler
  - verify summary is included above the threshold and absent below it.

---

# PR5 — Prompts upgrade: tags/meta/multi-message + elicitation-powered “wizards”

## Why

Prompts are the best place for:

- user-controlled workflows
- reusable parameterized templates
- interactive elicitation (“wizard”) flows

You already have basic prompts (`src/codeintel/serving/mcp/prompts.py`), but they are:

- untagged
- single-string
- not meta/versioned
- not using `PromptResult`

## Files

- `src/codeintel/serving/mcp/prompts.py`
- `src/codeintel/serving/mcp/app.py` (prompt duplicate behavior + optional prompt-tool injection)
- `src/codeintel/serving/settings.py` (prompt feature flags)

## Implementation notes

### 1) Upgrade to PromptResult + multi-message templates

Use:

- `PromptResult(messages=[Message(...), ...], meta={...})`

Add tags such as:

- `{"onboarding", "semantic"}`
- `{"search"}`
- `{"export"}`
- `{"ops"}`

Set `on_duplicate_prompts="error"` in `FastMCP(...)` so duplicates fail fast during startup.

### 2) Elicitation-powered prompts (“wizards”)

Add prompts that (when client supports elicitation) gather a few fields and return a ready-to-run plan:

- `wizard_export_data`:
  - elicit `view_id`
  - elicit `format` choice (`ndjson|json`)
  - elicit `limit` / confirm
  - returns messages that instruct to call `semantic_export(..., task=True)`

- `wizard_query_view`:
  - elicit `view_id`
  - elicit optional `select` and a small set of filters (shallow schema)
  - returns messages to call `semantic_query(...)`

If elicitation unsupported: return a prompt that explains how to call the non-wizard tools manually.

### 3) Optional: PromptToolMiddleware for tool-only clients

Gate behind a setting (default off) because it changes the visible tool list.

## Acceptance criteria

- `list_prompts()` shows prompts with tags/meta (and clear descriptions).
- `get_prompt()` returns multi-message templates suitable for direct LLM consumption.
- Wizard prompts use elicitation when available and degrade gracefully when not.

## Tests to add

- `tests/serving/test_mcp_prompts_metadata.py`
- `tests/serving/test_mcp_prompt_elicitation_wizard.py` (in-memory client with elicitation handler)

---

# PR6 — Export resources: chunked reads + TTL + cleanup policy

## Why

MCP resources are not streaming. Returning a full export payload as one string can:

- blow up client memory
- exceed payload limits
- make “big payloads via resources” impractical at scale

We need:

- chunked export reads (resource templates for slices)
- TTL + cleanup policy so the export store does not grow unbounded

## Files

- `src/codeintel/serving/mcp/resources.py`
- `src/codeintel/serving/mcp/resource_store.py`
- `src/codeintel/serving/settings.py`

## Implementation notes

### 1) Chunked resource templates

Add resource templates like:

- `codeintel://exports/{export_id}/lines?offset={offset}&limit={limit}`
  - returns NDJSON lines subset
  - supports agent pagination

Or path-based:

- `codeintel://exports/{export_id}/chunks/{chunk_index}`

### 2) TTL support

Add settings:

- `mcp_export_ttl_seconds: int | None`
- `mcp_export_cleanup_interval_seconds: int`

Write `expires_at` into metadata sidecar and implement cleanup:

- on startup (lifespan)
- periodically (async task)

### 3) “Large payload” strategy

Ensure the export handle response emphasizes:

- prefer `preview` and `meta` resources first
- fetch payload in chunks if needed

## Acceptance criteria

- Export payloads can be retrieved safely in small chunks.
- Exports are automatically cleaned up when TTL is configured.
- No tool/resource call loads multi-GB payloads into memory.

## Tests to add

- `tests/serving/test_resource_store_ttl_cleanup.py`
- `tests/serving/test_mcp_export_chunked_resource.py`

---

# PR7 — Integration + regression tests + operational docs

## Why

These changes span middleware, protocol errors, tasks, prompts, and resources. We need a tight test loop and an operator-friendly runbook.

## Files

- `tests/serving/` (new test modules)
- `docs/` (short runbook; can live in `docs/` or appended to this plan)
- `src/codeintel/serving/contracts/check_operation_contracts.py` (only if we intentionally change tool list/schema requirements)

## Implementation notes

### 1) Prefer in-memory FastMCP client tests

FastMCP supports `Client(mcp_server_instance)` (in-memory), which is ideal for deterministic protocol tests:

- no sockets
- no uvicorn
- no env leakage

Use this for:

- error shaping
- tasks
- prompts/elicitation
- resource reads

### 2) Validate manifests with `fastmcp inspect`

Add a documented command to generate a manifest snapshot (useful for debugging client-visible changes):

- `uv run fastmcp inspect src/codeintel/serving/mcp/server.py:create_mcp_server -o build/mcp-manifest.json`

### 3) Update/extend operator docs

Document:

- configuration knobs
- worker requirements
- how to run locally (stdio and HTTP)
- how to debug with `fastmcp dev` / Inspector (stdio)

## Acceptance criteria

- CI-quality gates pass locally (quality report + pytest).
- MCP manifest diff is predictable and intentionally changed.
- Clear “how to run” and “supported deployment modes” docs exist.

---

## Validation Commands (copy/paste)

Environment setup:

```bash
scripts/bootstrap.sh
uv sync
```

Quality gates:

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

Serving contract check:

```bash
uv run python -m codeintel.serving.contracts.check_operation_contracts
```

MCP manifest inspection:

```bash
uv run fastmcp inspect src/codeintel/serving/mcp/server.py:create_mcp_server -o build/mcp-manifest.json
```

---

## Appendix: “If we ever want multi-worker”

If requirements change and we need `uvicorn_workers > 1`:

1. Enable `stateless_http=True` for MCP.
2. Disable or redesign sessionful features:
   - elicitation (likely disabled)
   - sampling (likely disabled, or redesigned so client carries state)
3. Move task backend off in-memory:
   - use Redis/Valkey backend for tasks and event store
4. Make caching snapshot-aware and shared (Redis) or disable it.

This plan intentionally does **not** implement the multi-worker path.

