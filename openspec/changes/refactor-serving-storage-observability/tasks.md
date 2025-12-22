## Status (2025-12-22)
- Implementation is largely complete for observability bootstrap, DuckDB tracing, SQL tooling,
  UUID factory migration, and HTTP NDJSON streaming.
- Remaining work is focused on aligning MCP export read metadata with FastMCP constraints,
  unifying MCP NDJSON encoding with the shared UTF-8/msgspec encoder, and surfacing
  sql_fingerprint in the shared SemanticQueryResponse for HTTP + MCP.

## 1. Implementation
- [x] 1.1 Add shared observability bootstrap and context utilities for CLI/HTTP/MCP.
- [x] 1.2 Refactor CLI telemetry to use the shared observability bootstrap (no parallel stack).
- [x] 1.3 Wire FastAPI serving to shared observability, OTel instrumentation, and /metrics.
- [x] 1.4 Wire MCP serving to shared observability, per-call spans, and /metrics.
- [x] 1.5 Enable DuckDB DB-API tracing by default with SQL statement redaction and env toggles.
- [x] 1.6 Centralize SQL canonicalization/fingerprinting in storage SQLGlot tools and add
      semantic SQL diff helper for upgrade diagnostics.
- [x] 1.7 Replace ad hoc UUID generation with the canonical UUID factory across serving,
      error reporting, CLI jobs/context, storage staging, and build executor IDs.
- [ ] 1.8 Update export payload handling: MCP reads return content with registry MIME and
      msgspec-backed UTF-8 NDJSON streaming with stdlib fallback.
      Status: HTTP NDJSON streaming is complete; MCP export read metadata + MCP NDJSON encoding
      still require follow-up (see Remaining Items).

## 2. Optional Improvements
- [x] 2.1 Add outbound HTTP client instrumentation (httpx/requests) when present; safe no-op otherwise.
- [x] 2.2 Add optional auth gating for /metrics using existing serving auth policy.
- [x] 2.3 Add export filename and caching hints in MCP export metadata (meta resource and
      resource listing _meta) where useful.

## 3. Tests
- [x] 3.1 Extend SQL compiler upgrade gate tests to include semantic diff output on failure.
- [x] 3.2 Add NDJSON encoding parity tests (unicode, datetime, UUID) for streaming exports.
- [x] 3.3 Add OTel span/metric smoke tests for CLI bootstrap, HTTP, and MCP middleware.
- [x] 3.4 Add DuckDB span redaction tests for default and overridden statement modes.

## 4. Docs
- [x] 4.1 Document new observability env toggles and default-on DuckDB tracing.
- [x] 4.2 Fix stale references in docs (build_http_router signature, register_prompts call shape).

## 5. Remaining Items (Follow-ups)
- [ ] 5.1 Add sql_fingerprint to SemanticQueryResponse (HTTP + MCP share the same model).
      Design: compute in ServingKernel.query from compiled SQL when available, omit on failure.
- [ ] 5.2 Align MCP export read metadata with FastMCP constraints.
      Design: read returns content + MIME only; export metadata is served via
      codeintel://exports/{export_id}/meta and MAY be surfaced in resource listing _meta.
      Optional: custom read handler or FastMCP patch if per-read metadata becomes required.
- [ ] 5.3 Unify MCP NDJSON encoding with shared encoder (msgspec fast path, UTF-8,
      compact separators, ensure_ascii=False, default=str).
      Add parity tests for ResourceStore NDJSON artifacts.
