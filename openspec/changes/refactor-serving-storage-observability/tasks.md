## 1. Implementation
- [ ] 1.1 Add shared observability bootstrap and context utilities for CLI/HTTP/MCP.
- [ ] 1.2 Refactor CLI telemetry to use the shared observability bootstrap (no parallel stack).
- [ ] 1.3 Wire FastAPI serving to shared observability, OTel instrumentation, and /metrics.
- [ ] 1.4 Wire MCP serving to shared observability, per-call spans, and /metrics.
- [ ] 1.5 Enable DuckDB DB-API tracing by default with SQL statement redaction and env toggles.
- [ ] 1.6 Centralize SQL canonicalization/fingerprinting in storage SQLGlot tools and add
      semantic SQL diff helper for upgrade diagnostics.
- [ ] 1.7 Replace ad hoc UUID generation with the canonical UUID factory across serving,
      error reporting, CLI jobs/context, storage staging, and build executor IDs.
- [ ] 1.8 Update export payload handling: ResourceContent for MCP exports and msgspec-backed
      UTF-8 NDJSON streaming with stdlib fallback.

## 2. Optional Improvements
- [ ] 2.1 Add outbound HTTP client instrumentation (httpx/requests) when present; safe no-op otherwise.
- [ ] 2.2 Add optional auth gating for /metrics using existing serving auth policy.
- [ ] 2.3 Add export filename and caching hints in MCP ResourceContent metadata where useful.

## 3. Tests
- [ ] 3.1 Extend SQL compiler upgrade gate tests to include semantic diff output on failure.
- [ ] 3.2 Add NDJSON encoding parity tests (unicode, datetime, UUID) for streaming exports.
- [ ] 3.3 Add OTel span/metric smoke tests for CLI bootstrap, HTTP, and MCP middleware.
- [ ] 3.4 Add DuckDB span redaction tests for default and overridden statement modes.

## 4. Docs
- [ ] 4.1 Document new observability env toggles and default-on DuckDB tracing.
- [ ] 4.2 Fix stale references in docs (build_http_router signature, register_prompts call shape).
