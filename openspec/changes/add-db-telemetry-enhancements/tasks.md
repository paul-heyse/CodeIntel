## 1. Implementation
- [ ] 1.1 Add SQLGlot-based query summary generator with alias normalization,
      CTE-safe table extraction, multi-operation handling, and token-safe truncation.
- [ ] 1.2 Add a shared DB span attribute builder and integrate it into DuckDB tracing
      (span naming, canonical attributes, optional legacy keys).
- [ ] 1.3 Implement opt-in `db.query.text` emission policies (parameterized, redacted,
      parameterized-or-redacted, debug-only raw) with length caps and literal redaction.
- [ ] 1.4 Implement opt-in allowlisted `db.query.parameter.<key>` emission for named
      params only, skipping batches and enforcing scalar/length/hashing rules.
- [ ] 1.5 Consolidate SQL canonicalization/hashing so summary and hash use a shared
      SQLGlot path with safe fallback on parse failure.
- [ ] 1.6 Wire new settings into runtime configuration and document them in
      `docs/observability.md`.
- [ ] 1.7 Update DuckDB tracing to record exceptions/status, attach correlation IDs,
      and honor parent-span gating.
- [ ] 1.8 Add/expand tests for summary generation, span naming, query text policy,
      parameter allowlists, and parse-failure fallbacks.
- [ ] 1.9 Update observability guidance to group dashboards by `db.query.summary` and
      `codeintel.db.statement.sha256`.
