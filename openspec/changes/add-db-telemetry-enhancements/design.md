## Context
DuckDB tracing already emits redacted `db.statement` and a stable hash, but spans lack a
low-cardinality query summary and consistent attribute composition across callsites.
The db telemetry improvement proposals add summary generation, opt-in query text and
parameter emission, and tighter alignment with OTel database semantic conventions.

## Goals / Non-Goals
- Goals:
  - Provide `db.query.summary` for DuckDB spans and use it as the span name.
  - Centralize DB span attribute composition with safe defaults.
  - Keep raw SQL out of spans by default; allow opt-in sanitized query text/params.
  - Ensure statement hashes and summaries share a single canonicalization path.
  - Improve span fidelity with error status, exception recording, correlation IDs,
    and optional parent-span gating.
- Non-Goals:
  - Capturing raw SQL or parameter values by default.
  - Adding new metrics pipelines beyond existing OTel integration.
  - Supporting positional parameter emission (named only).

## Decisions
- Summary generation uses SQLGlot parsing with:
  - alias normalization (ignore aliases),
  - CTE-safe physical table extraction,
  - multi-operation summaries (e.g., INSERT ... SELECT ...),
  - token-safe truncation at 255 characters.
- Span naming uses `db.query.summary` when available; `db.operation.name` is not derived
  from SQL text.
- A shared DB span attribute builder owns `db.system.name`, `db.namespace`,
  `db.query.summary`, `codeintel.db.statement.sha256`, optional legacy keys, and the
  optional query text/parameter attributes.
- Query text emission is opt-in with policies:
  - parameterized-only,
  - SQLGlot-redacted,
  - parameterized-or-redacted,
  - debug-only raw.
- Named parameter emission is opt-in with strict allowlist, scalar-only coercion,
  string length caps, optional hashing, and batch execution suppression.
- Parent-span gating defaults on to avoid noisy root spans; can be disabled via config.

## Risks / Trade-offs
- SQLGlot parsing adds overhead; mitigated by caching/fast paths and fallback
  normalization when parsing fails.
- Summaries may omit some target tables to preserve low cardinality; explicit truncation
  preserves stability while keeping dashboards readable.
- Additional configuration knobs increase surface area; defaults remain safe and minimal.

## Migration Plan
Immediate transition: enable summary generation and new attribute builder without a
multi-phase deprecation. Legacy attribute emission remains optional for integration
compatibility but is disabled by default.

## Open Questions
- Final names for summary/text/parameter environment variables.
- Whether to cap the number of tables included in summaries beyond 255-char truncation.
