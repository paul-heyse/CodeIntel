# Change: Add DuckDB telemetry enhancements

## Why
DuckDB spans are already redacted, but dashboards still lack a low-cardinality query
summary and consistent attribute composition across storage/serving/CLI. We also need
explicit, opt-in controls for query text and parameter emission, plus consistent hashing
to prevent fingerprint drift.

## What Changes
- Add SQLGlot-based `db.query.summary` generation and use it as the DuckDB span name,
  including alias normalization, CTE-safe table extraction, multi-operation summaries,
  and token-safe 255-character truncation.
- Introduce a shared DB span attribute builder to centralize `db.system.name`,
  `db.namespace`, `db.query.summary`, and `codeintel.db.statement.sha256`, with optional
  legacy `db.system`/`db.name` emission and no SQL-derived `db.operation.name`.
- Add opt-in `db.query.text` emission (parameterized or SQLGlot-redacted) with strict
  defaults that keep raw SQL out of spans.
- Add opt-in allowlisted `db.query.parameter.<key>` emission for named parameters only,
  skipping batch executions and enforcing scalar + length limits.
- Consolidate SQL canonicalization and hashing so statement digests and summaries share
  a single SQLGlot path with safe regex/hash fallback on parse failure.
- Improve trace fidelity by recording exceptions/status, attaching correlation IDs, and
  supporting parent-span gating for DuckDB span emission.
- Extend observability settings and docs with the new configuration knobs and update
  test coverage for summaries, text/parameter policies, and span naming.

## Impact
- Affected specs: observability
- Affected code: src/codeintel/observability/duckdb_tracing.py,
  src/codeintel/observability/sql_redaction.py,
  src/codeintel/observability/context.py,
  src/codeintel/observability/otel.py,
  src/codeintel/storage/sqlglot_tools.py,
  src/codeintel/serving/semantic/fingerprints.py,
  src/codeintel/core/config/settings.py,
  src/codeintel/core/runtime/loader.py,
  docs/observability.md,
  tests/observability/test_duckdb_tracing.py,
  tests/storage/test_sqlglot_tools.py
