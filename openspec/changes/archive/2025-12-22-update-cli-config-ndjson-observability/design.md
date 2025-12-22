## Context
- CLI config precedence currently skips environment overrides and the Cyclopts config chain
  only includes TOML, which diverges from expected CLI behavior.
- NDJSON encoding uses msgspec datetime formatting that does not match str() output, and
  tests expect deterministic RFC3339 output across encoders.
- Pandera schema checks apply numeric constraints to JSON list columns, which fails
  validation for analytics.test_profile functions_covered payloads.
- Observability tests depend on env monkeypatching and async plugin gaps, while DuckDB
  tracing suppresses spans unless a parent span exists.

## Goals / Non-Goals
- Goals: deterministic CLI config precedence with top-level env allowlist, RFC3339 NDJSON
  timestamps, schema checks aligned with JSON columns, and observability behavior that is
  injectable for tests without monkeypatching.
- Goals: DuckDB tracing respects require_parent_span and statement_mode toggles; HTTP and
  MCP spans include correlation IDs; function_effects emits an info log for population.
- Non-Goals: support nested env overrides, change config file formats, or alter default
  redaction policies beyond documented toggles.

## Decisions
- Decision: Implement an explicit top-level env allowlist for CLI overrides and track env
  sources in ConfigService to preserve precedence and auditability.
- Decision: Keep runtime environment parsing centralized while allowing the CLI config
  allowlist as the only permitted env parsing outside the runtime loader.
- Decision: Standardize NDJSON datetime serialization on RFC3339 UTC with Z suffix and use
  a shared coercion layer for msgspec and stdlib json fallbacks.
- Decision: Apply non-negative constraints to *_count columns only and skip numeric checks
  for JSON list columns like functions_covered.
- Decision: Keep duckdb_require_parent_span default true in production but allow explicit
  overrides in injected ObservabilityConfig for tests.
- Decision: Use pytest-asyncio for asyncio tests and reserve anyio marks for anyio-native
  code paths (FastAPI/Starlette internals).

## Risks / Trade-offs
- RFC3339 changes may alter downstream NDJSON consumers expecting str() formatting; tests
  and docs must be updated together.
- Introducing env parsing in CLI config adds a second parsing surface; mitigate by
  restricting to an explicit allowlist and documenting precedence.
- Additional observability configuration paths increase setup complexity for tests.

## Migration Plan
1. Implement config/env override changes and update ConfigService source tracking.
2. Update NDJSON encoder and related tests to RFC3339 expectations.
3. Adjust Pandera constraint generation and profile validation tests.
4. Refactor observability tests to use injected config and async plugin support.
5. Run quality report and targeted pytest subsets, then full pytest.

## Open Questions
- None.
