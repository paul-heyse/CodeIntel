# Change: Update CLI config overlays, NDJSON serialization, and observability expectations

## Why
Full pytest runs show config precedence drift, NDJSON datetime formatting mismatches, and
observability test harness violations that block tracing validation. Aligning config
overrides, serialization, schema validation, and tracing behavior restores deterministic
runtime behavior and enables charter-compliant tests.

## What Changes
- Add explicit top-level CLI env override allowlist and source tracking so precedence is
  defaults < file < env < CLI flags, including CODEINTEL_COMMIT for project detection.
- Expand Cyclopts config chain to include env and optional TOML loaders so CLI parsing
  matches config precedence and reports two config chain entries.
- Standardize NDJSON serialization to RFC3339 UTC timestamps with consistent type coercion
  across msgspec and stdlib json encoders.
- Update Pandera constraint generation to avoid numeric checks on JSON list columns and
  enforce non-negative checks on corresponding count columns.
- Refine observability behavior: DuckDB tracing honors require_parent_span and
  statement_mode toggles, HTTP and MCP spans include correlation IDs, and function_effects
  emits info logs after row assembly.
- Refactor observability tests to use injected config and async plugin support without
  monkeypatch or unittest.mock.

## Impact
- Affected specs: config-injection, export-formats, schema-contracts, observability
  (in-flight from refactor-serving-storage-observability)
- Affected code: src/codeintel/cli/config/loader.py, src/codeintel/cli/config/service.py,
  src/codeintel/cli/project/_project.py, src/codeintel/serving/export/ndjson.py,
  src/codeintel/build/hamilton/contracts/schemas/pandera_schemas.py,
  src/codeintel/observability/duckdb_tracing.py, src/codeintel/observability/otel.py,
  src/codeintel/build/analytics/functions/function_effects.py, tests/cli/config/*,
  tests/serving/mcp/test_resources.py, tests/serving/test_streaming_ndjson.py,
  tests/analytics/test_profiles_and_functions.py, tests/analytics/functions/
  test_function_effects_runtime.py, tests/observability/*, pytest.ini
- Related changes: refactor-serving-storage-observability (observability + NDJSON) will
  need alignment during archive
