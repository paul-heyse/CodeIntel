## 1. Implementation
- [ ] 1.1 Add a CLI env override allowlist and precedence handling in the config loader
      (defaults < file < env < CLI flags) with source tracking.
- [ ] 1.2 Add a Cyclopts config chain env loader alongside the optional TOML loader and
      keep chain order consistent with precedence.
- [ ] 1.3 Honor CODEINTEL_COMMIT as an explicit override in project commit detection.
- [ ] 1.4 Add a shared NDJSON value coercion layer that emits RFC3339 UTC timestamps and
      consistent stringification for non-JSON types across encoders.
- [ ] 1.5 Update Pandera constraint generation to skip numeric checks on JSON columns and
      apply non-negative checks to count columns.
- [ ] 1.6 Refine observability configuration to honor duckdb require_parent_span and
      statement_mode toggles, and ensure span attributes align with expectations.
- [ ] 1.7 Emit a function_effects population INFO log with row counts and snapshot context.
- [ ] 1.8 Add asyncio test configuration (pytest-asyncio) for asyncio tests and keep anyio
      usage for anyio-native code paths.

## 2. Tests
- [ ] 2.1 Update config integration/unit tests for env overrides and config chain length.
- [ ] 2.2 Update NDJSON tests to expect RFC3339 UTC timestamps.
- [ ] 2.3 Update profile schema tests for JSON list columns and count checks.
- [ ] 2.4 Refactor observability tests to use injected config and remove monkeypatch and
      unittest.mock usage.
- [ ] 2.5 Verify function_effects logging assertions.

## 3. Validation
- [ ] 3.1 Run `uv run python -m tools.quality_report --output build/quality-results/
      quality_report.json`.
- [ ] 3.2 Run `uv run pytest -q` (or targeted subsets for updated areas if needed).
