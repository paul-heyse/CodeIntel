# Incident: `tools.quality_report` corrupts Python sources by stripping `#...`

## Summary

Running `uv run python -m tools.quality_report` modifies `*.py` files in-place. Its current
"comment stripping" logic removes everything after the first `#` on a line, even when that `#`
appears inside a string literal. This produces widespread syntax corruption (unterminated strings,
unterminated f-strings, and invalid Python).

This is not related to the SQL → Ibis migration; it's a tooling bug that impacts any file containing
`#` inside strings (URNs with fragments, markdown headings, CSS hex colors, etc.).

## Status

Mitigation applied: `tools/quality_report.py` no longer performs in-place source mutation. Note that
the report still runs `ruff check --fix`, so edits from Ruff autofix are expected (but should be
semantics-preserving).

## Root Cause

In the affected revision, `tools/quality_report.py` called `_strip__comments(repo_root)` from
`main()`. `_strip__comments` walks `repo_root.rglob("*.py")` and performs a naive
`raw_line.find("#")` slice to drop everything after `#`, then writes the result back to disk.

Because it is not token-aware, it treats `#` inside string literals the same as actual Python
comments.

## Evidence (Examples)

- `tests/test_tests_analytics_unit.py` contains URNs like
  `goid:demo/repo#python:function:test_mod.test_func`; after running the tool they become
  `goid:demo/repo` (unterminated string literal).
- `src/codeintel/analytics/graphs/plugin_catalog.py` contains markdown strings like
  `"# Analytics Plugin Catalog"` and CSS colors like `#333`; both are truncated, producing invalid
  Python.

## Impact / Symptoms

- Ruff / Pyright / Pyrefly emit parse errors ("invalid syntax") across many files.
- `python -m compileall -q src tests` reports syntax errors in 50 `src/` + `tests/` files.
- Additional syntax errors appear in `mkdocs_gen/` and `tools/quality_report.py` itself.

## Affected Files (From `compileall`)

### `src/` + `tests/` (50)

```text
src/codeintel/analytics/compute/functions/loc.py
src/codeintel/analytics/graphs/plugin_catalog.py
src/codeintel/cli/completions/bash_generator.py
src/codeintel/cli/completions/fish_generator.py
src/codeintel/cli/completions/powershell_generator.py
src/codeintel/cli/completions/zsh_generator.py
src/codeintel/cli/project/_project.py
src/codeintel/cli/rendering/service.py
src/codeintel/cli/shell/_shell.py
src/codeintel/config/datasets/introspection.py
src/codeintel/graphs/compute/goid.py
src/codeintel/storage/datasets/catalog.py
src/codeintel/storage/datasets/scaffold.py
tests/analytics/adapters/test_functions_adapter.py
tests/analytics/test_analytics_contracts.py
tests/analytics/test_functions_validation.py
tests/analytics/testing/coverage/test_edges.py
tests/analytics/test_model_config_heuristics.py
tests/analytics/test_profiles_and_functions.py
tests/architecture/test_analytics_imports.py
tests/architecture/test_ibis_only_queries.py
tests/config/test_datasets_introspection.py
tests/docs_export/conftest.py
tests/docs_export/test_mkdocs_generation.py
tests/graphs/test_adapters_libcst_extended.py
tests/_helpers/coverage.py
tests/_helpers/fakes/function_catalogs.py
tests/_helpers/ingestion.py
tests/_helpers/orchestration/coverage_orchestration.py
tests/_helpers/orchestration/entrypoints_orchestration.py
tests/_helpers/orchestration/history.py
tests/_helpers/orchestration/seeding_docs.py
tests/_helpers/rows.py
tests/_helpers/seeds/architecture.py
tests/_helpers/seeds/core.py
tests/_helpers/seeds/coverage.py
tests/_helpers/seeds/data_models.py
tests/_helpers/seeds/entrypoints.py
tests/_helpers/seeds/function_types.py
tests/_helpers/seeds/metrics.py
tests/_helpers/seeds/profile.py
tests/_helpers/seeds/subsystems_analytics.py
tests/ingestion/test_tools.py
tests/server/test_fastapi_endpoints.py
tests/serving/backend/test_function_backend.py
tests/storage/repositories/test_functions.py
tests/storage/test_dataset_catalog.py
tests/storage/test_dataset_scaffold.py
tests/storage/test_gateway_accessors.py
tests/test_tests_analytics_unit.py
```

### Other (3)

```text
mkdocs_gen/build_single_markdown.py
mkdocs_gen/gen_ref_pages.py
tools/quality_report.py
```

## How to Reproduce (Unsafe On Any Non-Disposable Working Tree)

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
```

## Recovery (After Corruption)

1. Stop running `tools.quality_report` until the stripping behavior is removed or fixed.
2. Restore corrupted files from git (choose a scope that preserves any local work you need):
   - Broad restore (fastest, but will discard local edits under these paths):
     - `git restore --source=HEAD -- src tests mkdocs_gen tools/quality_report.py`
   - Targeted restore:
     - Use `python -m compileall -q src tests` and `git diff` to identify and restore only damaged
       files.
3. Verify syntax is healthy again:
   - `uv run python -m compileall -q src tests`

## Prevention / Proposed Fix

`tools.quality_report` should never mutate repository sources. If we want a "no suppressions" policy,
the tool should detect and fail (or emit a report) rather than editing files. If automatic removal is
still desired, it must be token-aware (e.g., use `tokenize` to remove only actual comment tokens).
