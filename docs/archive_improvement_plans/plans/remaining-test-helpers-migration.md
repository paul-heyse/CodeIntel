# Remaining Test Helpers Migration Plan

## Overview

This document outlines the remaining work needed to complete the wider adoption of `_helpers` infrastructure across the test suite. The goal is to reduce code duplication, ensure consistent test conditions, and align with the Testing Charter (no mocking allowed).

## Critical Requirements

All changes MUST achieve:
- Zero `pyright` errors
- Zero `pyrefly` errors  
- Zero `ruff` errors
- NO `# type: ignore` or `# noqa` suppressions

---

## Completed Work (Summary)

The following phases have been completed:

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Create `TestExecutionContextBuilder` with real production types | ✅ Complete |
| 2 | Consolidate local `TEST_REPO`/`TEST_COMMIT` constants | ✅ Complete |
| 3 | Replace local `_create_config()` with `make_step_config()` | ✅ Complete |
| 5 | Update analytics adapters conftest to use shared fixtures | ✅ Complete |
| 6 | Consolidate RunContext creation in core plugins | ✅ Complete |

### New Infrastructure Created

- `tests/_helpers/fakes/execution_contexts.py` - `TestExecutionContextBuilder` and `create_test_execution_context()`
- `tests/_helpers/factories/step_config_factories.py` - `make_step_config()` and `make_snapshot()`

---

## Remaining Work

### Phase A: Migrate Mock Context Builders to Real Contexts (HIGH PRIORITY)

**Problem**: 14 plugin test files still use `_create_mock_context()` functions that return `MagicMock` objects, violating the Testing Charter.

**Files Affected** (14 files):
1. `tests/analytics/plugins/subsystems/test_build.py`
2. `tests/analytics/plugins/semantic_roles/test_compute.py`
3. `tests/analytics/plugins/history/test_timeseries.py`
4. `tests/analytics/plugins/functions/test_ast_features.py`
5. `tests/analytics/plugins/functions/test_effects.py`
6. `tests/analytics/plugins/functions/test_contracts.py`
7. `tests/analytics/plugins/risk/test_factors.py`
8. `tests/analytics/plugins/dependencies/test_external.py`
9. `tests/analytics/plugins/profiles/test_build.py`
10. `tests/analytics/plugins/data_models/test_build.py`
11. `tests/analytics/plugins/data_models/test_usage.py`
12. `tests/analytics/plugins/config_data_flow/test_compute.py`
13. `tests/analytics/plugins/test_plugins_tests/test_behavioral_coverage.py`
14. `tests/analytics/plugins/test_coverage_plugins/test_test_edges.py`

**Solution**: Replace `_create_mock_context()` with real context creation using `TestExecutionContextBuilder`:

```python
# Before (violates Testing Charter)
def _create_mock_context(*, has_config: bool = True) -> MagicMock:
    ctx = MagicMock()
    ctx.has_config.return_value = has_config
    if has_config:
        ctx.get_config.return_value = _create_config()
    else:
        ctx.get_config.side_effect = ValueError("Config not found")
    ctx.gateway = MagicMock()
    return ctx

# After (uses real production types)
def _create_context(
    tmp_path: Path,
    *,
    has_config: bool = True,
) -> PluginExecutionContext:
    builder = TestExecutionContextBuilder.create(tmp_path)
    if has_config:
        config = make_step_config(MyStepConfig, tmp_path)
        builder.with_config(MyStepConfig, config)
    return builder.build()
```

**Implementation Steps Per File**:
1. Remove `from unittest.mock import MagicMock` import
2. Add imports:
   ```python
   from tests._helpers.fakes import TestExecutionContextBuilder
   from tests._helpers.factories import make_step_config
   ```
3. Replace `_create_mock_context()` with `_create_context()` using real types
4. Update test functions to pass `tmp_path` fixture
5. Update assertions that rely on mock behavior (e.g., `assert_any_call`) to test actual behavior
6. Run linting and type checks

**Estimated Effort**: ~2 hours (8-10 min per file)

---

### Phase B: Consolidate Inline SnapshotRef Construction (MEDIUM PRIORITY)

**Problem**: 43 occurrences across 28 files still construct `SnapshotRef` objects inline instead of using factory functions.

**Target Files** (excluding `_helpers` internal files):

| Category | File Count | Files |
|----------|------------|-------|
| analytics | 14 | test_model_config_heuristics.py, test_tests_profiles_*.py, test_graph_runtime_cache.py, test_runtime_pool.py, test_feature_flags_behavior.py, test_function_profile_contract.py, test_coverage_analytics.py, test_executor_contracts.py, test_graph_metric_filters_integration.py, test_function_contracts_integration.py |
| ingestion | 5 | test_db_queries.py, test_resources.py, test_change_tracker.py, test_module_inventory.py, test_docstrings_inventory.py, test_recipe_executor.py |
| serving | 2 | test_backend_resource_runtime.py, test_auto_pipeline.py |
| storage | 1 | test_run_tracking.py |
| graphs | 1 | test_validation_flags.py |

**Solution**: Replace inline construction with `make_snapshot()` or `create_test_snapshot()`:

```python
# Before
snapshot = SnapshotRef(repo="test/repo", commit="abc123", repo_root=tmp_path)

# After
from tests._helpers.factories import make_snapshot
snapshot = make_snapshot(repo_root=tmp_path)
```

**Implementation Steps Per File**:
1. Add import: `from tests._helpers.factories import make_snapshot`
2. Replace `SnapshotRef(repo=..., commit=..., repo_root=...)` with `make_snapshot(repo_root=...)`
3. Remove `from codeintel.config.primitives import SnapshotRef` if no longer needed
4. Update any assertions using hardcoded repo/commit values to use `DEFAULT_REPO`/`DEFAULT_COMMIT`
5. Run linting and type checks

**Estimated Effort**: ~1.5 hours

---

### Phase C: Consolidate Direct Gateway Creation (LOW PRIORITY)

**Problem**: Some test files call `open_memory_gateway()` directly with varying options instead of using centralized helpers.

**Current Helpers Available**:
- `tests/_helpers.fakes.create_graph_gateway()` - standard gateway with schema
- `tests/_helpers.gateway.GatewayFactory` - configurable factory

**Files to Review**:
- `tests/analytics/functions/test_function_contracts.py`
- `tests/analytics/test_ast_metrics.py`
- `tests/analytics/core/test_config_registry.py`
- Others using direct `open_memory_gateway()` calls

**Solution**: Use existing gateway helpers where appropriate:

```python
# Before
gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)

# After
from tests._helpers.fakes import create_graph_gateway
gateway = create_graph_gateway()
```

**Estimated Effort**: ~30 minutes

---

## Prioritized Implementation Order

1. **Phase A** (Mock Context Migration) - Highest impact, addresses Testing Charter violations
2. **Phase B** (SnapshotRef Consolidation) - Medium impact, improves consistency
3. **Phase C** (Gateway Consolidation) - Low impact, nice-to-have cleanup

---

## Verification Strategy

After each file modification:

```bash
# Format and lint
uv run ruff format <file> && uv run ruff check <file> --fix

# Type check
uv run pyright <file>
uv run pyrefly check <file>

# Run tests
uv run pytest <file> -x -q
```

After completing each phase:

```bash
# Full verification
uv run pytest tests/analytics/plugins/ tests/analytics/functions/ -x -q
```

---

## Files Reference

### Infrastructure Files (DO NOT MODIFY during migration)

- `tests/_helpers/constants.py` - `DEFAULT_REPO`, `DEFAULT_COMMIT`, `DEFAULT_RUN_ID`
- `tests/_helpers/fakes/execution_contexts.py` - `TestExecutionContextBuilder`
- `tests/_helpers/fakes/configs.py` - `create_test_snapshot()`, `create_test_run_context()`
- `tests/_helpers/factories/step_config_factories.py` - `make_step_config()`, `make_snapshot()`
- `tests/_helpers/factories/config_factories.py` - `make_snapshot()` (alternative)

### Key Imports for Migration

```python
# Constants
from tests._helpers.constants import DEFAULT_REPO, DEFAULT_COMMIT

# Factories
from tests._helpers.factories import make_step_config, make_snapshot

# Fake builders
from tests._helpers.fakes import (
    TestExecutionContextBuilder,
    create_test_execution_context,
    create_test_snapshot,
    create_graph_gateway,
)
```

---

## Notes

- The `TestExecutionContextBuilder` uses real `PluginExecutionContext` from production code
- Tests that previously asserted mock interactions (e.g., `mock.assert_any_call()`) need to be restructured to test actual behavior
- Some tests may need additional fixtures (e.g., `tmp_path`, `fresh_gateway`) when migrating from mocks
- The `make_snapshot()` function is available in both `factories/config_factories.py` and `factories/step_config_factories.py` - prefer the one from `factories/__init__.py`

