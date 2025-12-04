# Analytics Cleanup Phase 3 - Detailed Implementation Plan

## Context and Rationale

### Background

The CodeIntel codebase has undergone a significant architectural migration to establish a unified core architecture in `codeintel.core`. This unified architecture provides canonical definitions for:

- **Plugins**: `PluginMetadata`, `PluginProtocol`, execution contexts
- **Resources**: `ResourceRegistry`, `ResourceProvider` pattern
- **Recipes**: `Recipe`, `RecipeStage`, `RecipeBuilder` for workflow composition

The analytics subsystem (`codeintel.analytics`) was one of the first domains to adopt this architecture. During the migration, backward-compatibility shim modules were created to allow gradual adoption without breaking existing code. Now that the migration is complete and all consumers have been updated, these shims are no longer needed.

### Why This Cleanup Is Necessary

1. **Eliminate Code Duplication**
   - The backward-compat shims (`analytics/tests/`, `analytics/tests_profiles/`) simply re-export from the canonical `analytics/testing/` module
   - This duplication increases maintenance burden and creates confusion about which import path to use
   - Removing shims forces all code to use the canonical paths, improving consistency

2. **Remove Dead Code**
   - Empty stub packages (`coverage/`, `hotspots/`, `risk/`, `config_data_flow/`) were placeholders that were never implemented
   - Empty directories (`pipeline/`, `compute/graphs/`) contain only `__pycache__/` from deleted code
   - Keeping these clutters the codebase and suggests features that don't exist

3. **Improve Naming Clarity**
   - `graph_metrics/` is easily confused with graph metric computations in `codeintel.graphs.compute.metrics/`
   - Renaming to `graph_primitives/` better reflects its purpose: providing primitive graph computation helpers used by analytics orchestration
   - This aligns with the established pattern where pure computations live in `graphs/` and analytics-specific orchestration lives in `analytics/`

4. **Consolidate RecipeBuilder**
   - The analytics `RecipeBuilder` duplicates most of the core `RecipeBuilder` implementation
   - By extending the core class, we reduce duplication and ensure feature parity
   - This follows the project principle of "shared functions and classes with subclasses and modification as needed"

### Architectural Alignment

This cleanup aligns the analytics subsystem with the established architecture:

```
codeintel/
├── core/                    # Canonical shared definitions
│   ├── plugins/             # Plugin protocol, metadata, registry
│   ├── resources/           # ResourceRegistry, ResourceProvider
│   └── recipes/             # Recipe, RecipeBuilder, RecipeStage
│
├── analytics/               # Analytics domain
│   ├── core/                # Analytics-specific plugin implementations
│   ├── testing/             # Test analytics (canonical location)
│   │   ├── coverage/        # Coverage edge computation
│   │   ├── behavioral/      # Behavioral tags, importance scoring
│   │   └── profiles/        # Test profile builders
│   ├── graph_primitives/    # Graph computation helpers (renamed)
│   └── recipes/             # Analytics recipes (extends core)
│
└── graphs/                  # Graph domain
    └── compute/metrics/     # Pure graph metric computations
```

### Quality Standards

All changes must satisfy:
- **Zero pyright errors** (strict mode)
- **Zero pyrefly errors** (sharp checks)
- **Zero ruff errors** (no suppressions allowed)
- **Full test suite passes**

These standards ensure the cleanup doesn't introduce regressions and maintains the codebase's type safety and code quality.

---

## Status Summary

**Completed:**
- Phase 1.1: Updated test files using `analytics.tests` → `analytics.testing`
- Phase 1.2: Updated production files using `analytics.tests` → `analytics.testing`

**Remaining:**
- Phase 1.3: Update test files using `analytics.tests_profiles`
- Phase 2: Delete backward-compat shim modules
- Phase 3: Delete empty stub packages
- Phase 4: Delete empty directories
- Phase 5: Rename `graph_metrics/` to `graph_primitives/`
- Phase 6: Consolidate RecipeBuilder to extend CoreRecipeBuilder
- Phase 7: Quality checks and validation

---

## Phase 1.3: Update Test Files Using `analytics.tests_profiles`

### Overview
Five test files import from the backward-compatibility shim `codeintel.analytics.tests_profiles`. These must be updated to import from the canonical `codeintel.analytics.testing` module structure.

### Files to Update

#### 1. `tests/analytics/test_tests_profiles_helpers.py`

**Current imports:**
```python
from codeintel.analytics.tests_profiles import behavioral_tags, coverage_inputs, importance, rows
from codeintel.analytics.tests_profiles.types import (
    BehavioralLLMRequest,
    BehavioralLLMResult,
    ImportanceInputs,
    IoFlags,
    TestAstInfo,
    TestProfileContext,
    TestRecord,
)
```

**New imports:**
```python
from codeintel.analytics.testing.behavioral import importance
from codeintel.analytics.testing.behavioral import tags as behavioral_tags
from codeintel.analytics.testing.coverage import inputs as coverage_inputs
from codeintel.analytics.testing.profiles import rows
from codeintel.analytics.testing.profiles.types import (
    BehavioralLLMRequest,
    BehavioralLLMResult,
    ImportanceInputs,
    IoFlags,
    TestAstInfo,
    TestProfileContext,
    TestRecord,
)
```

**Additional changes:**
- The test uses `_override(behavioral_tags, "ensure_schema", ...)` - this will work since we're aliasing the module
- References to `behavioral_tags.BehaviorRowHooks` and `behavioral_tags.BehavioralContext` remain valid

---

#### 2. `tests/analytics/test_function_profile_contract.py`

**Current imports:**
```python
from codeintel.analytics.tests_profiles import rows as test_rows
```

**New imports:**
```python
from codeintel.analytics.testing.profiles import rows as test_rows
```

---

#### 3. `tests/analytics/test_tests_profiles_registry_and_snapshots.py`

**Current imports:**
```python
from codeintel.analytics.tests_profiles import coverage_inputs, rows
from codeintel.analytics.tests_profiles.types import (
    IoFlags,
    TestAstInfo,
    TestProfileContext,
    TestRecord,
)
```

**New imports:**
```python
from codeintel.analytics.testing.coverage import inputs as coverage_inputs
from codeintel.analytics.testing.profiles import rows
from codeintel.analytics.testing.profiles.types import (
    IoFlags,
    TestAstInfo,
    TestProfileContext,
    TestRecord,
)
```

---

#### 4. `tests/analytics/test_tests_profiles_coverage_inputs.py`

**Current imports:**
```python
from codeintel.analytics.tests_profiles import coverage_inputs
```

**New imports:**
```python
from codeintel.analytics.testing.coverage import inputs as coverage_inputs
```

---

#### 5. `tests/analytics/test_tests_profiles_wrappers.py`

**Current imports:**
```python
from codeintel.analytics.tests_profiles.behavioral_tags import infer_behavior_tags
from codeintel.analytics.tests_profiles.coverage_inputs import (
    aggregate_test_coverage_by_function,
    aggregate_test_coverage_by_subsystem,
    load_test_graph_metrics,
)
from codeintel.analytics.tests_profiles.importance import (
    compute_flakiness_score,
    compute_importance_score,
)
from codeintel.analytics.tests_profiles.types import ImportanceInputs, IoFlags, TestAstInfo
```

**New imports:**
```python
from codeintel.analytics.testing.behavioral.tags import infer_behavior_tags
from codeintel.analytics.testing.coverage.inputs import (
    aggregate_test_coverage_by_function,
    aggregate_test_coverage_by_subsystem,
    load_test_graph_metrics,
)
from codeintel.analytics.testing.behavioral.importance import (
    compute_flakiness_score,
    compute_importance_score,
)
from codeintel.analytics.testing.profiles.types import ImportanceInputs, IoFlags, TestAstInfo
```

---

### Validation for Phase 1.3

```bash
# Run ruff check on updated test files
uv run ruff check tests/analytics/test_tests_profiles_helpers.py \
                  tests/analytics/test_function_profile_contract.py \
                  tests/analytics/test_tests_profiles_registry_and_snapshots.py \
                  tests/analytics/test_tests_profiles_coverage_inputs.py \
                  tests/analytics/test_tests_profiles_wrappers.py --fix

# Run pyright on updated files
uv run pyright tests/analytics/test_tests_profiles_*.py tests/analytics/test_function_profile_contract.py

# Run the specific tests to ensure they pass
uv run pytest tests/analytics/test_tests_profiles_helpers.py \
              tests/analytics/test_function_profile_contract.py \
              tests/analytics/test_tests_profiles_registry_and_snapshots.py \
              tests/analytics/test_tests_profiles_coverage_inputs.py \
              tests/analytics/test_tests_profiles_wrappers.py -q --no-cov
```

---

## Phase 2: Delete Backward-Compat Shim Modules

### 2.1 Delete `analytics/tests/` Directory

**Files to delete:**
```
src/codeintel/analytics/tests/__init__.py
src/codeintel/analytics/tests/coverage_edges.py
src/codeintel/analytics/tests/graph_metrics.py
src/codeintel/analytics/tests/profiles.py
```

**Command:**
```bash
rm -rf src/codeintel/analytics/tests/
```

### 2.2 Delete `analytics/tests_profiles/` Directory

**Files to delete:**
```
src/codeintel/analytics/tests_profiles/__init__.py
src/codeintel/analytics/tests_profiles/behavioral_tags.py
src/codeintel/analytics/tests_profiles/coverage_inputs.py
src/codeintel/analytics/tests_profiles/importance.py
src/codeintel/analytics/tests_profiles/rows.py
src/codeintel/analytics/tests_profiles/types.py
```

**Command:**
```bash
rm -rf src/codeintel/analytics/tests_profiles/
```

### Validation for Phase 2

```bash
# Verify no remaining imports to deleted modules
grep -r "from codeintel\.analytics\.tests import" src/ tests/
grep -r "from codeintel\.analytics\.tests\." src/ tests/
grep -r "from codeintel\.analytics\.tests_profiles import" src/ tests/
grep -r "from codeintel\.analytics\.tests_profiles\." src/ tests/

# All should return empty results
```

---

## Phase 3: Delete Empty Stub Packages

### Packages to Delete

These packages contain only `__init__.py` with no functional code:

1. **`src/codeintel/analytics/coverage/`**
   - Only contains `__init__.py`

2. **`src/codeintel/analytics/hotspots/`**
   - Only contains `__init__.py`

3. **`src/codeintel/analytics/risk/`**
   - Only contains `__init__.py`

4. **`src/codeintel/analytics/config_data_flow/`**
   - Only contains `__init__.py`

### Pre-deletion Verification

```bash
# Verify these are empty packages (only __init__.py)
ls -la src/codeintel/analytics/coverage/
ls -la src/codeintel/analytics/hotspots/
ls -la src/codeintel/analytics/risk/
ls -la src/codeintel/analytics/config_data_flow/

# Check for any imports to these packages
grep -r "from codeintel\.analytics\.coverage import" src/ tests/
grep -r "from codeintel\.analytics\.hotspots import" src/ tests/
grep -r "from codeintel\.analytics\.risk import" src/ tests/
grep -r "from codeintel\.analytics\.config_data_flow import" src/ tests/
```

### Deletion Commands

```bash
rm -rf src/codeintel/analytics/coverage/
rm -rf src/codeintel/analytics/hotspots/
rm -rf src/codeintel/analytics/risk/
rm -rf src/codeintel/analytics/config_data_flow/
```

---

## Phase 4: Delete Empty Directories

### Directories to Delete

These directories only contain `__pycache__/` and no Python files:

1. **`src/codeintel/analytics/pipeline/`**
   - Previously deleted Python files, only `__pycache__/` remains

2. **`src/codeintel/analytics/compute/graphs/`**
   - Only contains `__pycache__/`

### Deletion Commands

```bash
rm -rf src/codeintel/analytics/pipeline/
rm -rf src/codeintel/analytics/compute/graphs/

# Check if compute/ is now empty and can be removed
ls -la src/codeintel/analytics/compute/
# If empty, also remove:
rm -rf src/codeintel/analytics/compute/
```

---

## Phase 5: Rename `graph_metrics/` to `graph_primitives/`

### 5.1 Rename the Directory

```bash
mv src/codeintel/analytics/graph_metrics/ src/codeintel/analytics/graph_primitives/
```

### 5.2 Update Internal Module References

**File: `src/codeintel/analytics/graph_primitives/__init__.py`**

No changes needed - the internal imports reference `metrics.py` in the same directory.

### 5.3 Update External Imports

#### File: `src/codeintel/analytics/graph_service.py`

**Current:**
```python
from codeintel.analytics.graph_metrics import (
    # ... imports
)
```

**New:**
```python
from codeintel.analytics.graph_primitives import (
    # ... imports
)
```

#### File: `tests/analytics/test_feature_flags_behavior.py`

**Current:**
```python
from codeintel.analytics.graph_metrics import ...
```

**New:**
```python
from codeintel.analytics.graph_primitives import ...
```

### 5.4 Update `analytics/__init__.py` if Needed

Check if `graph_metrics` is exported from the main analytics package and update:

```python
# In src/codeintel/analytics/__init__.py
# Change any reference from graph_metrics to graph_primitives
```

### Validation for Phase 5

```bash
# Verify no remaining references to old name
grep -r "graph_metrics" src/codeintel/analytics/ --include="*.py"
grep -r "from codeintel\.analytics\.graph_metrics" src/ tests/

# Run tests that use graph metrics
uv run pytest tests/analytics/test_feature_flags_behavior.py -q --no-cov
```

---

## Phase 6: Consolidate RecipeBuilder to Extend CoreRecipeBuilder

### Current State Analysis

**Core RecipeBuilder (`src/codeintel/core/recipes/dsl.py`):**
- Full-featured builder with stages, plugins, configs, tags
- Has `add_stage()`, `timeout()`, `dry_run()`, `skip_on_unchanged()`, `max_parallel()`
- Builds `Recipe` from `codeintel.core.recipes.model`

**Analytics RecipeBuilder (`src/codeintel/analytics/recipes/dsl.py`):**
- Simpler version without stages
- Missing: `add_stage()`, `timeout()`, `dry_run()`, `skip_on_unchanged()`, `max_parallel()`
- Has `extend()` method (not in core)
- Builds `Recipe` from `codeintel.analytics.recipes.model`

### Refactoring Strategy

Since analytics uses a different `Recipe` model from `analytics.recipes.model`, we have two options:

**Option A: Re-export Core RecipeBuilder (if models are compatible)**
```python
# analytics/recipes/dsl.py
from codeintel.core.recipes import RecipeBuilder

__all__ = ["RecipeBuilder", "recipe"]

def recipe(name: str) -> RecipeBuilder:
    return RecipeBuilder(name)
```

**Option B: Extend Core RecipeBuilder (if analytics needs different behavior)**
```python
# analytics/recipes/dsl.py
from codeintel.analytics.recipes.model import Recipe, RecipeOptions
from codeintel.core.recipes import RecipeBuilder as CoreRecipeBuilder

class RecipeBuilder(CoreRecipeBuilder):
    """Analytics-specific recipe builder extending core."""
    
    def extend(self, recipe: Recipe) -> "RecipeBuilder":
        """Extend this recipe with plugins from another recipe."""
        for plugin in recipe.plugins:
            self.add(plugin)
        for plugin_name, config in recipe.default_configs.items():
            self.with_config(plugin_name, config)
        for t in recipe.tags:
            self.tag(t)
        return self
    
    def build(self) -> Recipe:
        """Build the analytics-specific recipe."""
        return Recipe(
            name=self._name,
            description=self._description,
            plugins=tuple(self._plugins),
            default_configs=dict(self._configs),
            tags=tuple(self._tags),
            options=RecipeOptions(
                fail_fast=self._fail_fast,
                max_duration_ms=self._max_duration_ms,
            ),
            version=self._version,
        )
```

### Implementation Steps

1. **Check model compatibility:**
   ```bash
   diff src/codeintel/core/recipes/model.py src/codeintel/analytics/recipes/model.py
   ```

2. **Refactor `analytics/recipes/dsl.py`:**
   - Remove duplicated methods that are identical to core
   - Only keep `extend()` and override `build()` if needed for analytics-specific Recipe model

3. **Update imports in `analytics/recipes/__init__.py`:**
   - Ensure `RecipeBuilder` is still exported

### Validation for Phase 6

```bash
# Run recipe-related tests
uv run pytest tests/analytics/core/test_recipes.py -q --no-cov

# Run any test that uses RecipeBuilder
grep -r "RecipeBuilder" tests/analytics/ --include="*.py" -l | xargs uv run pytest -q --no-cov
```

---

## Phase 7: Quality Checks and Validation

### 7.1 Run Ruff Check with Auto-fix

```bash
uv run ruff check src/codeintel/analytics/ --fix
uv run ruff format src/codeintel/analytics/
```

### 7.2 Run Pyright Type Checking

```bash
uv run pyright src/codeintel/analytics/ --pythonversion=3.13
```

### 7.3 Run Pyrefly Check

```bash
uv run pyrefly check src/codeintel/analytics/
```

### 7.4 Run Full Analytics Test Suite

```bash
uv run pytest tests/analytics/ -q --no-cov
```

### 7.5 Run Integration Tests

```bash
# Tests that cross analytics boundaries
uv run pytest tests/graphs/test_span_consistency_integration.py -q --no-cov
uv run pytest tests/orchestration/test_pipeline_catalog_entrypoint.py -q --no-cov
```

### 7.6 Full Quality Report

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
```

---

## Execution Order Summary

| Step | Phase | Description | Est. Time |
|------|-------|-------------|-----------|
| 1 | 1.3 | Update 5 test files for tests_profiles imports | 10 min |
| 2 | 1.3 | Validate Phase 1.3 (ruff, pyright, tests) | 5 min |
| 3 | 2.1 | Delete `analytics/tests/` directory | 1 min |
| 4 | 2.2 | Delete `analytics/tests_profiles/` directory | 1 min |
| 5 | 2 | Validate no remaining imports | 2 min |
| 6 | 3 | Delete 4 empty stub packages | 2 min |
| 7 | 4 | Delete empty directories | 1 min |
| 8 | 5.1 | Rename `graph_metrics/` → `graph_primitives/` | 1 min |
| 9 | 5.2-5.4 | Update all imports for renamed directory | 5 min |
| 10 | 5 | Validate Phase 5 | 3 min |
| 11 | 6 | Refactor RecipeBuilder to extend Core | 10 min |
| 12 | 6 | Validate Phase 6 | 3 min |
| 13 | 7 | Final quality checks (ruff, pyright, pyrefly) | 5 min |
| 14 | 7 | Full test suite validation | 10 min |

**Total estimated time: ~60 minutes**

---

## Risk Mitigation

### Potential Issues

1. **Circular imports after module restructuring**
   - Mitigation: Use `TYPE_CHECKING` guards for type-only imports

2. **Tests failing due to module path changes**
   - Mitigation: Run tests incrementally after each phase

3. **RecipeBuilder compatibility issues**
   - Mitigation: Check Recipe model compatibility before deciding approach

### Rollback Strategy

If issues arise:
1. Git stash or revert changes
2. Re-run tests on clean state
3. Identify specific failing component
4. Address incrementally rather than in bulk

---

## Success Criteria

- [ ] Zero pyright errors in `src/codeintel/analytics/`
- [ ] Zero pyrefly errors in `src/codeintel/analytics/`
- [ ] Zero ruff errors (no suppressions)
- [ ] All tests in `tests/analytics/` pass
- [ ] All integration tests pass
- [ ] No imports from deleted modules anywhere in codebase
- [ ] `graph_primitives/` is the canonical name
- [ ] `RecipeBuilder` extends core implementation

