# Config Refinement — Legacy/Compat Decommission Plan

**Status**: Implementation plan  
**Last updated**: 2025-12-18  
**Primary scope**: `src/codeintel/config/**`  
**Secondary scope**: call-site updates in `src/**` + `tests/**` required to delete config legacy/compat code

## 0) Objectives

1) **Delete dead/unused config code** that is not referenced by runtime or tests.
2) **Decommission compatibility/legacy shims** that exist only to bridge older module boundaries.
3) **Eliminate duplicated “domain logic” living under config** (e.g., analytics rows/helpers).
4) Preserve or improve:
   - type safety (pyright strict + pyrefly)
   - lint cleanliness (ruff, no suppressions)
   - functionality (pytest green)

## 1) Acceptance Gates (run after each phase)

```bash
uv run ruff check --fix
uv run pyright --warnings --pythonversion=3.13
uv run pyrefly check
uv run pytest -q
```

When iterating quickly, scope down to config-focused checks:

```bash
uv run ruff check --fix src/codeintel/config tests/config tests/_helpers
uv run pyright --warnings --pythonversion=3.13 src/codeintel/config tests/config tests/_helpers
uv run pyrefly check src/codeintel/config tests/config tests/_helpers
uv run pytest -q tests/config
```

## 2) Workstream Overview (What We’re Removing)

### A) Dead modules (no call sites)

These appear unused by both runtime and tests and can be deleted once confirmed:

- `src/codeintel/config/resolver.py`
- `src/codeintel/config/layering_checks.py`
- `src/codeintel/config/graph_helpers.py` (after removing re-exports)
- `src/codeintel/config/datasets/semantic_roles.py`

### B) Tests-only / duplicated domain logic

- `src/codeintel/config/datasets/dependencies.py`
  - Used by `tests/_helpers/rows.py`
  - Duplicates dependency-id logic already implemented elsewhere

### C) Legacy compat shims / cross-layer coupling

- `src/codeintel/config/datasets/__init__.py` wrapper functions
- `src/codeintel/config/datasets/columns.py`:
  - unused SQL constants
  - `load_columns_by_table()` depends on `codeintel.build.schemas` via lazy import

## 3) Phase Plan (Sequenced to Minimize Rework)

### Phase 0 — Confirm “Dead” with repo-wide search (no edits)

**Goal**: ensure we aren’t deleting something referenced indirectly (docs, scripts, CLI).

Run:

```bash
rg -n "resolve_tools_config|resolve_scan_profiles|resolve_graph_backend" -S .
rg -n "layering_checks" -S .
rg -n "GraphMetricWeights|GraphPluginPolicy|GraphRunScope|graph_helpers" -S .
rg -n "FunctionSemanticRoleRow|ModuleSemanticRoleRow" -S .
rg -n "DependencyCallRow|DependencyAggregateRow|compute_dep_id\\(" -S .
rg -n "AST_NODES_DELETE|CFG_BLOCKS_DELETE|CALL_GRAPH_EDGES_DELETE|SYMBOL_USE_DELETE|FILE_STATE_DELETE|TAGS_INDEX_DELETE" -S .
```

**Gate**: everything listed in §2 is either unused or has identified call sites to migrate.

---

### Phase 1 — Remove `graph_helpers.py` (re-export-only compat)

**Goal**: delete a module that exists only as a “moved-from steps_graphs.py” placeholder.

1) Remove re-exports from `src/codeintel/config/__init__.py`
   - `GraphMetricWeights`
   - `GraphMetricPluginSelection`
   - `GraphMetricPluginOverrides`
   - `GraphPluginPolicy`
   - `GraphRunScope`
   - `GraphMetricsTuning`
2) Confirm no imports exist besides the re-export:
   - `rg -n "codeintel\\.config\\.graph_helpers|GraphMetricWeights|GraphPluginPolicy|GraphRunScope" -S src tests`
3) Delete `src/codeintel/config/graph_helpers.py`

**Gate**
- `uv run ruff/pyright/pyrefly` succeed
- `uv run pytest -q tests/config` succeeds

---

### Phase 2 — Delete `layering_checks.py` (dead script-in-package)

**Goal**: remove unreferenced tooling from the runtime package surface.

Options:

- **Option A (recommended)**: move to `tools/` (if you want to keep the check)
- **Option B**: delete outright

Steps:

1) Confirm no call sites in code/tests.
2) Choose A or B, then remove `src/codeintel/config/layering_checks.py`.

**Gate**
- `uv run ruff/pyright/pyrefly` succeed
- `uv run pytest -q` succeeds

---

### Phase 3 — Remove `resolver.py` (dead + layering-odd)

**Goal**: eliminate unused “env override resolver” functions and remove the core→domain import smell.

Steps:

1) Confirm no call sites.
2) Delete `src/codeintel/config/resolver.py`.

If you later discover you need these helpers, reintroduce them in an appropriate layer:

- CLI runtime resolution belongs in `src/codeintel/cli/resolution/**`, or
- ingestion runtime resolution belongs in `src/codeintel/ingestion/**`.

**Gate**
- `uv run ruff/pyright/pyrefly` succeed
- `uv run pytest -q` succeeds

---

### Phase 4 — Remove dataset package compat wrappers (`datasets/__init__.py`)

**Goal**: stop encouraging “import from package root” patterns that hide true sources-of-truth.

Steps:

1) Enumerate imports like `from codeintel.config.datasets import ...` (if any appear later).
2) Update call sites to import directly:
   - `src/codeintel/config/datasets/contracts.py`
   - `src/codeintel/config/datasets/primitives.py`
   - `src/codeintel/config/datasets/columns.py` (or its replacement; see Phase 6)
3) Reduce `src/codeintel/config/datasets/__init__.py` to a minimal docstring (or delete exports).

**Gate**
- `uv run ruff/pyright/pyrefly` succeed
- `uv run pytest -q tests/config` succeeds

---

### Phase 5 — Remove `datasets/semantic_roles.py` (unused + duplicated domain logic)

**Goal**: ensure semantic role logic is owned by analytics and/or generated row models.

Steps:

1) Confirm no code/tests import `codeintel.config.datasets.semantic_roles`.
2) Delete `src/codeintel/config/datasets/semantic_roles.py`.

If a row type is needed for serialization, prefer:

- generated types in `src/codeintel/core/schemas/generated_rows/**`, and/or
- analytics-owned row builders under `src/codeintel/analytics/semantic_roles/**`.

**Gate**
- `uv run ruff/pyright/pyrefly` succeed
- `uv run pytest -q` succeeds

---

### Phase 6 — Decommission `datasets/dependencies.py` (tests-only + duplicated logic)

**Goal**: remove config-layer duplication of dependency IDs and row shapes.

Steps:

1) Migrate `tests/_helpers/rows.py`:
   - Replace imports from `src/codeintel/config/datasets/dependencies.py`
   - Use generated row models (`src/codeintel/core/schemas/generated_rows/analytics.py`) or a single
     canonical helper in analytics (if you want a stable “dep_id” constructor).
2) Ensure `dep_id` computation is sourced from one place:
   - Either expose a public helper from `src/codeintel/analytics/dependencies/**`
   - Or define a minimal shared helper under `src/codeintel/core/**` if both analytics and tests need it.
3) Delete `src/codeintel/config/datasets/dependencies.py`.

**Gate**
- `uv run ruff/pyright/pyrefly` succeed
- `uv run pytest -q` succeeds

---

### Phase 7 — Refactor `datasets/columns.py` (remove unused SQL constants + remove build coupling)

This is the most important “compat purge” because it hides a cross-layer dependency:
`codeintel.config.datasets.columns` (config/core) imports `codeintel.build.schemas` (domain) lazily.

#### 7.1 Remove dead SQL constants

**Goal**: delete unused constants that are exported but have no call sites.

Steps:

1) Confirm each constant is unused with repo-wide grep.
2) Delete unused constants and prune `__all__`.
3) Decide what to do with `TEST_CATALOG_UPDATE_GOIDS`:
   - keep it in `columns.py`, or
   - move it closer to the only call site (`src/codeintel/analytics/testing/coverage/edges.py`)

#### 7.2 Remove config→build coupling in `load_columns_by_table()`

**Goal**: `load_columns_by_table()` should not import build-owned providers.

Choose one:

- **Option A (recommended)**: derive from declared schemas
  - Build `columns_by_table` from `src/codeintel/config/datasets/declared_schemas.py`
  - This keeps config self-contained and deterministic.
- **Option B**: move `load_columns_by_table()` into a build/ingestion module
  - Update call sites (ingestion, analytics utilities) accordingly.
  - Keep `serialize_row(...)` where it logically belongs (likely core/util).

Update call sites identified today:

- `src/codeintel/ingestion/compute/typing_ingest.py`
- `src/codeintel/ingestion/compute/docstrings_extract.py`
- `src/codeintel/ingestion/adapters/hash_change_detection.py`
- `src/codeintel/ingestion/adapters/duckdb_storage.py`
- `src/codeintel/analytics/utilities/datasets.py`
- `src/codeintel/core/validation/reporters.py`

**Gate**
- `uv run ruff/pyright/pyrefly` succeed
- `uv run pytest -q` succeeds

## 4) Final “Purge Checklist”

By the end of the work:

- `src/codeintel/config/resolver.py` is gone.
- `src/codeintel/config/layering_checks.py` is gone (or moved to `tools/` and removed from package).
- `src/codeintel/config/graph_helpers.py` is gone and not re-exported by `src/codeintel/config/__init__.py`.
- `src/codeintel/config/datasets/semantic_roles.py` is gone.
- `src/codeintel/config/datasets/dependencies.py` is gone and tests no longer import it.
- `src/codeintel/config/datasets/__init__.py` no longer provides wrapper/compat exports.
- `src/codeintel/config/datasets/columns.py` no longer exports unused SQL constants.
- `load_columns_by_table()` no longer imports `codeintel.build.schemas` from config/core.

