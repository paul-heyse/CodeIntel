# Build Consolidation + Streamlining Implementation Plan

## TL;DR
Consolidate duplicated analytics logic, unify columnar utilities, and tighten layering
between pure compute, orchestration, and Hamilton-native IO. The end state is a
smaller, more consistent API surface with fewer sources of truth and clearer
boundaries for extension.

## Goals
- Enforce a strict layering boundary: pure compute has no IO dependencies.
- Reduce duplicate implementations across analytics, graphs, and columnar utilities.
- Standardize schema-driven row construction and column ordering.
- Simplify maintenance and extension points for new analytics targets.
- Improve testability by making compute modules deterministic and IO-free.

## Non-Goals
- No functional changes to analytics outputs beyond bug fixes needed for consolidation.
- No sweeping renames of existing target names or table keys.
- No behavioral changes to end-user CLI flows beyond internal refactors.

## Scope
Primary scope:
- `src/codeintel/build/analytics/**`
- `src/codeintel/build/graphs/**`
- `src/codeintel/build/tabular/**`
- `src/codeintel/build/hamilton/native/**` (wrappers only)
- `src/codeintel/build/analytics/utilities/**`

Secondary scope:
- `src/codeintel/core/columnar/**` (alignment helpers used by consolidated utilities)
- `src/codeintel/build/run_context.py` (cleanup of unused knobs)

## Target Architecture
Layering principles:
1. **Pure compute**: stateless, deterministic, no IO.
2. **Orchestration**: combines compute results with IO-backed inputs.
3. **Hamilton-native**: IO, materialization, and DAG-specific glue.

Target shape:
- `build/graphs/compute/**` is the canonical home for graph metrics primitives.
- `build/analytics/compute/**` is canonical for analytics-specific computation.
- `build/analytics/**` orchestration may read filesystem/DB and call compute.
- `build/tabular/**` is the single place for Arrow/Polars/DuckDB conversions.
- `build/hamilton/native/**` is thin integration; logic lives in compute/orchestration.

## Workstreams

### W1: Columnar + Frame Utilities Consolidation (foundation)
Problem: duplicated frame creation and schema alignment in
`hamilton/native/analytics/table_utils.py` and
`hamilton/native/ingestion/frame_utils.py`.

Plan:
- Create `src/codeintel/build/tabular/frames.py` with:
  - `empty_frame_for_table`
  - `rows_to_frame`
  - `lazyframe_for_table_columns`
  - `lazyframe_for_ingest_columns`
  - `dedupe_frame_for_table`
  - `to_records`
- Route all callers to `build/tabular/frames.py`.
- Keep old helpers as thin wrappers with deprecation notes, then delete.

Acceptance:
- Single column-ordering implementation and schema alignment path.
- No duplicate schema alignment logic across analytics and ingestion.

### W2: Dependency Detection Consolidation
Problem: duplicate call detection and alias mapping in
`analytics/dependencies/core.py` and
`analytics/compute/dependencies/detection.py`.

Plan:
- Make `analytics/compute/dependencies/detection.py` canonical for:
  - alias mapping
  - AST call detection
  - classification helpers
- Refactor `analytics/dependencies/core.py` to import and use compute helpers only.
- Remove duplicate `DependencyCallVisitor` and alias-map logic from core.
- Ensure all path normalization occurs in one place (compute layer).

Acceptance:
- One source of truth for dependency call detection and classification.
- Core/orchestration layer does not own compute logic.

### W3: Entrypoints Compute/IO Split
Problem: `analytics/compute/entrypoints/compute.py` depends on IO-heavy core helpers.

Plan:
- Move IO-dependent logic (filesystem scanning) into `analytics/entrypoints/runtime.py`.
- Keep `analytics/compute/entrypoints/detection.py` purely AST-based.
- Provide a pure compute API that accepts in-memory sources or ASTs.
- Update Hamilton native module to use runtime wrappers, not compute directly.

Acceptance:
- Compute module is IO-free and can run on in-memory inputs.
- Runtime and Hamilton native layers own filesystem access.

### W4: Data Models and Usage Consolidation
Problem: data model usage relies on Hamilton ingestion helpers and duplicated framing.

Plan:
- Replace `build/hamilton/native/ingestion/frame_utils` calls with `build/tabular/frames`.
- Extract any compute logic from `analytics/compute/data_models/usage.py` that is IO-related
  into orchestration modules.
- Ensure data model usage compute can run in tests without filesystem access.

Acceptance:
- Data model usage compute is pure and mockable.
- No direct Hamilton-native utility dependencies inside compute.

### W5: Graph Metrics Consolidation
Problem: overlapping implementations in `build/graphs/compute/**` and
`build/analytics/compute/graphs/**`.

Plan:
- Make `build/graphs/compute/**` the single source of graph metrics primitives.
- Convert `build/analytics/compute/graphs/**` to lightweight re-exports or remove.
- Merge CFG/DFG metric helpers so `analytics/cfg_dfg/**` uses `graphs/compute/**`.
- Align `GraphContext` usage across modules.

Acceptance:
- One implementation per metric primitive.
- Analytics modules only orchestrate, not re-implement primitives.

### W6: Schema-Driven Row Construction
Problem: repeated manual `*_COLS` lists and tuple ordering logic.

Plan:
- Adopt `columns_for_table_key` consistently.
- Add shared row-builder helpers in `analytics/compute/row_builders/` for tables
  that currently rely on manual column lists.
- Replace tuple assembly with schema-driven dict-to-row conversion where possible.

Acceptance:
- No manual column ordering in analytics modules unless required by performance.
- A single schema-driven path for row ordering.

### W7: Type Coercion + Lazy Import Cleanup
Problem: utility duplication across analytics and core.

Plan:
- Route analytics type coercion to `codeintel.core.query_results`.
- Replace `analytics/utilities/lazy_module.py` with `core/imports/lazy.py` helpers.
- Keep compatibility wrappers temporarily and then delete.

Acceptance:
- One canonical utility for coercion and lazy loading.
- Utility modules in analytics become thin or removed.

### W8: BuildRunContext Cleanup
Problem: `BuildRunContext.build_env` accepts knobs that are currently no-ops.

Plan:
- Either implement `load_catalogs` / `load_schema_service` or remove them.
- Update call sites to reflect the chosen behavior.

Acceptance:
- Public API surface matches actual behavior.

## Phased Implementation Plan

### Phase 0: Design + Inventory (1-2 days)
1. Catalog all duplicate utilities and call sites.
2. Define the canonical module for each utility.
3. Document a deprecation map for transitional wrappers.

Deliverables:
- Consolidation map (old path -> new path).
- List of call sites per module.

### Phase 1: Columnar/Frame Utilities (2-4 days)
1. Add `build/tabular/frames.py` with the consolidated API.
2. Switch callers in analytics + Hamilton native to new API.
3. Add thin wrappers in old modules with TODO deprecation notes.

Acceptance gate:
- All columnar frame construction paths go through `build/tabular/frames.py`.

### Phase 2: Dependency Detection (2-3 days)
1. Move canonical logic into `analytics/compute/dependencies/detection.py`.
2. Refactor `analytics/dependencies/core.py` to use compute utilities.
3. Remove duplicate visitor and alias-map logic.

Acceptance gate:
- Single detection implementation; no duplicated classes/functions.

### Phase 3: Entrypoints + Data Models (3-5 days)
1. Introduce `analytics/entrypoints/runtime.py` for IO-heavy scanning.
2. Make compute path pure and operate on provided sources/ASTs.
3. Replace ingestion frame helpers with tabular frames in data model usage.

Acceptance gate:
- Compute modules contain no filesystem or Hamilton-native imports.

### Phase 4: Graph Metrics (4-6 days)
1. Consolidate CFG/DFG primitives into `build/graphs/compute/**`.
2. Convert analytics compute graph modules to re-exports or remove.
3. Update analytics/cfg_dfg to use canonical primitives.

Acceptance gate:
- One canonical implementation per graph primitive.

### Phase 5: Schema-Driven Rows + Cleanup (2-4 days)
1. Implement row builder utilities for high-volume tables.
2. Remove manual `*_COLS` where feasible.
3. Remove deprecated wrappers and update docs.

Acceptance gate:
- Consistent schema ordering; fewer manual column lists.

## Migration + Deprecation Strategy
- Use transitional wrappers in old modules for one release window.
- Add a single `DEPRECATION_NOTES.md` entry or inline docstrings.
- Remove wrappers once all call sites are migrated and tests pass.

## Testing + Quality Gates
Mandatory:
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- `uv run pytest -q` for impacted subpackages:
  - `tests/build/analytics/**`
  - `tests/build/graphs/**`
  - `tests/build/tabular/**`
- Segmented pytest runs for major directories as per AGENTS.md.

Suggested additions:
- Unit tests for consolidated utility modules (tabular frames, detection).
- Snapshot tests for row ordering via `columns_for_table_key`.

## Documentation Updates
- Update `docs/architecture.md` with layering diagram and canonical locations.
- Add a short “Consolidated Utilities” section to `docs/build_consolidation_plan.md`.

## Risks + Mitigations
- **Risk**: subtle schema alignment changes.
  - Mitigation: snapshot tests for row ordering; schema hash checks.
- **Risk**: performance regressions in graph metrics.
  - Mitigation: run on representative dataset sizes; keep hot loops unchanged.
- **Risk**: accidental IO in compute modules.
  - Mitigation: lint rule or review checklist enforcing IO-free compute.

## Success Metrics
- 1 canonical implementation for each utility or compute function.
- Decrease in total module count in `build/analytics` and `build/graphs`.
- Fewer manual column lists and row tuple builders.

## Open Questions
- Should `analytics/*/core.py` be retained as orchestration, or merged into
  compute + runtime modules explicitly?
- Do we want a formal “compute purity” test or lint rule?
- Is it acceptable to rename module namespaces to reflect layering (e.g., `runtime`)?
