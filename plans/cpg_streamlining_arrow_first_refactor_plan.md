# CPG Streamlining + Arrow-First Refactor Plan

## Goal
Deliver the same CPG functionality with a smaller, clearer code footprint while
improving extensibility, robustness, and maintainability. Keep the data path
pyarrow-centric (RecordBatchReader, dataset scanning, and Arrow joins) and avoid
eager materialization unless strictly necessary.

## Scope
**In scope**
- `src/codeintel/build/hamilton/native/graphs/cpg.py` refactor into smaller modules.
- Shared graph assembly helpers (nodes/edges/IDs/contracts).
- Arrow-first conversion of graph assembly inputs and joins.
- Overlay registry for optional CPG edges (inspect, bytecode, etc.).
- Rollout gates and regression test segmentation.

**Out of scope**
- Changing semantic output schema or table keys.
- Altering Hamilton DAG topology beyond module boundaries and imports.
- Rewriting ingestion extractors (already Arrow-first).

## Principles
- Arrow-first at module boundaries: prefer `pa.RecordBatchReader` over tables/frames.
- Single alignment point: align to contract once per output stream.
- Single ID authority: all IDs originate from shared helpers.
- Minimal materialization: tables only at the last mile (if a join or view requires it).
- Strict compatibility: keep existing table keys, columns, and edge kinds.

## Phase 0: Baseline + Inventory
- [ ] Inventory graph assembly paths and identify redundant logic.
  - `src/codeintel/build/hamilton/native/graphs/cpg.py`
  - `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
  - `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`
  - `src/codeintel/build/hamilton/native/graphs/import_graph.py`
  - `src/codeintel/build/hamilton/native/graphs/pdg.py`
  - `src/codeintel/build/hamilton/native/graphs/symbol_use.py`
- [ ] Capture baseline metrics for memory and wall time on representative repos.
- [ ] Confirm current contract alignment points (and document required schemas).

## Phase 1: Modularize CPG Assembly
- [ ] Create a `cpg` package to split the monolith into focused modules:
  - `src/codeintel/build/hamilton/native/graphs/cpg/__init__.py`
  - `src/codeintel/build/hamilton/native/graphs/cpg/nodes.py`
  - `src/codeintel/build/hamilton/native/graphs/cpg/edges.py`
  - `src/codeintel/build/hamilton/native/graphs/cpg/bytecode.py`
  - `src/codeintel/build/hamilton/native/graphs/cpg/inspect_overlay.py`
  - `src/codeintel/build/hamilton/native/graphs/cpg/ids.py`
- [ ] Reduce `src/codeintel/build/hamilton/native/graphs/cpg.py` to a thin
  re-export layer so Hamilton node names remain stable.
- [ ] Consolidate CPG constants and table keys into a single module.

## Phase 2: Graph Assembly Helpers (Arrow-First)
- [ ] Introduce a shared graph assembly toolkit:
  - `src/codeintel/build/graphs/assembly/readers.py`
    - Reader helpers: alignment, single-consume safeguards, minimal peeking.
  - `src/codeintel/build/graphs/assembly/collectors.py`
    - Build edges/nodes with `ColumnarBatchCollector` and emit readers.
  - `src/codeintel/build/graphs/assembly/ids.py`
    - Stable ID construction, decimal-safe IDs, and normalization utilities.
  - `src/codeintel/build/graphs/assembly/contracts.py`
    - Contract alignment helpers (one place to enforce schema policy).
- [ ] Replace per-module row -> table conversions with shared helpers.
- [ ] Standardize edge/node assembly to return `pa.RecordBatchReader`.

## Phase 3: Arrow-First Inputs + Joins
- [ ] Update graph modules to accept `RecordBatchReader` inputs where possible.
  - Prefer `tabular_to_arrow_reader` + Arrow filters/projects.
  - Convert to `pa.Table` only when a join API requires it.
- [ ] Centralize Arrow joins via `src/codeintel/build/tabular/arrow_ops.py`.
  - Use `ArrowJoinSpec` consistently.
  - Keep join validation logic in one place.
- [ ] Ensure no eager `list(reader)` or `read_all()` in graph paths.

## Phase 4: Overlay Registry and Feature Gates
- [ ] Introduce a CPG overlay registry:
  - Each overlay returns `{table_key: RecordBatchReader}` plus metadata.
  - Registry applies feature flags and allowlists before execution.
- [ ] Move inspect overlay logic behind registry:
  - Keep allowlist-only behavior.
  - Require explicit enable flags for runtime overlays.
- [ ] Make bytecode DFG/REACHES overlays pluggable through the registry.

## Phase 5: ID Authority + Contract Enforcement
- [ ] Move all CPG node/edge ID logic to `graphs/assembly/ids.py`.
- [ ] Ensure decimal-safe ID columns for CPG edges/nodes (single authority).
- [ ] Align all outputs to contract schemas in one place.

## Phase 6: Migration of Adjacent Graphs
- [ ] Convert the following to the shared helpers:
  - `call_wiring.py` (call/arg/param edges)
  - `cfg_dfg.py` (cfg/dfg edges)
  - `import_graph.py` (import edges)
  - `pdg.py` (program dependence edges)
  - `symbol_use.py` (symbol use edges)
- [ ] Remove duplicated helpers after migration.

## Phase 7: Test Gating + Segmented Runs
- [ ] Codify regression gates using the segmented runner:
  - `uv run python -m tools.pytest_gate`
  - Default targeted subset: symtable/dis/ast/inspect ingestion tests.
  - Default segments: ingestion, build, graphs, storage, serving, runtime, analytics.
- [ ] Document the gate runner in the plan and dev docs.

## Phase 8: Cleanup + Deprecations
- [ ] Remove unused conversion helpers once new graph assembly is stable.
- [ ] Deprecate table-based helpers in graph modules when readers suffice.
- [ ] Update doc references to the new module layout.

## Deliverables
- Modular CPG codebase with clear ownership boundaries.
- Shared Arrow-first assembly toolkit.
- Overlay registry for optional edge providers.
- Single authority for IDs and contract alignment.
- Segmented regression gate runner.

## Acceptance Criteria
- CPG outputs remain schema-compatible and row-stable vs baseline.
- No eager materialization in graph assembly paths (unless explicitly justified).
- CPG overlays are enabled/disabled via explicit flags and allowlists.
- Regression gate runner passes for targeted + segmented runs.

## Validation Plan
- Run quality report:
  - `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Run regression gates:
  - `uv run python -m tools.pytest_gate`
- Spot-check key CPG tables for row-count deltas and schema drift.

## Risks and Mitigations
- **Risk:** Arrow joins may differ from Polars defaults.
  - **Mitigation:** enforce join validation in one helper and compare row counts.
- **Risk:** Refactor churn in `cpg.py` affects DAG nodes.
  - **Mitigation:** keep `cpg.py` as stable re-export and add tests for node names.
- **Risk:** Single-consume readers used twice.
  - **Mitigation:** enforce single-consume via helper and document usage patterns.

## Rollout Strategy
1) Land shared assembly toolkit and ID authority (no behavior change).
2) Migrate CPG internals to the toolkit; keep output parity tests.
3) Migrate other graphs (call/import/cfg/dfg/pdg) to the toolkit.
4) Enable overlay registry; keep inspect disabled by default.
5) Remove legacy helpers after parity is validated.
