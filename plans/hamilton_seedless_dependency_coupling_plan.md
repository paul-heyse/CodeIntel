# Hamilton Seedless Dependency Coupling Overhaul Plan

## Summary
This plan removes seeded datasets entirely and enforces end-to-end dependency coupling in the Hamilton DAG so that all derived outputs are produced from the current codebase run. It also introduces stronger preflight validation for dependency wiring and shifts schema validation to diagnostics (non-blocking) for compute-path outputs.

## Goals
- Eliminate seeded dataset support and any snapshot-driven inference that bypasses current-run compute.
- Ensure all query nodes (`q__*`) for produced datasets consume in-DAG data nodes, not snapshot loaders.
- Enforce explicit dependency coupling from targets to all downstream analytics/validation nodes.
- Preserve rich diagnostics for schema drift, nullability, and data quality without aborting DAG execution.

## Non-Goals
- No changes to storage layer behavior beyond removing build-time dependency on seeded datasets.
- No redesign of core AST/CST/SCIP extraction logic.
- No reliance on partial recompute logic; caching remains for performance only.

## Invariants
- No seed datasets exist. The only source of truth is the current repository snapshot.
- There is no partial recompute. Cached results may prove equality but never drive compute selection.
- All datasets produced by build must be computed in the current run and should not be loaded as inputs.

## Current Gaps (Observed)
- `q__*` nodes can load snapshots via `load_relation` before producers execute.
- Support spec treats view dependencies as `producer_target=None`, leading to snapshot-based loaders.
- Seeded dataset config is still available and can reintroduce snapshot-based inputs.
- Strict validation can abort run on empty outputs, preventing full inventory diagnostics.

---

## Proposed Design

### A) Build Config: Seedless, Compute-First
- Create or enforce `config/codeintel.build.toml` with:
  - `[hamilton] graph_backend = "compute"`
  - `schema_drift_mode = "warn"`
- Remove all seeded dataset config keys and manifest resolution.
- Explicitly reject any attempt to configure seeded datasets.

### B) Catalog-Backed Data Node Wiring
- Extend `DagCatalog` to expose a mapping of `table_key -> data_node` (derived from `ci.data_node` tag on saver nodes).
- Use that mapping to route `q__*` nodes directly to computed data nodes, not snapshot loaders.

### C) Support Spec: Accurate Producers for View Dependencies
- In `support_spec_from_catalog`, resolve view dependencies to their producer targets if present in the catalog.
- If a dependency is not produced by any target, classify it as an explicit external input.

### D) External Input Allowlist (Registry-Backed)
- The external input allowlist lives in `config/registry/external_inputs_allowlist.yaml` (registry-owned, versioned).
- Support spec reads this file directly; there is no build-config override.
- Proposed schema:
  ```yaml
  version: 1
  external_inputs:
    - table_key: "external.some_table"
      reason: "Required for X until producer target lands"
      owner: "team-or-module"
  ```

### E) Query Node Resolution Rules
- Produced datasets: `q__<table>` must `source(<data_node>)`.
- External datasets: `q__<table>` may `source(load_relation)` but only if allowlisted.
- Any non-allowlisted external dataset is a preflight error.

### F) Validation as Diagnostics (Non-Blocking)
- Validation failures (schema drift, nullability, min rows) must not abort DAG execution for compute-path datasets.
- Emit structured diagnostics under `build/diagnostics/` with fixed filenames and schemas:
  - `schema_drift.json`
    - Shape: `{ "generated_at": "...", "run_id": "...", "tables": [{ "table_key": "...", "drift_summary": {...} }] }`
  - `null_inventory.json`
    - Shape: `{ "generated_at": "...", "run_id": "...", "tables": [{ "table_key": "...", "row_count": 0, "null_counts": { "col": 123 } }] }`
  - `validation_findings.jsonl`
    - One JSON object per line: `{ "table_key": "...", "severity": "warn|error", "check": "...", "message": "...", "column": "...", "count": 0 }`
  - `external_input_usage.json`
    - Shape: `{ "generated_at": "...", "run_id": "...", "tables": [{ "table_key": "...", "loader_node": "...", "allowlisted": true }] }`

---

## Implementation Phases

### Phase 1: Config + Invariants
- [ ] Add `plans/architecture` note documenting invariants (seedless, compute-only, coupled DAG).
- [ ] Create `config/codeintel.build.toml` with `hamilton.graph_backend="compute"` and `schema_drift_mode="warn"`.
- [ ] Remove seed config keys from `BuildConfig` allowed keys and parsing (`src/codeintel/build/config.py`).
- [ ] Remove `seed_suite_manifest_path` and `ci_seeded_datasets` handling in `src/codeintel/build/hamilton/executor.py`.
- [ ] Update any docs/examples that mention seeded datasets.

### Phase 2: Catalog Enhancements
- [ ] Add `DagCatalog.table_data_nodes: Mapping[str, str]` (table_key -> data_node).
- [ ] Populate mapping from output tags (`ci.data_node`) during catalog compilation.
- [ ] Add preflight check: every table output must have a valid data node.

### Phase 3: Support Spec Accuracy
- [ ] Extend `SupportDatasetSpec` to include `data_node` and `producer_target`.
- [ ] In `support_spec_from_catalog`, resolve view base table keys to actual producers when possible.
- [ ] Load external input allowlist from `config/registry/external_inputs_allowlist.yaml` and validate against it.

### Phase 4: Query Node Rewire
- [ ] Update `_decorate_query_nodes` in `src/codeintel/build/hamilton/nodes/support_nodes.py`:
  - Produced dataset -> `source(data_node)`
  - External dataset -> `source(dataset_ref)` then `load_relation`
- [ ] Keep `dataset_ref` only for external inputs and artifacts.
- [ ] Add preflight validation for any dataset that uses `load_relation` without allowlist entry.

### Phase 5: Diagnostics-First Validation
- [ ] Adjust default validation behavior for build outputs to warn/diagnose without abort.
- [ ] Emit diagnostics with fixed filenames: `schema_drift.json`, `null_inventory.json`, `validation_findings.jsonl`, `external_input_usage.json`.
- [ ] Add table-level diagnostics for empty outputs (e.g., `graph.cfg_blocks` empty) without stopping execution.

### Phase 6: Tests + Safety Nets
- [ ] Add tests verifying `q__graph__cfg_blocks` depends on `cfg__blocks_table` (or data node) not `load_relation`.
- [ ] Add tests that seeded dataset config is rejected.
- [ ] Add preflight test for external inputs allowlist enforcement.
- [ ] Add run-level test that `t__goids` executes before `cfg` when `graph_backend=compute`.

---

## Validation & Acceptance Criteria
- `uv run codeintel build run --all --verbose=1` executes end-to-end without snapshot loaders for produced datasets.
- `q__*` nodes for produced tables depend on computed data nodes.
- Any schema mismatch or nullability issue appears in diagnostics without aborting the run.
- No seeded dataset config path remains in build code or config.

---

## Key Touchpoints
- Config: `src/codeintel/build/config.py`
- Executor: `src/codeintel/build/hamilton/executor.py`
- Catalog: `src/codeintel/build/hamilton/dag_catalog.py`
- Support Spec: `src/codeintel/build/hamilton/nodes/support_spec.py`
- Support Nodes: `src/codeintel/build/hamilton/nodes/support_nodes.py`
- Validation: `src/codeintel/build/hamilton/native/patterns/savers.py`, `src/codeintel/core/validation/engine.py`
- Diagnostics: `src/codeintel/build/hamilton/diagnostics.py`

---

## Open Questions
- Should validation diagnostics be table-key specific (per output) or aggregated in a single report?
- Should any dataset categories (e.g., SCIP) have stricter validation thresholds even in diagnostic mode?
