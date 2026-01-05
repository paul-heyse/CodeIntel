# PyArrow Repo-Wide Intensification Plan

## Goal
Expand PyArrow usage across the repo (build, ingestion, analytics, validation)
to reduce Polars/Python-row overhead, while preserving the hard boundary between
`src/codeintel/build` and `src/codeintel/storage`.

## Boundary constraints
- Build emits Arrow readers / Parquet snapshots; storage consumes Parquet/Arrow.
- No new imports from `src/codeintel/build` inside `src/codeintel/storage`.
- Storage changes must be internal to storage (no cross-layer coupling).

## References
- `docs/python_library_reference/pyarrow-advanced.md`
- `plans/arrow_first_build_maximal_plan.md`
- `plans/pyarrow_compute_deployment_plan.md`
- `docs/arrow_join_policy.md`

## Target state
- Arrow tables/readers are the default across build, ingestion, analytics,
  and validation layers.
- Polars is limited to explicit UI/presentation or legacy boundaries only.
- Parquet exports use Arrow metadata and dictionary encoding for categorical
  columns where appropriate.

---

## Phase 0: Inventory and guardrails
Purpose: establish an auditable map of where Arrow should replace Polars.

Checklist
- [ ] Inventory Polars usage under `src/codeintel/build` and `src/codeintel/ingestion`.
- [x] Expand `docs/arrow_join_policy.md` with any new ingestion/analytics joins.
- [x] Link `plans/pyarrow_compute_deployment_plan.md` back to this plan.
- [x] Centralize Arrow mask helpers + string_view handling
  (`src/codeintel/build/tabular/compute_masks.py`,
  `src/codeintel/build/tabular/arrow_ops.py`).

File targets
- `docs/arrow_join_policy.md`
- `scripts/ci/arrow_first_guard.sh` (extend scope if needed)
- `plans/pyarrow_compute_deployment_plan.md` (link to this plan)

Acceptance
- Join inventory reflects all join-like operations in build + ingestion.
- Guardrail(s) exist for Polars usage in build compute modules.

---

## Phase 1: Arrow-first ingestion pipeline
Purpose: remove Polars from ingestion joins and enrich steps.

Checklist
- [x] Convert join-heavy ingestion modules to Arrow tables + `arrow_join_tables`
  (`syntax_enrich.py`, `syntax_augment.py`, `scip_resolution.py`).
- [x] Replace LazyFrame filters with Arrow compute + shared mask helpers.
- [x] Align outputs to contract schema using `align_table_to_contract`.
- [x] Preserve existing dedupe behavior with `dedupe_table_for_table`.
- [ ] Sweep remaining ingestion modules for Polars usage
  (`scip.py`, `file_line_index.py`, `frame_utils.py`).

File targets
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- `src/codeintel/build/hamilton/native/ingestion/scip.py`
- `src/codeintel/build/hamilton/native/ingestion/file_line_index.py`
- `src/codeintel/build/hamilton/native/ingestion/frame_utils.py`

Acceptance
- No Polars usage in ingestion compute modules.
- Outputs remain schema-aligned and match current row counts.

---

## Phase 2: Build analytics to Arrow compute
Purpose: move high-volume analytics calculations to Arrow kernels.

Checklist
- [x] Use `pyarrow.compute` mask helpers in analytics + validation paths
  (`graph_metrics.py`, `semantic_roles/core.py`,
  `graphs/validation/runner.py`).
- [x] Convert analytics joins to Arrow where touched (subsystem cache join).
- [ ] Replace remaining Polars groupby/agg with `Table.group_by().aggregate(...)`
  across `src/codeintel/build/analytics/**`.

File targets (examples)
- `src/codeintel/build/analytics/**`
- `src/codeintel/build/hamilton/native/analytics/**`
- `src/codeintel/build/graphs/validation/**`

Acceptance
- Analytics pipelines operate on Arrow readers/tables.
- No regression in metric outputs (spot-check with existing tests).

---

## Phase 3: Graph validation and dataset helpers
Purpose: avoid Polars conversion in build-time validation and dataset scans.

Checklist
- [x] Arrow-first validation filters in `graphs/validation/runner.py`.
- [ ] Use `pyarrow.dataset.Scanner` for dataset scans with projection + filters.
- [ ] Replace `scan_snapshot_lazyframe` call sites with Arrow readers/tables.
- [ ] Apply row-group pruning where filters are present.

File targets
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/build/graphs/validation/runner.py`
- `src/codeintel/build/graphs/validation/checks/**`

Acceptance
- Validation paths run Arrow-first end-to-end.

---

## Phase 4: Schema evolution + metadata
Purpose: make Arrow schema metadata a first-class signal without changing
the build/storage boundary.

Checklist
- [x] Normalize Arrow join behavior and string_view handling
  (`arrow_ops.py`, `view_outputs.py`).
- [ ] Attach schema-level metadata on build outputs (snapshot, tool version).
- [ ] Use `pa.unify_schemas(..., promote_options="permissive")` before concat.
- [ ] If needed, use `Table.cast` to normalize types before concat.

File targets
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/exports/**`

Acceptance
- Parquet outputs carry schema metadata consistently.
- Concat paths use unified schema + cast alignment.

---

## Phase 5: Dictionary encoding for categorical columns
Purpose: reduce memory and improve join/groupby speed.

Checklist
- [x] Preserve dictionary encode toggles in export writers (`writers.py`).
- [ ] Identify categorical columns (kinds, enums, languages, roles).
- [ ] Apply dictionary encoding before export with `Table.dictionary_encode()`.
- [ ] Normalize dictionaries across batches with `Table.unify_dictionaries()`.
- [ ] Ensure exporters preserve dictionary types.

File targets
- `src/codeintel/build/exports/writers.py`
- `src/codeintel/build/exports/engine.py`
- `src/codeintel/build/tabular/arrow_ops.py`

Acceptance
- Dictionary-encoded columns are preserved in Parquet outputs.
- No schema drift in downstream consumption.

---

## Phase 6: Parquet metadata sidecars for fast scans
Purpose: speed dataset discovery and scanning.

Checklist
- [ ] Collect `FileMetaData` when writing Parquet shards.
- [ ] Emit `_metadata` / `_common_metadata` sidecars via
  `pyarrow.parquet.write_metadata`.
- [ ] Use sidecars in dataset scanners where available.

File targets
- `src/codeintel/build/exports/writers.py`
- `src/codeintel/storage/datasets/manifest_index.py`
- `src/codeintel/storage/datasets/arrow_store.py`

Acceptance
- Dataset scans reuse metadata when available.

---

## Phase 7: Storage internal Arrow upgrades (boundary-safe)
Purpose: improve storage performance without crossing build/storage boundary.

Checklist
- [ ] Keep storage APIs unchanged, but use Arrow readers internally where possible.
- [ ] Replace Polars in `src/codeintel/storage/repositories/datasets.py` with Arrow
  tables/readers (optional).
- [ ] Use Arrow compute for validation checks in storage (optional).

File targets
- `src/codeintel/storage/repositories/datasets.py`
- `src/codeintel/storage/validation/columnar.py`
- `src/codeintel/storage/warehouse.py`

Acceptance
- Storage APIs unchanged; Parquet/Arrow boundary preserved.

---

## Phase 8: Enforcement
Purpose: prevent regression back to Polars in compute paths.

Checklist
- [ ] Extend the Arrow-first guard to cover ingestion/analytics if desired.
- [ ] Add a lightweight CI check (ripgrep or grep) for disallowed Polars usage.

File targets
- `scripts/ci/arrow_first_guard.sh`
- `.github/workflows/arrow-first-guard.yml`

Acceptance
- CI fails on Polars usage in compute modules.

---

## Validation plan
- Run targeted Hamilton outputs for updated modules.
- Spot-check a small repo to validate unchanged row counts.
- Optional: add a small Arrow-only benchmark for scan + join paths.
