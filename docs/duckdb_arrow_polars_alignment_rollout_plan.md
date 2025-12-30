# DuckDB/Arrow/Polars Advanced Features Rollout Plan

## Intent

Provide a prioritized, risk-aware rollout plan for advanced Arrow/Polars features
with explicit decision points and "only if" gates for the risky items.

## Scope

- Dataset scanning with fragment/row-group pruning.
- IPC read/write options control plane.
- Vectorized validation via Arrow compute.
- Polars execution control plane (optimization visibility, profiling).
- Streaming-first writes with tuning knobs.
- Selectors DSL for schema-driven transforms.
- Sortedness metadata fast paths.
- Zero-copy interop channels.
- Dictionary encoding + unify_dictionaries.
- Observability and test hardening.

## Guiding Principles

- Preserve correctness before performance.
- Default to streaming; collect only with explicit intent.
- Keep control-plane knobs behind safe defaults.
- Gate risky changes behind measurable wins and clear rollbacks.

## Prioritized Phases

### Phase -1 (P-1): Arrow Contract Foundation (Required)

Objective: establish the Arrow schema contract as the runtime authority before
enabling streaming, IPC, or validation changes.

Scope:
- Canonical Arrow contract generation with metadata keys.
- Persist contract IPC bytes in renderer_cache for every table/view.
- Runtime schema loading prefers contract over DuckDB metadata.
- Contract-aligned ingest alignment with extras policy.

Files:
- `src/codeintel/core/schemas/arrow_gen.py`
- `src/codeintel/core/schemas/arrow_polars.py`
- `src/codeintel/storage/tracking/schema_catalog_compile.py`
- `src/codeintel/storage/tracking/schema_catalog.py`
- `src/codeintel/storage/schema/arrow_schema.py`
- `src/codeintel/core/columnar/schema_alignment.py` (new)
- `src/codeintel/build/hamilton/native/ingestion/frame_utils.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
- `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`
- `src/codeintel/core/exports/arrow_ipc.py`

Exit criteria:
- All tables and views have renderer_cache Arrow schema IPC bytes.
- IPC responses emit the contract schema with contract metadata intact.
- External ingest aligns to contract and retains extras in `_ci_extras`.

Decision point:
- Proceed only if contract round-trip is verified for at least one serving
  dataset and one ingestion flow.

Only-if gates:
- Only enable any streaming/IPC/validation features if contract is present
  in renderer_cache and schema hash matches.


### Phase 0 (P0): Baseline, Metrics, and Guardrails

Objective: establish consistent metrics and test scaffolding so the next phases
can be measured and reverted safely.

Scope:
- Add timing/memory counters around dataset scans and exports.
- Add minimal plan-level assertions for serving/analytics tests (stable invariants).

Files:
- `src/codeintel/serving/semantic/kernel.py`
- `src/codeintel/serving/semantic/engines/polars_engine.py`
- `tests/serving/semantic/*`
- `tests/analytics/*`

Exit criteria:
- Baseline latency/memory benchmarks captured for current flows.
- Test harness updated with stable invariants (row count, schema, basic plan markers).

Decision point:
- Proceed only if baseline metrics are captured for at least one representative
  dataset and serving query flow.


### Phase 1 (P1): Streaming-First Reads and Writes (High Value)

Objective: move IO to streaming-first with minimal behavior change.

Scope:
- Dataset scanning with fragment/row-group pruning via `ds.Dataset.scanner()` and
  `Dataset.get_fragments(filter=...)` with `Scanner.to_reader()`.
- Streaming-first writes via `LazyFrame.sink_parquet` where possible and
  fallback to `collect_batches + ds.write_dataset` only when required.

Files:
- `src/codeintel/storage/datasets/arrow_store.py`
- `src/codeintel/serving/semantic/engines/polars_engine.py`
- `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`

Exit criteria:
- Read path supports filter pushdown on a sample dataset with measurable
  reduction in scanned rows.
- Write path uses streaming sink for non-partitioned writes and falls back
  correctly for partitioned datasets.
- Arrow schema emitted on writes matches the contract schema from renderer_cache.

Decision point:
- Proceed only if scan metrics show actual pruning (row group or fragment) and
  output parity matches existing datasets.

Only-if gates:
- Only enable fragment pruning if partition metadata is available via Iceberg
  metadata or per-dataset export manifests, and filter translation is validated
  by tests.
- Only enable `sink_parquet` if row-group sizing and ordering controls are
  compatible with existing downstream readers.
- Only enable streaming reads/writes if contract schema load succeeds.


### Phase 2 (P2): IPC Control Plane (Medium Risk)

Objective: expose IPC knobs without breaking defaults.

Scope:
- Expose `IpcWriteOptions` and `IpcReadOptions` in streaming helpers.
- Add schema metadata injection and record-batch metadata policy.

Files:
- `src/codeintel/core/exports/arrow_ipc.py`
- `src/codeintel/serving/http/streaming.py`
- `src/codeintel/serving/semantic/kernel.py`

Exit criteria:
- Defaults remain unchanged for existing clients.
- IPC round-trip succeeds with new options enabled in a controlled test.
- IPC schema bytes match renderer_cache contract schema.

Decision point:
- Proceed only if client compatibility is verified for at least one Arrow IPC
  consumer (internal or external) and schema metadata appears intact.

Only-if gates:
- Only expose advanced options via config flags (no new required params).
- Only enable custom compression or recursion-depth changes if verified against
  large nested payloads.
- Only enable IPC options if contract schema is loaded from renderer_cache.


### Phase 3 (P3): Vectorized Validation (Medium Risk)

Objective: reduce per-row validation overhead while keeping parity for common
constraints.

Scope:
- Replace per-row JSON Schema validation with Arrow compute for nullability,
  type checks, ranges, and enum constraints.
- Keep a fallback to JSON Schema for complex or nested constraints.

Files:
- `src/codeintel/build/exports/validation.py`
- `src/codeintel/storage/validation/columnar.py`

Exit criteria:
- Validation time reduced on sample Parquet exports.
- Error messages remain actionable and stable.
- Contract nullability and type metadata are enforced in validation.

Decision point:
- Proceed only if parity is achieved for the top 80 percent of constraints in
  production schemas (nullability, types, ranges, enums).

Only-if gates:
- Only skip JSON Schema for a table when constraint coverage is proven by tests.
- Only enable contract-based validation if renderer_cache contract is present.


### Phase 4 (P4): Polars Execution Control Plane (Low Risk)

Objective: make optimizer behavior observable without changing results.

Scope:
- Surface QueryOptFlags, `profile`, and `inspect` in Polars plans.
- Add streaming fallback visibility and logging.

Files:
- `src/codeintel/serving/semantic/engines/polars_engine.py`
- `src/codeintel/serving/semantic/kernel.py`
- `src/codeintel/serving/semantic/polars_query_builder.py`

Exit criteria:
- Debug mode provides plan introspection without affecting outputs.
- Streaming fallback is visible in logs or metrics.

Decision point:
- Proceed only if plan visibility does not introduce nondeterministic behavior
  in tests.

Only-if gates:
- Only enable profiling/inspect under debug or explicit config flags.


### Phase 5 (P5): Schema-Driven Transforms and Sortedness (Medium Risk)

Objective: simplify wide-schema transforms and unlock fast paths with metadata.

Scope:
- Use `polars.selectors` for schema-driven column selection.
- Apply `set_sorted` when manifest exposes reliable sort keys.

Files:
- `src/codeintel/serving/semantic/polars_query_builder.py`
- `src/codeintel/build/hamilton/transforms/tabular_steps.py`
- `src/codeintel/serving/semantic/engines/polars_engine.py`

Exit criteria:
- Reduced transform boilerplate in at least one wide-schema flow.
- Join-asof or merge-sorted fast path can be triggered without extra sorts.

Decision point:
- Proceed only if manifests define sort keys with integrity and tests show
  correct ordering semantics.

Only-if gates:
- Only set sortedness when the dataset was written in that order and validated
  by a spot-check or manifest fingerprint.


### Phase 6 (P6): Dictionary Encoding and Unify Dictionaries (Risky)

Objective: improve memory and join performance for low-cardinality columns.

Scope:
- Apply dictionary encoding selectively to low-cardinality string columns.
- Call `Table.unify_dictionaries()` before joins/writes when beneficial.

Files:
- `src/codeintel/storage/datasets/arrow_store.py`
- `src/codeintel/serving/semantic/*`

Exit criteria:
- Reduced memory usage and equal-or-better join performance on representative
  workloads.

Decision point:
- Proceed only if profiling shows a win for real data distributions.

Only-if gates:
- Only enable dictionary encoding behind a config flag with a fallback to
  plain strings for high-cardinality columns.


### Phase 7 (P7): Zero-Copy Interop Channels (High Risk)

Objective: eliminate conversion overhead for ingestion/serving boundaries.

Scope:
- Accept `__dataframe__` and Arrow C Data Interface capsules in ingestion.
- Centralize adapters in columnar stream utilities.

Files:
- `src/codeintel/core/columnar/stream.py`
- `src/codeintel/storage/storage.py` (or equivalent ingestion boundary)

Exit criteria:
- End-to-end data flow with zero-copy inputs works without memory leaks.
- Interop inputs are aligned to contract schema via schema_alignment utilities.

Decision point:
- Proceed only if lifetime/ownership semantics are well-defined and tested.

Only-if gates:
- Only enable for known-good producers (feature flag per adapter).
- Only accept capsules when schema compatibility is explicitly validated.
- Only enable if contract-based alignment is in place for external ingest.


## Decision Matrix for Risky Items

- Vectorized validation: only when constraint coverage is proven for a table.
- Sortedness metadata: only when manifests assert and tests verify ordering.
- Dictionary encoding: only when profiling shows a benefit on real data.
- Zero-copy interop: only when ownership and lifecycle are proven in tests.
- IPC advanced options: only via opt-in config with compatibility tests.

## Rollback Strategy

- All risky items must be gated by a configuration flag.
- Each phase must preserve a fallback path to the prior implementation.
- Rollback criteria: any regression in correctness, latency, or memory beyond
  pre-defined thresholds.

## Metrics to Capture Per Phase

- Query latency (p50/p95) for serving queries.
- Peak memory during export and dataset writes.
- Rows scanned vs rows returned (pruning efficiency).
- Validation runtime and error parity.
- CPU time for joins and aggregations.

## Follow-up Implementation Checklist (Code Changes)

- Implement contract metadata injection in Arrow schema generation:
  `src/codeintel/core/schemas/arrow_gen.py`
- Add contract-aware Arrow schema reuse and validation:
  `src/codeintel/core/schemas/arrow_polars.py`
- Persist IPC schema bytes into renderer_cache for each table/view:
  `src/codeintel/storage/tracking/schema_catalog_compile.py`
- Load contract schema from renderer_cache at runtime:
  `src/codeintel/storage/schema/arrow_schema.py`
- Ensure schema_versions writes always include renderer_cache payload:
  `src/codeintel/storage/tracking/schema_catalog.py`
- Update IPC streaming to emit contract schema and metadata:
  `src/codeintel/core/exports/arrow_ipc.py`
- Add schema_alignment module with align_reader_to_contract:
  `src/codeintel/core/columnar/schema_alignment.py`
- Wire ingest/extract paths through contract alignment and extras policy:
  `src/codeintel/build/hamilton/native/ingestion/frame_utils.py`
  `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
  `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`
- Add contract round-trip tests and renderer_cache assertions:
  `tests/*` (arrow_gen/arrow_polars/schema_catalog/ingestion alignment)
