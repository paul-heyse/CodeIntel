# Inference-First Schema Implementation Plan (Hamilton + PyArrow)

## Summary

Deliver an inference-first schema system where observed schemas and statistics are the
authoritative contract for build, storage, and serving. Static schemas remain as hints
or fallbacks for non-inferable outputs and metadata tables. The plan integrates
advanced Hamilton DAG features and PyArrow interop to ensure robust, streaming-first
schema inference and alignment, while enforcing a clean serving boundary through
Parquet datasets imported into DuckDB.

## Objectives

- Make schema observations the primary source of truth for all inferable outputs.
- Ensure schema inference is streaming-first and does not require full materialization.
- Leverage Hamilton function modifiers to keep compute nodes inferable and modular.
- Standardize on PyArrow as the interchange layer for schema and data transport.
- Provide drift visibility, validation, and dataset tuning based on observed stats.
- Enforce a layered boundary: Hamilton + PyArrow write Parquet datasets; DuckDB reads only
  those datasets; SQLGlot serves based solely on DuckDB catalog/metadata.

## Scope

- Build-time inference, schema observation capture, and registry persistence.
- Runtime schema resolution, alignment, and validation based on observed schemas.
- Integration of Hamilton tagging and data quality hints as soft metadata.
- PyArrow datasets, schema metadata, and compute kernels for stats and alignment.
- Migration of static registries to "hint-only" for inferable outputs.

## Non-goals

- Schema pinning or enforced schema version gating.
- Sampling policies beyond streaming-first inference (optional config may follow).
- Blocking on drift or schema mismatch (drift is observable, not a gate).

## Guiding Principles

- Inference is authoritative for inferable outputs.
- Hints are additive only; they must not override observed types.
- Streaming-first everywhere (RecordBatchReader and scan_batches).
- Observability over enforcement; drift and mismatch are surfaced, not blocked.
- Minimize new config surface; derive defaults from observed stats.
- Parquet datasets are the single boundary between build outputs and serving inputs.

## Current State and Gaps

- Static TableSchema registries define many outputs that should be inferable.
- Inference seeds rely on empty frames derived from declared schemas.
- Validators and ingestion normalization use static schema sources by default.
- Hamilton tagging writes schema metadata based on static table schemas.
- Serving metadata is fixed JSON and does not reflect observed schema drift.
- View computation is centered in DuckDB SQL (SQLGlot builders + CREATE VIEW).
- Serving can still depend on registry/metadata beyond DuckDB, blurring the layer boundary.

## Target Architecture

### Build-time flow (authoritative)

Hamilton outputs -> RecordBatchReader -> SchemaObservation -> Registry
-> Derived TableSchema + Arrow schema + stats -> Parquet dataset (+ Arrow metadata)

### Runtime flow (authoritative)

Registry latest observation -> Alignment and validation -> Parquet dataset
-> DuckDB import -> SQLGlot planning + serving

### Schema authority chain

1) Observed schema (latest schema_observations for inferable outputs)
2) Override schema (non-inferable outputs)
3) Declared source schemas (external inputs, metadata tables)

### Layered serving boundary (new)

- Hamilton + PyArrow compute all build outputs and write Parquet datasets.
- Parquet datasets embed required metadata (table_key, schema_hash, column types,
  extras policy, stats) via Arrow schema metadata and dataset-level metadata.
- DuckDB imports only these Parquet datasets and becomes the sole schema source for
  SQLGlot planning and serving (information_schema + metadata tables).

### Parquet metadata checklist (authoritative boundary)

Required Arrow schema metadata keys (per dataset):
- codeintel.table_key (e.g., "analytics.function_metrics")
- codeintel.schema_hash (stable hash of inferred TableSchema)
- codeintel.schema_digest (fingerprint of schema JSON)
- codeintel.derivation_kind (e.g., "inferred_relation", "explicit_override")
- codeintel.derivation_source (target or source descriptor)
- codeintel.extras_policy (e.g., "drop", "retain", "error")
- codeintel.extras_column (default: "extras")
- codeintel.contract_version (arrow schema contract version)
- codeintel.observed_at (ISO timestamp of observation)

Required Arrow field metadata keys (per column):
- codeintel.column_type (normalized column type string)
- codeintel.description (optional)
- codeintel.nullable_observed (true/false)
- codeintel.pii_class (optional)
- codeintel.key_role (optional: primary_key, foreign_key)

Dataset-level metadata storage rules:
- Arrow schema metadata is embedded in Parquet file metadata (schema-level key/value pairs).
- Column metadata is embedded in Parquet field metadata.
- Observed stats are written to metadata tables in DuckDB on import:
  - metadata.schema_versions (schema JSON + renderer_cache/IPC)
  - metadata.schema_observations (stats + drift summary)
  - metadata.table_schema_registry (current pointer)
- DuckDB is the only serving-time schema source; no registry calls during query planning.

## Workstreams and Phases

### Phase 0: Inventory and readiness

Goals:
- Produce a canonical inventory of inferable outputs and static overrides.
- Baseline inference success and coverage metrics.

Tasks:
- Generate a table_key inventory with status: inferable, override, source-only.
- Map each output to its producing Hamilton node and IO boundary.
- Identify compute nodes that are non-inferable due to dependencies.

Deliverables:
- Inventory report with inferability status and owning target.
- Dependency graph of inference blockers.

Acceptance:
- All outputs are classified with an owner and current schema source.

---

### Phase 1: Observation emission from materializers

Goals:
- Emit SchemaObservation records for every inferable output.
- Ensure no eager materialization occurs for inference.

Tasks:
- Instrument Arrow dataset saver to emit observations during write.
- Instrument DuckDB relation saver to emit observations from fetch_arrow_reader().
- Persist observations in metadata.schema_observations.
- Attach derived stats and schema IPC bytes to schema_versions.
- Ensure Parquet datasets carry Arrow schema metadata required by DuckDB import.

Deliverables:
- Observation persistence hooks in materializers.
- End-to-end observation write for at least one target domain.

Acceptance:
- SchemaObservation records appear for both Arrow and DuckDB outputs.
- No to_table() or full materialization in inference paths.

---

### Phase 2: Streaming inference and hint merging

Goals:
- Inference service uses streaming batches and Arrow schema metadata.
- Hamilton hints are merged as soft metadata only.

Tasks:
- Extend inference service to accept RecordBatchReader and iterators of RecordBatch.
- Compute stats via pyarrow.compute: null counts, distinct counts, min/max, lengths.
- Derive TableSchema from observed Arrow schema with nested types.
- Merge Hamilton hints (nullable, description, PII, tags) without overriding types.
- Ensure inference can run without declared schema seeds.

Deliverables:
- Streaming inference interface with batch iterator support.
- Schema hints merge rules and tests.

Acceptance:
- Inference completes without collecting full tables.
- Output TableSchema reflects observed types plus hint metadata.

---

### Phase 3: Schema registry as inference-backed authority

Goals:
- Runtime resolution prefers inferred schema from observations.
- Static schemas become fallback only.

Tasks:
- Update SchemaService to load latest observation when present.
- Prefer renderer_cache IPC bytes from observations for Arrow schema.
- Maintain fallback for metadata tables and non-inferable outputs.

Deliverables:
- Registry resolution uses observed schema by default.
- Stable caching and refresh behavior for schema updates.

Acceptance:
- Latest observation is the schema returned at runtime for inferable outputs.
- Static schema definitions never override observations.

---

### Phase 4: Replace static seed harness for q__ inputs

Goals:
- Seed inference from observed datasets rather than declared schemas.

Tasks:
- Replace DatasetSeedHarness with a dataset scanner that returns empty or sampled
  RecordBatchReaders based on observed schema.
- Support q__ inputs via pyarrow.dataset scanners with projection-only reads.
- Allow optional "no-scan" mode to remain streaming-only without data reads.

Deliverables:
- New seed provider using observed schemas and Arrow datasets.
- Backward-compatible fallback for declared-only sources.

Acceptance:
- Inference can run without static declared schema seeds.
- q__ inputs resolve via observed schema or dataset scan.

---

### Phase 5: Ingestion and alignment upgrades

Goals:
- Align ingestion outputs to observed schema everywhere.
- Preserve extras policy and type promotion behavior in Arrow alignment.

Tasks:
- Align normalize_ingest_frame and frame_utils to use inferred schema.
- Ensure align_reader_to_contract uses Arrow schema metadata from observations.
- Encode extras policy and column_type metadata in Arrow field metadata.

Deliverables:
- Ingestion alignment using inferred schema and Arrow metadata.
- Consistent extras handling across ingestion, build, and serving.

Acceptance:
- Ingestion outputs match observed schema order and types.
- Extra columns handled according to policy and tracked in extras field.

---

### Phase 6: Hamilton DAG rework for inferability

Goals:
- Ensure compute nodes are inferable and IO is isolated.
- Use Hamilton function modifiers to preserve DAG structure and metadata.

Tasks:
- Move IO to @dataloader/@datasaver or @load_from/@save_to nodes.
- Keep compute nodes tabular (pl.LazyFrame, pa.Table, RecordBatchReader).
- Use @pipe_input/@pipe_output and @with_columns to express pipelines.
- Use @parameterize and @parameterized_subdag for repeated schemas.
- Use @resolve or @resolve_from_config for config-dependent DAG shaping.
- Apply @schema.output and @tag for soft schema hints only.

Checklist (remaining; aligned with schema_consolidation_implementation_plan.md):
- [ ] Run an inferability audit (per target) and list remaining non-inferable nodes.
- [ ] Refactor compute nodes that still return DuckDBRelation to return inferable tabular types
  (pl.LazyFrame, pa.Table, pa.RecordBatchReader) and move IO to loaders/savers, focusing on:
  `src/codeintel/build/hamilton/native/analytics/`,
  `src/codeintel/build/hamilton/native/graphs/`,
  `src/codeintel/build/hamilton/native/ingestion/`,
  `src/codeintel/build/hamilton/native/export/serving_artifacts.py`.
- [ ] Replace relation-first compute paths (relation_to_polars / DuckDBRelation inputs) with
  dataset-scanner based inputs once `src/codeintel/core/columnar/dataset_scanner.py` lands.
- [ ] Convert IO nodes to @dataloader/@datasaver or @load_from/@save_to; keep compute nodes pure
  and apply @pipe_input/@pipe_output or @with_columns for staged transforms.
- [ ] Tighten type hints so compute nodes use InferableTabularInput (remove DuckDBRelation from
  compute signatures; confine DuckDBRelation to IO adapters).
- [ ] Ensure all Arrow/Parquet writes use the consolidated observation pipeline + metadata codec
  from schema consolidation (no per-module schema metadata logic).
- [ ] Add/extend tests that enforce: inferable_table_keys covers all compute outputs, and no
  compute node depends on target nodes or data_saver nodes.

Deliverables:
- DAG patterns catalog for compute nodes and IO boundaries.
- Refactored targets to remove non-inferable dependencies.

Acceptance:
- inferable_table_keys covers all compute outputs with tabular returns.
- No compute node depends on target nodes or data_saver nodes.

---

### Phase 7: Serving and interoperability

Goals:
- DuckDB serves exclusively from Parquet datasets produced by Hamilton + PyArrow.
- SQLGlot planning and serving rely only on DuckDB catalog + metadata.

Tasks:
- Define a DuckDB import pipeline that registers only Parquet datasets.
- Load schema metadata into DuckDB (information_schema + metadata tables) from Parquet
  Arrow metadata and dataset-level metadata.
- Update semantic planners to resolve schema/columns from DuckDB only (no registry calls).
- Keep SQLGlot view definitions grounded in DuckDB tables, not external schema sources.

Deliverables:
- DuckDB import adapter for Parquet datasets with embedded metadata.
- Serving pipeline fully resolved from DuckDB catalog + metadata.

Acceptance:
- DuckDB only reads Parquet datasets produced by Hamilton.
- SQLGlot planning uses only DuckDB schema information; registry is not consulted at query time.

---

### Phase 8: View migration to Hamilton + PyArrow (DuckDB relegation)

Goals:
- Move view computation into the Hamilton DAG using Arrow-native pipelines.
- Relegate DuckDB to residual query serving and lightweight stitching only.
- Execute an aggressive migration with no feature-flagged dual path.

Tasks:
- Inventory SQLGlot views (by complexity: projection-only, simple joins, complex SQL).
- Create Hamilton view modules that emit tabular outputs (pa.Table or RecordBatchReader).
- Replace SQLGlot builders with DAG compute nodes that use:
  - pyarrow.dataset scans with projection/filter pushdown
  - Arrow joins/aggregations (pa.Table.join, pyarrow.acero where needed)
  - Hamilton @pipe_input/@pipe_output for staged transforms
- Materialize view outputs via @save_to.parquet/@datasaver (not CREATE VIEW).
- Update view discovery to read Hamilton tags from DAG nodes (output_kind=view/semantic_view).
- Remove DuckDB view materialization from build-time execution.
- Update schema inference for views to rely on observed outputs (not SQL parsing).
- Preserve lineage by emitting column lineage metadata from DAG edges where possible.
- Constrain SQLGlot view definitions to DuckDB-only sources:
  - Views may reference only DuckDB tables and metadata imported from Parquet datasets.
  - No view definition may consult registry schemas or filesystem scans at serve time.

Checklist (remaining; aligned with schema_consolidation_implementation_plan.md):
- [ ] Route view input loading through the shared dataset scanner utilities (once available) to
  enforce streaming-first reads in `src/codeintel/build/hamilton/native/views/view_outputs.py`.
- [ ] Ensure view dataset writes flow through the unified observation pipeline and metadata
  codec so `codeintel.*` metadata keys are consistent with the Parquet contract.
- [ ] Replace SQLGlot-derived lineage sync with DAG-derived lineage emission for view outputs;
  populate `metadata.derived_lineage_edges` and `metadata.derived_lineage_columns` from DAG edges.
- [ ] Retire or quarantine build-time SQLGlot schema inference (`src/codeintel/storage/views/schema_inference.py`)
  and view materialization orchestration (`src/codeintel/storage/views/materialization.py`).
- [ ] Verify view outputs are present in `metadata.schema_observations` and
  `metadata.table_schema_registry` after build runs.
- [ ] Keep SQLGlot view builders for serving only; validate all view SQL resolves against
  DuckDB-only sources imported from Parquet datasets.

Deliverables:
- Hamilton-based view output modules with Arrow-first execution.
- DuckDB view materialization removed from build runs.
- Observed schemas for view outputs registered in schema_observations.
- SQLGlot view catalog validated against DuckDB-only sources.

Acceptance:
- View outputs are produced by Hamilton DAG nodes, not DuckDB SQL views.
- DuckDB is only used for query serving or minor stitching.
- View schemas are inferred from observed Arrow outputs.
- SQLGlot view definitions resolve exclusively against DuckDB catalog/metadata.

#### View Migration Checklist (by tier)

Tier definitions (initial heuristics from SQL AST metrics):
- T0: projection-only or simple filters (no joins, no aggregates, no windows).
- T1: single-join or lightweight transforms (<=1 join, no windows/unions).
- T2: multi-step joins and/or aggregation (<=2 joins, group/agg present).
- T3: complex joins, windows, or unions (joins >2 or windows/unions present).

Ownership assignment is domain-based and should be confirmed:
- analytics.* -> Analytics
- core.* -> Core/Ingestion
- docs.* -> Docs/Serving
- graph.* -> Graph Analytics

| View key | Tier | Owner | Status |
| --- | --- | --- | --- |
| analytics.v_function_hotspots | T3 | Analytics | [ ] |
| analytics.v_function_summary | T1 | Analytics | [ ] |
| core.v_goid_crosswalk_join | T1 | Core/Ingestion | [ ] |
| core.v_goid_crosswalk_mismatches | T1 | Core/Ingestion | [ ] |
| docs.v_behavioral_classification_input | T1 | Docs/Serving | [ ] |
| docs.v_call_graph_enriched | T3 | Docs/Serving | [ ] |
| docs.v_cfg_block_architecture | T3 | Docs/Serving | [ ] |
| docs.v_config_data_flow | T1 | Docs/Serving | [ ] |
| docs.v_data_model_fields | T0 | Docs/Serving | [ ] |
| docs.v_data_model_relationships | T0 | Docs/Serving | [ ] |
| docs.v_data_model_usage | T2 | Docs/Serving | [ ] |
| docs.v_data_models | T0 | Docs/Serving | [ ] |
| docs.v_data_models_normalized | T0 | Docs/Serving | [ ] |
| docs.v_dfg_block_architecture | T3 | Docs/Serving | [ ] |
| docs.v_entrypoints | T0 | Docs/Serving | [ ] |
| docs.v_external_dependencies | T0 | Docs/Serving | [ ] |
| docs.v_external_dependency_calls | T0 | Docs/Serving | [ ] |
| docs.v_file_summary | T3 | Docs/Serving | [ ] |
| docs.v_function_architecture | T3 | Docs/Serving | [ ] |
| docs.v_function_summary | T2 | Docs/Serving | [ ] |
| docs.v_ide_hints | T3 | Docs/Serving | [ ] |
| docs.v_module_architecture | T3 | Docs/Serving | [ ] |
| docs.v_module_architecture_full | T3 | Docs/Serving | [ ] |
| docs.v_module_with_subsystem | T3 | Docs/Serving | [ ] |
| docs.v_subsystem_agreement | T0 | Docs/Serving | [ ] |
| docs.v_subsystem_coverage | T2 | Docs/Serving | [ ] |
| docs.v_subsystem_profile | T2 | Docs/Serving | [ ] |
| docs.v_subsystem_summary | T2 | Docs/Serving | [ ] |
| docs.v_symbol_module_graph | T0 | Docs/Serving | [ ] |
| docs.v_test_architecture | T1 | Docs/Serving | [ ] |
| docs.v_test_to_function | T3 | Docs/Serving | [ ] |
| docs.v_validation_summary | T3 | Docs/Serving | [ ] |
| graph.v_call_graph_degree | T2 | Graph Analytics | [ ] |
| graph.v_import_graph_degree | T2 | Graph Analytics | [ ] |

#### Canonical Hamilton View Module Layout

Module layout (domain-first, DAG-owned views):

```
src/codeintel/build/hamilton/native/views/
  __init__.py
  analytics/
    __init__.py
    function_summary.py
    function_hotspots.py
  docs/
    __init__.py
    module_architecture.py
    subsystem_summary.py
  graph/
    __init__.py
    call_graph_degree.py
```

Guidelines:
- Each view node returns pa.Table or pa.RecordBatchReader.
- Use @dataloader or @load_from.* to load Arrow datasets.
- Tag view outputs with output_kind=view or semantic_view + table_key.
- Persist view outputs with @save_to.parquet/@datasaver (no CREATE VIEW).

Example (join + aggregate using Arrow tables):

```python
from __future__ import annotations

import pyarrow as pa
import pyarrow.dataset as ds

from hamilton.function_modifiers import dataloader

from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.tagging_helpers import apply_raw_tags


@dataloader()
def analytics_function_metrics_dataset(path: str) -> tuple[ds.Dataset, dict[str, object]]:
    dataset = ds.dataset(path, format="parquet")
    return dataset, {"path": path}


def analytics_function_metrics(
    analytics_function_metrics_dataset: ds.Dataset,
) -> pa.Table:
    return analytics_function_metrics_dataset.to_table(
        columns=["function_goid_h128", "risk_score", "cyclomatic_complexity"]
    )


def docs_v_function_summary(
    analytics_function_metrics: pa.Table,
) -> pa.Table:
    return analytics_function_metrics.group_by("function_goid_h128").aggregate(
        [("risk_score", "mean"), ("cyclomatic_complexity", "max")]
    )


docs_v_function_summary = apply_raw_tags(
    docs_v_function_summary,
    tags={
        ht.TAG_OUTPUT_KIND: ht.OUTPUT_KIND_VIEW,
        ht.TAG_TABLE_KEY: "docs.v_function_summary",
    },
)
```

Example (multi-source join with Arrow tables):

```python
from __future__ import annotations

import pyarrow as pa

from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.tagging_helpers import apply_raw_tags


def docs_v_module_architecture(
    module_profile: pa.Table,
    subsystem_profile: pa.Table,
) -> pa.Table:
    joined = module_profile.join(
        subsystem_profile,
        keys=["repo", "commit", "subsystem_id"],
        join_type="left",
    )
    return joined


docs_v_module_architecture = apply_raw_tags(
    docs_v_module_architecture,
    tags={
        ht.TAG_OUTPUT_KIND: ht.OUTPUT_KIND_VIEW,
        ht.TAG_TABLE_KEY: "docs.v_module_architecture",
    },
)
```

---

### Phase 9: Validation and contracts

Goals:
- Validation derives from observed schema and stats.
- JSON Schema generation uses inferred TableSchema.

Tasks:
- Update data quality validators to use inferred schema by default.
- Derive nullability and type constraints from observation stats.
- Keep JSON Schema as a serialized view of the inferred TableSchema.

Checklist (remaining; aligned with schema_consolidation_implementation_plan.md):
- [ ] Implement the canonical resolver (`src/codeintel/core/schemas/resolution.py`) and migrate
  validation call sites to it (`src/codeintel/build/exports/validation.py`,
  `src/codeintel/storage/validation/columnar.py`, `src/codeintel/storage/schema/json_schema.py`).
- [ ] Introduce the unified validation engine (`src/codeintel/core/validation/schema_constraints.py`)
  and remove duplicate constraint derivation logic from legacy modules.
- [ ] Derive nullability/type constraints directly from observation stats; respect extras policy
  and column metadata from the SchemaMetadataCodec.
- [ ] Ensure JSON Schema generation uses inferred TableSchema (observed-first), via
  `src/codeintel/build/schemas/json_schema_registry.py` + `src/codeintel/core/schemas/service.py`.
- [ ] Add unit/integration tests to verify observed-first validation and JSON Schema parity.

Deliverables:
- Validation logic based on observations with fallback to static.
- Updated JSON Schema generation path.

Acceptance:
- Validation reports match observed schema, not static declarations.
- Drift is reported rather than blocked.

---

### Phase 10: Dataset tuning from observed stats

Goals:
- Use observed stats to tune Arrow and Parquet output settings.

Tasks:
- Derive dictionary encoding candidates from distinct counts.
- Set row group and data page sizes based on observed row counts and size.
- Persist derived settings in dataset manifests for future runs.

Checklist (remaining; aligned with schema_consolidation_implementation_plan.md):
- [ ] Extend the observation pipeline to compute stats needed for tuning (distinct counts,
  row counts, byte sizes) and persist them with observations.
- [ ] Add tuning logic in the Arrow dataset saver (`src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`)
  to set dictionary encoding, row group size, and data page sizing from observed stats.
- [ ] Persist tuning parameters in dataset manifests (`src/codeintel/core/manifests.py`) and
  ensure they round-trip during serving imports.
- [ ] Add tests that assert tuning metadata is written and applied for representative datasets.

Deliverables:
- Dataset tuning logic using observation stats.
- Stored tuning metadata in manifests.

Acceptance:
- Output datasets use inferred tuning parameters.
- Serving benefits from improved scan performance.

---

### Phase 11: Drift observability and reporting

Goals:
- Provide structured drift reporting for operators and developers.

Tasks:
- Compute drift summaries (missing/extra fields, type changes).
- Emit drift logs and metrics for each build run.
- Provide CLI and report views for drift history.

Checklist (remaining; aligned with schema_consolidation_implementation_plan.md):
- [ ] Implement drift diffing against the latest observation and persist summaries alongside
  observations (or in a dedicated metadata table).
- [ ] Emit drift logs/metrics during build materialization and include summaries in run records.
- [ ] Add CLI support for drift inspection (command + handler + result type) and a report view
  for drift history.
- [ ] Add tests that validate drift summaries for add/remove/type-change scenarios.

Deliverables:
- Drift summaries in metadata and build logs.
- CLI command for drift inspection.

Acceptance:
- Drift is visible without blocking execution.

---

### Phase 12: Static registry migration

Goals:
- Minimize static schemas for inferable outputs.
- Retain static schemas for metadata tables and non-inferable sources.

Tasks:
- Move inferable output schemas out of OUTPUT_TABLE_SCHEMAS.
- Keep only non-inferable outputs in override registry.
- Keep metadata and source-only schemas in TABLE_SCHEMAS.

Checklist (remaining; aligned with schema_consolidation_implementation_plan.md):
- [ ] Use the inferability audit to identify inferable outputs and remove them from
  `src/codeintel/core/schemas/output_registry.py`.
- [ ] Keep only non-inferable overrides in OUTPUT_TABLE_SCHEMAS and ensure metadata/source-only
  schemas remain in `src/codeintel/core/schemas/table_registry.py`.
- [ ] Update resolver/service paths to treat overrides as fallback only; remove direct registry
  access from consumers (per schema consolidation).
- [ ] Delete legacy helper paths called out in `docs/schema_consolidation_implementation_plan.md`
  once call sites have moved to the canonical modules.
- [ ] Add tests that assert inferable outputs resolve from observations, not static registries.

Deliverables:
- Reduced static registry footprint.
- Clear separation between inferable outputs and static sources.

Acceptance:
- Inferable outputs resolve solely from observations.
- Static schemas remain for metadata tables and true non-inferable outputs.

## Hamilton Design Patterns (Enforcement)

- IO isolation: @dataloader/@datasaver, @load_from/@save_to.
- Pipeline nodes: @pipe_input/@pipe_output for staged transforms.
- Column-level DAGs: @with_columns to express feature ops inside frames.
- Dynamic DAGs: @parameterize, @parameterized_subdag for multi-source graphs.
- Config-driven DAGs: @resolve or @resolve_from_config for runtime shape.
- Metadata hints: @tag, @schema.output, @tag_output (soft hints only).
- Validation hooks: @check_output and custom validators (warn or fail).

## PyArrow Interoperability Strategy

- Use pa.RecordBatchReader as the canonical interchange object.
- Prefer pyarrow.dataset for scan and write with projection/filter pushdown.
- Store column metadata (PII, nullability, extras policy) in Arrow schema.
- Use pa.unify_schemas for type promotion and drift handling.
- Use pyarrow.compute for stats; avoid full materialization.
- Use Arrow IPC for tool boundaries and internal streaming.

## Testing and Validation Plan

- Unit tests:
  - schema observation capture and persistence
  - streaming inference from RecordBatchReader
  - hint merging behavior (no type override)
  - Arrow schema metadata preservation
- Integration tests:
  - end-to-end inference for a representative target
  - serving query alignment with inferred schema
  - ingestion alignment for extra columns and type promotion
- Performance tests:
  - inference overhead for large outputs
  - dataset scan performance with projection and filters

## Risks and Mitigations

- Inference gaps due to non-tabular outputs:
  - Mitigation: standardize tabular returns and isolate IO.
- Schema drift noise:
  - Mitigation: normalization of types and robust diff logic.
- Backward compatibility with static schemas:
  - Mitigation: staged migration with fallbacks and feature flags.

## Rollout Strategy

1) Enable observations for a single domain (analytics) and validate.
2) Migrate a single docs view to Hamilton + Arrow and validate parity.
3) Migrate remaining views in ranked complexity order; remove DuckDB view materialization.
4) Expand to ingestion outputs and serving artifacts.
5) Switch SchemaService to prefer observations.
6) Remove static overrides for inferable outputs.
7) Turn on drift reporting across all domains.

## Open Questions

- Should we allow optional sampling for inference in very large outputs?
- Which outputs must remain static due to deterministic, non-tabular sources?
- What is the canonical policy for nullability when observed nulls appear?
