# Contract System (Build-Time) — End-to-End Design

This document explains how the CodeIntel **build-time contract system** works end-to-end: how
contracts are resolved via lazy references, how contract alignment is applied across Arrow/Polars
backends, how materialization records store contract metadata and diagnostics, and how optional
loader-side migrations keep older snapshots readable.

## Goals

- **Single source of truth** for table shape and semantics, derived from schema definitions.
- **Lazy, import-safe resolution**: targets declare contract refs that resolve only after schema
  services are configured.
- **Target-aware alignment**: the same `table_key` can be emitted by different targets without
  collisions in alignment behavior.
- **High-performance alignment**: reader-first schema alignment with caching and fast paths.
- **Strong observability**: surface alignment diffs in a structured way and persist them as build
  diagnostics and run-level summaries.
- **Evolution without fragility**: contract versions/hashes attached to outputs + optional loader
  migrations to support older snapshots.
- **Ergonomic authoring**: minimize boilerplate via contract refs and table-target context objects.

## Core Types and Concepts

### `table_key`

A fully-qualified table identifier: `"domain.table_name"` (e.g. `"analytics.graph_metrics_functions"`).

### Dataset schema vs build-time contract

- **Schema**: the canonical model of the dataset (fields, types, metadata, validation defaults).
  Schema is owned by the schema service and used to generate Arrow schemas.
- **Build-time contract spec**: the *target-specific* contract policy and wiring used by Hamilton
  build nodes and materializers, resolved at runtime from `ContractRef`.

### `ContractRef`

Lazy contract handle created at import time (safe before SchemaService exists). It captures:

- `table_key`
- `target_name`
- `input_name`
- optional overrides (policy, required cols, ops module, etc.)

### `ContractRuntime`

Runtime resolver configured after the schema service is available:

- resolves `ContractRef` to a concrete `TableContractSpec`
- applies policy defaults from `ContractPolicyRegistry`
- caches resolutions by `(table_key, target_name, input_name, overrides)`

Runtime wiring happens during composition in `src/codeintel/runtime/compose.py`.

### `ContractPolicy`

The policy surface for alignment/validation behavior.

Relevant fields (see `src/codeintel/build/contracts/types.py`):

- `extras_policy`: how to handle columns not present in the contract schema.
- `validation_profile`: optional validation profile.
- `coerce_types`: whether type coercion is allowed during alignment.
- `allow_nulls`: validation policy hook (used by validation stages).

Defaults are resolved via `ContractPolicyRegistry`, which is configured from
`config/codeintel.build.toml` (`[contracts]` section) during runtime composition.

### `ContractPolicyRegistry`

The policy registry centralizes defaults so targets do not hard-code policy choices. Configuration
keys under `[contracts]` include:

- `policy_profiles`: named policy definitions
- `policy_tables`: table key → profile mapping
- `policy_targets`: target name → profile mapping
- `default_profile`: fallback profile name

### `TableContractSpec`

Build-time contract spec used by targets.

Relevant fields:

- Identity: `table_key`, `domain`, `target`
- Wiring: `input_name` (Hamilton base node name), `ops_module` (optional transforms module),
  `columns_to_pass` (step configuration)
- Policy: `policy`, plus operational knobs `required_cols` and `clip_column`
- Versioning: `contract_version` and `contract_hash`

### Target-aware contracts (`target_name`)

The contract system keys contract resolution and alignment by:

- `table_key`
- `target_name` (the target producing the output)

This preserves output-specific shaping when a `table_key` is reused across multiple targets.

## Resolution: How a `TableContractSpec` Is Produced

### Contract runtime + registry

Targets declare **lazy contract refs** and resolve them at runtime:

1. Targets create a `ContractRef` via `contract_ref_for_table(...)`.
2. `configure_contract_runtime(...)` sets up a `ContractRuntime` with the schema service and policy
   registry during composition (`src/codeintel/runtime/compose.py`).
3. When a table target spec is built/attached, the runtime resolves the ref into a concrete
   `TableContractSpec`.

The resolver derives the contract spec from the canonical `TableSchema` and applies overrides. It
also attaches **identity metadata**:

- `contract_version`: derived from the dataset contract (if present).
- `contract_hash`: a stable hash of the `TableSchema` (the schema “fingerprint”).

### Canonical helper (`ContractRef`)

```python
from codeintel.build.contracts.ref import contract_ref_for_table

CONTRACT = contract_ref_for_table(
    table_key="analytics.graph_metrics_functions",
    target_name="graph_metrics",
    input_name="graph_metrics_functions__base",
    required_cols=(),
    clip_column=None,
)
```

For tests or specialized tooling you can still resolve contracts directly via the registry
(`contract_for_table`, `require_contract_for_target`), but Hamilton targets should use refs to avoid
import-time schema resolution.

## Contract Attachment: From Contract Specs to Hamilton Targets

### Table target specs

Hamilton targets are described by table-target specs and contexts:

- `src/codeintel/build/hamilton/native/patterns/table_target.py`

Key concepts:

- `TableTargetContext` / `TableTargetTableContext`: context objects used to build consistent
  `TableTargetSpec` and table specs.
- `build_single_table_target_spec(...)` / `build_multi_table_target_spec(...)`: build specs used to
  attach target templates.
- `TableTargetContext.from_contract_ref(...)` and
  `TableTargetTableContext.from_contract_ref(...)`: context constructors that ensure the **contract
  ref drives base node wiring** (the ref’s `input_name` becomes the single source of truth for the
  base node name). The ref is resolved to a `TableContractSpec` at attach time via
  `ContractRuntime`.

### Guardrail: contract provenance

Targets are only allowed to attach contracts that carry contract identity metadata. A guardrail in
`src/codeintel/build/hamilton/native/patterns/table_target.py` requires `contract_hash` whenever a
contract is attached to a table spec.

This ensures contract specs are built through the registry/factory, not hand-rolled.

## Contract Pipeline: How Contracts Shape Nodes and Outputs

Contracts are not just schemas; they also drive a **canonical, config-driven transform pipeline**
applied at table boundaries.

The pipeline is assembled in:

- `src/codeintel/build/hamilton/transforms/table_contract.py`

and applied automatically when a table target has a contract:

- `src/codeintel/build/hamilton/native/patterns/table_target.py` (via `table_contract(...)`)

### Pipeline stages (in order)

Given a `TableContractSpec`, the pipeline wires these stages:

1) **Input cleaning** (`pipe_clean_df`)
   - Uses `required_cols` and `clip_column`.
   - Targets the base input name via `input_name` (this is why
     `TableTargetContext.from_contract_ref(...)` is the canonical wiring pattern).
2) **Feature injection** (`with_features`, optional)
   - Enabled when `ops_module` is present.
   - Uses `columns_to_pass` and config-provided feature selections.
3) **Contract alignment** (`pipe_contract_alignment`)
   - Uses `table_key`, `target` (as `target_name`), and `policy`.
4) **Canonical output ordering** (`pipe_canonical_output`)
   - Uses `column_order_for_table_key(table_key)` when available.

The decorator factories live in:

- `src/codeintel/build/hamilton/transforms/decorators.py`

### Configuration surface (high level)

These stages are intentionally config-driven (via Hamilton `resolve_from_config`):

- Cleaning: `df_backend`, `clean_mode`, `null_policy`, `max_loc_clip`
- Alignment: `enable_contract_alignment`
- Canonicalization: `enable_canonicalization`
- Feature ops: `feature_sets` (table-key keyed selection)

This keeps the contract pipeline consistent across targets while allowing runtime tuning.

## Alignment: Enforcing Contract Shape on Outputs

### Where alignment lives

All contract alignment is implemented in Arrow-first helpers:

- `src/codeintel/build/tabular/arrow_ops.py`

There are two primary building blocks:

- `align_reader_to_contract(...)`: streaming alignment (preferred).
- `align_table_to_contract(...)`: table alignment (materializes batches as needed).

Both are target-aware and accept alignment options:

- `target_name`: used for contract resolution/policy decisions.
- `policy` / `extras_policy`: override policy knobs.
- `reporter`: an optional callback to record alignment diagnostics.

#### Policy resolution details

Alignment resolves policy in the following order:

1) If a `policy` override is provided, it is used directly.
2) Else, if `target_name` is provided, alignment resolves `TableContractSpec.policy` via
   `require_contract_for_target(table_key=..., target_name=...)`, which applies policy defaults
   from `ContractPolicyRegistry`.
3) Else, it falls back to a default `ContractPolicy()`.

If `extras_policy` is provided, it must be compatible with the resolved policy:

- `extras_policy=None` means “use the policy’s extras policy”.
- Otherwise, the override must match the policy’s `extras_policy` (or the policy must have
  `extras_policy=None`), or alignment raises to prevent silent drift.

#### Extras column semantics

Some tables store unknown fields in an “extras” column. Alignment report generation uses schema
metadata (`codeintel.extras_column`) to identify this column and avoid reporting it as an “extra”
column when present.

### Fast paths

`align_table_to_contract(...)` has fast paths for schema equality:

- If the incoming schema equals the contract schema (including metadata), return the table.
- If the only difference is metadata, replace metadata and return the table.

This avoids re-batching and re-materialization when alignment is a no-op.

### Cached alignment plan

Alignment uses a cached plan (`ContractAlignmentPlan`) keyed by:

- `table_key`
- `target_name`
- `extras_policy` (derived from the resolved `ContractPolicy`)

This keeps alignment overhead small, especially in large DAGs with repeated alignment calls.

## Cross-Tabular Alignment (Arrow + Polars)

Some pipeline steps and nodes operate on Polars frames while others operate on Arrow tables/readers.
To keep contract alignment consistent across backends, the system provides:

- `align_tabular_to_contract(...)` in `src/codeintel/build/tabular/arrow_ops.py`

This helper:

1. Detects whether the input is Arrow or Polars.
2. Normalizes to Arrow for alignment.
3. Converts back to preserve the input’s tabular type when possible.

Canonical usage from pipeline steps:

```python
from codeintel.build.tabular.arrow_ops import align_tabular_to_contract, emit_alignment_report

aligned = align_tabular_to_contract(
    "analytics.subsystem_profile_cache",
    df,
    target_name="subsystem_caches",
    reporter=emit_alignment_report,
)
```

The Hamilton step utilities route alignment through this helper:

- `src/codeintel/build/hamilton/transforms/tabular_steps.py`

## Diagnostics: Alignment Reports and Persistence

### Alignment report structure

Alignment can emit a structured report:

- `AlignmentReport` in `src/codeintel/build/tabular/arrow_ops.py`

It captures:

- missing columns
- extra columns
- coerced columns
- row count (when available)
- `table_key` and `target_name`

Implementation details worth knowing:

- Schema types are normalized for reporting so Arrow “view” types (e.g. string/binary view) do not
  cause noisy coerced-column diffs.
- Alignment reports exclude the configured extras column from the “extra columns” list.

### Reporter plumbing

Alignment helpers accept a `reporter` callback, and the build system provides a default reporter:

- `emit_alignment_report` in `src/codeintel/build/tabular/arrow_ops.py`

Call sites can use:

- `reporter=emit_alignment_report` to record a report into the per-run tracking surface.

Under the hood, `emit_alignment_report(...)`:

- Logs at most once per `(table_key, target_name)` to reduce log noise.
- Stores the last report in an in-process map so materialization records can attach it later.

Alignment diagnostics intended for persistence are captured separately:

- `record_alignment_diagnostic(...)` stores the latest diagnostic for a table target.
- `drain_alignment_diagnostics(...)` returns and clears pending diagnostics for persistence.

### Persistent diagnostics dataset

Alignment diagnostics are persisted as a build dataset at run finalization:

- `build.contract_alignment_issues` (per target/table)
- written via `persist_contract_alignment_issues` in
  `src/codeintel/build/hamilton/contract_alignment_issues.py`
- triggered from `src/codeintel/build/hamilton/executor.py`

Each row records run metadata, missing/extra/coerced counts, row count when available, and the
contract hash/version used for alignment.

### Run-level summary target

A lightweight analytics rollup aggregates per-run counts:

- `analytics.contract_alignment_summary`
- defined in `src/codeintel/build/hamilton/native/analytics/contract_alignment_summary.py`

### Persistence in materialization records

Materialization records attach contract metadata and alignment reports to dataset refs:

- `src/codeintel/build/hamilton/native/materialization_records.py`

This makes alignment diffs visible downstream (e.g., in manifests, metadata viewers, or diagnostics
tools).

Materialization records store contract information and alignment diffs under dataset ref metadata
keys:

- `contract_version`
- `contract_hash`
- `alignment_report` (a dict payload containing missing/extra/coerced columns and row count)
- `dataset_manifest_path` (when available)

## Output Metadata: Contract Identity on Materialized Datasets

### Dataset saver

When writing datasets, the Arrow dataset saver attaches contract identity metadata:

- `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`

The saver derives a `ContractDescriptor` for the output schema and stores:

- `contract_version`
- `contract_hash`

These values are written into:

- dataset manifest “extras”
- Parquet metadata (where supported)

This makes each snapshot self-describing and supports safe evolution.

In addition to metadata, the saver also performs a final **schema alignment** step for streamed
Arrow outputs (record batch readers) before materialization. This alignment:

- Uses the contract schema derived from the schema service.
- Applies `extras_policy` inferred from schema metadata.
- Uses runtime `schema_promote_options` to keep streaming alignment safe.
- Logs and records an event when alignment fails, but may continue materialization depending on
  failure handling paths.

### Tags on DAG nodes

Targets also attach contract identity as Hamilton tags (when available):

- tag keys are defined in `src/codeintel/core/hamilton/tags.py` and
  `src/codeintel/build/hamilton/tag_spec.py`
- table targets add contract tags via `src/codeintel/build/hamilton/native/patterns/table_target.py`

This helps introspection and target lineage/observability tooling.

## Loader-Side Migration: Keeping Old Snapshots Readable

### Motivation

When schemas evolve, older snapshots may have:

- missing columns
- renamed columns
- changed types or metadata

Rather than forcing every consumer to handle multiple historical shapes, loader-side migration
allows older datasets to be adapted to the current contract.

### Migration registry

Migrations are registered and applied by:

- `src/codeintel/build/contracts/migrations.py`

Conceptually:

- `register_contract_migration(...)` registers a function keyed by `table_key`.
- `apply_contract_migration(...)` applies the registered migration when versions differ.

Current behavior supports one migration per table key (no chained version routing yet). The
migration function receives `from_version` and `to_version` for context.

### Loader behavior

Loader nodes read the dataset manifest and compare it to the current contract version:

- `src/codeintel/build/hamilton/native/patterns/loaders.py`

Flow:

1. Load snapshot dataset and manifest.
2. Read `contract_version` from the manifest.
3. If it differs from the current contract version:
   - attempt to resolve a migration and apply it
   - warn when no migration exists
4. Validate and align the loaded table to the current schema contract.

This approach keeps “reader-first” alignment: migration happens before alignment/validation.

Note: loader alignment uses the lower-level alignment utilities in
`codeintel.core.columnar.schema_alignment` because loaders need to apply environment-specific
`schema_promote_options` while aligning streamed/loaded datasets.

## Pipeline Ordering: Where Contracts Apply

Contracts are enforced at well-defined points:

- In ingestion normalization steps (`src/codeintel/build/hamilton/transforms/ingestion_normalize.py`)
  before deduping/materialization.
- In target materialization templates (table targets) to ensure final outputs conform to the contract.
- In loader nodes for reading snapshot inputs consistently.

The general strategy is:

1) Normalize/coerce/align early for core inputs.
2) Compute (pure logic).
3) Align at output boundaries with consistent diagnostics.
4) Persist identity metadata to support future reads and migrations.

## Developer Workflow: Adding or Modifying a Contracted Table

1) Ensure the table exists in the schema service (TableSchema / dataset contract source).
2) In a target module, declare a contract ref via `contract_ref_for_table(...)`.
3) Use `TableTargetContext.from_contract_ref(...)` (single-table) or
   `TableTargetTableContext.from_contract_ref(...)` (multi-table) so `input_name` is the base-node
   source of truth.
4) If you need strict/lenient defaults, map the table/target in `config/codeintel.build.toml`
   under `[contracts]`.
5) Ensure output alignment happens through:
   - `align_table_to_contract(...)` / `align_reader_to_contract(...)` for Arrow, or
   - `align_tabular_to_contract(...)` if the node can return Polars.
6) If you change schema versions and need compatibility, add a migration in
   `src/codeintel/build/contracts/migrations.py`.

## File Index (Key Entry Points)

- Contract types: `src/codeintel/build/contracts/types.py`
- Contract refs + runtime: `src/codeintel/build/contracts/ref.py`,
  `src/codeintel/build/contracts/runtime.py`
- Contract registry (direct resolution): `src/codeintel/build/contracts/registry.py`
- Contract policy registry: `src/codeintel/build/contracts/policy_registry.py`
- Contract migrations: `src/codeintel/build/contracts/migrations.py`
- Alignment engine: `src/codeintel/build/tabular/arrow_ops.py`
- Graph assembly wrappers: `src/codeintel/build/graphs/assembly/contracts.py`
- Cross-tabular pipeline alignment: `src/codeintel/build/hamilton/transforms/tabular_steps.py`
- Contract pipeline decorator: `src/codeintel/build/hamilton/transforms/table_contract.py`
- Decorator factories: `src/codeintel/build/hamilton/transforms/decorators.py`
- Ingestion normalization alignment: `src/codeintel/build/hamilton/transforms/ingestion_normalize.py`
- Table target templates/contexts: `src/codeintel/build/hamilton/native/patterns/table_target.py`
- Loader migrations + validation: `src/codeintel/build/hamilton/native/patterns/loaders.py`
- Materialization metadata: `src/codeintel/build/hamilton/native/materialization_records.py`
- Alignment diagnostics persistence: `src/codeintel/build/hamilton/contract_alignment_issues.py`
- Alignment summary target:
  `src/codeintel/build/hamilton/native/analytics/contract_alignment_summary.py`

## Deprecations

- `tools/generate_accessor_inserts.py` and its generated registry
  (`src/codeintel/storage/gateway/registry_generated.py`) are decommissioned.
- Schema propagation relies on dynamic row bindings in
  `src/codeintel/core/data_models/rows.py` and contract-driven schemas via the registry.
