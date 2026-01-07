# Contract System (Build-Time) — End-to-End Design

This document explains how the CodeIntel **build-time contract system** works end-to-end: how
contracts are resolved, how contract alignment is applied across Arrow/Polars backends, how
materialization records store contract metadata and diagnostics, and how loader-side migrations keep
older snapshots readable.

## Goals

- **Single source of truth** for table shape and semantics, derived from schema definitions.
- **Target-aware alignment**: the same `table_key` can be emitted by different targets without
  collisions in alignment behavior.
- **High-performance alignment**: reader-first schema alignment with caching and fast paths.
- **Strong observability**: surface alignment diffs in a structured way and persist them with
  materialization metadata.
- **Evolution without fragility**: contract versions/hashes attached to outputs + optional loader
  migrations to support older snapshots.
- **Ergonomic authoring**: minimize boilerplate via contract factories and table-target context
  objects.

## Core Types and Concepts

### `table_key`

A fully-qualified table identifier: `"domain.table_name"` (e.g. `"analytics.graph_metrics_functions"`).

### Dataset schema vs build-time contract

- **Schema**: the canonical model of the dataset (fields, types, metadata, validation defaults).
  Schema is owned by the schema service and used to generate Arrow schemas.
- **Build-time contract spec**: the *target-specific* contract policy and wiring used by Hamilton
  build nodes and materializers.

### `ContractPolicy`

The policy surface for alignment/validation behavior.

Relevant fields (see `src/codeintel/build/contracts/types.py`):

- `extras_policy`: how to handle columns not present in the contract schema.
- `validation_profile`: optional validation profile.
- `coerce_types`: whether type coercion is allowed during alignment.
- `allow_nulls`: validation policy hook (used by validation stages).

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

### Registry

The build-time entrypoint is the contract registry:

- `src/codeintel/build/contracts/registry.py`

Conceptually:

1. The schema service provides a canonical `TableSchema`.
2. The registry derives a `TableContractSpec` from the schema defaults.
3. Target-specific override values (input name, optional policy tweaks) are applied.

The registry also attaches **identity metadata**:

- `contract_version`: derived from the dataset contract (if present).
- `contract_hash`: a stable hash of the `TableSchema` (the schema “fingerprint”).

#### Convenience APIs and injection

The registry module exposes both low-level and target-aware helpers:

- `get_contract(...)` / `require_contract(...)`: resolve by explicit `domain` + `target`.
- `get_contract_for_target(...)` / `require_contract_for_target(...)`: resolve by `table_key` +
  `target_name` (domain is derived from `table_key`).

For testing or specialized deployments, the global registry can be replaced:

- `set_contract_registry(registry=...)` and `get_contract_registry()`

This enables swapping out contract resolution behavior without changing target code.

### Factory helpers (`Scope 8`)

To reduce repetitive `ContractOverrides(...)` scaffolding, targets use a factory:

```python
from codeintel.build.contracts.registry import contract_for_table

CONTRACT = contract_for_table(
    table_key="analytics.graph_metrics_functions",
    target_name="graph_metrics",
    input_name="graph_metrics_functions__base",
    required_cols=(),
    clip_column=None,
)
```

This is the canonical pattern for targets under `src/codeintel/build/hamilton/native/analytics/*`.

There is also a batch helper for multi-table targets:

```python
from codeintel.build.contracts.registry import ContractForTableInput, contracts_for_target

contracts = contracts_for_target(
    target_name="config_graph_metrics",
    specs=(
        ContractForTableInput(
            table_key="analytics.config_graph_metrics_keys",
            input_name="config_graph_metrics_keys__base",
            required_cols=(),
            clip_column=None,
        ),
        # ...
    ),
)
```

## Contract Attachment: From Contract Specs to Hamilton Targets

### Table target specs

Hamilton targets are described by table-target specs and contexts:

- `src/codeintel/build/hamilton/native/patterns/table_target.py`

Key concepts:

- `TableTargetContext` / `TableTargetTableContext`: context objects used to build consistent
  `TableTargetSpec` and table specs.
- `build_single_table_target_spec(...)` / `build_multi_table_target_spec(...)`: build specs used to
  attach target templates.
- `TableTargetContext.from_contract(...)` and `TableTargetTableContext.from_contract(...)`:
  context constructors that ensure the **contract drives base node wiring** (the contract’s
  `input_name` becomes the single source of truth for the base node name).

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
     `TableTargetContext.from_contract(...)` is the canonical wiring pattern).
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
   `require_contract_for_target(table_key=..., target_name=...)`.
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

## Cross-Tabular Alignment (Arrow + Polars) (`Scope 7`)

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

## Diagnostics: Alignment Reports and Materialization Metadata (`Scope 6`)

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

## Output Metadata: Contract Identity on Materialized Datasets (`Scope 6 + 5`)

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

## Loader-Side Migration: Keeping Old Snapshots Readable (`Scope 5`)

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

- `register_contract_migration(...)` registers a function keyed by `(table_key, from_version, to_version)`.
- `apply_contract_migration(...)` applies the registered migration to an Arrow table.

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

## Pipeline Ordering: Where Contracts Apply (`Scope 4`)

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
2) In a target module, declare the contract spec via `contract_for_table(...)`.
3) Use `TableTargetContext.from_contract(...)` (single-table) or
   `TableTargetTableContext.from_contract(...)` (multi-table) so `input_name` is the base-node
   source of truth.
4) Ensure output alignment happens through:
   - `align_table_to_contract(...)` / `align_reader_to_contract(...)` for Arrow, or
   - `align_tabular_to_contract(...)` if the node can return Polars.
5) If you change schema versions, add a migration in `src/codeintel/build/contracts/migrations.py`
   when older snapshots must remain readable.

## File Index (Key Entry Points)

- Contract types: `src/codeintel/build/contracts/types.py`
- Contract registry + factories: `src/codeintel/build/contracts/registry.py`
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
