# Contract System Best In Class Plan

## Goals
- Make contracts the single source of truth for schema, alignment, and validation behavior.
- Improve runtime performance with fast-path alignment and streaming-first flows.
- Make contract behavior explicit, testable, and easy to evolve without drift.
- Improve developer ergonomics with factories and consistent patterns.

## Non-Goals
- Change table keys, table names, or output formats.
- Replace the existing schema registry or rewrite contracts end-to-end in one pass.

## Scope 1: Contract Registry and Resolver (single source of truth)
### Overview
Centralize contract creation and retrieval so `TableContractSpec` is derived from
the schema registry plus target overrides. This eliminates duplicated contract
definitions and keeps schema and contracts in sync.

### Status
Completed.

### Implementation Steps
- Introduce a `ContractRegistry` interface with `get_contract` / `require_contract`.
- Add a concrete `SchemaBackedContractRegistry` that uses the schema service to
  build `TableContractSpec` on demand.
- Provide a `ContractResolver` helper to merge schema defaults with target-level
  overrides (input_name, ops_module, columns_to_pass).
- Switch contract constants in analytics modules to use the registry.

### Code Pattern
```python
@dataclass(frozen=True, slots=True)
class ContractOverrides:
    input_name: str | None = None
    ops_module: ModuleType | None = None
    columns_to_pass: Sequence[str] = ()
    required_cols: Sequence[str] | None = None
    clip_column: str | None = None


class ContractRegistry(Protocol):
    def require_contract(
        self,
        *,
        table_key: str,
        domain: str,
        target: str,
        overrides: ContractOverrides | None = None,
    ) -> TableContractSpec: ...


def require_contract(
    *,
    table_key: str,
    domain: str,
    target: str,
    overrides: ContractOverrides | None = None,
) -> TableContractSpec:
    registry = get_contract_registry()
    return registry.require_contract(
        table_key=table_key,
        domain=domain,
        target=target,
        overrides=overrides,
    )
```

### Target Files
- New: `src/codeintel/build/contracts/registry.py`
- New: `src/codeintel/build/contracts/types.py`
- `src/codeintel/build/hamilton/transforms/table_contract.py`
- `src/codeintel/build/hamilton/native/analytics/*` (replace contract constants)
- `src/codeintel/build/contracts/__init__.py`

### Completed Work
- Added registry + resolver + overrides/policy types in `src/codeintel/build/contracts/registry.py`
  and `src/codeintel/build/contracts/types.py`.
- Wired policy-aware `TableContractSpec` defaulting in
  `src/codeintel/build/hamilton/transforms/table_contract.py`.
- Migrated analytics contracts to `require_contract(..., overrides=...)` across
  `src/codeintel/build/hamilton/native/analytics/*`.

## Scope 2: Explicit Contract Policy Surface
### Overview
Make alignment/validation behavior explicit via a policy object. This removes
implicit behavior in alignment helpers and makes changes easy to reason about.

### Status
Completed.

### Implementation Steps
- Define a `ContractPolicy` dataclass (extras policy, type coercion, null policy,
  validation profile, clipping behavior).
- Attach `policy` to `TableContractSpec`.
- Update `align_table_to_contract` to accept or derive `ContractPolicy`.
- Add a default policy in the registry that can be overridden per target.

### Code Pattern
```python
@dataclass(frozen=True, slots=True)
class ContractPolicy:
    extras_policy: ExtrasPolicy | None = None
    validation_profile: ValidationProfile | None = None
    coerce_types: bool = True
    allow_nulls: bool = True


@dataclass(frozen=True, slots=True)
class TableContractSpec:
    table_key: str
    domain: str
    target: str
    ops_module: ModuleType | None
    columns_to_pass: Sequence[str]
    required_cols: Sequence[str] = ("loc", "cyclo")
    clip_column: str | None = "loc"
    input_name: str = "df"
    policy: ContractPolicy = ContractPolicy()
```

### Target Files
- `src/codeintel/build/hamilton/transforms/table_contract.py`
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/contracts/registry.py` (default policy)
- `src/codeintel/build/graphs/assembly/contracts.py`

### Completed Work
- Added `ContractPolicy` and attached it to `TableContractSpec` in
  `src/codeintel/build/hamilton/transforms/table_contract.py`.
- Made alignment policy-aware with explicit type-coercion control in
  `src/codeintel/build/tabular/arrow_ops.py`.
- Threaded policy support through graph assembly alignment in
  `src/codeintel/build/graphs/assembly/contracts.py`.

## Scope 3: Fast-Path Alignment + Streaming First
### Overview
Avoid unnecessary table materialization when schemas already match and favor
reader-based alignment for large data.

### Status
Completed.

### Implementation Steps
- Add a schema-equality fast path in `align_table_to_contract`.
- Introduce `ContractAlignmentPlan` cached per table to reduce repeated work.
- Prefer `align_reader_to_contract` for pipeline paths already in reader form.

### Code Pattern
```python
def align_table_to_contract(
    table_key: str,
    table: pa.Table,
    *,
    policy: ContractPolicy | None = None,
) -> pa.Table:
    contract_schema = get_contract_schema(table_key, policy=policy)
    if table.schema.equals(contract_schema, check_metadata=False):
        return table
    reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches())
    aligned = align_reader_to_contract(table_key, reader, policy=policy)
    return reader_to_table(aligned)
```

### Target Files
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/hamilton/transforms/ingestion_normalize.py`
- `src/codeintel/build/graphs/assembly/contracts.py`

### Completed Work
- Added schema equality fast paths (including metadata-only updates) in
  `src/codeintel/build/tabular/arrow_ops.py`.
- Introduced cached `ContractAlignmentPlan` keyed by `table_key`, `target_name`,
  and extras policy in `src/codeintel/build/tabular/arrow_ops.py`.
- Routed table alignment through reader-first logic when schemas do not match.

## Scope 3.1: Target-Aware Contract Alignment (table_key + target_name)
### Overview
Make alignment keyed to the target output so the same table key can be shaped
slightly differently across targets without collisions or drift.

### Status
In progress (major call sites updated).

### Implementation Steps
- Add a target-aware alignment helper that accepts `table_key` + `target_name`.
- Update `align_table_to_contract` call sites in target modules to pass target name.
- Update contract resolution to use `table_key + target_name` as the lookup key.

### Code Pattern
```python
def align_table_to_contract(
    table_key: str,
    table: pa.Table,
    *,
    target_name: str,
    policy: ContractPolicy | None = None,
) -> pa.Table:
    contract = resolve_contract(
        table_key=table_key,
        target_name=target_name,
        policy=policy,
    )
    schema = contract_to_schema(contract)
    return align_table_to_schema(table, schema)
```

### Target Files
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/graphs/assembly/contracts.py`
- `src/codeintel/build/hamilton/transforms/ingestion_normalize.py`
- `src/codeintel/build/hamilton/native/analytics/*`
- `src/codeintel/build/hamilton/native/ingestion/*`

### Completed Work
- Added `target_name` support to alignment helpers in
  `src/codeintel/build/tabular/arrow_ops.py`.
- Threaded target-aware alignment through graph assembly helpers in
  `src/codeintel/build/graphs/assembly/contracts.py`.
- Added `target_name` parameter to ingestion normalization in
  `src/codeintel/build/hamilton/transforms/ingestion_normalize.py`.
- Passed `target_name` through primary ingestion and graph alignment call sites,
  including `ingest_targets`, `scip_resolution`, `syntax_enrich`, `syntax_augment`,
  `call_wiring`, `cdg`, `pdg`, and `cpg2` assembly.

### Remaining Work
- Confirm any remaining alignment call sites (if any) pass `target_name`
  consistently and fold in diagnostics reporting.

## Scope 4: Contract Pipeline as a First-Class Stage
### Overview
Ensure a consistent ordering for contract steps: clean, feature, align, validate,
persist. This makes runtime behavior predictable and testable.

### Status
Completed.

### Implementation Steps
- Create a `contract_pipeline` helper that composes the steps in a fixed order.
- Update `table_contract` to use the pipeline helper.
- Ensure saver nodes in `table_target` call the pipeline before persistence.

### Code Pattern
```python
def contract_pipeline(
    *,
    spec: TableContractSpec,
    policy: ContractPolicy,
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    def _apply(fn: Callable[..., object]) -> Callable[..., object]:
        fn = pipe_clean_df(
            required_cols=spec.required_cols,
            clip_column=spec.clip_column,
            input_name=spec.input_name,
            namespace=f"prep__{sanitize_pipeline_component(spec.table_key)}",
        )(fn)
        if spec.ops_module is not None:
            fn = with_features(
                table_key=spec.table_key,
                columns_to_pass=tuple(spec.columns_to_pass),
                ops_module=spec.ops_module,
            )(fn)
        fn = pipe_contract_alignment(
            table_key=spec.table_key,
            target_name=spec.target,
            policy=spec.policy,
            namespace=f"align__{sanitize_pipeline_component(spec.table_key)}",
        )(fn)
        fn = pipe_canonical_output(
            table_key=spec.table_key,
            namespace=f"post__{sanitize_pipeline_component(spec.table_key)}",
        )(fn)
        return fn
    return _apply
```

### Target Files
- `src/codeintel/build/hamilton/transforms/table_contract.py`
- `src/codeintel/build/hamilton/native/patterns/table_target.py`
- `src/codeintel/build/hamilton/transforms/ingestion_normalize.py`

### Completed Work
- Added `contract_pipeline` helper and routed `table_contract` through it in
  `src/codeintel/build/hamilton/transforms/table_contract.py`.
- Added contract alignment as a pipeline stage via `pipe_contract_alignment`,
  keeping validation at saver-level decorators.

## Scope 5: Contract Versioning + Migration Hooks
### Overview
Attach a contract version and hash to outputs so future schema changes are
auditable and compatible across releases.

### Implementation Steps
- Add `contract_version` and `contract_hash` to contract descriptors.
- Include this metadata in materialization records and target tags.
- Provide a migration hook that can transform data when contract versions differ.

### Code Pattern
```python
@dataclass(frozen=True, slots=True)
class ContractDescriptor:
    table_key: str
    version: str
    schema_hash: str


def record_contract_metadata(
    *,
    context: MaterializationRecordContext,
    contract: ContractDescriptor,
) -> None:
    context.extra_metadata["contract_version"] = contract.version
    context.extra_metadata["contract_hash"] = contract.schema_hash
```

### Target Files
- `src/codeintel/build/contracts/types.py`
- `src/codeintel/build/schemas/schema_index.py`
- `src/codeintel/build/hamilton/native/materialization_records.py`
- `src/codeintel/build/hamilton/native/patterns/table_target.py`

## Scope 6: Alignment Diagnostics
### Overview
Provide actionable diagnostics for missing/extra columns and type coercions.
This helps debugging and enables observability at scale.

### Status
Completed.

### Implementation Steps
- Add `AlignmentReport` with missing/extra/coerced columns and row counts.
- Return the report from alignment helpers (or optional callback).
- Emit the report into materialization metadata or logs once per target.

### Code Pattern
```python
@dataclass(frozen=True, slots=True)
class AlignmentReport:
    table_key: str
    missing_columns: tuple[str, ...]
    extra_columns: tuple[str, ...]
    coerced_columns: tuple[str, ...]


def align_table_to_contract(
    table_key: str,
    table: pa.Table,
    *,
    policy: ContractPolicy | None = None,
    report: AlignmentReport | None = None,
) -> tuple[pa.Table, AlignmentReport]:
    # build report and return aligned table
    ...
```

### Target Files
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/hamilton/native/materialization_records.py`
- `src/codeintel/build/hamilton/native/patterns/table_target.py`
- `src/codeintel/build/graphs/assembly/contracts.py`

### Completed Work
- Added `AlignmentReport`, `AlignmentReporter`, and `emit_alignment_report` in
  `src/codeintel/build/tabular/arrow_ops.py`.
- Added reporter hooks to alignment helpers in
  `src/codeintel/build/tabular/arrow_ops.py` and
  `src/codeintel/build/graphs/assembly/contracts.py`.
- Threaded reporter usage through ingestion and graph call sites (including
  `ingest_targets`, `scip_resolution`, `syntax_enrich`, `syntax_augment`,
  `call_wiring`, `cdg`, `pdg`, `cpg2` assembly, and `subsystem_cache`).
- Attached alignment reports to materialization metadata via
  `src/codeintel/build/hamilton/native/materialization_records.py`.

## Scope 7: Cross-Tabular Alignment (Arrow + Polars)
### Overview
Provide a unified alignment helper that accepts Arrow tables/readers and Polars
frames to ensure consistent contract alignment across backends.

### Implementation Steps
- Add `align_tabular_to_contract` to handle Arrow and Polars inputs.
- Normalize to Arrow for alignment, then convert back when needed.
- Update call sites that manually align Polars to use the unified helper.

### Code Pattern
```python
def align_tabular_to_contract(
    table_key: str,
    value: pa.Table | pa.RecordBatchReader | pl.DataFrame | pl.LazyFrame,
    *,
    policy: ContractPolicy | None = None,
) -> pa.Table | pa.RecordBatchReader | pl.DataFrame | pl.LazyFrame:
    table = tabular_to_arrow_table(value)
    aligned = align_table_to_contract(table_key, table, policy=policy)
    return convert_back(value, aligned)
```

### Target Files
- `src/codeintel/build/tabular/conversion.py`
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/hamilton/native/patterns/loaders.py`
- `src/codeintel/build/hamilton/native/analytics/subsystem_cache.py`

## Scope 8: Developer Ergonomics (Factories and Codegen)
### Overview
Reduce repetitive contract definitions by adding helpers that synthesize contract
specs from schemas and target metadata.

### Implementation Steps
- Add `contract_for_table` and `contracts_for_target` helpers.
- Provide `TableTargetContext.from_contract` and `TableTargetTableContext.from_contract`
  as the canonical pattern (already present; migrate remaining call sites).
- Add a lint/guardrail that flags contract specs not created via the factory.

### Code Pattern
```python
def contract_for_table(
    *,
    table_key: str,
    domain: str,
    target: str,
    input_name: str,
    ops_module: ModuleType | None = None,
) -> TableContractSpec:
    overrides = ContractOverrides(
        input_name=input_name,
        ops_module=ops_module,
    )
    return require_contract(
        table_key=table_key,
        domain=domain,
        target=target,
        overrides=overrides,
    )
```

### Target Files
- `src/codeintel/build/contracts/registry.py`
- `src/codeintel/build/contracts/types.py`
- `src/codeintel/build/hamilton/native/analytics/*`
- `src/codeintel/build/hamilton/native/patterns/table_target.py`

## Suggested Rollout
1) Land ContractRegistry + ContractPolicy (scopes 1-2).
2) Align fast-path + streaming improvements (scope 3).
3) Introduce pipeline step and diagnostics (scopes 4 and 6).
4) Add versioning and cross-tabular alignment (scopes 5 and 7).
5) Migrate contracts to factory usage + guardrails (scope 8).
