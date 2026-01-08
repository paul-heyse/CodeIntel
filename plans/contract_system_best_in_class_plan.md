# Contract System Best-in-Class Implementation Plan

This document defines a comprehensive redesign and implementation plan for the
build-time contract system. The goal is to strengthen validation and evolution
while making schema/contract authoring frictionless and robust under runtime
composition.

## Target State Summary

- Contract resolution is **lazy** and safe at import time (no schema service required).
- Contract policy is centralized in a **registry + profiles**, not embedded in targets.
- Schema evolution is **classified, gated, and visible** (compat vs breaking changes).
- Alignment diagnostics are **first-class datasets** with durable metadata.
- Migrations are **formal, chained, and testable** for older snapshots.
- Authoring new outputs is **scaffolded** and validated with coverage reporting.

## Foundational Redesign (System-Level)

This redesign removes contract resolution from import-time execution and turns
contracts into lightweight references resolved after SchemaService is configured.

Design changes:
- Introduce `ContractRef` as the primary contract handle used by targets.
- Move contract resolution to a `ContractRuntime` initialized during runtime
  composition (after SchemaService is configured).
- Require that alignment and materialization use resolved contracts, ensuring
  `contract_hash` and `contract_version` are attached to outputs without import-time
  schema resolution.
- Centralize policy selection in a registry to remove policy duplication across targets.

Representative pattern:
```python
from codeintel.build.contracts.ref import contract_ref_for_table
from codeintel.build.hamilton.native.patterns import TableTargetContext

CONTRACT = contract_ref_for_table(
    table_key="analytics.graph_metrics_functions",
    target_name="graph_metrics",
    input_name="graph_metrics_functions__base",
)

TABLE_CONTEXT = TableTargetContext.from_contract_ref(CONTRACT)
```

Target files (foundation):
- `src/codeintel/build/contracts/ref.py`
- `src/codeintel/build/contracts/runtime.py`
- `src/codeintel/build/hamilton/native/patterns/table_target.py`
- `src/codeintel/runtime/compose.py`

## Scope Items

### 1) Lazy Contract Resolution (ContractRef + ContractRuntime)

Status: Completed

Goal:
- Remove import-time dependency on SchemaService while preserving contract hashes.

Redesign details:
- `contract_for_table` becomes `contract_ref_for_table` and returns a `ContractRef`
  instead of an eagerly resolved `TableContractSpec`.
- `ContractRuntime` resolves refs on-demand using SchemaService, caching resolved
  `TableContractSpec` by `(table_key, target_name, input_name, overrides)`.
- `TableTargetContext.from_contract_ref` wires the base node name from the ref and
  injects a resolver callback into the target spec.

Implementation steps:
1. Add `ContractRef` dataclass + resolver hook.
2. Add `ContractRuntime` with cached `resolve(ref)` method.
3. Update table-target patterns to accept ContractRef, resolve at execution/attach time.
4. Keep guardrail: validate `contract_hash` once resolved and before materialization.

Completed scope (current implementation):
- Added `ContractRef` and `contract_ref_for_table` for lazy contract references.
- Added `ContractRuntime` with cached resolution and a fallback SchemaService.
- Updated `TableTargetContext`/`TableTargetTableContext` to support `from_contract_ref`.
- Updated `TableTargetTableSpec` to carry `contract_ref` and resolve it at attach time.
- Wired `configure_contract_runtime` into runtime composition so refs resolve after schema config.
- Migrated analytics Hamilton targets to use `contract_ref_for_table` to avoid import‑time
  resolution.
- Added ContractRef runtime tests (resolution + table target attachment).
- Documented deprecation guidance for `contract_for_table`.

Remaining scope (item 1):
- None. Item 1 is complete.

Representative code pattern:
```python
@dataclass(frozen=True)
class ContractRef:
    table_key: str
    target_name: str
    input_name: str
    overrides: ContractOverrides | None = None

class ContractRuntime:
    def __init__(self, schema_service: SchemaService) -> None:
        self._schema_service = schema_service
        self._cache: dict[ContractRef, TableContractSpec] = {}

    def resolve(self, ref: ContractRef) -> TableContractSpec:
        if ref in self._cache:
            return self._cache[ref]
        contract = require_contract_for_target(
            table_key=ref.table_key,
            target_name=ref.target_name,
            overrides=ref.overrides,
        )
        self._cache[ref] = contract
        return contract
```

Target files:
- `src/codeintel/build/contracts/ref.py`
- `src/codeintel/build/contracts/runtime.py`
- `src/codeintel/build/contracts/registry.py`
- `src/codeintel/build/hamilton/native/patterns/table_target.py`
- `src/codeintel/runtime/compose.py`
- `src/codeintel/build/hamilton/native/analytics/*` (ref migration for targets)

### 2) Contract Policy Registry + Profiles

Status: Proposed

Goal:
- Centralize policy selection while keeping target code minimal.

Redesign details:
- Add `ContractPolicyRegistry` with per-table and per-target profiles.
- Support profiles in config (e.g., `contracts.policy_profiles`).
- `ContractRuntime.resolve` applies policy defaults based on table/target.

Implementation steps:
1. Add registry with `resolve_policy(table_key, target_name)` API.
2. Load policy profiles from config into registry during compose.
3. Remove per-target policy tweaks unless explicitly overridden.

Representative code pattern:
```python
registry = ContractPolicyRegistry()
registry.register_profile(
    name="strict",
    extras_policy="reject",
    coerce_types=False,
)
registry.attach_table_profile("analytics.graph_metrics_functions", "strict")
```

Target files:
- `src/codeintel/build/contracts/policy_registry.py`
- `src/codeintel/core/config/settings.py`
- `src/codeintel/runtime/compose.py`
- `config/codeintel.build.toml`

Active policy defaults (current config):
- `default_profile = "default"` (coerce types + allow nulls).
- `strict` rejects extras and disables coercion; `lenient` retains extras.
- Strict targets: core graph + ingestion (`call_graph`, `cfg`, `cpg`, `scip`, etc.).
- Lenient targets: analytics metrics + config graph outputs.
- Table-level overrides for selected graph edges + analytics graph/config tables.

### 3) Schema Evolution Classification + Gating

Status: Proposed

Goal:
- Detect and gate breaking schema changes systematically.

Redesign details:
- Add a schema diff tool that classifies changes (compatible/additive vs breaking).
- Integrate with `tools.quality_report` as a required gate.
- Require explicit approvals for breaking changes via a small registry file.

Implementation steps:
1. Implement schema diff classifier (column adds, nullability changes, type changes).
2. Add a `schema_breaks.yaml` allowlist with explicit approvals.
3. Add a quality_report step that fails on unapproved breaking changes.

Representative code pattern:
```python
diff = classify_schema_change(old_schema, new_schema)
if diff.breaking and not is_approved(diff, approvals):
    raise SystemExit(f"Breaking schema change: {diff.summary}")
```

Target files:
- `tools/schema_diff.py`
- `tools/quality_report.py`
- `config/schema_breaks.yaml`

### 4) Contract Diagnostics as First-Class Outputs

Status: Proposed

Goal:
- Persist alignment diffs for auditability and quality enforcement.

Redesign details:
- Add `build.contract_alignment_issues` (per-run) with counts of missing/extra/coerced.
- Update alignment reporter to emit structured events during materialization.
- Attach `contract_hash` + `contract_version` to each diagnostic row.

Implementation steps:
1. Add table schema for diagnostics.
2. Extend `align_*_to_contract` reporters to emit structured diff rows.
3. Persist diagnostics as a build dataset in each run.

Representative code pattern:
```python
def report_alignment(table_key: str, target: str, diff: AlignmentDiff) -> None:
    diagnostics.append(
        {
            "table_key": table_key,
            "target": target,
            "missing_count": diff.missing_count,
            "extra_count": diff.extra_count,
            "coerced_count": diff.coerced_count,
            "contract_hash": diff.contract_hash,
            "recorded_at": utc_now(),
        }
    )
```

Target files:
- `src/codeintel/core/schemas/table_registry.py`
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`
- `src/codeintel/build/hamilton/native/diagnostics.py`

### 5) Migrations Registry + Loader Integration

Status: Proposed

Goal:
- Maintain backward compatibility for older snapshots.

Redesign details:
- Add migration registry keyed by `(table_key, from_hash, to_hash)`.
- Loader applies migration chain when reading data with older hash.
- Migrations are explicit, deterministic Arrow transforms.

Implementation steps:
1. Add migration registry with registration decorators.
2. Add loader hook to apply migrations to older snapshots.
3. Provide a migration test harness for deterministic validation.

Representative code pattern:
```python
@register_migration(
    table_key="analytics.graph_metrics_functions",
    from_hash="abc",
    to_hash="def",
)
def migrate_v1_to_v2(table: pa.Table) -> pa.Table:
    return ensure_table_columns(table, ["new_col"])
```

Target files:
- `src/codeintel/core/schemas/migrations.py`
- `src/codeintel/core/datasets/arrow_store.py`
- `src/codeintel/build/exports/writers.py`

### 6) Schema Authoring Ergonomics (Scaffold + Coverage)

Status: Proposed

Goal:
- Make adding new outputs repeatable and low-friction.

Redesign details:
- Add a scaffold CLI that generates a target module, contract wiring, and schema stub.
- Add a schema coverage report (DAG outputs vs declared schemas).

Implementation steps:
1. Create scaffold CLI with templates for analytics and ingestion targets.
2. Build a coverage report tool that lists missing contracts or schemas.
3. Integrate coverage report into quality_report (warning or required gate).

Representative code pattern:
```python
def missing_contracts(catalog: DagCatalog) -> list[str]:
    return sorted(
        table_key for table_key in catalog.table_outputs
        if table_key not in TABLE_SCHEMAS
    )
```

Target files:
- `tools/contract_scaffold.py`
- `tools/schema_coverage.py`
- `tools/quality_report.py`
- `templates/contracts/*`

## Rollout Phases

1) Introduce ContractRef + ContractRuntime and update targets to use it.
2) Add policy registry and move policy decisions out of target code.
3) Add schema diff classification + gating.
4) Add alignment diagnostics table and hook into materializers.
5) Add migrations registry + loader support.
6) Add scaffolding + schema coverage tooling.

## Validation Strategy

- Unit tests for contract resolution and policy registry.
- Schema diff tests with representative breaking and non-breaking changes.
- Migration tests with real snapshot fixtures.
- Integration tests for alignment diagnostics emission and persistence.
