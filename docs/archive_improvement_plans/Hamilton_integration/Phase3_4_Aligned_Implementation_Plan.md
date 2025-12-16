# Hamilton Phase 3 + Phase 4 Implementation Plan (Storage-Architecture Aligned)

**Last updated:** 2025-12-13

This document is a **comprehensive, implementation-oriented plan** to finish the remaining Hamilton
**Phase 3** and **Phase 4** scope, **aligned with the updated storage architecture** that has been
streamlined since the original planning.

---

## 0) Executive Summary

### Current Architecture State (Post-Streamlining)

The codebase has undergone significant architecture improvements since the original Phase 3/4 planning:

**Storage Layer Centralization:**
- `DuckDBPolicyBackend` is now the single point for all DDL and mutation operations
- All table schemas are registered in `codeintel.config.datasets.contracts` via `TABLE_SCHEMAS`
- Views are managed through `VIEW_BUILDERS` registry in `storage/views/ibis_registry.py`
- The `IbisGateway` provides typed access via the `gateway.ibis` interface

**Contract System Maturity:**
- `OutputContract` now references `TableSchema` definitions from `_DATASET_TABLE_SCHEMAS`
- Most targets in `registry.py` already have explicit, complete contracts
- `ContractEnforcer` exists with `for_target()` context manager and validation methods
- Artifact specifications use templated paths (`{scip_dir}`, `{export_dir}`, etc.)

**Asset Catalog (Phase 4 v1) Implementation:**
- Full schema support: `build.asset_versions`, `build.run_asset_versions`, `build.asset_lineage`,
  `build.asset_aliases`, `build.asset_diffs`
- `AssetTracking` accessor with batch operations and UPSERT semantics
- `persist_asset_catalog_for_run()` emitter invoked after Hamilton executions
- Fingerprinting via `compute_fast_version_hash()` and `compute_table_schema_hash()`

**Hamilton Driver Architecture:**
- `HamiltonRuntime` bundles Driver, TargetGraph, and bidirectional mappings
- `auto` mode is already supported in `build_driver()` and CLI validation
- Native target modules loadable via `load_native_modules()` registry

### What's Actually Remaining (Gap Analysis)

Based on code review, the **true remaining gaps** are:

**Phase 3 Remaining (Correctness):**
1. ✅ ~~Contracts are incomplete~~ → Most targets now have complete contracts
2. ✅ ~~`call_graph_views` contract mismatch~~ → Fixed; has proper `graph.v_*` tables
3. ✅ ~~CLI cannot run `auto` mode~~ → Already allowed (`valid_modes` includes `auto`)
4. ⚠️ **Strict contract enforcement not wired** → `ContractEnforcer` exists but not activated
5. ⚠️ **Wrapper allowlist is warning-only** → Needs hard gate option
6. ⚠️ **Tool targets (`typing`) incomplete** → May not materialize all contract tables
7. ⚠️ **Graph export lacks node metadata** → Missing Hamilton node tags/kinds

**Phase 4 Remaining (Feature Work):**
1. ⚠️ **Fingerprint policy includes `commit`** → Blocks cross-commit reuse
2. ⚠️ **Impact analysis CLI missing** → Need `build impact` command
3. ⚠️ **Cross-commit reuse not implemented** → Need `--reuse-from` mechanics
4. ⚠️ **Run environment capture missing** → Need reproducibility records
5. ⚠️ **Graph exports 2.0 not complete** → Need asset/version graph exports

---

## 1) Phase 3 Remaining Work (Aligned Implementation)

### Acceptance Criteria (Phase 3 "Done")

Phase 3 is complete when:
1. Every target has a validated `OutputContract` with explicit tables/artifacts
2. `--strict-contracts` actively enforces write boundaries at the storage layer
3. `--wrapper-allowlist` can optionally fail planning (not just warn)
4. Native tool targets (`typing`, `scip`) materialize their declared outputs
5. Graph export includes Hamilton node metadata (tags, node_kind, impl_kind)

---

### P3-PR-01 — Validate and Audit Contract Completeness

**Status:** Low priority (most work already done)

The original plan identified ~20 targets with empty contracts. Code review shows most now have
complete contracts via `_DATASET_TABLE_SCHEMAS` references.

#### Validation Tasks

1. **Run contract audit script** to confirm no empty contracts remain:

```python
# tools/audit_contracts.py
from codeintel.build.registry import get_target_graph

def audit_contracts() -> list[str]:
    """Return target names with empty or incomplete contracts."""
    graph = get_target_graph()
    issues = []
    for target in graph.all_targets:
        if not target.contract.tables and not target.contract.artifacts:
            if target.plugin:  # Only flag plugin targets
                issues.append(f"{target.name}: empty contract but has plugin '{target.plugin}'")
    return issues
```

2. **Verify contract table keys exist in schema registry:**

```python
from codeintel.config.datasets.contracts import get_table_schemas

def validate_contract_tables() -> list[str]:
    """Return contract table keys not in TABLE_SCHEMAS."""
    schemas = get_table_schemas()
    graph = get_target_graph()
    missing = []
    for target in graph.all_targets:
        for table_key in target.contract.table_keys:
            if table_key not in schemas:
                missing.append(f"{target.name}: contract references unknown table '{table_key}'")
    return missing
```

#### Tests

- Add: `tests/build/hamilton/test_contract_completeness.py`
  - Every plugin target has non-empty contract
  - All contract table keys exist in `get_table_schemas()`
  - Artifact templates are valid for `BuildPaths`

---

### P3-PR-02 — Wire Strict Contract Enforcement at Storage Layer

**Status:** High priority (infrastructure gap)

The `ContractEnforcer` class exists but is not activated during target execution. Writes can
currently bypass contract validation.

#### Current State

```python
# src/codeintel/build/hamilton/contracts/enforcement.py
class ContractEnforcer:
    _current_target: OutputTarget | None = None
    _strict: bool = False
    
    @classmethod
    @contextmanager
    def for_target(cls, target: OutputTarget, *, strict: bool) -> Iterator[None]:
        # This context manager exists but is never entered during execution
```

#### Implementation Tasks

1. **Wrap target execution in enforcement context:**

   In `src/codeintel/build/hamilton/nodes/targets_phase0.py` (or equivalent executor):

   ```python
   from codeintel.build.hamilton.contracts.enforcement import ContractEnforcer
   
   def execute_target(target: OutputTarget, env: BuildEnv, ...) -> TargetRunRecord:
       with ContractEnforcer.for_target(target, strict=env.strict_contracts):
           # Execute plugin or native implementation
           result = _run_plugin(target, env, ...)
       return result
   ```

2. **Hook enforcement into `IbisGateway.write()`:**

   Create `EnforcedIbisGateway` wrapper or add check in `IbisGateway.write()`:

   ```python
   # src/codeintel/build/hamilton/contracts/enforced_gateway.py
   class EnforcedIbisGateway:
       """Wrapper that validates writes against active contract."""
       
       def __init__(self, inner: IbisGateway) -> None:
           self._inner = inner
       
       def write(self, table_key: str, data: Any, **kwargs: Any) -> WriteResult:
           ContractEnforcer.validate_table_write(table_key)
           return self._inner.write(table_key, data, **kwargs)
   ```

3. **Inject enforced gateway during target execution:**

   The `BuildEnv` should expose an enforced gateway when `strict_contracts=True`.

#### Files to Modify

- `src/codeintel/build/hamilton/contracts/enforced_gateway.py` (new)
- `src/codeintel/build/hamilton/nodes/targets_phase0.py`
- `src/codeintel/build/hamilton/executor.py`
- `src/codeintel/build/env.py`

#### Tests

- Add: `tests/build/hamilton/test_strict_contracts_enforcement.py`
  - Fake plugin writes to undeclared table → `ContractViolationError`
  - Native node writes to undeclared table → `ContractViolationError`
  - Writes to declared tables succeed

---

### P3-PR-03 — Make Wrapper Allowlist a Hard Gate

**Status:** Medium priority

The wrapper allowlist currently emits `DeprecationWarning` but does not fail planning.

#### Current Behavior

```python
# src/codeintel/build/hamilton/planner.py (approximately)
if wrapper_target not in allowlist:
    warnings.warn(f"Wrapper target '{wrapper_target}' not in allowlist", DeprecationWarning)
```

#### Implementation Tasks

1. **Add `--wrapper-allowlist-mode` parameter:**
   - `warn` (default): current behavior
   - `strict`: fail planning if any wrapper not in allowlist

2. **Update planner to respect mode:**

   ```python
   if wrapper_target not in allowlist:
       if mode == "strict":
           raise WrapperNotAllowedError(f"Wrapper '{wrapper_target}' not in allowlist")
       else:
           warnings.warn(...)
   ```

3. **Surface in CLI:**

   ```python
   wrapper_allowlist_mode: Annotated[
       str,
       Parameter(
           name=["--wrapper-allowlist-mode"],
           help="How to handle wrappers not in allowlist: warn (default) or strict.",
       ),
   ] = "warn"
   ```

#### Tests

- Add: `tests/build/hamilton/test_wrapper_allowlist_enforcement.py`
  - `mode=warn`: planning succeeds with warning
  - `mode=strict`: planning fails with `WrapperNotAllowedError`

---

### P3-PR-04 — Complete Native Tool Targets (typing, scip)

**Status:** High priority (correctness)

Tool-style native targets may execute successfully but not materialize their contract tables.

#### Current State

The `typing` target has contract:
```python
contract=OutputContract(
    tables=(
        _DATASET_TABLE_SCHEMAS["analytics.typedness"],
        _DATASET_TABLE_SCHEMAS["analytics.static_diagnostics"],
    )
)
```

But the native implementation in `src/codeintel/build/hamilton/native/ingestion/typing.py`
may not actually write these tables.

#### Implementation Tasks

1. **Audit each tool target's actual output:**
   - Run `typing` target and verify `analytics.typedness` has rows
   - Run `typing` target and verify `analytics.static_diagnostics` has rows

2. **Implement table materialization in tool nodes:**

   ```python
   # src/codeintel/build/hamilton/native/ingestion/typing.py
   
   def materialize_typing_results(
       env: BuildEnv,
       pyright_output: PyRightOutput,
       pyrefly_output: PyReflyOutput,
       ruff_output: RuffOutput,
   ) -> list[DatasetRef]:
       """Convert tool outputs to table rows and materialize."""
       typedness_rows = _compute_typedness_rows(pyright_output, pyrefly_output)
       diagnostics_rows = _compute_diagnostics_rows(pyright_output, pyrefly_output, ruff_output)
       
       # Use DuckDBPolicyBackend for writes
       backend = DuckDBPolicyBackend(env.gateway)
       backend.delete_for_snapshot("analytics.typedness", ...)
       backend.bulk_insert("analytics.typedness", typedness_rows, ...)
       
       backend.delete_for_snapshot("analytics.static_diagnostics", ...)
       backend.bulk_insert("analytics.static_diagnostics", diagnostics_rows, ...)
       
       return [
           DatasetRef(table_key="analytics.typedness", row_count=len(typedness_rows)),
           DatasetRef(table_key="analytics.static_diagnostics", row_count=len(diagnostics_rows)),
       ]
   ```

3. **Add skip gating based on input hash:**

   ```python
   def typing__should_run(env: BuildEnv, typing__input_hash: str) -> bool:
       """Check if typing target needs to run based on input hash."""
       # Look up previous run's input_hash from build.run_asset_versions
       tracking = AssetTracking(env.gateway)
       latest = tracking.get_latest_version_hash(
           repo=env.snapshot.repo,
           commit=env.snapshot.commit,
           asset_kind="table",
           asset_key="analytics.typedness",
       )
       if latest is None:
           return True  # No prior output, must run
       # Compare input hashes...
       return prior_input_hash != typing__input_hash
   ```

#### Tests

- Add: `tests/build/hamilton/test_typing_materializes_tables.py`
  - Mock tool executor with deterministic output
  - Assert `analytics.typedness` and `analytics.static_diagnostics` have expected rows

---

### P3-PR-05 — Graph Export Enrichment (Node Metadata)

**Status:** Medium priority

Graph export should include Hamilton node metadata, not just TargetGraph metadata.

#### Desired Export Shape

```json
{
  "nodes": [
    {
      "node_name": "t__function_metrics",
      "target": "function_metrics",
      "impl_kind": "wrapper",
      "node_kind": "target",
      "tags": {"module": "analytics", "outputs": ["analytics.function_metrics"]},
      "contract": {
        "tables": ["analytics.function_metrics", "analytics.function_types"],
        "artifacts": []
      }
    }
  ],
  "edges": [...]
}
```

#### Implementation Tasks

1. **Collect Hamilton node metadata during driver construction:**

   ```python
   # src/codeintel/build/hamilton/observability.py
   
   def collect_node_metadata(driver: Driver) -> dict[str, NodeMetadata]:
       """Extract metadata from Hamilton DAG nodes."""
       metadata = {}
       for node in driver.graph.get_nodes():
           tags = getattr(node.callable, "_tags", {})
           metadata[node.name] = NodeMetadata(
               node_name=node.name,
               tags=tags,
               node_kind=_infer_node_kind(node, tags),
               impl_kind=_infer_impl_kind(node),
           )
       return metadata
   ```

2. **Expose in graph export CLI:**

   ```python
   # src/codeintel/cli/handlers/build.py
   
   def graph_export_handler(ctx: CommandContext) -> CliResult[GraphExportResult]:
       runtime = build_driver(mode=params.hamilton_mode)
       node_metadata = collect_node_metadata(runtime.dr)
       export = build_graph_export(runtime.graph, node_metadata)
       return CliResult.ok(export)
   ```

#### Tests

- Add: `tests/build/hamilton/test_graph_export_includes_metadata.py`
  - Export graph in JSON format
  - Assert each node has `impl_kind`, `node_kind`, `tags` fields

---

### P3-PR-06 — Phase 3 Closure PR

**Status:** Low priority (finalization)

Once the above items are complete:

1. **Update stale documentation** in `docs/Hamilton_integration/`
2. **Add CLI snapshot tests** for new/modified commands
3. **Create Phase 3 completion checklist** in `CHANGELOG.md`

---

## 2) Phase 4 Remaining Work (Aligned Implementation)

### Current Phase 4 Baseline

The following are already implemented:

- ✅ Asset catalog schemas (`build.asset_versions`, etc.)
- ✅ `AssetTracking` accessor with CRUD operations
- ✅ `persist_asset_catalog_for_run()` emitter
- ✅ CLI commands: `build assets`, `build lineage`, `build promote`, `build resolve`, `build diff`
- ✅ Fast fingerprinting via `compute_fast_version_hash()`

### Remaining Phase 4 Acceptance Criteria

Phase 4 is complete when:
1. `version_hash` is stable across commits (content-addressed, not commit-dependent)
2. `build impact` CLI returns downstream impacted assets
3. `--reuse-from-commit` enables cross-commit asset reuse
4. Run environment is captured for reproducibility
5. Asset/version graph can be exported

---

### P4-PR-01 — Stable Fingerprinting Policy (Cross-Commit)

**Status:** High priority (enables reuse)

#### Current Problem

```python
# src/codeintel/build/assets/emitter.py
version_hash = compute_fast_version_hash(
    "table",
    dataset.table_key,
    schema_hash,
    row_count,
    record.input_hash,  # <-- includes repo+commit via hashing.py
    record.options_hash,
)
```

The `input_hash` computation in `hashing.py` includes `snapshot.repo` and `snapshot.commit`,
making version hashes commit-dependent.

#### Implementation Tasks

1. **Create `FingerprintPolicy` abstraction:**

   ```python
   # src/codeintel/build/assets/fingerprinting.py
   
   from enum import Enum
   from dataclasses import dataclass
   
   class FingerprintMode(Enum):
       FAST = "fast"           # Current behavior (commit-dependent)
       STABLE_V1 = "stable_v1" # Content-addressed, commit-independent
   
   @dataclass(frozen=True)
   class FingerprintPolicy:
       mode: FingerprintMode = FingerprintMode.STABLE_V1
       
       def compute_table_version(
           self,
           *,
           table_key: str,
           schema_hash: str | None,
           row_count: int | None,
           upstream_versions: list[str],
           options_hash: str | None,
       ) -> str:
           if self.mode == FingerprintMode.FAST:
               # Legacy behavior
               return compute_fast_version_hash(...)
           
           # Stable v1: exclude repo+commit, include upstream versions
           parts = [
               "stable_v1",
               table_key,
               schema_hash or "",
               str(row_count or 0),
               options_hash or "",
               *sorted(upstream_versions),
           ]
           return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
   ```

2. **Update emitter to use policy:**

   ```python
   # src/codeintel/build/assets/emitter.py
   
   def _dataset_version_record(
       env: BuildEnv,
       *,
       run_id: str,
       record: TargetRunRecord,
       dataset: DatasetRef,
       policy: FingerprintPolicy,
       upstream_versions: list[str],
   ) -> tuple[AssetVersionRecord, RunAssetVersionRecord, _AssetVersionKey]:
       version_hash = policy.compute_table_version(
           table_key=dataset.table_key,
           schema_hash=compute_table_schema_hash(dataset.table_key),
           row_count=dataset.row_count,
           upstream_versions=upstream_versions,
           options_hash=record.options_hash,
       )
       # ...
   ```

3. **Add policy selection to BuildEnv:**

   ```python
   # src/codeintel/build/env.py
   
   @dataclass
   class BuildEnv:
       fingerprint_policy: FingerprintPolicy = field(
           default_factory=lambda: FingerprintPolicy(FingerprintMode.STABLE_V1)
       )
   ```

#### Tests

- Add: `tests/build/hamilton/test_stable_fingerprinting.py`
  - Same content with different commits → same `version_hash`
  - Different content with same commit → different `version_hash`
  - Upstream version change → downstream version changes

---

### P4-PR-02 — Impact Analysis CLI

**Status:** High priority

#### CLI Design

```bash
# Show downstream impacted assets from a specific asset version
codeintel build impact --asset-kind table --asset-key analytics.function_metrics

# Show impact from a specific version hash
codeintel build impact --version-hash abc123...

# Show targets that would need to re-run if this asset changed
codeintel build impact --asset-key analytics.function_metrics --show-targets
```

#### Implementation Tasks

1. **Add impact analysis logic:**

   ```python
   # src/codeintel/build/assets/impact.py
   
   def compute_impact(
       gateway: StorageGateway,
       *,
       asset_kind: str,
       asset_key: str,
       version_hash: str | None = None,
   ) -> ImpactResult:
       """Compute downstream assets impacted by changes to an asset."""
       tracking = AssetTracking(gateway)
       
       # BFS over asset_lineage edges
       impacted: set[tuple[str, str]] = set()
       queue = [(asset_kind, asset_key, version_hash)]
       
       while queue:
           kind, key, version = queue.pop(0)
           downstream = _get_downstream_edges(tracking, kind, key, version)
           for edge in downstream:
               if (edge.downstream_kind, edge.downstream_key) not in impacted:
                   impacted.add((edge.downstream_kind, edge.downstream_key))
                   queue.append((edge.downstream_kind, edge.downstream_key, edge.downstream_version))
       
       return ImpactResult(impacted_assets=list(impacted))
   ```

2. **Add CLI command:**

   ```python
   # src/codeintel/cli/commands/build.py
   
   @build_app.command("impact")
   def build_impact(
       asset_kind: Annotated[str, Parameter(help="Asset kind (table, artifact)")],
       asset_key: Annotated[str, Parameter(help="Asset key")],
       version_hash: Annotated[str | None, Parameter(help="Specific version")] = None,
       show_targets: Annotated[bool, Parameter(help="Map to target names")] = False,
   ) -> None:
       """Analyze downstream impact of an asset change."""
       ...
   ```

#### Tests

- Add: `tests/build/hamilton/test_impact_analysis.py`
  - Synthetic lineage graph → correct BFS traversal
  - CLI integration test with JSON output

---

### P4-PR-03 — Cross-Commit Reuse

**Status:** High priority (major feature)

#### CLI Design

```bash
# Reuse outputs from a previous run where inputs match
codeintel build run --goals profiles --reuse-from-run <run_id>

# Reuse from a specific base commit
codeintel build run --goals profiles --reuse-from-commit abc123
```

#### Implementation Tasks

1. **Add reuse detection in planner:**

   ```python
   # src/codeintel/build/hamilton/planner.py
   
   class PlanStatus(Enum):
       COMPUTE = "compute"    # Must run
       SKIP = "skip"          # Already up-to-date in current snapshot
       INHERIT = "inherit"    # Can reuse from base_run/base_commit
   
   def plan_with_reuse(
       graph: TargetGraph,
       goals: list[str],
       *,
       base_run_id: str | None = None,
       base_commit: str | None = None,
   ) -> list[PlannedTarget]:
       """Plan execution considering cross-commit reuse."""
       ...
   ```

2. **Implement version compatibility check:**

   ```python
   def _can_inherit_version(
       current_env: BuildEnv,
       base_version: AssetVersionRecord,
       current_upstream_versions: list[str],
   ) -> bool:
       """Check if a base version can be inherited."""
       # Schema must match
       if base_version.schema_hash != compute_table_schema_hash(base_version.asset_key):
           return False
       
       # Upstream versions must match (or be inheritable themselves)
       # ...
       return True
   ```

3. **Record inheritance in run_asset_versions:**

   ```python
   run_map = RunAssetVersionRecord(
       run_id=run_id,
       asset_kind="table",
       asset_key=table_key,
       version_hash=base_version.version_hash,
       resolution_kind="inherited",  # New resolution kind
       meta={"inherited_from_run": base_run_id},
   )
   ```

#### Tests

- Add: `tests/build/hamilton/test_cross_commit_reuse.py`
  - Run A produces version V1
  - Run B with `--reuse-from-run A` inherits V1 (no re-execution)
  - `resolution_kind="inherited"` recorded correctly

---

### P4-PR-04 — Run Environment Capture

**Status:** Medium priority (reproducibility)

#### Schema Addition

```python
# Add to build.runs or new build.run_environments table

RUN_ENVIRONMENT_COLUMNS = (
    Column("run_id", "VARCHAR", nullable=False),
    Column("python_version", "VARCHAR"),
    Column("os_name", "VARCHAR"),
    Column("os_version", "VARCHAR"),
    Column("tool_versions", "JSON"),  # {"pyright": "1.1.x", ...}
    Column("config_hash", "VARCHAR"),
    Column("git_dirty", "BOOLEAN"),
    Column("captured_at", "TIMESTAMPTZ"),
)
```

#### Implementation Tasks

1. **Add environment capture utility:**

   ```python
   # src/codeintel/build/environment.py
   
   import platform
   import subprocess
   import sys
   
   @dataclass(frozen=True)
   class RunEnvironment:
       python_version: str
       os_name: str
       os_version: str
       tool_versions: dict[str, str]
       config_hash: str
       git_dirty: bool
       
       @classmethod
       def capture(cls, config: BuildConfig) -> RunEnvironment:
           return cls(
               python_version=sys.version,
               os_name=platform.system(),
               os_version=platform.release(),
               tool_versions=_capture_tool_versions(),
               config_hash=_hash_config(config),
               git_dirty=_is_git_dirty(),
           )
   ```

2. **Persist with each run:**

   ```python
   # src/codeintel/build/hamilton/executor.py
   
   def execute_plan(...):
       env = RunEnvironment.capture(config)
       tracking = AssetTracking(gateway)
       tracking.record_run_environment(run_id, env)
       # ... execute targets ...
   ```

#### Tests

- Add: `tests/build/hamilton/test_run_environment_capture.py`
  - Environment recorded with each run
  - Tool versions captured correctly

---

### P4-PR-05 — Asset/Version Graph Export

**Status:** Medium priority

#### CLI Design

```bash
# Export logical asset graph
codeintel build graph --format json --output assets.json

# Export version graph for a specific run
codeintel build graph --run-id abc123 --include-versions --format mermaid
```

#### Implementation Tasks

1. **Add graph export formats:**

   ```python
   # src/codeintel/build/assets/graph_export.py
   
   def export_asset_graph(
       tracking: AssetTracking,
       *,
       repo: str,
       commit: str,
       format: Literal["json", "mermaid", "dot"] = "json",
   ) -> str:
       """Export the asset lineage graph in the specified format."""
       ...
   ```

2. **Add to CLI:**

   ```python
   @build_app.command("graph")
   def build_graph_export(...) -> None:
       """Export asset or version dependency graph."""
       ...
   ```

---

### P4-PR-06 — Quality Gates 2.0 (Policy Enforcement)

**Status:** Medium priority

Consolidate and formalize quality gate policies:

```python
# src/codeintel/build/policies.py

@dataclass
class BuildPolicies:
    strict_contracts: bool = False
    wrapper_allowlist_mode: Literal["warn", "strict"] = "warn"
    require_schema_validation: bool = False
    fingerprint_mode: FingerprintMode = FingerprintMode.STABLE_V1
    
    @classmethod
    def for_ci(cls) -> BuildPolicies:
        """Strict policies for CI environments."""
        return cls(
            strict_contracts=True,
            wrapper_allowlist_mode="strict",
            require_schema_validation=True,
        )
```

---

## 3) Implementation Sequence (Dependency-Aware)

### Phase 3 Ordering

| Order | PR | Priority | Dependencies |
|-------|-----|----------|--------------|
| 1 | P3-PR-01 Contract Audit | Low | None |
| 2 | P3-PR-02 Strict Enforcement | High | None |
| 3 | P3-PR-04 Tool Targets | High | P3-PR-02 |
| 4 | P3-PR-03 Wrapper Gate | Medium | None |
| 5 | P3-PR-05 Graph Export | Medium | None |
| 6 | P3-PR-06 Closure | Low | All above |

### Phase 4 Ordering

| Order | PR | Priority | Dependencies |
|-------|-----|----------|--------------|
| 1 | P4-PR-01 Stable Fingerprinting | High | None |
| 2 | P4-PR-02 Impact Analysis | High | P4-PR-01 |
| 3 | P4-PR-03 Cross-Commit Reuse | High | P4-PR-01 |
| 4 | P4-PR-04 Environment Capture | Medium | None |
| 5 | P4-PR-05 Graph Export | Medium | P4-PR-02 |
| 6 | P4-PR-06 Quality Gates | Medium | P3-PR-02, P3-PR-03 |

---

## 4) Key Architecture Patterns to Follow

### Storage Operations

All write operations should use `DuckDBPolicyBackend`:

```python
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend

backend = DuckDBPolicyBackend(gateway)
backend.delete_for_snapshot(table_key, repo=..., commit=...)
backend.bulk_insert(table_key, rows, columns=...)
backend.upsert(table_key, rows, conflict_columns=..., update_columns=...)
```

### Contract Enforcement

Wrap target execution with enforcement context:

```python
from codeintel.build.hamilton.contracts.enforcement import ContractEnforcer

with ContractEnforcer.for_target(target, strict=env.strict_contracts):
    result = execute_plugin(...)
```

### Asset Tracking

Use `AssetTracking` for catalog operations:

```python
from codeintel.storage.tracking.asset_tracking import AssetTracking

tracking = AssetTracking(gateway)
tracking.record_asset_versions_batch(versions)
tracking.record_run_asset_versions_batch(run_maps)
tracking.record_lineage_edges_batch(edges)
```

### Schema Access

Reference schemas from centralized registry:

```python
from codeintel.config.datasets.contracts import get_table_schemas

schemas = get_table_schemas()
schema = schemas["analytics.function_metrics"]
```

---

## 5) Working Conventions

For each PR:

```bash
# 1. Quality checks
uv run python -m tools.quality_report --output build/quality-results/quality_report.json

# 2. Type checking
uv run pyright --warnings --pythonversion=3.13
uv run pyrefly check

# 3. Tests
uv run pytest -q

# 4. CLI snapshots (if modified)
pytest -m cli_snapshot --update-cli-snapshots
```

---

## 6) Migration Notes

### From Original Plan

The following items from `Phase3_4_Remaining_Implementation_Plan.md` are **already complete**:

- ✅ Most contract parity work (targets have explicit contracts)
- ✅ `call_graph_views` schema/contract alignment
- ✅ CLI `auto` mode support
- ✅ Basic asset catalog v1 (schemas, persistence, emitter)
- ✅ CLI commands for assets, lineage, promote, resolve, diff

### Architecture Alignment

The following patterns have changed from the original plan:

| Original Approach | Current Architecture |
|-------------------|---------------------|
| `ensure_schema()` per module | `DuckDBPolicyBackend.ensure_table()` |
| Direct SQL writes | `IbisGateway.write()` + policy backend |
| Scattered schema definitions | Centralized `TABLE_SCHEMAS` registry |
| Plugin-specific contracts | `OutputContract` with `TableSchema` refs |

---

## 7) Success Metrics

### Phase 3 Complete When:

- [ ] Contract audit script reports zero issues
- [ ] `--strict-contracts` causes failure on undeclared writes
- [ ] `typing` target produces rows in both declared tables
- [ ] Graph export JSON includes `impl_kind` and `node_kind` fields

### Phase 4 Complete When:

- [ ] Same content across commits produces same `version_hash`
- [ ] `build impact` returns correct downstream assets
- [ ] `--reuse-from-run` skips re-execution for matching inputs
- [ ] Run environment captured and queryable

