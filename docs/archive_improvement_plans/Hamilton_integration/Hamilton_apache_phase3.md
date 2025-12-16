
Below is an aggressive, Hamilton‑first **Phase 3** scope + implementation plan that is **explicitly grounded in your current Phase 2 codebase** (what’s in `src/codeintel/build/hamilton/*`, the generated module approach, manifest index, DatasetRef/ArtifactRef, loader nodes, run_targets persistence, graph exports, snapshot test runner, etc.). The goal is to **finish the migration to a best‑in‑class Hamilton DAG** where:

* targets are **contract‑accurate** (tables/artifacts are canonical, not “best effort”),
* “targets” become **pure compute nodes + explicit materializers** (Hamilton style),
* plugins become **optional shims** (or disappear),
* and build UX/observability/testing become first‑class.

I’m going to propose this as **PR‑16 … PR‑27** so it maps cleanly onto your existing PR‑08…PR‑15 Phase 2 structure and tests (`tests/build/hamilton/…` + snapshot manifest).

---

## Phase 3 North Star

### What “best-in-class Hamilton” means for CodeIntel

1. **Contract-first DAG**

   * Every `OutputTarget` has a complete `OutputContract`:

     * exact table keys (including multi-table targets like `function_metrics` producing both `analytics.function_metrics` and `analytics.function_types`)
     * exact artifact outputs (exports, SCIP artifacts, etc.)
   * The *Hamilton DAG* is generated from these contracts, so dataset/loader nodes are always correct.

2. **Two-layer execution model**

   * **Compute layer:** pure Hamilton nodes returning **Ibis expressions / DataFrames / artifact specs**
   * **Materialization layer:** a small set of **materializer nodes** that:

     * apply skip logic
     * write outputs
     * validate outputs (if enabled)
     * save manifest
     * produce `TargetRunRecord`

3. **Progressive migration without duplicated node names**

   * Per-target implementation kind:

     * `wrapper` = call plugin via `_run_target` (your Phase 0/1/2 wrapper approach)
     * `native` = Hamilton compute + materializer for that target
   * You can flip a target from wrapper → native with a one-line registry change.

4. **Observability that scales with a bigger DAG**

   * Today you persist `build.run_targets` from `TargetRunRecord`s.
   * In Phase 3 we keep that, but also add optional node-level telemetry (“run_nodes”) so you can see:

     * which internal compute nodes are slow
     * which tables were validated / row counted
     * where failures happen inside a native target

---

## Phase 3 Deliverables

### Core deliverables (strongly recommended)

* **(A)** Full contract parity in `codeintel/build/registry.py`
* **(B)** A native-target framework:

  * native registry
  * driver composition that combines:

    * generated *assets module* (dataset/loader/artifact nodes for all targets)
    * generated *wrapper targets module* (t__ nodes only for non-native targets)
    * native target modules for the native targets
* **(C)** Native target runner + materialization utilities (shared code)
* **(D)** Migrate a meaningful slice of targets to native (start with the “highest leverage / most downstream / most reused” ones)
* **(E)** Upgrade tests + snapshot goldens to lock behavior

### “Best-in-class” enhancements (optional but I’d do them since you’re pre-prod)

* **(F)** Add `build.run_nodes` telemetry table (node durations, errors)
* **(G)** Add a “strict contracts” mode that fails if anything writes outside its contract
* **(H)** Add “native SQL view library” for complex targets (risk scoring, profiles, etc.)

---

# PR-by-PR Implementation Plan (Phase 3)

## PR‑16 — Contract parity across *all* targets (table keys + artifacts)

### Why this is critical

Right now, several targets **produce tables that are not reflected in `OutputTarget.contract`**, which means:

* dataset/loader nodes are wrong or missing
* `TargetRunRecord.datasets` don’t match what is actually written
* downstream native targets cannot rely on the contract graph

This PR makes contracts authoritative and correct.

### What to change

File: `src/codeintel/build/registry.py`

1. Add `OutputContract(tables=(...))` for every target:

   * Ingestion:

     * `modules` → `core.modules`, `core.file_state`, `core.repo_map` (whatever is truly produced)
     * `typing` → `analytics.typedness`, `analytics.static_diagnostics` (etc.)
     * `tests_ingest` → `analytics.test_results`
     * `coverage_ingest` → `analytics.coverage_lines`
     * `scip` already has artifacts; it should also list any produced tables (if any)
   * Graphs:

     * `call_graph` already has multi-table contract
     * `cfg`, `dfg` should have correct contracts (today `dfg` has a contract, but `plugin=""` in registry is a red flag you can fix here)
   * Analytics:

     * `function_metrics` must include `analytics.function_metrics` **and** `analytics.function_types`
     * `risk_factors` must include `analytics.goid_risk_factors` (not `analytics.risk_factors`)
     * `graph_metrics` already includes multiple
     * etc.
   * Export:

     * `export_jsonl` and `export_parquet` should be artifacts with `{export_dir}/…` templates

2. Make the contract table schemas come from your `_DATASET_TABLE_SCHEMAS` (already present):

   ```py
   contract=OutputContract(
       tables=(
           _DATASET_TABLE_SCHEMAS["analytics.function_metrics"],
           _DATASET_TABLE_SCHEMAS["analytics.function_types"],
       ),
   )
   ```

### Tests to add

Create: `tests/build/hamilton/test_pr16_contract_parity.py`

Suggested assertions:

* every `target.contract.table_keys` exists in `get_table_schemas()`
* no target has empty plugin unless it’s intentionally native-only (Phase 3 will make this explicit)
* optionally: compare plugin metadata `produces_tables` against contract table keys (allow exceptions if plugin metadata is stale)

---

## PR‑17 — Split generated modules: “assets” vs “wrapper targets”

### Why

To safely mix native and wrapper targets **without duplicate function names**, you want:

* **Assets module**: dataset nodes + loader nodes + artifact nodes for *all targets*
  (but **no** `t__` nodes)
* **Wrapper targets module**: `t__` nodes for *non-native targets only*
  (but **no** dataset/loader/artifact nodes)

Right now `node_factory.build_target_module()` can’t generate “assets without target node” for a target because dataset/loader nodes are only emitted when that target node is generated.

### Code changes

File: `src/codeintel/build/hamilton/nodes/node_factory.py`

Add a new option to `GenerationOptions`:

```py
@dataclass(frozen=True)
class GenerationOptions:
    include_target_nodes: bool = True
    include_dataset_nodes: bool = True
    include_loader_nodes: bool = True
    include_artifact_nodes: bool = True
    include_targets: tuple[str, ...] | None = None
    exclude_targets: frozenset[str] = frozenset()
```

Then in `_generate_nodes_for_target(...)`, gate the `t__` creation:

```py
if options.include_target_nodes:
    target_fn = make_target_node(...)
    functions.append(target_fn)
    target_mapping[target.name] = target_node(target.name)
```

But **still allow dataset/loader/artifact nodes** even if `include_target_nodes=False`.

### Tests to add

Create: `tests/build/hamilton/test_pr17_generated_assets_module.py`

Assert:

* assets module exposes `d__...`, `q__...`, `df__...`, `a__...`
* assets module does **not** expose `t__...`
* wrapper targets module exposes `t__...` only

---

## PR‑18 — Native target registry + driver composition

### Goal

Have one runtime “auto mode” driver that composes:

1. native target modules (for selected targets)
2. wrapper targets module (for everything else)
3. assets module (for dataset/loader/artifact nodes)

### Code changes

#### 1) Add native registry

New file: `src/codeintel/build/hamilton/native/registry.py`

```py
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType

@dataclass(frozen=True)
class NativeTargetSpec:
    target: str
    module: str  # import path

NATIVE_TARGETS: tuple[NativeTargetSpec, ...] = (
    NativeTargetSpec("risk_factors", "codeintel.build.hamilton.native.analytics.risk_factors"),
    # more later...
)

def native_target_names() -> frozenset[str]:
    return frozenset(spec.target for spec in NATIVE_TARGETS)

def load_native_modules() -> list[ModuleType]:
    return [import_module(spec.module) for spec in NATIVE_TARGETS]
```

#### 2) Update driver_factory

File: `src/codeintel/build/hamilton/driver_factory.py`

Add a new mode `"auto"` (or reuse `"generated"` but with native injected). Recommended explicit:

```py
def build_driver(*, mode: Literal["phase0", "generated", "auto"] = "auto"):
    graph = get_target_graph()

    if mode == "auto":
        from codeintel.build.hamilton.native.registry import native_target_names, load_native_modules

        native = native_target_names()

        assets = get_generated_module(
            graph=graph,
            options=GenerationOptions(
                include_target_nodes=False,
                include_dataset_nodes=True,
                include_loader_nodes=True,
                include_artifact_nodes=True,
            ),
        )
        wrappers = get_generated_module(
            graph=graph,
            options=GenerationOptions(
                include_target_nodes=True,
                include_dataset_nodes=False,
                include_loader_nodes=False,
                include_artifact_nodes=False,
                exclude_targets=native,
            ),
        )
        native_mods = load_native_modules()

        modules = [*native_mods, wrappers, assets]
        return driver.Driver({}, *modules, adapter=SimplePythonGraphAdapter())
```

*(Exact `driver.Driver` invocation may differ depending on your Hamilton version; keep the conceptual composition.)*

#### 3) Planner uses impl_kind

File: `src/codeintel/build/hamilton/planner.py`

Set:

```py
from codeintel.build.hamilton.native.registry import native_target_names

native = native_target_names()
...
impl_kind = "native" if name in native else "wrapper"
```

### Tests to add

Create: `tests/build/hamilton/test_pr18_native_driver_composition.py`

* Ensure `build_driver(mode="auto")` contains nodes from:

  * native module
  * wrapper targets module
  * assets module
* Ensure plan shows `impl_kind="native"` for those targets

---

## PR‑19 — Native runner + materialization utilities

### Goal

Centralize all “Phase 0 wrapper logic” (hashing, skip, manifest, record building) into a reusable runner that native targets can call.

### Key improvements to make now

1. **Every TargetRunRecord should always include expected DatasetRef / ArtifactRef**, even on:

   * upstream-blocked skip
   * plugin-missing failure
   * execution failure

This is what enables native compute nodes to safely depend on loader nodes *without dataset-node explosions*.

### Code changes

#### 1) Extract “expected output refs”

New file: `src/codeintel/build/hamilton/outputs.py`

```py
from __future__ import annotations

from codeintel.build.targets import TargetGraph
from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.build.hamilton.io.artifact_ref import ArtifactRef
from pathlib import Path

def expected_dataset_refs(graph: TargetGraph, target: str, repo: str, commit: str) -> tuple[DatasetRef, ...]:
    t = graph.targets[target]
    return tuple(DatasetRef(table_key=k, repo=repo, commit=commit) for k in t.table_keys)

def expected_artifact_refs(graph: TargetGraph, target: str, repo: str, commit: str, *, paths, repo_root) -> tuple[ArtifactRef, ...]:
    t = graph.targets[target]
    refs: list[ArtifactRef] = []
    for spec in t.contract.artifacts:
        path = spec.path_template.format(
            build_dir=str(paths.build_dir),
            scip_dir=str(paths.scip_dir),
            export_dir=str(paths.document_output_dir),
            repo_root=str(repo_root),
        )
        refs.append(ArtifactRef(name=spec.name, artifact_type="file", repo=repo, commit=commit, path=path))
    return tuple(refs)
```

#### 2) Native runner

New file: `src/codeintel/build/hamilton/native/runner.py`

Core interface:

```py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Any

import ibis.expr.types as it
import pandas as pd

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.manifest_hook import ManifestSaveRequest, save_manifest
from codeintel.build.hamilton.hashing import compute_target_input_hash_with_deps
from codeintel.build.hamilton.nodes.targets_phase0 import _check_upstream_failures, _should_skip_target
from codeintel.build.hamilton.outputs import expected_dataset_refs, expected_artifact_refs
from codeintel.build.hamilton.manifest_hook import TargetRunRecord

@dataclass(frozen=True)
class NativeMaterializeSpec:
    tables: Mapping[str, it.Table | pd.DataFrame]  # table_key -> expression/df
    artifacts: Mapping[str, str] = None            # artifact name -> path

def run_native_target(
    *,
    env: BuildEnv,
    graph,
    target_name: str,
    upstream: tuple[TargetRunRecord, ...],
    spec: NativeMaterializeSpec | None,
    plugin_label: str,
) -> TargetRunRecord:
    start = time.time()

    # Always attach expected refs
    datasets = expected_dataset_refs(graph, target_name, env.repo, env.commit)
    artifacts = expected_artifact_refs(graph, target_name, env.repo, env.commit, paths=env.paths, repo_root=env.snapshot.repo_root)

    # Upstream gate
    gate = _check_upstream_failures(upstream)
    if gate is not None:
        return TargetRunRecord(
            target=target_name,
            plugin_name=plugin_label,
            status="skipped",
            input_hash=None,
            options_hash=None,
            duration_ms=(time.time() - start) * 1000,
            datasets=datasets,
            artifacts=artifacts,
            error=gate,
        )

    input_hash = compute_target_input_hash_with_deps(
        env=env,
        graph=graph,
        target=target_name,
        upstream=upstream,
    )

    if _should_skip_target(env, target_name, input_hash):
        return TargetRunRecord(
            target=target_name,
            plugin_name=plugin_label,
            status="skipped",
            input_hash=input_hash,
            options_hash=None,
            duration_ms=(time.time() - start) * 1000,
            datasets=datasets,
            artifacts=artifacts,
            error=None,
        )

    if spec is None:
        return TargetRunRecord(
            target=target_name,
            plugin_name=plugin_label,
            status="failed",
            input_hash=input_hash,
            options_hash=None,
            duration_ms=(time.time() - start) * 1000,
            datasets=datasets,
            artifacts=artifacts,
            error="native_target_returned_no_spec",
        )

    # Materialize tables (simple pattern: delete snapshot then insert)
    row_counts: dict[str, int] = {}
    for table_key, payload in spec.tables.items():
        # implement: write_table_snapshot(env.gateway, table_key, payload, env.repo, env.commit)
        # implement: count rows -> row_counts[table_key]
        ...

    duration_ms = (time.time() - start) * 1000
    save_manifest(
        gateway=env.gateway,
        request=ManifestSaveRequest(
            target=target_name,
            repo=env.repo,
            commit=env.commit,
            plugin=plugin_label,
            duration_ms=duration_ms,
            input_hash=input_hash,
            row_count=sum(row_counts.values()) if row_counts else None,
        ),
    )

    return TargetRunRecord(
        target=target_name,
        plugin_name=plugin_label,
        status="succeeded",
        input_hash=input_hash,
        options_hash=None,
        duration_ms=duration_ms,
        datasets=datasets,
        artifacts=artifacts,
        row_counts=row_counts,
        error=None,
    )
```

#### 3) Patch wrapper `_run_target` to always include expected refs

File: `src/codeintel/build/hamilton/nodes/targets_phase0.py`

* On upstream_failed skip: include expected dataset/artifact refs
* On plugin missing: include expected refs
* On failed result: include expected refs

This prevents dataset/loader nodes from breaking the graph shape.

### Tests to add

Create: `tests/build/hamilton/test_pr19_records_always_have_refs.py`

* Simulate upstream failure and verify:

  * the record has datasets and artifacts attached
  * dataset nodes can be called without raising

---

## PR‑20 — First full native target: `risk_factors` (compute + materialize)

This is the ideal first migration because it’s highly downstream, touches many sources, and benefits from contract correctness.

### Code changes

#### 1) Native module

New file: `src/codeintel/build/hamilton/native/analytics/risk_factors.py`

Sketch:

```py
from __future__ import annotations

import ibis
import ibis.expr.types as it

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.runner import NativeMaterializeSpec, run_native_target

def risk_factors_expr(
    env: BuildEnv,
    t__function_metrics: TargetRunRecord,
    t__typing: TargetRunRecord,
    t__coverage_functions: TargetRunRecord,
    t__hotspots: TargetRunRecord,
    t__coverage_test_edges: TargetRunRecord,
) -> it.Table | None:
    # gate early so we don't touch tables when upstream failed
    upstream = (t__function_metrics, t__typing, t__coverage_functions, t__hotspots, t__coverage_test_edges)
    if any(r.status == "failed" for r in upstream):
        return None

    gw = env.gateway
    repo, commit = env.repo, env.commit

    fm = gw.ibis.table("analytics.function_metrics").filter((_.repo == repo) & (_.commit == commit))  # fill properly
    # repeat for function_types, coverage_functions, hotspots, typedness, static_diagnostics, test_coverage_edges, core.modules
    # then build the score expression (Ibis CASE / COALESCE / joins)

    # return final table expression with schema matching analytics.goid_risk_factors
    return result

def t__risk_factors(
    env: BuildEnv,
    graph,
    risk_factors_expr: it.Table | None,
    t__function_metrics: TargetRunRecord,
    t__typing: TargetRunRecord,
    t__coverage_functions: TargetRunRecord,
    t__hotspots: TargetRunRecord,
    t__coverage_test_edges: TargetRunRecord,
) -> TargetRunRecord:
    upstream = (t__function_metrics, t__typing, t__coverage_functions, t__hotspots, t__coverage_test_edges)

    spec = None if risk_factors_expr is None else NativeMaterializeSpec(
        tables={"analytics.goid_risk_factors": risk_factors_expr}
    )
    return run_native_target(
        env=env,
        graph=graph,
        target_name="risk_factors",
        upstream=upstream,
        spec=spec,
        plugin_label="native.risk_factors",
    )
```

#### 2) Register as native

Update: `src/codeintel/build/hamilton/native/registry.py`
Add risk_factors spec.

### Tests to add

Create: `tests/build/hamilton/test_pr20_native_risk_factors.py`

* Unit test of DAG construction:

  * `build_driver(mode="auto")` should include `risk_factors_expr` node and `t__risk_factors`
* Plan test:

  * `compute_plan(...targets=("risk_factors",))` shows impl_kind native
* Optional integration test using in-memory gateway with minimal seed tables

---

## PR‑21 — Native migration wave 1 (SQL/Ibis‑friendly analytics targets)

After risk_factors, migrate the set that are primarily “transform tables from other tables”:

* `entrypoints`
* `external_deps`
* `config_data_flow`
* `data_model_usage`
* `behavioral_coverage`
* `subsystems` + `subsystem_graph_metrics` + `subsystem_agreement` (depending on complexity)
* `test_profile` + `test_graph_metrics` + `symbol_graph_metrics` (if they’re mostly DB transforms)

Pattern:

* each native module defines `*_expr` compute node(s)
* `t__target` materializer writes the contract tables

### Tests

Add one test file per PR or group:

* `tests/build/hamilton/test_pr21_native_wave1_plan.py`
* plus snapshot golden updates for `codeintel build plan` output (now showing more native targets)

---

## PR‑22 — Native export targets (artifact materializers)

Make export targets first-class Hamilton nodes producing ArtifactRefs.

### Code changes

* Add contracts in registry:

  * `export_jsonl` contract artifacts like `{export_dir}/codeintel.jsonl` (or a directory)
  * `export_parquet` similarly

* Native module `codeintel/build/hamilton/native/export/jsonl.py`:

  * compute node builds an export manifest
  * materializer writes files
  * returns TargetRunRecord with artifact refs

### Tests

* add CLI snapshot tests for:

  * `codeintel build run --target export_jsonl` help/plan output
* add unit tests to validate artifact refs are attached to record

---

## PR‑23 — Native migration wave 2 (refactor “compute_*” functions to be pure)

For targets like `coverage_functions` that currently:

* build Ibis expressions
* **and write them inside the compute function**

Refactor pattern:

* extract pure expression builder:

  * `coverage_functions_expr(gateway, cfg) -> it.Table`
* materializer node handles:

  * delete snapshot rows
  * insert expression
  * row count
  * manifest

This is where you gain *real Hamilton reusability*:

* other targets can depend on the expression node
* you can visualize “how coverage_functions is derived” in the DAG

### Tests

* verify expression builder returns a valid Ibis table with expected columns
* integration test writes rows and counts them

---

## PR‑24 — Native migration wave 3 (ingestion/tool targets as Hamilton nodes)

This is the “aggressive architecture optimization” part.

Even tool-based targets can become clean Hamilton nodes if you separate:

* **tool execution nodes** (produce artifacts)
* **parsing/materialization nodes** (write tables)
* **final `t__target` node** that wraps skip/manifest and returns TargetRunRecord

Example for `scip`:

* node `scip_index_path` produces path
* node `scip_run` executes tool and returns ArtifactRef
* node `scip_ingest_tables` ingests into DB
* node `t__scip` ties it all together + manifest

This gives:

* better observability (“tool step vs ingest step”)
* ability to cache based on artifact hashes
* better graph exports

### Tests

* unit tests for the node graph shape
* mocked tool runner tests

---

## PR‑25 — Observability upgrade: optional `build.run_nodes`

Once native targets explode into subgraphs, per-target `run_targets` is not enough.

Add:

* `build.run_nodes` table (node_name, duration_ms, status, error)
* a lightweight Hamilton hook (or driver wrapper) that records node execution events

This becomes your internal “build profiler”.

### Tests

* schema creation test
* “records inserted” test with a small graph

---

## PR‑26 — Strict contracts mode

Add a build config flag:

* `strict_contracts: bool`

When enabled:

* if anything attempts to write to a table not in the current target’s contract → fail target immediately.

How:

* wrap gateway write calls during execution (context manager around target run)
* maintain a whitelist for the current target

This prevents “silent drift” forever.

---

## PR‑27 — Cleanup + final migration gates

* remove Phase0-only code paths you no longer need (optional)
* make `"auto"` driver the only supported path
* ensure CLI graph export uses `"auto"` driver so it includes native nodes
* update docs + snapshots

---

# Phase 3 Recommended Target Migration Order

If you want the most impact early:

1. **Contract parity first** (PR‑16)
2. **risk_factors native** (PR‑20)
3. “transform-heavy analytics” wave (PR‑21)
4. exports (PR‑22)
5. refactor + make compute pure (PR‑23)
6. ingestion/tool nodes (PR‑24)

This order maximizes:

* correctness early (contracts)
* downstream payoff early (risk_factors)
* architecture payoff (pure compute DAG + materializers)

---

## Phase 3 Definition of Done (DoD)

By end of Phase 3, you should be able to say:

* **All targets have correct OutputContracts** (tables + artifacts)
* `build_driver(mode="auto")` produces:

  * native nodes for native targets
  * wrapper nodes only where explicitly allowed
  * dataset/loader/artifact nodes for everything
* `codeintel build plan` shows `impl_kind` correctly and matches what executes
* `codeintel build graph` includes native internal nodes (compute + materialize)
* `TargetRunRecord` always includes expected DatasetRefs/ArtifactRefs (even on skip/failure)
* Snapshot tests cover:

  * plan output with impl_kind
  * graph export output shape
  * at least one native target end-to-end

---

## If you want, I can also generate a Phase 3 PR-by-PR tracking board

In the exact format you liked in Phase 2:

* per PR:

  * tasks checklist
  * tests checklist under `tests/build/hamilton/`
  * snapshot commands + exact expected snapshot filenames to add under `tests/build/hamilton/snapshots/`

…and I can include a suggested “native tag taxonomy” for compute nodes (domain/module/target/output_kind) so your graph exports become genuinely readable at scale.


# Phase 3 Board #

Below is a **Phase 3 PR-by-PR tracking board (PR‑16 … PR‑27)** in the same “execution-ready” style as Phase 2: each PR has

* **Tasks checklist**
* **Tests checklist** (under `tests/build/hamilton/…`)
* **CLI snapshot cases** with **exact commands + snapshot filenames** to add under `tests/build/hamilton/snapshots/`

This plan assumes your Phase 2 baseline is already in place (Hamilton default, planner + explain, manifest prefetch, DatasetRef v2 + ArtifactRef, loader nodes, run_targets persistence, Mermaid/DOT exports, CLI snapshot framework, and the `impl_kind` scaffolding). 
It also explicitly builds off the Phase 3 “pure compute + explicit materializers” preview you’ve been following. 
And it preserves the Phase 1 correctness guarantees (closure execution, upstream failure gating, run tracking, universal dataset lineage). 

---

# Phase 3 Native Tag Taxonomy (apply across PR‑16+)

This is the standard that makes Phase 3 graphs readable at scale.

## Required tags

Use Hamilton `@tag(...)` consistently across nodes:

* `domain`: `ingestion|graphs|analytics|export|build`
* `target`: canonical target name (e.g., `risk_factors`)
* `impl_kind`: `wrapper|native`
* `node_kind`:

  * `target` (final node returning `TargetRunRecord`)
  * `compute` (pure expression builder)
  * `materialize` (writes assets, saves manifest)
  * `validate` (pandera checks / invariants)
  * `dataset` (d__* nodes)
  * `query` (q__* nodes)
  * `dataframe` (df__* nodes)
  * `artifact` (a__* nodes)
  * `tool` (external tool invocation)
* `asset_kind`: `table|view|artifact`
* `asset_key`: table_key like `analytics.function_metrics` or artifact name like `scip_index`

## Recommended extra tags (optional but very useful)

* `module`: target module (`analytics`, `graphs`, etc.)
* `output_key`: for multi-output targets (table_key or artifact name)
* `step`: for compute graphs (`join`, `aggregate`, `score`, etc.)
* `cacheable`: `true|false`

### Naming convention for native internal nodes

To avoid confusion with `t__*` (targets), use these prefixes:

* **Compute nodes**: `c__<target>__<step>`
* **Materializers**: `m__<target>` (optional)
* **Final target node**: keep `t__<target>` returning `TargetRunRecord`

Example:

* `c__risk_factors__fan_in_out`
* `c__risk_factors__score`
* `m__risk_factors` (materialize)
* `t__risk_factors` (returns record; may call `m__risk_factors`)

You can implement these prefixes via `naming.py` helpers in PR‑19.

---

# CLI Snapshot conventions for Phase 3

You already have the YAML-driven CLI snapshot framework from Phase 2. 
For Phase 3, keep snapshots **small and deterministic**:

* Prefer **`build plan`** + **`build graph`** + **`build explain`** snapshots
* Use `--format json` wherever possible
* Use tags in the manifest (`pr20`, `native`, `graph`, `plan`, etc.)

---

# Phase 3 PR-by-PR Tracking Board

## PR‑16 — Contract parity: make contracts authoritative for all targets

### Tasks

* [ ] Audit `OutputTarget.contract` for every target:

  * all produced **table_keys** must be present
  * all produced **artifacts** must be present (SCIP index dirs, exports, etc.)
* [ ] Ensure multi-table targets have complete contract tables (e.g., `function_metrics` includes both tables).
* [ ] Ensure `planner` and graph export use contract table_keys/artifact_keys (not plugin metadata fallbacks).
* [ ] Add a lightweight contract linter function:

  * `validate_contracts(graph) -> list[str]` returning problems

### Tests (`tests/build/hamilton/`)

* [ ] `test_pr16_contract_parity.py`

  * every `contract.table_keys` exists in `_DATASET_TABLE_SCHEMAS`
  * no “empty contract” for targets that claim outputs
  * artifact templates render without KeyError given `paths`
* [ ] If you add `validate_contracts()`: unit test it returns empty on the real graph

### CLI snapshots (`tests/build/hamilton/snapshots/`)

Add cases to `manifest.yaml`:

* **Command**

  ```bash
  codeintel build plan function_metrics --format json
  ```

  **Snapshot file**

  * `pr16_plan_function_metrics_contract.json`

* **Command**

  ```bash
  codeintel build plan risk_factors --format json
  ```

  **Snapshot file**

  * `pr16_plan_risk_factors_contract.json`

What these snapshots should lock:

* `entries[].table_keys` includes the complete contract outputs for those targets.

---

## PR‑17 — Split generated modules: `assets` vs `wrapper_targets`

### Tasks

* [ ] Extend `GenerationOptions` to support:

  * include/exclude targets
  * include_target_nodes / include_dataset_nodes / include_loader_nodes / include_artifact_nodes
* [ ] Generate two modules:

  * **Assets module**: dataset + loader + artifact nodes for all targets
  * **Wrapper targets module**: `t__*` nodes only for wrapper targets
* [ ] Ensure mappings (`TARGET_TO_NODE`, `DATASET_TO_NODE`, `QUERY_TO_NODE`, `DATAFRAME_TO_NODE`, `ARTIFACT_TO_NODE`) remain correct and discoverable.

### Tests

* [ ] `test_pr17_generated_assets_module.py`

  * assets module has `d__/q__/df__/a__` nodes
  * assets module has **no** `t__*`
* [ ] `test_pr17_generated_wrapper_targets_module.py`

  * wrapper module has `t__*`
  * wrapper module has **no** `d__/q__/df__/a__` nodes

### CLI snapshots

This PR is mostly internal. Add one small plan snapshot that should not change shape but is a smoke test for driver assembly:

* **Command**

  ```bash
  codeintel build plan modules --format json
  ```

  **Snapshot**

  * `pr17_plan_smoke.json`

---

## PR‑18 — Native registry + `auto` driver composition + CLI toggles

### Tasks

* [ ] Add `native/registry.py` describing native targets (`target -> module`).
* [ ] Add driver mode `"auto"` (or make `generated` behave like auto):

  * loads native modules for enabled targets
  * loads wrapper target module for everything else
  * loads assets module always
* [ ] Add CLI support:

  * `--native-target <name>` (repeatable)
  * `--native-none` (force wrapper-only)
  * optionally `--native-all` (all targets that have native impl available)
* [ ] Ensure `plan` includes `impl_kind` = `native|wrapper`. (You already have scaffolding.) 

### Tests

* [ ] `test_pr18_native_registry_and_driver.py`

  * enabling native target results in:

    * native node present
    * wrapper node absent for that target
* [ ] `test_pr18_plan_impl_kind.py`

  * plan marks `impl_kind` correctly based on registry/CLI

### CLI snapshots

* **Command**

  ```bash
  codeintel build plan risk_factors --format json --native-target risk_factors
  ```

  **Snapshot**

  * `pr18_plan_impl_kind_native.json`

* **Command**

  ```bash
  codeintel build plan risk_factors --format json --native-none
  ```

  **Snapshot**

  * `pr18_plan_impl_kind_wrapper.json`

---

## PR‑19 — Native materialization framework + “records always have refs”

### Tasks

* [ ] Add `materializers/duckdb.py`:

  * `write_table_snapshot(table_key, expr_or_df, repo, commit, mode=overwrite)`
  * optional: staging+swap for atomicity (even pre-prod, it’s worth doing)
* [ ] Add `native/runner.py`:

  * a single `run_native_target(...) -> TargetRunRecord`
  * handles upstream gating, skip check, manifest save, dataset/artifact refs, validation hook
* [ ] Ensure wrapper path also **always attaches expected DatasetRefs/ArtifactRefs** even on skip/failure/blocked.

### Tests

* [ ] `test_pr19_records_always_have_refs.py`

  * upstream_failed skip includes datasets/artifacts
  * plugin_missing failure includes datasets/artifacts
* [ ] `test_pr19_materializer_writes_snapshot.py`

  * write snapshot then read back and confirm row count or schema

### CLI snapshots

* **Command**

  ```bash
  codeintel build plan risk_factors --format json
  ```

  **Snapshot**

  * `pr19_plan_refs_present.json`

(Here you’re locking that the plan includes `table_keys` and `artifact_keys` correctly for later use.)

---

## PR‑20 — Native target #1: `risk_factors` (compute graph + explicit materialize)

### Tasks

* [ ] Implement native module: `native/analytics/risk_factors.py`

  * compute nodes: `c__risk_factors__...`
  * materializer: `m__risk_factors`
  * final: `t__risk_factors` returns `TargetRunRecord`
* [ ] Compute uses loader nodes (`q__*`) as inputs (Phase 2 capability). 
* [ ] Add optional validation node: `v__risk_factors` (Pandera + invariants)
* [ ] Register in native registry

### Tests

* [ ] `test_pr20_native_risk_factors_plan_and_graph.py`

  * plan shows impl_kind native
  * graph contains compute/materialize node names (or at least more nodes than wrapper)
* [ ] `test_pr20_native_risk_factors_materialization.py` (integration-ish)

  * seed minimal upstream tables, run native risk_factors, assert output table exists

### CLI snapshots

* **Command**

  ```bash
  codeintel build plan risk_factors --format json --native-target risk_factors
  ```

  **Snapshot**

  * `pr20_plan_risk_factors_native.json`

* **Command**

  ```bash
  codeintel build graph risk_factors --format mermaid --native-target risk_factors
  ```

  **Snapshot**

  * `pr20_graph_risk_factors_native.mmd`

---

## PR‑21 — Native wave 1: analytics transforms (high ROI set)

### Tasks

* [ ] Migrate 3–6 transform-heavy analytics targets to native:

  * pick those with clean SQL/Ibis derivations + strong downstream value
  * keep ingestion/tool targets wrapper for now
* [ ] Ensure each migrated target has:

  * compute nodes (pure)
  * materializer node
  * final `t__target` record node
* [ ] Add registry entries

### Tests

* [ ] `test_pr21_native_wave1_impl_kind.py`

  * plan marks all migrated targets as native
* [ ] `test_pr21_native_wave1_smoke_run.py`

  * run a small closure and assert all native outputs materialize successfully (can be tiny-seeded)

### CLI snapshots

* **Command**

  ```bash
  codeintel build plan --module analytics --format json
  ```

  **Snapshot**

  * `pr21_plan_analytics_native_mix.json`

(Goal: lock that `impl_kind` values show a mix until full migration.)

---

## PR‑22 — Native graphs layer: call graph “views” and graph metrics

### Tasks

* [ ] Introduce a “graphs dataflow” package for derived graph tables/views:

  * `native/graphs/call_graph_views.py`
  * `native/graphs/graph_metrics.py`
* [ ] Decide whether outputs should be:

  * materialized tables (preferred for stability), or
  * DuckDB views (fine if you add view materializer support)
* [ ] Add contracts for views if you implement view assets (`asset_kind=view`).

### Tests

* [ ] `test_pr22_native_graphs_views.py`

  * compute returns ibis expressions
  * materializer creates table/view and it’s readable
* [ ] `test_pr22_graph_export_contains_graph_nodes.py`

  * graph export includes these nodes (or their presence in plan closure)

### CLI snapshots

* **Command**

  ```bash
  codeintel build graph call_graph --format dot
  ```

  **Snapshot**

  * `pr22_graph_call_graph_native.dot`

---

## PR‑23 — Native exports: artifact materializers (JSONL/Parquet/etc.)

### Tasks

* [ ] Add `materializers/artifacts.py`:

  * atomic writes (tmp + rename)
  * record ArtifactRefs
* [ ] Convert export targets to native:

  * compute nodes build “export spec” from tables
  * materializer writes files
* [ ] Ensure `a__*` nodes cleanly expose exported artifacts (Phase 2 capability). 

### Tests

* [ ] `test_pr23_native_exports_artifacts.py`

  * run export target in a temp directory
  * assert file exists + artifact refs stored

### CLI snapshots

* **Command**

  ```bash
  codeintel build plan export_jsonl --format json --native-target export_jsonl
  ```

  **Snapshot**

  * `pr23_plan_export_jsonl_native.json`

---

## PR‑24 — Native ingestion refactor: tool steps as Hamilton subgraphs (aggressive)

### Tasks

* [ ] Convert one ingestion/tool target end-to-end (start with SCIP):

  * `tool` node invokes external tool → produces ArtifactRef
  * `parse/materialize` nodes ingest results into tables
  * final `t__scip` returns TargetRunRecord
* [ ] Add caching semantics for tool outputs:

  * hash tool version + inputs + config
  * skip tool step if artifact already present and matches
* [ ] Add targeted parallelism hooks if safe (optional now)

### Tests

* [ ] `test_pr24_native_scip_graph_shape.py`

  * node graph includes tool+ingest steps
* [ ] `test_pr24_native_scip_mocked_tool.py`

  * mock tool invocation, verify materializer called and artifact refs recorded

### CLI snapshots

* **Command**

  ```bash
  codeintel build graph scip --format mermaid --native-target scip
  ```

  **Snapshot**

  * `pr24_graph_scip_native.mmd`

---

## PR‑25 — Asset catalog + “what exists?” CLI (best-in-class UX)

### Tasks

* [ ] Create `build.assets` persisted table(s):

  * latest materialization per asset_key per repo/commit
  * owner target, schema version, row counts, timestamps
* [ ] Add CLI:

  * `codeintel build assets --format json`
  * `codeintel build assets --target risk_factors`
  * `codeintel build assets --changed-since <commit>` (optional)

### Tests

* [ ] `test_pr25_assets_catalog_persistence.py`

  * after a run, assets rows exist
* [ ] `test_pr25_assets_cli.py`

  * CLI returns deterministic JSON for a tiny run

### CLI snapshots

* **Command**

  ```bash
  codeintel build assets --format json
  ```

  **Snapshot**

  * `pr25_assets_catalog.json`

---

## PR‑26 — Node-level telemetry: `build.run_nodes` + graph metadata enrichment

### Tasks

* [ ] Add `build.run_nodes` table:

  * run_id, node_name, status, duration_ms, error, tags (json)
* [ ] Add Hamilton hook or wrapper so node execution writes run_nodes rows.
* [ ] Enrich graph export JSON to include:

  * node tags (from taxonomy)
  * node_kind, impl_kind
  * asset_key for compute/materialize/dataset/artifact nodes

### Tests

* [ ] `test_pr26_run_nodes_persisted.py`

  * run produces N node records
* [ ] `test_pr26_graph_export_includes_tags.py`

  * exported JSON includes taxonomy tags for nodes

### CLI snapshots

* **Command**

  ```bash
  codeintel build history --run-id hamilton-test-0001 --format json
  ```

  **Snapshot**

  * `pr26_history_with_run_nodes.json`

(If you keep history output stable, add a new CLI like `build run-info` instead and snapshot that.)

---

## PR‑27 — Strict contracts mode + wrapper deprecation gate

### Tasks

* [ ] Add config flag: `strict_contracts=true`

  * if a materializer writes outside declared contract → fail target
* [ ] Add “wrapper shrink” policy:

  * wrappers allowed only for a specific allowlist (temporary)
  * everything else must be native
* [ ] Documentation pass:

  * how to write a native target
  * how to use native registry flags
  * how to interpret run_nodes + assets tables

### Tests

* [ ] `test_pr27_strict_contracts_violations_fail.py`

  * intentionally write outside contract and assert failure
* [ ] `test_pr27_wrapper_allowlist_enforced.py`

  * wrapper target not in allowlist fails planning (or fails run)

### CLI snapshots

* **Command**

  ```bash
  codeintel build plan --all --format json --native-all
  ```

  **Snapshot**

  * `pr27_plan_all_native.json`

---

# Phase 3 Snapshot Manifest Template (YAML)

Add these cases progressively to `tests/build/hamilton/snapshots/manifest.yaml`:

```yaml
# Phase 3 examples
- name: "pr20_plan_risk_factors_native"
  tags: ["pr20", "plan", "native", "json", "tiny"]
  args: ["build", "plan", "risk_factors", "--format", "json", "--native-target", "risk_factors"]
  snapshot: "pr20_plan_risk_factors_native.json"

- name: "pr20_graph_risk_factors_native"
  tags: ["pr20", "graph", "native", "mermaid", "text", "tiny"]
  kind: "text"
  args: ["build", "graph", "risk_factors", "--format", "mermaid", "--native-target", "risk_factors"]
  snapshot: "pr20_graph_risk_factors_native.mmd"
```

Then you can run:

```bash
pytest -m cli_snapshot --cli-snapshot-tags pr20 --update-cli-snapshots
```

---

# Quick guidance on “native vs wrapper” migration pacing

To stay aggressive but keep the system stable:

* **PR‑16 → PR‑19** is the platform foundation (contracts, module split, registry, runner/materializer).
* **PR‑20 → PR‑23** is the high-impact migration (analytics + exports).
* **PR‑24** is the “big swing” (ingestion/tool targets become real Hamilton subgraphs).
* **PR‑25/26/27** is what makes it best-in-class as a platform (catalog + telemetry + strict correctness).

---


