Got it – I’ve updated the plan to target `tests/build` for the new Phase 1 tests.


Below is a Phase 1 implementation plan mapped 1:1 to concrete PRs (PR‑01…PR‑07), using your current `codeintel/build/hamilton` structure as the backbone, and targeting new tests under `tests/build/`.

I’m treating “Phase 1” here as: **make Hamilton a real, end‑to‑end orchestration engine for the entire build graph** (not just the Phase 0 chain), with **CLI parity (force, accurate executed/skipped/failed lists, run tracking)**, and **production‑grade correctness (don’t run downstream if upstream failed)**—building on the Phase 0 foundation you already implemented (dependency DAG, skip-if-unchanged manifests, DatasetRef/Pandera hooks, dynamic node generation). 

---

## PR‑01 — Hamilton “node mode” + correct target↔node mapping everywhere

### Goal

Make node lookup/mapping correct and explicit so **Hamilton can address all targets** reliably (not just those in `targets_phase0.TARGET_TO_NODE`). This is foundational for every later PR.

### Why now?

Right now, `driver_factory.target_to_node_name()` and `list_available_nodes()` are wired to the Phase 0 static mapping. That prevents the generated node path from being a first-class runtime mode.

### Files / functions to touch

**`codeintel/build/hamilton/driver_factory.py`**

* Update:

  * `HamiltonRuntime` dataclass (add mode + mappings)
  * `build_driver(...)`
  * `list_available_nodes(...)`
  * `target_to_node_name(...)` (make it runtime-aware)

**`codeintel/build/hamilton/executor.py`**

* Update:

  * `HamiltonBuildResult.get_record(...)` to use the mapping stored in the result (not a global mapping function)

**`codeintel/build/hamilton/nodes/node_factory.py`**

* Verify/ensure:

  * generated module exposes `TARGET_TO_NODE` for all targets (it already does per your report snippet) 

### Implementation details

#### 1) Add an explicit “mode”

Use a `Literal` or `Enum`:

```python
# codeintel/build/hamilton/driver_factory.py
from typing import Literal

HamiltonNodeMode = Literal["phase0", "generated"]
```

#### 2) Expand the runtime object to carry mappings

```python
# codeintel/build/hamilton/driver_factory.py
@dataclass(frozen=True)
class HamiltonRuntime:
    dr: driver.Driver
    graph: TargetGraph
    mode: HamiltonNodeMode
    target_to_node: dict[str, str]
    node_to_target: dict[str, str]
```

#### 3) Build the mapping based on mode

* `phase0`: use `targets_phase0.TARGET_TO_NODE`
* `generated`: use the generated module’s `TARGET_TO_NODE` (or compute using `target_node()`)

```python
def _build_target_to_node_map(
    graph: TargetGraph,
    *,
    mode: HamiltonNodeMode,
) -> dict[str, str]:
    if mode == "phase0":
        from codeintel.build.hamilton.nodes.targets_phase0 import TARGET_TO_NODE
        return dict(TARGET_TO_NODE)

    # generated
    from codeintel.build.hamilton.nodes.node_factory import get_generated_module
    mod = get_generated_module()
    mapping = getattr(mod, "TARGET_TO_NODE", None)
    if isinstance(mapping, dict) and mapping:
        return dict(mapping)

    # fallback: compute stable names
    from codeintel.build.hamilton.naming import target_node
    return {t.name: target_node(t.name) for t in graph.all_targets}
```

#### 4) Update `build_driver`

```python
def build_driver(
    *,
    config: dict[str, Any] | None = None,
    profile: str | None = None,
    mode: HamiltonNodeMode = "generated",
) -> HamiltonRuntime:
    graph = get_target_graph(profile=profile)

    if mode == "generated":
        from codeintel.build.hamilton.nodes.node_factory import get_generated_module
        nodes_module = get_generated_module()
    else:
        from codeintel.build.hamilton.nodes import targets_phase0 as nodes_module

    dr = driver.Driver(config or {}, nodes_module)

    t2n = _build_target_to_node_map(graph, mode=mode)
    n2t = {v: k for k, v in t2n.items()}
    return HamiltonRuntime(dr=dr, graph=graph, mode=mode, target_to_node=t2n, node_to_target=n2t)
```

#### 5) Make `target_to_node_name` runtime-aware

```python
def target_to_node_name(
    target_name: str,
    *,
    runtime: HamiltonRuntime | None = None,
    mode: HamiltonNodeMode = "generated",
) -> str | None:
    if runtime is not None:
        return runtime.target_to_node.get(target_name)

    # fallback if used outside runtime
    from codeintel.build.hamilton.naming import target_node
    return target_node(target_name) if mode == "generated" else None
```

### Tests to add (`tests/build/...`)

**`tests/build/hamilton/test_driver_factory_mode_mapping.py`**

* `test_build_driver_generated_exposes_mapping_for_many_targets()`

  * `runtime = build_driver(mode="generated")`
  * assert `len(runtime.target_to_node) == len(runtime.graph.all_targets)`
* `test_build_driver_phase0_only_has_phase0_mapping()`

  * `runtime = build_driver(mode="phase0")`
  * assert mapping contains the Phase 0 chain targets, and does **not** contain some known non-phase0 target.

---

## PR‑02 — Execute dependency closure + return complete results (not just requested)

### Goal

When user requests `targets=["risk_factors"]`, Hamilton should:

1. execute the **entire dependency closure**
2. return a result object that has **accurate executed/skipped/failed** lists across the closure
3. enable the CLI adapter to report correctly.

### Files / functions to touch

**`codeintel/build/hamilton/executor.py`**

* Update `HamiltonBuildResult` fields (add `computed_targets`, `skipped_targets`, `failed_targets`, maybe `all_targets`)
* Update `HamiltonBuildExecutor.run(...)` to:

  * compute closure via `TargetGraph.topological_order(...)`
  * request final vars for the full closure (not only requested)

**`codeintel/cli/handlers/build.py`**

* Update `_HamiltonResultAdapter` to use the new result fields rather than iterating only `requested`

### Implementation details

#### 1) Expand targets to closure (topologically ordered)

Your `TargetGraph` already supports this (`topological_order` expands dependencies). Use it.

```python
# codeintel/build/hamilton/executor.py
closure = runtime.graph.topological_order(targets)
```

#### 2) Execute *all* closure target nodes

```python
final_vars = []
missing = []
for t in closure:
    node = target_to_node_name(t, runtime=runtime)
    if node is None:
        missing.append(t)
    else:
        final_vars.append(node)

if missing:
    return HamiltonBuildResult(
        requested=tuple(targets),
        outputs={},
        success=False,
        failed_targets=tuple(missing),
        # include computed/skipped empty
    )

outputs = runtime.dr.execute(
    final_vars=final_vars,
    inputs={"env": env, "graph": runtime.graph},
)
```

#### 3) Derive computed/skipped/failed from returned `TargetRunRecord`s

```python
computed: list[str] = []
skipped: list[str] = []
failed: list[str] = []

for t in closure:
    node = target_to_node_name(t, runtime=runtime)
    rec = outputs.get(node) if node else None
    if not isinstance(rec, TargetRunRecord):
        failed.append(t)
        continue
    if rec.status == "succeeded":
        computed.append(t)
    elif rec.status == "skipped":
        skipped.append(t)
    else:
        failed.append(t)
```

#### 4) Update the CLI adapter to use the lists

```python
# codeintel/cli/handlers/build.py
class _HamiltonResultAdapter:
    def __init__(self, result: HamiltonBuildResult): ...
    def completed_targets(self) -> tuple[str, ...]:
        return result.computed_targets
    def skipped_targets(self) -> tuple[str, ...]:
        return result.skipped_targets
    def failed_targets(self) -> tuple[str, ...]:
        return result.failed_targets
```

### Tests to add (`tests/build/...`)

**`tests/build/hamilton/test_executor_closure_results.py`**

* Build a tiny `TargetGraph` in-test (A→B→C), patch `build_driver()` to return:

  * that graph
  * a fake `dr.execute(...)` returning `TargetRunRecord`s for `t__A`, `t__B`, `t__C`
* Assert:

  * `result.computed_targets == ("A","B","C")` when all succeed
  * requested only includes the requested target, but computed includes closure

---

## PR‑03 — Correctness: do not run a target if any upstream target failed

### Goal

Match the legacy planner semantics: **if upstream fails, downstream must not execute** (otherwise you risk stale DB reads + nonsensical “success” runs).

### Where to implement?

You already pass upstream records into node functions (generated nodes accept `**deps_records`; phase0 nodes accept explicit `t__...` args) but currently ignore them.

### Files / functions to touch

**`codeintel/build/hamilton/nodes/targets_phase0.py`**

* `_run_target(...)`: accept upstream records and enforce “upstream failure gating”
* Update all `t__...` functions to pass their upstream records into `_run_target(...)`

**`codeintel/build/hamilton/nodes/node_factory.py`**

* In `_make_node_function(...)`, pass upstream records into `_run_target(...)`

### Implementation details

#### 1) Add upstream gating in `_run_target`

```python
# codeintel/build/hamilton/nodes/targets_phase0.py
def _run_target(
    *,
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    upstream: tuple[TargetRunRecord, ...] = (),
) -> TargetRunRecord:
    failed_upstream = tuple(r.target for r in upstream if r.status == "failed")
    if failed_upstream:
        meta = from_plugin_or_target(plugin=..., target=graph.get(target_name))
        return TargetRunRecord(
            target=target_name,
            plugin_name=meta.name,
            status="skipped",
            input_hash=None,
            options_hash=None,
            duration_ms=0.0,
            row_counts={},
            error=f"upstream_failed:{failed_upstream}",
        )

    # ... existing skip/execute logic ...
```

#### 2) Wire upstream into generated nodes

```python
# codeintel/build/hamilton/nodes/node_factory.py
def _make_node_function(target_name: str, deps: tuple[str, ...], ...):
    def node_fn(env: BuildEnv, graph: TargetGraph, **deps_records: object) -> TargetRunRecord:
        upstream = tuple(
            r for r in deps_records.values()
            if isinstance(r, TargetRunRecord)
        )
        return _run_target(env=env, graph=graph, target_name=target_name, upstream=upstream)
```

#### 3) Wire upstream into phase0 explicit nodes

Example:

```python
def t__call_graph(env: BuildEnv, graph: TargetGraph, t__goids: TargetRunRecord) -> TargetRunRecord:
    return _run_target(env=env, graph=graph, target_name="call_graph", upstream=(t__goids,))
```

### Tests to add (`tests/build/...`)

**`tests/build/hamilton/test_upstream_failure_gating.py`**

* Patch the internal plugin execution path so nothing actually runs.
* Create a fake upstream `TargetRunRecord(status="failed")`
* Call `_run_target(... upstream=(failed,))`
* Assert:

  * returned status is `"skipped"`
  * error contains `"upstream_failed"`

---

## PR‑04 — CLI parity: implement `--force` for Hamilton (skip bypass)

### Goal

If the user passes `--force target_x`, Hamilton must recompute it even if manifests say it’s fresh.

### Files / functions to touch

**`codeintel/build/hamilton/env.py`**

* Add `force_targets: frozenset[str] = frozenset()`

**`codeintel/cli/handlers/build.py`**

* When constructing `BuildEnv`, pass `force_targets=frozenset(force or ())`

**`codeintel/build/hamilton/nodes/targets_phase0.py`**

* In `_run_target`, bypass `should_skip(...)` if target in `env.force_targets`

### Implementation details

#### 1) Extend BuildEnv

```python
# codeintel/build/hamilton/env.py
@dataclass(frozen=True)
class BuildEnv:
    ...
    force_targets: frozenset[str] = frozenset()
```

#### 2) Pass it from CLI handler

```python
# codeintel/cli/handlers/build.py inside _execute_build_hamilton(...)
env = BuildEnv(
    gateway=gateway,
    snapshot=runtime.snapshot,
    paths=runtime.paths,
    providers=runtime.providers,
    config=runtime.config,
    profile=runtime.profile,
    force_targets=frozenset(force or ()),
)
```

#### 3) Bypass skip

```python
if target_name not in env.force_targets and should_skip(...):
    return TargetRunRecord(... status="skipped" ...)
```

### Tests to add (`tests/build/...`)

**`tests/build/hamilton/test_force_bypasses_skip.py`**

* Monkeypatch `should_skip` to always return `True`
* Provide `env.force_targets={target_name}`
* Assert result status is not `"skipped"` (i.e. it attempts execution; in test you can patch plugin execution to “success”)

---

## PR‑05 — Run tracking parity: start/complete `build.runs` for Hamilton executions

### Goal

A Hamilton build should show up in build history/status the same way legacy builds do.

### Files / functions to touch

**`codeintel/build/hamilton/executor.py`**

* In `HamiltonBuildExecutor.run(...)`:

  * generate a run_id
  * `gateway.build.start_run(...)`
  * `gateway.build.complete_run(...)` with computed/skipped lists + error summary
  * set duration_ms on completion

**`codeintel/build/hamilton/env.py`** (optional)

* Optionally carry run_id (not required; can stay executor-local)

### Implementation details

Hamilton doesn’t prevent you from doing this; you already have `StorageGateway` in `BuildEnv`.

Pseudo-code:

```python
from datetime import UTC, datetime
from codeintel.build.manifest import BuildRunRecord

run_id = _generate_run_id()
env.gateway.build.start_run(
    BuildRunRecord(
        run_id=run_id,
        repo=env.repo,
        commit=env.commit,
        requested_targets=tuple(targets),
        computed_targets=(),
        skipped_targets=(),
        started_at=datetime.now(tz=UTC),
        status="running",
    )
)

start = datetime.now(tz=UTC)
try:
    # execute closure (PR-02)
    ...
    status = "failed" if failed else "succeeded"
    error_summary = f"{len(failed)} targets failed" if failed else None
finally:
    duration_ms = (datetime.now(tz=UTC) - start).total_seconds() * 1000
    env.gateway.build.complete_run(
        run_id=run_id,
        status=status,
        computed_targets=tuple(computed),
        skipped_targets=tuple(skipped),
        error_summary=error_summary,
    )
```

### Tests to add (`tests/build/...`)

**`tests/build/hamilton/test_run_tracking.py`**

* Use `codeintel.storage.gateway.factory.create_in_memory_gateway_for_tests(...)`
* Patch `build_driver()` to return a stub driver + stub graph
* Run `HamiltonBuildExecutor.run(...)`
* Assert:

  * `gateway.build.fetch_run(run_id)` exists
  * computed/skipped stored match what you returned

---

## PR‑06 — Dataset lineage scale-out: auto-populate `TargetRunRecord.datasets` + generate dataset nodes for *all* contract tables

### Goal

Turn DatasetRef/lineage into a first-class, universal feature:

* every successful target run returns `TargetRunRecord.datasets` populated from the target contract
* generated Hamilton module also defines `d__...` dataset nodes for each contract table (not just the Phase 0 hardcoded ones) 

### Files / functions to touch

**`codeintel/build/hamilton/nodes/targets_phase0.py`**

* In `_run_target` success path:

  * derive `table_keys = target.contract.table_keys`
  * set `datasets=tuple(refs_from_target_result(...).values())`

**`codeintel/build/hamilton/nodes/node_factory.py`**

* Extend `build_target_module(...)` to also generate dataset nodes for each table in each target contract.

**`codeintel/build/hamilton/nodes/dataset_nodes.py`**

* Either:

  * keep as “Phase 0 examples” (fine), OR
  * refactor to be purely helper utilities and stop adding new hardcoded nodes.

### Implementation details

#### 1) Populate datasets on success

```python
# codeintel/build/hamilton/nodes/targets_phase0.py
from codeintel.build.hamilton.io.dataset_ref import refs_from_target_result, refs_to_tuple

table_keys = target.contract.table_keys
refs = refs_from_target_result(
    target_name=target_name,
    table_keys=table_keys,
    row_counts=dict(row_counts),
)
datasets = refs_to_tuple(refs)

return TargetRunRecord(
    ...,
    row_counts=row_counts,
    datasets=datasets,
)
```

#### 2) Generate dataset nodes in `node_factory`

Add a second generator pass:

```python
# codeintel/build/hamilton/nodes/node_factory.py
from codeintel.build.hamilton.naming import dataset_node, target_node
from codeintel.build.hamilton.io.dataset_ref import DatasetRef

def _make_dataset_node_function(*, table_key: str, target_name: str) -> Callable[..., DatasetRef]:
    d_name = dataset_node(table_key)           # e.g. d__analytics__hotspots
    t_name = target_node(target_name)          # e.g. t__hotspots

    def fn(**kwargs: object) -> DatasetRef:
        rec = kwargs[t_name]
        assert isinstance(rec, TargetRunRecord)
        ds = rec.get_dataset(table_key)
        if ds is None:
            # “missing dataset” is a contract violation; choose fail-fast or raise
            raise ValueError(f"Missing DatasetRef for {table_key} from {target_name}")
        return ds

    fn.__name__ = d_name
    fn.__module__ = "codeintel.build.hamilton.nodes.generated"
    fn.__signature__ = inspect.Signature(
        parameters=[
            inspect.Parameter(t_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=TargetRunRecord),
        ],
        return_annotation=DatasetRef,
    )
    return tag(domain=table_key.split(".", 1)[0], table=table_key)(fn)
```

…and register them into the dynamically built module.

### Tests to add (`tests/build/...`)

**`tests/build/hamilton/test_datasets_populated.py`**

* Create a fake `OutputTarget` with `contract.tables=(TableSchema(...),)`
* Patch plugin execution to return success + row_counts
* Assert `record.get_dataset("schema.table")` exists and `row_count` matches.

**`tests/build/hamilton/test_dataset_nodes_generated.py`**

* Build a tiny graph with one target producing one table
* Call `build_target_module()`
* Assert:

  * module has attribute `d__schema__table`
  * calling it with `{ "t__target": record }` returns the DatasetRef

---

## PR‑07 — Observability + UX: export/visualize DAG for a build request, plus docs

### Goal

Provide developer tooling to:

* inspect what Hamilton will execute
* export a DAG representation for troubleshooting / PR review

Hamilton’s `Driver.execute(...)` and `Driver.export_execution(...)` are explicitly designed for this kind of use. ([hamilton.apache.org][1])

### Files / functions to touch

**`codeintel/build/hamilton/driver_factory.py`** (or a new `observability.py`)

* Add helper:

  * `export_execution_json(runtime, targets, env) -> str`

**`codeintel/cli/commands/build.py`** and/or **`codeintel/cli/handlers/build.py`**

* Add a subcommand, e.g.:

  * `codeintel build graph --targets ... --format json --out dag.json`
  * (or `codeintel build run --export-dag dag.json`)

### Implementation details

```python
# codeintel/build/hamilton/observability.py
def export_execution_json(
    runtime: HamiltonRuntime,
    *,
    targets: list[str],
    env: BuildEnv,
) -> str:
    closure = runtime.graph.topological_order(targets)
    final_vars = [runtime.target_to_node[t] for t in closure]
    return runtime.dr.export_execution(
        final_vars=final_vars,
        inputs={"env": env, "graph": runtime.graph},
        overrides=None,
    )
```

Per the Hamilton driver reference, `export_execution(...)` returns a JSON string representation of the graph for the requested `final_vars`. ([hamilton.apache.org][1])

### Tests to add (`tests/build/...`)

**`tests/build/hamilton/test_export_execution_json.py`**

* Create a runtime (or stub driver with `export_execution`)
* Assert returned string parses as JSON and includes expected node names.

---

# Cross-PR notes / sequencing constraints

* **PR‑01 must land before anything else** (it makes mode/mapping explicit).
* **PR‑02 + PR‑03 are “correctness blockers”** for trusting Hamilton output (closure completeness + no downstream after upstream failure).
* **PR‑04/PR‑05 deliver CLI parity** (force + run tracking).
* **PR‑06 universalizes dataset lineage** beyond the Phase 0 hardcoded examples. 
* **PR‑07 is optional but very high leverage** for adoption/debugging; Hamilton explicitly supports `execute(final_vars, inputs=...)` and `export_execution(...)` for this purpose. ([hamilton.apache.org][1])

---

## One concrete “Phase 1 definition of done”

You can treat Phase 1 as complete when:

1. `codeintel build run --engine hamilton --all` executes successfully for a real repo (not just the Phase 0 chain)
2. `--force` forces recompute of a previously “fresh” target
3. build history/status shows Hamilton runs (via `build.runs`)
4. downstream targets do not run when upstream fails
5. `TargetRunRecord.datasets` is populated for successful targets, and dataset nodes can be generated for all contract tables

---


[1]: https://hamilton.apache.org/reference/drivers/Driver/ "Builder - Hamilton"
