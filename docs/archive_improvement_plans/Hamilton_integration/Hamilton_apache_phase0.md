Below is a **concrete “Phase 0 Hamilton integration shape”** that fits your repo as‑is, **doesn’t require rewriting plugins**, and still cleanly sets you up for the “policy + metadata‑driven unified engine” end‑state.

It’s structured as:

1. **Repo/module layout** (what you add)

2. **How plugin metadata becomes Hamilton nodes** (exact mechanism)

3. **How DuckDB + Ibis materialization plugs in** (minimal but real)

4. **How to wire it into build + CLI without a big‑bang rewrite**

5. **Phase‑0 acceptance tests / health checks**

Throughout, the guiding principle is:

> **Hamilton owns dependency ordering + observability; your plugins still own computation + writes (for now).**
> We use Hamilton’s hooks to layer in contracts, profiles, caching/skip, and materialization gradually.

---

## 0) Phase‑0 scope decisions (explicit)

### Phase‑0 “yes”

* Use Hamilton as an **execution orchestrator** for a *subset path* first (e.g. `risk_factors` chain).
* Wrap each existing target plugin execution as a Hamilton node.
* Keep **existing plugin code paths** (including their DB writes).
* Add **a small bridge layer** that:

  * reads **canonical metadata** (or falls back to current target definitions),
  * resolves options,
  * computes hashing/skip decisions,
  * records execution (manifest/records),
  * emits “dataset refs” for lineage.

### Phase‑0 “not yet”

* Don’t attempt to turn every plugin into “pure functions returning DataFrames/Ibis expressions” immediately.
* Don’t replace your entire config system immediately.
* Don’t migrate *all* build targets to Hamilton on day 1 (we build a pattern that scales to that).

---

## 1) Proposed module layout (additive, safe)

Add a new “Hamilton bridge” package in build:

```
src/codeintel/build/hamilton/
  __init__.py
  executor.py                 # HamiltonBuildExecutor (Phase 0 executor)
  driver_factory.py           # builds the Hamilton Driver + adapters
  naming.py                   # stable node/dataset naming conventions
  metadata_bridge.py          # unify plugin metadata source of truth
  node_factory.py             # generate nodes from metadata/targets
  nodes/
    __init__.py
    env.py                    # provides BuildEnv input types
    targets_phase0.py         # explicit nodes for a first chain (fastest to land)
  io/
    __init__.py
    duckdb_ibis_adapter.py    # DataLoader/DataSaver for DuckDB/Ibis
  hooks/
    __init__.py
    manifest_hook.py          # skip/rehash + execution recording
    dq_hook.py                # Pandera @check_output integration (optional)
```

Why both `nodes/targets_phase0.py` **and** `node_factory.py`?

* `targets_phase0.py` lets you land Phase‑0 quickly with **fully type‑checked, explicit nodes**.
* `node_factory.py` is the path to scale to “all targets generated from metadata” once you like the pattern.

---

## 2) Canonical naming rules (critical for Hamilton)

Hamilton node names must be valid Python identifiers, but your “logical IDs” likely aren’t (dots, slashes, etc.). So Phase‑0 should standardize naming so that:

* metadata remains stable (`analytics.function_metrics`)
* Hamilton nodes are stable (`t__analytics__function_metrics`)

**`naming.py`**

```python
# src/codeintel/build/hamilton/naming.py
from __future__ import annotations

import re

def to_node_name(logical_name: str, *, prefix: str) -> str:
    """
    Convert stable logical ids like:
      analytics.function_metrics
      graph.call_graph_edges
    into Hamilton node identifiers like:
      t__analytics__function_metrics
    """
    s = logical_name.strip()
    s = s.replace(".", "__").replace("-", "_").replace("/", "__")
    s = re.sub(r"[^a-zA-Z0-9_]", "_", s)
    return f"{prefix}__{s}"

def target_node(target_name: str) -> str:
    # targets are already snake_case but keep consistent prefixing
    return to_node_name(target_name, prefix="t")

def dataset_node(dataset_key: str) -> str:
    return to_node_name(dataset_key, prefix="d")
```

This is the keystone that makes “metadata → nodes” scalable.

---

## 3) Metadata bridge: one way to ask “what is this thing?”

Phase‑0 must not depend on “did we fully migrate metadata yet?”
So we implement a small bridge:

* If a plugin exposes canonical metadata (your new common format): use it.
* Else fall back to `OutputTarget` info (what build already knows).

**`metadata_bridge.py`**

```python
# src/codeintel/build/hamilton/metadata_bridge.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Type

from codeintel.build.targets import OutputTarget

@dataclass(frozen=True)
class CanonicalPluginMeta:
    # stable id for policy/config/lineage
    name: str                 # e.g. "analytics.function_metrics" or "graphs.call_graph"
    version: str              # plugin version string
    domain: str               # "ingestion"|"graphs"|"analytics"
    description: str

    # dependency semantics (capabilities or targets)
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    # dataset IO contracts
    produces_tables: tuple[str, ...] = ()
    consumes_tables: tuple[str, ...] = ()

    # options
    options_model: Type[Any] | None = None

def from_target(target: OutputTarget) -> CanonicalPluginMeta:
    # Phase-0 fallback: derive from target graph knowledge
    return CanonicalPluginMeta(
        name=f"{target.module}.{target.name}",          # stable-ish
        version="0.0.0",                                # until plugins expose it
        domain=target.module,
        description=f"Target {target.name} ({target.module})",
        requires=tuple(f"{target.module}.{d}" for d in target.dependencies),
        provides=(f"{target.module}.{target.name}",),
        produces_tables=tuple(target.contract.tables.keys()),
        consumes_tables=(),  # can be enriched later
        options_model=None,  # can be enriched later
    )

def from_plugin_or_target(*, plugin: Any, target: OutputTarget) -> CanonicalPluginMeta:
    meta = getattr(plugin, "metadata", None)
    if meta is not None:
        # Expect your unified metadata shape
        return CanonicalPluginMeta(
            name=meta.name,
            version=meta.version,
            domain=meta.domain,
            description=meta.description,
            requires=tuple(meta.requires),
            provides=tuple(meta.provides),
            produces_tables=tuple(meta.produces_tables),
            consumes_tables=tuple(meta.consumes_tables),
            options_model=getattr(meta, "options_model", None),
        )
    return from_target(target)
```

This gives you one answer to:

> “what’s the identity, IO contract, and config model of this unit of work?”

That’s your “metadata brain” entry point.

---

## 4) The key Phase‑0 object: a single “BuildEnv” input

Hamilton works best when your nodes are **pure functions** of upstream values + inputs.
We won’t make plugins pure yet, but we’ll make the *orchestration interface* pure by passing one `BuildEnv`.

**`nodes/env.py`**

```python
# src/codeintel/build/hamilton/nodes/env.py
from __future__ import annotations

from dataclasses import dataclass
from codeintel.config.primitives import SnapshotRef, BuildPaths
from codeintel.storage.gateway import StorageGateway
from codeintel.build.providers import Providers
from codeintel.build.config import BuildConfig

@dataclass(frozen=True)
class BuildEnv:
    gateway: StorageGateway
    snapshot: SnapshotRef
    paths: BuildPaths
    providers: Providers
    config: BuildConfig
    profile: str | None = None  # policy profile name: "fast"|"full"|...
```

The Hamilton driver will be invoked with:

```python
dr.execute([target_node("risk_factors")], inputs={"env": env})
```

---

## 5) Manifest + hashing + skip logic: implemented once (Phase‑0)

Hamilton has its own caching/data versioning capabilities, but in Phase‑0 you should:

* keep your current manifest tables,
* or use your new core manifest utilities.

Your Hamilton nodes can implement a consistent “skip if unchanged” check.

Your Hamilton overview doc highlights that Hamilton has **content-based caching keyed by code + data versions** (including `node_name`, `code_version`, `data_version`) and pluggable stores, which you can adopt later once the bridge is stable. 

### Phase‑0 approach

Implement skip inside node execution:

**`hooks/manifest_hook.py`**

```python
# src/codeintel/build/hamilton/hooks/manifest_hook.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping

from codeintel.build.hashing import compute_options_hash
from codeintel.build.manifest import OutputManifest
from codeintel.core.plugins.execution.manifest import InputHashPayload, compute_input_hash as core_input_hash
from codeintel.storage.tracking.build_tracking import BuildTracking

@dataclass(frozen=True)
class TargetRunRecord:
    target: str
    plugin_name: str
    status: str               # "succeeded"|"failed"|"skipped"
    input_hash: str | None
    options_hash: str | None
    duration_ms: float
    row_counts: Mapping[str, int] = ()
    error: str | None = None

def compute_target_input_hash(
    *,
    repo: str,
    commit: str,
    plugin_name: str,
    plugin_version: str | None,
    options_hash: str | None,
    upstream_hashes: Mapping[str, str | None],
) -> str:
    payload = InputHashPayload(
        repo=repo,
        commit=commit,
        plugin_name=plugin_name,
        version_hash=plugin_version,
        options_hash=options_hash,
        extra_fields={"upstream": dict(sorted(upstream_hashes.items()))},
    )
    return core_input_hash(payload)

def should_skip(
    *,
    tracking: BuildTracking,
    target: str,
    repo: str,
    commit: str,
    input_hash: str,
) -> bool:
    prior = tracking.load_manifest(target=target, repo=repo, commit=commit)
    return prior is not None and prior.input_hash == input_hash

def save_manifest(
    *,
    tracking: BuildTracking,
    target: str,
    repo: str,
    commit: str,
    plugin: str,
    duration_ms: float,
    input_hash: str,
    row_count: int | None,
    options_hash: str | None,
) -> None:
    tracking.save_manifest(
        OutputManifest(
            target=target,
            repo=repo,
            commit=commit,
            plugin=plugin,
            computed_at=datetime.now(tz=UTC),
            duration_ms=duration_ms,
            input_hash=input_hash,
            row_count=row_count,
            options_hash=options_hash,
        )
    )
```

This is intentionally “Phase‑0”: it reuses your manifest tables, but the hash payload is already shaped like the unified core hashing.

---

## 6) Phase‑0 Hamilton nodes: wrap existing plugins (explicit, type-safe)

This is the “fastest to land” part: create a module defining nodes for one end-to-end chain.

Pick a chain that spans all domains. A great one is:

* ingestion: `modules`, `scip`, `ast`
* graphs: `goids`, `call_graph`
* analytics: `function_metrics`, `risk_factors`

(Your target graph already expresses dependencies.)

### `nodes/targets_phase0.py`

```python
# src/codeintel/build/hamilton/nodes/targets_phase0.py
from __future__ import annotations

import asyncio
import time
from typing import Mapping

from hamilton.function_modifiers import tag

from codeintel.build.context import ContextResources, TargetExecutionContext
from codeintel.build.plugin_registry import get_plugin_for_target
from codeintel.build.targets import TargetGraph
from codeintel.build.hamilton.nodes.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import (
    TargetRunRecord,
    compute_target_input_hash,
    save_manifest,
    should_skip,
)
from codeintel.build.hamilton.metadata_bridge import from_plugin_or_target
from codeintel.build.hamilton.naming import target_node

def _run_target(*, env: BuildEnv, graph: TargetGraph, target_name: str, upstream: Mapping[str, TargetRunRecord]) -> TargetRunRecord:
    target = graph.get(target_name)
    plugin = get_plugin_for_target(target_name)
    meta = from_plugin_or_target(plugin=plugin, target=target)

    # Options hashing: phase-0 can use current build config parameters_for(target)
    raw_params = env.config.parameters_for(target_name)
    options_hash = None
    if raw_params:
        options_hash = compute_options_hash(raw_params)

    # Upstream state signature
    upstream_hashes = {k: v.input_hash for k, v in upstream.items()}

    # Compute a stable input hash (repo/commit/plugin/version/options/upstream)
    input_hash = compute_target_input_hash(
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        plugin_name=meta.name,
        plugin_version=meta.version,
        options_hash=options_hash,
        upstream_hashes=upstream_hashes,
    )

    tracking = env.gateway.build  # BuildTracking accessor
    if should_skip(tracking=tracking, target=target_name, repo=env.snapshot.repo, commit=env.snapshot.commit, input_hash=input_hash):
        return TargetRunRecord(
            target=target_name,
            plugin_name=meta.name,
            status="skipped",
            input_hash=input_hash,
            options_hash=options_hash,
            duration_ms=0.0,
            row_counts={},
        )

    # Build existing TargetExecutionContext exactly like BuildExecutor does
    resources = ContextResources(
        providers=env.providers,
        gateway=env.gateway,
        modules=(),  # load lazily from DB if needed
    )
    ctx = TargetExecutionContext(
        target=target,
        snapshot=env.snapshot,
        paths=env.paths,
        resources=resources,
        parameters=raw_params,
    )

    start = time.perf_counter()
    try:
        result = asyncio.run(plugin.execute(ctx))
    except Exception as e:  # noqa: BLE001
        dur_ms = (time.perf_counter() - start) * 1000
        return TargetRunRecord(
            target=target_name,
            plugin_name=meta.name,
            status="failed",
            input_hash=input_hash,
            options_hash=options_hash,
            duration_ms=dur_ms,
            row_counts={},
            error=str(e),
        )

    dur_ms = (time.perf_counter() - start) * 1000
    row_counts = dict(result.row_counts or {})

    # Persist build manifest
    save_manifest(
        tracking=tracking,
        target=target_name,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        plugin=target.plugin,
        duration_ms=dur_ms,
        input_hash=input_hash,
        row_count=sum(row_counts.values()) if row_counts else None,
        options_hash=options_hash,
    )

    status = "succeeded" if result.success else "failed"
    return TargetRunRecord(
        target=target_name,
        plugin_name=meta.name,
        status=status,
        input_hash=input_hash,
        options_hash=options_hash,
        duration_ms=dur_ms,
        row_counts=row_counts,
        error=result.error_message if not result.success else None,
    )

# ---- Now define explicit nodes (phase-0) ----

@tag(domain="ingestion", target="modules")
def t__modules(env: BuildEnv, graph: TargetGraph) -> TargetRunRecord:
    return _run_target(env=env, graph=graph, target_name="modules", upstream={})

@tag(domain="ingestion", target="scip")
def t__scip(env: BuildEnv, graph: TargetGraph, t__modules: TargetRunRecord) -> TargetRunRecord:
    return _run_target(env=env, graph=graph, target_name="scip", upstream={"modules": t__modules})

@tag(domain="ingestion", target="ast")
def t__ast(env: BuildEnv, graph: TargetGraph, t__modules: TargetRunRecord) -> TargetRunRecord:
    return _run_target(env=env, graph=graph, target_name="ast", upstream={"modules": t__modules})

@tag(domain="graphs", target="goids")
def t__goids(env: BuildEnv, graph: TargetGraph, t__ast: TargetRunRecord) -> TargetRunRecord:
    return _run_target(env=env, graph=graph, target_name="goids", upstream={"ast": t__ast})

@tag(domain="graphs", target="call_graph")
def t__call_graph(env: BuildEnv, graph: TargetGraph, t__goids: TargetRunRecord, t__scip: TargetRunRecord) -> TargetRunRecord:
    return _run_target(env=env, graph=graph, target_name="call_graph", upstream={"goids": t__goids, "scip": t__scip})

@tag(domain="analytics", target="function_metrics")
def t__function_metrics(env: BuildEnv, graph: TargetGraph, t__goids: TargetRunRecord, t__ast: TargetRunRecord) -> TargetRunRecord:
    return _run_target(env=env, graph=graph, target_name="function_metrics", upstream={"goids": t__goids, "ast": t__ast})

@tag(domain="analytics", target="risk_factors")
def t__risk_factors(env: BuildEnv, graph: TargetGraph, t__function_metrics: TargetRunRecord, t__call_graph: TargetRunRecord) -> TargetRunRecord:
    return _run_target(
        env=env,
        graph=graph,
        target_name="risk_factors",
        upstream={"function_metrics": t__function_metrics, "call_graph": t__call_graph},
    )
```

This gives you a *real, executable* “Phase‑0” Hamilton DAG without touching plugin internals.

### Why `@tag` now?

Because later you can:

* filter nodes by domain in Hamilton UI
* attach policy metadata
* drive contract checks by tags

Hamilton supports tagging and makes those tags queryable in tooling/UI. 

---

## 7) Driver factory: build a Hamilton Driver from your repo config

Hamilton supports:

* config-driven DAG variants (`@config.when(...)`) 
* IO modifiers via `@dataloader/@datasaver` + materialization 

Phase‑0 driver factory just wires the modules and passes config.

**`driver_factory.py`**

```python
# src/codeintel/build/hamilton/driver_factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hamilton import driver

from codeintel.build.registry import get_target_graph
from codeintel.build.hamilton.nodes import targets_phase0

@dataclass(frozen=True)
class HamiltonRuntime:
    dr: driver.Driver
    graph: Any  # TargetGraph

def build_driver(*, config: dict[str, Any]) -> HamiltonRuntime:
    graph = get_target_graph()
    dr = driver.Driver(config=config, modules=[targets_phase0])
    return HamiltonRuntime(dr=dr, graph=graph)
```

You’ll later extend this to attach:

* caching adapters
* lifecycle hooks
* result builders

Your overview doc notes Hamilton supports rich lifecycle hooks and adapters (progress bars, tracing, lineage, etc.). 

---

## 8) Phase‑0 HamiltonBuildExecutor (drop-in for build executor)

**`executor.py`**

```python
# src/codeintel/build/hamilton/executor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.nodes.env import BuildEnv

@dataclass(frozen=True)
class HamiltonBuildResult:
    requested: tuple[str, ...]
    outputs: dict[str, Any]

class HamiltonBuildExecutor:
    def __init__(self, *, profile: str | None = None) -> None:
        self._profile = profile

    def run(self, *, env: BuildEnv, targets: list[str]) -> HamiltonBuildResult:
        # Phase-0: map requested targets -> phase0 nodes by name
        # For now we only support the explicit chain (e.g. risk_factors).
        config = {"profile": self._profile or "default"}

        runtime = build_driver(config=config)

        # Provide TargetGraph as input too (so nodes can look up OutputTarget)
        outputs = runtime.dr.execute(
            final_vars=[f"t__{t}" for t in targets],
            inputs={"env": env, "graph": runtime.graph},
        )
        return HamiltonBuildResult(requested=tuple(targets), outputs=outputs)
```

**CLI integration** (minimal):

* Add `--engine legacy|hamilton` option to `codeintel build run`
* If `hamilton`, call this executor instead of `BuildExecutor`

---

## 9) DuckDB + Ibis materialization (Phase‑0 minimal, but sets the standard)

Right now, your Phase‑0 nodes “materialize” by side effect (plugins write to DuckDB).
To align with your *future architecture* (datasets first-class), Phase‑0 should introduce:

* `DatasetRef` values in the DAG
* `@dataloader/@datasaver` adapters so Hamilton can treat datasets as first-class artifacts 

### 9.1 DatasetRef type

```python
# src/codeintel/build/hamilton/io/duckdb_ibis_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class DatasetRef:
    table: str                   # "graph.call_graph_edges"
    schema_version: str | None = None
    meta: dict[str, Any] | None = None
```

### 9.2 Create “dataset nodes” from a plugin run (output shaping)

Your Hamilton overview doc calls out `@extract_fields` as a strong way to split composite outputs into nodes. 

So we can have each target node return a dict `{table_name: DatasetRef}` and then split them with `@extract_fields`.

Example pattern for one target:

```python
from hamilton.function_modifiers import extract_fields
from codeintel.build.hamilton.io.duckdb_ibis_adapter import DatasetRef
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord

@extract_fields(
    {"graph__call_graph_edges": DatasetRef, "graph__call_graph_nodes": DatasetRef}
)
def d__call_graph_datasets(t__call_graph: TargetRunRecord) -> dict[str, DatasetRef]:
    # The plugin already wrote tables; we just return refs for lineage
    return {
        "graph__call_graph_edges": DatasetRef("graph.call_graph_edges"),
        "graph__call_graph_nodes": DatasetRef("graph.call_graph_nodes"),
    }
```

Now Hamilton has explicit dataset artifacts in the DAG even though the plugin wrote them.

### 9.3 Add true Ibis loaders/savers (optional in Phase‑0)

Hamilton supports IO modifiers that annotate load/save semantics for materialization. 

Phase‑0 can implement an adapter that:

* loads a table into an Ibis expression (or dataframe)
* saves an Ibis expression to DuckDB

Skeleton:

```python
# src/codeintel/build/hamilton/io/duckdb_ibis_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ibis
import ibis.expr.types as it

from hamilton.function_modifiers import dataloader, datasaver

@dataclass(frozen=True)
class DuckDBIbisConfig:
    db_path: str

def _ibis_backend(cfg: DuckDBIbisConfig):
    # if you already have an ibis gateway wrapper, use that instead
    return ibis.duckdb.connect(cfg.db_path)

@dataloader()
def load_duckdb_table(table: str, duckdb_cfg: DuckDBIbisConfig) -> tuple[it.Table, dict[str, Any]]:
    backend = _ibis_backend(duckdb_cfg)
    t = backend.table(table)
    return t, {"source": "duckdb", "table": table}

@datasaver()
def save_duckdb_table(output: it.Table, table: str, duckdb_cfg: DuckDBIbisConfig) -> dict[str, Any]:
    backend = _ibis_backend(duckdb_cfg)
    # In a “real” impl: create/replace table, handle schema enforcement, etc.
    backend.create_table(table, output, overwrite=True)
    return {"saved_to": "duckdb", "table": table}
```

Even if you don’t use it everywhere immediately, this establishes the “standard interface” you wanted.

### 9.4 Pandera checks (Phase‑0 optional, but recommended)

Your Hamilton overview doc notes Hamilton can attach data quality checks directly to node outputs with `@check_output` and can integrate with Pandera. 

So you can put Pandera validation at:

* dataset nodes (after load)
* derived view nodes
* outputs intended for serving

This makes “contracts live with computation,” which is exactly what you want as you move toward schema-driven DAGs.

---

## 10) Policy profiles in Phase‑0 (don’t overbuild yet)

Hamilton supports config-driven DAG variants (`@config.when(...)`) so you can select “fast” vs “full” implementations without branching orchestration code. 

Phase‑0 **does not need** to introduce DAG variants for everything. But you should:

* pass `profile` into Hamilton driver config
* start reading `env.profile` in nodes to adjust options resolution

Example:

```python
def resolved_profile(env: BuildEnv) -> str:
    return env.profile or "default"
```

Later you can do:

```python
from hamilton.function_modifiers import config

@config.when(profile="fast")
def t__call_graph(...): ...

@config.when(profile="full")
def t__call_graph(...): ...
```

But Phase‑0 can simply adjust options.

---

## 11) Scaling beyond Phase‑0: generate nodes from metadata (the real “move”)

Once the explicit Phase‑0 chain works, you can scale to “all targets” by generating nodes from the target graph + canonical metadata.

Hamilton can support this because its “nodes are functions”; you can dynamically build a module that contains one function per target.

**`node_factory.py` (sketch)**

```python
# src/codeintel/build/hamilton/node_factory.py
from __future__ import annotations

import inspect
from types import ModuleType
from typing import Callable

from codeintel.build.registry import get_target_graph
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.nodes.env import BuildEnv
from codeintel.build.hamilton.naming import target_node
from codeintel.build.hamilton.nodes.targets_phase0 import _run_target  # reuse core runner

def build_target_module() -> ModuleType:
    graph = get_target_graph()
    m = ModuleType("codeintel.build.hamilton.nodes.targets_generated")

    for target in graph.all_targets:
        node_name = target_node(target.name)

        dep_node_names = [target_node(dep) for dep in target.dependencies]

        def _fn_factory(target_name: str, dep_names: list[str]) -> Callable[..., TargetRunRecord]:
            def _node(**kwargs):
                env: BuildEnv = kwargs["env"]
                g = kwargs["graph"]
                upstream = {dep: kwargs[target_node(dep)] for dep in graph.get(target_name).dependencies}
                return _run_target(env=env, graph=g, target_name=target_name, upstream=upstream)

            # Tell Hamilton what the signature is (env, graph, deps...)
            params = [
                inspect.Parameter("env", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=BuildEnv),
                inspect.Parameter("graph", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ] + [
                inspect.Parameter(dn, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=TargetRunRecord)
                for dn in dep_names
            ]
            _node.__signature__ = inspect.Signature(params)  # Hamilton inspects deps via signature
            _node.__name__ = node_name
            return _node

        setattr(m, node_name, _fn_factory(target.name, dep_node_names))

    return m
```

This is how you get to “metadata → nodes” at scale without manually writing 40+ functions.

---

## 12) Phase‑0 acceptance criteria (health checks)

You want Phase‑0 to be “safe to iterate” with clear checks.

### 12.1 DAG build validation

* Running Hamilton driver construction must succeed.
* The DAG must include the expected dependencies (e.g. `t__risk_factors` depends on `t__call_graph` and `t__function_metrics`).

### 12.2 Dry-run style execution

* Execute `t__risk_factors` on a tiny repo fixture and confirm:

  * all upstream nodes ran in order,
  * manifests were written,
  * a second run skips unchanged nodes.

### 12.3 Observability

* Ensure tags show up (domain/target tags on nodes).
* Later, turn on a progress adapter / telemetry hooks (Hamilton supports these through lifecycle APIs and adapters). 

### 12.4 Storage sanity

* Ensure all target tables exist in DuckDB after run.
* If you enable Ibis loaders, ensure `load_duckdb_table()` returns an Ibis table and can perform at least one composed query.

---

## Recommendation on ordering (direct answer)

### Should you “replace build folder to Hamilton first” vs “integrate plugins into Hamilton first”?

They are the same action in Phase‑0 if you implement it as above:

* **Build is the place that defines DAG selection and execution entrypoints**, so Phase‑0 lives in build.
* **Plugins are wrapped, not rewritten**, so plugin integration is “thin”.

So: **start in `build/`** by adding the Hamilton executor + Phase‑0 node module, then expand coverage.

---

## What I’d do immediately after Phase‑0 lands

Once `risk_factors` runs end-to-end:

1. Convert `targets_phase0.py` → generated module from target graph (scale-out)

2. Introduce dataset refs + `@extract_fields` systematically (so datasets become first-class artifacts) 

3. Start adopting `@dataloader/@datasaver` for *serving-facing* datasets first (where the lineage and contracts help the most) 

4. Add `@check_output` Pandera checks on a few critical outputs (shift-left contracts) 

---


