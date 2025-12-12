Below is a **Phase 2 implementation plan** that is intentionally **aggressive** (since you’re still in design), and pushes much harder into “Hamilton-first” strengths: **planning/explain**, **materialization & dataset I/O nodes**, **stronger caching semantics**, and **observability/run introspection**. I’m also structuring this as a **PR-mapped checklist** (PR‑08 … PR‑15) with **exact files/functions to touch** and **tests to add under `tests/build`**.

---

## Phase 2 goals

### What we optimize for

1. **Hamilton becomes the “source of truth” orchestration layer**

   * “Run”, “plan”, “explain”, “graph”, “validate” all come from the Hamilton DAG/runtime.
2. **Best‑in‑class incremental builds**

   * Fast “what will run & why?” planning.
   * Deterministic cache keys that cascade correctly through dependencies.
   * Minimize repeated manifest I/O.
3. **Asset-centric DAG**

   * Dataset outputs aren’t just metadata — they become *first-class nodes* you can load/validate/materialize.
4. **Deep observability**

   * Per-target records persisted and queryable by run_id.
   * Rich graph exports (JSON + Mermaid + DOT).
5. **Foundation for Phase 3: true Hamilton-native compute**

   * Phase 2 introduces the scaffolding so Phase 3 can migrate targets from “plugin side-effects” → “pure nodes + materializers”.

---

## Phase 2 deliverables (high level)

* **Hamilton default engine + mode control** (fix the current “default is phase0” footgun)
* **A real Hamilton build planner** (dry-run outputs computed/skipped/missing + reasons + hashes)
* **Improved hashing/caching semantics** (dependency hash cascade; manifest prefetch/index)
* **DatasetRef v2 + ArtifactRef + loader nodes** (d__/q__/df__ nodes)
* **Optional post-write validation** (Pandera validation on produced datasets)
* **Persist per-target run records** into DuckDB (build.run_targets) for history/debugging
* **Graph exports**: JSON + Mermaid + DOT + “why is node stale?” metadata

---

# PR roadmap (Phase 2)

## PR‑08 — Make Hamilton the default + add explicit Hamilton mode

**Theme:** remove ambiguity and make the Hamilton path the “normal” path.

### Changes

1. **Default Hamilton mode should be `generated` everywhere** (driver factory + executor + CLI path).
2. Add CLI support for `--hamilton-mode` (`phase0|generated`) to preserve Phase0 debugging.

### Files to touch

* `build/hamilton/driver_factory.py`

  * Change defaults: `mode: HamiltonNodeMode = "generated"` in:

    * `build_driver(...)`
    * `list_available_nodes(...)`
    * `target_to_node_name(...)`
* `build/hamilton/executor.py`

  * `HamiltonBuildExecutor.__init__(..., mode="generated")`
* `cli/commands/build.py`

  * Add field to `BuildRunCommand`:

    * `hamilton_mode: str = "generated"`
  * Consider flipping `engine` default to `"hamilton"` (aggressive transition)
* `cli/handlers/build.py`

  * Pass the CLI mode through to `HamiltonBuildExecutor(..., mode=params.hamilton_mode)`

### Code sketch

**`cli/commands/build.py`**

```python
@dataclass
class BuildRunCommand:
    ...
    engine: Annotated[
        str,
        Parameter(
            name=["--engine", "-e"],
            help="Build engine to use: hamilton (default) or legacy.",
            show_choices=True,
        ),
    ] = "hamilton"

    hamilton_mode: Annotated[
        str,
        Parameter(
            name=["--hamilton-mode"],
            help="Hamilton node mode: phase0 (debug) or generated (full DAG).",
            show_choices=True,
        ),
    ] = "generated"
```

**`cli/handlers/build.py`** (inside `_execute_build_hamilton`)

```python
executor = HamiltonBuildExecutor(profile="default", mode=params.hamilton_mode)
hamilton_result = executor.run(env=env, targets=goals)
```

### Tests to add/update (`tests/build`)

* Update existing Hamilton tests to explicitly set mode when needed:

  * `tests/build/test_hamilton_phase0.py`: pass `mode="phase0"` where appropriate.
* Add: `tests/build/test_hamilton_phase2_defaults.py`

  * Ensure that calling `build_driver(..., mode="generated")` exposes nodes for *all* targets.

---

## PR‑09 — Hamilton planner: “what will run & why?” + real dry-run output

**Theme:** best-in-class DX. In Phase 2, `--dry-run` should not be a stub — it should be **actionable**.

### Design

Introduce a Hamilton-centric planner that:

* Computes closure using the **TargetGraph** (topological order)
* Computes `options_hash` + `input_hash` per target
* Looks up prior manifests (prefer preloaded index — PR‑10)
* Produces a plan containing:

  * `status`: `compute|skip|missing|blocked`
  * `reason`: `forced|no_manifest|hash_changed|up_to_date|upstream_missing`
  * `input_hash` (+ prior input hash)
  * node name mapping (t__/d__/…)
  * contract outputs (table_keys/artifacts) for reporting

### Files to add/touch

* **Add** `build/hamilton/planner.py`
* `cli/handlers/build.py`

  * When `dry_run=True` and engine==hamilton: run planner and return meaningful output.
* (Optional but recommended) **Add** `cli/commands/build_plan.py` or extend `build` group:

  * `codeintel build plan ...` → returns plan JSON

### Code sketch

**`build/hamilton/planner.py`**

```python
from dataclasses import dataclass
from typing import Literal

from codeintel.build.hamilton.driver_factory import target_to_node_name
from codeintel.build.hamilton.manifest_hook import compute_target_options_hash, compute_target_input_hash
from codeintel.build.targets import TargetGraph

PlanStatus = Literal["compute", "skip", "missing", "blocked"]

@dataclass(frozen=True)
class PlanEntry:
    target: str
    node: str
    module: str
    status: PlanStatus
    reason: str
    input_hash: str | None
    options_hash: str | None
    prior_input_hash: str | None
    dependencies: tuple[str, ...]
    table_keys: tuple[str, ...]
    artifact_keys: tuple[str, ...]  # Phase2+; can be () initially

@dataclass(frozen=True)
class HamiltonBuildPlan:
    requested: tuple[str, ...]
    closure: tuple[str, ...]
    entries: tuple[PlanEntry, ...]

    @property
    def to_compute(self) -> tuple[str, ...]:
        return tuple(e.target for e in self.entries if e.status == "compute")

    @property
    def to_skip(self) -> tuple[str, ...]:
        return tuple(e.target for e in self.entries if e.status == "skip")

def compute_plan(*, env, graph: TargetGraph, requested: tuple[str, ...], mode: str) -> HamiltonBuildPlan:
    closure = tuple(graph.topological_order(requested))
    entries: list[PlanEntry] = []

    # prior manifests (PR-10 will replace this with env.manifest_index)
    prior = {m.target: m for m in env.gateway.build.list_manifests(repo=env.snapshot.repo, commit=env.snapshot.commit)}

    for t in closure:
        target = graph.get(t)
        node = target_to_node_name(t, mode=mode)
        opts = env.config.parameters_for(t)
        options_hash = compute_target_options_hash(opts)
        input_hash = compute_target_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            options_hash=options_hash,
        )
        p = prior.get(t)
        prior_hash = p.input_hash if p else None

        if env.is_forced(t):
            status, reason = "compute", "forced"
        elif p is None:
            status, reason = "compute", "no_manifest"
        elif p.input_hash != input_hash:
            status, reason = "compute", "hash_changed"
        else:
            status, reason = "skip", "up_to_date"

        entries.append(
            PlanEntry(
                target=t,
                node=node,
                module=target.module,
                status=status,
                reason=reason,
                input_hash=input_hash,
                options_hash=options_hash,
                prior_input_hash=prior_hash,
                dependencies=tuple(target.dependencies),
                table_keys=tuple(target.table_keys),
                artifact_keys=tuple(a.name for a in (target.contract.artifacts if target.contract else ())),
            )
        )

    return HamiltonBuildPlan(requested=requested, closure=closure, entries=tuple(entries))
```

### Tests (`tests/build`)

* Add `tests/build/test_hamilton_phase2_planner.py`

  * Uses a tiny fake gateway/build accessor returning manifests.
  * Verifies:

    * forced → compute
    * no manifest → compute
    * hash mismatch → compute
    * hash match → skip
  * Verifies closure ordering matches TargetGraph.

---

## PR‑10 — Fix cache semantics + manifest index prefetch (performance + correctness)

**Theme:** incremental builds must cascade correctly and must be fast.

### Why this matters

Right now, hashing/skip decisions happen target-by-target and can involve repetitive manifest fetches. Phase 2 should:

* **Prefetch manifests once** per run
* **Compute input hashes using dependency hashes that actually cascade**
* Make “plan” and “run” use the same cached view of prior state

### Changes

1. Extend `BuildEnv` to carry a **manifest index**
2. Update `compute_input_hash()` to optionally use a preloaded manifest map and (recommended) to use dependency **input_hash** for cascade.
3. Update Hamilton skip logic to use the manifest index (no per-target DB round trips).

### Files to touch

* `build/hamilton/env.py`

  * Add `manifest_index: Mapping[str, OutputManifest] | None = None`
* `cli/handlers/build.py`

  * When constructing env: load manifests once and inject.
* `build/hashing.py`

  * Add optional `manifests: Mapping[str, OutputManifest] | None = None`
  * Update dependency hash choice (recommended: dependency manifest **input_hash**)
* `build/hamilton/nodes/targets_phase0.py`

  * `_should_skip_target` should consult `env.manifest_index` first
  * `_compute_hashes` passes manifests through for fast hashing (if you thread it)

### Code sketch

**`build/hashing.py`**

```python
def compute_input_hash(
    target: OutputTarget,
    snapshot: SnapshotRef,
    gateway: StorageGateway,
    options_hash: str | None = None,
    *,
    manifests: Mapping[str, OutputManifest] | None = None,
) -> str:
    dep_parts: list[str] = []
    for dep in target.dependencies:
        m = manifests.get(dep) if manifests is not None else gateway.build.load_manifest(
            target=dep, repo=snapshot.repo, commit=snapshot.commit
        )
        # cascade on input_hash (more robust than output_hash in this codebase)
        dep_hash = (m.input_hash if m is not None else "MISSING")
        dep_parts.append(f"{dep}:{dep_hash}")

    deps_blob = ",".join(sorted(dep_parts))
    opts_blob = options_hash or ""
    combined = f"{snapshot.repo}:{snapshot.commit}|{target.name}|{deps_blob}|{opts_blob}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]
```

**`cli/handlers/build.py`** (when building `BuildEnv`)

```python
prior_manifests = gateway.build.list_manifests(repo=runtime.snapshot.repo, commit=runtime.snapshot.commit)
manifest_index = {m.target: m for m in prior_manifests}

env = BuildEnv(
    gateway=gateway,
    snapshot=runtime.snapshot,
    paths=runtime.paths,
    providers=providers,
    config=config,
    profile="default",
    force_targets=frozenset(force or ()),
    manifest_index=manifest_index,
)
```

### Tests (`tests/build`)

* Update `tests/build/test_hashing_plan_targets.py`

  * Adjust `test_compute_input_hash_differentiates_dependency_hashes` to match new behavior (dependency input_hash cascade).
* Add `tests/build/test_hamilton_phase2_manifest_index.py`

  * Ensures skip check uses env.manifest_index and does not require per-target load calls (can be asserted via a fake accessor that raises if called).

---

## PR‑11 — DatasetRef v2 + “skipped targets still yield dataset refs” + ArtifactRef

**Theme:** asset-centric DAG requires that dataset/artifact nodes are reliable even when targets are skipped due to cache freshness.

### Changes

1. Upgrade `DatasetRef` to include snapshot identity (`repo`, `commit`) so loaders can safely filter.
2. Ensure `TargetRunRecord` for **fresh-cache skips** includes dataset refs (row_count can be `None`).
3. Introduce `ArtifactRef` + optional `a__*` nodes for artifact lineage.

### Files to touch/add

* `build/hamilton/io/dataset_ref.py`

  * Add fields: `repo: str`, `commit: str`
  * Update `refs_from_target_result(...)` to accept `snapshot`
* `build/hamilton/manifest_hook.py`

  * `TargetRunRecord`: add `artifacts: tuple[ArtifactRef, ...] = ()` (new)
* `build/hamilton/nodes/targets_phase0.py`

  * In “skip due to manifest match” branch: return a `TargetRunRecord` that includes dataset refs
* **Add** `build/hamilton/io/artifact_ref.py`
* `build/hamilton/naming.py`

  * Add `artifact_node(...)` helper
* `build/hamilton/nodes/node_factory.py`

  * Generate artifact nodes and attach `ARTIFACT_TO_NODE`

### Code sketch

**`build/hamilton/io/dataset_ref.py`**

```python
@dataclass(frozen=True)
class DatasetRef:
    table_key: str
    repo: str
    commit: str
    schema_version: str | None = None
    row_count: int | None = None
    source_target: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
```

**`build/hamilton/nodes/targets_phase0.py`** (skip branch)

```python
if _should_skip_target(env, target_name, hashes.input_hash):
    datasets = tuple(
        refs_from_target_result(
            snapshot=env.snapshot,
            target_name=target_name,
            table_keys=target.table_keys,
            row_counts=None,  # unknown on skip
        ).values()
    )
    return TargetRunRecord(
        target=target_name,
        plugin_name=meta.name,
        status="skipped",
        input_hash=hashes.input_hash,
        options_hash=hashes.options_hash,
        duration_ms=0.0,
        datasets=datasets,
    )
```

### Tests (`tests/build`)

* Update `tests/build/test_hamilton_phase1.py`

  * Add assertion: dataset refs exist even when record is skipped (construct a record or exercise skip path with fake manifest index).
* Add `tests/build/test_hamilton_phase2_datasetref_v2.py`

  * Validate schema/table parsing still works.
  * Validate repo/commit are present and preserved.

---

## PR‑12 — Dataset loader nodes: `q__*` (Ibis) + `df__*` (pandas) + optional validation

**Theme:** this is where Hamilton starts to feel *powerful* — not just orchestration, but actual *asset flows*.

### Changes

1. Add loader nodes per dataset:

   * `q__schema__table` returns an **Ibis table expression** filtered to repo/commit when possible.
   * `df__schema__table` returns a **pandas DataFrame** (usually for debugging/validation).
2. Add a “safe filter” layer: only filter on repo/commit if those columns exist.
3. Add **optional** output validation after target success:

   * If enabled, load produced datasets and run Pandera validation (using `storage.pandera_schemas.validate_dataset_df`).

### Files to touch

* `build/hamilton/io/ibis_adapter.py`

  * Add `load_dataset_ibis(...)` and `load_dataset_df(...)` with filtering logic.
* `build/hamilton/naming.py`

  * Add `query_node(table_key)` and `dataframe_node(table_key)`
* `build/hamilton/nodes/node_factory.py`

  * Generate these loader nodes per dataset
* `cli/commands/build.py`

  * Add `--validate-outputs` flag (optional but recommended)
* `cli/handlers/build.py`

  * Pass flag into `BuildEnv` (add field in env)

### Code sketch

**`build/hamilton/io/ibis_adapter.py`**

```python
def load_dataset_ibis(*, gateway: IbisGateway, ref: DatasetRef):
    t = gateway.table(ref.table_key)
    cols = set(t.columns)
    if "repo" in cols and "commit" in cols:
        t = t.filter((t.repo == ref.repo) & (t.commit == ref.commit))
    return t

def load_dataset_df(*, gateway: IbisGateway, ref: DatasetRef):
    t = load_dataset_ibis(gateway=gateway, ref=ref)
    return t.execute()
```

**`build/hamilton/nodes/node_factory.py`** (per-table loader nodes)

```python
def _create_dataset_query_node_function(table_key: str) -> Callable[..., object]:
    d_name = dataset_node(table_key)
    q_name = query_node(table_key)

    def query_fn(env: BuildEnv, **kwargs: object):
        ref = kwargs[d_name]
        return load_dataset_ibis(gateway=env.gateway.ibis, ref=ref)

    query_fn.__name__ = q_name
    ...
    return tag(domain="io", table=table_key)(query_fn)
```

### Tests (`tests/build`)

* Add `tests/build/test_hamilton_phase2_loader_nodes.py`

  * Ensure generated module includes:

    * `d__analytics__function_metrics`
    * `q__analytics__function_metrics`
    * `df__analytics__function_metrics`
* Add a small unit test for filtering behavior with a fake ibis object if available in test env; otherwise keep it as a contract test on function shape.

---

## PR‑13 — Persist per-target run records: `build.run_targets`

**Theme:** make runs inspectable and debuggable without reading logs.

### Changes

1. Add a new dataset schema for `build.run_targets`
2. Persist `TargetRunRecord` entries for each target in a run:

   * run_id, repo, commit, target, status, input_hash, options_hash, duration_ms, error
   * row_counts as JSON
3. Extend build history to optionally include per-target breakdown.

### Files to touch

* `config/datasets/schemas.py`

  * Add `build.run_targets` `TableSchema`
* `storage/pandera_schemas.py`

  * Optionally add checks for run_targets (status in allowed set, duration non-negative, etc.)
* `storage/tracking/build_tracking.py`

  * Add `save_run_target(...)` or a generic insert helper
* `build/hamilton/executor.py`

  * After execution completes, persist per-target records with the run_id
* `cli/handlers/build.py`

  * When `build history --run-id X`: include run_targets rows in JSON output (optional but very useful)

### Code sketch

**`config/datasets/schemas.py`**

```python
"build.run_targets": TableSchema(
    schema="build",
    name="run_targets",
    columns=[
        Column("run_id", "VARCHAR", nullable=False),
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("target", "VARCHAR", nullable=False),
        Column("plugin", "VARCHAR", nullable=False),
        Column("status", "VARCHAR", nullable=False),
        Column("input_hash", "VARCHAR"),
        Column("options_hash", "VARCHAR"),
        Column("duration_ms", "DOUBLE", nullable=False),
        Column("row_counts", "JSON", nullable=False),
        Column("error", "VARCHAR"),
        Column("recorded_at", "TIMESTAMPTZ", nullable=False),
    ],
    primary_key=("run_id", "target"),
    indexes=(Index("idx_build_run_targets_repo_commit", ("repo", "commit")),),
)
```

### Tests (`tests/build`)

* Add `tests/build/test_hamilton_phase2_run_targets_persistence.py`

  * Use an in-memory gateway or temp duckdb file gateway.
  * Run a tiny synthetic “execution” or directly call persistence method with a fake record.
  * Assert rows inserted.

---

## PR‑14 — Graph exports: Mermaid + DOT + enriched node metadata

**Theme:** make DAG inspection a joy.

### Changes

1. Extend `build/hamilton/observability.py` to export:

   * Mermaid `graph TD`
   * Graphviz DOT
2. Fix `codeintel build graph --output-format ...` so it’s not ignored.

### Files to touch

* `build/hamilton/observability.py`

  * add:

    * `export_dag_mermaid(info: DagInfo) -> str`
    * `export_dag_dot(info: DagInfo) -> str`
* `cli/handlers/build.py`

  * In `build_graph_handler`: respect `output_format` and `output_file`.
* `cli/commands/build.py`

  * Expand `BuildGraphCommand.output_format` choices to include `mermaid` and `dot`.

### Mermaid export sketch

```python
def export_dag_mermaid(info: DagInfo) -> str:
    lines = ["graph TD"]
    for node_name, node_data in info.nodes.items():
        label = node_data.get("node_type", "node")
        lines.append(f'  {node_name}["{node_name} ({label})"]')
    for src, dsts in info.dependencies.items():
        for dst in dsts:
            lines.append(f"  {src} --> {dst}")
    return "\n".join(lines) + "\n"
```

### Tests (`tests/build`)

* Add `tests/build/test_hamilton_phase2_graph_exports.py`

  * Build a DagInfo from a tiny module and verify output contains expected edges.

---

## PR‑15 — (Stretch, but very Phase‑2‑worthy) “Explain staleness” + “plan diff”

**Theme:** truly best-in-class incremental system UX.

### Goal

Add a command that answers:

* “Why will target X run?”
* “Which dependency hash changed?”
* “What’s the minimal set of upstream changes causing recompute?”

### Implementation idea

* Extend planner entries to include:

  * `dep_hashes: dict[dep, dep_input_hash]`
  * `prior_dep_hashes: dict[...]` (from prior manifests)
* Then produce a “diff” listing changed deps.

### Files

* `build/hamilton/planner.py` (extend PlanEntry)
* `cli/commands/build.py` add `build explain <target>` OR `build plan --diff`

### Tests

* Add `tests/build/test_hamilton_phase2_explain_diff.py`

  * Fake manifests for deps and assert the diff highlights the right dep.

---

# Implementation sequencing recommendation (fastest payoff order)

If you want the “wow” quickly:

1. **PR‑08** (defaults + mode) — removes a major sharp edge.
2. **PR‑10** (manifest index + hash cascade) — correctness + speed foundation.
3. **PR‑09** (planner) — immediately useful DX.
4. **PR‑11/12** (DatasetRef v2 + loader nodes) — turns DAG into an asset graph.
5. **PR‑14** (mermaid/dot) — huge dev UX multiplier.
6. **PR‑13** (persist run targets) — long-term observability & debugging.

---

# Notes on aggressiveness and breaking changes

Because you’re not in prod, I *do* recommend being willing to break:

* CLI JSON output shape (add fields like plan entries)
* DatasetRef structure (v2 repo/commit)
* Hash semantics (dependency cascade)

Those changes pay off massively once you start migrating targets to Hamilton-native “pure compute + materialize” patterns in Phase 3.

---

