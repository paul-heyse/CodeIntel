
Here’s the **paired narrative** for what (5) and (6) *actually change* in the system—i.e., how the build/serve/explain loop feels after the refactor, and what conceptual “kinks” we’re removing.

---

## 5) Plan/explain becomes a first-class DAG product

### Before (sidecar planning)

“Plan” and “explain” live as **external control logic**:

* A planner traverses metadata, computes closure, reasons about “skip/compute/blocked,” and emits an explanation payload.
* Execution is then orchestrated as a second step, with the planner’s outputs acting as a kind of control plane.
* The artifacts (decision trace / explain output) are *downstream* of execution but not structurally represented as DAG outputs. They’re computed “off to the side,” using bespoke code paths that can drift from real execution semantics.

This makes planning:

* hard to cache coherently,
* hard to serve uniformly (because it isn’t part of the same catalog/output system),
* and fragile (because planner logic is not guaranteed to match what execution + caching actually do).

### After (planning is “just another target”)

We change the mental model to:

> Planning and explanation are **Hamilton nodes** that emit **normal saver-derived outputs**, like any other computation.

What you implement is a **planning subgraph** that depends only on *structural* + *availability* inputs:

* `catalog` (targets, deps, IO, output inventory)
* `env` / config digests
* `cache_index` (probe cache presence for predicted reuse)
* `plan_request` (what the user asked for)
* `preflight` signals (missing prereqs, missing schemas, invalid environment)

…and explicitly does **not** depend on any compute tables/artifacts.

In other words, planning becomes a **referentially transparent DAG computation**.

#### What “plan” is now

A `BuildPlan` object is produced inside the DAG:

* closure list (toposorted)
* per-target entries:

  * deps, reads, writes (from catalog)
  * predicted action: reuse/compute/blocked
  * block reasons (from preflight)
  * optional node-level cache hit/miss breakdown (from cache probe)

#### What “explain” is now

“Explain” is no longer special logic; it is a **rendering node** over the `BuildPlan`:

* `ci.plan.json` (machine)
* `ci.plan.explain.md` (human)
* optionally `ci.plan.entries` (DuckDB table for joins/diffs/serving)

These are emitted by standard saver nodes with `output_role="contract"`, so:

* they’re visible to your catalog output inventory,
* they’re enforceable via the same write enforcement mechanism,
* they’re servable via the same semantic/serving pipeline.

#### The big behavioral shift

* `--plan` no longer means “call planner code.”
* It means “execute target `ci_plan`” (or equivalent), which **materializes plan artifacts**.
* Plans can be cached, diffed in CI, served via MCP/HTTP like anything else, and correlated with manifest/cache telemetry because they’re first-class artifacts.

---

## 6) Enforce a single composition root (no driver rebuilds inside DAG nodes)

### Before (multiple implicit composition roots)

Even with a good architecture, large systems tend to accrete “helper” paths where some module:

* rebuilds a Hamilton driver to introspect tags,
* rebuilds a driver to compile semantic views,
* rebuilds a driver per request in serving,
* rebuilds a driver inside an export step “just to list variables,” etc.

That creates subtle incoherence:

* two drivers built with slightly different module sets/config/adapters,
* different tagging inventories (because support nodes differ),
* different caching adapters wired,
* different catalog compilations and therefore different “truth.”

This is also a maintainability killer: you can’t reason about “the system” because there are multiple ways to assemble it.

### After (one “composition root” produces a RuntimeBundle)

We force the entire program into a strict topology:

> **Exactly one place builds the runtime object graph** (driver + catalog + caches + registries + stores).
> Everything else consumes it.

You introduce a canonical `RuntimeBundle` (or `HamiltonRuntime`) that contains:

* the final `Driver`
* the final `DagCatalog`
* cache adapter + cache probe index
* schema index + semantic registry (compiled once, or loaded from snapshot)
* artifact store / DuckDB context
* runtime fingerprint (config + modules + versions)

And you introduce one constructor:

* `compose_runtime(env, cfg) -> RuntimeBundle`

No other module is allowed to:

* discover modules,
* call DriverBuilder,
* rebuild the driver,
* construct “mini runtimes” ad hoc.

#### How this changes execution and serving

* CLI build paths: compose runtime once → execute targets against the same runtime.
* Serving startup: compose runtime once → store it in `app.state` → requests only read/execute, never rebuild.
* Export tasks: use the already-composed runtime’s driver/catalog/tag_query, never build their own.

#### How you prevent regressions

You add both:

* **static enforcement** (AST linter that forbids driver-building imports/calls under DAG node directories), and
  * Linter entrypoint: `tools/lint_no_driver_build_in_nodes.py`
  * Test gate: `tests/lint/test_no_driver_build_in_nodes.py`
* **runtime enforcement** (thread-local guards that prevent composition from being called during DAG execution).

Optionally, you introduce snapshotting so serving can load catalog/registry artifacts without importing Hamilton at all when it doesn’t need to execute.

---

## How (5) and (6) reinforce each other

These two changes lock together tightly:

### Planning as DAG product only works cleanly if runtime construction is singular

If planning nodes were allowed to rebuild drivers, you’d get a paradox:

* “plan” would be computed against one driver configuration,
* execution might run against another,
* and your plan would become an unreliable narrative rather than a deterministic product.

By enforcing a single composition root:

* planning DAG nodes always see the same `catalog`, `cache_index`, `schema_index`, and tags that execution uses.
* “plan vs actual” discrepancies become meaningful signals (cache drift, environment drift), not assembly drift.

### The end state feels like this

1. Compose runtime once (deterministic).
2. Execute `ci_plan` target to materialize plan outputs.
3. Execute compute targets; caching decides reuse.
4. Plan/explain artifacts + cache/manifest telemetry can be joined because both are emitted from the same runtime fingerprint and the same output inventory model.

Net effect: you stop having a “planning subsystem” and a “build subsystem.” You have **one DAG-driven system** where planning, execution, export, and serving are all just different DAG products computed from the same immutable runtime.



Below is a **repo-concrete, breaking-change–friendly**, **high lexical density** implementation plan for:

> **(5) Turn “plan/explain” into a first-class DAG product, not an external side path**

Assumptions (because this phase composes directly on prior refactors):

* **(1)** is in place: `DagCatalog` is the sole structural view; no `TargetGraph`.
* **(3)** is in place: **saver-tag-derived outputs** are the only output inventory.
* **(4)** is in place: **Hamilton caching is authoritative**; manifests are audit/telemetry.

The objective here is to eliminate the “planner sidecar” pattern (Python functions computing plan/explain outside the DAG) and replace it with a **DAG-native planning subgraph** that:

1. is computed via Hamilton execution,
2. is persisted via standard saver nodes (artifact + optional table),
3. is served/queried like any other output product,
4. can be diffed/validated in CI and correlated with cache/manifest telemetry.

---

# 0) End-state contract

## 0.1 Planning is a DAG product

* “Plan” and “Explain” are emitted by DAG nodes (Hamilton variables) and materialized via saver decorators.
* Plan generation is *purely* a function of:

  * `DagCatalog` (targets/IO/deps)
  * `BuildEnv` + config digests
  * `CacheIndex` / cache store probe (for predicted hits/misses)
  * optional “availability indexes” (external inputs, pre-materialized datasets)
  * `requested_targets` + mode flags

No external “planner.py computes plan then executor runs” coupling remains.

## 0.2 Planning does not pull compute nodes into its dependency cone

Planning nodes must be **structurally isolated**: they must depend on `catalog` and cache/index primitives, not on “real compute” outputs (tables/artifacts). This prevents accidental “plan triggers compute”.

## 0.3 Planning outputs are standard outputs

* Planning emits:

  * `artifact: ci.plan.json` (canonical, machine-parseable)
  * `artifact: ci.plan.explain.md` (human-facing)
  * optional `table_key: ci.plan.entries` (structured table in DuckDB for BI/serving)
* These outputs are declared via savers with `output_role="contract"` so they participate in the same:

  * output inventory compilation (catalog),
  * schema enforcement,
  * serving artifact compilation.

---

# 1) Introduce a first-class planning data model (typed, serializable, stable)

### Files CREATED

1. `src/codeintel/build/planning/model.py`

Define “plan as product” dataclasses (frozen/slots) with explicit schema stability guarantees:

* `PlanRequest`

  * `requested_targets: tuple[str, ...]`
  * `mode: Literal["predict", "audit"]`

    * `predict`: pre-execution cache probe
    * `audit`: post-execution view (optional future extension; can join manifest/cache telemetry)
  * `include_node_details: bool`
  * `include_io_details: bool`
  * `include_cache_details: bool`
* `PlanNodeStat`

  * `node: str`
  * `version: str`
  * `cache_status: Literal["hit","miss","unknown"]`
* `PlanTargetEntry`

  * `target: str`
  * `domain: str`
  * `deps: tuple[str, ...]` (target-level)
  * `reads: tuple[str, ...]` (table_keys)
  * `writes_tables: tuple[str, ...]`
  * `writes_artifacts: tuple[str, ...]`
  * `predicted_action: Literal["compute","reuse","blocked"]`
  * `block_reasons: tuple[str, ...]`
  * `cache_hit_ratio: float | None`
  * `miss_nodes: tuple[str, ...]` (optional; gated by `include_node_details`)
* `BuildPlan`

  * `request: PlanRequest`
  * `closure: tuple[str, ...]` (deps-first)
  * `entries: tuple[PlanTargetEntry, ...]`
  * `created_at_utc: str`
  * `build_fingerprint: str` (env/config digest; must match cache key resolver inputs)

Add canonical JSON encoding:

* `to_dict()` methods (no `pydantic` required unless already pervasive)
* stable field ordering
* explicit version header: `plan_schema_version: "v1"`

---

# 2) Add a cache probe/index abstraction usable inside DAG nodes

Planning requires pre-execution cache introspection without invoking driver caching execution. Do not couple planning to `CacheAdapter` internals; instead, define a read-only interface.

### Files CREATED

2. `src/codeintel/build/hamilton/cache_index.py`

Define minimal probe surface:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Iterable

@dataclass(frozen=True, slots=True)
class CacheProbeResult:
    node: str
    version: str
    hit: bool

class CacheIndex(Protocol):
    def has(self, *, node: str, version: str) -> bool: ...
    def batch_has(self, pairs: Iterable[tuple[str, str]]) -> tuple[CacheProbeResult, ...]: ...
```

### Files MODIFIED

* `src/codeintel/build/hamilton/cache_adapter.py` (from (4))

  * Factor persistence store into a shared component implementing `CacheIndex` semantics:

    * `DuckDBCacheStore` / `ParquetCacheStore` implements both:

      * write-path for adapter (`put`)
      * read-path for index (`has`, `batch_has`)
  * The adapter composes `CacheStore` rather than embedding probe logic.

### Wiring rule

* `cache_index` is injected as a runtime input to Hamilton execution:

  * `inputs={"env": env, "catalog": catalog, "cache_index": cache_index, "plan_request": req}`

---

# 3) Implement planning as a Hamilton native module (pure, taggable, saver-driven)

### Files CREATED

3. `src/codeintel/build/hamilton/native/planning/plan_nodes.py`

This is the **planning subgraph**. It must be self-contained and depend only on `catalog/env/cache_index/plan_request` and lightweight helpers.

#### Key node inventory (recommended minimum)

* `plan_request(plan_request: PlanRequest) -> PlanRequest`
  (identity node; allows uniform injection, tagging, and downstream type clarity)
* `plan_target_closure(catalog: DagCatalog, plan_request: PlanRequest) -> tuple[str,...]`

  * `closure = catalog.closure(plan_request.requested_targets)`
* `plan_target_subgraph_nodes(catalog: DagCatalog, plan_target_closure: tuple[str,...]) -> dict[str, tuple[str,...]]`

  * For each target `t`, compute node set reachable upstream of `anchor_node(t)` until hitting other anchors (same stop condition used for target dep derivation).
  * This is the **per-target node cone** needed for cache prediction.
* `plan_node_versions(cache_key_resolver: CacheKeyResolver, catalog: DagCatalog, plan_target_subgraph_nodes: dict[...], env: BuildEnv) -> dict[str, str]`

  * Compute version per node via the unified resolver from (4).
  * The resolver must be deterministic and stable across process boundaries.
* `plan_cache_probe(cache_index: CacheIndex, plan_node_versions: dict[str,str]) -> dict[str, Literal["hit","miss","unknown"]]`

  * Use `batch_has` for amortized IO.
  * If cache_index is `None` (allowed), mark all as `"unknown"` and degrade gracefully.
* `plan_entries(catalog: DagCatalog, plan_target_closure: tuple[str,...], plan_cache_probe: dict[...], plan_target_subgraph_nodes: dict[...], plan_request: PlanRequest) -> BuildPlan`

  * For each target:

    * IO: from `catalog.targets[target].io`
    * deps: from `catalog.targets[target].deps`
    * predicted action:

      * `"reuse"` iff **all** nodes in target cone are hits (or sufficiently high hit ratio; define strict semantics)
      * `"compute"` if any miss
      * `"blocked"` if preflight indicates missing external prerequisites (see §4)
    * compute `cache_hit_ratio` if `include_cache_details`
    * include node miss list if `include_node_details`

#### Minimal code snippet (shows the “compute is not pulled” discipline)

```python
# plan_nodes.py (illustrative)
def plan_target_closure(catalog: DagCatalog, plan_request: PlanRequest) -> tuple[str, ...]:
    return catalog.closure(plan_request.requested_targets)

def plan_entries(
    catalog: DagCatalog,
    plan_target_closure: tuple[str, ...],
    plan_target_subgraph_nodes: dict[str, tuple[str, ...]],
    plan_cache_probe: dict[str, str],
    plan_request: PlanRequest,
) -> BuildPlan:
    # Only uses catalog + probe; never touches compute artifacts.
    ...
```

### Files CREATED

4. `src/codeintel/build/hamilton/native/planning/plan_savers.py`

Emit planning products via standard saver decorators (artifact + optional DuckDB table). The saver nodes are the “first-class output boundary”.

Recommended outputs:

* `artifact="ci.plan.json"` (canonical)
* `artifact="ci.plan.explain.md"` (human)
* `table_key="ci.plan.entries"` (optional; if you want queryable plan entries in DuckDB)

Example (conceptual; adapt to your `save_to` wrapper):

```python
# plan_savers.py (illustrative)
from codeintel.build.hamilton.save_to import save_to_artifact, save_to_duckdb_table

@save_to_artifact(artifact="ci.plan.json", output_role="contract", path_template="plans/{run_id}/plan.json")
def m__ci_plan_json(plan: BuildPlan) -> str:
    return plan_to_json(plan)

@save_to_artifact(artifact="ci.plan.explain.md", output_role="contract", path_template="plans/{run_id}/plan.md")
def m__ci_plan_explain_md(plan: BuildPlan) -> str:
    return render_plan_markdown(plan)

@save_to_duckdb_table(table_key="ci.plan.entries", output_role="contract")
def m__ci_plan_entries(plan: BuildPlan):
    return plan_entries_to_arrow(plan)  # Arrow/Polars preferred
```

### Files MODIFIED

* `src/codeintel/build/hamilton/driver_factory.py`

  * Include planning modules in the driver’s module list:

    * `native_modules += [codeintel.build.hamilton.native.planning.plan_nodes, plan_savers]`
  * Provide feature gating via config:

    * `ci.enable_planning_nodes: bool = True`
  * Ensure planning nodes are not filtered out by any “targets only” module selection mechanism.

---

# 4) Preflight/block reasoning: make “blocked” a DAG derivation, not a planner heuristic

To classify `"blocked"` deterministically inside the DAG, implement a preflight subgraph that checks *availability of prerequisites* required to execute targets. This is strictly “environment readiness”, not compute.

### Files CREATED

5. `src/codeintel/build/planning/preflight.py`

* Implement preflight signals:

  * missing external inputs (e.g., required source datasets not present)
  * missing schema registry entries for produced outputs (if you want plan to fail-fast)
  * missing toolchain prerequisites (scip index not found, repo snapshot invalid)
* Expose a compact model:

  * `PreflightIssue(kind, target|global, message, severity)`

### Files CREATED

6. `src/codeintel/build/hamilton/native/planning/preflight_nodes.py`

* `preflight_issues(env: BuildEnv, catalog: DagCatalog, plan_target_closure: tuple[str,...]) -> tuple[PreflightIssue, ...]`
* `preflight_block_map(preflight_issues) -> dict[str, tuple[str,...]]`

  * map `target -> block_reasons[]`

### Files MODIFIED

* `plan_entries(...)` in `plan_nodes.py`

  * incorporate `preflight_block_map`:

    * if target has block reasons → `predicted_action="blocked"`
    * else predicted action computed from cache probe results

> This yields a single coherent DAG-native explanation story: “blocked because X prerequisite absent,” without bespoke executor-side checks.

---

# 5) Make plan/explain invocable via the same target mechanism as compute

Because your orchestration is target-driven (Hamilton anchor nodes `t__*`), provide explicit “planning targets” so the system can:

* run plan-only without compute,
* persist plan outputs with standard outputs,
* expose via serving/MCP uniformly.

### Files CREATED

7. `src/codeintel/build/hamilton/native/planning/plan_targets.py`

Define materialize anchors:

* `t__ci_plan` (domain e.g. `"ops"` or `"control"`)
* `t__ci_explain` (optional; can be same target emitting both artifacts)

Anchor semantics:

* `t__ci_plan` returns `BuildPlan` (or references it) and is tagged with:

  * `node_type=materialize`
  * `target="ci_plan"`
  * `domain="ops"`
  * `target_spec_version="v1"`

This creates a clean operational path:

* `codeintel build --target ci_plan` produces plan artifacts.
* `codeintel build --target ci_plan --plan-for <targets...>` injects `plan_request`.

> Critical: this target must not depend on “compute target anchors”; it depends on `catalog` only.

---

# 6) Replace external planner functions with DAG execution wrappers

### Files MODIFIED

* `src/codeintel/build/hamilton/planner.py`

  * Convert `compute_plan(...)` into a thin wrapper:

    * builds runtime (driver+catalog)
    * executes the planning target or planning artifact saver nodes
    * returns `BuildPlan` (python object) and/or serialized report
  * Convert `explain_plan(...)` into:

    * execute `m__ci_plan_explain_md` (or return its in-memory output)
  * Delete all logic that manually:

    * computes closure,
    * computes skip reasons,
    * synthesizes decision trace payloads.
  * The planner becomes orchestration sugar, not a second computation system.

### Files MODIFIED

* `src/codeintel/cli/handlers/build.py`

  * Replace `--plan/--explain` codepaths that call planner side APIs with:

    * create `PlanRequest` from CLI args
    * execute `t__ci_plan` (or directly `m__ci_plan_json` / `m__ci_plan_explain_md`)
  * Ensure `plan_request` is passed as an input to Hamilton execution:

    * `inputs={"plan_request": req, "env": env, "catalog": runtime.catalog, "cache_index": cache_index}`

### Files MODIFIED (Serving)

* `src/codeintel/serving/api/plan.py` (or the equivalent plan endpoint module)

  * Replace endpoint logic that calls `planner.compute_plan` with:

    * execute planning DAG node(s) on-demand
    * optionally load cached plan artifacts from artifact store for identical `PlanRequest` fingerprint

---

# 7) Persist plan as both artifact and structured table (optional but high leverage)

If you persist `ci.plan.entries` into DuckDB, you gain:

* cheap diffing (`EXCEPT` queries),
* serving query endpoints,
* observability correlation (join with cache events / manifests).

### Files MODIFIED

* `src/codeintel/build/hamilton/materializers/duckdb_saver.py`

  * Ensure saver supports Arrow/Polars ingestion for plan entries efficiently (avoid pandas conversion).
  * Enforce stable schema for `ci.plan.entries`.

### Files CREATED (Schema)

8. `src/codeintel/core/schemas/tables/ci_plan_entries.py`

* Define explicit schema (if non-inferable) keyed by `table_key="ci.plan.entries"`:

  * columns: `run_id, created_at_utc, requested_targets, target, domain, action, cache_hit_ratio, block_reasons, miss_nodes, reads, writes_tables, writes_artifacts, build_fingerprint, plan_schema_version`
* Register in table registry so (3) schema enforcement passes.

---

# 8) Delete/reduce legacy “decision trace” bespoke machinery (reframe as a derived view)

Under (4), manifest is audit-of-cache-events; plan is now predicted pre-run. Decision trace should become either:

* a view over `ci.plan.entries` + cache/manifest tables, or
* an artifact rendered from those tables.

### Files MODIFIED

* `src/codeintel/build/hamilton/native/export/decision_trace.py`

  * Replace “planner-derived decision trace payload” with:

    * `decision_trace = render_decision_trace(plan_entries_table, cache_events_table)`
  * If cache events are not persisted as tables, read them from manifest audit logs.

### Files DELETED (if present)

* Any bespoke `explain_plan(...) -> JSON payload` builder utilities that are not reused once plan is a DAG output.

---

# 9) Tests: enforce the new invariants (plan is DAG output, deterministic, isolated)

### Files CREATED

* `tests/build/planning/test_plan_is_dag_product.py`

  * Build a minimal synthetic DAG with:

    * two targets with known deps,
    * one saver output each,
    * cache_index stub returning controlled hits/misses.
  * Execute `t__ci_plan` and assert:

    * `BuildPlan` object produced
    * `m__ci_plan_json` emits stable JSON
    * `ci.plan.entries` has expected rows (if enabled)
    * Plan graph does not execute compute savers (assert via side-effect counters on compute nodes)

* `tests/build/planning/test_plan_cache_probe_semantics.py`

  * Validate `predicted_action` rules:

    * all-hit → reuse
    * any-miss → compute
    * preflight missing → blocked

* `tests/build/planning/test_plan_schema_stability.py`

  * Assert `ci.plan.entries` schema matches registry definition and is stable across runs.

### Files MODIFIED

* Existing tests that refer to `planner.compute_plan` must be updated to execute the planning target/node(s).

---

# 10) File index summary (P0)

## Created

* `src/codeintel/build/planning/model.py`
* `src/codeintel/build/planning/preflight.py`
* `src/codeintel/build/hamilton/cache_index.py`
* `src/codeintel/build/hamilton/native/planning/plan_nodes.py`
* `src/codeintel/build/hamilton/native/planning/preflight_nodes.py`
* `src/codeintel/build/hamilton/native/planning/plan_savers.py`
* `src/codeintel/build/hamilton/native/planning/plan_targets.py`
* `src/codeintel/core/schemas/tables/ci_plan_entries.py`
* `tests/build/planning/test_plan_is_dag_product.py`
* `tests/build/planning/test_plan_cache_probe_semantics.py`
* `tests/build/planning/test_plan_schema_stability.py`

## Modified

* `src/codeintel/build/hamilton/driver_factory.py`
* `src/codeintel/build/hamilton/planner.py`
* `src/codeintel/cli/handlers/build.py`
* `src/codeintel/build/hamilton/cache_adapter.py` (factor store/index)
* `src/codeintel/build/hamilton/materializers/duckdb_saver.py`
* `src/codeintel/build/hamilton/native/export/decision_trace.py`
* Serving endpoint module(s) handling plan/explain

## Deleted (likely; depends on current structure)

* Any legacy `compute_plan/explain_plan` implementations that directly traverse graphs and compute skip decisions (once wrappers exist).

---

# 11) Definition of Done (hard gates)

1. **Plan artifacts are produced via saver nodes** and therefore appear in `DagCatalog` output inventory (from saver tags).
2. `--plan` / `--explain` CLI paths execute Hamilton variables (planning target or planning savers), not bespoke Python planner logic.
3. Planning nodes depend only on `(catalog, env, cache_index, plan_request, preflight)`; they do not depend on any compute outputs (tables/artifacts).
4. Plan action classification is deterministic and test-covered:

   * `blocked` supersedes cache semantics
   * `reuse` iff all nodes in target cone are cache hits (or your explicitly defined policy)
5. Optional but recommended: `ci.plan.entries` table schema is registry-defined and stable.
6. “Decision trace” is either eliminated or becomes a derived rendering over `ci.plan.entries` + cache/manifest events, not a planner-specific bespoke structure.

---

## Key architectural payoff (what this unlocks immediately)

Once plan/explain is a DAG product, you can:

* serve it via the same registry/MCP machinery as any other output,
* diff plan tables across branches in CI without custom tooling,
* correlate predicted misses vs actual cache misses (post-run) using shared fingerprints,
* reuse caching to avoid recomputing plan for identical requests (plan itself becomes cacheable).

If you want the next step after (5): **(6) enforce “single composition root” by eliminating any “driver rebuild inside DAG nodes” and pushing all graph/catalog/state construction into a single runtime factory**—this completes the “no sidecar compute” story across planning, serving, and export.



Below is a **repo-concrete, breaking-change–friendly**, **high lexical density** implementation plan for:

> **(6) Remove “driver rebuilds inside DAG nodes” and enforce a single composition root**

This phase is fundamentally about **topology of construction**: we take every code path that *reconstructs* a Hamilton `Driver` (or rebuilds module lists/config/adapters) from within “business logic” (and especially from within Hamilton DAG nodes), and we **force all orchestration wiring into one composition root**. Every other subsystem consumes a **stable, immutable runtime bundle** (driver + catalog + caches + registries) and is forbidden—structurally and mechanically—from building new drivers.

---

# 0) End-state contract

## 0.1 Single composition root (SCR)

* There exists exactly one module/function responsible for **constructing the runtime object graph**:

  * module discovery
  * config normalization + digesting
  * Hamilton adapter assembly (caching, lifecycle adapters, hooks)
  * support node expansion config (from catalog)
  * final driver build
  * final `DagCatalog` compile
  * cache store/index instantiation
  * schema/semantic registry compilation (or loading precompiled snapshots)

All other layers (CLI, serving, exports, DAG nodes) **receive** a `RuntimeBundle` (or `HamiltonRuntime`) and may only use it. They may not call driver builder APIs.

## 0.2 DAG nodes are referentially transparent w.r.t. driver construction

* No Hamilton node (anything under `codeintel.build.hamilton.native.*` and `codeintel.build.hamilton.nodes.*`) may:

  * import `driver_factory` / `runtime_factory`
  * call `build_driver(...)`, `DriverBuilder(...).build()`, `module_discovery(...)`
  * construct `Driver` objects, `Builder`, or `DriverBuilder` directly
* DAG nodes are allowed to depend on **derived** immutable products:

  * `catalog: DagCatalog`
  * `schema_index`, `semantic_registry`
  * `cache_index`
  * `runtime_fingerprint` (digest)
  * `plan_request`, etc.

## 0.3 Serving path does not rebuild drivers per request

* FastAPI startup builds (or loads) a single runtime bundle per tenant/config key.
* Requests only:

  * route to `runtime.driver.execute(...)` with injected inputs, or
  * read from persisted artifacts/tables, or
  * query DuckDB tables.

---

# 1) Introduce canonical runtime bundle + composition root API

### Files CREATED

1. `src/codeintel/runtime/runtime_bundle.py`

Define the immutable container for all “wired” subsystems:

* `RuntimeKey` (hashable)

  * `repo_fingerprint`
  * `config_fingerprint`
  * `modules_fingerprint`
  * `build_profile` (serving/build/export)
* `RuntimeBundle` (frozen/slots)

  * `driver: hamilton.driver.Driver`
  * `catalog: DagCatalog`
  * `cache_adapter: CacheAdapter | None`
  * `cache_index: CacheIndex | None`
  * `schema_index: SchemaIndex | None`
  * `semantic_registry: SemanticRegistry | None`
  * `artifact_store: ArtifactStore`
  * `duckdb: DuckDBConnectionProvider` (if used)
  * `fingerprint: str` (single canonical digest used everywhere)
  * `created_at_utc: str`

2. `src/codeintel/runtime/compose.py`

The single composition root entrypoint:

```python
# compose.py (illustrative; keep highly deterministic)
def compose_runtime(*, env: BuildEnv, cfg: RuntimeConfig) -> RuntimeBundle:
    # 1) normalize cfg, compute digests
    # 2) discover modules (native + plugins)
    # 3) build native driver
    # 4) compile native catalog
    # 5) derive support spec from catalog (if (2) is present)
    # 6) build final driver (native + support_nodes + adapters + caching)
    # 7) compile final catalog
    # 8) build cache store/index/adapter
    # 9) compile or load schema_index / semantic_registry
    # 10) return RuntimeBundle
```

3. `src/codeintel/runtime/registry.py`

Provide an in-process runtime cache (LRU) so you do not rebuild drivers repeatedly in serving/CLI loops:

* `RuntimeRegistry.get_or_create(key: RuntimeKey, factory: Callable[[], RuntimeBundle]) -> RuntimeBundle`
* thread-safe (RWLock) and optionally async-safe (if your serving stack is async).

---

# 2) Demote driver_factory to a private implementation detail

Right now, driver construction logic typically lives in `src/codeintel/build/hamilton/driver_factory.py` and is imported by multiple call sites (including “export” nodes and serving compilation paths). That fan-out is the root cause.

### Files MODIFIED

* `src/codeintel/build/hamilton/driver_factory.py`

  * Convert into a *private* helper module used only by `codeintel.runtime.compose` (or inline it and delete this file).
  * Remove any public API that can be imported by nodes; rename exports to underscore-prefixed:

    * `build_driver(...)` → `_build_driver_impl(...)`
  * Add an explicit guard: `__all__ = []` (or keep only internal symbols).
  * Optionally: move this file into `src/codeintel/runtime/_driver_build.py` and remove it from `build/hamilton/`.

### Files DELETED (optional but high-payoff)

* `src/codeintel/build/hamilton/driver_factory.py`
  if you fully migrate construction into `runtime/compose.py` and kill the old import path.

> The goal is to make it *difficult* for DAG code to accidentally depend on the builder.

---

# 3) Canonical “inputs injection” contract for DAG execution

To prevent nodes from self-assembling missing services, define a stable injection surface and enforce it in the executor.

### Files CREATED

4. `src/codeintel/runtime/inputs.py`

Define `ExecutionInputs` (frozen/slots) that the executor unpacks into Hamilton `inputs={...}`:

* `env: BuildEnv`
* `catalog: DagCatalog`
* `cache_index: CacheIndex | None`
* `schema_index: SchemaIndex | None`
* `semantic_registry: SemanticRegistry | None`
* `runtime_fingerprint: str`
* `plan_request: PlanRequest | None` (phase (5))
* future: `feature_flags`, `resource_limits`, etc.

### Files MODIFIED

* `src/codeintel/build/hamilton/executor.py`

  * Change execution entry to accept a `RuntimeBundle` (not raw driver/cfg):

    * `execute_targets(runtime: RuntimeBundle, requested: Sequence[str], *, inputs: ExecutionInputs | None=None)`
  * Ensure `inputs` are *always* populated from `RuntimeBundle` (single source of injection truth).
  * Hard-delete any fallback behavior that says “if missing schema_index, build it here” or “if missing semantic registry, compile it here”.

---

# 4) Purge driver rebuild call sites (mechanical refactor sweep)

This is the central effort: find all secondary composition roots.

## 4.1 Grep targets (non-negotiable)

Perform a repository sweep for any of the following in non-runtime modules:

* `build_driver(` / `compose_driver(` / `DriverBuilder(` / `Builder(` (Hamilton)
* `module_discovery` / `discover_modules`
* `Driver(` instantiation
* `with_modules(` / `with_config(` builder calls in places other than composition root

Any hits inside:

* `codeintel.build.hamilton.native.*`
* `codeintel.build.hamilton.nodes.*`
* `codeintel.build.serving.*`
* `codeintel.build.spec.*`
* `codeintel.build.schemas.*`
* `codeintel.build.assets.*`
  …are refactor targets.

## 4.2 Typical offenders and their replacement patterns

### A) Export/serving artifact generation nodes

Common pattern: a DAG node (or export function) compiles a driver “just to introspect tags” or “list variables”.

**Replace** with catalog-based compilation:

* If you need “semantic registry”, compile it from `catalog.find_nodes(...)` (phase (1)) rather than building a driver.
* If you need “available targets”, use `runtime.catalog.targets`.

**Files MODIFIED (likely)**

* `src/codeintel/build/hamilton/native/export/serving_artifacts.py`
* `src/codeintel/build/serving/semantic_compile.py`
* `src/codeintel/build/assets/emitter.py`
* `src/codeintel/build/spec/compile.py`

**Change signatures**

* from `compile_* (modules, config, ...)` → `compile_* (catalog: DagCatalog, schema_index: SchemaIndex, ...)`

### B) Schema inference services that build drivers

Replace any schema service that builds a driver to learn “what tables exist” with:

* `runtime.catalog.table_outputs`
* `runtime.schema_index` compiled at composition root

**Files MODIFIED**

* `src/codeintel/build/schemas/inference_service.py`
* `src/codeintel/build/schemas/schema_index.py`
* `src/codeintel/build/schemas/provider_unified.py`

### C) Plan/explain path that rebuilds drivers (phase (5))

Planning DAG products must run on the already-built runtime; no planning path may rebuild driver internally.

**Files MODIFIED**

* `src/codeintel/build/hamilton/planner.py`
* `src/codeintel/cli/handlers/build.py`
* `src/codeintel/serving/api/plan.py`

---

# 5) Serving runtime lifecycle: build once at startup, not per request

### Files MODIFIED

* `src/codeintel/serving/app.py` (or equivalent FastAPI entry)

  * On startup:

    * `runtime = runtime_registry.get_or_create(runtime_key, lambda: compose_runtime(env, cfg))`
    * store `app.state.runtime = runtime`
  * Provide FastAPI dependency:

    * `def get_runtime(request) -> RuntimeBundle: return request.app.state.runtime`

### Code snippet (dependency injection shape)

```python
# serving/deps.py (illustrative)
from fastapi import Request

def get_runtime(req: Request) -> RuntimeBundle:
    return req.app.state.runtime

@router.get("/targets")
def list_targets(runtime: RuntimeBundle = Depends(get_runtime)):
    return sorted(runtime.catalog.targets.keys())
```

### Files CREATED

5. `src/codeintel/serving/deps.py` (if you don’t already have it)

### Critical behavioral constraint

* Ban any endpoint logic that calls `compose_runtime()` directly.
* The only allowed composition invocation is app startup (or an explicit admin “reload runtime” endpoint guarded by auth).

---

# 6) Add snapshotting: serialize derived products so serving can avoid driver rebuild entirely

This is optional but synergistic: if your serving tier does not need to execute Hamilton computations (only serve artifacts/tables), it can load precompiled metadata without importing Hamilton.

### Files CREATED

6. `src/codeintel/runtime/snapshot.py`

* `RuntimeSnapshot` artifacts:

  * `ci.runtime.catalog.json`
  * `ci.runtime.schema_index.json` (or duckdb table)
  * `ci.runtime.semantic_registry.json`

* `write_snapshot(runtime: RuntimeBundle, store: ArtifactStore) -> None`

* `load_snapshot(store, key) -> SnapshotBundle`

### Files MODIFIED

* `src/codeintel/build/hamilton/native/export/serving_artifacts.py`

  * If this is a “build/export target”, make it emit snapshot artifacts via saver nodes (aligns with “DAG products” pattern).

> This makes “serving” a pure consumer of artifacts and eliminates the last reason to rebuild drivers outside the build process.

---

# 7) Enforce boundaries with static + runtime guards

## 7.1 Static enforcement (AST linter)

### Files CREATED

7. `tools/lint_no_driver_build_in_nodes.py`

Scan Python AST under forbidden roots and reject:

* imports of `codeintel.runtime.compose`, `codeintel.build.hamilton.driver_factory`
* usage of `hamilton.driver.Builder`, `DriverBuilder`, `Driver`
* calls named `build_driver`, `compose_runtime`, `discover_modules`

Minimal pattern:

```python
DISALLOWED_IMPORTS = {
  "codeintel.runtime.compose",
  "codeintel.build.hamilton.driver_factory",
  "hamilton.driver",
}
DISALLOWED_CALLS = {"build_driver", "compose_runtime", "DriverBuilder", "Builder", "Driver"}
FORBIDDEN_ROOTS = ["src/codeintel/build/hamilton/native", "src/codeintel/build/hamilton/nodes"]
```

Wire into CI (pytest or pre-commit).

### Files MODIFIED

* `pyproject.toml` or `tox.ini` or CI workflow to execute the linter.

## 7.2 Runtime guard (belt-and-suspenders)

### Files MODIFIED

* `src/codeintel/runtime/compose.py`

  * Set a thread-local “composition in progress” flag during runtime build and clear after.
* `src/codeintel/build/hamilton/executor.py`

  * Set a thread-local “DAG execution in progress” flag; composition root asserts it is not called under DAG execution.

This catches accidental re-entrancy during tests even if static guard misses a dynamic call.

---

# 8) Consolidate “module discovery” into composition root exclusively

If module discovery currently happens in multiple contexts (CLI, serving, export tasks), centralize it.

### Files CREATED

8. `src/codeintel/runtime/module_resolver.py`

* `resolve_modules(cfg: RuntimeConfig) -> tuple[ModuleType, ...]`
* Implements:

  * filesystem discovery
  * plugin entrypoints (if you add them later)
  * deterministic ordering + digest

### Files MODIFIED

* Any call site performing discovery moves to `compose_runtime()`.

---

# 9) Tests: prove single composition root and no in-DAG driver rebuild

### Files CREATED

* `tests/runtime/test_no_driver_build_in_nodes.py`

  * Executes the AST linter (or imports and runs it) as a test gate.

* `tests/runtime/test_compose_runtime_idempotent.py`

  * Compose runtime twice with same config key; assert `RuntimeRegistry` returns same instance or at least same fingerprint and no additional module discovery executed (spy/mocks).

* `tests/runtime/test_execute_does_not_compose.py`

  * Execute a DAG target; assert:

    * compose function was not called
    * driver_factory builder was not invoked
  * Use monkeypatch to raise if called.

### Files MODIFIED

* Any existing tests that call `build_driver` inside helper functions should be rewritten to use `compose_runtime` once per test session fixture and pass `RuntimeBundle`.

---

# 10) File index summary (P0)

## Created

* `src/codeintel/runtime/runtime_bundle.py`
* `src/codeintel/runtime/compose.py`
* `src/codeintel/runtime/registry.py`
* `src/codeintel/runtime/inputs.py`
* `src/codeintel/runtime/snapshot.py` (optional but recommended)
* `src/codeintel/runtime/module_resolver.py`
* `src/codeintel/serving/deps.py` (if needed)
* `tools/lint_no_driver_build_in_nodes.py`
* `tests/runtime/test_no_driver_build_in_nodes.py`
* `tests/runtime/test_compose_runtime_idempotent.py`
* `tests/runtime/test_execute_does_not_compose.py`

## Modified (core)

* `src/codeintel/build/hamilton/executor.py`
* `src/codeintel/build/hamilton/planner.py`
* `src/codeintel/cli/handlers/build.py`
* `src/codeintel/serving/app.py`
* `src/codeintel/build/serving/semantic_compile.py`
* `src/codeintel/build/schemas/{schema_index,inference_service,provider_unified}.py`
* `src/codeintel/build/spec/compile.py`
* `src/codeintel/build/assets/emitter.py`
* `src/codeintel/build/hamilton/native/export/serving_artifacts.py`
* `src/codeintel/build/hamilton/driver_factory.py` (or delete after migration)

## Deleted (targeted; depends on what remains after earlier phases)

* Any “secondary factory” modules that exist solely to build drivers in subcomponents (e.g., `*_driver_utils.py`, `serving_driver_factory.py`).

> If exact filenames differ, the sweep is still deterministic: delete/replace any module whose sole responsibility is “build a driver” outside `codeintel.runtime.compose`.

---

# 11) Definition of Done (hard gates)

1. **Static linter passes**: no disallowed imports/calls exist under DAG node directories.
2. Runtime composition is invoked **only** from:

   * CLI entrypoints (build/export commands)
   * serving startup initialization
   * explicit admin reload (if implemented)
3. All “plan/explain/semantic registry/schema index/buildspec/export” computations consume `RuntimeBundle` or `DagCatalog` and do not rebuild drivers.
4. Executing any target cannot trigger driver rebuild:

   * verified by `test_execute_does_not_compose` + runtime guard.
5. Serving endpoints never rebuild the driver per request:

   * runtime is stored in `app.state` and reused.
6. Optional but strongly recommended: serving can run without Hamilton by loading `RuntimeSnapshot` artifacts (catalog/registry) when compute is not needed.

---

## Architectural payoff

Once this lands, the system’s *construction semantics* become trivially auditable:

* **Driver build** is a pure function of `(env, config, discovered_modules)` and happens exactly once per runtime key.
* All DAG execution is done against a stable, immutable runtime bundle.
* All derived metadata (catalog/schema/registry/plan) becomes a first-class artifact pipeline rather than ad hoc recomputation.

This is the prerequisite for “production-grade” coherence: correctness bugs stop coming from “which driver did we build in which codepath,” because there is only one driver.
