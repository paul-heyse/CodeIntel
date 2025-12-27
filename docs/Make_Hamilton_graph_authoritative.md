
According to a document from **December 26, 2025** (your `architecture_ph6.md`), you’ve already achieved the *right* north star: “the DAG is the truth,” with stable naming, tag-driven target compilation, saver-derived IO surfaces, manifest-driven incrementality, and a generated support module that makes reads/writes DAG-visible. The biggest remaining wins are now about **eliminating the last “two sources of truth,” collapsing bespoke plumbing into Hamilton-native mechanisms, and turning your derived metadata into a single, reusable “catalog” that every surface (build, export, serving, MCP) consumes**.

Below are the **highest-leverage opportunities** I see to tighten integration, dramatically consolidate code, and harden extensibility.

---

## 1) Kill the “dual graph world”: make Hamilton graph + tags the only authoritative structure

Your architecture doc explicitly calls out a lingering duplication: targets exist both as `TargetGraph` objects and as DAG nodes (`t__*`). That duplication creates drift pressure and forces glue code (closure computation, dependency propagation, mapping dictionaries, etc.).

You can push further and make the *Hamilton graph* the single source of truth for:

* Target inventory (what targets exist)
* Dependencies and closure computation
* IO surfaces (reads/writes)
* What is “servable” / “MCP-visible” / “semantic” (via tags)

Today, you still:

* build a **base TargetGraph**, then build an enriched graph after introspection, then build a final driver with support module,
* compute closure via `TargetGraph.topological_order(...)`,
* and maintain runtime mappings (`target_to_node`, `node_to_target`) as state.

### Breaking-change recommendation

Create one canonical object, something like `DagCatalog`, built **once per driver build**, that exposes:

* `targets`: derived from tag-filtering anchor/materialize nodes (no separate registration)
* `deps(target)`: derived from Hamilton node dependencies (or a cached adjacency extracted once)
* `writes(target)` / `reads(target)`: derived from saver tags + loader tags (already derivable via introspection)
* `node_index`: parsed `TagSpec` per node (so every consumer uses the same tag parsing rules)

Then:

* Delete (or severely shrink) the parts of `TargetGraph` that duplicate graph semantics.
* Make **closure** a method on `DagCatalog` that uses the Hamilton dependency graph (or an extracted adjacency), not a separate graph structure.

This one move tends to cascade into big simplifications in planner, executor, validation, and serving compilation.


## 2) Replace bespoke support-module “function factories” with Hamilton parameterization primitives

Your support module generation is already a strong design: dataset refs/loaders/artifact refs become DAG-visible and are derived from the native graph + saver-derived outputs.

But the current implementation relies on:

* dynamic module construction,
* per-node function factories,
* manual signature injection,
* a lot of bespoke dependency wiring (`**kwargs` patterns for loader nodes).

### Opportunity: use Hamilton’s own “generate many nodes from one template” facilities

Hamilton’s function modifiers let you generate many nodes from one function using parameterization patterns (e.g., `parameterize_sources`, `parameterize_values`, `source(...)`, `value(...)`) and select behaviors via `resolve_from_config`.

**Concrete refactor direction**
Instead of programmatically attaching N functions, define *one* template per support-node kind:

* `dataset_ref_template(record: TargetRunRecord, table_key: str) -> DatasetRef`
* `query_loader_template(env: BuildEnv, ref: DatasetRef) -> ibis.Table`
* `df_loader_template(env: BuildEnv, ref: DatasetRef) -> DataFrame`
* `artifact_ref_template(record: TargetRunRecord, artifact_name: str) -> ArtifactRef`

Then generate all nodes via parameterization:

* each `d__{table_key}` points `record=source(t__producer_target)` and `table_key=value(table_key)`
* each `q__{table_key}` points `ref=source(d__{table_key})`
* etc.

This removes:

* manual signature mutation,
* `**kwargs` dependency capture,
* most of the custom module-attach surface (you can keep it for templates where it truly adds value).

Net effect: **support nodes become declarative**, and the codebase loses a lot of “meta-programming scaffolding.”


Here’s the **paired narrative** for the two changes—i.e., what the system *becomes*, how the build/execution path feels end-to-end, and what conceptual “kinks” we’re straightening out.

---

## 1) “Kill the dual graph world” — what’s changing in plain terms

### Before (today’s mental model)

You effectively have **two parallel representations** of “what exists and how it depends on what”:

1. **Hamilton’s FunctionGraph** (node→node deps, tags, variables), which is the *actual* execution substrate.
2. A bespoke **TargetGraph / TargetSystem / TagIndex** world that:

   * re-describes targets and dependencies (target→target),
   * maintains mappings like `target_to_node` / `node_to_target`,
   * provides closure/toposort, IO surfaces, serving views, etc.

Even if the TargetGraph is *derived* from the DAG, it becomes a second “truthy” structure because planners/executors/serving code paths accept it as input, cache it, and re-derive things again from it. That introduces drift pressure and requires glue code to keep both worlds aligned.

### After (new mental model)

There is **exactly one authoritative structure**:

> **Hamilton graph + node tags** are the only truth about structure.

Everything else is a *derived, immutable catalog* extracted from the Hamilton graph once per driver-build:

* **`DagCatalog` is not a “graph”** you maintain; it’s a *read-only index* over Hamilton nodes and tags.
* It contains the minimal derived objects the rest of the system needs:

  * `TargetDescriptor` (derived from `t__*` anchors + tags)
  * `OutputDescriptor` (derived from `m__*` saver tags)
  * `IOSurface` (reads/writes derived from tag reachability upstream of anchors)
  * `closure()` (toposorted target dependency closure computed from Hamilton edges)

So your architecture becomes:
**Hamilton graph is the source**, `DagCatalog` is the *compiled view*.

### What “targets” become

A “target” is no longer something you register in a TargetGraph—**it exists iff a `t__*` node exists and is tagged**.

* Target identity and metadata: parsed only from anchor tags (`target`, `domain`, `spec_version`, etc.).
* Target dependencies: derived by walking upstream from the anchor node across Hamilton edges until encountering other anchors, collapsing node→node into target→target.
* This means **target closure/toposort is never computed from a separate structure**—it’s always computed from Hamilton graph edges (via the catalog’s derived adjacency).

### What “IO surfaces” become

IO surfaces stop being “target-declared” objects that can diverge from behavior. Instead they are derived:

* **Writes**: saver nodes (`hamilton.data_saver == True`) tagged with `output_role="contract"` + `table_key` or `artifact`.
* **Reads**: table_keys discovered via upstream loader/dataset nodes reachable from the anchor (with stop conditions at other anchors).

Serving, schema inference, preflight, and contract checks all pivot to “IO is what tags say,” not “IO is what a separate model says.”

### What this *feels like* for the rest of the codebase

Planner/executor/serving/CLI/state computation stop passing around a `TargetGraph` and instead take a single object:

* `HamiltonRuntime(driver, catalog)`

Everywhere that previously needed:

* closure/toposort,
* dependency lists,
* IO surface,
* producer-by-table_key indexes,

…now reads those from `catalog` (or from `catalog.targets[target]`).

The payoff is not just conceptual purity—it’s deletion: you stop maintaining an extra graph, extra mappings, extra indexes, and a big family of helper functions whose only job was reconciling those.

---

## 2) “Replace bespoke support-module factories” — what’s changing in plain terms

### Before (today’s mental model)

Support nodes (the `d__*`, `q__*`, `df__*`, `a__*`, `p__*` family) are created by a **dynamic module generator**:

* You introspect the graph / targets / outputs.
* You generate N Python callables in-memory.
* You mutate their signatures so Hamilton recognizes dependencies.
* You attach them into a `ModuleType`, inject it into `sys.modules`, and rebuild the driver.

This works, but it is bespoke meta-programming with a lot of failure modes:

* signature mutation and dependency wiring are easy to break,
* introspection is harder because the module is synthetic,
* test failures become “did codegen emit the function correctly?” rather than “does the DAG describe what we want?”

### After (new mental model)

Support nodes become a **static, checked-in Hamilton module** with **template functions**.

Instead of generating N functions, you define *one* function per support-node kind, and let Hamilton expand it into many nodes at compile time via **parameterization primitives**:

* `@resolve_from_config(decorate_with=...)` chooses whether to expand and how.
* `@parameterize(...)` emits many node variants from one function by binding:

  * dependency inputs via `source(<upstream_node_name>)`
  * constants via `value(<literal>)`

So support node generation is no longer “codegen”; it becomes “compile-time expansion driven by config.”

### How the inventory is produced (important connection to (1))

With (1), you now have `DagCatalog` as the canonical derived view of:

* which targets exist,
* which contract outputs exist (table_keys, artifact names, producer targets).

So driver construction becomes:

1. Build native driver (targets, materializers).
2. Compile catalog from it.
3. Derive a **SupportNodeSpec** (dataset/artifact lists) from catalog outputs.
4. Rebuild final driver including the static `support_nodes` module + config containing the spec.

In other words: **catalog → support spec → parameterized support nodes**.

No dynamic module injection required; no signature surgery required.

### Tagging and discoverability improves (and becomes uniform)

One subtle but major win: per-node tags are applied using Hamilton’s `target_=` tag targeting in decorators during expansion. That means:

* Every expanded `d__X` node has canonical `node_type=dataset`, `table_key=X`, `target=<producer>`, `domain=<domain>` tags.
* Every expanded `q__X` / `df__X` node has canonical loader tags.
* Every expanded `a__Y` / `p__Y` node has canonical artifact tags.

So the catalog compiler, serving compilation, and any future “semantic registry” logic can treat support nodes exactly like native nodes: they’re just Hamilton nodes with tags.

### What this *feels like* operationally

Support nodes stop being a magical generated module; they become:

* something you can open and read,
* something linters/type checkers can analyze,
* something that fails deterministically in the same place every time (driver compile) if config is wrong.

And critically: their behavior becomes “pure Hamilton semantics,” which makes the DAG more transparent to tooling and to AI agents.

---

## The combined “new flow” narrative (the whole pipeline)

### Driver build becomes the single composition root

The whole pipeline collapses into a stable, repeatable shape:

1. **Assemble native modules** (targets, materializers, export, serving artifacts).
2. Build a **native Driver**.
3. Compile **`DagCatalog`** from the Driver graph:

   * anchors → targets,
   * savers → outputs,
   * upstream reachability → reads/writes + deps.
4. Compute **SupportNodeSpec** from catalog’s contract outputs.
5. Build **final Driver** including `support_nodes` module + config (spec + feature flags).
6. Compile **final DagCatalog** (now includes support nodes too, but targets still anchored by `t__*`).

Runtime now is simply: `runtime = (driver, catalog)`.

### Planning and execution simplify

Planner:

* asks catalog for closure/toposort
* asks catalog for IO surface
* produces plan + decision trace

Executor:

* runs anchor nodes for requested targets
* uses catalog to compute dependency hashes and IO expectations
* uses tag-derived saver outputs to determine “what got produced”

Serving compilation:

* enumerates semantic views / MCP-visible objects using catalog tag queries
* uses `table_outputs` indexes for producer/lineage mapping

Everything is “DAG + tags → derived catalog → consumption,” and there are no side structures competing for authority.

---

## The “why this matters” narrative (what pain we’re explicitly removing)

* **Drift elimination:** no more “TargetGraph says X but Hamilton graph says Y.”
* **Deletion leverage:** most bespoke index/mapping logic becomes redundant once `DagCatalog` exists.
* **Extensibility becomes mechanical:** adding a new computation is:

  * add a `t__*` anchor and tagged savers,
  * rerun compilation,
  * catalog and support nodes update automatically.
* **Debuggability improves:** because the graph you see (Hamilton) is the graph you execute, and the catalog is a deterministic compilation artifact.

If you want, the next helpful narrative pairing is for (3): “make saver-derived outputs the only output inventory,” because it dovetails directly with the catalog/output indexing story and further reduces duplication in contracts/specs.

---

Below is a **repo-concrete, breaking-change–friendly** implementation plan for:

> **(1) Kill the “dual graph world”: make Hamilton graph + tags the only authoritative structure**

End state: **Hamilton Driver graph is the sole dependency truth**; everything else is a **derived, immutable catalog** built from `Driver.graph.nodes[*].tags` + node deps. No `TargetGraph`, no separate tag index, no `target_to_node` lookup tables as primary state.

---

## 0) End-state contract (what must be true)

### Authoritative structure

* **Only** `hamilton.driver.Driver.graph` (FunctionGraph) determines:

  * dependency edges (node→node)
  * target closure (target→target, derived *from* node edges)
* Targets exist **only** because a `t__*` node is present and tagged:

  * `TAG_NODE_TYPE == NODE_TYPE_MATERIALIZE`
  * `TAG_TARGET` (string), `TAG_DOMAIN` (string), `TAG_TARGET_SPEC_VERSION` (string/int/… per existing validator)
* Everything consumed by planner/executor/serving is derived from:

  * target-anchor tags (`t__*`)
  * saver tags (`m__*`, `hamilton.data_saver == True`, `output_role in {"contract","internal"}`, plus `TAG_TABLE_KEY` or `TAG_ARTIFACT`)
  * loader/dataset tags (`q__/df__/d__` etc, `TAG_TABLE_KEY`, `TAG_NODE_TYPE`)

### Only derived caches allowed

* A “catalog” object may store precomputed:

  * `targets` (metadata)
  * `deps` (target-level adjacency derived from node-level graph)
  * `io_surface` (reads/writes derived from tags)
  * `output indexes` (table_key→producer target, artifact→producer target)
* But: **no second graph** with independent edge storage; the catalog is a *view* of the Hamilton graph.

---

## 1) New canonical artifact: `DagCatalog` (single object everyone consumes)

### Files created

1. `src/codeintel/build/hamilton/dag_catalog.py`
   **Implement** minimal immutable descriptors + catalog API (slots/frozen).

   * `NodeDescriptor(name, deps, tags)`
   * `TargetDescriptor(name, domain, module?, anchor_node, deps, resources/execution/parameters/description, io, outputs)`
   * `OutputDescriptor(kind={"table","artifact"}, key, role, producer_target, saver_node, sink, artifact_path_template?)`
   * `IOSurface(read_table_keys, write_tables, write_artifacts)` (you can keep richer `TableRead/TableWrite` forms if you need provenance)
   * `DagCatalog(nodes, targets, table_outputs, artifact_outputs, …)` with:

     * `closure(targets: Sequence[str]) -> tuple[str, ...]` (toposorted deps-first)
     * `target_node(target: str) -> str` (returns anchor node name; eliminates `target_to_node` dict)
     * `producer_of_table(table_key) / producer_of_artifact(name)`
     * `find_nodes(tag_key, tag_value=None)` (replaces TagIndex use-cases)

2. `src/codeintel/build/hamilton/dag_catalog_compiler.py`
   **Implement** pure compiler: `compile_dag_catalog(driver: Driver, *, strict: bool=True) -> DagCatalog`

   * Extract `NodeDescriptor` for all nodes: `node.name`, `[dep.name for dep in node.dependencies]`, `node.tags`
   * Identify anchors (`t__*`): tag-based, **not** name-based (name convention is helpful but non-authoritative)
   * Compile `TargetDescriptor`:

     * reuse/refactor existing parsing logic from `target_spec_compiler.py` (domain/resources/execution/parameters/spec_version/description) but **no TargetGraph output**
   * Derive **target→target deps** by collapsing through non-materialize nodes:

     * start from anchor node deps; DFS/BFS upstream until encountering other anchors; stop traversal past encountered anchor(s)
     * enforce “one anchor per target” invariant (existing runtime already assumes this)
   * Derive outputs from saver tags (contract role):

     * scan nodes where `hamilton.data_saver is True` and `output_role == "contract"`
     * exactly one of (`TAG_TABLE_KEY`, `TAG_ARTIFACT`) must be present
     * artifact outputs require `TAG_ARTIFACT_PATH_TEMPLATE`
     * build global uniqueness indexes:

       * `table_key` unique across contract outputs
       * `artifact_name` unique across contract outputs
   * Derive IO surface per target:

     * writes: from saver tags filtered by `TAG_TARGET == target` and `output_role == "contract"`
     * reads: from loader/dataset tags reachable upstream of anchor (stop at other anchors), collecting `TAG_TABLE_KEY`
   * Emit `DagCatalog` with frozen mappings (`MappingProxyType`) for stability.

### Files modified (refactor-only)

* `src/codeintel/build/hamilton/target_spec_compiler.py`
  Convert from “compile `OutputTarget` objects” to “compile `TargetDescriptor` fields”.

  * Keep: tag parsing helpers, overrides, validation structure
  * Delete/stop emitting: `TargetGraph`, `OutputTarget`
  * New public entrypoint: `compile_target_descriptors_from_driver(driver, *, strict=True, overrides=...) -> tuple[TargetDescriptor,...]`

* `src/codeintel/build/hamilton/validate.py`
  Remove any **parity checks** or APIs that accept `(base_graph: TargetGraph)`; validator becomes purely “Hamilton graph + tags” invariant gate.

---

## 2) Remove `TargetGraph` as a type and as a runtime input

### Files deleted

* `src/codeintel/build/hamilton/tag_index.py`
  Replaced by `DagCatalog.find_nodes(...)` + direct tag scanning in catalog compiler.
* `tests/build/test_targets.py`
  Entirely TargetGraph-centric; replaced by catalog tests (below).

> If you want to go harder: you can delete `TargetGraph` itself from `src/codeintel/build/targets.py`, but you don’t need to delete the file—just delete the class + all graph logic.

### Files modified

* `src/codeintel/build/targets.py`

  * Keep `OutputTarget` only if you still want “target metadata” as a standalone type; otherwise **replace** it with `TargetDescriptor` import/export.
  * **Remove `TargetGraph`** class and its associated helpers (`topological_order`, `dependents_of`, `validate`, etc.). Those responsibilities move to `DagCatalog`.

* `src/codeintel/build/hamilton/runtime.py`

  * Replace `graph: TargetGraph` with `catalog: DagCatalog`
  * Remove `target_to_node` / `node_to_target` fields entirely; use `catalog.target_node(target)` and reverse lookup via tags if needed.

* `src/codeintel/build/hamilton/driver_factory.py`

  * Stop building any `TargetGraph`.
  * New flow:

    1. build native driver (native modules only)
    2. compile minimal catalog from native driver (needed for support-module generation boundaries)
    3. build support module (still allowed, but it consumes catalog outputs, not a TargetGraph)
    4. build final driver (native + support module + adapters)
    5. compile final catalog (includes support nodes, but target anchors remain the authoritative set)
    6. return `HamiltonRuntime(dr=..., catalog=...)`
  * Update `adapter_factory` signature: `Callable[[DagCatalog], Sequence[LifecycleAdapter]]`

* `src/codeintel/build/__init__.py`

  * Replace lazy exports: remove `TargetGraph`; export `DagCatalog` (+ descriptors if you expose them)

---

## 3) Planner/executor/state: swap `TargetGraph` → `DagCatalog` everywhere

### Hashing and skip logic (must change because dependencies no longer live in TargetGraph)

#### Files modified

* `src/codeintel/build/hashing.py`

  * Change hashing surface to accept `(target_name, deps)` or a `TargetDescriptor` that already includes `deps`.
  * The hashing algorithm itself stays identical: it still hashes `{dep_name:dep_input_hash}`; only the dep-source changes.

* `src/codeintel/build/hamilton/run_records.py`

  * `compute_target_input_hash*` should resolve dependencies via catalog (or descriptor.deps), not via `target.dependencies` in a graph.
  * `expected_*` helpers should use catalog outputs (table_outputs/artifact_outputs) as canonical inventory.

* `src/codeintel/build/hamilton/native/executor.py`

  * `NativeTargetExecutor.for_target(env, catalog, target_name, …)`
  * Resolve the target descriptor from catalog; resolve deps from descriptor; compute hashes/skip unchanged.

### Planning

#### Files modified

* `src/codeintel/build/hamilton/planner.py`

  * Replace any `TargetGraph` usage:

    * closure = `catalog.closure(requested)`
    * per-target metadata from `catalog.targets[target]`
  * Remove `target_graph_from_hamilton` call entirely (that function/class disappears).

### Execution

#### Files modified

* `src/codeintel/build/hamilton/executor.py`

  * Replace all `runtime.graph` with `runtime.catalog`
  * Replace `target_to_node_name(...)` mapping usage with `runtime.catalog.target_node(target)`
  * Execution inputs: pass `inputs={"env": execution_env, "catalog": runtime.catalog}` (not `"graph"`)
  * IO discovery: stop calling `derive_target_io_surface(runtime)`; use `catalog.targets[target].io` (precomputed) or a `catalog.io_surface(target)` accessor.

### State computation / policy

#### Files modified

* `src/codeintel/build/state.py`
* `src/codeintel/build/state_computer.py`
* `src/codeintel/build/execution_policy.py`
* `src/codeintel/build/hash_evaluator.py` (if it takes graph/targets)

All of these must be refactored to accept `DagCatalog` and use `catalog.closure()` / `descriptor.deps` for traversal.

---

## 4) Native Hamilton nodes: swap `graph: TargetGraph` input to `catalog: DagCatalog`

This is the largest mechanical change but also the biggest coherence win: nodes stop depending on a parallel structure.

### Canonical signature rule

* Any node that currently takes `graph: TargetGraph` must take:

  * `catalog: DagCatalog` (name “catalog” is deliberate; “graph” is overloaded in your codebase)

### Files modified (complete list from repo scan)

Update parameter typing + any internal lookups in:

**Native patterns / infra**

* `src/codeintel/build/hamilton/native/patterns/tool_target.py`
* `src/codeintel/build/hamilton/native/materialization_records.py`
* `src/codeintel/build/hamilton/native/executor.py`

**Native ingestion**

* `src/codeintel/build/hamilton/native/ingestion/scip.py`
* `src/codeintel/build/hamilton/native/ingestion/scip_proto.py`
* `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
* `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`

**Native graphs**

* `src/codeintel/build/hamilton/native/graphs/call_graph.py`
* `src/codeintel/build/hamilton/native/graphs/import_graph.py`
* `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`
* `src/codeintel/build/hamilton/native/graphs/graph_targets.py`

**Native analytics**

* `src/codeintel/build/hamilton/native/analytics/*` (all files currently typed with TargetGraph; see list below)

  * `classification_targets.py`
  * `config_graph_targets.py`
  * `coverage_targets.py`
  * `dependency_targets.py`
  * `function_detail_targets.py`
  * `function_metrics.py`
  * `hotspots.py`
  * `metadata_targets.py`
  * `metrics_targets.py`
  * `risk_factors.py`
  * `subsystem_cache_targets.py`
  * `subsystem_targets.py`

**Native export**

* `src/codeintel/build/hamilton/native/export/decision_trace.py`
* `src/codeintel/build/hamilton/native/export/export_targets.py`
* `src/codeintel/build/hamilton/native/export/serving_artifacts.py`

**Materializers / hooks**

* `src/codeintel/build/hamilton/materializers/base.py`
* `src/codeintel/build/hamilton/materializers/duckdb_saver.py`
* `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py`
* `src/codeintel/build/hamilton/materializers/artifact_saver.py`
* `src/codeintel/build/hamilton/hooks/__init__.py`
* `src/codeintel/build/hamilton/hooks/contract_hook.py`
* `src/codeintel/build/hamilton/contracts/check_target_contracts.py`

**Support module generator (must stop consuming TargetGraph)**

* `src/codeintel/build/hamilton/nodes/support_factory.py`

---

## 5) Target metadata + serving compilation: consume catalog, not tag index + target graph

### Files modified

* `src/codeintel/build/target_metadata.py`

  * `TargetSystem.graph` → `TargetSystem.catalog`
  * indexes (`by_name`, `by_table_key`, `by_artifact_name`) derived from catalog outputs
  * `closure()` delegates to `catalog.closure()`
  * remove `TagIndex` creation; swap to `catalog.find_nodes(...)` patterns

* `src/codeintel/build/serving/semantic_compile.py`

  * Replace `compile_semantic_registry(tag_index: TagIndex)` with `compile_semantic_registry(catalog: DagCatalog)`
  * semantic view discovery becomes: scan nodes where `TAG_OUTPUT_KIND == OUTPUT_KIND_SEMANTIC_VIEW` and `TAG_MCP_VISIBLE == "1"` and collect `TAG_TABLE_KEY` (+ semantic tag payload)

* `src/codeintel/build/hamilton/native/export/serving_artifacts.py`

  * Stop constructing `TagIndex.from_modules`; use a catalog compiled from a Driver built with those modules (or directly walk driver nodes by tag).

* `src/codeintel/core/registry/service.py`

  * `targets = {t.name: t for t in get_target_system().graph.all_targets}` becomes `get_target_system().catalog.targets`

---

## 6) CLI + schema inference: graph parameter renamed and rewired

### Files modified

* `src/codeintel/cli/handlers/build.py`

* `src/codeintel/cli/handlers/ops.py`

  * anywhere you call `derive_target_io_surface(...)` or pass `graph` into driver execution becomes catalog-based.

* `src/codeintel/build/schemas/inference_service.py`

  * replace `graph.all_targets` loops with `catalog.targets.values()`
  * `_producers_by_table_key(graph)` becomes `_producers_by_table_key(catalog)` using `catalog.table_outputs`

* `src/codeintel/build/schemas/compile.py`

* `src/codeintel/build/spec/compile.py`

* `src/codeintel/build/assets/emitter.py`

  * all these currently accept/require TargetGraph; migrate to DagCatalog indexes.

---

## 7) Tests: delete TargetGraph tests, add DagCatalog tests

### Files created

* `tests/build/hamilton/test_dag_catalog_compiler.py`

  * Asserts:

    * duplicate anchors rejected
    * closure(toposort) stable/deterministic
    * saver-derived output inventory uniqueness enforced
    * read/write surfaces correctly derived on representative fixture DAG (use a tiny synthetic module)

* `tests/build/test_hashing_with_catalog_deps.py`

  * Ensures input hash changes iff dep manifest hash changes (same behavior as today)

### Files modified

(Replace any helper that constructs a TargetGraph.)

* `tests/_helpers/build.py`
* `tests/_helpers/hamilton_execution.py`
* `tests/build/hamilton/conftest.py`
* `tests/build/hamilton/test_pr09_planner.py`
* `tests/build/hamilton/test_materializer.py`
* `tests/build/hamilton/test_graph_targets.py`
* `tests/build/hamilton/test_metrics_targets.py`
* `tests/build/hamilton/test_coverage_targets.py`
* `tests/build/hamilton/test_schema_index_overrides.py`
* `tests/build/test_state.py`
* `tests/build/test_state_computer.py`
* `tests/build/test_hashing_plan_targets.py`
* `tests/build/test_contracts_parameters_state.py`
* `tests/build/test_registry_has_no_static_dependencies.py`
* `tests/build/hamilton/test_pr10_manifest_index.py`
* `tests/build/hamilton/validators/test_validation_poc.py`
* `tests/build/hamilton/native/test_skip_logic.py` (if it asserts graph presence)

### Files deleted

* `tests/build/test_targets.py` (replaced by catalog tests)

---

## 8) Final “files modified / created / deleted” index (single place)

### Created

* `src/codeintel/build/hamilton/dag_catalog.py`
* `src/codeintel/build/hamilton/dag_catalog_compiler.py`
* `tests/build/hamilton/test_dag_catalog_compiler.py`
* `tests/build/test_hashing_with_catalog_deps.py`

### Deleted

* `src/codeintel/build/hamilton/tag_index.py`
* `tests/build/test_targets.py`

### Modified (core set)

* `src/codeintel/build/targets.py` (remove TargetGraph; keep/adjust OutputTarget or replace with TargetDescriptor)
* `src/codeintel/build/__init__.py`
* `src/codeintel/build/target_metadata.py`
* `src/codeintel/build/hashing.py`
* `src/codeintel/build/state.py`
* `src/codeintel/build/state_computer.py`
* `src/codeintel/build/execution_policy.py`
* `src/codeintel/build/schemas/inference_service.py`
* `src/codeintel/build/schemas/compile.py`
* `src/codeintel/build/spec/compile.py`
* `src/codeintel/build/assets/emitter.py`
* `src/codeintel/core/registry/service.py`
* `src/codeintel/cli/handlers/build.py`
* `src/codeintel/cli/handlers/ops.py`
* `src/codeintel/build/serving/semantic_compile.py`

### Modified (Hamilton subsystem)

* `src/codeintel/build/hamilton/runtime.py`
* `src/codeintel/build/hamilton/driver_factory.py`
* `src/codeintel/build/hamilton/target_spec_compiler.py`
* `src/codeintel/build/hamilton/validate.py`
* `src/codeintel/build/hamilton/planner.py`
* `src/codeintel/build/hamilton/executor.py`
* `src/codeintel/build/hamilton/run_records.py`
* `src/codeintel/build/hamilton/nodes/support_factory.py`
* `src/codeintel/build/hamilton/contracts/check_target_contracts.py`
* `src/codeintel/build/hamilton/materializers/{base,duckdb_saver,duckdb_rows_saver,artifact_saver}.py`
* `src/codeintel/build/hamilton/hooks/{__init__,contract_hook}.py`
* all `src/codeintel/build/hamilton/native/**` files listed in §4

### Modified (tests)

* all files listed in §7 “Files modified”

---

## 9) Definition of Done (hard checks)

1. `TargetGraph` type is absent from runtime: no import path or execution input named `graph` representing a target graph.
2. All dependency closure used by planner/executor/state comes from `DagCatalog.closure()` (derived from Hamilton graph).
3. Single authoritative output inventory: `DagCatalog.table_outputs` and `DagCatalog.artifact_outputs` (derived from saver tags) drive:

   * schema inference candidate discovery
   * contract checks
   * serving semantic compilation
4. Native nodes compile and run with `inputs={"env": ..., "catalog": runtime.catalog}`.
5. No stale imports: `python -m compileall` passes + unit test suite passes.

If you want the next increment after this: I can produce the same style plan for **(2) replace bespoke support-module function factories with Hamilton parameterization primitives**, but the above fully scopes “dual graph world elimination” end-to-end.



According to a document from **December 26, 2025** (your `architecture_ph6.md`), you’ve already achieved the *right* north star: “the DAG is the truth,” with stable naming, tag-driven target compilation, saver-derived IO surfaces, manifest-driven incrementality, and a generated support module that makes reads/writes DAG-visible. The biggest remaining wins are now about **eliminating the last “two sources of truth,” collapsing bespoke plumbing into Hamilton-native mechanisms, and turning your derived metadata into a single, reusable “catalog” that every surface (build, export, serving, MCP) consumes**.

Below are the **highest-leverage opportunities** I see to tighten integration, dramatically consolidate code, and harden extensibility.

---

## 1) Kill the “dual graph world”: make Hamilton graph + tags the only authoritative structure

Your architecture doc explicitly calls out a lingering duplication: targets exist both as `TargetGraph` objects and as DAG nodes (`t__*`). That duplication creates drift pressure and forces glue code (closure computation, dependency propagation, mapping dictionaries, etc.).

You can push further and make the *Hamilton graph* the single source of truth for:

* Target inventory (what targets exist)
* Dependencies and closure computation
* IO surfaces (reads/writes)
* What is “servable” / “MCP-visible” / “semantic” (via tags)

Today, you still:

* build a **base TargetGraph**, then build an enriched graph after introspection, then build a final driver with support module,
* compute closure via `TargetGraph.topological_order(...)`,
* and maintain runtime mappings (`target_to_node`, `node_to_target`) as state.

### Breaking-change recommendation

Create one canonical object, something like `DagCatalog`, built **once per driver build**, that exposes:

* `targets`: derived from tag-filtering anchor/materialize nodes (no separate registration)
* `deps(target)`: derived from Hamilton node dependencies (or a cached adjacency extracted once)
* `writes(target)` / `reads(target)`: derived from saver tags + loader tags (already derivable via introspection)
* `node_index`: parsed `TagSpec` per node (so every consumer uses the same tag parsing rules)

Then:

* Delete (or severely shrink) the parts of `TargetGraph` that duplicate graph semantics.
* Make **closure** a method on `DagCatalog` that uses the Hamilton dependency graph (or an extracted adjacency), not a separate graph structure.

This one move tends to cascade into big simplifications in planner, executor, validation, and serving compilation.

---

Below is a **repo-concrete, breaking-change–friendly** implementation plan for:

> **(2) Replace bespoke support-module function factories with Hamilton parameterization primitives**

Target delta: delete `support_factory.py`’s “N functions + signature mutation + sys.modules injection” pattern; replace with **1 template function per support-node kind**, expanded into many DAG nodes via `@parameterize` (constants via `value(...)`, deps via `source(...)`), with per-node canonical tags applied using Hamilton’s **`@tag(..., target_=<expanded_node_name>)` targeting** (no bespoke per-node wrappers).

This is written assuming **(1) “dual graph removal” has landed** (i.e., you have `DagCatalog` and you no longer treat `TargetGraph` as authoritative). Where Phase6 still has `graph`, treat those references as “rename in-flight”.

---

# A) End-state semantics (non-negotiable)

## A.1 Support nodes become *static module* + *compile-time expansion*

* Support nodes are defined in a **real Python module** (checked in), not a dynamic `ModuleType` injected into `sys.modules`.
* The node *inventory* is supplied by driver construction via **Hamilton config**, and expanded at compile time using `resolve_from_config` + `parameterize` (per Hamilton advanced patterns).
* Support-node expansion is deterministic for a given DAG/catalog: `dataset_node(table_key)` etc.

## A.2 No per-node function factories, no signature surgery, no kwargs dependency capture

Remove the patterns currently used in `support_factory.py`:

* `inspect.Signature` mutation via `set_signature(...)`
* `**kwargs` lookup keyed by dynamically named parameters
* `attach_node(...)` per output
* mapping dicts (`DATASET_TO_NODE`, etc.) as required runtime infrastructure

Support-node dependencies are expressed as **parameter bindings**:

* `record=source(target_node(producer_target))`
* `ref=source(dataset_node(table_key))`
* constants via `value(table_key)` / `value(artifact_name)`

## A.3 Tagging remains canonical and queryable

Every expanded node must carry canonical tags (from `codeintel.build.hamilton.tagging` / `TagSpec`):

* datasets: `node_type=dataset`, `domain`, `target`, `table_key`
* loaders: `node_type=loader.query|loader.dataframe`, `domain`, `target`, `table_key`
* artifacts: `node_type=artifact`, `domain`, `target`, `artifact`
* paths: `node_type=helper` (optionally also `artifact` if you decide to enrich)

Per-node differentiation is achieved by stacking tag decorators with `target_=<expanded_name>`.

---

# B) Concrete implementation (files created / modified / deleted)

## B.1 Files CREATED

### 1) `src/codeintel/build/hamilton/nodes/support_spec.py`

**Purpose:** typed + validated spec object that driver_factory computes from `DagCatalog` (contract outputs only) and injects into Hamilton config.

**Implement:**

* `SupportDatasetSpec`: `{table_key: str, producer_target: str, domain: str}`
* `SupportArtifactSpec`: `{artifact_name: str, producer_target: str, domain: str}`
* `SupportNodeSpec`:

  * `datasets: tuple[SupportDatasetSpec, ...]`
  * `artifacts: tuple[SupportArtifactSpec, ...]`
  * `include_dataset_nodes: bool`
  * `include_loader_nodes: bool`
  * `include_artifact_nodes: bool`
  * `include_artifact_path_nodes: bool`
* `SupportNodeSpec.validate()` invariants:

  * unique `table_key` across datasets
  * unique `artifact_name` across artifacts
  * `producer_target` exists in catalog targets (if built from catalog)
* `SupportNodeSpec.to_hamilton_config()` emits a **flat config dict** with stable keys:

  * `ci_support_datasets`: `tuple[dict[str,str], ...]` (or list)
  * `ci_support_artifacts`: `tuple[dict[str,str], ...]`
  * `ci_support_include_dataset_nodes`: bool
  * `ci_support_include_loader_nodes`: bool
  * `ci_support_include_artifact_nodes`: bool
  * `ci_support_include_artifact_path_nodes`: bool

**Build hook:** `support_spec_from_catalog(catalog: DagCatalog, *, flags...) -> SupportNodeSpec`

> Keep config keys flat because `resolve_from_config` matches callable arg names (no dotted keys).

---

### 2) `src/codeintel/build/hamilton/nodes/support_nodes.py`

**Purpose:** one template function per support-node kind; compile-time expansion via `resolve_from_config` returning a decorator that applies:

1. `@parameterize(...)` with per-node bindings (`source(...)` + `value(...)`)
2. per-node tagging via stacked canonical tag decorators using `target_=expanded_node_name`

**Required imports (conceptual):**

* Hamilton: `resolve_from_config`, `parameterize`, `source`, `value`
* Naming: `dataset_node`, `query_node`, `dataframe_node`, `artifact_node`, `path_node`, `target_node`
* Tagging helpers: `tag_dataset`, `tag_loader_query`, `tag_loader_dataframe`, `tag_artifact`, `tag_helper`
* Types: `BuildEnv`, `TargetRunRecord`, `DatasetRef`, `ArtifactRef`, `ibis.expr.types as ir`, `pandas as pd`, `Path`

**Core pattern (single illustrative snippet; implement all 5 similarly):**

```python
# support_nodes.py (illustrative pattern)
from hamilton.function_modifiers import parameterize, resolve_from_config, source, value
from codeintel.build.hamilton.naming import dataset_node, target_node
from codeintel.build.hamilton.tagging import tag_dataset
from codeintel.core.hamilton.records import TargetRunRecord
from codeintel.build.hamilton.io.dataset_ref import DatasetRef

def _apply(decs):
    def wrapper(fn):
        out = fn
        for d in decs:  # order already encoded by caller
            out = d(out)
        return out
    return wrapper

def _decorate_dataset_nodes(ci_support_datasets, ci_support_include_dataset_nodes):
    if not ci_support_include_dataset_nodes or not ci_support_datasets:
        return lambda fn: fn

    mapping = {
        dataset_node(spec["table_key"]): {
            "record": source(target_node(spec["producer_target"])),
            "table_key": value(spec["table_key"]),
            "producer_target": value(spec["producer_target"]),
        }
        for spec in ci_support_datasets
    }
    tag_decs = [
        tag_dataset(
            domain=spec["domain"],
            target=spec["producer_target"],
            table_key=spec["table_key"],
            target_=dataset_node(spec["table_key"]),
        )
        for spec in ci_support_datasets
    ]
    # decorator order: parameterize closest to fn, tags above it
    return _apply([parameterize(**mapping), *tag_decs])

@resolve_from_config(decorate_with=_decorate_dataset_nodes)
def dataset_ref(record: TargetRunRecord, table_key: str, producer_target: str) -> DatasetRef:
    ds = record.get_dataset(table_key)
    if ds is None:
        raise ValueError(f"Missing DatasetRef for {table_key} from {producer_target}")
    if isinstance(ds, DatasetRef):
        return ds
    return DatasetRef(
        table_key=ds.table_key,
        repo=ds.repo,
        commit=ds.commit,
        row_count=ds.row_count,
        source_target=producer_target,
    )
```

**You will implement analogous resolvers for:**

* `load_ibis(env: BuildEnv, ref: DatasetRef) -> ir.Table` expanded to `q__*`
* `load_df(env: BuildEnv, ref: DatasetRef) -> pd.DataFrame` expanded to `df__*`
* `artifact_ref(env: BuildEnv, record: TargetRunRecord, artifact_name: str) -> ArtifactRef` expanded to `a__*`
* `artifact_path(ref: ArtifactRef) -> Path | None` expanded to `p__*`

For loaders:

* expansion depends on both dataset list and `ci_support_include_loader_nodes`
* bindings:

  * `ref = source(dataset_node(table_key))`
* tags:

  * `tag_loader_query(... target_=query_node(table_key))`
  * `tag_loader_dataframe(... target_=dataframe_node(table_key))`

For artifacts:

* bindings:

  * `record = source(target_node(producer_target))`
  * `artifact_name = value(artifact_name)`
* tags:

  * `tag_artifact(... target_=artifact_node(artifact_name))`

For paths:

* bindings:

  * `ref = source(artifact_node(artifact_name))`
* tags:

  * `tag_helper(... target_=path_node(artifact_name))` (optionally enrich with artifact tag if you elect to)

**Important:** `resolve_from_config` is a **compile-time decorator synthesis** feature; ensure driver build enables Hamilton power-user mode if required by your pinned Hamilton (see wiring below).

---

## B.2 Files MODIFIED

### 1) `src/codeintel/build/hamilton/driver_factory.py`

**Replace** dynamic support module generation (`build_support_module(...)`) with:

1. base/native driver build
2. catalog compile (from (1))
3. support spec compile from catalog outputs
4. final driver build including `support_nodes` module + merged config containing `ci_support_*` keys

**Concrete edits (Phase6 → end state):**

* Remove imports:

  * `SupportGenerationOptions`, `build_support_module`
* Add imports:

  * `from codeintel.build.hamilton.nodes.support_spec import support_spec_from_catalog`
  * `from codeintel.build.hamilton.nodes import support_nodes` (module import)
* Replace `_build_support_graph_and_module(...)` with `_build_support_spec(...)` returning `(base_driver, base_catalog, support_spec)`
* Build final driver via:

  * `builder.with_config({**user_config, **support_spec.to_hamilton_config()})`
  * `builder.with_modules(*native_mods, support_nodes)` (no dynamic module)

**Power-user mode wiring:**

* If your Hamilton build requires explicit enablement for `resolve_from_config`, set it **once** at composition root (driver_factory) before any drivers are built; isolate to build DAG assembly only.

**Keep the “two-pass build”** (native → introspect outputs → final) but eliminate module injection + per-node factories.

---

### 2) `src/codeintel/build/hamilton/nodes/__init__.py`

* Stop exporting `build_support_module` APIs.
* Export only:

  * `SupportNodeSpec` (optional public)
  * or nothing (keep it internal and only import `support_nodes` as module by path)

---

### 3) Tests (replace support_factory direct tests with driver-level node presence tests)

#### Modify:

* `tests/build/hamilton/test_pr12_loader_nodes.py`

  * Remove imports of `build_support_module`, `SupportGenerationOptions`
  * Replace with:

    * `runtime = build_driver()` (or build_driver with config toggles)
    * Assert `q__*` and `df__*` appear in `runtime.dr.graph.nodes` or `list_available_variables()`
    * For “disabled” behavior: pass config flag `ci_support_include_loader_nodes=False` and ensure `q__/df__` absent (or keep always-on and delete that test)

* `tests/build/hamilton/test_pr17_generated_assets_module.py`

  * Replace “module contains nodes” with “driver graph contains nodes”
  * Validate:

    * `any(name.startswith("d__") ...)`
    * `any(name.startswith("a__") ...)`
    * `not any(name.startswith("t__") for name in *support module*)` becomes irrelevant; instead assert that support node module doesn’t define `t__` templates and that graph’s `t__` nodes still originate from native modules.

* `tests/build/test_hamilton_phase1.py`

  * Remove `TestSupportFactory` section entirely or rewrite as `TestSupportNodesCompilation`:

    * “build_driver produces dataset nodes”
    * “exclude/include flags operate via config”

---

## B.3 Files DELETED

1. `src/codeintel/build/hamilton/nodes/support_factory.py`
   Entirely replaced by static `support_nodes.py` + `SupportNodeSpec` compile.

2. `src/codeintel/build/hamilton/nodes/mappings.py`
   No longer needed; node naming is deterministic via `naming.py`, and inventories come from `DagCatalog`/support spec.

> Keep `module_attach.py` and `signature_tools.py` because other dynamic-generation patterns (e.g., tool targets) still use them.

---

# C) Operational invariants + validation gates (must implement)

Add these checks either in `support_spec_from_catalog(...)` or in `support_nodes.py` resolver functions (prefer spec build-time):

1. **Uniqueness:**

   * `table_key` unique across `ci_support_datasets`
   * `artifact_name` unique across `ci_support_artifacts`

2. **Producer validity:**

   * `producer_target` exists in catalog targets
   * `target_node(producer_target)` exists as a node (anchor)

3. **Identifier safety:**

   * enforce `dataset_node(table_key).isidentifier()` etc (should already hold by `naming.py`, but validate once)

4. **Spec emptiness behavior:**

   * empty spec must compile cleanly (no `parameterize()` called with 0 kwargs if that errors in Hamilton; handle with “identity decorator” fallback)

---

# D) Expected functional output (observable behaviors)

After this change:

* `build_driver()` produces a driver whose DAG includes:

  * `d__*` for every **contract table output**
  * `q__*`/`df__*` for each dataset if loaders enabled
  * `a__*` and `p__*` for each **contract artifact output** if enabled
* No dynamic `ModuleType("codeintel.build.hamilton.nodes.support")` exists; support nodes are compiled from `support_nodes.py`.
* Support-node compile logic is reduced to:

  * building `SupportNodeSpec` (pure derivation from catalog)
  * passing `ci_support_*` config into Builder
* You delete the heaviest bespoke parts: signature mutation, function factories, and sys.modules injection.

---

# E) File index summary (what changes where)

### Created

* `src/codeintel/build/hamilton/nodes/support_spec.py`
* `src/codeintel/build/hamilton/nodes/support_nodes.py`

### Modified

* `src/codeintel/build/hamilton/driver_factory.py`
* `src/codeintel/build/hamilton/nodes/__init__.py`
* `tests/build/hamilton/test_pr12_loader_nodes.py`
* `tests/build/hamilton/test_pr17_generated_assets_module.py`
* `tests/build/test_hamilton_phase1.py`

### Deleted

* `src/codeintel/build/hamilton/nodes/support_factory.py`
* `src/codeintel/build/hamilton/nodes/mappings.py`

---

If you want the *next* “best-in-class squeeze” after this: once support nodes are static + config-expanded, you can make **loader generation cost near-zero** by splitting `support_nodes.py` into micro-modules (`support_datasets.py`, `support_loaders.py`, `support_artifacts.py`) and only including the loader module when a serving/export workflow actually needs loaders (avoids importing pandas/ibis in minimal build profiles).


---

#
