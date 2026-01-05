# 1) Executive architecture summary

* **Hamilton DAG is authoritative**: the Driver graph defines targets and dependencies, compiled into a `DagCatalog`.
  `src/codeintel/build/hamilton/dag_catalog_compiler.py :: compile_dag_catalog(...)`

* **Targets are single-source**: a target exists iff a `t__*` node with materialize tags exists; discovery is tag-filter driven.
  `src/codeintel/core/hamilton/tag_query.py :: TagQuery` / `Driver.list_available_variables(tag_filter=...)`

* **BuildEnv is the only Hamilton input**: it bundles gateway, snapshot/paths, config/settings, providers, and execution context.
  `src/codeintel/build/hamilton/env.py :: BuildEnv`

* **Incrementality is cache-first**: cache keys and cache presence drive skip/compute; manifests are audit-only.
  `src/codeintel/build/hamilton/cache_key_resolver.py :: CacheKeyResolver`
  `src/codeintel/build/hamilton/cache_index.py :: CacheIndex`
  `src/codeintel/build/session.py :: BuildSession`

* **Support nodes are static/parameterized**: dataset/artifact refs and loaders are expanded via config, not a dynamic support factory.
  `src/codeintel/build/hamilton/nodes/support_nodes.py`
  `src/codeintel/build/hamilton/nodes/module_attach.py :: attach_support_nodes(...)`

# 2) Repository map (build-focused)

## 2.1 Build package top-level

* Package facade + exports: `src/codeintel/build/__init__.py`
* Dag catalog + target descriptors: `src/codeintel/build/hamilton/dag_catalog.py :: DagCatalog` / `TargetDescriptor`
* Target metadata service: `src/codeintel/build/target_metadata.py :: TargetMetadataService`
* Run context wiring: `src/codeintel/build/run_context.py :: BuildRunContext`
* State computation: `src/codeintel/build/state.py :: StateValidator` / `src/codeintel/build/state_computer.py :: StateComputer`
* Session cache: `src/codeintel/build/session.py :: BuildSession`

## 2.2 Hamilton subtree

* Composition: `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)`
* Planning: `src/codeintel/build/hamilton/planner.py :: compute_plan(...)` / `PlanEntry`
* Execution: `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor`
* Tagging: `src/codeintel/build/hamilton/tag_spec.py :: TagSpec` / `tag_spec_from_tags(...)`
* Tag queries: `src/codeintel/core/hamilton/tag_query.py :: TagQuery`
* Support nodes: `src/codeintel/build/hamilton/nodes/support_nodes.py` / `support_spec.py`
* Observability exports: `src/codeintel/build/hamilton/observability.py :: export_dag_json(...)`

# 3) Composition and catalog flow

* Native module discovery loads Hamilton modules from:
  `src/codeintel/build/hamilton/native/discovery.py :: native_module_paths()`

* Driver composition attaches native modules and parameterized support nodes:
  `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)`
  `src/codeintel/build/hamilton/nodes/module_attach.py :: attach_support_nodes(...)`

* DagCatalog compilation is derived directly from the Driver graph:
  `src/codeintel/build/hamilton/dag_catalog_compiler.py :: compile_dag_catalog(...)`

# 4) Planning, state, and incrementality

* Cache authority contract:

  * Cache key == input identity for a target.
  * Cache presence == current state for that target.

* State computation is cache-driven, with dependency blocking computed via DagCatalog closure:
  `src/codeintel/build/state_computer.py :: StateComputer.compute_all(...)`
  `src/codeintel/build/hamilton/dag_catalog.py :: DagCatalog.closure(...)`

* Planner and explain nodes use cache probes and catalog metadata (no manifest gating):
  `src/codeintel/build/hamilton/planner.py :: compute_plan(...)`
  `src/codeintel/build/hamilton/native/planning/plan_nodes.py`

# 5) IO boundaries and run records

* IO boundaries are explicit saver nodes that emit `MaterializationResult`:
  `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py :: DuckDBRowsSaver`
  `src/codeintel/build/hamilton/boundary_types.py :: MaterializationResult`

* Run records and target records are written by the executor:
  `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter`
  `src/codeintel/build/hamilton/run_records.py :: create_run_record(...)`

# 6) Glossary

* **DagCatalog**: Immutable view over Driver nodes/tags/deps.
* **TargetDescriptor**: Parsed target metadata (resources, execution, parameters, description).
* **TagQuery**: Typed tag filter helper for Driver tag queries.
* **BuildSession**: Per-run cache key and cache index wrapper.
* **ExecutionContext**: Unified runtime primitives + settings.
