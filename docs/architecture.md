


# 1) Executive architecture summary

* **Hamilton is the build execution substrate**: build targets execute as Hamilton nodes in a Driver built from native modules + generated support module.
  `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)`

* **Targets are DAG-defined**:

  * A target is anchored by a `t__<target>` node tagged `node_type="materialize"` and decorated with target spec tags (resources/execution/parameters/spec_version).
    `src/codeintel/build/hamilton/native/target_decorators.py :: codeintel_target(...)` / `TargetSpecDescriptor`
    `src/codeintel/core/hamilton/tags.py :: TAG_NODE_TYPE` / `NODE_TYPE_MATERIALIZE`
  * Target metadata (`OutputTarget`) is compiled from the Hamilton driver graph (anchors + tags + docstrings) and then dependency edges are derived from upstream traversal.
    `src/codeintel/build/hamilton/target_spec_compiler.py :: compile_output_targets_from_driver(...)`
    `src/codeintel/build/hamilton/introspect.py :: derive_target_dependencies(...)`

* **BuildEnv is the sole Hamilton input** for node execution; it bundles gateway, snapshot/paths, config/settings, storage helpers, manifests/force flags, and optional registry/execution context.
  `src/codeintel/build/hamilton/env.py :: BuildEnv`

* **Incremental execution is manifest-driven**:

  * Planning and skipping compare computed input hashes vs stored `OutputManifest` entries and support `force_targets` bypass.
    `src/codeintel/build/hash_evaluator.py :: compute_hash_evaluation(...)` / `evaluate_hash_state(...)`
    `src/codeintel/build/hamilton/run_record_utils.py :: should_skip_native_target(...)`
    `src/codeintel/core/build_manifest.py :: OutputManifest`

* **Non-execution build operations exist as first-class CLI workflows**:

  * Status (state computation): `src/codeintel/cli/handlers/build.py :: build_status_handler(...)` → `src/codeintel/build/state.py :: StateValidator.validate(...)`
  * Plan (dry run): `src/codeintel/cli/handlers/build.py :: build_plan_handler(...)` → `src/codeintel/build/hamilton/planner.py :: compute_plan(...)`
  * Graph export: `src/codeintel/cli/handlers/build.py :: build_graph_handler(...)` → `src/codeintel/build/hamilton/observability.py :: export_dag_mermaid(...)` / `export_dag_dot(...)` / `export_dag_json(...)`
  * Schema manifest (compile/diff/migrate): `src/codeintel/cli/handlers/build_schema.py :: build_schema_compile_handler(...)` / `build_schema_diff_handler(...)` / `build_schema_migrate_handler(...)`
  * BuildSpec compile: `src/codeintel/cli/handlers/build_spec.py :: build_spec_compile_handler(...)`

* **RegistryService is a canonical cross-cutting discovery surface** (targets + dataset contracts), and includes an on-disk DAG output inventory YAML.
  `src/codeintel/core/registry/service.py :: RegistryService.from_gateway(...)`
  `src/codeintel/core/registry/service.py :: _DAG_OUTPUT_INVENTORY_PATH`

* **Optional Hamilton UI tracker integration exists in executor** (if adapter is available), including deterministic tags (repo/commit/run_id + decision trace metadata).
  `src/codeintel/build/hamilton/executor.py :: _create_tracker(...)` / `_build_tracker_tags(...)` / `_apply_tracker_constants(...)`
  `src/codeintel/core/config/settings.py :: HamiltonTrackerSettings`

# 2) Repository map (build-focused)

## 2.1 `src/codeintel/build/` (top-level)

* Public facade + lazy exports: `src/codeintel/build/__init__.py :: _LAZY_IMPORTS` / `__getattr__(...)`
* Target model + dependency graph: `src/codeintel/build/targets.py :: OutputTarget` / `TargetGraph`
* Contracts: `src/codeintel/build/contracts.py :: OutputContract` / `ArtifactSpec`
* Config + parameters + resources:

  * `src/codeintel/build/config.py :: BuildConfig` / `BuildConfigStack` / `load_build_config(...)`
  * `src/codeintel/build/parameters.py :: TargetParameters`
  * `src/codeintel/build/resources.py :: TargetResources` / `TargetExecution`
* Hashing + evaluation + session/state:

  * `src/codeintel/build/hashing.py :: compute_input_hash_with_deps(...)` / `compute_target_options_hash(...)`
  * `src/codeintel/build/hash_evaluator.py :: compute_hash_evaluation(...)` / `evaluate_hash_state(...)`
  * `src/codeintel/build/session.py :: BuildSession`
  * `src/codeintel/build/state.py :: StateValidationOptions` / `StateValidator`
* Engine version (hash invalidation): `src/codeintel/build/engine_version.py :: get_build_engine_version(...)`
* Run context wiring: `src/codeintel/build/run_context.py :: BuildRunContext.build_env(...)`
* Target metadata caching layer: `src/codeintel/build/target_metadata.py :: get_target_metadata_service(...)` / `get_target_system(...)` / `clear_target_metadata_cache(...)`

## 2.2 `src/codeintel/build/hamilton/` (high-leverage modules)

* Composition/runtime: `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)`, `src/codeintel/build/hamilton/runtime.py :: HamiltonRuntime`
* Execution: `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor.run(...)`, `src/codeintel/build/hamilton/execution_options.py :: BuildExecutionOptions`
* Planning: `src/codeintel/build/hamilton/planner.py :: compute_plan(...)` / `PlanEntry`
* Tagging system:

  * Typed tag spec: `src/codeintel/build/hamilton/tag_spec.py :: TagSpec` / `TagKey` / `NodeType` / `validate_tag_spec(...)`
  * Canonical tag decorators: `src/codeintel/build/hamilton/tagging.py :: tag_materialize(...)` / `tag_compute(...)` / `tag_tool(...)`
  * Tag index: `src/codeintel/build/hamilton/tag_index.py :: TagIndex.from_runtime(...)` / `semantic_view_tags(...)` / `data_saver_nodes(...)`
* Introspection: `src/codeintel/build/hamilton/introspect.py :: derive_target_dependencies(...)` / `derive_target_outputs_from_savers(...)` / `derive_target_io_surface(...)`
* Options loading (plan/execution alignment): `src/codeintel/build/hamilton/options_loading.py :: load_target_options(...)`
* Graph runtime options loader: `src/codeintel/build/hamilton/graph_runtime_options.py :: load_graph_runtime_options(...)`
* Impl-kind detection: `src/codeintel/build/hamilton/impl_kind.py :: target_impl_kind(...)`
* Materialization records: `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor`
* Decision trace: `src/codeintel/build/hamilton/decision_trace.py :: build_decision_trace(...)` / `read_decision_trace(...)`
* Observability exports: `src/codeintel/build/hamilton/observability.py :: export_dag_json(...)` / `export_dag_mermaid(...)` / `export_dag_dot(...)`
* Support nodes: `src/codeintel/build/hamilton/nodes/support_factory.py :: build_support_module(...)`
* Save-to and materializers: `src/codeintel/build/hamilton/save_to.py :: SaveToObjectMetadataDecorator`, `src/codeintel/build/hamilton/materializers/*`

## 2.3 Build products subsystems (still present)

* Assets: `src/codeintel/build/assets/fingerprinting.py :: FingerprintPolicy`, `src/codeintel/build/assets/impact.py :: compute_impact(...)`
* Schemas:

  * Schema service factory: `src/codeintel/build/schemas/service.py :: get_schema_service(...)`
  * Contract service: `src/codeintel/build/schemas/contract_service.py :: ContractService` / `get_enriched_contract_service(...)`
  * Schema index/inference: `src/codeintel/build/schemas/schema_index.py :: SchemaIndex.get_table_schema(...)`
  * Schema manifest compile: `src/codeintel/build/schemas/compile.py :: compile_schema_manifest(...)` (entrypoints referenced by CLI handler)
* BuildSpec: `src/codeintel/build/spec/compile.py :: compile_buildspec(...)`, `src/codeintel/build/spec/serdes.py :: buildspec_to_json(...)`
* Exports: `src/codeintel/build/exports/runner.py :: ExportRunner`
* Serving snapshot: `src/codeintel/build/serving/publisher.py :: publish_serving_snapshot(...)`

# 3) Hamilton subsystem map (deep dive)

## 3.1 Composition root: `build_driver(...)`

* Base driver is built from native modules: `src/codeintel/build/hamilton/driver_factory.py :: _build_base_graph(...)`, `src/codeintel/build/hamilton/native/discovery.py :: load_native_modules(...)`
* OutputTarget specs compiled from the driver: `src/codeintel/build/hamilton/target_spec_compiler.py :: compile_output_targets_from_driver(...)`
* Dependencies derived from graph traversal: `src/codeintel/build/hamilton/introspect.py :: derive_target_dependencies(...)`, `target_graph_from_hamilton(...)`
* Support module generated using saver-derived outputs:

  * Derived outputs: `src/codeintel/build/hamilton/introspect.py :: derive_target_outputs_from_savers(...)`
  * Support module: `src/codeintel/build/hamilton/nodes/support_factory.py :: build_support_module(...)`
  * Wiring: `src/codeintel/build/hamilton/driver_factory.py :: _build_support_graph_and_module(...)`

## 3.2 Execution runtime

* Executor runs full closure and writes run/target/node records:

  * `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor.run(...)`
  * Writer integration: `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.start_run(...)` / `save_run_targets(...)` / `save_run_nodes(...)` / `complete_run(...)`

## 3.3 Contracts enforcement and validation (runtime gates)

* Contract enforcement is context-driven via ContractEnforcer:

  * `src/codeintel/build/hamilton/contracts/enforcement.py :: ContractEnforcer.for_target(...)`
* Strict enforcement is applied by wrapping the gateway:

  * `src/codeintel/build/hamilton/contracts/enforced_gateway.py :: ContractEnforcingStorageGateway`
* Optional Pandera/schema validation helpers exist:

  * `src/codeintel/build/hamilton/contracts/pandera_hook.py :: validate_dataframe(...)` (and schema lookup)
  * `src/codeintel/build/hamilton/validators/dataframe.py :: (validators)`

## 3.4 Planning and decision trace utilities

* Planner produces per-target `PlanEntry` with `status` and `reason` plus hashes/deps:

  * `src/codeintel/build/hamilton/planner.py :: PlanStatus` / `PlanReason`
  * `src/codeintel/build/hamilton/planner.py :: PlanEntry`
* Decision trace utilities serialize plan contexts:

  * `src/codeintel/build/hamilton/decision_trace.py :: build_decision_trace_payload(...)` / `read_decision_trace(...)` / `write_decision_trace(...)`
  * CLI reader: `src/codeintel/cli/handlers/build.py :: build_decision_trace_handler(...)`
  * **Writing decision trace from build run is not confirmed** (no call site found outside the utility module).

# 4) Core runtime concepts & types (type map)

* Build runtime settings:

  * `src/codeintel/core/config/settings.py :: BuildSettings`
  * `src/codeintel/core/config/settings.py :: HamiltonExecutionSettings`
  * `src/codeintel/core/config/settings.py :: HamiltonTrackerSettings`
* Execution context bundle:

  * `src/codeintel/core/execution/context.py :: ExecutionContext` / `RunContext`
* Build env + run context:

  * `src/codeintel/build/run_context.py :: BuildRunContext`
  * `src/codeintel/build/hamilton/env.py :: BuildEnv` (fields include `execution_context`, `registry`, `fingerprint_policy`, `history_options`)
* Registry and storage facades:

  * `src/codeintel/core/registry/service.py :: RegistryService`
  * `src/codeintel/storage/facade.py :: StorageFacade`
* Schema types:

  * `src/codeintel/build/schemas/service.py :: get_schema_service(...)` (returns SchemaService)
  * `src/codeintel/build/schemas/contract_service.py :: ContractService`
  * `src/codeintel/build/schemas/schema_index.py :: SchemaIndex`
  * Schema manifest types are re-exported from core manifests: `src/codeintel/build/schemas/manifest.py :: SchemaManifest` (imported from `src/codeintel/core/manifests.py`)
* State computation:

  * `src/codeintel/build/state.py :: StateValidationOptions`
  * `src/codeintel/build/state.py :: StateValidator`
  * `src/codeintel/build/state_types.py :: TargetState` / `BuildState`
* Planning:

  * `src/codeintel/build/hamilton/planner.py :: PlanEntry`
* IO surface derivation:

  * `src/codeintel/build/hamilton/introspect.py :: TargetIOSurface` / `derive_target_io_surface(...)`

# 5) Data & IO model (as implemented)

* Target node outputs:

  * `src/codeintel/core/hamilton/records.py :: TargetRunRecord`
* Dataset/artifact handle types used by support nodes:

  * `src/codeintel/build/hamilton/io/dataset_ref.py :: DatasetRef`
  * `src/codeintel/build/hamilton/io/artifact_ref.py :: ArtifactRef`
* BuildEnv IO accessors:

  * Warehouse façade resolution: `src/codeintel/build/hamilton/env.py :: BuildEnv.warehouse` (uses `storage.warehouse` when `storage` present, else wraps gateway)
  * Manifest service shortcut: `src/codeintel/build/hamilton/env.py :: BuildEnv.manifest_service`
* Materialization result boundary (saver metadata nodes):

  * `src/codeintel/build/hamilton/boundary_types.py :: MaterializationResult`
  * Saver tags built in: `src/codeintel/build/hamilton/save_to.py :: _build_saver_tags(...)`
* DuckDB writes:

  * `src/codeintel/build/hamilton/materializers/duckdb_saver.py :: DuckDBIbisTableSaver.save_data(...)`
  * `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py :: DuckDBRowsSaver.save_data(...)` (columns resolved via `src/codeintel/build/schemas/column_resolution.py :: resolve_columns(...)`)
* Artifact writes:

  * `src/codeintel/build/hamilton/materializers/artifact_saver.py :: FileArtifactSaver`
  * Template validation: `src/codeintel/build/hamilton/materializers/path_templates.py :: validate_path_template(...)` / `format_path_template(...)`
* Support read nodes:

  * `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_dataset_node_function(...)` / `_create_query_node_function(...)` / `_create_dataframe_node_function(...)` / `_create_artifact_node_function(...)`

# 6) Target orchestration model (derived behavior)

* Status (state computation):

  * CLI entrypoint: `src/codeintel/cli/handlers/build.py :: build_status_handler(...)`
  * Uses target system graph + StateValidator: `src/codeintel/build/target_metadata.py :: get_target_system(...)`, `src/codeintel/build/state.py :: StateValidator.validate(...)`
* Planning:

  * CLI entrypoint: `src/codeintel/cli/handlers/build.py :: build_plan_handler(...)`
  * Planner: `src/codeintel/build/hamilton/planner.py :: compute_plan(...)` → `PlanEntry`
* Execution:

  * CLI entrypoint: `src/codeintel/cli/handlers/build.py :: build_run_handler(...)` → `_execute_build_hamilton(...)`
  * Env assembly: `src/codeintel/build/run_context.py :: BuildRunContext.build_env(...)`

    * Loads registry when enabled: `src/codeintel/core/registry/service.py :: RegistryService.from_gateway(...)`
    * Ensures schema service cache is initialized: `src/codeintel/build/schemas/service.py :: get_schema_service(...)`
    * Wraps gateway in StorageFacade: `src/codeintel/storage/facade.py :: StorageFacade.from_gateway(...)`
  * Execution: `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._execute_dag(...)` uses `inputs={"env": execution_env, "graph": graph}`
* Native target execution helper:

  * `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor` (hash/skip/records)
* Build graph export:

  * `src/codeintel/cli/handlers/build.py :: build_graph_handler(...)` calls `src/codeintel/build/hamilton/observability.py :: export_dag_*`
* Build history:

  * CLI entrypoint: `src/codeintel/cli/handlers/build.py :: build_history_handler(...)`
  * Env supports timeseries options: `src/codeintel/build/hamilton/env.py :: BuildEnv.history_options`, type: `src/codeintel/analytics/history/history_timeseries.py :: HistoryTimeseriesOptions`
* Build schema/spec products:

  * Schema manifest compile/diff/migrate: `src/codeintel/cli/handlers/build_schema.py :: build_schema_compile_handler(...)` / `build_schema_diff_handler(...)` / `build_schema_migrate_handler(...)`
  * BuildSpec compile: `src/codeintel/cli/handlers/build_spec.py :: build_spec_compile_handler(...)`

# 7) Walkthrough: “request one target” end-to-end trace

* `build run` handler path:

  * `src/codeintel/cli/handlers/build.py :: build_run_handler(...)` → `_execute_build_hamilton(...)`
* Env assembly:

  * `src/codeintel/build/run_context.py :: BuildRunContext.build_env(load_catalogs=True, load_schema_service=True)`
  * Populates: `registry=RegistryService.from_gateway(...)`, `storage=StorageFacade.from_gateway(...)`, `fingerprint_policy=DEFAULT_FINGERPRINT_POLICY`, `execution_context=self.execution_context`. `src/codeintel/build/run_context.py :: BuildRunContext.build_env(...)`
* Execution:

  * `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor.run(...)` computes closure and calls `_execute_dag(...)`
  * Driver call uses `inputs={"env": ..., "graph": ...}`: `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._execute_dag(...)`
* Representative target still follows `t__<target>` anchor convention (example module present):

  * `src/codeintel/build/hamilton/native/ingestion/scip.py :: t__scip(...)`
  * Decorator: `src/codeintel/build/hamilton/native/target_decorators.py :: codeintel_target(...)`

# 8) Extension mechanics (how new behavior is introduced today)

* New target:

  * Add `t__<target>` node and decorate with `codeintel_target(domain=..., target=..., spec=...)`. `src/codeintel/build/hamilton/native/target_decorators.py :: codeintel_target(...)`
* Configuration loading convention for targets:

  * Load strongly-typed options from `env.config` via `load_target_options(...)`. `src/codeintel/build/hamilton/options_loading.py :: load_target_options(...)`
  * Graph-oriented targets use `load_graph_runtime_options(...)`. `src/codeintel/build/hamilton/graph_runtime_options.py :: load_graph_runtime_options(...)`
* Output materialization:

  * Use `SaveToObjectMetadataDecorator` to attach DataSavers and emit saver tags. `src/codeintel/build/hamilton/save_to.py :: SaveToObjectMetadataDecorator`
  * Executor-style targets materialize via `NativeTargetExecutor` inside `t__<target>` nodes. `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor`
* Tagging:

  * Preferred tagging path uses typed TagSpec helpers: `src/codeintel/build/hamilton/tag_spec.py :: TagSpec.for_*`, applied via `src/codeintel/build/hamilton/tagging.py :: tag_*`

# 9) Conventions & invariants (observed)

* Canonical naming:

  * `src/codeintel/build/hamilton/naming.py :: target_node(...)` / `dataset_node(...)` / `query_node(...)` / `dataframe_node(...)` / `materialize_node(...)` / `artifact_node(...)`
* Canonical tag keys/types:

  * `src/codeintel/core/hamilton/tags.py :: TAG_DOMAIN` / `TAG_TARGET` / `TAG_TABLE_KEY` / `TAG_ARTIFACT` / `TAG_NODE_TYPE`
  * `src/codeintel/core/hamilton/tags.py :: NODE_TYPE_MATERIALIZE` / `NODE_TYPE_COMPUTE` / `NODE_TYPE_DATASET` / `NODE_TYPE_ARTIFACT`
* Typed tag spec validation:

  * `src/codeintel/build/hamilton/tag_spec.py :: validate_tag_spec(...)`
  * Tagging decorators construct TagSpec via `TagSpec.for_*` factories: `src/codeintel/build/hamilton/tagging.py :: tag_*`
* Saver tag schema invariants:

  * Tag builder: `src/codeintel/build/hamilton/save_to.py :: _build_saver_tags(...)`
  * Artifact savers must provide validated `path_template`: `src/codeintel/build/hamilton/save_to.py :: _resolve_artifact_path_template(...)`, `src/codeintel/build/hamilton/materializers/path_templates.py :: validate_path_template(...)`
* Graph validation invariant checks (anchors/support/savers; optional compute IO purity):

  * `src/codeintel/build/hamilton/validate.py :: validate_nodes(..., enforce_compute_io_purity: bool = False, ...)`
  * Wrapper: `src/codeintel/build/hamilton/graph_validation.py :: validate_graph(...)`

# 10) Glossary (project-specific vocabulary)

* **BuildEnv**: the single Hamilton inputs bundle for all nodes (gateway/config/settings/snapshot/registry/storage/etc.). `src/codeintel/build/hamilton/env.py :: BuildEnv`
* **Target anchor**: `t__<target>` node tagged materialize and decorated with `codeintel_target(...)`. `src/codeintel/build/hamilton/native/target_decorators.py :: codeintel_target(...)`
* **OutputTarget**: compiled target metadata derived from the driver graph (tags/docstrings + derived outputs). `src/codeintel/build/hamilton/target_spec_compiler.py :: compile_output_targets_from_driver(...)`
* **TagSpec**: typed tag schema for Hamilton build nodes; used by tagging helpers. `src/codeintel/build/hamilton/tag_spec.py :: TagSpec`
* **DataSaver node**: node created by SaveToObjectMetadataDecorator that performs persistence and emits saver tags. `src/codeintel/build/hamilton/save_to.py :: SaveToObjectMetadataDecorator.create_saver_node(...)`
* **RegistryService**: canonical discovery bundle for dataset contracts and target catalog; also owns an on-disk DAG output inventory YAML. `src/codeintel/core/registry/service.py :: RegistryService`
* **StorageFacade**: unified storage access surface (warehouse/exports/datasets) created from a gateway. `src/codeintel/storage/facade.py :: StorageFacade`
* **SchemaIndex**: resolves table schemas using inference + overrides, tracks inference errors. `src/codeintel/build/schemas/schema_index.py :: SchemaIndex`
* **PlanEntry**: per-target plan decision record (`compute|skip|missing|blocked` + reason/hashes/deps). `src/codeintel/build/hamilton/planner.py :: PlanEntry`
* **Decision trace**: plan decision payload utilities; CLI reads latest trace file. `src/codeintel/build/hamilton/decision_trace.py :: read_decision_trace(...)`, `src/codeintel/cli/handlers/build.py :: build_decision_trace_handler(...)`
* **StateValidator**: computes current/stale/missing/blocked state from manifests + hashes. `src/codeintel/build/state.py :: StateValidator`

If you want the next step to be maximally efficient, I can take one pass over Phase5 and produce a “deep dive map” for the *new* elements that weren’t in the prior statement set (RegistryService + DAG output inventory YAML, TagSpec/tagging, decision trace, and the expanded build CLI workflows), keeping the same evidence rules.
