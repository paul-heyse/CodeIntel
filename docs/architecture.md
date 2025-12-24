# 0) What I inspected

* **Codebase**: extracted `CodeIntel_Centralizing_Phase4.zip` and inspected:

  * `AGENTS.md` (agent constraints + repo “do-not-edit zones” inventory). `AGENTS.md :: (document)`
  * `pyproject.toml` (package layout + tool config). `pyproject.toml :: (project metadata)`
  * Build layer package tree: `src/codeintel/build/**` (all modules/subpackages; emphasis on `build/hamilton/**`). `src/codeintel/build/__init__.py :: __getattr__(...)`
  * CLI build entrypoints and handlers (for composition-root behavior): `src/codeintel/cli/handlers/build.py :: build_run_handler(...)` / `build_plan_handler(...)` / `build_explain_handler(...)`

* **Implementation plans provided (context only; code is authoritative)**:

  * `centralization_big_move_1.md` 
  * `centralization_big_move_2.md` 
  * `centralization_big_move_3.md` 

---

# 1) Executive architecture summary

* **Single orchestration engine**: build execution is driven by a Hamilton `Driver` constructed from *native target modules* plus a generated support module. `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)`
* **Targets are DAG-defined (not registry-defined)**:

  * A build target is anchored by a `t__<target>` node tagged `node_type="materialize"`, with `domain`, `target`, and spec tags. `src/codeintel/build/hamilton/native/target_decorators.py :: codeintel_target(...)`, `src/codeintel/core/hamilton/tags.py :: TAG_NODE_TYPE` / `NODE_TYPE_MATERIALIZE`
  * `OutputTarget` specs are compiled from DAG tags + docstrings + saver-derived outputs. `src/codeintel/build/hamilton/target_spec_compiler.py :: compile_output_targets_from_driver(...)`
* **Output inventory can be “declared vs DAG-derived”**:

  * Inventory modes `declared|compare|dag` are resolved via settings; DAG derivation comes from **DataSaver tags** (contract-only `output_role="contract"`). `src/codeintel/build/target_inventory.py :: resolve_output_inventory(...)`, `src/codeintel/build/hamilton/introspect.py :: derive_target_outputs_from_savers(...)`, `src/codeintel/core/config/settings.py :: BuildSettings.output_inventory_source`
* **Support nodes (dataset/loader/artifact nodes) can be generated from contracts or from DAG-derived saver outputs** based on `BuildSettings.support_nodes_source`. `src/codeintel/build/hamilton/driver_factory.py :: _build_support_graph_and_module(...)`, `src/codeintel/build/hamilton/nodes/support_factory.py :: build_support_module(...)`
* **Runtime execution (“build run”) path**:

  * CLI resolves goals using the target graph from the target metadata service, builds a `BuildRunContext`, constructs `BuildEnv`, and runs `HamiltonBuildExecutor.run(...)`. `src/codeintel/cli/handlers/build.py :: build_run_handler(...)`, `_execute_build_hamilton(...)`, `src/codeintel/build/run_context.py :: BuildRunContext.build_env(...)`, `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor.run(...)`
* **Incremental behavior**:

  * “Skip” decisions use manifest hash evaluation (`evaluate_hash_state`) against `OutputManifest` loaded from the storage gateway, with a force-target bypass. `src/codeintel/build/hamilton/run_record_utils.py :: should_skip_native_target(...)`, `src/codeintel/build/hash_evaluator.py :: evaluate_hash_state(...)`, `src/codeintel/core/build_manifest.py :: OutputManifest`
* **“DAG-first invariants” have an explicit validator**:

  * `validate_nodes(...)` checks materialize anchors, support nodes, saver tags (including artifact templates), and optional compute I/O purity. `src/codeintel/build/hamilton/validate.py :: validate_nodes(...)`
  * `validate_graph()` builds the driver then validates it. `src/codeintel/build/hamilton/graph_validation.py :: validate_graph(...)`

---

# 2) Repository map (build-focused)

## 2.1 `src/codeintel/build/` (Phase4 notable modules)

* Public facade + lazy exports:

  * `src/codeintel/build/__init__.py :: _LAZY_IMPORTS` / `__getattr__(...)`
* Core target primitives:

  * `src/codeintel/build/targets.py :: OutputTarget` / `TargetGraph` / `TargetModule`
  * Target errors: `src/codeintel/build/errors.py :: TargetNotFoundError` / `InvalidTargetSpecError` (and related domain errors)
* Contracts & contract outputs:

  * `src/codeintel/build/contracts.py :: OutputContract` / `ArtifactSpec`
* Config, parameters, resources:

  * `src/codeintel/build/config.py :: BuildConfig` / `BuildConfigStack` / `load_build_config(...)`
  * `src/codeintel/build/parameters.py :: TargetParameters` / `EMPTY_PARAMETERS`
  * `src/codeintel/build/resources.py :: TargetResources` / `TargetExecution` / `DEFAULT_*`
* Hashing / evaluation / session/state:

  * `src/codeintel/build/hashing.py :: compute_input_hash_with_deps(...)` / `compute_target_options_hash(...)` / `InputHashOptions`
  * `src/codeintel/build/hash_evaluator.py :: evaluate_hash_state(...)` / `compute_hash_evaluation(...)`
  * `src/codeintel/build/session.py :: BuildSession`
  * `src/codeintel/build/state.py :: StateValidator.validate(...)`
  * `src/codeintel/build/state_computer.py :: StateComputer.compute_all(...)`
  * `src/codeintel/build/state_types.py :: TargetState` / `BuildState`
* **New/centralized metadata & inventory services**:

  * Target catalog (cached canonical target specs): `src/codeintel/build/target_catalog.py :: load_target_specs(...)`, `target_graph_from_catalog(...)`
  * Target metadata service (runtime + indexes + tag index + schema index): `src/codeintel/build/target_metadata.py :: get_target_metadata_service(...)`, `TargetSystem`, `TargetMetadataService`
  * Output inventory types: `src/codeintel/build/output_inventory.py :: OutputInventory`
  * Output inventory resolution modes: `src/codeintel/build/target_inventory.py :: resolve_output_inventory(...)` / `OutputInventoryResolver`
* Runtime settings façade:

  * `src/codeintel/build/settings.py :: get_build_settings()` / `get_hamilton_execution_settings()`

## 2.2 `src/codeintel/build/catalogs/` (canonical catalogs)

* Canonical catalogs build/load + metadata cache:

  * `src/codeintel/build/catalogs/canonical.py :: load_target_catalog(...)` / `load_contract_catalog(...)`
  * Target catalog is built from a freshly constructed Hamilton runtime graph: `src/codeintel/build/catalogs/canonical.py :: _build_target_catalog()`

## 2.3 `src/codeintel/build/hamilton/` (Phase4 expanded)

* Composition root and runtime:

  * `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)`
  * `src/codeintel/build/hamilton/runtime.py :: HamiltonRuntime`
* Env and execution:

  * `src/codeintel/build/hamilton/env.py :: BuildEnv`
  * `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor` / `HamiltonBuildResult`
  * Options: `src/codeintel/build/hamilton/execution_options.py :: BuildExecutionOptions`
* Target spec compilation & validation:

  * DAG→`OutputTarget` compiler: `src/codeintel/build/hamilton/target_spec_compiler.py :: compile_output_targets_from_driver(...)`
  * Graph validation entrypoint: `src/codeintel/build/hamilton/graph_validation.py :: validate_graph(...)`
  * Validator: `src/codeintel/build/hamilton/validate.py :: validate_nodes(...)`
* Naming/tagging/indexing/introspection:

  * `src/codeintel/build/hamilton/naming.py :: target_node(...)` / `materialize_node(...)` / loader/dataset/artifact node naming
  * `src/codeintel/build/hamilton/tagging.py :: tag_materialize(...)` / `tag_compute(...)` / `tag_tool(...)` / `tag_helper(...)`
  * `src/codeintel/build/hamilton/tag_index.py :: TagIndex.from_runtime(...)` (+ `data_saver_nodes()` groupings)
  * Introspection (deps, outputs, IO surface): `src/codeintel/build/hamilton/introspect.py :: derive_target_dependencies(...)`, `derive_target_outputs_from_savers(...)`, `derive_target_io_surface(...)`
* Support node generation:

  * `src/codeintel/build/hamilton/nodes/support_factory.py :: build_support_module(...)` (optionally driven by `DerivedTargetOutputs`)
* IO boundary + materializers:

  * Typed save-to decorator: `src/codeintel/build/hamilton/save_to.py :: SaveToObjectMetadataDecorator`
  * DataSavers:

    * `src/codeintel/build/hamilton/materializers/duckdb_saver.py :: DuckDBIbisTableSaver`
    * `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py :: DuckDBRowsSaver`
    * `src/codeintel/build/hamilton/materializers/artifact_saver.py :: FileArtifactSaver`
  * Artifact template helpers: `src/codeintel/build/hamilton/materializers/path_templates.py :: validate_path_template(...)` / `format_path_template(...)`
* Native targets + helpers:

  * Discovery: `src/codeintel/build/hamilton/native/discovery.py :: load_native_modules()`
  * Native target executor utility: `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor`
  * Target anchor decorator: `src/codeintel/build/hamilton/native/target_decorators.py :: codeintel_target(...)`
  * Materialization record builders: `src/codeintel/build/hamilton/native/materialization_records.py :: record_from_file_artifact_materializations(...)`
* Operational registries derived from DataSaver tags:

  * `src/codeintel/build/hamilton/io_registry.py :: compile_write_registry(...)`

## 2.4 CLI build composition roots

* Build run (execute): `src/codeintel/cli/handlers/build.py :: build_run_handler(...)` → `_execute_build_hamilton(...)`
* Build plan (dry run): `src/codeintel/cli/handlers/build.py :: build_plan_handler(...)`
* Build explain (includes IO surface option): `src/codeintel/cli/handlers/build.py :: build_explain_handler(...)` (uses `derive_target_io_surface(...)`), `src/codeintel/cli/commands/build.py :: BuildExplainCommand.io_surface`

---

# 3) Hamilton subsystem map (deep dive)

## 3.1 Composition root: `build_driver(...)`

* Constructs a **native-only** driver and a derived `TargetGraph`:

  * Builds a base Driver from `load_native_modules()` and compiles `OutputTarget` specs from DAG tags: `src/codeintel/build/hamilton/driver_factory.py :: _build_base_graph(...)`, `src/codeintel/build/hamilton/native/discovery.py :: load_native_modules()`, `src/codeintel/build/hamilton/target_spec_compiler.py :: compile_output_targets_from_driver(...)`
  * Derives target dependencies from Hamilton graph, then produces the final `TargetGraph`: `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)`, `src/codeintel/build/hamilton/introspect.py :: derive_target_dependencies(...)`, `target_graph_from_hamilton(...)`
* Generates a support module containing dataset/loader/artifact nodes:

  * Support module build: `src/codeintel/build/hamilton/driver_factory.py :: _build_support_graph_and_module(...)` → `src/codeintel/build/hamilton/nodes/support_factory.py :: build_support_module(...)`
  * Support outputs source can switch to saver-derived outputs: `src/codeintel/core/config/settings.py :: BuildSettings.support_nodes_source`, `src/codeintel/build/hamilton/driver_factory.py :: _build_support_graph_and_module(...)`
* Runtime container:

  * Returned as `HamiltonRuntime(dr, graph, target_to_node, node_to_target)`: `src/codeintel/build/hamilton/runtime.py :: HamiltonRuntime`

## 3.2 “Target spec lives on the DAG”

* Canonical target anchor decorator:

  * `@codeintel_target(domain=..., target=..., spec=TargetSpecDescriptor(...))` wraps `tag_materialize(...)` and injects spec tags:

    * `target_resources`, `target_execution`, `target_parameters`, `target_spec_version`, optional `target_estimated_duration_ms`. `src/codeintel/build/hamilton/native/target_decorators.py :: codeintel_target(...)`, `TargetSpecDescriptor`
    * Tag keys: `src/codeintel/core/hamilton/tags.py :: TAG_TARGET_RESOURCES` / `TAG_TARGET_EXECUTION` / `TAG_TARGET_PARAMETERS` / `TAG_TARGET_SPEC_VERSION`
* DAG→`OutputTarget` compilation:

  * Validates graph invariants then reads:

    * anchor `domain/target/spec_version` tags and docstring summary
    * JSON-encoded resources/execution/parameters tags
    * contract outputs derived from DataSaver tags (`output_role="contract"`). `src/codeintel/build/hamilton/target_spec_compiler.py :: compile_output_targets_from_driver(...)`, `_resources_from_tags(...)`, `_execution_from_tags(...)`, `_parameters_from_tags(...)`, `src/codeintel/build/hamilton/introspect.py :: derive_target_outputs_from_savers(...)`

## 3.3 Saver nodes are first-class “DAG-visible IO boundary”

* Typed save-to decorator produces a saver metadata node (a Hamilton node) with tags used for:

  * output derivation, validation, and IO registries. `src/codeintel/build/hamilton/save_to.py :: SaveToObjectMetadataDecorator.create_saver_node(...)`
* Saver tag identity requirements:

  * `output_role` must be static `value(...)` if present; must be `"contract"` or `"internal"`. `src/codeintel/build/hamilton/save_to.py :: _resolve_output_role(...)`
  * Contract savers enforce exactly one of `{table_key, artifact_name}`; artifact savers require `path_template` and validate placeholders. `src/codeintel/build/hamilton/save_to.py :: _resolve_output_identity(...)`, `_resolve_artifact_path_template(...)`, `src/codeintel/build/hamilton/materializers/path_templates.py :: validate_path_template(...)`
* DataSaver tag consumers:

  * Output derivation: `src/codeintel/build/hamilton/introspect.py :: derive_target_outputs_from_savers(...)`
  * Operational registry: `src/codeintel/build/hamilton/io_registry.py :: compile_write_registry(...)`
  * TagIndex grouping: `src/codeintel/build/hamilton/tag_index.py :: TagIndex.data_saver_nodes(...)`

## 3.4 Executor + hooks/adapters

* Executor entrypoint:

  * `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor.run(...)`
* Parallel adapter:

  * Built from runtime/graph policy: `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._build_runtime(...)` → `src/codeintel/build/hamilton/adapters/parallel.py :: create_parallel_adapter(...)`
* Hooks (telemetry + contracts + lifecycle):

  * Hook builder: `src/codeintel/build/hamilton/hooks/__init__.py :: build_hooks(...)`
  * Node telemetry hook flushes records via `BuildRunWriter`: `src/codeintel/build/hamilton/hooks/telemetry_hook.py :: NodeTelemetryHook.flush(...)`, `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.save_run_nodes(...)`
  * Contract enforcement hook activates per node based on node tags: `src/codeintel/build/hamilton/hooks/contract_hook.py :: ContractEnforcementHook.pre_node_execute(...)`

## 3.5 Validation subsystem

* Node graph validation:

  * Validates materialize anchors, dataset/artifact nodes, saver outputs (tables/artifacts/templates), and optional compute I/O purity: `src/codeintel/build/hamilton/validate.py :: validate_nodes(...)`
* Runtime “validate graph” helper:

  * Constructs runtime then validates it: `src/codeintel/build/hamilton/graph_validation.py :: validate_graph(...)`

---

# 4) Core runtime concepts & types (type map)

## 4.1 Targets, graphs, contracts

* Target metadata:

  * `src/codeintel/build/targets.py :: OutputTarget(name, module, contract, dependencies, resources, execution, parameters, description, ...)`
* Dependency graph:

  * `src/codeintel/build/targets.py :: TargetGraph.register(...)`, `topological_order(...)`, `validate()`
* Contract types:

  * `src/codeintel/build/contracts.py :: OutputContract`
  * `src/codeintel/build/contracts.py :: ArtifactSpec`

## 4.2 Target spec compilation and overrides

* Target spec override type (compiler-local “small override layer”):

  * `src/codeintel/build/hamilton/target_spec_compiler.py :: TargetSpecOverride`
* Anchor decorator spec bundle:

  * `src/codeintel/build/hamilton/native/target_decorators.py :: TargetSpecDescriptor`

## 4.3 Build environment, run context, execution options

* Build environment passed to Hamilton DAG:

  * `src/codeintel/build/hamilton/env.py :: BuildEnv`
* Run context builder:

  * `src/codeintel/build/run_context.py :: BuildRunContext`
  * Overrides bundle: `src/codeintel/build/run_context.py :: BuildRunContextOverrides`
* Hamilton execution options:

  * `src/codeintel/build/hamilton/execution_options.py :: BuildExecutionOptions`
* Build runtime settings (env-injected):

  * `src/codeintel/core/config/settings.py :: BuildSettings(output_inventory_source, output_inventory_strict, support_nodes_source, ...)`
  * Hamilton execution settings: `src/codeintel/core/config/settings.py :: HamiltonExecutionSettings(...)`

## 4.4 Output inventory and target metadata service

* Inventory data model:

  * `src/codeintel/build/output_inventory.py :: OutputInventory(datasets_by_target, artifacts_by_target, artifact_templates_by_target)`
* Inventory resolver:

  * `src/codeintel/build/target_inventory.py :: resolve_output_inventory(...)`
  * Mode literal: `src/codeintel/build/target_inventory.py :: OutputInventoryMode`
* Target graph/spec catalog:

  * `src/codeintel/build/target_catalog.py :: load_target_specs(...)`, `target_graph_from_catalog(...)`
* Target metadata service:

  * Runtime+graph+indexes: `src/codeintel/build/target_metadata.py :: TargetSystem`
  * Service bundle: `src/codeintel/build/target_metadata.py :: TargetMetadataService`
  * Singleton loader: `src/codeintel/build/target_metadata.py :: get_target_metadata_service(...)`

## 4.5 Incremental artifacts: manifests and hash evaluation

* Persisted manifest record:

  * `src/codeintel/core/build_manifest.py :: OutputManifest`
* Hash evaluation:

  * `src/codeintel/build/hash_evaluator.py :: HashEvaluation` / `evaluate_hash_state(...)`
  * Planning-level evaluation: `src/codeintel/build/hash_evaluator.py :: compute_hash_evaluation(...)`
* Skip decision utilities:

  * `src/codeintel/build/hamilton/run_record_utils.py :: SkipCheckRequest`
  * `src/codeintel/build/hamilton/run_record_utils.py :: should_skip_native_target(...)`
* Native executor’s per-target input hash computation:

  * `src/codeintel/build/hamilton/run_records.py :: compute_target_input_hash(...)`
  * Wrapped in: `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor.for_target(...)`

## 4.6 Run records and execution results

* Per-target runtime record type (Hamilton node output for `t__*` targets):

  * `src/codeintel/build/hamilton/run_records.py :: TargetRunRecord` (re-export)
  * Underlying core type: `src/codeintel/core/hamilton/records.py :: TargetRunRecord`
* Standard compute-step result (used by executor-style targets):

  * `src/codeintel/build/hamilton/execution_result.py :: ExecutionResult` / `to_execution_result(...)`
* Build run record (run-level persistence):

  * `src/codeintel/core/build_manifest.py :: BuildRunRecord`
  * Writer: `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.start_run(...)`, `complete_run(...)`

## 4.7 DAG-derived IO surface types

* Derived outputs from savers:

  * `src/codeintel/build/hamilton/introspect.py :: DerivedTargetOutputs`
* Per-target IO surface map:

  * `src/codeintel/build/hamilton/introspect.py :: TargetIOSurface` / `TableRead` / `TableWrite` / `ArtifactWrite`
  * Derivation function: `src/codeintel/build/hamilton/introspect.py :: derive_target_io_surface(...)`

---

# 5) Data & IO model (as implemented)

## 5.1 Primary “data objects” that flow through the DAG

* **Target node output**: `TargetRunRecord` (status + hashes + row_counts + datasets/artifacts). `src/codeintel/core/hamilton/records.py :: TargetRunRecord`
* **Dataset handle**: `DatasetRef` (not the data), used by loader nodes:

  * `src/codeintel/build/hamilton/io/dataset_ref.py :: DatasetRef`
* **Artifact handle**: `ArtifactRef` (not the data), produced by artifact support nodes:

  * `src/codeintel/build/hamilton/io/artifact_ref.py :: ArtifactRef`
* **Materialization metadata** (typed at DAG boundary):

  * `src/codeintel/build/hamilton/boundary_types.py :: MaterializationMetadata`

## 5.2 Write boundaries: DataSaver metadata nodes (`m__*`)

* Save-to decorator wraps a compute node with a saver metadata node:

  * `src/codeintel/build/hamilton/save_to.py :: SaveToObjectMetadataDecorator.transform_node(...)`
* Saver nodes emit `MaterializationMetadata` dicts (parsed downstream via typed metadata models):

  * DuckDB metadata: `src/codeintel/build/hamilton/materializers/metadata.py :: DuckDBMaterializationMetadata`
  * Artifact metadata: `src/codeintel/build/hamilton/materializers/metadata.py :: FileArtifactMaterializationMetadata`

## 5.3 DuckDB materialization: tables and rows

* Ibis table → DuckDB:

  * Saver: `src/codeintel/build/hamilton/materializers/duckdb_saver.py :: DuckDBIbisTableSaver.save_data(...)`
  * Uses `env.warehouse.materialize_table(...)`: `src/codeintel/storage/warehouse.py :: Warehouse.materialize_table(...)`
* Row tuples → DuckDB:

  * Saver: `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py :: DuckDBRowsSaver.save_data(...)`
  * Uses deferred column resolution: `src/codeintel/build/schemas/__init__.py :: DeferredColumns` / `deferred_columns_for_table_key(...)`
  * Warehouse write path: `src/codeintel/storage/warehouse.py :: Warehouse.materialize_dataframe(...)` / `materialize_rows(...)` (called inside saver)

## 5.4 Artifact materialization: template-driven paths (DAG metadata)

* Artifact saver accepts `path_template` and resolves output path via template formatting:

  * `src/codeintel/build/hamilton/materializers/artifact_saver.py :: FileArtifactSaver(path_template=...)`
  * Resolution is template-first (raises if missing): `src/codeintel/build/hamilton/materializers/artifact_saver.py :: _resolve_artifact_path(...)`
* Template validation and formatting:

  * `src/codeintel/build/hamilton/materializers/path_templates.py :: validate_path_template(...)`
  * `src/codeintel/build/hamilton/materializers/path_templates.py :: format_path_template(...)`, `default_formatter(...)`
* Saver node tags require `artifact_path_template` for contract artifact outputs (enforced during saver tag derivation and during output derivation):

  * `src/codeintel/core/hamilton/tags.py :: TAG_ARTIFACT_PATH_TEMPLATE`
  * Output derivation requires it: `src/codeintel/build/hamilton/introspect.py :: derive_target_outputs_from_savers(...)`

## 5.5 Read boundaries: support nodes `d__/q__/df__/a__`

* Dataset node `d__<table_key>` extracts `DatasetRef` from producing target record:

  * `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_dataset_node_function(...)`
* Loader nodes:

  * Ibis query loader `q__*`: `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_query_node_function(...)` → `src/codeintel/build/hamilton/io/ibis_adapter.py :: load_dataset_ibis(...)`
  * DataFrame loader `df__*`: `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_dataframe_node_function(...)` → `src/codeintel/build/hamilton/io/ibis_adapter.py :: load_dataset_df(...)`
* Artifact node `a__<artifact_name>` extracts `ArtifactRef` from producing target record:

  * `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_artifact_node_function(...)`

## 5.6 Persistence surfaces (storage gateway boundary)

* Manifest and run tracking types are in `codeintel.core` to avoid layering violations:

  * `src/codeintel/core/build_manifest.py :: OutputManifest`, `BuildRunRecord`
* Build writer persists:

  * run start/complete: `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.start_run(...)`, `complete_run(...)`
  * per-target records: `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.save_run_targets(...)`
  * per-node telemetry: `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.save_run_nodes(...)`

---

# 6) Target orchestration model (derived behavior)

## 6.1 Target discovery and dependency derivation (DAG → TargetGraph)

* Build driver compiles base target specs then derives dependencies from the Hamilton graph:

  * `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)`
  * Dependency derivation: `src/codeintel/build/hamilton/introspect.py :: derive_target_dependencies(...)`
  * Graph materialization: `src/codeintel/build/hamilton/introspect.py :: target_graph_from_hamilton(...)`

## 6.2 Closure computation (requested targets → full dependency closure)

* Closure is computed using the `TargetGraph` topological order:

  * `src/codeintel/build/targets.py :: TargetGraph.topological_order(...)`
* CLI “build run” uses the target graph from the target metadata service:

  * `src/codeintel/cli/handlers/build.py :: build_run_handler(...)` (uses `get_target_metadata_service().system.graph`)

## 6.3 Planning (“build plan”): compute vs skip vs blocked vs missing

* Plan builds a Hamilton runtime and reads the target graph:

  * `src/codeintel/build/hamilton/planner.py :: compute_plan(...)` (calls `build_driver()`)
* Hash evaluation uses current input hash / options hash vs stored manifest:

  * `src/codeintel/build/hash_evaluator.py :: compute_hash_evaluation(...)`
  * `src/codeintel/build/hash_evaluator.py :: evaluate_hash_state(...)`
  * `src/codeintel/core/build_manifest.py :: OutputManifest`
* Plan statuses and reasons are encoded on `PlanEntry`:

  * `src/codeintel/build/hamilton/planner.py :: PlanEntry(status, reason, ...)`

## 6.4 Execution (“build run”): Hamilton driver execute

* CLI constructs `BuildEnv` and runs the executor:

  * `src/codeintel/cli/handlers/build.py :: _execute_build_hamilton(...)` → `BuildRunContext.build_env(...)` → `HamiltonBuildExecutor.run(...)`
  * `src/codeintel/build/run_context.py :: BuildRunContext.build_env(...)`
  * `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor.run(...)`
* Executor runs the DAG with `inputs={"env": env}` and `final_vars=[t__* nodes for closure]` (final vars mapping is target→node):

  * `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._execute_dag(...)`
  * Node mapping: `src/codeintel/build/hamilton/driver_factory.py :: target_to_node_name(...)`

## 6.5 Runtime incremental behavior (native executor and manifest skip)

* Many native targets use `NativeTargetExecutor` to unify:

  * input hash computation, skip decision, record creation, manifest persistence. `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor`
* Skip check:

  * `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor.should_skip(...)` → `src/codeintel/build/hamilton/run_record_utils.py :: should_skip_native_target(...)`
* Manifest persistence on success:

  * `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor.execute(...)` calls `save_manifest(...)` on success via the run-record layer. `src/codeintel/build/hamilton/run_records.py :: save_manifest(...)`

## 6.6 DAG-derived IO registry and surface introspection

* IO registry (writes grouped by sink) is computed from DataSaver tags:

  * `src/codeintel/build/hamilton/io_registry.py :: compile_write_registry(...)`
* Per-target IO surface (reads/writes) can be derived by traversing upstream dependencies and collecting loader/saver tags:

  * `src/codeintel/build/hamilton/introspect.py :: derive_target_io_surface(...)`
  * CLI build explain can include this: `src/codeintel/cli/handlers/build.py :: build_explain_handler(...)` (gate via `ctx.params.get_bool("io_surface")`)

---

# 7) Walkthrough: “request one target” end-to-end trace

> Representative target: **`scip`** (ingestion domain). This module demonstrates:
>
> * a tool step (`t__scip__run`)
> * artifact savers with `path_template`
> * row materialization savers for DuckDB
> * a final target anchor `t__scip` with `@codeintel_target(...)` and `TargetRunRecord` output.
>
> `src/codeintel/build/hamilton/native/ingestion/scip.py :: SCIP_TARGET_NAME`, `t__scip(...)`

## 7.1 CLI to executor

1. **CLI resolves the target graph and goals**:

* `src/codeintel/cli/handlers/build.py :: build_run_handler(...)` uses `get_target_metadata_service().system.graph`.
* Resolves goals: `src/codeintel/cli/handlers/build.py :: _resolve_goals(...)`.

2. **CLI opens a gateway, loads manifests, and constructs `BuildEnv`**:

* Manifest list/index: `src/codeintel/cli/handlers/build.py :: _execute_build_hamilton(...)` (`gateway.build.list_manifests(...)`, `manifest_index = {...}`)
* Run context construction: `src/codeintel/build/run_context.py :: BuildRunContext.from_execution_context(...)`
* Env build: `src/codeintel/build/run_context.py :: BuildRunContext.build_env(...)` → `src/codeintel/build/hamilton/env.py :: BuildEnv(...)`

3. **CLI runs Hamilton executor**:

* `src/codeintel/cli/handlers/build.py :: _execute_build_hamilton(...)` instantiates `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor` and calls `.run(env=env, targets=goals)`.

## 7.2 Executor closure and DAG execution

4. **Executor constructs a Hamilton runtime** (native modules + support module):

* `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._build_runtime(...)` calls `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)`.

5. **Executor computes dependency closure and maps to `t__` nodes**:

* Closure: `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._compute_closure(...)` → `src/codeintel/build/targets.py :: TargetGraph.topological_order(...)`
* Target→node mapping: `src/codeintel/build/hamilton/executor.py :: _map_closure_to_nodes(...)` uses `src/codeintel/build/hamilton/driver_factory.py :: target_to_node_name(...)`.

6. **Driver executes final vars**:

* `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._execute_dag(...)` executes `runtime.dr.execute(final_vars, inputs={"env": execution_env})`.

## 7.3 `scip` native module: key nodes

7. **Tool step** (not the target anchor; tagged as tool):

* `src/codeintel/build/hamilton/native/ingestion/scip.py :: t__scip__run(env, graph, t__modules) -> ScipRunResult`
* Skip check uses `NativeTargetExecutor`:

  * `src/codeintel/build/hamilton/native/ingestion/scip.py :: t__scip__run(...)` calls `NativeTargetExecutor.for_target(...).should_skip()`.
  * `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor.should_skip(...)`

8. **Artifact compute nodes + saver metadata nodes**:

* Artifact compute nodes:

  * `src/codeintel/build/hamilton/native/ingestion/scip.py :: scip__index_artifact(...) -> Path | None`
  * `src/codeintel/build/hamilton/native/ingestion/scip.py :: scip__json_artifact(...) -> Path | None`
* Each compute node is decorated with `SaveToObjectMetadataDecorator([FileArtifactSaver], ...)` supplying:

  * `target_name=value("scip")`, `artifact_name=value(...)`, and **`path_template=value(...)`**:

    * `src/codeintel/build/hamilton/native/ingestion/scip.py :: SaveToObjectMetadataDecorator(...)` (for both artifact nodes)
* The resulting saver nodes contribute tags consumed by output derivation:

  * Tag emission: `src/codeintel/build/hamilton/save_to.py :: SaveToObjectMetadataDecorator.create_saver_node(...)`
  * Output derivation reads tags: `src/codeintel/build/hamilton/introspect.py :: _iter_contract_saver_tags(...)`

9. **Row materialization nodes**:

* Row compute nodes:

  * `src/codeintel/build/hamilton/native/ingestion/scip.py :: scip__symbol_rows(...)`
  * `src/codeintel/build/hamilton/native/ingestion/scip.py :: scip__occurrence_rows(...)`
* Each uses `SaveToObjectMetadataDecorator([DuckDBRowsSaver], output_name_=materialize_node(<table_key>), ...)`:

  * `src/codeintel/build/hamilton/native/ingestion/scip.py :: SaveToObjectMetadataDecorator(...)` (DuckDBRowsSaver usage)
  * Table-key naming: `src/codeintel/build/hamilton/naming.py :: materialize_node(...)`

10. **Target anchor node** (`t__scip`) produces `TargetRunRecord`:

* Decorated with `@codeintel_target(domain="ingestion", target="scip", spec=TargetSpecDescriptor(...))`:

  * `src/codeintel/build/hamilton/native/ingestion/scip.py :: t__scip(...)`
  * Decorator implementation: `src/codeintel/build/hamilton/native/target_decorators.py :: codeintel_target(...)`
* Returns a target record assembled from artifact materializations + row counts:

  * `src/codeintel/build/hamilton/native/ingestion/scip.py :: t__scip(...)` calls `record_from_file_artifact_materializations(...)`
  * Record builder: `src/codeintel/build/hamilton/native/materialization_records.py :: record_from_file_artifact_materializations(...)`

## 7.4 Post-run persistence

11. **Executor categorizes computed/skipped/failed** based on `TargetRunRecord.status`:

* `src/codeintel/build/hamilton/executor.py :: _categorize_outputs(...)`

12. **Build run writer persists run + target records (+ telemetry if enabled)**:

* `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._run_with_state(...)`
* Writer: `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.start_run(...)`, `save_run_targets(...)`, `complete_run(...)`

---

# 8) Extension mechanics (how new behavior is introduced today)

## 8.1 Add a new build target (DAG-native spec anchor)

* Create a new native module under `src/codeintel/build/hamilton/native/<domain>/...` (module discovery):

  * `src/codeintel/build/hamilton/native/discovery.py :: native_module_paths(...)`, `load_native_modules()`
* Define a target anchor function `t__<target>` and decorate with:

  * `src/codeintel/build/hamilton/native/target_decorators.py :: codeintel_target(domain=..., target=..., spec=...)`
* Ensure the target anchor has a docstring summary (used for `OutputTarget.description`):

  * Docstring extraction: `src/codeintel/build/hamilton/target_spec_compiler.py :: _node_docstring(...)`, `_summary(...)`

## 8.2 Add table outputs (DuckDB)

* Add compute nodes returning either:

  * ibis table expressions (`DuckDBIbisTableSaver`) or
  * row tuples (`DuckDBRowsSaver`),
    typically tagged as compute:
  * `src/codeintel/build/hamilton/tagging.py :: tag_compute(...)`
* Attach a saver metadata node with `SaveToObjectMetadataDecorator(...)` supplying static identity via `value(...)`:

  * `src/codeintel/build/hamilton/save_to.py :: SaveToObjectMetadataDecorator`
* Table schema must exist in the table schema registry (compiler requires it):

  * `src/codeintel/build/hamilton/target_spec_compiler.py :: _resolve_table_schemas(...)` (calls `src/codeintel/core/schemas/table_registry.py :: get_table_schema(...)`)

## 8.3 Add artifact outputs (filesystem)

* Add a compute node returning `Path | None` (or other supported artifact payload types as accepted by `FileArtifactSaver`), tagged as compute:

  * Example pattern: `src/codeintel/build/hamilton/native/ingestion/scip.py :: scip__index_artifact(...)`
* Attach a `FileArtifactSaver` via `SaveToObjectMetadataDecorator`, with:

  * `target_name=value(<target>)`
  * `artifact_name=value(<artifact>)`
  * `path_template=value(<template>)` (validated for allowed placeholders)
  * `src/codeintel/build/hamilton/save_to.py :: _resolve_artifact_path_template(...)`, `src/codeintel/build/hamilton/materializers/path_templates.py :: validate_path_template(...)`

## 8.4 Control what counts as a “contract output” vs “internal output”

* Saver metadata nodes use `output_role` tag to classify outputs; derivation functions filter to `output_role="contract"`:

  * Saver tag enforcement: `src/codeintel/build/hamilton/save_to.py :: _resolve_output_role(...)`
  * Derivation filter: `src/codeintel/build/hamilton/introspect.py :: _require_output_role(...)`

## 8.5 Add support-node surfaces for new outputs

* Support nodes are generated per target from either:

  * target contracts (`OutputTarget.contract`), or
  * saver-derived outputs (`DerivedTargetOutputs`) when enabled:

    * `src/codeintel/build/hamilton/driver_factory.py :: _build_support_graph_and_module(...)` (checks `BuildSettings.support_nodes_source`)
    * `src/codeintel/build/hamilton/nodes/support_factory.py :: build_support_module(..., derived_outputs=...)`

---

# 9) Conventions & invariants (observed)

## 9.1 Naming conventions (stable node identity)

* Node naming functions define stable prefixes:

  * `t__*` targets: `src/codeintel/build/hamilton/naming.py :: target_node(...)`
  * `m__*` materializers: `src/codeintel/build/hamilton/naming.py :: materialize_node(...)`
  * `d__*` dataset nodes: `src/codeintel/build/hamilton/naming.py :: dataset_node(...)`
  * `q__*` and `df__*` loader nodes: `src/codeintel/build/hamilton/naming.py :: query_node(...)`, `dataframe_node(...)`
  * `a__*` artifact nodes: `src/codeintel/build/hamilton/naming.py :: artifact_node(...)`

## 9.2 Tag conventions (canonical keys and node types)

* Canonical tag keys and node type values are centralized:

  * `src/codeintel/core/hamilton/tags.py :: TAG_DOMAIN` / `TAG_TARGET` / `TAG_TABLE_KEY` / `TAG_ARTIFACT` / `TAG_NODE_TYPE`
  * Node types: `src/codeintel/core/hamilton/tags.py :: NODE_TYPE_MATERIALIZE` / `NODE_TYPE_COMPUTE` / `NODE_TYPE_DATASET` / `NODE_TYPE_ARTIFACT` / loader kinds

## 9.3 Target anchor invariants (validated)

* Materialize anchor nodes must have:

  * `domain` and `target` tags, and `target_spec_version == "1"`. `src/codeintel/build/hamilton/validate.py :: _collect_materialize_index(...)`
* DAG validation can be run via:

  * `src/codeintel/build/hamilton/graph_validation.py :: validate_graph(...)`

## 9.4 Saver node invariants (contract outputs are tag-derivable)

* DataSaver nodes are identified by `hamilton.data_saver == True` tags (and related sink/class tags):

  * Tag construction: `src/codeintel/build/hamilton/save_to.py :: _build_saver_tags(...)`, `SaverTagContext`
  * Tag indexing: `src/codeintel/build/hamilton/tag_index.py :: TagIndex.data_saver_nodes(...)`
* Contract outputs must be attributable and self-identifying:

  * `output_role` must be `"contract"` or `"internal"`: `src/codeintel/build/hamilton/introspect.py :: _require_output_role(...)`
  * Contract savers must have exactly one of `{table_key, artifact}`: `src/codeintel/build/hamilton/introspect.py :: _resolve_output_identity(...)`
  * Contract artifact savers must have `artifact_path_template`: `src/codeintel/build/hamilton/introspect.py :: _iter_contract_saver_tags(...)`

## 9.5 Output inventory drift handling (declared vs DAG)

* Inventory mismatch computation compares:

  * table sets, artifact sets, and artifact templates per target. `src/codeintel/build/target_inventory.py :: _diff_inventories(...)`
* Settings control strictness and source:

  * `src/codeintel/core/config/settings.py :: BuildSettings.output_inventory_source`, `output_inventory_strict`

---

# 10) Glossary (project-specific vocabulary)

* **Target anchor**: A `t__<name>` node tagged `node_type="materialize"` and decorated with `@codeintel_target(...)`; used to define the existence/spec of a build target. `src/codeintel/build/hamilton/native/target_decorators.py :: codeintel_target(...)`, `src/codeintel/core/hamilton/tags.py :: NODE_TYPE_MATERIALIZE`
* **DAG-native target spec**: `OutputTarget` compiled from Hamilton node tags + docstring + saver-derived outputs. `src/codeintel/build/hamilton/target_spec_compiler.py :: compile_output_targets_from_driver(...)`
* **DataSaver node**: A Hamilton node created by `SaveToObjectMetadataDecorator` that performs persistence and returns `MaterializationMetadata`; tagged with `hamilton.data_saver` and identity tags (`target`, `table_key`/`artifact`). `src/codeintel/build/hamilton/save_to.py :: SaveToObjectMetadataDecorator.create_saver_node(...)`
* **output_role**: Saver tag that classifies outputs as `"contract"` (official output inventory) vs `"internal"` (excluded from contract/output derivation). `src/codeintel/build/hamilton/save_to.py :: _resolve_output_role(...)`
* **OutputInventory**: Per-target inventory of dataset keys, artifact names, and artifact path templates; can be derived from declared targets or from saver tags. `src/codeintel/build/output_inventory.py :: OutputInventory`, `src/codeintel/build/target_inventory.py :: resolve_output_inventory(...)`
* **Support nodes**: Generated nodes (`d__*`, `q__*`, `df__*`, `a__*`) that expose datasets/artifacts and loader access patterns to the DAG. `src/codeintel/build/hamilton/nodes/support_factory.py :: build_support_module(...)`
* **Manifest**: Stored record (`OutputManifest`) of a target computation including `input_hash` (and optional dep hashes/options hash), used for skip decisions and planning. `src/codeintel/core/build_manifest.py :: OutputManifest`
* **NativeTargetExecutor**: Helper that centralizes per-target input hash computation, skip checks, record creation, and manifest persistence for native targets. `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor`
* **Graph validation**: Deterministic validator that checks DAG invariants required by this build architecture (anchors, output tags, templates). `src/codeintel/build/hamilton/validate.py :: validate_nodes(...)`, `src/codeintel/build/hamilton/graph_validation.py :: validate_graph(...)`
