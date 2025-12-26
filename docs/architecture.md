Below is a **revised** version of the Phase4 architecture overview you shared earlier (the `architecture.md` you attached) with the **audit corrections and additions integrated**. 

I have **only removed/changed** content where it was **confirmed incorrect** (notably: `InvalidTargetSpecError`, the `DeferredColumns` path, and the missing `graph` input in execution). Everything else is preserved and expanded.

---

# 0) What I inspected

* **Codebase**: extracted `CodeIntel_Centralizing_Phase4.zip` and inspected:

  * `AGENTS.md` (agent constraints + repo “do-not-edit zones” inventory). `AGENTS.md :: (document)`
  * `pyproject.toml` (package layout + tool config). `pyproject.toml :: (project metadata)`
  * Build layer package tree: `src/codeintel/build/**` (all modules/subpackages; emphasis on `build/hamilton/**`). `src/codeintel/build/__init__.py :: __getattr__(...)`
  * CLI build entrypoints and handlers (composition roots):

    * `src/codeintel/cli/handlers/build.py :: build_run_handler(...)`
    * `src/codeintel/cli/handlers/build.py :: build_plan_handler(...)`
    * `src/codeintel/cli/handlers/build.py :: build_explain_handler(...)`
    * plus additional Phase4 build handlers (see §2.4). `src/codeintel/cli/handlers/build.py :: build_history_handler(...)` / `build_graph_handler(...)` / `build_assets_handler(...)` / `build_lineage_handler(...)` / `build_promote_handler(...)` / `build_resolve_handler(...)` / `build_diff_handler(...)` / `build_impact_handler(...)`

* **Implementation plans provided (context only; code is authoritative)**:

  * `centralization_big_move_1.md`
  * `centralization_big_move_2.md`
  * `centralization_big_move_3.md`

---

# 1) Executive architecture summary

* **Single orchestration engine**: build execution is driven by a Hamilton `Driver` constructed from *native target modules* plus a generated support module.
  `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)`

* **Targets are DAG-defined (not registry-defined)**:

  * A build target is anchored by a `t__<target>` node tagged `node_type="materialize"`, with `domain`, `target`, and spec tags.
    `src/codeintel/build/hamilton/native/target_decorators.py :: codeintel_target(...)`
    `src/codeintel/core/hamilton/tags.py :: TAG_NODE_TYPE` / `NODE_TYPE_MATERIALIZE`

  * `OutputTarget` specs are compiled from DAG tags + docstrings + saver-derived outputs.
    `src/codeintel/build/hamilton/target_spec_compiler.py :: compile_output_targets_from_driver(...)`

* **Output inventory is DAG-derived**:

  * Output identity and artifact templates are derived from **DataSaver tags** (contract-only `output_role="contract"`).
    `src/codeintel/build/hamilton/introspect.py :: derive_target_outputs_from_savers(...)`

* **Support nodes (dataset/loader/artifact nodes) are generated from DAG-derived saver outputs**.
  `src/codeintel/build/hamilton/driver_factory.py :: _build_support_graph_and_module(...)`
  `src/codeintel/build/hamilton/nodes/support_factory.py :: build_support_module(...)`

* **Runtime execution (“build run”) path**:

  * CLI resolves goals using the target graph from the target metadata service, builds a `BuildRunContext`, constructs `BuildEnv`, and runs `HamiltonBuildExecutor.run(...)`.
    `src/codeintel/cli/handlers/build.py :: build_run_handler(...)` / `_execute_build_hamilton(...)`
    `src/codeintel/build/run_context.py :: BuildRunContext.build_env(...)`
    `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor.run(...)`

* **Incremental behavior**:

  * “Skip” decisions use manifest hash evaluation (`evaluate_hash_state`) against `OutputManifest` loaded from the storage gateway, with a force-target bypass.
    `src/codeintel/build/hamilton/run_record_utils.py :: should_skip_native_target(...)`
    `src/codeintel/build/hash_evaluator.py :: evaluate_hash_state(...)`
    `src/codeintel/core/build_manifest.py :: OutputManifest`

* **Hamilton cache (execution feature)**:

  * Driver construction supports enabling Hamilton cache and setting cache dir.
    `src/codeintel/build/hamilton/driver_factory.py :: build_driver(..., enable_cache: bool, cache_dir: ...)`
  * Execution options include cache toggles.
    `src/codeintel/build/hamilton/execution_options.py :: BuildExecutionOptions(enable_hamilton_cache, cache_dir, ...)`

* **“DAG-first invariants” have an explicit validator**:

  * `validate_nodes(...)` checks materialize anchors, support nodes, saver tags (including artifact templates), and optional compute I/O purity.
    `src/codeintel/build/hamilton/validate.py :: validate_nodes(..., enforce_compute_io_purity: bool = False, ...)`

  * `validate_graph()` builds the driver then validates it.
    `src/codeintel/build/hamilton/graph_validation.py :: validate_graph(...)`

---

# 2) Repository map (build-focused)

## 2.1 `src/codeintel/build/` (Phase4 notable modules)

* Public facade + lazy exports:

  * `src/codeintel/build/__init__.py :: _LAZY_IMPORTS` / `__getattr__(...)`

* Core target primitives:

  * `src/codeintel/build/targets.py :: OutputTarget` / `TargetGraph` / `TargetModule`
  * Target errors (example): `src/codeintel/build/errors.py :: TargetNotFoundError`
    *(Removed: `InvalidTargetSpecError` — not present in code.)* `src/codeintel/build/errors.py :: (module)`

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

* Engine version (hash invalidation surface):

  * `src/codeintel/build/engine_version.py :: get_build_engine_version(...)`

* Build “types” bundle (build-wide data/result types):

  * `src/codeintel/build/types.py :: (module-level types)`

* Providers and run config (still present and used in run-context wiring):

  * `src/codeintel/build/providers.py :: (module)`
  * `src/codeintel/build/run_config.py :: (module)`

* **New/centralized metadata & inventory services**:

  * Target metadata service (runtime + indexes + tag index + schema index):
    `src/codeintel/build/target_metadata.py :: get_target_metadata_service(...)` / `TargetSystem` / `TargetMetadataService`
  * Target graph (DAG-derived, canonical):
    `src/codeintel/build/target_metadata.py :: get_target_system(...)` / `TargetSystem.graph`
  * DAG output inventory (checked-in artifact):
    `src/codeintel/core/registry/service.py :: DagOutputInventory` /
    `src/codeintel/core/registry/dag_output_inventory.yaml`
  * DAG-derived outputs for runtime checks:
    `src/codeintel/build/hamilton/introspect.py :: derive_target_outputs_from_savers(...)`

* Runtime settings façade:

  * `src/codeintel/build/settings.py :: get_build_settings()` / `get_hamilton_execution_settings()`

## 2.2 Build subpackages still present (high-level inventory)

*(These existed in earlier phases and remain part of the build surface; they should be included in a “build-focused” map.)*

* Assets: `src/codeintel/build/assets/*`

  * `src/codeintel/build/assets/fingerprinting.py :: FingerprintPolicy`
  * `src/codeintel/build/assets/impact.py :: compute_impact(...)`

* Schemas: `src/codeintel/build/schemas/*`

  * Column resolution: `src/codeintel/build/schemas/column_resolution.py :: DeferredColumns` / `resolve_columns(...)`
  * Manifest/diff: `src/codeintel/build/schemas/manifest.py :: (module)` / `diff.py :: (module)`

* Spec: `src/codeintel/build/spec/*`

  * `src/codeintel/build/spec/compile.py :: compile_buildspec(...)`

* Exports: `src/codeintel/build/exports/*`

  * `src/codeintel/build/exports/runner.py :: ExportRunner`

* Serving: `src/codeintel/build/serving/*`

  * `src/codeintel/build/serving/publisher.py :: publish_serving_snapshot(...)`

## 2.3 Catalog cache (removed)

Canonical build catalogs have been removed. Registry consumers now derive contracts from the
schema service and targets from the Hamilton DAG via `TargetMetadataService`.

## 2.4 `src/codeintel/build/hamilton/` (Phase4 expanded)

* Composition root and runtime:

  * `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)`
  * `src/codeintel/build/hamilton/runtime.py :: HamiltonRuntime`

* Env and execution:

  * `src/codeintel/build/hamilton/env.py :: BuildEnv`
  * `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor` / `HamiltonBuildResult`
  * Options: `src/codeintel/build/hamilton/execution_options.py :: BuildExecutionOptions`

* Target spec compilation & validation:

  * DAG→`OutputTarget` compiler:
    `src/codeintel/build/hamilton/target_spec_compiler.py :: compile_output_targets_from_driver(...)`
  * Graph validation entrypoint: `src/codeintel/build/hamilton/graph_validation.py :: validate_graph(...)`
  * Validator: `src/codeintel/build/hamilton/validate.py :: validate_nodes(...)`

* Naming/tagging/indexing/introspection:

  * `src/codeintel/build/hamilton/naming.py :: target_node(...)` / `materialize_node(...)` / loader/dataset/artifact node naming
  * `src/codeintel/build/hamilton/tagging.py :: tag_materialize(...)` / `tag_compute(...)` / `tag_tool(...)` / `tag_helper(...)`
  * `src/codeintel/build/hamilton/tag_index.py :: TagIndex.from_runtime(...)` (+ `data_saver_nodes()` groupings)
  * Introspection (deps, outputs, IO surface):
    `src/codeintel/build/hamilton/introspect.py :: derive_target_dependencies(...)` / `derive_target_outputs_from_savers(...)` / `derive_target_io_surface(...)`

* Options loading / runtime options normalization:

  * `src/codeintel/build/hamilton/options_loading.py :: load_target_options(...)`
  * `src/codeintel/build/hamilton/graph_runtime_options.py :: load_graph_runtime_options(...)`

* Impl-kind detection:

  * `src/codeintel/build/hamilton/impl_kind.py :: target_impl_kind(...)`

* Materialization helper (executor-driven materialize targets):

  * `src/codeintel/build/hamilton/materialization_helpers.py :: executor_materialize(...)`

* Observability exports (graph render/export):

  * `src/codeintel/build/hamilton/observability.py :: export_dag_json(...)` (and related exports)

* Support node generation:

  * `src/codeintel/build/hamilton/nodes/support_factory.py :: build_support_module(...)` (optionally driven by `DerivedTargetOutputs`)

* IO boundary + materializers:

  * Typed save-to decorator: `src/codeintel/build/hamilton/save_to.py :: SaveToObjectMetadataDecorator`
  * DataSavers:

    * `src/codeintel/build/hamilton/materializers/duckdb_saver.py :: DuckDBIbisTableSaver`
    * `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py :: DuckDBRowsSaver`
    * `src/codeintel/build/hamilton/materializers/artifact_saver.py :: FileArtifactSaver`
  * Artifact template helpers:
    `src/codeintel/build/hamilton/materializers/path_templates.py :: validate_path_template(...)` / `format_path_template(...)`

* Native targets + helpers:

  * Discovery: `src/codeintel/build/hamilton/native/discovery.py :: load_native_modules()`
  * Native target executor utility: `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor`
  * Target anchor decorator: `src/codeintel/build/hamilton/native/target_decorators.py :: codeintel_target(...)`
  * Materialization record builders:
    `src/codeintel/build/hamilton/native/materialization_records.py :: record_from_file_artifact_materializations(...)`

* Contracts/validators subpackages (still present; IO enforcement and validation):

  * `src/codeintel/build/hamilton/contracts/enforcement.py :: (module)`
  * `src/codeintel/build/hamilton/contracts/enforced_gateway.py :: (module)`
  * `src/codeintel/build/hamilton/contracts/pandera_hook.py :: (module)`
  * `src/codeintel/build/hamilton/validators/contracts.py :: (module)`
  * `src/codeintel/build/hamilton/validators/dataframe.py :: (module)`

* Operational registries derived from DataSaver tags:

  * `src/codeintel/build/hamilton/io_registry.py :: compile_write_registry(...)`
  * Additional IO registry helpers: `src/codeintel/build/hamilton/io_registry.py :: duckdb_materializations(...)` / `artifact_writes(...)`

## 2.5 CLI build composition roots (expanded Phase4 surface)

* Run/Plan/Explain/Validate:

  * `src/codeintel/cli/handlers/build.py :: build_run_handler(...)` → `_execute_build_hamilton(...)`
  * `src/codeintel/cli/handlers/build.py :: build_plan_handler(...)`
  * `src/codeintel/cli/handlers/build.py :: build_explain_handler(...)`
  * `src/codeintel/cli/handlers/build.py :: build_validate_handler(...)`
  * Commands: `src/codeintel/cli/commands/build.py :: BuildExplainCommand`, `BuildValidateCommand`

* Operational surfaces:

  * `src/codeintel/cli/handlers/build.py :: build_history_handler(...)` / command `src/codeintel/cli/commands/build.py :: BuildHistoryCommand`
  * `src/codeintel/cli/handlers/build.py :: build_graph_handler(...)` / command `src/codeintel/cli/commands/build.py :: BuildGraphCommand`
  * `src/codeintel/cli/handlers/build.py :: build_assets_handler(...)` / command `src/codeintel/cli/commands/build.py :: BuildAssetsCommand`
  * `src/codeintel/cli/handlers/build.py :: build_lineage_handler(...)` / command `src/codeintel/cli/commands/build.py :: BuildLineageCommand`
  * `src/codeintel/cli/handlers/build.py :: build_promote_handler(...)` / command `src/codeintel/cli/commands/build.py :: BuildPromoteCommand`
  * `src/codeintel/cli/handlers/build.py :: build_resolve_handler(...)` / command `src/codeintel/cli/commands/build.py :: BuildResolveCommand`
  * `src/codeintel/cli/handlers/build.py :: build_diff_handler(...)` / command `src/codeintel/cli/commands/build.py :: BuildDiffCommand`
  * `src/codeintel/cli/handlers/build.py :: build_impact_handler(...)` / command `src/codeintel/cli/commands/build.py :: BuildImpactCommand`

---

# 3) Hamilton subsystem map (deep dive)

## 3.1 Composition root: `build_driver(...)`

* Constructs a **native-only** driver and a derived `TargetGraph`:

  * Builds a base Driver from `load_native_modules()` and compiles `OutputTarget` specs from DAG tags:
    `src/codeintel/build/hamilton/driver_factory.py :: _build_base_graph(...)`
    `src/codeintel/build/hamilton/native/discovery.py :: load_native_modules()`
    `src/codeintel/build/hamilton/target_spec_compiler.py :: compile_output_targets_from_driver(...)`

  * Derives target dependencies from Hamilton graph, then produces the final `TargetGraph`:
    `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)`
    `src/codeintel/build/hamilton/introspect.py :: derive_target_dependencies(...)`
    `src/codeintel/build/hamilton/introspect.py :: target_graph_from_hamilton(...)`

* Generates a support module containing dataset/loader/artifact nodes:

  * Support module build (always uses saver-derived outputs):
    `src/codeintel/build/hamilton/driver_factory.py :: _build_support_graph_and_module(...)` →
    `src/codeintel/build/hamilton/nodes/support_factory.py :: build_support_module(...)`

* Hamilton cache integration:

  * Driver builder can be configured with cache settings:
    `src/codeintel/build/hamilton/driver_factory.py :: build_driver(..., enable_cache: bool, cache_dir: ...)`

* Runtime container:

  * Returned as `HamiltonRuntime(dr, graph, target_to_node, node_to_target)`:
    `src/codeintel/build/hamilton/runtime.py :: HamiltonRuntime`

## 3.2 “Target spec lives on the DAG”

* Canonical target anchor decorator:

  * `@codeintel_target(domain=..., target=..., spec=TargetSpecDescriptor(...))` wraps `tag_materialize(...)` and injects spec tags:

    * `target_resources`, `target_execution`, `target_parameters`, `target_spec_version`, optional `target_estimated_duration_ms`.
      `src/codeintel/build/hamilton/native/target_decorators.py :: codeintel_target(...)` / `TargetSpecDescriptor`
    * Tag keys:
      `src/codeintel/core/hamilton/tags.py :: TAG_TARGET_RESOURCES` / `TAG_TARGET_EXECUTION` / `TAG_TARGET_PARAMETERS` / `TAG_TARGET_SPEC_VERSION`

* DAG→`OutputTarget` compilation:

  * Validates graph invariants then reads:

    * anchor `domain/target/spec_version` tags and docstring summary
    * JSON-encoded resources/execution/parameters tags
    * contract outputs derived from DataSaver tags (`output_role="contract"`).
      `src/codeintel/build/hamilton/target_spec_compiler.py :: compile_output_targets_from_driver(...)`
      `_resources_from_tags(...)` / `_execution_from_tags(...)` / `_parameters_from_tags(...)`
      `src/codeintel/build/hamilton/introspect.py :: derive_target_outputs_from_savers(...)`

## 3.3 Saver nodes are first-class “DAG-visible IO boundary”

* Typed save-to decorator produces a saver metadata node (a Hamilton node) with tags used for:

  * output derivation, validation, IO registries.
    `src/codeintel/build/hamilton/save_to.py :: SaveToObjectMetadataDecorator.create_saver_node(...)`

* Saver tag identity requirements:

  * `output_role` must be static `value(...)` if present; must be `"contract"` or `"internal"`.
    `src/codeintel/build/hamilton/save_to.py :: _resolve_output_role(...)`
  * Contract savers enforce exactly one of `{table_key, artifact_name}`; artifact savers require `path_template` and validate placeholders.
    `src/codeintel/build/hamilton/save_to.py :: _resolve_output_identity(...)` / `_resolve_artifact_path_template(...)`
    `src/codeintel/build/hamilton/materializers/path_templates.py :: validate_path_template(...)`

* Saver tag schema is assembled centrally (for invariants and downstream derivation):

  * `src/codeintel/build/hamilton/save_to.py :: _build_saver_tags(...)` / `SaverTagContext`

* DataSaver tag consumers:

  * Output derivation: `src/codeintel/build/hamilton/introspect.py :: derive_target_outputs_from_savers(...)`
  * Operational registry: `src/codeintel/build/hamilton/io_registry.py :: compile_write_registry(...)`
  * TagIndex grouping: `src/codeintel/build/hamilton/tag_index.py :: TagIndex.data_saver_nodes(...)`

## 3.4 Executor + hooks/adapters

* Executor entrypoint:

  * `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor.run(...)`

* Parallel adapter:

  * Built from runtime/graph policy:
    `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._build_runtime(...)` →
    `src/codeintel/build/hamilton/adapters/parallel.py :: create_parallel_adapter(...)`

* Hooks (telemetry + contracts + lifecycle):

  * Hook builder: `src/codeintel/build/hamilton/hooks/__init__.py :: build_hooks(...)`
  * Node telemetry hook flushes records via `BuildRunWriter`:
    `src/codeintel/build/hamilton/hooks/telemetry_hook.py :: NodeTelemetryHook.flush(...)`
    `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.save_run_nodes(...)`
  * Contract enforcement hook activates per node based on node tags:
    `src/codeintel/build/hamilton/hooks/contract_hook.py :: ContractEnforcementHook.pre_node_execute(...)`

## 3.5 Validation subsystem

* Node graph validation:

  * Validates materialize anchors, dataset/artifact nodes, saver outputs (tables/artifacts/templates), and optional compute I/O purity:
    `src/codeintel/build/hamilton/validate.py :: validate_nodes(..., enforce_compute_io_purity: bool = False, ...)`

* Runtime “validate graph” helper:

  * Constructs runtime then validates it:
    `src/codeintel/build/hamilton/graph_validation.py :: validate_graph(...)`

---

# 4) Core runtime concepts & types (type map)

## 4.1 Targets, graphs, contracts

* Target metadata:

  * `src/codeintel/build/targets.py :: OutputTarget(...)`

* Dependency graph:

  * `src/codeintel/build/targets.py :: TargetGraph.register(...)` / `topological_order(...)` / `validate()`

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
  * Additional env fields impacting behavior:

    * `src/codeintel/build/hamilton/env.py :: BuildEnv.execution_settings`
    * `src/codeintel/build/hamilton/env.py :: BuildEnv.storage`
    * `src/codeintel/build/hamilton/env.py :: BuildEnv.history_options`

* Core primitives commonly embedded in BuildEnv / run context:

  * `src/codeintel/config/primitives.py :: SnapshotRef`
  * `src/codeintel/config/primitives.py :: BuildPaths`

* Run context builder:

  * `src/codeintel/build/run_context.py :: BuildRunContext`
  * Overrides bundle: `src/codeintel/build/run_context.py :: BuildRunContextOverrides`

* Hamilton execution options:

  * `src/codeintel/build/hamilton/execution_options.py :: BuildExecutionOptions`

* Build runtime settings (env-injected):

  * `src/codeintel/core/config/settings.py :: BuildSettings(...)`
  * `src/codeintel/core/config/settings.py :: HamiltonExecutionSettings(...)`

## 4.4 DAG outputs and target metadata service

* DAG output inventory artifact:

  * `src/codeintel/core/registry/dag_output_inventory.yaml`
  * `src/codeintel/core/registry/service.py :: DagOutputInventory`

* DAG-derived outputs (saver tag introspection):

  * `src/codeintel/build/hamilton/introspect.py :: derive_target_outputs_from_savers(...)`
  * Expected output helpers: `src/codeintel/build/hamilton/native/outputs.py`

* Target metadata service:

  * `src/codeintel/build/target_metadata.py :: TargetSystem`
  * `src/codeintel/build/target_metadata.py :: TargetMetadataService`
  * `src/codeintel/build/target_metadata.py :: get_target_metadata_service(...)`

## 4.5 Incremental artifacts: manifests and hash evaluation

* Persisted manifest record:

  * `src/codeintel/core/build_manifest.py :: OutputManifest`

* Hash evaluation:

  * `src/codeintel/build/hash_evaluator.py :: HashEvaluation` / `evaluate_hash_state(...)`
  * `src/codeintel/build/hash_evaluator.py :: compute_hash_evaluation(...)`

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

* Standard compute-step result (executor-style targets):

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

* **Target node output**: `TargetRunRecord` (status + hashes + row_counts + datasets/artifacts).
  `src/codeintel/core/hamilton/records.py :: TargetRunRecord`

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
  * Uses deferred column resolution (corrected path):
    `src/codeintel/build/schemas/column_resolution.py :: DeferredColumns` / `resolve_columns(...)`
  * Warehouse write path: `src/codeintel/storage/warehouse.py :: Warehouse.materialize_dataframe(...)` / `materialize_rows(...)`

* `BuildEnv.warehouse` IO boundary behavior (explicit accessor):

  * `src/codeintel/build/hamilton/env.py :: BuildEnv.warehouse(...)`

## 5.4 Artifact materialization: template-driven paths (DAG metadata)

* Artifact saver accepts `path_template` and resolves output path via template formatting:

  * `src/codeintel/build/hamilton/materializers/artifact_saver.py :: FileArtifactSaver(path_template=...)`
  * Resolution is template-first (raises if missing):
    `src/codeintel/build/hamilton/materializers/artifact_saver.py :: _resolve_artifact_path(...)`

* Template validation and formatting:

  * `src/codeintel/build/hamilton/materializers/path_templates.py :: validate_path_template(...)`
  * `src/codeintel/build/hamilton/materializers/path_templates.py :: format_path_template(...)` / `default_formatter(...)`

* Saver node tags require `artifact_path_template` for contract artifact outputs:

  * `src/codeintel/core/hamilton/tags.py :: TAG_ARTIFACT_PATH_TEMPLATE`
  * Output derivation requires it: `src/codeintel/build/hamilton/introspect.py :: derive_target_outputs_from_savers(...)`

## 5.5 Read boundaries: support nodes `d__/q__/df__/a__`

* Dataset node `d__<table_key>` extracts `DatasetRef` from producing target record:

  * `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_dataset_node_function(...)`

* Loader nodes:

  * Ibis query loader `q__*`:
    `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_query_node_function(...)` →
    `src/codeintel/build/hamilton/io/ibis_adapter.py :: load_dataset_ibis(...)`
  * DataFrame loader `df__*`:
    `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_dataframe_node_function(...)` →
    `src/codeintel/build/hamilton/io/ibis_adapter.py :: load_dataset_df(...)`

* Artifact node `a__<artifact_name>` extracts `ArtifactRef` from producing target record:

  * `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_artifact_node_function(...)`

## 5.6 Persistence surfaces (storage gateway boundary)

* Manifest and run tracking types are in `codeintel.core`:

  * `src/codeintel/core/build_manifest.py :: OutputManifest` / `BuildRunRecord`

* Build writer persists:

  * run start/complete: `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.start_run(...)` / `complete_run(...)`
  * per-target records: `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.save_run_targets(...)`
  * per-node telemetry: `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.save_run_nodes(...)`

* Contract enforcement/validation surfaces at IO boundary (present; invoked via hooks/wrappers as configured):

  * `src/codeintel/build/hamilton/contracts/enforcement.py :: (module)`
  * `src/codeintel/build/hamilton/contracts/enforced_gateway.py :: (module)`
  * `src/codeintel/build/hamilton/contracts/pandera_hook.py :: (module)`
  * `src/codeintel/build/hamilton/validators/dataframe.py :: (module)`

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

  * `src/codeintel/cli/handlers/build.py :: build_run_handler(...)` (via `get_target_metadata_service().system.graph`)

## 6.3 Planning (“build plan”): compute vs skip vs blocked vs missing

* Plan builds a Hamilton runtime and reads the target graph:

  * `src/codeintel/build/hamilton/planner.py :: compute_plan(...)` (calls `build_driver()`)

* Hash evaluation uses current input hash / options hash vs stored manifest:

  * `src/codeintel/build/hash_evaluator.py :: compute_hash_evaluation(...)`
  * `src/codeintel/build/hash_evaluator.py :: evaluate_hash_state(...)`
  * `src/codeintel/core/build_manifest.py :: OutputManifest`

* Plan statuses and reasons are encoded on `PlanEntry`:

  * `src/codeintel/build/hamilton/planner.py :: PlanEntry(...)`

## 6.4 Execution (“build run”): Hamilton driver execute

* CLI constructs `BuildEnv` and runs the executor:

  * `src/codeintel/cli/handlers/build.py :: _execute_build_hamilton(...)` → `BuildRunContext.build_env(...)` → `HamiltonBuildExecutor.run(...)`
  * `src/codeintel/build/run_context.py :: BuildRunContext.build_env(...)`
  * `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor.run(...)`

* **Corrected execution inputs**: executor runs the DAG with **both** `env` and `graph`:

  * `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._execute_dag(...)` (uses `inputs={"env": execution_env, "graph": graph}`)
  * Node mapping: `src/codeintel/build/hamilton/driver_factory.py :: target_to_node_name(...)`

## 6.5 Runtime incremental behavior (native executor and manifest skip)

* Many native targets use `NativeTargetExecutor` to unify:

  * input hash computation, skip decision, record creation, manifest persistence.
    `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor`

* Skip check:

  * `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor.should_skip(...)` →
    `src/codeintel/build/hamilton/run_record_utils.py :: should_skip_native_target(...)`

* Manifest persistence on success:

  * `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor.execute(...)` calls `save_manifest(...)`.
    `src/codeintel/build/hamilton/run_records.py :: save_manifest(...)`

## 6.6 DAG-derived IO registry and surface introspection

* IO registry (writes grouped by sink) is computed from DataSaver tags:

  * `src/codeintel/build/hamilton/io_registry.py :: compile_write_registry(...)`

* Per-target IO surface (reads/writes) derived from loader/saver tags:

  * `src/codeintel/build/hamilton/introspect.py :: derive_target_io_surface(...)`
  * CLI build explain can include this:

    * `src/codeintel/cli/handlers/build.py :: build_explain_handler(...)`
    * `src/codeintel/cli/commands/build.py :: BuildExplainCommand`

## 6.7 Options loading and runtime normalization (plan/execution alignment)

* Canonical per-target option loading:

  * `src/codeintel/build/hamilton/options_loading.py :: load_target_options(...)`

* Graph runtime option loading:

  * `src/codeintel/build/hamilton/graph_runtime_options.py :: load_graph_runtime_options(...)`

## 6.8 Additional build orchestration surfaces (CLI)

*(These are build-facing orchestration paths beyond run/plan/explain/validate.)*

* `src/codeintel/cli/handlers/build.py :: build_history_handler(...)`
* `src/codeintel/cli/handlers/build.py :: build_graph_handler(...)` (uses observability exporters)
* `src/codeintel/cli/handlers/build.py :: build_assets_handler(...)`
* `src/codeintel/cli/handlers/build.py :: build_lineage_handler(...)`
* `src/codeintel/cli/handlers/build.py :: build_promote_handler(...)`
* `src/codeintel/cli/handlers/build.py :: build_resolve_handler(...)`
* `src/codeintel/cli/handlers/build.py :: build_diff_handler(...)`
* `src/codeintel/cli/handlers/build.py :: build_impact_handler(...)`

---

# 7) Walkthrough: “request one target” end-to-end trace

> Representative target: **`scip`** (ingestion domain).
> `src/codeintel/build/hamilton/native/ingestion/scip.py :: SCIP_TARGET_NAME`, `t__scip(...)`

## 7.1 CLI to executor

1. **CLI resolves the target graph and goals**:

* `src/codeintel/cli/handlers/build.py :: build_run_handler(...)` uses `get_target_metadata_service().system.graph`
* `src/codeintel/cli/handlers/build.py :: _resolve_goals(...)`

2. **CLI opens a gateway, loads manifests, and constructs `BuildEnv`**:

* `src/codeintel/cli/handlers/build.py :: _execute_build_hamilton(...)` (manifest index creation)
* `src/codeintel/build/run_context.py :: BuildRunContext.from_execution_context(...)`
* `src/codeintel/build/run_context.py :: BuildRunContext.build_env(...)` → `src/codeintel/build/hamilton/env.py :: BuildEnv(...)`

3. **CLI runs Hamilton executor**:

* `src/codeintel/cli/handlers/build.py :: _execute_build_hamilton(...)` → `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor.run(...)`

## 7.2 Executor closure and DAG execution

4. **Executor constructs a Hamilton runtime**:

* `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._build_runtime(...)` → `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)`

5. **Executor computes dependency closure and maps to `t__` nodes**:

* `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._compute_closure(...)`
* `src/codeintel/build/targets.py :: TargetGraph.topological_order(...)`
* `src/codeintel/build/hamilton/executor.py :: _map_closure_to_nodes(...)` → `src/codeintel/build/hamilton/driver_factory.py :: target_to_node_name(...)`

6. **Driver executes final vars (corrected inputs)**:

* `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._execute_dag(...)` executes driver with `inputs={"env": execution_env, "graph": graph}`

## 7.3 `scip` native module: key nodes

7. **Tool step** (tagged as tool):

* `src/codeintel/build/hamilton/native/ingestion/scip.py :: t__scip__run(env, graph, t__modules) -> ScipRunResult`
* Skip check uses `NativeTargetExecutor`:

  * `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor.should_skip(...)`

8. **Artifact compute nodes + saver metadata nodes**:

* `src/codeintel/build/hamilton/native/ingestion/scip.py :: scip__index_artifact(...) -> Path | None`
* `src/codeintel/build/hamilton/native/ingestion/scip.py :: scip__json_artifact(...) -> Path | None`
* Decorated with `SaveToObjectMetadataDecorator([FileArtifactSaver], ...)`:

  * `src/codeintel/build/hamilton/save_to.py :: SaveToObjectMetadataDecorator`
  * `src/codeintel/build/hamilton/materializers/artifact_saver.py :: FileArtifactSaver`

9. **Row materialization nodes**:

* `src/codeintel/build/hamilton/native/ingestion/scip.py :: scip__symbol_rows(...)`
* `src/codeintel/build/hamilton/native/ingestion/scip.py :: scip__occurrence_rows(...)`
* Decorated with `SaveToObjectMetadataDecorator([DuckDBRowsSaver], ...)`:

  * `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py :: DuckDBRowsSaver`

10. **Target anchor node** (`t__scip`) produces `TargetRunRecord`:

* `src/codeintel/build/hamilton/native/ingestion/scip.py :: t__scip(...)`
* Decorator: `src/codeintel/build/hamilton/native/target_decorators.py :: codeintel_target(...)`
* Record builder: `src/codeintel/build/hamilton/native/materialization_records.py :: record_from_file_artifact_materializations(...)`

## 7.4 Post-run persistence

11. **Executor categorizes computed/skipped/failed**:

* `src/codeintel/build/hamilton/executor.py :: _categorize_outputs(...)`

12. **Build run writer persists run + target records (+ telemetry if enabled)**:

* `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.start_run(...)` / `save_run_targets(...)` / `complete_run(...)`

---

# 8) Extension mechanics (how new behavior is introduced today)

## 8.1 Add a new build target (DAG-native spec anchor)

* Create a new native module under `src/codeintel/build/hamilton/native/<domain>/...`:

  * `src/codeintel/build/hamilton/native/discovery.py :: native_module_paths(...)` / `load_native_modules()`

* Define a target anchor function `t__<target>` and decorate with:

  * `src/codeintel/build/hamilton/native/target_decorators.py :: codeintel_target(domain=..., target=..., spec=...)`

* Ensure the target anchor has a docstring summary:

  * `src/codeintel/build/hamilton/target_spec_compiler.py :: _node_docstring(...)` / `_summary(...)`

* (Conventional alignment) load target options via the canonical loader:

  * `src/codeintel/build/hamilton/options_loading.py :: load_target_options(...)`
  * `src/codeintel/build/hamilton/graph_runtime_options.py :: load_graph_runtime_options(...)`

## 8.2 Add table outputs (DuckDB)

* Add compute nodes returning:

  * ibis table expressions (`DuckDBIbisTableSaver`) or row tuples (`DuckDBRowsSaver`)
  * tagging: `src/codeintel/build/hamilton/tagging.py :: tag_compute(...)`

* Attach a saver metadata node with `SaveToObjectMetadataDecorator(...)` (static identity via `value(...)`):

  * `src/codeintel/build/hamilton/save_to.py :: SaveToObjectMetadataDecorator`

* Table schema must exist in the registry:

  * `src/codeintel/build/hamilton/target_spec_compiler.py :: _resolve_table_schemas(...)` → `src/codeintel/core/schemas/table_registry.py :: get_table_schema(...)`

## 8.3 Add artifact outputs (filesystem)

* Add a compute node returning `Path | None` (or other supported payload types for `FileArtifactSaver`), tagged as compute.
* Attach a `FileArtifactSaver` via `SaveToObjectMetadataDecorator` with:

  * `target_name=value(<target>)`, `artifact_name=value(<artifact>)`, `path_template=value(<template>)`
  * Template validation: `src/codeintel/build/hamilton/materializers/path_templates.py :: validate_path_template(...)`

## 8.4 Control what counts as a “contract output” vs “internal output”

* Saver metadata nodes use `output_role`:

  * Enforcement: `src/codeintel/build/hamilton/save_to.py :: _resolve_output_role(...)`
  * Derivation filter: `src/codeintel/build/hamilton/introspect.py :: _require_output_role(...)`

## 8.5 Add support-node surfaces for new outputs

* Support nodes are generated from:

  * contracts (`OutputTarget.contract`) OR derived outputs (`DerivedTargetOutputs`)
  * `src/codeintel/build/hamilton/driver_factory.py :: _build_support_graph_and_module(...)`
  * `src/codeintel/build/hamilton/nodes/support_factory.py :: build_support_module(..., derived_outputs=...)`

---

# 9) Conventions & invariants (observed)

## 9.1 Naming conventions (stable node identity)

* Stable prefixes:

  * `t__*`: `src/codeintel/build/hamilton/naming.py :: target_node(...)`
  * `m__*`: `src/codeintel/build/hamilton/naming.py :: materialize_node(...)`
  * `d__*`: `src/codeintel/build/hamilton/naming.py :: dataset_node(...)`
  * `q__*`, `df__*`: `src/codeintel/build/hamilton/naming.py :: query_node(...)` / `dataframe_node(...)`
  * `a__*`: `src/codeintel/build/hamilton/naming.py :: artifact_node(...)`

## 9.2 Tag conventions (canonical keys and node types)

* Canonical tags and node types:

  * `src/codeintel/core/hamilton/tags.py :: TAG_DOMAIN` / `TAG_TARGET` / `TAG_TABLE_KEY` / `TAG_ARTIFACT` / `TAG_NODE_TYPE`
  * `src/codeintel/core/hamilton/tags.py :: NODE_TYPE_MATERIALIZE` / `NODE_TYPE_COMPUTE` / `NODE_TYPE_DATASET` / `NODE_TYPE_ARTIFACT` / loader kinds

## 9.3 Target anchor invariants (validated)

* Materialize anchor nodes must have `domain`, `target`, and `target_spec_version == "1"`:

  * `src/codeintel/build/hamilton/validate.py :: _collect_materialize_index(...)`

* DAG validation runnable via:

  * `src/codeintel/build/hamilton/graph_validation.py :: validate_graph(...)`

## 9.4 Saver node invariants (contract outputs are tag-derivable)

* DataSaver nodes identified via `hamilton.data_saver == True` tags:

  * Tag construction: `src/codeintel/build/hamilton/save_to.py :: _build_saver_tags(...)` / `SaverTagContext`
  * Tag indexing: `src/codeintel/build/hamilton/tag_index.py :: TagIndex.data_saver_nodes(...)`

* Contract outputs must be self-identifying:

  * `output_role` must be `"contract"` or `"internal"`: `src/codeintel/build/hamilton/introspect.py :: _require_output_role(...)`
  * Exactly one of `{table_key, artifact}`: `src/codeintel/build/hamilton/introspect.py :: _resolve_output_identity(...)`
  * Contract artifact savers require `artifact_path_template`: `src/codeintel/build/hamilton/introspect.py :: _iter_contract_saver_tags(...)`

## 9.5 DAG output drift handling

* Contract vs DAG outputs are compared directly from saver tags:

  * `src/codeintel/build/hamilton/contracts/check_target_contracts.py :: main(...)`
  * `src/codeintel/build/hamilton/introspect.py :: derive_target_outputs_from_savers(...)`

---

# 10) Glossary (project-specific vocabulary)

* **Target anchor**: a `t__<name>` node tagged `node_type="materialize"` and decorated with `@codeintel_target(...)`.
  `src/codeintel/build/hamilton/native/target_decorators.py :: codeintel_target(...)`
  `src/codeintel/core/hamilton/tags.py :: NODE_TYPE_MATERIALIZE`

* **DAG-native target spec**: `OutputTarget` compiled from node tags + docstring + saver-derived outputs.
  `src/codeintel/build/hamilton/target_spec_compiler.py :: compile_output_targets_from_driver(...)`

* **DataSaver node**: Hamilton node created by `SaveToObjectMetadataDecorator` that performs persistence and returns `MaterializationMetadata`; tagged with saver identity tags.
  `src/codeintel/build/hamilton/save_to.py :: SaveToObjectMetadataDecorator.create_saver_node(...)`

* **output_role**: saver tag classifying outputs as `"contract"` vs `"internal"`.
  `src/codeintel/build/hamilton/save_to.py :: _resolve_output_role(...)`

* **DAG output inventory**: inventory artifact for target outputs and contracts.
  `src/codeintel/core/registry/dag_output_inventory.yaml`
  `src/codeintel/core/registry/service.py :: DagOutputInventory`

* **DerivedTargetOutputs**: saver-tag-derived datasets/artifacts/templates per target.
  `src/codeintel/build/hamilton/introspect.py :: DerivedTargetOutputs`

* **Support nodes**: generated `d__*`, `q__*`, `df__*`, `a__*` nodes for data/artifact access.
  `src/codeintel/build/hamilton/nodes/support_factory.py :: build_support_module(...)`

* **Manifest**: `OutputManifest` persisted per target used for skip/planning.
  `src/codeintel/core/build_manifest.py :: OutputManifest`

* **NativeTargetExecutor**: native-target helper centralizing hash/skip/record/manifest logic.
  `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor`

* **Graph validation**: deterministic DAG invariant checks (anchors/output tags/templates; optional compute IO purity).
  `src/codeintel/build/hamilton/validate.py :: validate_nodes(...)`
  `src/codeintel/build/hamilton/graph_validation.py :: validate_graph(...)`

* **BuildPaths / SnapshotRef**: core primitives used to define snapshot identity + filesystem layout for a run.
  `src/codeintel/config/primitives.py :: BuildPaths` / `SnapshotRef`

* **BuildSpec / Schema manifest / Serving snapshot / Exports**: build subsystems for contractable surfaces and downstream consumption.
  `src/codeintel/build/spec/compile.py :: compile_buildspec(...)`
  `src/codeintel/build/schemas/manifest.py :: (module)`
  `src/codeintel/build/serving/publisher.py :: publish_serving_snapshot(...)`
  `src/codeintel/build/exports/runner.py :: ExportRunner`
