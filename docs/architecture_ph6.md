

# 1) Executive architecture summary

* **Top-level build control plane is CLI-driven, delegating to Hamilton execution.**
  `src/codeintel/cli/handlers/build.py :: build_run_handler(ctx: CommandContext) -> CliResult[BuildRunResult]` calls `_build_run_result(...)` and (for execution) `_execute_build_hamilton(...)`.
  `_execute_build_hamilton(...)` constructs a `BuildRunContext` and invokes `HamiltonBuildExecutor.run(...)`.
  Evidence: `src/codeintel/cli/handlers/build.py :: _execute_build_hamilton(...)`, `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor.run(...)`.

* **Hamilton is composed in a “native-only” mode with a generated support module.**
  Native modules are discovered by filesystem scanning (`*.py`) under `src/codeintel/build/hamilton/native/{ingestion,graphs,analytics,export}` and imported as modules.
  Support nodes (dataset refs, loaders, artifact refs) are generated into a dynamic module and included in the Driver.
  Evidence: `src/codeintel/build/hamilton/native/discovery.py :: native_module_paths()`, `src/codeintel/build/hamilton/native/discovery.py :: load_native_modules()`, `src/codeintel/build/hamilton/driver_factory.py :: _build_support_graph_and_module(...)`, `src/codeintel/build/hamilton/nodes/support_factory.py :: build_support_module(...)`.

* **Targets are represented twice: as “build targets” in `TargetGraph`, and as Hamilton nodes named `t__*`.**
  `TargetGraph` is populated by compiling `t__*` anchors with required tags (node_type “materialize” + spec tags) into `OutputTarget` objects.
  Runtime also carries mappings `{target_name → t__node_name}`.
  Evidence: `src/codeintel/build/targets.py :: TargetGraph`, `src/codeintel/build/hamilton/target_spec_compiler.py :: compile_output_targets_from_driver(...)`, `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)` (builds `t2n`/`n2t`), `src/codeintel/build/hamilton/runtime.py :: HamiltonRuntime`.

* **Incrementality is “manifest + input_hash (+ options_hash)” driven, reused across planning, native tool gating, and materializers.**
  Input hash includes engine version + repo/commit + target name + dependency manifest hashes + optional file-state and options hash.
  Skip checks compare computed hashes to stored `OutputManifest` for the same repo/commit/target.
  Evidence: `src/codeintel/build/hashing.py :: compute_input_hash_with_deps(...)`, `src/codeintel/build/hash_evaluator.py :: evaluate_hash_state(...)`, `src/codeintel/build/hamilton/run_record_utils.py :: should_skip_native_target(...)`, `src/codeintel/core/build_manifest.py :: OutputManifest` (type), `src/codeintel/build/hamilton/run_records.py :: save_manifest(...)`.

* **IO boundaries are explicit Hamilton saver nodes (`m__*`) that write to DuckDB or to filesystem artifacts.**
  Row outputs are persisted by `DuckDBRowsSaver.save_data(...)` and return `MaterializationResult` records; these are then summarized into a `TargetRunRecord` by finalization helpers.
  Evidence: `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py :: DuckDBRowsSaver.save_data(...)`, `src/codeintel/build/hamilton/native/materialization_records.py :: record_from_materializations(...)`, `src/codeintel/build/hamilton/boundary_types.py :: MaterializationResult`.

* **Run-level persistence captures both “run lifecycle” and “per-target records”, best-effort.**
  The executor writes `BuildRunRecord` start/complete, persists per-target `TargetRunRecord` list, and emits an asset catalog.
  Evidence: `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.start_run(...)`, `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.complete_run(...)`, `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.save_run_targets(...)`, `src/codeintel/build/hamilton/executor.py :: _finalize_run(...)`.


# 2) Repository map (build-focused)

## 2.1 Build package top-level: `src/codeintel/build/*` (Phase 1, carried forward)

* Package facade + lazy exports
  `src/codeintel/build/__init__.py :: __all__`
  `src/codeintel/build/__init__.py :: __getattr__(name: str) -> object`

* Target model + dependency graph
  `src/codeintel/build/targets.py :: OutputTarget`
  `src/codeintel/build/targets.py :: TargetGraph`

* Output contracts (tables + artifacts)
  `src/codeintel/build/contracts.py :: OutputContract`
  `src/codeintel/build/contracts.py :: ArtifactSpec`

* Resource/execution “hints” attached to targets
  `src/codeintel/build/resources.py :: TargetResources`
  `src/codeintel/build/resources.py :: TargetExecution`

* Parameter container and config system
  `src/codeintel/build/parameters.py :: TargetParameters`
  `src/codeintel/build/config.py :: BuildConfig / load_build_config(...)`

* Hashing + session cache + state types
  `src/codeintel/build/hashing.py :: compute_input_hash(...)`
  `src/codeintel/build/session.py :: BuildSession`
  `src/codeintel/build/state_types.py :: TargetState / BuildState`

* Execution policy and errors
  `src/codeintel/build/execution_policy.py :: ExecutionPolicy`
  `src/codeintel/build/errors.py :: BuildError / BuildErrorCollection`

* Build run context surface (bridges into Hamilton env/runtime)
  `src/codeintel/build/run_context.py :: BuildRunContext`

## 2.2 Hamilton subtree expansion (Phase 2)

### 2.2.1 Hamilton package entrypoints / orchestration helpers

* Facade exports: `src/codeintel/build/hamilton/__init__.py :: __all__ / __getattr__(...)`
* Driver/runtime:

  * `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)`
  * `src/codeintel/build/hamilton/runtime.py :: HamiltonRuntime`
* Planning:

  * `src/codeintel/build/hamilton/planner.py :: compute_plan(...)`
  * `src/codeintel/build/hamilton/planner.py :: explain_plan(...)`
* Execution:

  * `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor`
  * `src/codeintel/build/hamilton/executor.py :: HamiltonBuildResult`
* Observability exports:

  * `src/codeintel/build/hamilton/observability.py :: export_dag_json(...) / export_execution_json(...) / export_dag_mermaid(...)`

### 2.2.2 Tagging + naming + validation (Hamilton graph hygiene)

* Naming: `src/codeintel/build/hamilton/naming.py :: target_node(...) / dataset_node(...) / materialize_node(...) / to_node_name(...)`
* Typed tag parsing/validation: `src/codeintel/build/hamilton/tag_spec.py :: TagSpec / validate_tag_spec(...)`
* Graph validation utility: `src/codeintel/build/hamilton/validate.py :: validate_nodes(...)`

### 2.2.3 Target compilation + introspection

* Target compilation from Hamilton nodes:
  `src/codeintel/build/hamilton/target_spec_compiler.py :: compile_output_targets_from_driver(...)`
* Runtime introspection (deps, IO surface, outputs):
  `src/codeintel/build/hamilton/introspect.py :: derive_target_dependencies(...)`
  `src/codeintel/build/hamilton/introspect.py :: derive_target_outputs_from_savers(...)`
  `src/codeintel/build/hamilton/introspect.py :: derive_target_io_surface(...)`
  `src/codeintel/build/hamilton/introspect.py :: target_graph_from_hamilton(...)`

### 2.2.4 Support node generation (auto nodes for datasets/loaders/artifacts)

* `src/codeintel/build/hamilton/nodes/support_factory.py :: build_support_module(...)`
* `src/codeintel/build/hamilton/nodes/module_attach.py :: tagged_attach_node(...)`
* `src/codeintel/build/hamilton/nodes/mappings.py :: SupportNodeMappings`

### 2.2.5 IO adapters + materializers

* IO refs:
  `src/codeintel/build/hamilton/io/dataset_ref.py :: DatasetRef`
  `src/codeintel/build/hamilton/io/artifact_ref.py :: ArtifactRef`
* IO adapter wrappers (delegating to storage IO):
  `src/codeintel/build/hamilton/io/ibis_adapter.py :: save_dataframe(...) / load_dataset_df(...)`
* Materializers / savers:

  * `src/codeintel/build/hamilton/materializers/duckdb_saver.py :: DuckDBIbisTableSaver`
  * `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py :: DuckDBRowsSaver`
  * `src/codeintel/build/hamilton/materializers/artifact_saver.py :: FileArtifactSaver`
  * Materialization results: `src/codeintel/core/execution/materialization.py :: MaterializationResult`

### 2.2.6 Hooks/adapters, run records, and manifests

* Hooks assembly: `src/codeintel/build/hamilton/hooks/__init__.py :: build_hooks(...) / HookOptions`
* Telemetry hook: `src/codeintel/build/hamilton/hooks/telemetry_hook.py :: NodeTelemetryHook`
* Contract hook: `src/codeintel/build/hamilton/hooks/contract_hook.py :: ContractEnforcementHook`
* Run writing: `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter`
* Run records / manifest persistence:

  * `src/codeintel/build/hamilton/run_records.py :: create_run_record(...) / save_manifest(...)`
  * Skip/manifest service: `src/codeintel/build/hamilton/run_record_utils.py :: BuildManifestService / should_skip(...)`

### 2.2.7 Native modules layout

* Discovery: `src/codeintel/build/hamilton/native/discovery.py :: native_module_paths() / load_native_modules()`
* Native modules (examples of inventory; content varies by module file):

  * Analytics: `src/codeintel/build/hamilton/native/analytics/*`
  * Graphs: `src/codeintel/build/hamilton/native/graphs/*`
  * Ingestion: `src/codeintel/build/hamilton/native/ingestion/*`
  * Export: `src/codeintel/build/hamilton/native/export/*`
  * Options: `src/codeintel/build/hamilton/native/options/*`
  * Patterns/templates used by native targets: `src/codeintel/build/hamilton/native/patterns/*`
  * Target decorator surface: `src/codeintel/build/hamilton/native/target_decorators.py :: codeintel_target(...)`

## 2.3 Composition roots that invoke build (Phase 1, carried forward)

* CLI entrypoint registration
  `pyproject.toml :: [project.scripts] codeintel = "codeintel.cli:main"`

* Build command wiring
  `src/codeintel/cli/commands/build.py :: BuildRunCommand`
  `src/codeintel/cli/commands/build.py :: @cli_command("build.run", handler=build_run_handler, ...)`

# 3) Hamilton subsystem map (deep dive)

## 3.1 Canonical tag keys + node-type taxonomy

* Canonical tag keys and node type string values live in:
  `src/codeintel/core/hamilton/tags.py :: TAG_DOMAIN / TAG_TARGET / TAG_TABLE_KEY / TAG_NODE_TYPE / ...`
  `src/codeintel/core/hamilton/tags.py :: NODE_TYPE_MATERIALIZE / NODE_TYPE_DATASET / NODE_TYPE_LOADER_QUERY / ...`
* Typed tag interpretation (build-layer) wraps the canonical tags:
  `src/codeintel/build/hamilton/tag_spec.py :: TagSpec`
  `src/codeintel/build/hamilton/tag_spec.py :: validate_tag_spec(tags: Mapping[str, object]) -> list[str]`
  `src/codeintel/build/hamilton/tag_spec.py :: tag_spec_from_tags(...) -> TagSpec`

## 3.2 Stable naming conventions for DAG-visible nodes

* Node name canonicalization:
  `src/codeintel/build/hamilton/naming.py :: to_node_name(logical_id: str, *, prefix: str) -> str`
  Example shown in docstring: `to_node_name("analytics.function_metrics", prefix="t") -> "t__analytics__function_metrics"`.
* Role-specific naming helpers (all return `str` node names):

  * Targets (logical target name → `t__*`): `src/codeintel/build/hamilton/naming.py :: target_node(target_name: str) -> str`
  * Datasets (table key → `d__*`): `src/codeintel/build/hamilton/naming.py :: dataset_node(dataset_key: str) -> str`
  * Loaders (table key → `q__*` / `df__*`): `src/codeintel/build/hamilton/naming.py :: query_node(...) / dataframe_node(...)`
  * Materializers and artifacts: `src/codeintel/build/hamilton/naming.py :: materialize_node(...) / artifact_node(...) / path_node(...)`
* Reverse mapping helper: `src/codeintel/build/hamilton/naming.py :: node_to_target(node_name: str) -> str | None` (string parsing; exact rules in code).

## 3.3 Native module discovery and module list construction

* Native module enumeration is filesystem-driven:
  `src/codeintel/build/hamilton/native/discovery.py :: native_module_paths() -> list[Path]`
* Native module import/instantiation for Hamilton:

  * `src/codeintel/build/hamilton/native/discovery.py :: load_native_modules() -> list[ModuleType]`
  * Uses the paths returned by `native_module_paths()` to load modules (import mechanism is in-file).

## 3.4 Driver construction: base graph, support module, adapters, cache

### 3.4.1 Base driver and base TargetGraph from native modules

* Base driver build:

  * `src/codeintel/build/hamilton/driver_factory.py :: _build_base_graph(config: dict[str, Any] | None) -> tuple[TargetGraph, h_driver.Driver]`
  * Constructs driver from `load_native_modules()` and config:
    `Builder().with_config(...).with_modules(*native_mods).allow_module_overrides().build()` (builder chain appears in `_build_base_graph`).
* OutputTarget compilation from the Hamilton driver:

  * `src/codeintel/build/hamilton/driver_factory.py :: _build_base_graph(...)` calls
    `src/codeintel/build/hamilton/target_spec_compiler.py :: compile_output_targets_from_driver(driver, strict=True)`
  * Compiled targets are registered into `TargetGraph`:
    `src/codeintel/build/hamilton/driver_factory.py :: TargetGraph.register(...)` within `_build_base_graph`.

### 3.4.2 Support module generation and “full graph” build

* Support module build is derived from native graph + derived output surfaces:

  * `src/codeintel/build/hamilton/driver_factory.py :: _build_support_graph_and_module(...)`

    * Builds a native runtime: `src/codeintel/build/hamilton/runtime.py :: HamiltonRuntime(dr=native_driver, graph=base_graph)`
    * Derives deps: `src/codeintel/build/hamilton/introspect.py :: derive_target_dependencies(runtime)`
    * Builds an enriched graph: `src/codeintel/build/hamilton/introspect.py :: target_graph_from_hamilton(...)`
    * Derives saver outputs: `src/codeintel/build/hamilton/introspect.py :: derive_target_outputs_from_savers(runtime)`
    * Generates a support module:

      * `src/codeintel/build/hamilton/nodes/support_factory.py :: build_support_module(options=SupportGenerationOptions(...), graph=native_graph, outputs=derived_outputs, strict=...)`
* Final driver build includes native modules + support module:

  * `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...) -> HamiltonRuntime` constructs a `Builder()` with modules `(*native_mods, support_module)` and adapters list, and then `.build()`.

### 3.4.3 Adapters (hooks) and cache toggles

* `build_driver` accepts adapters directly or via a factory:

  * `src/codeintel/build/hamilton/driver_factory.py :: build_driver(*, adapters: Sequence[LifecycleAdapter] | None = None, adapter_factory: Callable[[TargetGraph], Sequence[LifecycleAdapter]] | None = None, ...)`
* Cache toggle is explicit and uses Hamilton’s builder cache API:

  * `src/codeintel/build/hamilton/driver_factory.py :: build_driver(..., enable_cache: bool = False, cache_dir: str | Path | None = None, ...)`
  * When enabled: `builder.with_cache(path=..., default_behavior="disable", default_loader_behavior="disable", default_saver_behavior="disable")` (exact call in `build_driver`).

## 3.5 Target compilation from Hamilton nodes (tags → OutputTarget)

* Primary compiler entrypoint:
  `src/codeintel/build/hamilton/target_spec_compiler.py :: compile_output_targets_from_driver(driver: h_driver.Driver, *, strict: bool = True, overrides: Sequence[TargetSpecOverride] | None = None) -> tuple[OutputTarget, ...]`
* Anchor-node detection for targets is tag-driven:

  * `src/codeintel/build/hamilton/target_spec_compiler.py :: _is_target_anchor(node: Node) -> bool`
  * Condition includes: `tags.get(ht.TAG_NODE_TYPE) == ht.NODE_TYPE_MATERIALIZE` and presence of string tags for target and domain. (Uses `src/codeintel/core/hamilton/tags.py` constants via `ht` import.)
* Per-target fields are parsed from tags:

  * Resources: `src/codeintel/build/hamilton/target_spec_compiler.py :: _resources_from_tags(...) -> TargetResources`
  * Execution hints: `src/codeintel/build/hamilton/target_spec_compiler.py :: _execution_from_tags(...) -> TargetExecution`
  * Parameters: `src/codeintel/build/hamilton/target_spec_compiler.py :: _parameters_from_tags(...) -> TargetParameters`
  * Target “spec version” validation: `src/codeintel/build/hamilton/target_spec_compiler.py :: _validate_spec_version(...)`
* OutputContract resolution used during compilation:

  * Table schemas: `src/codeintel/build/hamilton/target_spec_compiler.py :: _resolve_table_schemas(...)`
  * Artifact specs: `src/codeintel/build/hamilton/target_spec_compiler.py :: _artifact_specs(...)`
  * OutputTarget assembly: `src/codeintel/build/hamilton/target_spec_compiler.py :: _build_output_target(...) -> OutputTarget`

## 3.6 Support node generation (dataset refs, loaders, artifacts) as a derived module

* Support module factory:

  * `src/codeintel/build/hamilton/nodes/support_factory.py :: build_support_module(graph: TargetGraph, outputs: DerivedTargetOutputs, *, options: SupportGenerationOptions, strict: bool) -> ModuleType`
* SupportGenerationOptions controls which kinds of support nodes are created:

  * `src/codeintel/build/hamilton/nodes/support_factory.py :: SupportGenerationOptions` (dataclass)
* Support node creation is function-factory based:

  * DatasetRef nodes (`d__*`): `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_dataset_node_function(...)`
  * Query loader nodes (`q__*`): `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_query_node_function(...)`
  * DataFrame loader nodes (`df__*`): `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_dataframe_node_function(...)`
  * ArtifactRef nodes + path nodes: `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_artifact_node_function(...) / _create_artifact_path_node_function(...)`
* Attaching nodes to a dynamic module with tags:

  * `src/codeintel/build/hamilton/nodes/module_attach.py :: tagged_attach_node(module: ModuleType, *, name: str, fn: Callable[..., object], tags: Mapping[str, object]) -> None`

## 3.7 Runtime introspection: deriving deps and IO surfaces from the Hamilton graph

* Target list and dependency derivation:

  * `src/codeintel/build/hamilton/introspect.py :: target_names_from_runtime(runtime: HamiltonRuntime) -> frozenset[str]`
  * `src/codeintel/build/hamilton/introspect.py :: derive_target_dependencies(runtime: HamiltonRuntime) -> dict[str, tuple[str, ...]]`
* Derived outputs from saver tags (table writes / artifact writes):

  * `src/codeintel/build/hamilton/introspect.py :: derive_target_outputs_from_savers(runtime: HamiltonRuntime) -> DerivedTargetOutputs`
  * IO surface model types:

    * `src/codeintel/build/hamilton/introspect.py :: TableWrite`
    * `src/codeintel/build/hamilton/introspect.py :: ArtifactWrite`
    * `src/codeintel/build/hamilton/introspect.py :: TargetIOSurface`
    * `src/codeintel/build/hamilton/introspect.py :: derive_target_io_surface(runtime: HamiltonRuntime, ...) -> dict[str, TargetIOSurface]`
* Building a TargetGraph “as seen from Hamilton” (with derived deps attached):

  * `src/codeintel/build/hamilton/introspect.py :: target_graph_from_hamilton(runtime: HamiltonRuntime, *, base_graph: TargetGraph, derived_deps: Mapping[str, tuple[str, ...]], strict: bool) -> TargetGraph`

## 3.8 Planning (dry-run) subsystem

* Plan entry model:

  * `src/codeintel/build/hamilton/planner.py :: PlanEntry` (dataclass)
  * Field inventory excerpt: `target`, `node`, `module`, `status`, `reason`, `input_hash`, `options_hash`, `prior_input_hash`, `dependencies`, `table_keys`, `artifact_keys`, `dep_hashes`, `prior_dep_hashes`, `impl_kind`.
* Plan computation:

  * `src/codeintel/build/hamilton/planner.py :: compute_plan(env: BuildEnv, runtime: HamiltonRuntime, *, selection: Sequence[str] | None = None, allow_missing: bool = False, force: frozenset[str] = frozenset()) -> HamiltonBuildPlan`
* Plan explanation formatting:

  * `src/codeintel/build/hamilton/planner.py :: explain_plan(plan: HamiltonBuildPlan, *, show_hashes: bool = False, show_deps: bool = False, show_outputs: bool = True) -> str`

## 3.9 Execution subsystem (HamiltonBuildExecutor)

* Executor orchestrator:

  * `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor`
  * Result container:

    * `src/codeintel/build/hamilton/executor.py :: HamiltonBuildResult` (dataclass)
* Executor-internal phases are encoded as helper functions and private dataclasses (inventory examples):

  * Preflight: `src/codeintel/build/hamilton/executor.py :: _preflight_missing_inputs(...) / _apply_preflight(...)`
  * Closure mapping: `src/codeintel/build/hamilton/executor.py :: _map_closure_to_nodes(...)`
  * Failure record synthesis: `src/codeintel/build/hamilton/executor.py :: _ensure_failure_records(...)`
  * Finalization: `src/codeintel/build/hamilton/executor.py :: _finalize_run(...)`

## 3.10 Hooks/adapters subsystem

* Hook options + assembly:

  * `src/codeintel/build/hamilton/hooks/__init__.py :: HookOptions`
  * `src/codeintel/build/hamilton/hooks/__init__.py :: build_hooks(run_id: str, writer: BuildRunWriter, graph: TargetGraph, *, options: HookOptions | None = None) -> Sequence[LifecycleAdapter]`
* Telemetry hook:

  * `src/codeintel/build/hamilton/hooks/telemetry_hook.py :: NodeTelemetryHook` (Hamilton lifecycle adapter base classes appear in class inheritance)
* Contract enforcement hook:

  * `src/codeintel/build/hamilton/hooks/contract_hook.py :: ContractEnforcementHook`
  * Captures “validation results” types:

    * `src/codeintel/build/hamilton/hooks/contract_hook.py :: ValidationResult`
    * `src/codeintel/build/hamilton/hooks/contract_hook.py :: ValidationSummary`
* Progress/timing hooks:

  * `src/codeintel/build/hamilton/hooks/lifecycle.py :: ProgressBarHook`
  * `src/codeintel/build/hamilton/hooks/lifecycle.py :: BuildTimingHook`
  * `src/codeintel/build/hamilton/hooks/lifecycle.py :: create_progress_hook(...)`

## 3.11 Materialization and saver tagging

* Saver tagging/decorator logic:

  * `src/codeintel/build/hamilton/save_to.py :: SaveToObjectMetadataDecorator`
  * Saver tag context type used during tag construction:

    * `src/codeintel/build/hamilton/save_to.py :: SaverTagContext`
  * Tag-building helper: `src/codeintel/build/hamilton/save_to.py :: _build_saver_tags(...)`
  * Artifact path template resolution helper: `src/codeintel/build/hamilton/save_to.py :: _resolve_artifact_path_template(...)`
* Materialize option derivation for warehouse writes:

  * `src/codeintel/build/hamilton/materialize_options.py :: materialize_options(...) -> MaterializeOptions`
  * Config type: `src/codeintel/build/hamilton/materialize_options.py :: MaterializeOptionsConfig`
* Write policy helpers:

  * `src/codeintel/build/hamilton/materializers/write_policy.py :: resolve_materialize_options(...) -> MaterializeOptions`

## 3.12 Graph validation (DAG consistency checks)

* Main validation entrypoint:

  * `src/codeintel/build/hamilton/validate.py :: validate_nodes(runtime: HamiltonRuntime, *, graph: TargetGraph | None = None) -> GraphValidationResult`
* Validation “issue” model types:

  * `src/codeintel/build/hamilton/validate.py :: GraphValidationIssue`
  * `src/codeintel/build/hamilton/validate.py :: GraphValidationResult`
* Validation includes saver output collection and mismatch checking:

  * `src/codeintel/build/hamilton/validate.py :: _collect_saver_outputs(...)`
  * `src/codeintel/build/hamilton/validate.py :: _derived_outputs_mismatch_issues(...)`

# 4) Core runtime concepts & types (type map)

## 4.1 Build-layer types (Phase 1, carried forward)

* Targets/contracts/resources:

  * `src/codeintel/build/targets.py :: OutputTarget`
  * `src/codeintel/build/targets.py :: TargetGraph`
  * `src/codeintel/build/contracts.py :: OutputContract`
  * `src/codeintel/build/contracts.py :: ArtifactSpec`
  * `src/codeintel/build/resources.py :: TargetResources / TargetExecution`
* Config/parameters/hashing/session/state:

  * `src/codeintel/build/config.py :: BuildConfig / ConfigSection / BuildConfigOverrides / BuildConfigStack`
  * `src/codeintel/build/parameters.py :: TargetParameters`
  * `src/codeintel/build/hashing.py :: InputHashOptions / compute_input_hash(...)`
  * `src/codeintel/build/session.py :: BuildSession`
  * `src/codeintel/build/state_types.py :: TargetState / BuildState`
* Errors/policy:

  * `src/codeintel/build/errors.py :: BuildError / BuildErrorCollection`
  * `src/codeintel/build/execution_policy.py :: ExecutionPolicy`

## 4.2 Hamilton/runtime + env types (Phase 2 extension)

* Driver/runtime containers:

  * `src/codeintel/build/hamilton/runtime.py :: HamiltonRuntime`

    * Fields: `dr: h_driver.Driver`, `graph: TargetGraph`, `target_to_node: dict[str, str]`, `node_to_target: dict[str, str]`
* Build environment passed into Hamilton execution:

  * `src/codeintel/build/hamilton/env.py :: BuildEnv`

    * Key fields: `gateway: StorageGateway`, `snapshot: SnapshotRef`, `paths: BuildPaths`, `providers: Providers`, `config: BuildConfig`, `settings: BuildSettings`, `manifest_index: Mapping[str, OutputManifest] | None`, `strict_contracts: bool`, `validate_outputs: bool`
* Execution option bundles:

  * `src/codeintel/build/hamilton/execution_options.py :: BuildExecutionOptions`

    * Fields include: `profile`, `parallel_backend`, `max_workers`, `enable_hamilton_cache`, `cache_dir`, `enable_telemetry`, `enable_progress`, `enable_timing`
* Planning types:

  * `src/codeintel/build/hamilton/planner.py :: PlanEntry`
  * `src/codeintel/build/hamilton/planner.py :: StalenessExplanation`
  * `src/codeintel/build/hamilton/planner.py :: HamiltonBuildPlan`
* Execution result for executor-style compute nodes:

  * `src/codeintel/build/hamilton/execution_result.py :: ExecutionResult`

    * Fields include: `success`, `table_counts`, `error`, `skipped`, `skip_reason`, `warnings`
* Boundary aliases used across Hamilton IO surfaces:

  * `src/codeintel/build/hamilton/boundary_types.py :: MaterializationResult`
  * `src/codeintel/build/hamilton/boundary_types.py :: RowCounts = dict[str, int]`
  * `src/codeintel/build/hamilton/boundary_types.py :: TargetName = str`

## 4.3 Tag constants / node-type values (shared)

* Canonical constants:

  * `src/codeintel/core/hamilton/tags.py :: TAG_DOMAIN / TAG_TARGET / TAG_TABLE_KEY / TAG_NODE_TYPE / TAG_OUTPUT_KIND / ...`
  * `src/codeintel/core/hamilton/tags.py :: NODE_TYPE_MATERIALIZE / NODE_TYPE_DATASET / NODE_TYPE_COMPUTE / ...`

## 4.4 IO reference types (Hamilton DAG lineage handles)

* Dataset reference:

  * `src/codeintel/build/hamilton/io/dataset_ref.py :: DatasetRef(NamedTuple)`
* Artifact reference:

  * `src/codeintel/build/hamilton/io/artifact_ref.py :: ArtifactRef(NamedTuple)`
* Run-record protocol types they align with:

  * `src/codeintel/core/hamilton/records.py :: DatasetRefProtocol / ArtifactRefProtocol`
  * `src/codeintel/core/hamilton/records.py :: TargetRunRecord` (dataclass)

## 4.5 Materialization context + result models

* Materializer context resolution:

  * `src/codeintel/build/hamilton/materializers/base.py :: MaterializationContext`
  * `src/codeintel/build/hamilton/materializers/base.py :: resolve_materialization_context(...)`
* Canonical materialization result model:

  * `src/codeintel/core/execution/materialization.py :: MaterializationResult`

## 4.6 Manifests and run-level records

* Output manifest:

  * `src/codeintel/core/build_manifest.py :: OutputManifest`
* Manifest persistence entrypoint (build/hamilton layer):

  * `src/codeintel/build/hamilton/run_records.py :: save_manifest(env: BuildEnv, record: TargetRunRecord, *, change_delta: Mapping[str, object] | None = None) -> None`
* Run writer (run-level metadata capture):

  * `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter`

## 4.7 Storage boundary / warehouse IO types referenced by build/hamilton

* Gateway protocol used by `BuildEnv.gateway`:

  * `src/codeintel/storage/gateway/protocol.py :: StorageGateway`
* Warehouse materialize options:

  * `src/codeintel/storage/warehouse.py :: MaterializeOptions`
  * `src/codeintel/storage/warehouse.py :: UpsertConfig`
* Ibis IO config used by adapter wrappers:

  * `src/codeintel/storage/io/ibis_io.py :: IbisIOConfig`
* Table key type and helpers:

  * `src/codeintel/storage/helpers/table_key.py :: TableKey / parse_table_key(...) / validate_table_key(...)`

# 5) Data & IO model (as implemented)

## 5.1 Primary execution inputs to Hamilton

* The “single input object” pattern is `BuildEnv`, passed via driver `inputs`:

  * `src/codeintel/build/hamilton/env.py :: BuildEnv` (docstring: “single frozen input passed to Hamilton nodes”)
  * Example execution in runtime docstring:
    `src/codeintel/build/hamilton/runtime.py :: HamiltonRuntime` shows `runtime.dr.execute([...], inputs={"env": env, "graph": runtime.graph})` (graph also passed as an input mapping key in the example).
* `BuildEnv` identity and resources:

  * Snapshot identity: `src/codeintel/build/hamilton/env.py :: BuildEnv.snapshot: SnapshotRef`

    * `src/codeintel/config/primitives.py :: SnapshotRef`
  * Storage access: `src/codeintel/build/hamilton/env.py :: BuildEnv.gateway: StorageGateway`

    * `src/codeintel/storage/gateway/protocol.py :: StorageGateway`
  * Optional manifest preloading for incremental/skip decisions:

    * `src/codeintel/build/hamilton/env.py :: BuildEnv.manifest_index: Mapping[str, OutputManifest] | None`
    * `src/codeintel/core/build_manifest.py :: OutputManifest`

## 5.2 Tables and datasets: table keys and dataset refs

* Table identity uses a “fully qualified” table key string (e.g., `"analytics.function_metrics"`), treated as a `TableKey`:

  * `src/codeintel/storage/helpers/table_key.py :: TableKey` and helpers (parse/validate/split)
* DatasetRef is a lightweight handle that identifies a table without loading data:

  * `src/codeintel/build/hamilton/io/dataset_ref.py :: DatasetRef`
  * Fields include: `table_key`, `repo`, `commit`, `schema_version`, `row_count`, `source_target`, `metadata` (per class docstring and NamedTuple fields in-file).

## 5.3 Artifacts: artifact refs and path templates

* ArtifactRef is a lightweight handle for non-tabular outputs:

  * `src/codeintel/build/hamilton/io/artifact_ref.py :: ArtifactRef`
  * Fields include: `name`, `artifact_type`, `repo`, `commit`, `path`, `metadata`.
* Artifact path templates are validated/formatted in materializer utilities:

  * `src/codeintel/build/hamilton/materializers/path_templates.py :: validate_path_template(template: str) -> None`
  * `src/codeintel/build/hamilton/materializers/path_templates.py :: format_path_template(template: str, values: Mapping[str, object], *, formatter: Callable[[object], str] | None = None) -> str`
* Resolver used when saver tags include an artifact path template:

  * `src/codeintel/build/hamilton/save_to.py :: _resolve_artifact_path_template(...) -> str | None`

## 5.4 IO “write” boundary: materializers/savers produce MaterializationResult

* Cross-module materialization shape is a shared dataclass:

  * `src/codeintel/build/hamilton/boundary_types.py :: MaterializationResult`
* Table materialization saver (DuckDB/Ibis):

  * `src/codeintel/build/hamilton/materializers/duckdb_saver.py :: DuckDBIbisTableSaver.save_data(self, data: object) -> MaterializationResult`
  * Materialization context resolution used inside saver:

    * `src/codeintel/build/hamilton/materializers/base.py :: resolve_materialization_context(env: BuildEnv, graph: TargetGraph, *, target_name: str, options_hash: str | None, hash_options: InputHashOptions | None) -> MaterializationContext | MaterializationContextError`
* Row-based materialization saver (DuckDB rows):

  * `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py :: DuckDBRowsSaver.save_data(self, data: object) -> MaterializationResult`
* File artifact saver:

  * `src/codeintel/build/hamilton/materializers/artifact_saver.py :: FileArtifactSaver.save_data(self, data: object) -> MaterializationResult`
  * Path resolution helpers:

    * `src/codeintel/build/hamilton/materializers/artifact_saver.py :: _resolve_artifact_path_from_template(...) -> Path`

## 5.5 IO “read” boundary: support nodes + loader nodes

Support nodes are generated into a dynamic module and added to the driver:

* Support module generation:

  * `src/codeintel/build/hamilton/nodes/support_factory.py :: build_support_module(...) -> ModuleType`

Within that module, loaders and dataset/artifact ref nodes are function-factories:

* Dataset ref node pattern (produces `DatasetRef` from run record context):

  * `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_dataset_node_function(...) -> Callable[..., DatasetRef]`
* DataFrame loader node loads via gateway:

  * Factory: `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_dataframe_node_function(...)`
  * Loader body uses storage IO wrappers:

    * `src/codeintel/build/hamilton/nodes/support_factory.py :: dataframe_fn(env: BuildEnv, **kwargs: object) -> pd.DataFrame`
    * Calls: `src/codeintel/build/hamilton/io/ibis_adapter.py :: load_dataset_df(gateway=env.gateway, ref=ds_ref) -> pd.DataFrame`
* Query loader node loads “query result” via gateway (query semantics are in the wrapper implementation):

  * Factory: `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_query_node_function(...)`
  * Calls: `src/codeintel/build/hamilton/io/ibis_adapter.py :: load_dataset_ibis(gateway=env.gateway, ref=ds_ref) -> ir.Table`

## 5.6 Storage boundary: gateway protocol + ibis IO config

* Build layer treats storage as a gateway interface:

  * `src/codeintel/storage/gateway/protocol.py :: StorageGateway` (protocol with attributes like `ibis`, `build`, etc.)
* The hamilton IO adapter wrappers delegate to storage/io implementations using an IO config:

  * Wrapper module: `src/codeintel/build/hamilton/io/ibis_adapter.py :: save_dataframe(...) / save_rows(...) / upsert_dataframe(...) / load_dataset_df(...)`
  * Underlying implementation + config type:

    * `src/codeintel/storage/io/ibis_io.py :: IbisIOConfig`
    * `src/codeintel/storage/io/ibis_io.py :: save_dataframe(...) / save_rows(...) / upsert_dataframe(...) / load_dataset_df(...)`

## 5.7 MaterializeOptions derivation for snapshot-scoped writes

* Build-layer helper to create warehouse write options (snapshot-scoped):

  * `src/codeintel/build/hamilton/materialize_options.py :: materialize_options(env: BuildEnv, *, owner_target: str, config: MaterializeOptionsConfig | None = None) -> MaterializeOptions`
  * Config object:

    * `src/codeintel/build/hamilton/materialize_options.py :: MaterializeOptionsConfig`
    * Annotated fields: `mode`, `replace_scope`, `input_hash`, `upsert`, `use_staging`, `fallback_upsert_on_conflict`
* Additional “write policy” resolution helpers:

  * `src/codeintel/build/hamilton/materializers/write_policy.py :: resolve_materialize_options(...) -> MaterializeOptions`

## 5.8 Derived “what outputs exist” model (tables/artifacts) used for support generation and validation

* Derived target outputs from saver tags:

  * `src/codeintel/build/hamilton/introspect.py :: DerivedTargetOutputs`
  * Computation entrypoint:

    * `src/codeintel/build/hamilton/introspect.py :: derive_target_outputs_from_savers(runtime: HamiltonRuntime) -> DerivedTargetOutputs`
* Expected output inventories used by native/runtime utilities:

  * `src/codeintel/build/hamilton/native/outputs.py :: expected_table_keys_for_target(...)`
  * `src/codeintel/build/hamilton/native/outputs.py :: expected_artifact_names_for_target(...)`
  * `src/codeintel/build/hamilton/native/outputs.py :: artifact_templates_for_target(...)`

## 5.9 Manifests and skip decisions

* Manifest persistence is a first-class function in build/hamilton:

  * `src/codeintel/build/hamilton/run_records.py :: save_manifest(env: BuildEnv, record: TargetRunRecord, *, change_delta: Mapping[str, object] | None = None) -> None`
  * Constructs: `src/codeintel/core/build_manifest.py :: OutputManifest(...)` and calls gateway persistence:

    * `src/codeintel/build/hamilton/run_records.py :: save_manifest(...)` uses `env.gateway.build.save_manifest(manifest)` (gateway “build tracking” surface is via `StorageGateway.build`; concrete implementation not confirmed from protocol alone).
* Skip logic / manifest lookup service:

  * `src/codeintel/build/hamilton/run_record_utils.py :: BuildManifestService`
  * Primary skip decision function:

    * `src/codeintel/build/hamilton/run_record_utils.py :: should_skip(env: BuildEnv, *, request: SkipCheckRequest) -> bool`

## 5.10 Contract/schema validation in the IO flow (as referenced)

* Contract-enforcing gateway wrappers exist (used when strict/validation is enabled):

  * `src/codeintel/build/hamilton/contracts/enforced_gateway.py :: ContractEnforcingStorageGateway`
  * `src/codeintel/build/hamilton/contracts/enforced_gateway.py :: ContractEnforcingIbisGateway`
* Pandera schema/validation hook utilities (contract layer):

  * `src/codeintel/build/hamilton/contracts/pandera_hook.py :: get_pandera_schema(table_key: str, *, registry: SchemaRegistry) -> pa.DataFrameSchema | None`
  * `src/codeintel/build/hamilton/contracts/pandera_hook.py :: validate_dataframe(...) -> ValidationResult`
* Lifecycle hook that activates contract enforcement around node execution:

  * `src/codeintel/build/hamilton/hooks/contract_hook.py :: ContractEnforcementHook`


# 6) Target orchestration model (derived behavior)

## 6.1 Goal selection and target closure

* **Goal resolution from CLI arguments**:

  * `src/codeintel/cli/handlers/build.py :: _resolve_goals(targets, module, target_scope, graph) -> list[str]` validates requested names via `graph.get(...)` or enumerates module/all targets via `TargetGraph.targets_for_module(...)` / `TargetGraph.all_targets`.
  * Domain inference for telemetry is derived from node tags when possible:
    `src/codeintel/cli/handlers/build.py :: _resolve_domain_for_goals(goals: Sequence[str]) -> str | None` reads `tag_index.tags_by_node[...]` and `ht.TAG_DOMAIN`.
* **Closure computation**:

  * Execution uses `TargetGraph.topological_order(...)`, which expands dependencies via `TargetGraph.transitive_deps(...)` and sorts using Kahn-style in-degree reduction.
    Evidence: `src/codeintel/build/targets.py :: TargetGraph.topological_order(names: Iterable[str]) -> tuple[str, ...]`.

## 6.2 Planning (dry-run / explain) vs execution (run)

* **Planning** is computed without executing nodes:

  * `src/codeintel/build/hamilton/planner.py :: compute_plan(env, graph, requested, graph_source="hamilton") -> HamiltonBuildPlan`:

    * Loads manifests via `env.manifest_index` or `env.gateway.build.list_manifests(...)`.
    * Computes per-target `PlanEntry` via `_compute_entry_for_target(...)`.
  * Plan reasons are encoded in `PlanEntry.reason: PlanReason` (`"forced"|"no_manifest"|"hash_changed"|"up_to_date"|"upstream_missing"|"no_impl"`).
    Evidence: `src/codeintel/build/hamilton/planner.py :: PlanEntry`, `src/codeintel/build/hamilton/planner.py :: _compute_entry_for_target(...)`.
* **Execution** runs Hamilton DAG nodes for the closure:

  * `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._run_with_state(...)`:

    * Computes closure (`_compute_closure(...)`).
    * Converts closure targets to node names via `target_to_node_name(...)` (`t__*`).
      Evidence: `src/codeintel/build/hamilton/driver_factory.py :: target_to_node_name(...)`.
    * Applies preflight for missing input tables (`_apply_preflight(...)`).
    * Executes driver: `context.runtime.dr.execute(list(final_vars), inputs={"env": execution_env, "graph": context.runtime.graph})`.
      Evidence: `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._execute_dag(...)`.

## 6.3 Skip behavior: where and how it is enforced

* **Native-target skip** (tool/compute steps):
  `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor.should_skip() -> bool` calls `should_skip_native_target(env, target, input_hash, options_hash=...)`.
  `should_skip_native_target(...)` bypasses skipping when `target.name in env.force_targets`.
  Evidence: `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor.for_target(...)`, `src/codeintel/build/hamilton/run_record_utils.py :: should_skip_native_target(...)`.
* **Materializer skip** (IO boundary):
  `src/codeintel/build/hamilton/materializers/base.py :: resolve_materialization_context(...)` computes `input_hash` and sets `should_skip` via `should_skip_native_target(...)`.
  Example: `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py :: DuckDBRowsSaver.save_data(...)` returns a “skipped” metadata dict when `context.should_skip` is True or when upstream output is `None`.
  Evidence: `src/codeintel/build/hamilton/materializers/base.py :: resolve_materialization_context(...)`, `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py :: DuckDBRowsSaver.save_data(...)`.

## 6.4 Preflight missing-input blocking (table existence)

* Before execution, the executor can block targets that read tables not produced in-closure and not present in the warehouse:

  * Missing-table detection: `src/codeintel/build/hamilton/executor.py :: _preflight_missing_inputs(...)` derives `TargetIOSurface` via `derive_target_io_surface(...)` and checks existence via `_table_key_exists(...)` (`env.gateway.policy.table_exists(schema, table)`).
    Evidence: `src/codeintel/build/hamilton/executor.py :: _preflight_missing_inputs(...)`, `src/codeintel/build/hamilton/introspect.py :: derive_target_io_surface(...)`, `src/codeintel/build/hamilton/executor.py :: _table_key_exists(...)`.
  * Blocking propagation: `src/codeintel/build/hamilton/executor.py :: _blocked_targets(graph, roots) -> set[str]` expands to downstream dependents via `TargetGraph.dependents_of(...)`.
    Evidence: `src/codeintel/build/hamilton/executor.py :: _blocked_targets(...)`, `src/codeintel/build/targets.py :: TargetGraph.dependents_of(name: str)`.

## 6.5 Failure/record synthesis at the orchestration layer

* If the DAG execution errors, the executor synthesizes missing `TargetRunRecord` objects:

  * `src/codeintel/build/hamilton/executor.py :: _ensure_failure_records(...)` injects failure records for targets whose output is missing or not a `TargetRunRecord`.
  * Failure record construction uses input hash when possible: `src/codeintel/build/hamilton/executor.py :: _failure_record(...)` → `_safe_input_hash(...)` → `compute_target_input_hash(...)`.
    Evidence: `src/codeintel/build/hamilton/executor.py :: _ensure_failure_records(...)`, `src/codeintel/build/hamilton/executor.py :: _failure_record(...)`, `src/codeintel/build/hamilton/run_records.py :: compute_target_input_hash(...)`.

## 6.6 Run finalization and persistence

* Finalization aggregates per-target statuses from returned `TargetRunRecord` objects:

  * `src/codeintel/build/hamilton/executor.py :: _categorize_outputs(closure, outputs, runtime) -> (computed, skipped, failed)`.
* Persistence is done via a writer:

  * `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.start_run(...)`
  * `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.save_run_targets(...)`
  * `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.persist_asset_catalog(...)`
  * `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.complete_run(...)`
    Evidence: `src/codeintel/build/hamilton/executor.py :: _finalize_run(...)`.

# 7) Walkthrough: “request one target” end-to-end trace

> Representative target: **`ast`** (ingestion). Anchor node: `t__ast`.
> The trace below is the *observed* code path and DAG structure (no runtime logs).

## 7.1 CLI → execution context → BuildEnv

1. **CLI handler resolves runtime + goals + execution args**

   * Entry: `src/codeintel/cli/handlers/build.py :: build_run_handler(ctx)`.
   * Goal selection: `src/codeintel/cli/handlers/build.py :: _resolve_goals(...)` (validates `graph.get("ast")`).
   * Execution context creation: `src/codeintel/cli/handlers/build.py :: _build_execution_context(runtime, requested_datasets=tuple(goals)) -> ExecutionContext`.
   * Execution args include force/validation/caching/parallel: `src/codeintel/cli/handlers/build.py :: BuildExecutionArgs` (dataclass).

2. **BuildEnv constructed with manifests + config + providers**

   * Manifests loaded and indexed: `src/codeintel/cli/handlers/build.py :: _execute_build_hamilton(...)` calls `gateway.build.list_manifests(repo, commit)` and builds `manifest_index = {m.target: m for m in manifests_list}`.
   * Build context: `src/codeintel/build/run_context.py :: BuildRunContext.from_execution_context(...)`.
   * BuildEnv: `src/codeintel/build/run_context.py :: BuildRunContext.build_env(...) -> BuildEnv`.
   * Manifest service is `env.gateway.build`: `src/codeintel/build/hamilton/env.py :: BuildEnv.manifest_service`.

## 7.2 Executor startup → closure → node mapping

3. **Executor run() creates run_id + builds runtime + begins run persistence**

   * `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor.run(env, targets=["ast", ...])`.
   * Run id: `src/codeintel/build/hamilton/executor.py :: _generate_run_id() -> new_run_id("hamilton")`.
   * Runtime build (driver + graph + mappings): `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._build_runtime(...)` → `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)`.
   * Run record start: `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.start_run(...)`.

4. **Closure computation uses TargetGraph dependencies**

   * `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._compute_closure(graph, targets, run_id)` calls `TargetGraph.topological_order(targets)`.
   * Evidence: `src/codeintel/build/targets.py :: TargetGraph.topological_order(...)`.

5. **Targets mapped to Hamilton node names (`t__*`)**

   * `src/codeintel/build/hamilton/executor.py :: _map_closure_to_nodes(closure, runtime)` calls `target_to_node_name(...)`.
   * Mapping source: `src/codeintel/build/hamilton/runtime.py :: HamiltonRuntime.target_to_node`.
   * Mapping construction: `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)` sets `t2n = {t.name: target_node(t.name) ...}`.

## 7.3 DAG execution for `ast`: tool step → ingest → row nodes → materializers → target record

6. **`t__ast__run` tool step performs manifest skip gating before extraction**

   * Node: `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py :: t__ast__run(env, graph, t__modules, module_records, ast__hash_options) -> AstToolOutput`.
   * It constructs `ToolRunContext(..., target_name=AST_TARGET_NAME, hash_options=ast__hash_options, skip_reason="AST extraction skipped")` and calls `run_tool_step(context=..., run=_execute)`.
     Short excerpt (≤25 words): `output = run_tool_step(context=context, run=_execute)`
     Evidence: `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py :: t__ast__run(...)`.
   * `run_tool_step(...)` gates on `NativeTargetExecutor.should_skip()`:

     * `src/codeintel/build/hamilton/native/patterns/tool_target.py :: run_tool_step(...)` creates `NativeTargetExecutor.for_target(..., hash_options=...)`.
     * `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor.should_skip()` delegates to `should_skip_native_target(...)`.
       Evidence: `src/codeintel/build/hamilton/native/patterns/tool_target.py :: run_tool_step(...)`, `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor.should_skip()`.

7. **`t__ast__ingest` packages row payloads and produces an ExecutionResult**

   * Node: `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py :: t__ast__ingest(t__ast__run) -> IngestStep[...]`.
   * Behavior is based on `t__ast__run.result`:

     * If skipped → `ExecutionResult.skip(...)`
     * If failed → `ExecutionResult.failed(...)`
     * If ok → payload `{AST_NODES_TABLE_KEY: rows, AST_METRICS_TABLE_KEY: rows}` + `table_counts` lengths.
       Evidence: `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py :: t__ast__ingest(...)`, `src/codeintel/build/hamilton/execution_result.py :: ExecutionResult` (type).

8. **Row-producing nodes return `None` when upstream skipped/failed**

   * Nodes:

     * `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py :: ast__node_rows(t__ast__ingest) -> tuple[...] | None`
     * `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py :: ast__metric_rows(t__ast__ingest) -> tuple[...] | None`
   * Both contain the same gate: `if t__ast__ingest.result.skipped or not t__ast__ingest.result.success: return None`.
     Evidence: `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py :: ast__node_rows(...)`, `... :: ast__metric_rows(...)`.

9. **Save-to decorators generate materializer nodes (`m__core__ast_nodes`, `m__core__ast_metrics`)**

   * Decorator factory: `src/codeintel/build/hamilton/native/patterns/savers.py :: save_rows(context=..., spec=TableSaveSpec(table_key=...))`.
   * It wraps with `SaveToObjectMetadataDecorator([DuckDBRowsSaver], output_name_=materialize_node(spec.table_key), ...)`.
     Evidence: `src/codeintel/build/hamilton/native/patterns/savers.py :: save_rows(...)`, `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py :: DuckDBRowsSaver(DataSaver)`, `src/codeintel/build/hamilton/naming.py :: materialize_node(table_key: str) -> str`.

10. **`DuckDBRowsSaver.save_data(...)` handles skip + writes + returns metadata dict**

* Save path:

  * Resolves context: `src/codeintel/build/hamilton/materializers/base.py :: resolve_materialization_context(...)` (computes `input_hash`, `should_skip` via `should_skip_native_target(...)`).
  * If should_skip → returns skipped metadata; if data is None → returns skipped metadata.
  * Else writes to warehouse and returns succeeded metadata.
    Evidence: `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py :: DuckDBRowsSaver.save_data(...)`, `src/codeintel/build/hamilton/materializers/base.py :: resolve_materialization_context(...)`.

11. **`t__ast` finalizes from collected materialization metadata**

* Node: `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py :: t__ast(ast__finalize_context, t__ast__run, t__ast__ingest, ast__table_materializations) -> TargetRunRecord`.
* The finalizer call: `finalize_target_from_materializations(context=..., tool_step=..., ingest_step=..., table_materializations=...)`.
  Evidence: `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py :: t__ast(...)`, `src/codeintel/build/hamilton/native/patterns/tool_target.py :: finalize_target_from_materializations(...)`.

12. **Finalization synthesizes a `TargetRunRecord` and persists an `OutputManifest`**

* `finalize_target_from_materializations(...)` calls:
  `src/codeintel/build/hamilton/native/materialization_records.py :: record_from_materializations(...)`.
* `record_from_materializations(...)` creates a success `TargetRunRecord` and calls `save_manifest(env, record, change_delta=...)`.
  Evidence: `src/codeintel/build/hamilton/native/materialization_records.py :: record_from_materializations(...)`, `src/codeintel/build/hamilton/run_records.py :: save_manifest(...)`.

## 7.4 Run-level persistence after DAG execution returns

13. **Executor aggregates `TargetRunRecord` outputs and persists run summaries**

* `_finalize_run(...)` collects all `TargetRunRecord` instances found in `outputs.values()` and writes them via `BuildRunWriter`.
* Evidence: `src/codeintel/build/hamilton/executor.py :: _finalize_run(...)`, `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.save_run_targets(...)`, `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.complete_run(...)`.

# 8) Extension mechanics (how new behavior is introduced today)

## 8.1 Add a new native target module (discovery + driver composition)

* **Add a new `.py` file under a native domain directory**:
  Discovery scans `*.py` (excluding `__init__.py`) under `codeintel.build.hamilton.native.{ingestion,graphs,analytics,export}`.
  Evidence: `src/codeintel/build/hamilton/native/discovery.py :: native_module_paths() -> tuple[str, ...]`, `_NATIVE_DOMAINS = ("ingestion","graphs","analytics","export")`.

* **New module is automatically imported into the Driver**:
  Evidence: `src/codeintel/build/hamilton/native/discovery.py :: load_native_modules() -> tuple[ModuleType, ...]`, `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)` calls `load_native_modules()` and includes those modules in `.with_modules(*native_mods, support_module)`.

## 8.2 Declare a new target anchor (`t__*`) and attach target metadata via tags

* **Anchor decorator is `@codeintel_target(...)`** (materialize-node tag + embedded spec metadata as JSON tags):
  Evidence: `src/codeintel/build/hamilton/native/target_decorators.py :: codeintel_target(domain: str, target: str, spec: TargetSpecDescriptor | None) -> Decorator`.

* **Target spec payload is encoded into tags** (resources/execution/parameters/spec_version/estimated_duration):
  Evidence: `src/codeintel/build/hamilton/native/target_decorators.py :: TargetSpecDescriptor`, `... :: codeintel_target(...)` uses `ht.TAG_TARGET_RESOURCES`, `ht.TAG_TARGET_EXECUTION`, `ht.TAG_TARGET_PARAMETERS`, `ht.TAG_TARGET_SPEC_VERSION`.

* **Compiler turns anchor nodes into `OutputTarget`**:
  Evidence: `src/codeintel/build/hamilton/target_spec_compiler.py :: compile_output_targets_from_driver(...)`, `src/codeintel/build/targets.py :: OutputTarget`.

## 8.3 Add datasets/artifacts to a target via saver-decorator patterns

* **Row/table outputs**: `save_rows(...)` generates a `DuckDBRowsSaver`-backed saver node and tags it for contract output identity (`table_key`, `target_name`, `output_role`, etc.).
  Evidence: `src/codeintel/build/hamilton/native/patterns/savers.py :: save_rows(...)`, `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py :: DuckDBRowsSaver`.

* **Ibis table outputs**: `save_ibis_table(...)` creates `DuckDBIbisTableSaver` saver nodes.
  Evidence: `src/codeintel/build/hamilton/native/patterns/savers.py :: save_ibis_table(...)`, `src/codeintel/build/hamilton/materializers/duckdb_saver.py :: DuckDBIbisTableSaver`.

* **File artifacts**: `save_artifact(...)` creates `FileArtifactSaver` nodes with `artifact_name` and `path_template`.
  Evidence: `src/codeintel/build/hamilton/native/patterns/savers.py :: save_artifact(...)`, `src/codeintel/build/hamilton/materializers/artifact_saver.py :: FileArtifactSaver`.

## 8.4 Add generalized tool-backed scaffolds via template attachers

* **Template generator attaches multiple nodes (run/ingest/savers/collectors/final anchor) to a module**:
  Evidence: `src/codeintel/build/hamilton/native/patterns/tool_target.py :: attach_tool_target_template(...)`, `src/codeintel/build/hamilton/nodes/module_attach.py :: tagged_attach_node(...)`.

## 8.5 Extend runtime behavior via adapters/hooks (telemetry, contracts, parallelism)

* **Hooks are assembled into Hamilton lifecycle adapters**:

  * `src/codeintel/build/hamilton/hooks/__init__.py :: build_hooks(run_id, writer, graph, options=...)`
  * Hook options derived from execution options: `src/codeintel/build/hamilton/execution_options.py :: BuildExecutionOptions.hook_options(env=...)`.

* **Parallel adapter selection** is based on `parallel_backend` and `max_workers`:
  Evidence: `src/codeintel/build/hamilton/adapters/parallel.py :: create_parallel_adapter(...)`, `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._build_runtime(...)`.

* **Contract enforcement can wrap the gateway during execution** when `env.strict_contracts` is set:
  Evidence: `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._execute_dag(...)` wraps with `ContractEnforcingStorageGateway(...)`, `src/codeintel/build/hamilton/contracts/enforced_gateway.py :: ContractEnforcingStorageGateway`.

# 9) Conventions & invariants (observed)

## 9.1 Naming conventions (node identifiers)

* Canonical conversions:

  * Targets: `src/codeintel/build/hamilton/naming.py :: target_node(target_name) -> "t__..."`.
  * Materializers: `src/codeintel/build/hamilton/naming.py :: materialize_node(table_key) -> "m__..."`.
  * Dataset loaders: `src/codeintel/build/hamilton/naming.py :: query_node(table_key) -> "q__..."`, `... :: dataframe_node(table_key) -> "df__..."`.
  * Reverse: `src/codeintel/build/hamilton/naming.py :: node_to_target(node_name: str) -> str | None`.

## 9.2 “Native-only” driver composition is explicit

* `src/codeintel/build/hamilton/driver_factory.py` docstring states “native-only”; code constructs driver with `load_native_modules()` and a generated support module.
  Evidence: `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)`, `src/codeintel/build/hamilton/driver_factory.py :: _build_support_graph_and_module(...)`.

## 9.3 Planning assumes every target resolves to a native implementation

* `_compute_entry_for_target(...)` raises `RuntimeError` if `target_name not in native_names`.
  Evidence: `src/codeintel/build/hamilton/planner.py :: _compute_entry_for_target(...)` (excerpt ≤25 words): `if target_name not in native_names: raise RuntimeError(...)`.

## 9.4 Manifest/hash invariants

* Input hash includes engine version + repo/commit + target name + sorted dependency manifest hashes (+ optional file_state_hash + options_hash).
  Evidence: `src/codeintel/build/hashing.py :: compute_input_hash_with_deps(...)`, `src/codeintel/build/engine_version.py :: get_build_engine_version(...)`.
* Skip check compares computed values to manifest:

  * `src/codeintel/build/hash_evaluator.py :: evaluate_hash_state(...)` returns `HashEvaluation(status="current", reason="up_to_date")` when stored input hash matches.
  * `src/codeintel/build/hamilton/run_record_utils.py :: should_skip(...)` returns True when evaluation status is `"current"`.

## 9.5 Strict contract invariants (row counts and write validation)

* For **native target record creation**, strict mode requires row_counts to exist and match exactly the target contract table keys:

  * `src/codeintel/build/hamilton/run_records.py :: _validate_strict_row_counts(...)`.
* For **row materialization**, strict contract enforcement validates table writes:

  * `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py :: DuckDBRowsSaver.save_data(...)` calls `ContractEnforcer.validate_table_write(self.table_key)`.
    Evidence: `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py :: DuckDBRowsSaver.save_data(...)`, `src/codeintel/build/hamilton/contracts/enforcement.py :: ContractEnforcer.validate_table_write(...)` (symbol location).

## 9.6 TargetGraph invariants around output identity collisions

* Duplicate table_key or artifact name declared by multiple targets raises `ValueError` when building TargetSystem indexes.
  Evidence: `src/codeintel/build/target_metadata.py :: _build_indexes(...)` (checks `by_table_key` and `by_artifact_name` collisions).

## 9.7 Saver-tag invariants for output derivation

* `derive_target_outputs_from_savers(...)` requires saver nodes with `output_role == "contract"` to have exactly one of `{table_key, artifact}` tags set, and requires `artifact_path_template` for artifact outputs.
  Evidence: `src/codeintel/build/hamilton/introspect.py :: derive_target_outputs_from_savers(...)`, `src/codeintel/build/hamilton/introspect.py :: _resolve_output_identity(...)`, `src/codeintel/build/hamilton/introspect.py :: _iter_contract_saver_tags(...)`.

## 9.8 Preflight table existence invariant

* Executor preflight blocks targets that read non-produced tables that do not exist in storage:

  * Existence check uses `env.gateway.policy.table_exists(schema, table)`.
    Evidence: `src/codeintel/build/hamilton/executor.py :: _table_key_exists(...)`, `src/codeintel/build/hamilton/executor.py :: _preflight_missing_inputs(...)`.

# 10) Glossary (project-specific vocabulary

* **Target**: A build unit identified by a string name, represented as `OutputTarget`.
  Evidence: `src/codeintel/build/targets.py :: OutputTarget(name: str, module: TargetModule, contract: OutputContract, ...)`.

* **Target anchor node (`t__*`)**: A Hamilton node function that represents the “materialize/anchor” for a target and returns a `TargetRunRecord`.
  Evidence: `src/codeintel/build/hamilton/native/target_decorators.py :: codeintel_target(...)` (applies `tag_materialize`), examples: `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py :: t__ast(...)`.

* **Closure**: The dependency-expanded set of requested targets, ordered with dependencies first.
  Evidence: `src/codeintel/build/targets.py :: TargetGraph.topological_order(...)`.

* **Manifest**: A persisted record of a target computation for a specific repo/commit, used for skip decisions.
  Evidence: `src/codeintel/core/build_manifest.py :: OutputManifest`, `src/codeintel/build/hamilton/run_records.py :: save_manifest(...)`.

* **Input hash**: Content-addressable 16-hex hash for a target derived from repo/commit, dependencies’ manifest hashes, engine version, and optional hashes.
  Evidence: `src/codeintel/build/hashing.py :: compute_input_hash_with_deps(...)`.

* **Options hash**: A 16-hex hash derived from serialized per-target config parameters.
  Evidence: `src/codeintel/build/hashing.py :: compute_target_options_hash(...)`, `src/codeintel/build/hamilton/run_record_utils.py :: options_hash_for_target(...)`.

* **Skip**: A decision that a target (or materializer) does not need to execute because a manifest matches current hashes (unless forced).
  Evidence: `src/codeintel/build/hamilton/run_record_utils.py :: should_skip_native_target(...)`, `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor.should_skip()`.

* **Materializer node (`m__*`)**: A Hamilton saver node representing an IO boundary; returns `MaterializationResult` describing the write/skip/failure.
  Evidence: `src/codeintel/build/hamilton/naming.py :: materialize_node(...)`, `src/codeintel/build/hamilton/boundary_types.py :: MaterializationResult`, `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py :: DuckDBRowsSaver.save_data(...)`.

* **Support module**: A generated Hamilton module containing synthetic nodes (dataset refs/loaders/artifact refs) derived from the target graph and saver output tags.
  Evidence: `src/codeintel/build/hamilton/driver_factory.py :: _build_support_graph_and_module(...)`, `src/codeintel/build/hamilton/nodes/support_factory.py :: build_support_module(...)`.

* **`TargetRunRecord`**: The per-target output record emitted by `t__*` nodes (status, hashes, datasets/artifacts, errors).
  Evidence: `src/codeintel/core/hamilton/records.py :: TargetRunRecord` (type is imported broadly), record creation paths: `src/codeintel/build/hamilton/run_records.py :: create_run_record(...)`, `src/codeintel/build/hamilton/native/materialization_records.py :: record_from_materializations(...)`.

* **Decision trace**: A JSON artifact capturing the plan entries (compute/skip/missing/blocked + reasons + hashes) for requested targets.
  Evidence: `src/codeintel/build/hamilton/decision_trace.py :: build_decision_trace_payload(...)`, native target: `src/codeintel/build/hamilton/native/export/decision_trace.py :: decision_trace__content(...) / t__decision_trace(...)`.



PHASE 3 OUTPUT

# 0) What I inspected

**Phase 1–2 (carried forward, authoritative):**

* Codebase archive extracted at: `/mnt/data/phase6/` (from `/mnt/data/CodeIntel_Centralizing_phase6.zip`).
* Build system source tree: `src/codeintel/build/**` (inventory + symbol scan).
* Hamilton build subtree (in depth): `src/codeintel/build/hamilton/**` (driver/planner/executor, native discovery, support nodes, IO/materializers, hooks/contracts, introspection/validation).
* Composition roots that invoke build (CLI boundary): `src/codeintel/cli/handlers/build.py :: build_run_handler(...)`, `src/codeintel/cli/handlers/build.py :: _execute_build_hamilton(...)`, `pyproject.toml :: [project.scripts] codeintel = "codeintel.cli:main"`.
* Repo docs requested: `AGENTS.md`, `README_METADATA.md`, `docs/*` **not found** under `/mnt/data/phase6` (filesystem search only; not confirmed beyond that).

**Phase 3 additions (per prompt: execution entrypoints + incremental logic + one target trace + invariants scans):**

* Build execution path (Hamilton execution):
  `src/codeintel/cli/handlers/build.py :: _execute_build_hamilton(...)` →
  `src/codeintel/build/run_context.py :: BuildRunContext.from_execution_context(...)` →
  `src/codeintel/build/run_context.py :: BuildRunContext.build_env(...)` →
  `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor.run(...)` →
  `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)` →
  `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._execute_dag(...)`
* Incremental logic chain:

  * Hash computation: `src/codeintel/build/hashing.py :: compute_input_hash_with_deps(...)`
  * Hash evaluation: `src/codeintel/build/hash_evaluator.py :: evaluate_hash_state(...)`, `src/codeintel/build/hash_evaluator.py :: compute_hash_evaluation(...)`
  * Skip decision (manifest-based): `src/codeintel/build/hamilton/run_record_utils.py :: should_skip_native_target(...)`
  * Manifest persistence: `src/codeintel/build/hamilton/run_records.py :: save_manifest(...)`
  * Materializer skip (same primitive): `src/codeintel/build/hamilton/materializers/base.py :: resolve_materialization_context(...)`
* Representative target trace selected: **`ast`** (ingestion domain)

  * Target anchor: `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py :: t__ast(...)`
  * Tool run step: `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py :: t__ast__run(...)`
  * Row producers + savers:
    `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py :: ast__node_rows(...)` (decorated by `save_rows(...)`)
    `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py :: ast__metric_rows(...)` (decorated by `save_rows(...)`)
    `src/codeintel/build/hamilton/native/patterns/savers.py :: save_rows(...)` (uses `SaveToObjectMetadataDecorator([DuckDBRowsSaver], ...)`)
  * Finalization from saver metadata: `src/codeintel/build/hamilton/native/patterns/tool_target.py :: finalize_target_from_materializations(...)`
  * Record synthesis + manifest: `src/codeintel/build/hamilton/native/materialization_records.py :: record_from_materializations(...)`

