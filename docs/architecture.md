PHASE 3 OUTPUT

# 0) What I inspected

* Codebase source: extracted `CodeIntel_Centralizing_Phase3.zip` into a working directory and scanned repo structure (`pyproject.toml`, `src/`, `tests/`, `tools/`, `scripts/`).
* Build-layer full inventory:

  * Full package tree under `src/codeintel/build/**` (all `.py` modules + subpackages). `src/codeintel/build/__init__.py :: __getattr__(...)`
* Hamilton integration deep dive (Phase 2 scope confirmed by direct file inspection):

  * Composition root + dependency derivation: `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)`, `src/codeintel/build/hamilton/introspect.py :: derive_target_dependencies(...)`
  * Execution + hooks/adapters: `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor.run(...)`, `src/codeintel/build/hamilton/hooks/contract_hook.py :: ContractEnforcementHook.pre_node_execute(...)`
  * IO/materializers + run-record/manifest bridge: `src/codeintel/build/hamilton/save_to.py :: SaveToObjectMetadataDecorator`, `src/codeintel/build/hamilton/run_records.py :: save_manifest(...)`
* Phase 3 additional inspection (derived behavior, end-to-end trace, invariants):

  * CLI orchestration entrypoints: `src/codeintel/cli/handlers/build.py :: build_run_handler(...)`, `build_plan_handler(...)`, `build_status_handler(...)`
  * Planning / incremental reasoning: `src/codeintel/build/hamilton/planner.py :: compute_plan(...)`, `_compute_entry_for_target(...)`
  * Hashing + manifest evaluation: `src/codeintel/build/hashing.py :: compute_input_hash_with_deps(...)`, `src/codeintel/build/hash_evaluator.py :: compute_hash_evaluation(...)`
  * State/blocked propagation used by `build status`: `src/codeintel/build/state.py :: StateValidator.validate()`, `src/codeintel/build/state_computer.py :: StateComputer._propagate_blocking(...)`
  * Representative target trace chosen: `scip` native target implementation: `src/codeintel/build/hamilton/native/ingestion/scip.py :: t__scip(...)`, `t__scip__run(...)`, `scip__symbol_rows(...)`, `scip__materializations(...)`

# 1) Executive architecture summary

* **Build execution is orchestrated via Hamilton, with a single composition root that constructs a Driver from native modules + generated support nodes**: `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)`
* **Target metadata (contracts/resources/execution/params/description) is declared alongside native nodes and registered into an in-memory registry; dependencies are derived from the Hamilton graph**: `src/codeintel/build/hamilton/native/target_spec_helpers.py :: register_output_targets(...)`, `src/codeintel/build/hamilton/introspect.py :: derive_target_dependencies(...)`
* **A TargetGraph is produced whose dependency edges are Hamilton-derived (materialize-node tags → target-to-target edges), and is used for closure computation**: `src/codeintel/build/hamilton/introspect.py :: target_graph_from_hamilton(...)`, `src/codeintel/build/targets.py :: TargetGraph.topological_order(...)`
* **Execution path (“build run”)**:

  * CLI builds a `BuildRunContext` and `BuildEnv`, then calls `HamiltonBuildExecutor.run(env, targets)` which computes the dependency closure, maps targets to `t__*` node names, and executes the Hamilton Driver with inputs `{"env": ..., "graph": ...}`: `src/codeintel/cli/handlers/build.py :: _execute_build_hamilton(...)`, `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._execute_dag(...)`
  * The executor persists run lifecycle + per-target records + optional node telemetry via `BuildRunWriter`: `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._run_with_state(...)`, `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.start_run(...)`
* **Planning path (“build plan”)**:

  * CLI constructs `BuildEnv` (including `manifest_index` and `force_targets`) and calls `compute_plan(...)`, which topologically orders the closure and creates `PlanEntry` per target using hash evaluation vs manifests, plus upstream blocking: `src/codeintel/cli/handlers/build.py :: build_plan_handler(...)`, `src/codeintel/build/hamilton/planner.py :: compute_plan(...)`
* **Incremental logic (hashing + manifests)**:

  * Input hash is computed from engine version + repo/commit + target name + dependency manifest input hashes (or `"MISSING"`) + optional options/file-state hash; manifest comparison yields `missing/current/stale`: `src/codeintel/build/hashing.py :: compute_input_hash_with_deps(...)`, `src/codeintel/build/hash_evaluator.py :: evaluate_hash_state(...)`
  * Native targets can skip at runtime via `should_skip_native_target(...)` (force-targets bypass skip) and write a new `OutputManifest` on success: `src/codeintel/build/hamilton/run_records.py :: should_skip_native_target(...)`, `save_manifest(...)`
* **Data model of “target outputs” is standardized as `TargetRunRecord` carrying `datasets` (DatasetRef) and `artifacts` (ArtifactRef), produced by native target materialize nodes (`t__*`)**: `src/codeintel/core/hamilton/records.py :: TargetRunRecord`, `src/codeintel/build/hamilton/native/ingestion/scip.py :: t__scip(...)`
* **Downstream dependency enforcement is achieved structurally via support nodes**:

  * Dataset nodes `d__*` extract `DatasetRef` from the producing `TargetRunRecord` and raise if missing; loader nodes `q__*`/`df__*` then read from storage using that ref: `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_dataset_node_function(...)`, `_create_query_node_function(...)`

# 2) Repository map (build-focused)

## 2.1 `src/codeintel/build/` top-level modules

* Build public facade + lazy exports:

  * Re-exports build-layer symbols via `_LAZY_IMPORTS` and `__getattr__`. `src/codeintel/build/__init__.py :: _LAZY_IMPORTS` / `__getattr__(...)`
* Target model + dependency graph:

  * Target definition and dependency graph container. `src/codeintel/build/targets.py :: OutputTarget` / `TargetGraph`
* Contract model (what targets produce):

  * Tables + artifacts contract dataclasses and helpers. `src/codeintel/build/contracts.py :: OutputContract` / `ArtifactSpec`
* Target resource + execution hints:

  * Declares per-target resource requirements and execution hints. `src/codeintel/build/resources.py :: TargetResources` / `TargetExecution`
* Configuration (TOML build config) and stacking:

  * Loads + models build config sections, and composes per-run overrides via stack. `src/codeintel/build/config.py :: BuildConfig` / `BuildConfigStack` / `load_build_config(...)`
* Per-run config (profiles + plugin option layering):

  * Per-run configuration for resolving plugin options and per-target overrides. `src/codeintel/build/run_config.py :: BuildRunConfig`
* Run context → Build environment assembly:

  * Builds a build-time environment object and resolves run options. `src/codeintel/build/run_context.py :: BuildRunContext.build_env(...)` / `BuildRunContext.build_execution_options(...)`
* Providers (DI container for tool services):

  * Provider container + production factory wiring. `src/codeintel/build/providers.py :: Providers` / `create_default_providers(...)`
* Build settings accessors:

  * Thin wrappers over core runtime settings loader for build + execution settings. `src/codeintel/build/settings.py :: get_build_settings()` / `get_hamilton_execution_settings()`
* Hashing + hash evaluation:

  * Input hash computation (target+deps+options) and options hashing. `src/codeintel/build/hashing.py :: compute_input_hash(...)` / `compute_options_hash(...)`
  * Hash comparison helpers used by state computation. `src/codeintel/build/hash_evaluator.py :: evaluate_hash_state(...)` / `HashEvaluation`
* Session-scoped caching:

  * Caches input hashes and manifests for a run. `src/codeintel/build/session.py :: BuildSession`
* State computation + validation:

  * Unified state types. `src/codeintel/build/state_types.py :: TargetState` / `BuildState`
  * Computes state and propagates dependency blocking. `src/codeintel/build/state_computer.py :: StateComputer.compute_all()`
  * Validator wrapper that builds session/computer and returns `BuildState`. `src/codeintel/build/state.py :: StateValidator.validate()`
* Execution policy consolidation:

  * Combines run-level options and target execution hints into a resolved policy. `src/codeintel/build/execution_policy.py :: ExecutionPolicy`

## 2.2 `src/codeintel/build/assets/`

* Asset fingerprinting (dataset/artifact “version” hashes):

  * Fingerprint mode + input dataclasses + policy compute methods. `src/codeintel/build/assets/fingerprinting.py :: FingerprintPolicy` / `FingerprintMode` / `TableVersionInput` / `ArtifactVersionInput`
* Asset impact computation (build-facing):

  * Entry surface referenced by CLI build handler. `src/codeintel/build/assets/impact.py :: compute_impact(...)` (import site: `src/codeintel/cli/handlers/build.py :: build_impact_handler(...)`)
* Asset emission helpers:

  * Emits/serializes asset-related records. `src/codeintel/build/assets/emitter.py :: (module-level API)`

## 2.3 `src/codeintel/build/catalogs/`

* Canonical catalog loading/building for contracts and targets:

  * Loads/builds dataset-contract catalog and output-target catalog. `src/codeintel/build/catalogs/canonical.py :: load_contract_catalog(...)` / `load_target_catalog(...)`
* Catalog hashing:

  * Computes global catalog hash inputs. `src/codeintel/build/catalogs/hashing.py :: compute_global_catalog_hash(...)` / `CatalogHashInputs`
* Target serialization:

  * JSON serde for `OutputTarget`. `src/codeintel/build/catalogs/target_serde.py :: output_target_to_json_obj(...)` / `output_target_from_json_obj(...)`

## 2.4 `src/codeintel/build/spec/`

* BuildSpec primitives:

  * Serialized contract types: targets, datasets, semantic pointer. `src/codeintel/build/spec/primitives.py :: BuildSpec` / `TargetSpec` / `DatasetSpec`
* BuildSpec compilation:

  * Compiles `BuildSpec` from the canonical target catalog + schema provider; ensures a buildspec hash. `src/codeintel/build/spec/compile.py :: compile_buildspec(...)` / `BuildSpecCompileOptions`
* Serde + hash enforcement:

  * BuildSpec hash enforcement helper. `src/codeintel/build/spec/serdes.py :: ensure_buildspec_hash(...)`

## 2.5 `src/codeintel/build/schemas/`

* Build-owned schema service wiring:

  * Wires schema service. `src/codeintel/build/schemas/service.py :: get_schema_service()`
* Schema inference / provider implementations:

  * Provider and inference service. `src/codeintel/build/schemas/inference_service.py :: SchemaInferenceService` / `HamiltonSchemaProvider`
* Schema indexing and derivation metadata:

  * `src/codeintel/build/schemas/schema_index.py :: SchemaIndex` / `SchemaDerivation`
* Schema manifest compilation + diffs:

  * `src/codeintel/build/schemas/compile.py :: SchemaManifestRequest`, `src/codeintel/build/schemas/diff.py :: ManifestDiffResult`

## 2.6 `src/codeintel/build/exports/`

* Export orchestration types:

  * `src/codeintel/build/exports/runner.py :: ExportOptions` / `ExportRunner`
  * `src/codeintel/build/exports/common.py :: ExportCallOptions` / `ExportTarget`

## 2.7 `src/codeintel/build/serving/`

* Serving snapshot publication:

  * `src/codeintel/build/serving/publisher.py :: publish_serving_snapshot(...)` / `PublishServingSnapshotRequest`
* Serving snapshot manifest model:

  * `src/codeintel/build/serving/manifest.py :: ServingSnapshotManifest`
* Semantic compilation for serving:

  * `src/codeintel/build/serving/semantic_compile.py :: CompiledSemanticRegistry`
* Search index builders:

  * `src/codeintel/build/serving/search_index.py :: build_search_documents_table(...)` / `ensure_fts_index(...)`

## 2.8 `src/codeintel/build/hamilton/` (expanded)

* Driver factory / composition root:

  * `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)`
* Runtime + mapping container:

  * `src/codeintel/build/hamilton/runtime.py :: HamiltonRuntime`
* Execution:

  * `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor`
* Planning:

  * `src/codeintel/build/hamilton/planner.py :: compute_plan(...)` / `HamiltonBuildPlan`
* Naming/tagging/tag index:

  * `src/codeintel/build/hamilton/naming.py :: target_node(...)` / `materialize_node(...)`
  * `src/codeintel/build/hamilton/tagging.py :: tag_materialize(...)` / `tag_compute(...)`
  * `src/codeintel/build/hamilton/tag_index.py :: TagIndex.from_runtime(...)`
* Native module discovery + target metadata:

  * `src/codeintel/build/hamilton/native/discovery.py :: load_native_modules()`
  * `src/codeintel/build/hamilton/native/target_spec_helpers.py :: register_output_targets(...)`
* Support node generation:

  * `src/codeintel/build/hamilton/nodes/support_factory.py :: build_support_module(...)`
* Materializers and “save_to” decorator:

  * `src/codeintel/build/hamilton/save_to.py :: SaveToObjectMetadataDecorator`
  * `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py :: DuckDBRowsSaver`
  * `src/codeintel/build/hamilton/materializers/artifact_saver.py :: FileArtifactSaver`

## 2.9 Composition roots that invoke build (CLI)

* CLI run / plan / status:

  * `src/codeintel/cli/handlers/build.py :: build_run_handler(...)`
  * `src/codeintel/cli/handlers/build.py :: build_plan_handler(...)`
  * `src/codeintel/cli/handlers/build.py :: build_status_handler(...)`

# 3) Hamilton subsystem map (deep dive)

## 3.1 Composition root and runtime container

* **Driver construction path**:

  * Loads native modules and builds a Driver, then generates support nodes and re-builds the Driver including support module: `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)`, `_build_support_graph_and_module(...)`
* **Native target discovery is tag-driven**:

  * Targets are discovered from nodes tagged as `node_type == materialize` with a `target` tag: `src/codeintel/build/hamilton/introspect.py :: target_names_from_nodes(...)`
* **Target metadata registry must match DAG-discovered targets**:

  * Resolves registered `OutputTarget` specs for discovered target names and raises on missing/extra registrations: `src/codeintel/build/hamilton/native/target_spec_helpers.py :: resolve_registered_targets(...)`

## 3.2 Target dependency graph derivation

* **Dependency edges are derived from Hamilton’s function graph** by walking upstream dependencies from each materialize node and collapsing intermediate nodes until upstream materialize nodes are encountered: `src/codeintel/build/hamilton/introspect.py :: derive_target_dependencies(...)`, `_direct_target_dependencies(...)`
* **TargetGraph is built by cloning OutputTarget metadata and replacing dependencies with derived edges**: `src/codeintel/build/hamilton/introspect.py :: target_graph_from_hamilton(...)`, `_clone_target_with_dependencies(...)`

## 3.3 Execution runtime

* **Executor computes closure from TargetGraph**, maps targets to node names, executes driver, and categorizes outcomes based on `TargetRunRecord.status`: `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._run_with_state(...)`, `_categorize_outputs(...)`
* **Driver execution inputs** always include `env` and `graph`: `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._execute_dag(...)`

## 3.4 Hooks/adapters and persistence

* **BuildRunWriter** persists run lifecycle, per-target records, and node telemetry (best-effort): `src/codeintel/build/hamilton/run_writer.py :: BuildRunWriter.start_run(...)`, `save_run_targets(...)`, `save_run_nodes(...)`, `complete_run(...)`
* **Contract enforcement hook** activates a `ContractEnforcer` based on `target` node tags (per-node): `src/codeintel/build/hamilton/hooks/contract_hook.py :: ContractEnforcementHook.pre_node_execute(...)`
* **Parallel adapter** supports a threadpool backend and injects per-thread gateways: `src/codeintel/build/hamilton/adapters/parallel.py :: ThreadPoolAdapter._maybe_inject_thread_gateway(...)`

# 4) Core runtime concepts & types (type map)

(Phase 1 + Phase 2 content preserved; key Phase 3-relevant additions are below.)

## 4.1 Core orchestration types referenced directly in Phase 3 sections

* Target dependency closure computation (build + Hamilton executor and planner):

  * `src/codeintel/build/targets.py :: TargetGraph.topological_order(...)`
* Build plan and plan entries:

  * `src/codeintel/build/hamilton/planner.py :: HamiltonBuildPlan`
  * `src/codeintel/build/hamilton/planner.py :: PlanEntry`
* Hash evaluation result:

  * `src/codeintel/build/hash_evaluator.py :: HashEvaluation`
* Manifest record type used by both plan and skip checks:

  * `src/codeintel/core/build_manifest.py :: OutputManifest`
* Target execution record (DAG-visible target node output):

  * `src/codeintel/core/hamilton/records.py :: TargetRunRecord`
* Support-node dataset handle that gates downstream loaders:

  * `src/codeintel/build/hamilton/io/dataset_ref.py :: DatasetRef`
* Build-state types used by `build status`:

  * `src/codeintel/build/state_types.py :: TargetState` / `BuildState`

# 5) Data & IO model (as implemented)

(Phase 2 content preserved; Phase 3-relevant IO/record details referenced directly in derived behavior below.)

# 6) Target orchestration model (derived behavior)

## 6.1 “Run” entrypoint and orchestration (CLI → BuildEnv → Hamilton executor)

* **CLI resolves goal targets (explicit targets, module, or all)**:

  * Resolution + validation: `src/codeintel/cli/handlers/build.py :: _resolve_goals(...)`
  * Run entrypoint: `src/codeintel/cli/handlers/build.py :: build_run_handler(...)`
* **Run builds a BuildEnv that carries** `manifest_index`, `force_targets`, `validate_outputs`, `strict_contracts`, and other run inputs:

  * `src/codeintel/build/run_context.py :: BuildRunContext.build_env(...)`
  * `src/codeintel/build/hamilton/env.py :: BuildEnv` (properties: `force_targets`, `manifest_index`, `validate_outputs`, `strict_contracts`)
* **CLI invokes Hamilton build executor**:

  * `src/codeintel/cli/handlers/build.py :: _execute_build_hamilton(...)` constructs `HamiltonBuildExecutor(...)` and calls `executor.run(env=env, targets=execution.goals)`

## 6.2 Dependency closure computation and node selection

* **Closure is computed from the TargetGraph** (dependencies-first topological order for requested targets + transitive deps):

  * `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._compute_closure(...)` calls `TargetGraph.topological_order(targets)`
  * Underlying algorithm aggregates transitive deps then runs Kahn-style topo sort: `src/codeintel/build/targets.py :: TargetGraph.topological_order(...)`
* **Target → node mapping uses the runtime’s `target_to_node` mapping** created by `build_driver`:

  * Mapping population: `src/codeintel/build/hamilton/driver_factory.py :: build_driver(...)` (constructs `t2n = {t.name: target_node(t.name) ...}`)
  * Lookup at execution time: `src/codeintel/build/hamilton/executor.py :: _map_closure_to_nodes(...)` (calls `target_to_node_name(...)`)

## 6.3 Hamilton execution invocation (how the DAG actually runs)

* **Driver executes the closure’s `t__*` nodes as final vars**, and Hamilton resolves and executes dependencies of those nodes automatically:

  * `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._execute_dag(...)`
  * Invocation: `runtime.dr.execute(list(final_vars), inputs={"env": execution_env, "graph": graph})`

## 6.4 Incremental logic: hashing → manifest lookup → compute/skip outcomes

### Planning-time incremental decisions (`build plan`)

* **Plan computes per-target `PlanEntry` from**:

  * Upstream blocking status (computed as it iterates targets in closure): `src/codeintel/build/hamilton/planner.py :: _compute_entry_for_target(...)`
  * Hash evaluation vs stored manifest: `src/codeintel/build/hamilton/planner.py :: _compute_entry_for_target(...)` (calls `compute_hash_evaluation(...)`)
  * Force override: `src/codeintel/build/hamilton/planner.py :: _compute_entry_for_target(...)` (checks `env.is_forced(target_name)`)
* **Hash evaluation loads manifest then computes input hash + dep-hashes**:

  * Manifest lookup: `src/codeintel/build/hash_evaluator.py :: compute_hash_evaluation(...)` (uses `gateway.build.load_manifest(...)` if not in provided index)
  * Input hash includes dependency manifest hashes and yields `dep_hashes` (with `"MISSING"` sentinel): `src/codeintel/build/hashing.py :: compute_input_hash_with_deps(...)`
  * Decision for `missing/current/stale` and reason (`no_manifest`, `up_to_date`, `options_hash_mismatch`, `input_hash_mismatch`): `src/codeintel/build/hash_evaluator.py :: evaluate_hash_state(...)`
* **Plan produces statuses**:

  * `compute` when forced, missing manifest, or stale: `src/codeintel/build/hamilton/planner.py :: _compute_entry_for_target(...)`
  * `skip` when evaluation is current: `src/codeintel/build/hamilton/planner.py :: _compute_entry_for_target(...)`
  * `blocked` when upstream statuses include missing/blocked: `src/codeintel/build/hamilton/planner.py :: _compute_entry_for_target(...)`
  * `missing/no_plugin` when target not found in graph: `src/codeintel/build/hamilton/planner.py :: compute_plan(...)` (KeyError branch)

### Runtime incremental decisions (native target skip + manifest persistence)

* **Runtime skip check is target-local** (native target code calls `NativeTargetExecutor.should_skip()`):

  * `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor.should_skip(...)`
  * Uses `should_skip_native_target(env, target, input_hash, options_hash)`; bypassed when `target.name in env.force_targets`: `src/codeintel/build/hamilton/run_records.py :: should_skip_native_target(...)`
* **On successful native completion, manifest is persisted**:

  * `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor._create_success_record(...)` calls `save_manifest(...)`
  * `save_manifest` writes `OutputManifest(...)` via `env.gateway.build.save_manifest(...)`: `src/codeintel/build/hamilton/run_records.py :: save_manifest(...)`

## 6.5 Dependency propagation and “blocking” behavior surfaces

### Build-state blocking (used by `build status`)

* **`build status` uses StateValidator/StateComputer**, which computes preliminary states from manifests+hashes, then marks targets `blocked` when any dependency is not `current`:

  * Entry: `src/codeintel/cli/handlers/build.py :: build_status_handler(...)` constructs `StateValidator(...)` then calls `validate()`
  * Two-pass implementation: `src/codeintel/build/state_computer.py :: StateComputer.compute_all(...)`, `_propagate_blocking(...)`
  * Blocking reason mapping by dependency status: `src/codeintel/build/state_computer.py :: StateComputer._check_dep_blocking(...)`

### Runtime dependency gating via support nodes (dataset/loader failures)

* **Dataset support nodes require the producing target to provide a DatasetRef**:

  * If a target record lacks the dataset ref, `d__*` raises `ValueError("Missing DatasetRef...")`: `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_dataset_node_function(...)`
* **Loader support nodes require a DatasetRef**:

  * `q__*` raises `TypeError` if upstream `d__*` did not return `DatasetRef`: `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_query_node_function(...)`

# 7) Walkthrough: “request one target” end-to-end trace

> Representative target selected: **`scip`** (ingestion domain), because it exercises: tool execution + artifact materialization + DuckDB row materialization + final `TargetRunRecord` assembly. `src/codeintel/build/hamilton/native/ingestion/scip.py :: SCIP_TARGET_NAME`

## 7.1 CLI invocation to Hamilton executor

1. **User requests target `scip` via CLI run handler**:

* `src/codeintel/cli/handlers/build.py :: build_run_handler(...)` resolves goals list (includes `"scip"`) via `_resolve_goals(...)`.

2. **CLI opens a writable gateway and loads manifest index**:

* Manifest list and index: `src/codeintel/cli/handlers/build.py :: _execute_build_hamilton(...)` (builds `manifest_index = {m.target: m for m in gateway.build.list_manifests(...)}`)

3. **CLI constructs BuildRunContext and BuildEnv** (includes force targets, strict/validate flags, manifest_index):

* `src/codeintel/cli/handlers/build.py :: _execute_build_hamilton(...)` → `BuildRunContext.from_execution_context(...)` → `BuildRunContext.build_env(...)`
* `src/codeintel/build/run_context.py :: BuildRunContext.build_env(...)` returns `BuildEnv(...)`

4. **CLI executes HamiltonBuildExecutor**:

* `src/codeintel/cli/handlers/build.py :: _execute_build_hamilton(...)` calls `HamiltonBuildExecutor.run(env=env, targets=goals)`.

## 7.2 Executor closure and Driver execution

5. **Executor builds a Hamilton runtime and persists run start**:

* Runtime construction: `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._build_runtime(...)` calls `build_driver(...)`
* Run lifecycle start: `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._run_with_state(...)` calls `BuildRunWriter.start_run(...)`

6. **Executor computes dependency closure** for requested targets:

* `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._compute_closure(...)` calls `TargetGraph.topological_order(requested_targets)`
* (TargetGraph topo-order implementation): `src/codeintel/build/targets.py :: TargetGraph.topological_order(...)`

7. **Executor maps closure targets to `t__*` node names** and executes:

* Mapping: `src/codeintel/build/hamilton/executor.py :: _map_closure_to_nodes(...)`
* Execute: `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._execute_dag(...)`

## 7.3 Key Hamilton nodes involved for `scip` (native module)

> Node names below are the function names used by Hamilton; the naming convention for targets is `t__<target>` per `target_node(...)`. `src/codeintel/build/hamilton/naming.py :: target_node(...)`

8. **Tool execution step (`t__scip__run`)**:

* Node: `src/codeintel/build/hamilton/native/ingestion/scip.py :: t__scip__run(env, graph, t__modules) -> ScipRunResult`
* Skip check is performed here via `NativeTargetExecutor`:

  * `src/codeintel/build/hamilton/native/ingestion/scip.py :: t__scip__run(...)` calls `NativeTargetExecutor.for_target(...)` and `executor.should_skip()`
  * Skip logic: `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor.should_skip(...)` → `src/codeintel/build/hamilton/run_records.py :: should_skip_native_target(...)`

9. **Artifact compute nodes and saver metadata nodes**:

* Compute nodes produce artifact paths (or None) and are tagged as compute:

  * `src/codeintel/build/hamilton/native/ingestion/scip.py :: scip__index_artifact(t__scip__run) -> Path | None`
  * `src/codeintel/build/hamilton/native/ingestion/scip.py :: scip__json_artifact(t__scip__run) -> Path | None`
* Each is decorated with `SaveToObjectMetadataDecorator` which produces a **materialization metadata node** named by `materialize_node("artifact.scip_*")`:

  * Decorator: `src/codeintel/build/hamilton/save_to.py :: SaveToObjectMetadataDecorator`
  * Node name builder: `src/codeintel/build/hamilton/naming.py :: materialize_node(...)`
  * Saver used: `src/codeintel/build/hamilton/materializers/artifact_saver.py :: FileArtifactSaver`

10. **Ingestion row preparation (`t__scip__ingest`) and table row compute nodes**:

* Ingest node: `src/codeintel/build/hamilton/native/ingestion/scip.py :: t__scip__ingest(env, graph, t__modules, t__scip__run) -> ScipIngestResult`
* Row compute nodes return row-tuples or None:

  * `src/codeintel/build/hamilton/native/ingestion/scip.py :: scip__symbol_rows(t__scip__ingest) -> tuple[...] | None`
  * `src/codeintel/build/hamilton/native/ingestion/scip.py :: scip__occurrence_rows(t__scip__ingest) -> tuple[...] | None`
* Each is decorated to produce DuckDB row saver metadata nodes named by `materialize_node(<table_key>)`:

  * `src/codeintel/build/hamilton/native/ingestion/scip.py :: scip__symbol_rows(...)` uses `SaveToObjectMetadataDecorator([DuckDBRowsSaver], output_name_=materialize_node(SCIP_SYMBOLS_TABLE_KEY), ...)`
  * Saver: `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py :: DuckDBRowsSaver`

11. **Helper nodes bundle materialization metadata into mappings**:

* Artifact materializations mapping:

  * `src/codeintel/build/hamilton/native/ingestion/scip.py :: scip__materializations(m__artifact__scip_index, m__artifact__scip_json) -> dict[str, MaterializationMetadata]`
* Table materializations mapping:

  * `src/codeintel/build/hamilton/native/ingestion/scip.py :: scip__table_materializations(m__core__scip_symbols, m__core__scip_occurrences) -> dict[str, MaterializationMetadata]`
* Aggregated inputs:

  * `src/codeintel/build/hamilton/native/ingestion/scip.py :: scip__inputs(...) -> ScipMaterializationInputs`

12. **Final target node emits the target’s `TargetRunRecord` (`t__scip`)**:

* Node: `src/codeintel/build/hamilton/native/ingestion/scip.py :: t__scip(env, graph, t__modules, scip__inputs) -> TargetRunRecord`
* Final record is created from artifact materializations (and row_counts from table materializations) via:

  * `src/codeintel/build/hamilton/native/materialization_records.py :: record_from_file_artifact_materializations(...)`

## 7.4 Post-execution persistence and result reporting

13. **Executor categorizes computed/skipped/failed targets from returned outputs**:

* `src/codeintel/build/hamilton/executor.py :: _categorize_outputs(...)` inspects `TargetRunRecord.status` for each closure target.

14. **Executor persists per-target records and completes the run**:

* Save target records: `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._run_with_state(...)` calls `BuildRunWriter.save_run_targets(...)`
* Complete run: `src/codeintel/build/hamilton/executor.py :: HamiltonBuildExecutor._run_with_state(...)` calls `BuildRunWriter.complete_run(...)`

# 8) Extension mechanics (how new behavior is introduced today)

## 8.1 Add a new build target implemented in Hamilton native modules

* **Create a new native module file under the domain directory**:

  * Discovery scans and imports native modules: `src/codeintel/build/hamilton/native/discovery.py :: native_module_paths(...)`, `load_native_modules()`
* **Register the target’s metadata (contract/resources/execution/parameters)**:

  * Use `make_output_target(...)` + `register_output_targets(...)`: `src/codeintel/build/hamilton/native/target_spec_helpers.py :: make_output_target(...)`, `register_output_targets(...)`
  * Validation at registration time includes:

    * table key validation + duplicate checks: `src/codeintel/build/hamilton/native/target_spec_helpers.py :: _resolve_table_schemas(...)`
    * artifact path_template placeholder allowlist: `src/codeintel/build/hamilton/native/target_spec_helpers.py :: _validate_artifact_specs(...)`
* **Define a final “target/materialize” node named `t__<target>`**:

  * Convention is enforced by the name itself (function name) and by `tag_materialize(...)` applying `node_type="materialize"` plus `target` tag: `src/codeintel/build/hamilton/tagging.py :: tag_materialize(...)`
  * Example final node: `src/codeintel/build/hamilton/native/ingestion/scip.py :: t__scip(...)`

## 8.2 Add new table outputs for a target (DuckDB)

* **Declare produced tables in the target contract** (table_keys):

  * `src/codeintel/build/hamilton/native/target_spec_helpers.py :: TargetSpecOptions(table_keys=...)`
* **Implement a compute node that returns either**:

  * an Ibis table expression (for table saver), or
  * row tuples (for row saver),
    and tag as compute: `src/codeintel/build/hamilton/tagging.py :: tag_compute(...)`
* **Attach a saver via `SaveToObjectMetadataDecorator`** to produce a materialization metadata node (dict) with a deterministic `m__*` name:

  * Decorator: `src/codeintel/build/hamilton/save_to.py :: SaveToObjectMetadataDecorator`
  * Materialize-node naming: `src/codeintel/build/hamilton/naming.py :: materialize_node(...)`
  * Row saver example: `src/codeintel/build/hamilton/native/ingestion/scip.py :: scip__symbol_rows(...)` (uses `DuckDBRowsSaver`)
  * Saver implementations:

    * row saver: `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py :: DuckDBRowsSaver.save_data(...)`
    * ibis table saver: `src/codeintel/build/hamilton/materializers/duckdb_saver.py :: DuckDBIbisTableSaver.save_data(...)`
* **Downstream read access is automatic via generated support nodes**:

  * Dataset node `d__<table_key>` (extracts DatasetRef from producing TargetRunRecord): `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_dataset_node_function(...)`
  * Loader nodes `q__*`/`df__*`: `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_query_node_function(...)`, `_create_dataframe_node_function(...)`

## 8.3 Add new artifact outputs for a target (filesystem)

* **Declare artifacts in the target contract**:

  * `ArtifactSpec(name, path_template, description)` in `TargetSpecOptions(artifacts=...)`: `src/codeintel/build/contracts.py :: ArtifactSpec`, `src/codeintel/build/hamilton/native/target_spec_helpers.py :: TargetSpecOptions`
* **Artifact path templates are validated against an allowlist of placeholders**:

  * `src/codeintel/build/hamilton/native/target_spec_helpers.py :: _ALLOWED_ARTIFACT_TEMPLATE_KEYS`, `_validate_artifact_specs(...)`
* **Implement a compute node yielding the artifact payload/path and attach `FileArtifactSaver`**:

  * `src/codeintel/build/hamilton/materializers/artifact_saver.py :: FileArtifactSaver`
  * Example: `src/codeintel/build/hamilton/native/ingestion/scip.py :: scip__index_artifact(...)` (decorated with `SaveToObjectMetadataDecorator([FileArtifactSaver], ...)`)

## 8.4 Add or change target-to-target dependency edges

* **Dependency edges are derived from the Hamilton function graph**, not declared directly in target metadata:

  * Derivation requires each target to have a single materialize node tagged with `node_type="materialize"` and `target=<name>`: `src/codeintel/build/hamilton/introspect.py :: derive_target_dependencies(...)`
  * Dependency edges change when upstream dependencies of the materialize node change (walking upstream until other materialize nodes): `src/codeintel/build/hamilton/introspect.py :: _direct_target_dependencies(...)`

## 8.5 Make a dataset visible as a “semantic view” for serving/consumption

* **Semantic visibility is tag-driven**:

  * `TagIndex.semantic_view_tags()` selects nodes tagged with `output_kind` in `{semantic_view, semantic}` and `mcp_visible == "1"` and requires a `table_key` tag: `src/codeintel/build/hamilton/tag_index.py :: TagIndex.semantic_view_tags(...)`
* **Nodes can attach these secondary tags via `extra_tags`** on tagging helpers:

  * `src/codeintel/build/hamilton/tagging.py :: tag_compute(..., extra_tags=...)` (supports keys like `output_kind`, `semantic_id`, `entity`, `grain`, `mcp_visible`)

# 9) Conventions & invariants (observed)

## 9.1 Naming conventions (stable node IDs)

* Canonical conversions for logical identifiers:

  * `to_node_name(logical_name, prefix=...)` replaces `.` and `/` with `__`, `-` with `_`, strips invalid chars: `src/codeintel/build/hamilton/naming.py :: to_node_name(...)`
* Prefix conventions used across the build DAG:

  * Targets: `t__...` via `target_node(...)`: `src/codeintel/build/hamilton/naming.py :: target_node(...)`
  * Datasets: `d__...` via `dataset_node(...)`: `src/codeintel/build/hamilton/naming.py :: dataset_node(...)`
  * Loaders: `q__...` / `df__...`: `src/codeintel/build/hamilton/naming.py :: query_node(...)`, `dataframe_node(...)`
  * Materializers: `m__...`: `src/codeintel/build/hamilton/naming.py :: materialize_node(...)`
  * Artifacts: `a__...`: `src/codeintel/build/hamilton/naming.py :: artifact_node(...)`

## 9.2 Tagging invariants used for dependency derivation and discovery

* Canonical tag keys and node types are centralized:

  * `TAG_TARGET`, `TAG_TABLE_KEY`, `TAG_ARTIFACT`, `TAG_NODE_TYPE`, etc.: `src/codeintel/core/hamilton/tags.py :: TAG_TARGET` / `TAG_NODE_TYPE`
  * Node type values used by derivation/filtering: `src/codeintel/core/hamilton/tags.py :: NODE_TYPE_MATERIALIZE` / `NODE_TYPE_DATASET` / `NODE_TYPE_ARTIFACT`
* Tagging helpers apply canonical tags:

  * Materialize nodes: `tag_materialize(...)` sets `node_type="materialize"` and optionally `domain/target`: `src/codeintel/build/hamilton/tagging.py :: tag_materialize(...)`
  * Dataset nodes set `table_key` tag: `src/codeintel/build/hamilton/tagging.py :: tag_dataset(...)`
* Dependency derivation assumes:

  * a materialize node has `node_type == materialize` and a non-empty `target` tag: `src/codeintel/build/hamilton/introspect.py :: _is_materialize_node(...)`
  * duplicates (two materialize nodes for same target) are rejected: `src/codeintel/build/hamilton/introspect.py :: derive_target_dependencies(...)`

## 9.3 Registry/DAG alignment invariants (target metadata ↔ DAG)

* The registered target metadata must exactly match DAG-discovered target names:

  * Missing registrations raise: `src/codeintel/build/hamilton/native/target_spec_helpers.py :: resolve_registered_targets(...)`
  * Extra registrations raise: `src/codeintel/build/hamilton/native/target_spec_helpers.py :: resolve_registered_targets(...)`

## 9.4 Contract-related invariants (strict contracts, row counts)

* Strict contracts validate that:

  * artifact-only targets have empty row_counts; table-producing targets require row_counts keys to match `contract.table_keys` exactly; counts must be non-negative: `src/codeintel/build/hamilton/run_records.py :: _validate_strict_row_counts(...)`
* NativeTargetExecutor enforces that `row_counts` returned by compute must match the contract’s table_keys exactly (else emits a failed record):

  * `src/codeintel/build/hamilton/native/executor.py :: NativeTargetExecutor._create_success_record(...)`

## 9.5 Support node invariants (dataset ref must exist)

* Dataset nodes require the producing `TargetRunRecord` to contain the dataset ref; missing causes a raised exception:

  * `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_dataset_node_function(...)` (raises ValueError on missing `DatasetRef`)
* Loader nodes require a `DatasetRef` type from upstream dataset node:

  * `src/codeintel/build/hamilton/nodes/support_factory.py :: _create_query_node_function(...)` (raises TypeError if not `DatasetRef`)

## 9.6 Artifact path-template invariants

* Artifact path templates may only use placeholders from a fixed allowlist:

  * `src/codeintel/build/hamilton/native/target_spec_helpers.py :: _ALLOWED_ARTIFACT_TEMPLATE_KEYS`
  * Enforcement: `src/codeintel/build/hamilton/native/target_spec_helpers.py :: _validate_artifact_specs(...)`

# 10) Glossary (project-specific vocabulary

* **Target**: a named build unit with metadata (contract/resources/execution/params) and derived dependencies; materialized as a Hamilton node `t__<name>` producing `TargetRunRecord`. `src/codeintel/build/targets.py :: OutputTarget`, `src/codeintel/build/hamilton/native/ingestion/scip.py :: t__scip(...)`
* **TargetGraph**: dependency graph of `OutputTarget` objects used for closure/topological order (and for “blocked” propagation in state computations). `src/codeintel/build/targets.py :: TargetGraph`, `src/codeintel/build/targets.py :: TargetGraph.topological_order(...)`
* **Contract / OutputContract**: declaration of produced tables (table_keys/schemas) and artifacts for a target. `src/codeintel/build/contracts.py :: OutputContract`
* **Table key**: fully-qualified dataset identifier (schema.table) used in contracts and tags and for loader node generation. `src/codeintel/build/hamilton/native/target_spec_helpers.py :: _validate_table_key(...)`, `src/codeintel/core/hamilton/tags.py :: TAG_TABLE_KEY`
* **ArtifactSpec**: artifact declaration (name + optional path_template) used in contracts; validated for allowed template placeholders. `src/codeintel/build/contracts.py :: ArtifactSpec`, `src/codeintel/build/hamilton/native/target_spec_helpers.py :: _validate_artifact_specs(...)`
* **BuildEnv**: frozen input bundle provided to Hamilton execution containing gateway/snapshot/paths/providers/config/settings plus flags (force_targets, strict_contracts, validate_outputs) and manifest_index. `src/codeintel/build/hamilton/env.py :: BuildEnv`
* **OutputManifest**: stored record of a target’s computation, keyed by (repo, commit, target), including input_hash/options_hash; used for skip/planning/state. `src/codeintel/core/build_manifest.py :: OutputManifest`
* **Input hash**: 16-hex digest computed from engine version + repo/commit + target name + dependency manifest hashes (+ optional options/file-state hash). `src/codeintel/build/hashing.py :: compute_input_hash_with_deps(...)`
* **PlanEntry / HamiltonBuildPlan**: dry-run planning output describing per-target status (`compute/skip/missing/blocked`) and reason plus hashes and deps. `src/codeintel/build/hamilton/planner.py :: PlanEntry`, `HamiltonBuildPlan`
* **TargetRunRecord**: per-target execution result emitted by the DAG-visible target node; includes status, hashes, row_counts, datasets (DatasetRef), and artifacts. `src/codeintel/core/hamilton/records.py :: TargetRunRecord`
* **DatasetRef**: typed handle referencing a produced dataset (table_key + repo/commit + optional row_count) extracted from `TargetRunRecord` and used by loaders. `src/codeintel/build/hamilton/io/dataset_ref.py :: DatasetRef`
* **Support nodes**: generated nodes for datasets and loaders (e.g., `d__*`, `q__*`, `df__*`) constructed from contracts + TargetGraph and attached as a support module. `src/codeintel/build/hamilton/nodes/support_factory.py :: build_support_module(...)`
* **Materialization metadata node**: dict output produced by `SaveToObjectMetadataDecorator`/DataSaver nodes (often named `m__*`) describing persisted writes. `src/codeintel/build/hamilton/save_to.py :: SaveToObjectMetadataDecorator`, `src/codeintel/build/hamilton/boundary_types.py :: MaterializationMetadata`
