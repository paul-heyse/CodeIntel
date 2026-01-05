
According to a document from **December 26, 2025** (your `architecture_ph6.md`), you’ve already achieved the *right* north star: “the DAG is the truth,” with stable naming, tag-driven target compilation, saver-derived IO surfaces, manifest-driven incrementality, and a generated support module that makes reads/writes DAG-visible. The biggest remaining wins are now about **eliminating the last “two sources of truth,” collapsing bespoke plumbing into Hamilton-native mechanisms, and turning your derived metadata into a single, reusable “catalog” that every surface (build, export, serving, MCP) consumes**.

Below are the **highest-leverage opportunities** I see to tighten integration, dramatically consolidate code, and harden extensibility.



## 3) Make saver-derived outputs the *only* output inventory; stop declaring outputs twice

You already derive outputs from saver tags (`derive_target_outputs_from_savers(...)`) and use them to generate the support module. You also have a separate “OutputContract” compilation path that resolves table schemas and artifact specs during `OutputTarget` compilation.

That’s still two “truths” about outputs:

* what the target *claims* it produces (contract),
* what the DAG *actually produces* (savers).

### Breaking-change recommendation

Make “what gets produced” **100% saver-derived**:

* Output identity (table_key, artifact_name, path template) is defined only by saver decorators.
* Contracts should only supply **validation metadata** (schema, constraints, semantics), keyed by the output identity.

So `OutputTarget` becomes:

* `resources/execution/parameters/spec_version` from target-anchor tags
* `outputs` from saver-tag introspection (not from target spec payloads)
* `schemas` from a registry keyed by `table_key` (not re-declared in per-target payload)

This will:

* remove an entire class of “declared vs actual” drift,
* simplify validation (you validate that saver-tag identity exists + schema is known),
* and make adding outputs “just add a saver decorator.”

---

## 4) Unify manifest-driven incrementality with Hamilton caching for a massive code shrink

Your architecture states incrementality is “manifest + input_hash (+ options_hash)” and that the same primitive is reused in planning, tool gating, and materializers.

You also already expose a Hamilton cache toggle in `build_driver(...)`, but it’s configured conservatively (`default_behavior="disable"`, etc.).

### Major opportunity

Instead of maintaining *parallel* skip logic in:

* planner/explain,
* native executor gating,
* materializer gating,

…you can consolidate around Hamilton’s caching adapter *as the skip engine*, and treat manifests as either:

* the cache metadata store, or
* an external “persistence layer” that the caching adapter writes to.

Hamilton caching provides:

* inspectable cache keys/data versions,
* code + data versioning hooks,
* extensible hashing for custom types,
* operational patterns like “opt-in caching + JSONL audit trail.”

**High-payoff design**

* Define a canonical “data version” for each target based on your existing `input_hash`.
* Register hashing for your key internal types (e.g., snapshot refs, tool digests) so versions are stable and deterministic.
* Let caching decide recompute/skip per node; then manifests become a reporting artifact rather than the primary control plane.

This can eliminate a *lot* of bespoke “skip propagation” logic while preserving the exact semantics you want.

---

Below is the **narrative pairing** for (3) and (4): what the system *becomes*, how the mental model shifts, and how the execution/serving/contracts story reads end-to-end once these changes land.

---

## 3) Saver-derived outputs become the only output inventory

### Before (what’s “weird” today)

You have **two independent declarations of “what outputs exist”**:

1. **Reality**: the DAG contains `m__*` saver nodes whose tags encode `table_key` / `artifact` + `output_role`, and those nodes are what actually write.
2. **Declared contract**: targets carry an `OutputContract`/`ArtifactSpec` that separately lists outputs (often with positional alignment hacks: “json_schema_ids[i] aligns to tables[i]”).

Even if you reconcile them, it means every subsystem has a choice:

* sometimes it trusts the contract list,
* sometimes it trusts the saver-derived list,
* sometimes it computes a “derived_outputs” intermediate and compares.

That’s the duplication that forces constant glue logic and mismatch checks.

### After (new truth model)

There is exactly one answer to “what outputs exist?”:

> **Outputs exist iff a DataSaver node exists and advertises them via tags.**

So output identity and inventory are *not* a property of “targets” anymore; they’re a property of **DAG nodes**.

Concretely:

* The `DagCatalog` compilation step scans the DAG and collects:

  * **contract outputs**: `hamilton.data_saver=True` AND `output_role="contract"` AND (`table_key` XOR `artifact`)
  * it builds `catalog.table_outputs` and `catalog.artifact_outputs` and per-target `io.writes`
* Every downstream consumer uses that catalog inventory. There is no “declared output list” anywhere else.

### What contracts turn into

“Contracts” stop being an alternate inventory declaration. They become **metadata keyed by output identity**, not a parallel list.

So the question shifts from:

* “What tables does this target declare?”
  to:
* “For this table_key (which is already declared by the saver), what validation/schema/semantic metadata applies?”

Practically:

* The authoritative list of tables/artifacts is saver tags.
* The authoritative schema metadata is:

  * table registry (explicit schemas) and/or
  * inference service (inferable schemas),
  * with optional per-output overrides attached **as tags on the saver node** (or via a keyed registry, but the more “DAG-derived” approach is saver tags).

### Enforcement becomes simpler and more correct

Write enforcement no longer tries to enforce “you may only write the outputs listed in OutputContract,” because OutputContract is gone.

Instead:

* During execution of a target, the hook activates an allowlist derived from:

  * `catalog.targets[target].io.contract_writes()`
* Materializers call `validate_table_write(table_key)` / `validate_artifact_write(name)`
* Violations are now framed as:

  * “You attempted to write an output that is not declared by a saver tag for this target.”

This flips the direction: instead of contracts describing outputs and savers trying to match, savers describe outputs and enforcement ensures runtime writes stay within that declared set.

### Validation “tightens around the truth”

Once saver tags define inventory, validation becomes sharper:

* uniqueness constraints (table_key/artifact_name must be globally unique among contract outputs)
* artifact outputs must include path templates
* produced tables must have schemas (explicit registry or inferable)
* orphan savers are invalid (saver claims `target=X` but is not actually on a path to `t__X`)

So the validator stops doing “mismatch between contract list and saver list” checks — because there is no contract list — and instead becomes a **hard gate on saver correctness**.

### How this feels in daily development

Adding a new output becomes trivially mechanical and drift-proof:

1. Add a saver node for it (or update an existing saver) with:

   * `output_role="contract"`, `table_key="..."` (or `artifact="..."` + template)
2. Ensure schema exists (registry or inference).
3. Everything else updates automatically:

   * catalog inventory, support nodes, serving compilation, buildspec, enforcement, manifests.

No one adds outputs in two places anymore. The DAG is the contract.

---

## 4) Manifest-driven incrementality collapses into Hamilton caching

### Before (what’s “weird” today)

You essentially have **two recomputation decision engines**:

1. Your manifest system:

   * computes input hashes (and sometimes options hashes),
   * decides skip/compute at target level,
   * propagates skip/blocked reasons through closure,
   * writes manifests and decision traces.
2. Hamilton’s caching (present but conservative or secondary):

   * can cache nodes, but it’s not the authoritative control plane.

Even if they roughly align, they are conceptually distinct:

* your planner “decides” whether to run,
* executor performs gating,
* manifests are partially control and partially audit.

This is expensive in code and fragile in semantics: you must maintain consistent hashing rules and skip propagation logic outside Hamilton.

### After (new truth model)

There is exactly one recomputation decision engine:

> **Hamilton cache decides compute/skip.**

Everything else becomes *observability* and *analytics*.

Your manifests stop being a skip-control mechanism and become a **record of cache events**:

* hit
* miss
* store

So:

* planner no longer “proves skip” — it just requests closure.
* executor no longer “gates” — it executes requested targets and lets cache choose which nodes compute.
* decision trace becomes a derived view of cache behavior + runtime timing.

### What “incrementality” becomes

Incrementality shifts from “target-level planning logic” to “node-level cache semantics” (with target-level aggregation).

Key changes in mental model:

* A target execution is “execute `t__X`”, and the cache decides what upstream nodes are reused.
* You no longer need explicit “skip propagation” logic, because the cache naturally yields reuse wherever input versions match.

### The keystone: a single cache key/version strategy

Previously you had:

* input_hash logic,
* options_hash logic,
* sometimes ad-hoc digests and per-target heuristics.

After unification:

* a single `CacheKeyResolver` produces versions for nodes based on:

  * upstream dependencies’ versions,
  * environment/tool digests (repo snapshot, tool versions),
  * node code hash where appropriate,
  * config toggles that materially affect outputs.

This gives you one deterministic key path:

* used by Hamilton caching,
* recorded in manifests,
* used for later debugging/diffing.

No more “manifest hash != cache hash” mismatches because there isn’t a second hash system.

### How manifests and “decision traces” change

Manifests become “what happened” rather than “what should happen.”

Instead of recording:

* compute/skip/blocked reasons derived by planner logic

You record:

* cache hit/miss/store per node (and optionally per target aggregated)
* path/size/timing metadata

Decision trace then becomes:

* a view that says “target X was fast because 90% of its nodes were cache hits”
* or “target X recomputed because node Y’s version changed due to tool digest change”

That’s much closer to how production systems are debugged: by tracing cache behavior, not by trusting an external skip oracle.

### What this feels like operationally

Running the build is simpler:

* You request targets.
* Hamilton executes.
* Cache determines what recomputes.
* Manifests are written as cache events occur.

This reduces moving parts:

* less bespoke planner logic,
* less executor gating,
* less “incremental correctness” code to maintain.

---

## The combined (3)+(4) end-to-end story

Once both land, the system becomes extremely “single-source-of-truth” in two dimensions:

1. **Outputs**: defined by saver tags (DAG truth)
2. **Incrementality**: defined by cache keys + cache behavior (execution truth)

And those truths reinforce each other:

* Saver nodes are boundary points where durable outputs are written.
* Cache keys version the computation that leads to those saver nodes.
* Manifests become a durable record linking:

  * target → saver outputs produced
  * node versions → cache hits/misses
  * outputs → schema/contract metadata keyed by output identity

So the runtime becomes:

* “execute DAG with caching”
* “output inventory is what savers declare”
* “schema/contract metadata is keyed by output identity”
* “manifests are an audit log of execution/cache behavior”

---

## What gets deleted conceptually (the “why this shrinks code” narrative)

### (3) deletes “inventory reconciliation code”

* no more “derived outputs” layer
* no more “contract list vs saver list mismatch” checks
* no more positional alignment of metadata tuples to declared outputs
* no more per-target output declaration surfaces

### (4) deletes “bespoke skip propagation code”

* no more target-level “should_skip” machinery
* no more planner-level skip/block reasoning as the control plane
* no more duplicative hashing pipelines that exist only to power a second recomputation engine

What remains is cleaner:

* compile a catalog from DAG tags
* execute with caching
* record outcomes

---

## What you, as a developer, now “do” differently

* To add a table: add/adjust a saver (tags) + add schema/registry entry if non-inferable.
* To change incrementality semantics: adjust cache key resolver inputs (tool digests/config) rather than patching skip logic in multiple places.
* To debug “why didn’t it recompute”: look at cache versions/hits, not planner skip reasons.

That’s the core narrative: you’re replacing two duplicated declarative planes (output inventory + skip logic) with two execution-native truth planes (saver tags + cache behavior), and turning everything else into deterministic derived artifacts.



Below is a **repo-concrete, breaking-change–friendly** implementation plan for:

> **(3) Make saver-derived outputs the only output inventory; stop declaring outputs twice**

Assumption: **(1)** landed (Hamilton graph + tags authoritative; `DagCatalog` exists and is consumed everywhere) and **(2)** landed (support nodes are parameterized/static; no dynamic support_factory inventories). Where Phase6 still references `OutputTarget/TargetGraph/OutputContract`, treat those as **legacy surfaces to delete or downgrade** in this phase.

---

# 0) End-state contract (what must be true)

## 0.1 Output identity + inventory

* **Only** `SaveToObjectMetadataDecorator`-emitted DataSaver nodes (`tags["hamilton.data_saver"]=True`) define **what outputs exist**:

  * `tags["output_role"] in {"contract","internal"}`
  * `tags["target"]=<target_name>`
  * exactly one of `tags["table_key"]` or `tags["artifact"]`
  * artifacts require `tags["artifact_path_template"]`
* “Produced outputs” are compiled once into `DagCatalog.table_outputs` and `DagCatalog.artifact_outputs` (and per-target IO surfaces) and are **the** inventory used by:

  * planner closure/explain
  * contract/write enforcement
  * expected outputs + run record strictness
  * buildspec compilation
  * schema index + inference graph
  * serving compilation

## 0.2 Contracts become metadata-only (no output lists)

* There is **no second list** of outputs anywhere (no `OutputContract.tables/artifacts`, no “declared outputs” in target specs).
* Schema + validation metadata are keyed by output identity (`table_key` / `artifact_name`) and resolved via:

  * **table schemas**: `codeintel.core.schemas.table_registry` (+ optional inference)
  * **per-table contract overrides** (json schema id / export filenames / owner / profile): derived from **tags on saver nodes** OR from a dedicated keyed registry (choose one; below plan prefers tags to stay DAG-derived).

---

# 1) Output inventory compilation: make `DagCatalog` the sole output index

### Files MODIFIED

* `src/codeintel/build/hamilton/dag_catalog_compiler.py` (from (1))

  * Ensure output derivation is **exclusively** from DataSaver tags (contract role) and does **not** consult any per-target “declared outputs”.
  * Emit:

    * `OutputDescriptor(kind="table", key=table_key, role="contract", producer_target=tags["target"], saver_node=node.name, sink=tags["hamilton.data_saver.sink"], tags=<pass-through>)`
    * `OutputDescriptor(kind="artifact", key=artifact_name, role="contract", ..., artifact_path_template=tags["artifact_path_template"])`
  * Enforce global uniqueness:

    * `table_key` unique across contract outputs
    * `artifact_name` unique across contract outputs
  * Populate each `TargetDescriptor.io.writes` from saver nodes by `tags["target"]==target_name` and `output_role=="contract"` (and optionally also track internal writes separately if you keep the concept).

### Files DELETED (post-(1) cleanup; output inventory duplication)

* `src/codeintel/build/hamilton/introspect.py`

  * Specifically, delete all output-inventory duplication:

    * `DerivedTargetOutputs`
    * `derive_target_outputs_from_savers()`
    * `derive_target_outputs()`
    * `expected_*` style helpers that read derived outputs
  * If `introspect.py` still contains non-output utilities you need, split into:

    * `dag_walk.py` (generic reachability utilities)
    * and delete the output inventory portion.

> Rationale: once `DagCatalog` is the compiled view, `derive_target_outputs_from_savers()` is a second inventory API and must disappear.

---

# 2) Remove `OutputContract` as an output list (stop declaring outputs twice)

## 2.1 Delete the build-layer contract type (or downgrade it to metadata-only)

### Option A (recommended, maximal deletion): **remove `OutputContract` entirely**

Outputs already have identity via saver tags; table schemas already exist in `core.schemas`. Keep “dataset contract” concerns inside `codeintel.core.schemas.*` rather than `codeintel.build.contracts`.

#### Files DELETED

* `src/codeintel/build/contracts.py`

#### Files MODIFIED (remove exports)

* `src/codeintel/build/__init__.py`

  * Remove lazy exports: `OutputContract`, `ArtifactSpec`, `EMPTY_CONTRACT`.
  * Update package docstring: replace “OutputContract is single source of truth” with:

    * “DataSaver tags are the output inventory”
    * “schemas/contract metadata are keyed by output identity”

#### Files MODIFIED (remove dependencies on OutputContract)

* `src/codeintel/build/targets.py`

  * If still present post-(1), delete `OutputTarget.contract` field entirely; targets (or descriptors) no longer embed output lists.
  * In practice post-(1): this file should already be on the chopping block; ensure any residual uses are removed.

* `src/codeintel/build/errors.py`

  * Rewrite actionable hints:

    * “Add to OutputContract” → “add/adjust saver tags (`table_key`/`artifact`, `output_role`)” and/or “register schema in table registry”
  * Update `ContractViolationError` messaging to reference **DAG-declared outputs** (not “contract object”).

### Option B (transitional): keep `OutputContract` as metadata-only

If you need a temporary adapter for downstream code, redefine it as:

* `OutputContractMetadata(owner, family, retention_policy, validation_profile, …)`
  …and ensure it contains **no** `tables`/`artifacts` inventories.

(Plan below assumes Option A for maximal consolidation.)

---

# 3) Contract enforcement rewrite: enforce against catalog output inventory (not a contract list)

Current enforcement uses `OutputTarget.contract.table_keys` / `.artifacts`. That must be replaced with **catalog-derived allowlists**.

### Files MODIFIED

* `src/codeintel/build/hamilton/contracts/enforcement.py`

  * Replace `OutputTarget` dependency with a pure runtime write-context:

    * `ContextVar[str | None] current_target_name`
    * `ContextVar[bool] strict`
    * `ContextVar[frozenset[str]] allowed_table_keys`
    * `ContextVar[frozenset[str]] allowed_artifacts`
  * New API:

    * `activate(target_name: str, *, strict: bool, allowed_tables: frozenset[str], allowed_artifacts: frozenset[str])`
    * `deactivate()`
    * `validate_table_write(table_key: str)` checks membership in `allowed_tables` when strict
    * `validate_artifact_write(artifact_name: str)` checks membership in `allowed_artifacts` when strict

* `src/codeintel/build/hamilton/hooks/contract_hook.py`

  * Rename conceptually to `WriteEnforcementHook` (file rename optional; leaving filename is fine but update docstrings).
  * Replace ctor `(graph: TargetGraph)` with `(catalog: DagCatalog)` (post-(1) this should already be the pattern).
  * In `pre_node_execute`:

    * read `target_name = node.tags["target"]`
    * resolve `writes = catalog.targets[target_name].io.contract_writes()`
    * compute allowlists:

      * `allowed_tables = frozenset(o.key for o in writes if o.kind=="table")`
      * `allowed_artifacts = frozenset(o.key for o in writes if o.kind=="artifact")`
    * call `ContractEnforcer.activate(target_name, strict=strict, allowed_tables=..., allowed_artifacts=...)`
  * `post_node_execute`: `ContractEnforcer.deactivate()`

* `src/codeintel/build/hamilton/contracts/enforced_gateway.py`

  * No semantic change; it continues to call `ContractEnforcer.validate_table_write(table_key)`, which now consults catalog-derived allowlists instead of `OutputContract`.

### Files MODIFIED (materializers)

* `src/codeintel/build/hamilton/materializers/duckdb_saver.py`
* `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py`
* `src/codeintel/build/hamilton/materializers/artifact_saver.py`

  * Keep the enforcement call (`ContractEnforcer.validate_*`) but remove any dependency on `OutputContract`-typed targets.
  * (Post-(1)) these materializers should no longer carry `graph: TargetGraph`; they should rely on `env + catalog` or `env + runtime hashing` inputs. This is orthogonal but must be consistent.

---

# 4) Validation gate: remove “contract mismatch” checks, add saver-centric hardening

You already validate saver tags and unknown schemas in `validate.py`; it currently also contains **legacy mismatch logic** comparing derived saver outputs to `base_graph` contracts. That must go away, and you should **tighten saver correctness** because saver tags become authoritative.

### Files MODIFIED

* `src/codeintel/build/hamilton/validate.py`

  * Delete:

    * `_derived_outputs_mismatch_issues(...)` (compares derived outputs vs contract outputs)
    * any callsites passing `base_graph: TargetGraph`
  * Keep and promote saver-derived checks as *the* inventory validation:

    * `_collect_saver_outputs(...)` remains authoritative for “produced contract outputs”
    * `_unknown_schema_issues(...)` stays (schema must exist for all contract tables)
  * Add a new **reachability hardening check** (high leverage now that tags define producer):

    * For each contract DataSaver node `s`, ensure there exists at least one downstream materialize node `t__` with `tags["target"] == s.tags["target"]`.
    * Implementation: build reverse adjacency once: `dependents[node] = {nodes that depend on node}`; BFS from saver node; stop if matching anchor encountered; error if none.
    * This catches mis-tagged savers that “claim” to belong to a target they do not actually feed.

---

# 5) Schema subsystem refactor: stop reading output inventory from per-target contracts

You currently build schema derivations by iterating `target.contract.table_keys` and using `target.contract.get_table(...)` as override source. That entire pattern is the “outputs declared twice” problem in schema land.

## 5.1 Build schema index from catalog output keys + table registry (+ inference)

### Files MODIFIED

* `src/codeintel/build/schemas/schema_index.py`

  * Change `build_schema_index(system: TargetSystem, ...)` to iterate produced tables from `system.catalog.table_outputs` (or `system.catalog.targets[*].io.contract_writes()`), not from `system.graph.all_targets`.
  * Replace “explicit_override via OutputContract.get_table” with “explicit_registry via `codeintel.core.schemas.table_registry.get_table_schema(table_key)`”.
  * New derivation logic:

    * `inferable = inference_service.inferable_table_keys(catalog=system.catalog)` (signature change below)
    * for each produced `table_key`:

      * if `table_key in inferable`: `kind="inferred_relation"`; `override_schema=None` (or optionally use registry schema as fallback)
      * else: require `table_registry.get_table_schema(table_key)` is present; set `kind="explicit_registry"`; attach schema if you still store it inside derivation
    * if missing explicit schema for non-inferable: raise `ValueError` (or emit validation issues earlier; choose one consistent gate—prefer validator gate + exception here for deterministic failure)

* `src/codeintel/build/schemas/inference_service.py`

  * Replace `_producers_by_table_key(graph: TargetGraph)` with `_producers_by_table_key(catalog: DagCatalog)`:

    * `producers = {table_key: [desc.producer_target]}` from `catalog.table_outputs`
  * Update inference requirement scanning: if compute nodes previously depended on `"graph"`, they now depend on `"catalog"` (post-(1)); adjust `_inference_requirements` accordingly.

* `src/codeintel/build/schemas/provider_unified.py`

  * Update `declared_schema_provider()` exclusion set:

    * previously `exclude_table_keys = service.system.all_table_keys` (computed from contract)
    * now `exclude_table_keys = frozenset(service.system.catalog.table_outputs)` (or `system.all_table_keys` computed from catalog)

* `src/codeintel/build/schemas/registry.py`

  * Update docstring fallback chain: remove “Target-declared schemas from OutputContract.tables”; replace with:

    * “explicit table registry schemas for produced outputs”
    * “Hamilton inference for inferable nodes”
    * “declared source schemas last”

---

# 6) Dataset contract overrides: migrate away from OutputContract-based metadata alignment

Right now `build/schemas/contract_service.py` derives overrides from `OutputContract` using *positional alignment* (`json_schema_ids[i]` aligns to `tables[i]`). Once output inventory isn’t in `OutputContract`, this has to become **keyed by `table_key`**.

## Preferred approach (max DAG-derived): put per-table metadata on saver tags

### Files MODIFIED

* `src/codeintel/build/hamilton/save_to.py`

  * Extend tag-only kwargs (`_TAG_ONLY_KWARGS`) and `_build_saver_tags(...)` to allow optional per-output metadata tags, **provided via `value(...)` so they exist at compile time**:

    * `ci.json_schema_id`
    * `ci.jsonl_filename`
    * `ci.parquet_filename`
    * (optional) `ci.dataset_owner`, `ci.validation_profile`, etc.
  * Canonical reference: `docs/contract_override_tags.md`.
  * These tags are attached to the DataSaver metadata node, so `DagCatalog` can carry them in `OutputDescriptor.tags`.

### Files MODIFIED

* `src/codeintel/build/schemas/contract_service.py`

  * Delete `overrides_from_output_contract(contract, table_key=...)`.
  * Replace with `overrides_from_output_descriptor(output: OutputDescriptor) -> DatasetContractOverrides`:

    * read `output.tags.get("ci.json_schema_id")`, etc.
  * In `ContractService.get_dataset_contract(table_key)`:

    * resolve producing output descriptor via `target_metadata.target_for_table_key(table_key)` **or** directly via `catalog.table_outputs[table_key]`
    * build overrides from output descriptor tags (not from target.contract)

### Files MODIFIED

* `src/codeintel/core/schemas/contract_service.py`

  * Remove the `OutputContract`-typed wrapper `overrides_from_output_contract` export.
  * Replace with `overrides_for_table_key(table_key)` (or remove entirely if not used externally).

### Files MODIFIED

* `src/codeintel/build/schemas/__init__.py`

  * Remove exports/imports for `overrides_from_output_contract` and `OutputContract`.

---

# 7) BuildSpec + expected outputs: replace derived_outputs/contract fallbacks with catalog

### Files MODIFIED

* `src/codeintel/build/spec/compile.py`

  * Replace `derive_target_outputs_from_savers(runtime)` with `catalog = runtime.catalog`.
  * `_compile_target_specs(...)` becomes catalog-based:

    * outputs = `tuple(o.key for o in catalog.targets[target].io.contract_writes() if o.kind=="table")`
    * artifacts = `tuple(o.key for o in catalog.targets[target].io.contract_writes() if o.kind=="artifact")`
    * artifact path template comes **only** from output descriptor (`artifact_path_template`); delete fallback to `target.contract.get_artifact(...)`.

* `src/codeintel/build/hamilton/native/outputs.py`

  * Delete caching of `DerivedTargetOutputs` and runtime auto build for derived outputs.
  * Rewrite helpers to read from `DagCatalog`:

    * `expected_table_keys_for_target(target_name, *, catalog=None)`
    * `expected_artifact_names_for_target(...)`
    * `artifact_templates_for_target(...)` resolves from `catalog.artifact_outputs` filtered by producer_target and key.
  * Update `expected_datasets/expected_artifacts` to accept `target_name: str` (or `TargetDescriptor`) instead of `OutputTarget`.

---

# 8) Target metadata service: indexes must be built from catalog outputs, not contracts

### Files MODIFIED

* `src/codeintel/build/target_metadata.py`

  * Post-(1) you likely already swapped `TargetSystem.graph` → `TargetSystem.catalog`; ensure output indexes are now derived from:

    * `catalog.table_outputs` (table_key → producer target)
    * `catalog.artifact_outputs` (artifact_name → producer target)
  * Remove any indexing logic that iterates `target.contract.table_keys` / `artifact_names`.

---

# 9) Tests: remove OutputContract fixtures; assert saver-tag inventory is authoritative

### Files DELETED

* `tests/_helpers/contracts.py` (OutputContract builder)

### Files MODIFIED (representative; grep-driven sweep required)

* `tests/_helpers/build.py` (docstring + helpers: remove “use OutputContract/OutputTarget directly” guidance; pivot to synthetic Hamilton modules with `@save_to` tags or to `DagCatalog` fixtures)
* `tests/_helpers/harnesses/hamilton_build.py`

  * Replace `expected_tables = sorted(target.contract.table_keys)` with catalog-based expectations.
* `tests/build/test_contracts_parameters_state.py`

  * Entire file becomes obsolete if `build/contracts.py` deleted; replace with:

    * `test_saver_tag_validation_uniqueness_table_key`
    * `test_artifact_requires_path_template_tag`
    * `test_unknown_schema_issues_for_produced_tables`
* `tests/build/hamilton/test_schema_index_overrides.py`

  * Rewrite to test `explicit_registry` vs `inferred_relation` derivations based on table registry presence and inferability; remove OutputContract override cases.
* `tests/build/test_hashing_plan_targets.py`, `tests/build/hamilton/test_pr09_planner.py`, `tests/build/test_state.py`, etc.

  * Remove `OutputContract(...)` construction; build a minimal synthetic DAG module using `SaveToObjectMetadataDecorator` to declare outputs (contract role) and assert catalog inventories drive planner/state.

### Files CREATED

* `tests/build/hamilton/test_saver_declared_output_inventory.py`

  * Assert catalog output inventory equals saver tag inventory (one-to-one) and uniqueness errors are deterministic.
* `tests/build/hamilton/test_write_enforcement_allowlist.py`

  * Assert `ContractEnforcer.validate_*` allows only catalog-declared contract outputs when strict.
* `tests/build/schemas/test_schema_index_from_catalog_outputs.py`

  * Assert schema index derivations built from catalog outputs; non-inferable missing registry schema fails.

---

# 10) File operations summary (P0 list)

## Created

* `tests/build/hamilton/test_saver_declared_output_inventory.py`
* `tests/build/hamilton/test_write_enforcement_allowlist.py`
* `tests/build/schemas/test_schema_index_from_catalog_outputs.py`

## Modified

* `src/codeintel/build/hamilton/dag_catalog_compiler.py`
* `src/codeintel/build/hamilton/save_to.py`
* `src/codeintel/build/hamilton/contracts/enforcement.py`
* `src/codeintel/build/hamilton/hooks/contract_hook.py`
* `src/codeintel/build/hamilton/contracts/enforced_gateway.py`
* `src/codeintel/build/hamilton/materializers/{duckdb_saver,duckdb_rows_saver,artifact_saver}.py`
* `src/codeintel/build/hamilton/validate.py`
* `src/codeintel/build/schemas/{schema_index,inference_service,provider_unified,registry,contract_service,__init__}.py`
* `src/codeintel/core/schemas/contract_service.py`
* `src/codeintel/build/spec/compile.py`
* `src/codeintel/build/hamilton/native/outputs.py`
* `src/codeintel/build/target_metadata.py`
* `src/codeintel/build/errors.py`
* `src/codeintel/build/__init__.py`
* test helper surfaces under `tests/_helpers/**` that referenced OutputContract

## Deleted

* `src/codeintel/build/contracts.py`
* `src/codeintel/build/hamilton/introspect.py` (or at minimum delete the output inventory portion)
* `tests/_helpers/contracts.py`
* `tests/build/test_contracts_parameters_state.py` (replace with saver-inventory tests)

---

# 11) DoD (hard checks)

1. **No output inventory types** remain besides `DagCatalog` output descriptors:

   * no `OutputContract.tables`, no `ArtifactSpec` inventory, no `DerivedTargetOutputs`.
2. All “what does this target produce?” questions are answered by:

   * `catalog.targets[target].io.contract_writes()` and/or `catalog.table_outputs/artifact_outputs`.
3. `validate_nodes(...)` errors are the canonical gate for:

   * missing saver tags
   * duplicate `table_key` / `artifact`
   * missing `artifact_path_template`
   * missing explicit schema for non-inferable produced tables
   * orphan saver nodes (no downstream matching anchor)
4. Schema index + inference operates from catalog outputs (not contracts) and produces deterministic derivation records.
5. BuildSpec compilation uses catalog outputs and artifact templates strictly from saver tags.
6. Test suite contains **zero** direct construction of `OutputContract` / `ArtifactSpec` / `OutputTarget.contract`.

If you want the next increment after (3): the natural follow-on is **“move per-output export/ownership metadata fully into saver tags (or a keyed registry) and delete the positional tuple alignment logic entirely”**—but the plan above already sets you up to do that cleanly.


Below is a **repo-concrete, breaking-change–ready implementation plan** for:

> **(4) Unify manifest-driven incrementality with Hamilton caching for a massive code shrink**

Goal: collapse the two independent “skip engines” — (1) your manifest-based incrementality (hashes, input/option digests, per-target manifests, decision traces) and (2) Hamilton’s native caching subsystem — into **one authoritative caching layer**.
After this, **Hamilton’s cache determines recomputation or reuse**, and the **manifest becomes an audit artifact**, not a control plane.

---

## 0 · End-State Contract (what must be true)

| Layer                                         | Role                                                                         | Truth Source                                  |
| --------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------- |
| **Hamilton cache**                            | Decides compute/skip per node.  Stores materialized results + hash metadata. | Hamilton → `CacheAdapter`                     |
| **Manifest (pipeline_runs / pipeline_steps)** | Descriptive record: what ran, what was skipped, what cache-hit.              | Emitted by cache events + executor telemetry. |
| **Decision trace**                            | Derived analytics view of manifest + cache metadata.                         | SQL/Parquet join – no bespoke planner logic.  |

No second “skip propagation” logic remains in planner or executor; everything funnels through Hamilton’s cache.

---

## 1 · Introduce a unified cache adapter

### Files CREATED

`src/codeintel/build/hamilton/cache_adapter.py`

**Implements** a thin shim over Hamilton’s `CacheAdapter` API that writes/reads the same metadata used by your manifest system.

Key classes + functions:

```python
# cache_adapter.py  (conceptual)
from hamilton.caching import CacheAdapter, CacheMetadata
from codeintel.build.manifest.records import ManifestWriter, ManifestEntry

class ManifestBackedCacheAdapter(CacheAdapter):
    def __init__(self, store, manifest_writer: ManifestWriter, *, strict=True):
        self._store = store              # e.g. DuckDB/Parquet store path
        self._manifest_writer = manifest_writer
        self._strict = strict

    def get(self, node_name, version):
        record = self._store.lookup(node_name, version)
        if record is None:
            self._manifest_writer.record_miss(node_name, version)
            return CacheMetadata(hit=False)
        self._manifest_writer.record_hit(node_name, version)
        return CacheMetadata(hit=True, path=record.path, size=record.size)

    def put(self, node_name, version, value):
        path = self._store.save(node_name, version, value)
        self._manifest_writer.record_store(node_name, version, path)
        return path
```

Notes:

* `store` → any pluggable persistence (DuckDB table, Parquet, filesystem).
* `manifest_writer` → writes to `pipeline_steps` table or JSONL for audit.

---

## 2 · Wire Hamilton caching into runtime

### Files MODIFIED

`src/codeintel/build/hamilton/driver_factory.py`

1. Add `enable_cache` / `cache_dir` / `cache_adapter_cls` to the builder pipeline.
2. When constructing the final driver:

   * build or reuse a `ManifestBackedCacheAdapter`
   * pass into Hamilton builder:

     ```python
     builder.with_caching(
         adapter=cache_adapter,
         default_behavior="enable",
         cache_version_strategy="per_node",
     )
     ```
3. Remove explicit planner/executor “skip gating”.

### Files DELETED

* `src/codeintel/build/hamilton/runtime_incremental.py`
* `src/codeintel/build/incremental/skip_logic.py`
* any `should_skip_target(...)` utilities.
  These become redundant once cache is authoritative.

---

## 3 · Replace per-target manifest update calls with cache hooks

### Files MODIFIED

`src/codeintel/build/hamilton/executor.py`

Replace:

```python
if should_skip_target(...):
    record_skip(...)
else:
    execute_target(...)
    write_manifest(...)
```

with:

```python
result = driver.execute_with_caching(target_name)
```

and rely on the cache adapter’s internal manifest writer to record hit/miss/store events.

Executor responsibility reduces to:

* configuring cache adapter,
* feeding cache metrics to observability.

### Files MODIFIED

`src/codeintel/build/hamilton/run_records.py`

* Remove `compute_target_input_hash()` / `options_hash()` duplication; replace with `CacheKeyResolver` (see § 4).
* Simplify `RunRecord` structure: only `target_name`, `cache_key`, `cache_hit`, `cache_path`, `runtime_ns`.

### Files MODIFIED

`src/codeintel/build/hamilton/planner.py`

* Eliminate manifest diff computation.
* Planner now only builds *requested* target list; skip/execute decisions are delegated to cache.

---

## 4 · Unify hash / version computation

### Files CREATED

`src/codeintel/build/hamilton/cache_key_resolver.py`

Implements deterministic cache key generation used by both Hamilton caching and old manifests.

```python
class CacheKeyResolver:
    def for_node(self, node: NodeDescriptor, catalog: DagCatalog, env: BuildEnv) -> str:
        # combine input hashes, config hashes, code hash
        return blake3(
            node.tags.get("target","") +
            node.tags.get("domain","") +
            env.digest() +
            node.code_hash()
        ).hexdigest()
```

### Files MODIFIED

* `src/codeintel/build/hashing.py` → delegate to `CacheKeyResolver`
* `src/codeintel/build/state_computer.py` → remove per-target input hash evaluation

Now there is a single deterministic hash path used for cache key + manifest record.

---

## 5 · Manifest subsystem → audit only

### Files MODIFIED

`src/codeintel/build/manifest/records.py`

* Reduce schema: keep only `target`, `cache_key`, `status` ("hit"/"miss"/"stored"), `duration`, `size`, `timestamp`.
* Remove input/options hash columns.
* Add `cache_path` and `cache_version`.

`src/codeintel/build/manifest/writer.py`

* Simplify API: `record_hit`, `record_miss`, `record_store`, each taking `(node, version, path=None, duration=None)`.

`src/codeintel/build/manifest/reader.py`

* Drop skip-decision logic; becomes pure telemetry loader.

`src/codeintel/build/hamilton/native/export/decision_trace.py`

* Replace per-planner “skip reason” field with cache metadata snapshot (`hit`, `version`, `path`, `duration_ns`).

---

## 6 · Validation & telemetry integration

### Files MODIFIED

`src/codeintel/build/hamilton/validate.py`

* Add cache-consistency validator:

  * walk DAG; ensure each contract saver node has a cache metadata entry (hit/miss/store)
  * flag dangling manifest entries without node.

`src/codeintel/core/telemetry/hooks/cache_events.py` (new module)

* Subscribe to `CacheAdapter` events to emit OpenTelemetry metrics:

  * `codeintel.cache.hit`, `miss`, `store` counters
  * `cache_duration_ns` histogram
* Remove old `manifest.write_event` hooks.

---

## 7 · Configuration surface

### Files MODIFIED

`src/codeintel/build/config.py`
Add:

```python
@dataclass
class CacheConfig:
    enabled: bool = True
    dir: Path = Path(".codeintel/cache")
    backend: Literal["duckdb","parquet","filesystem"] = "duckdb"
    manifest_integration: bool = True
    strict: bool = False
```

### Files MODIFIED

`src/codeintel/cli/handlers/build.py`
Expose CLI flags:

```
--cache-enable / --cache-disable
--cache-dir <path>
--cache-backend <type>
--no-manifest
```

These populate `CacheConfig` and feed into driver_factory.

---

## 8 · Tests

### Files CREATED

`tests/build/hamilton/test_cache_adapter.py`

* round-trip test: store → get → manifest entries written
  `tests/build/hamilton/test_cache_manifest_integration.py`
* ensure manifest records mirror cache events, not planner decisions
  `tests/build/hamilton/test_cache_key_resolver.py`
* verify deterministic key across runs with identical DAG + env
  `tests/build/hamilton/test_cache_skip_behavior.py`
* build small DAG with cache enabled; assert only cache-missed nodes execute (via side-effect counter)

### Files DELETED

* `tests/build/test_incremental_skip_logic.py`
* `tests/build/test_manifest_skip_propagation.py`

---

## 9 · File Index Summary

### Created

* `src/codeintel/build/hamilton/cache_adapter.py`
* `src/codeintel/build/hamilton/cache_key_resolver.py`
* `src/codeintel/core/telemetry/hooks/cache_events.py`
* new test files above

### Modified

* `src/codeintel/build/hamilton/driver_factory.py`
* `src/codeintel/build/hamilton/executor.py`
* `src/codeintel/build/hamilton/planner.py`
* `src/codeintel/build/hamilton/run_records.py`
* `src/codeintel/build/hashing.py`
* `src/codeintel/build/state_computer.py`
* `src/codeintel/build/manifest/{records,writer,reader}.py`
* `src/codeintel/build/config.py`
* `src/codeintel/build/hamilton/validate.py`
* `src/codeintel/cli/handlers/build.py`

### Deleted

* `src/codeintel/build/hamilton/runtime_incremental.py`
* `src/codeintel/build/incremental/skip_logic.py`
* `tests/build/test_incremental_skip_logic.py`
* `tests/build/test_manifest_skip_propagation.py`

---

## 10 · Definition of Done

1. No function named `should_skip_target`, `skip_logic`, or `manifest_diff` exists.
2. Planner and executor call Hamilton Driver execution directly; skip/hit decisions come from cache.
3. Manifests are written *after* cache events; removing manifest write does not change compute decisions.
4. `pipeline_steps.status` ∈ {hit, miss, store}.
5. Cache metrics appear in OTel under `codeintel.cache.*`.
6. Tests confirm deterministic key stability and single-node recomputation semantics.

---

After this phase the codebase’s incremental execution path is *mathematically identical* to Hamilton’s caching semantics.
All higher-level constructs (planner, manifest, decision trace) become **read-only analytics layers**, yielding a leaner, faster, and much easier-to-reason-about execution core.
