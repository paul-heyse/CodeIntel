
According to a document from **December 24, 2025**, your current `scip` native Hamilton module already demonstrates the “Phase 4” pattern clearly: a tool step, artifact savers with `path_template`, DuckDB row savers, and a `t__scip` target anchor that assembles a `TargetRunRecord` from materialization metadata. The architecture overview explicitly calls out that saver-node tags (emitted by `SaveToObjectMetadataDecorator`) are the backbone for contract/output derivation and IO surface introspection. Since the SCIP upgrade, `index.scip` + protobuf parsing (via `scip_proto`) are the canonical path, and incremental indexing + richer tables are part of the baseline. This is the exact leverage point to push `scip.py` from “DAG-driven” to “maximally DAG-derived” with minimal remaining boilerplate and minimal drift risk. 

Below is a **fully integrated, narrative implementation plan** to enact the **10 consolidations** for `src/codeintel/build/hamilton/native/ingestion/scip.py` (representative target), plus advanced optimizations and further consolidation opportunities. It’s intentionally written so an experienced Python/Hamilton engineer can implement without line-by-line diffs.

---

## North-star: what “maximally DAG-driven” should mean for `scip`

For `scip`, the “end state” should satisfy these invariants:

1. **Module-state + manifest are authoritative for reuse/skip** — no ad-hoc file existence checks can override hash/manifest decisions. Module-level correctness is anchored in `core.scip_module_state`, with the shard manifest as a cache that can be regenerated. (This is crucial for correctness when options, tool versions, or upstream semantics change.)
2. **Artifacts and tables are “declared by savers”, not by anchor logic** — collectors, run record assembly, and output inventory should be derived from saver tags and contracts, not hand-maintained lists inside `scip.py`. This aligns with DAG-derived output derivation and IO registry compilation described in the architecture. 
3. **All configuration flows through DAG-visible nodes** — the target’s options are loaded once (as a node), hashed once, and used consistently in tool execution + ingestion. No hidden configuration pathways.
4. **The anchor node is “boring”** — it should only: (a) validate upstream target health, (b) call a shared “record builder” that consumes materialization metadata, and (c) return `TargetRunRecord`. No bespoke artifact/table enumeration or custom summarizers per target.
5. **Adding a new output requires minimal edits** — ideally only “add a new saver-decorated node” (or add an item to a small spec list) and the rest is automatically reflected in: output inventory, IO registry, materialization collection, and record assembly.
6. **Single-target truth for mixed outputs** — tool run, artifact materialization, ingest, and table writes remain within one target DAG for a single calculation scope/context. Do not split tool/ingest into separate targets. Avoid consuming `a__*` / `ArtifactRef` inside the same target (it depends on the target record and creates a cycle); use tool outputs or saver metadata nodes instead.
7. **Output inventory is DAG-derived in steady state** — DAG saver tags are the sole source of truth; transitional modes are removed. Utilities that read templates must fall back to explicit templates when DAG-derived inventory is unavailable.
8. **Protobuf is canonical for SCIP** — `index.scip` + `scip_proto` codegen are the source of truth; JSON artifacts are not part of the pipeline. SCIP project identity is stable (`--project-name CodeIntel`).

Everything below is designed to make those invariants true for `scip`, using the system you already have (SaveTo tags → output derivation; compile_write_registry; support nodes; etc.). 

---

## Integrated implementation sequence

### Phase ordering matters

To avoid regressions and reduce rework, implement in this order:

1. **Correctness + single-source-of-truth config** (Consolidations 1–3)
2. **DAG-pure dataflow: artifact-driven ingest and unified result models** (Consolidations 4–5)
3. **Auto-collection + generic run record assembly** (Consolidations 6–7)
4. **Efficiency + derived resources + observability** (Consolidations 8–10)
5. **Advanced optimizations + “beyond the 10”** (optional, but recommended)

After each target migration (starting with `scip`), add explicit **settings + validation gates**:

* DAG-derived outputs and support nodes are now the only mode; no per-run toggles remain.
* Run `build validate` with `validate_graph(..., enforce_compute_io_purity=True)` to ensure no manual IO remains and DAG purity is preserved as you scale the pattern.
* Treat any drift reported by `build validate` as a blocking defect before moving to the next target.

---

# The 10 consolidations for `scip.py`

## 1) Eliminate non-manifest reuse: make tool execution manifest-authoritative

### What’s wrong today (and why it matters)

The current `t__scip__run()` performs:

* `executor.should_skip()` (good),
* then **checks for existing `index.scip`** and returns success without running tools (dangerous if it becomes a skip override).

That file-existence short-circuit can create a correctness hole whenever:

* target options change,
* tool versions change,
* hashing semantics change,
* or upstream dependency hashes change in a way that should force recompute.

The architecture doc emphasizes that skip checks are computed from the Hamilton graph + manifests and used to determine recompute. Your target should not “override” that with filesystem heuristics. 

### Target state

`scip` tool execution must satisfy:

* If `NativeTargetExecutor.should_skip()` says **skip**, do **not** run tools.
* If it says **do not skip**, you must **run tools deterministically**, regardless of existing files, unless you can prove they correspond to the current computed input hash.
* Module-level state is authoritative via `core.scip_module_state`; the shard manifest is a cache that can be regenerated.

### Concrete implementation plan

Implement this as a **unified mechanism** across the incremental runner + tool runner:

1. **Ensure file-existence checks are only a safety valve**, not a skip override:

   * If `executor.should_skip()` is true, only skip when `index.scip` exists; if missing, force rebuild.
2. **Use DAG-derived hash inputs end-to-end**:

   * `scip__hash_options` should include options hash + module scan file_state_hash + manifest index.
   * Pass these into `NativeTargetExecutor` and `ScipIncrementalConfig` so skip/reuse is hash-driven.
   * Include `tool_version` at the module-state level so tooling changes invalidate only affected shards.
3. **Drive incremental updates from module deltas**:

   * Use `ModuleScanResult.change_set` when available; if missing, force a full rebuild.
4. **Make reuse content-addressed**:

   * Use `core.scip_module_state` as the source of truth for module-level reuse.
   * Regenerate the shard manifest (`{scip_dir}/shards/manifest.json`) from the table when missing or inconsistent.
   * Do not accept file existence alone as proof of correctness.
5. **Apply deterministic merge policy when updating `index.scip`**:

   * Symbol_information: prefer newest shard by `updated_at` **only** when the
     new row is strictly more informative (non-empty documentation/signature/
     display_name). Otherwise keep the existing row.
   * External_symbols: dedupe by full symbol string; prefer the most complete
     package triple (manager+name+version), then newest shard.
6. Add a regression test:

   * Run `scip` once → success.
   * Change a scip option or tool version stamp → ensure executor marks as not-skipped → ensure tool runtime does not reuse old files.

**Advanced optimization:** once you have `expected_input_hash`, incremental reuse becomes safe and fast: “reuse only if module-state rows report the same input hash + tool version and outputs exist”. That gives you correctness + speed.

---

## 2) Make ingestion artifact-driven: remove “in-memory documents” coupling

### What’s wrong today

Earlier iterations returned parsed documents from the tool runtime, and ingestion relied on them directly. Even though parsing is now protobuf-based, that pattern still couples ingestion to in-memory payloads rather than the canonical artifact (`index.scip`), and hides the dependency on the generated protobuf module. Keep the artifact-driven parse boundary explicit to avoid regressions.

This is subtly anti-DAG:

* ingestion becomes dependent on tool runtime behavior and memory return payloads,
* not strictly on the artifacts that are declared and materialized via savers.

The architecture explicitly positions savers + derived IO as the canonical write boundary and source of output derivation. Lean into that: treat artifacts as the contract and always parse from artifacts. 

### Target state

* Tool node returns: status + `index.scip` location (only).
* Ingest parses `index.scip` via protobuf using `scip__proto_module_path` (artifact-driven, not in-memory payloads).
* If you want a more explicit DAG boundary, extract parsing into a helper node later.

### Concrete implementation plan

1. Ensure `ScipRunResult` (alias of `ToolStepOutput`) carries only output paths + status (no documents).
2. In `t__scip__ingest`, parse `index.scip` via `parse_index(index_path, proto_module_path)` using `scip__proto_module_path`.
3. Remove JSON fallback logic entirely.
4. If you later add a `scip__parsed_index` node, ensure it returns `None` when upstream execution skipped/failed so downstream row materializers skip cleanly.
5. Add tests:

   * tool run success → `index.scip` exists → protobuf parse succeeds,
   * tool run skipped → ingest returns skip.

**Advanced optimization:** move parsing to a streaming parser if `index.scip` is large (see optimization section). Even if you don’t implement streaming now, isolating parsing into a dedicated node makes it easy later.

---

## 3) Convert `scip` to first-class target options: single-source config + hashed inputs

### What’s wrong today

Even with `ScipIngestOptions` wired in, it is easy for options to drift out of the execution path (tool invocation, incremental config, row filtering) unless they are enforced as a single source of truth. This creates:

* configuration drift risk,
* inability to tune scip behavior via the same mechanism as other targets,
* and the potential that options change won’t be represented consistently in hashing/skip.

### Target state

* `scip__options` is a DAG node (so introspection can see it).
* `t__scip__run` and `t__scip__ingest` consume `scip__options`.
* The **same options** contribute to:

  * tool execution parameters,
  * ingestion logic behavior,
  * and **options hashing** (manifest key).
* `scip__run_config` bundles `scip__options`, `scip__hash_options`, and `scip__proto_module_path` so tool execution has a single, explicit config surface.

### Concrete implementation plan

1. Add a node: `scip__options(env: BuildEnv) -> ScipIngestOptions` using the existing options loader (mirroring other targets).
2. Ensure options affect tool execution:

   * `scope_paths` → passed into the incremental config (per-module `--target-only`).
   * `max_file_size_kb` → passed into the incremental config to avoid oversized files.
   * `timeout_seconds` → passed to the tool runner.
   * `scip_output_dir` → used to resolve the canonical `index.scip` output path.
   * `scip_project_name` → sourced from `ToolsConfig.scip_project_name` (default `"CodeIntel"`) and passed into `build_scip_python_args`.
3. Ensure options affect ingestion:

   * `include_references`, `include_implementations` and any future flags should be passed into row builders (symbols, occurrences, symbol_information, relationships, diagnostics, external_symbols) or used to filter parsed documents.
4. Ensure options affect hashing:

   * Either by:

     * relying on your existing `options_hash_for_target()` (if it already hashes these parameters), or
     * explicitly passing `hash_options=InputHashOptions(options_hash=...)` into the savers and executor (see consolidation #4 and #7).

5. Add a shared hash node as the single hash source of truth:

   * `scip__hash_options(env: BuildEnv, t__modules__scan: ModuleScanResult) -> InputHashOptions`
   * Include `options_hash_for_target(env, "scip")` plus `t__modules__scan.file_state_hash` (module state) and manifest index.
   * Use it in `NativeTargetExecutor.for_target(..., hash_options=...)`, in `SaveToObjectMetadataDecorator` calls, and in incremental execution paths that need consistent reuse checks.
   * Normalize all hash inputs here (e.g., sorted/normalized `scope_paths`) so every consumer reuses the exact same hash.

**Advanced optimization:** if `scope_paths` is frequently used, also hash a normalized canonical representation (sorted, normalized paths). This prevents “same semantics, different ordering” from forcing reruns.

---

## 4) Unify the result model: collapse `ScipRunResult` and `ScipIngestResult` into a shared status/payload pattern

### What’s wrong today

`scip.py` has:

* `ScipRunResult` (alias of `ToolStepOutput`, carrying `ExecutionResult` + output paths)
* `ScipIngestResult(result: ExecutionResult, rows...)`

This is a fractured status model:

* success/skipped/error semantics exist in multiple shapes,
* anchors must manually interpret them,
* it’s harder to build generic orchestration.

### Target state

Use one “status carrier” everywhere:

* `ExecutionResult` is already your standard “target step status” abstraction.
* Tool steps should also speak `ExecutionResult` via `ToolStepOutput`.

### Concrete implementation plan

Create a reusable pattern used by other tool-backed targets:

1. Use the existing standard tool step output:

   * `ToolStepOutput(result: ExecutionResult, outputs: Mapping[str, Path])`

2. Update:

   * `t__scip__run` → returns `ToolStepOutput`.
   * `t__scip__ingest` → returns `ExecutionResult` plus rows, or returns a single dataclass with `result` like today but keep it consistent (recommended: `ExecutionResult` + “payload-only nodes”).

3. Anchor `t__scip` should interpret **only** `ExecutionResult` plus saver metadata status — no custom booleans.

**Advanced optimization:** adopt a tiny protocol interface like `HasExecutionResult` so generic helpers can accept either tool step outputs, ingest outputs, etc. This allows writing generic “if failed, return executor.fail” logic without coupling to type names.

---

## 5) Parameterize repetitive nodes: generate artifacts/tables via Hamilton modifiers instead of duplicating code

This is where you begin to *materially* reduce per-target boilerplate while increasing extensibility.

### What’s wrong today

`scip.py` has near-identical sets of:

* table row nodes (`scip__symbol_rows`, `scip__occurrence_rows`, `scip__symbol_info_rows`, `scip__relationship_rows`, `scip__diagnostic_rows`, `scip__external_symbol_rows`, `scip__module_state_rows`)
* collector nodes (artifact materializations + table materializations)

This isn’t terrible for a few outputs, but it scales badly as tables and internal artifacts grow.

### Target state

Use Hamilton’s “multi-node from one function” patterns:

* `@parameterize` / `@parameterize_sources` / `@parameterize_values` to generate a family of nodes from one function. ([Hamilton][1])
* `inject + group(source(...))` and/or `resolve_from_config` to wire lists dynamically at compile time.  ([Hamilton][2])

### Concrete implementation plan

#### 5A) Artifact nodes

1. Keep `scip__index_artifact` as the single contract artifact node for now.
2. If you add additional artifacts (e.g., shard manifest as `output_role="internal"`), replace per-artifact functions with **one** generic artifact accessor and parameterize it.
3. Make sure each parameterized output is still decorated with a saver (this may require:

   * keeping small wrappers, *or*
   * creating a mini-factory helper that returns decorated functions, *or*
   * using a custom “multi-save-to” decorator you create once.)

The key: adding a new artifact should become “add one entry in a small mapping”, not “copy/paste a function”.

#### 5B) Table row nodes

Similarly, replace multiple row nodes with one:

* `scip__rows_for_table(table_key: str, ingest_payload: ...) -> rows`
* Parameterize it for each table key.

Then the only place to update for a new table output becomes:

* add the new table key (and row builder) to the parameterization mapping.

**Advanced optimization:** if you later switch from tuples-of-tuples to Arrow/Polars, you’ll do it in exactly one implementation.

---

## 6) Auto-collect materialization metadata based on saver tags (not manual lists)

### What’s wrong today

`scip__materializations` and `scip__table_materializations` manually enumerate `m__artifact__scip_index`, etc. That means:

* adding an artifact/table requires editing collectors,
* drift is possible (“saver exists but collector forgot it”),
* generic record builders are harder.

But your architecture already centralizes the idea that saver nodes emit tags, which are used for output derivation and IO registry compilation. The same mechanism should drive materialization collection. 

### Target state

For each target, have standard “collector nodes” that are:

* generated automatically (preferred),
* or wired dynamically via `resolve_from_config` (compile-time), or
* at minimum, use `inject+group(source(...))` to reduce manual enumerations.

### Concrete implementation plan

You have three viable approaches; choose based on how aggressively you want “DAG-derived everything”:

#### Option A (best-in-class): generate collectors in the support module factory

Extend your support node generation system to also generate:

* `c__contract_artifacts__<target>` → dict[artifact_name, MaterializationMetadata]
* `c__contract_tables__<target>` → dict[table_key, MaterializationMetadata]

Mechanics:

1. During driver build, you already scan the graph for saver tags to derive target outputs. Use the exact same tag scan to list saver nodes for the target. 
2. The support factory can emit a collector function whose signature includes those saver-node names as parameters (static), and returns a dict mapping from tag value (artifact/table key) to metadata.
3. `scip.py` then deletes its custom `scip__materializations` and `scip__table_materializations` functions and depends on the generated collectors. Use a `c__` prefix (not `m__`) so collectors are clearly distinct from saver metadata nodes.

This is the “maximal DAG-derived” option: adding a saver node automatically updates collectors, record builders, and output inventory.

#### Option B: per-target collector using `resolve_from_config` + inject+group

If you’d rather not modify support_factory yet:

1. Store the list of saver metadata node names for a target in config at driver build time (or in env.registry).
2. Use `resolve_from_config` to inject `group(source(...))` and build collectors without enumerating each input manually.  ([Hamilton][2])

#### Option C: keep manual collectors, but centralize the mapping

Least preferable: keep collectors but make them read `SCIP_TABLE_KEYS` and `SCIP_ARTIFACTS` mapping so there is only one list.

**Recommendation:** Option A is the “limited room for improvement” end state.

---

## 7) Replace bespoke `_summarize_scip_table_materializations` with a shared “mixed materialization” record builder

### What’s wrong today

`scip.py` duplicates table materialization parsing/validation logic in `_summarize_scip_table_materializations`, even though you already have generalized record builders for DuckDB materializations and artifact materializations.

This prevents:

* consistent failure message semantics across targets,
* consistent row-count derivation,
* easy evolution of metadata formats.

### Target state

A single shared function should assemble the `TargetRunRecord` for targets that produce:

* artifacts only,
* tables only,
* or a mix of artifacts and tables.

### Concrete implementation plan

1. Create a new helper in `native/materialization_records.py`, e.g.:

* `record_from_materializations(env, graph, target_name, artifacts: dict|None, tables: dict|None, expected_contract: OutputContract|None=...)`

or two helpers:

* `validate_duckdb_materializations(...) -> (status, row_counts, error)`
* `record_from_*` calls `validate_*` and then writes manifest.

2. That helper should:

   * parse table materialization metadata via existing typed metadata (`DuckDBMaterializationMetadata.from_mapping(...)`),
   * handle missing metadata robustly,
   * compute row_counts consistently,
   * mark skipped vs succeeded vs failed,
   * and return a `TargetRunRecord` in one place.

3. Update `t__scip` to call the new helper; delete `_summarize_scip_table_materializations`.

**Advanced optimization:** Make it contract-driven:

* Instead of scip passing `SCIP_TABLE_KEYS`, the helper should consult the target contract derived from saver tags (since you already derive contracts from savers). That’s what makes it auto-extensible.

---

## 8) Fix artifact persistence efficiency: avoid redundant “read then rewrite same file”

### What’s wrong today

`FileArtifactSaver` is used to make persistence DAG-visible (good), but the tool step also writes artifacts directly into the same paths those savers resolve to. This can lead to:

* double work (tool writes file; saver reads file; saver writes file again),
* potential correctness pitfalls if the saver uses atomic rename over the same file it reads.

The architecture describes artifact savers as template-driven path resolvers and write boundaries. Ideally, either:

* savers write the artifact (compute produces bytes/plan), or
* savers validate and record an already-written artifact without copying. 

### Target state

Choose one of these “best-in-class” patterns:

#### Pattern A: Saver-only writes via write plan

Tool node returns an `ArtifactWritePlan` that, when invoked, runs the tool and writes directly to the resolved output path (so tool execution happens at materialization boundary).

* This is the most “pure boundary” model.
* But it’s tricky when one tool run produces multiple artifacts (e.g., `index.scip` + shard manifest).

#### Pattern B (recommended): “ExistingArtifactSaver” records metadata without rewriting

Introduce a specialized DataSaver for tool-produced files:

* If file exists at resolved path, it:

  * stats it, maybe hashes it,
  * records materialization metadata,
  * writes manifest metadata,
  * **does not copy**.

This keeps DAG visibility, keeps contract enforcement, and avoids redundant IO.

### Concrete implementation plan

1. Implement Pattern B as a new saver:

   * `ExistingFileArtifactSaver` (or extend `FileArtifactSaver` with `mode="record_only"` / `allow_in_place=True`).
2. In `scip.py`, use this saver for `scip_index` materialization (and any internal shard/manifest artifacts if you choose to surface them).
3. Ensure the saver still:

   * computes input hash (manifest key),
   * respects manifest skip,
   * validates contract (ContractEnforcer),
   * participates in saver tag derivation/validation (`derive_target_outputs_from_savers`, IO registry, `validate_nodes`),
   * and returns consistent metadata schema.

**Advanced optimization:** include a “content hash” (e.g., sha256) in `MaterializationMetadata` for artifacts and persist it in the manifest (not a sidecar). This enables stronger provenance, centralized reuse checks, and future dedupe without introducing a second metadata authority.

---

## 9) Derive target resources (especially tool requirements) from DAG tags instead of duplicating them in `TargetSpecDescriptor`

### What’s wrong today

`scip` lists required tools in `TargetSpecDescriptor.resources.tools = ("scip-python",)`, but the DAG already knows which tool step exists (tagged as tool). If the tool list changes and spec isn’t updated, you get drift.

Your architecture already relies on tags for IO derivation. Extend that philosophy to resources.

### Target state

Resources are computed as:

* explicit overrides when necessary,
* otherwise derived from node tags (especially tool nodes), and/or from output types.

### Concrete implementation plan

1. Extend `tag_tool(...)` usage in `t__scip__run` to include tool names as tags, e.g.:

   * `extra_tags={"tools": ["scip-python"]}` (or whatever canonical tag key you choose).
2. Extend target spec compilation to:

   * scan nodes for tool tags within a target scope,
   * union them into `TargetResources.tools` for that target.
3. Add canonical tag keys in `codeintel/core/hamilton/tags.py` for tool identity, and treat `TargetSpecDescriptor.resources.tools` as an explicit override when provided.
4. Update `scip.py` to remove explicit tools list *once the compiler is proven*.

**Advanced optimization:** do the same for “modules dependency”. If a target depends on `t__modules`, you can infer `resources.modules=True` automatically instead of specifying.

---

## 10) Harden DAG-derived correctness with stronger validation + observability gates

This consolidation is about ensuring the refactor stays correct and scalable as you add targets/outputs.

### What’s wrong today

The system is already strong, but after consolidations 1–9 you’ll be relying even more heavily on:

* saver tags as canonical declarations,
* auto-generated collectors,
* and generic record builders.

You need guardrails so mistakes surface early.

### Target state

Add or extend validations so “bad DAG definitions” fail fast during driver build or execution.

### Concrete implementation plan

#### 10A) Validation rules to add/extend

1. **Tool node reuse rule**: if a target has a tool node and also materializes artifacts, ensure:

   * tool node does not “skip via filesystem” when manifest says rerun (enforce by unit test + optionally static scan).
2. **Contract coverage**: for each target:

   * every contract output derived from savers must have a corresponding saver node,
   * every saver node tagged as contract output must specify required tags (`artifact_path_template`, `table_key`, etc.). The architecture notes these tags are required for output derivation. 
3. **Collector completeness** (if you implement auto collectors):

   * ensure collector dict keys match derived contract outputs (no missing).
4. **Run record correctness**:

   * record builder must mark failed if any required artifact write failed,
   * must include row_counts for all tables in contract (even if 0 when skipped).
5. **Compute IO purity gate**:

   * run `validate_graph(..., enforce_compute_io_purity=True)` as part of migration gating to ensure no manual IO has crept back into compute/tool nodes.
6. **Module-state integrity gate**:

   * ensure `core.scip_module_state` is populated and consistent for the current snapshot; regenerate shard manifest from the table when missing.

#### 10B) Observability / telemetry enhancements

Leverage Hamilton’s “build-time vs run-time” separation and execution instrumentation:

* You already have node telemetry persistence (architecture mentions per-node telemetry persisted by BuildRunWriter). Strengthen:

  * consistent tag keys for node types,
  * consistent “semantic id” tags for tool invocations (so you can aggregate metrics). 

Optionally integrate Hamilton’s own lifecycle/caching/structured logs features:

* The Hamilton Builder can attach caching and other adapters at driver build time. ([Hamilton][3])
* Hamilton caching/logging facilities exist via `.with_cache(...)` and structured logs. ([GitHub][4])
* Lifecycle customization can be used to instrument execution behavior. ([DAGWorks][5])

You don’t have to adopt these immediately (your manifest system already covers skip), but it’s useful for *in-run* observability and diagnosing performance regressions.

---

# Putting it together: the “new scip” dataflow shape

Once the 10 consolidations land, `scip` should look like this conceptually:

1. `t__scip_proto__run` → `scip__proto_artifact` → `t__scip_proto` → `scip__proto_module_path`
2. `scip__options` + `scip__hash_options` + `scip__module_inputs` + `scip__run_config`
3. `t__scip__run` (incremental tool node) → returns `index.scip` path (no documents)
4. `scip__index_artifact` (artifact node) → materialized via `FileArtifactSaver` (record-only optional)
5. `t__scip__ingest` (compute node) → parses `index.scip` via protobuf and returns `ExecutionResult` + payload
6. `scip__rows_for_table` (parameterized) → seven outputs for table keys (including `core.scip_module_state`) → materialized via DuckDBRowsSaver
7. `c__contract_*__scip` collectors (auto-generated)
8. `t__scip` anchor → calls shared `record_from_materializations(...)` and returns TargetRunRecord

Importantly:

* No per-target table/materialization summarizer.
* No manual collectors.
* Options are first-class.
* Tool execution correctness is guaranteed by hash/manifest.
* Protobuf (`index.scip`) is canonical; JSON artifacts are not part of the pipeline.

This aligns strongly with the architecture’s emphasis on:

* saver tags driving output derivation,
* IO registry compilation from DataSaver tags,
* and the representative `scip` pattern being the “showcase” module. 

---

# Advanced optimizations applied to the 10 consolidations

These are optional but materially improve scalability/performance while keeping the DAG-centric model.

## A) Stream rows instead of building huge tuples in memory

Today ingestion constructs `tuple(build_symbol_rows(...))` etc. That will not scale on large repos.

Two best-in-class options:

1. **Chunked writer saver**:

   * Extend `DuckDBRowsSaver` (or create a new saver) to accept an iterator/generator of rows and write in chunks using `executemany()` or Arrow ingestion.
   * This preserves the “compute → saver boundary” while reducing peak memory.

2. **Arrow/Polars path**:

   * Build a `pyarrow.Table` or `polars.DataFrame` for symbols/occurrences in chunks and let DuckDB ingest Arrow directly (DuckDB reads Arrow efficiently). ([Hamilton][6])
   * Then use a saver variant that materializes from Arrow/Polars (your architecture already uses “deferred columns” and a Warehouse write path; this is a natural evolution). 

## B) Parallelize ingestion row building by document partitions

If you factor parsing into a `scip__parsed_index` node, you can:

* partition documents by language or file path,
* build rows in parallel (thread/process) as long as serializer is thread-safe,
* then concatenate for write.

This can be gated by execution settings.

## C) Make `created_at` stable across the entire run

Instead of `datetime.now(UTC)` inside `t__scip__ingest`, use a run-scoped timestamp from env execution context (BuildRunContext). This improves reproducibility and keeps all targets consistent.

## D) Adopt profile-driven DAG variants (fast/full)

If you later want:

* “fast profile” = skip scip occurrences, or only index subset,
* “full profile” = everything,
  use config-driven DAG shaping (`resolve_from_config`, or Hamilton config gating) at **driver build time** (not runtime). Hamilton docs emphasize build-time DAG vs runtime inputs. ([Hamilton][3])
  Your local Hamilton advanced doc also highlights compile-time wiring via resolve. 

---

# Additional consolidation opportunities beyond the 10

If you want the “limited room for improvement” end state, these are the next layer:

## 1) Auto-generate `t__<target>` anchors (or make them declarative)

Right now, each target has a custom anchor function. You can push further:

* Introduce a **generic anchor factory** that:

  * reads the target’s derived contract (from saver tags),
  * reads auto-collected materialization metadata,
  * checks upstream target statuses,
  * and produces `TargetRunRecord`.

Then per-target code only supplies:

* tool nodes / compute nodes,
* and the target spec decorator (or even that could become derived).

This would make the system extremely “DAG-defined”.

## 2) Treat artifacts as first-class refs in the DAG, not raw Paths

The architecture already has artifact support nodes `a__<artifact_name>` that return `ArtifactRef`. 

You can push `scip` in that direction:

* Make downstream nodes consume `ArtifactRef` instead of `Path` **only when they are in a different target**. Within the same target, `a__*` depends on `t__<target>` and will introduce a cycle; keep same-target consumption tied to tool outputs or saver metadata.
* That allows:

  * consistent location resolution,
  * future remote storage,
  * and better provenance.

## 3) A single “IO contract” registry driven entirely by saver tags

You already compile an IO registry from DataSaver tags. The next step is to make:

* runtime orchestration decisions (e.g., “which sinks to warm up, which DB schemas to ensure”) driven by this registry at driver build time. 

## 4) Use Hamilton lifecycle hooks to unify execution logging and failure semantics

Hamilton’s lifecycle APIs can be used to implement:

* consistent timing metrics,
* consistent exception handling,
* consistent structured logs, independent of target code. ([DAGWorks][5])

You already persist telemetry; lifecycle hooks could eliminate per-target logging drift.

---

## Implementation “definition of done” checklist for `scip`

After implementing the 10 consolidations, you should be able to demonstrate:

1. **Change scip options** → input hash changes → `scip` reruns even if old files exist.
2. **Add a new scip table or internal artifact** (e.g., shard manifest or module-state rows) → output inventory and collectors include it automatically.
3. **Add a new scip table saver** → row_counts includes it automatically; no edits to summarizers/collectors.
4. **Protobuf path is enforced** → `t__scip_proto` runs, `scip__proto_module_path` is used in ingestion, and no JSON artifacts are required.
5. **Module-state is authoritative** → `core.scip_module_state` rows are populated and used to regenerate shard manifests when missing.
6. `t__scip` anchor becomes small and generic: no bespoke lists, no bespoke parsing of metadata.
7. Peak memory improves (if you do streaming/chunking), but even without it, the code path is now set up for it cleanly.
8. DAG saver templates are viable for scip: artifact templates resolve from saver tags, and no declared inventory is needed for correctness.

---

The shared library extraction plan below is the fully detailed, implementation-ready scope for the shared libraries work.

[1]: https://hamilton.apache.org/reference/decorators/parameterize/?utm_source=chatgpt.com "parameterize - Hamilton - The Apache Software Foundation"
[2]: https://hamilton.staged.apache.org/reference/decorators/resolve/?utm_source=chatgpt.com "resolve - Hamilton - The Apache Software Foundation"
[3]: https://hamilton.apache.org/concepts/builder/?utm_source=chatgpt.com "Builder - Hamilton - The Apache Software Foundation"
[4]: https://github.com/dagworks-inc/hamilton/blob/main/docs/concepts/caching.rst?plain=true&utm_source=chatgpt.com "hamilton/docs/concepts/caching.rst at main"
[5]: https://blog.dagworks.io/p/customizing-hamiltons-execution-with?utm_source=chatgpt.com "Customizing Hamilton's Execution with the new Lifecycle API"
[6]: https://hamilton.staged.apache.org/concepts/materialization/?utm_source=chatgpt.com "Materialization - Hamilton - The Apache Software Foundation"

# shared library extraction plan #

Below is a **shared library extraction plan** that turns the “`scip.py` consolidation pattern” into **reusable, low‑variance building blocks** you can apply across *all* native targets (tool-driven, artifact-only, table-only, mixed) while staying faithful to the repo’s core architectural premise: **targets + IO surfaces are DAG-defined via `t__*` anchors and DataSaver tags, and support nodes can be generated from either contracts or saver-derived outputs**.

I’m going to name **exact modules/utilities to add** (and the exact existing modules to extend), and describe the **public APIs, invariants, and adoption steps** so an expert can implement this directly.

---

## 0) Scope and goals (shared libraries)

- Reuse existing run-result primitives: `ExecutionResult`, `ToolStepOutput`, `MaterializationMetadata`, and `NativeTargetExecutor`.
- Keep manifest semantics unchanged: `save_manifest(...)` remains the single authority; no new manifest schema.
- Enforce DAG-derivable saver tags: `target_name`, `table_key` or `artifact_name`, `path_template`, and `output_role` must be static via `value(...)`.
- Preserve output inventory behavior: use DAG templates and fall back to explicit templates when needed.
- Keep shared modules outside native discovery: `native/patterns` must not register as targets.
- Make generated nodes deterministic and tag collectors with `tag_helper` to keep IO surfaces clean.

## 1) New shared package: `src/codeintel/build/hamilton/native/patterns/`

Create a dedicated patterns package **outside** domain directories (so discovery won’t treat them as targets). This becomes the “standard library” for DAG-native target authoring.

### 1.1 `native/patterns/specs.py`

Define the typed spec objects used across the shared library:

* `ArtifactOutputSpec` (artifact name, path template, output_role)
* `TableOutputSpec` (table_key, columns/deferred_columns, output_role)
* `ToolTargetSpec` (domain, target_name, `TargetSpecDescriptor`, artifacts, tables, tool tags)

These specs are the single source of truth for “what a target outputs” when using templates.

### 1.2 `native/patterns/__init__.py`

Export the public, stable surface:

* decorator factories (`save_artifact`, `save_rows`, `save_ibis_table`)
* collectors (`make_artifact_materializations_collector`, `make_table_materializations_collector`, `make_mixed_materializations_collector`)
* tool-step helpers (`run_tool_step`, `run_tool_and_ingest`)
* finalization helpers (`finalize_target_from_materializations`)

Re-export spec types from `specs.py` so target modules only import from `native.patterns.*` for orchestration/IO, and “business logic” remains in target modules.

---

## 2) Standardize SaveTo + tagging in one place

### 2.1 Add: `src/codeintel/build/hamilton/native/patterns/savers.py`

**Purpose:** Remove repetitive `SaveToObjectMetadataDecorator(...)` boilerplate, enforce tag invariants consistently, and make “contract vs internal output” *a one-liner*.

#### Public API (proposed)

* `save_artifact(...) -> Callable[[Fn], Fn]`
* `save_rows(...) -> Callable[[Fn], Fn]`
* `save_ibis_table(...) -> Callable[[Fn], Fn]` (for `DuckDBIbisTableSaver` targets)
* optional: `save_artifact_internal(...)`, `save_rows_internal(...)` convenience wrappers (sets `output_role="internal"`)

#### Required behavior

Each factory composes:

1. **canonical compute tagging** (domain/target + extra tags including `artifact` or `table_key`), and
2. **SaveToObjectMetadataDecorator** with:

   * `env=source("env")`, `graph=source("graph")` (always)
   * `target_name=value(target)`
   * static identity (`table_key=value(...)` or `artifact_name=value(...)`)
   * for artifacts: `path_template=value(...)` (must stay a `value()` to satisfy DAG-derived tag constraints)
   * `output_name_ = materialize_node(...)`
3. **hash and schema wiring**:

   * pass `hash_options=source("<target>__hash_options")` when a hash node exists
   * for row-based tables, pass `columns=value(deferred_columns_for_table_key(table_key))`
4. **no new saver classes**:

   * wrap existing `FileArtifactSaver`, `DuckDBRowsSaver`, and `DuckDBIbisTableSaver` only

#### Why this matters

* Guarantees saver tags stay DAG-derivable (your architecture depends on this)
* Makes it trivial to shift outputs between `contract` vs `internal` without editing multiple places
* Enables *factory-driven* node generation later (tool templates / table templates)

---

## 3) Unify run-record construction for mixed outputs

### 3.1 Extend: `src/codeintel/build/hamilton/native/materialization_records.py`

Add the missing “unified record builder” you explicitly called out:

#### Add: `record_from_materializations(...)`

**Goal:** replace ad-hoc mixed-output logic (like `scip.py`’s custom table summarization + `record_from_file_artifact_materializations`) with a single canonical flow.

##### Proposed signature shape

* Inputs:

  * `env: BuildEnv`
  * `graph: TargetGraph`
  * `target_name: str`
  * `artifact_materializations: Mapping[str, MaterializationMetadata] | None`
  * `table_materializations: Mapping[str, MaterializationMetadata] | None`
  * `change_delta: MaterializationDelta | None = None`
  * optional strictness knobs:

    * `require_artifacts: bool | None = None` (default derive from contract/inventory)
    * `require_tables: bool | None = None`
* Output:

  * `TargetRunRecord`

##### Canonical semantics

* **Expected sets** come from output inventory overrides when present, else contract. (mirror existing behavior in `_expected_artifact_names()` and `_expected_table_keys()`.)
* Validate:

  * key sets match expected (extra/missing → fail record)
  * metadata types parse correctly (`FileArtifactMaterializationMetadata.parse`, `DuckDBMaterializationMetadata.parse`)
  * status resolution:

    * any `failed` → `failed`
    * all `skipped` → `skipped`
    * otherwise → `succeeded`
* **Input hash + options hash**

  * require `input_hash` consistency across all provided materializations; mismatches fail with explicit error
  * compute `options_hash` via `options_hash_for_target(env, target_name)` (do not derive from metadata)
* **Row counts**

  * derived from DuckDB materialization metadata row_count (normalize across expected keys)
* **Artifact details**

  * update `ArtifactRef.path` and `size_bytes` from parsed artifact metadata (same as `record_from_file_artifact_materializations`)
* **Manifest persistence**

  * if overall status is `succeeded`, call `save_manifest(...)` once (single source) — do not save separately for artifacts and tables.

##### Implementation approach

Refactor existing `record_from_duckdb_materializations` and `record_from_file_artifact_materializations` into:

* shared internal helpers that:

  * validate + parse + extract (status, input_hash, options_hash, duration, row_counts, artifact_results)
* thin wrappers:

  * `record_from_duckdb_materializations` calls unified helper with tables only
  * `record_from_file_artifact_materializations` calls unified helper with artifacts only
  * `record_from_materializations` calls unified helper with both, and commits manifest once

This aligns with the architecture expectation that **target anchor returns a TargetRunRecord assembled from saver outputs**.

---

## 4) Standard “materialization collectors” to eliminate per-target dict boilerplate

### 4.1 Add: `src/codeintel/build/hamilton/native/patterns/materialization_collectors.py`

**Purpose:** eliminate repeated patterns like:

* `scip__materializations(...) -> dict[...]`
* `scip__table_materializations(...) -> dict[...]`
* `serving_artifacts__materializations_base/extras(...) -> dict[...]`

#### Core idea

You already have stable naming utilities (`materialize_node`, `to_node_name`, etc.)
So build collectors *from declared IO specs*.

#### Public API

* `make_artifact_materializations_collector(*, domain, target, artifacts: Sequence[str], node_name: str | None = None) -> Callable`
* `make_table_materializations_collector(*, domain, target, table_keys: Sequence[str], node_name: str | None = None) -> Callable`
* `make_mixed_materializations_collector(*, domain, target, artifacts, table_keys, node_name: str | None = None) -> Callable`

#### Implementation details

* Use the existing dynamic node plumbing you already have (`module_attach.attach_node` + `signature_tools.set_signature`) to generate a function whose parameters are the `m__*` node names, and whose return is a normalized dict keyed by artifact/table identity.
* Tag collector nodes as `helper` (or compute with helper tags) so they don’t pollute “compute surfaces”.

#### Payoff

* You can convert many targets to:

  * define IO spec constants once (lists of artifact names/table keys)
  * let collectors + unified record builder do the rest
* This sets you up for the next step: **spec-driven target templates**

---

## 5) Spec-driven tool target template

### 5.1 Add: `src/codeintel/build/hamilton/native/patterns/tool_target.py`

This is the keystone that makes “apply the same consolidation pattern across targets” real.

**Purpose:** Provide a reusable “tool-run → optional ingest → materialize outputs → build TargetRunRecord” template where:

* skip logic is canonical (manifest-based)
* tool failure handling is canonical
* output IO is driven by SaveTo + output inventory templates
* the anchor record is built purely from saver metadata (via `record_from_materializations`)

This directly operationalizes the design principle that targets are DAG-defined and incremental behavior is manifest/hash-driven.

#### Recommended types (reuse existing primitives)

Use the existing result containers to avoid type drift:

* `ToolStepOutput` (from `native/tool_results.py`) for tool runs; it already carries `ExecutionResult` and output paths.
* `ExecutionResult` for ingest status; payloads can remain target-specific dataclasses (e.g., `ScipIngestResult`) or a rows-by-table mapping.
* `HasExecutionResult` protocol is the common gate for helper functions that need to branch on success/skipped/failed.

Avoid new per-target “success/skipped/error” dataclasses unless the payload shape truly demands it.

#### Core helpers

1. `run_tool_step(...) -> ToolStepOutput`

   * constructs `NativeTargetExecutor.for_target(...)`
   * calls `.should_skip()` to gate tool execution (exactly as `scip` already does)
   * invokes the tool service call (sync or async)
   * catches exceptions and returns `ToolStepOutput(result=ExecutionResult.failed(...))`

2. `run_tool_and_ingest(...) -> tuple[ToolStepOutput, ExecutionResult | Payload]`

   * executes tool step, then ingest step only if tool succeeded and not skipped
   * keeps ingest payloads separate from tool run outputs to avoid coupling

3. `finalize_target_from_materializations(...) -> TargetRunRecord`

   * performs precondition checks:

     * tool failed → return `executor.fail(...)` (failed target)
     * ingest failed → fail
   * otherwise calls `record_from_materializations(...)`

#### Advanced: attach a full template to a module

Provide:

* `attach_tool_target_template(module: ModuleType, *, spec: ToolTargetSpec, run_fn, ingest_fn, ...) -> None`

Where `ToolTargetSpec` (from `native/patterns/specs.py`) includes:

* `domain`, `target_name`
* `spec_descriptor` for `@codeintel_target(...)`
* `artifacts: Mapping[str, str]` (artifact_name → path_template)
* `tables: Sequence[str]` (table keys)
* optional:

  * `output_roles` per artifact/table (contract/internal)
  * table schema hints (if needed)
  * tool metadata tags (tool id/version), used by `tag_tool(extra_tags=...)`

This function can **auto-attach**:

* `t__<target>__run` (tool node, tagged tool)
* `t__<target>__ingest` (ingest node)
* one artifact compute+saver node per artifact spec
* one rows compute+saver node per table key
* collectors (`<target>__materializations`, `<target>__table_materializations`)
* anchor `t__<target>` calling `record_from_materializations`

This is the point where **adding a new tool target can become “define spec + implement run_fn/ingest_fn”** rather than writing 40–200 lines of repeated scaffolding.

---

## 6) `paths` utilities: make path templates the single source of truth for tool outputs

### 6.1 Add: `src/codeintel/build/hamilton/native/patterns/paths.py`

**Problem:** tool nodes frequently “reconstruct” output paths manually (e.g., `env.paths.scip_dir / ...`), which can drift from the `path_template` defined on saver tags.

**Goal:** Ensure tool steps and artifact nodes resolve paths using the same template system as the artifact saver (your architecture explicitly treats artifact paths as template-driven DAG metadata).

#### Public API

* `resolve_artifact_output_path(env: BuildEnv, *, target: str, artifact: str, fallback_template: str | None = None) -> Path`

* uses saver-derived outputs (via `derive_target_outputs_from_savers`) to resolve templates when available
* otherwise uses provided fallback_template to avoid a second source of truth
  * formats with `format_path_template(...)` and `default_formatter(build_dir, scip_dir, export_dir, repo_root)`

* `resolve_artifact_output_paths(env, *, target, artifacts: Sequence[str]) -> dict[str, Path]`

#### Adoption pattern

* Tool node depends on resolved paths (or resolved output directory) rather than hardcoding.
* Artifact compute nodes can simply return the resolved path (and optionally validate existence after tool succeeds).

This makes IO maximally DAG-driven:

* the template stored in saver tags becomes the authoritative path spec
* tool run writes where the DAG says it writes

---

## 7) Extend `support_factory` so downstream DAG code consumes artifacts with zero ad-hoc glue

Your architecture already supports generating support nodes from contracts or saver-derived outputs depending on settings. Extend that “support surface” slightly so other targets can use artifacts without writing repeated “Path conversion” nodes.

### 7.1 Extend: `src/codeintel/build/hamilton/nodes/support_factory.py`

#### Add optional generation of artifact-path nodes

For each artifact `a__<artifact>` (returns `ArtifactRef`), also generate:

* `p__<artifact>` → `Path | None`

  * returns `Path(artifact_ref.path)` when present
  * intended for downstream targets/consumers; do not use `p__*` inside the same target DAG (it depends on `t__<target>`)

Add a `path_node(...)` helper in `src/codeintel/build/hamilton/naming.py` to keep `p__*` node naming stable.

Add this behind a settings flag or an argument to `build_support_module(...)`:

* `include_artifact_path_nodes: bool = True`

#### Why it’s worth it

* Many downstream consumers want the filesystem path, not just metadata.
* This removes one more per-target boilerplate pattern and further centralizes IO boundary semantics in the DAG’s generated support layer.

(You can go further later: JSON readers, parquet readers, etc., but path nodes are the high‑leverage minimum.)

---

## 8) Migration playbook: how to apply these patterns everywhere (low variance)

Once the library exists, the systematic refactor is straightforward:

### 8.1 First wave: mixed outputs (artifact + tables)

Targets like `scip` should become:

* tool step uses `run_tool_step` (and paths resolved from templates)
* artifact/table saver nodes use `save_artifact` / `save_rows`
* collectors generated via `make_*_collector`
* anchor uses `record_from_materializations` (no bespoke summarizers)

### 8.2 Second wave: artifact-only targets

E.g. modules like `export/serving_artifacts.py` become:

* compute nodes ideally return `ArtifactWritePlan` (to avoid heavy compute on skip)
* saver nodes use `save_artifact`
* anchor uses `record_from_materializations(artifacts=..., tables=None)`

### 8.3 Third wave: table-only targets

Targets that already use `record_from_duckdb_materializations` can stay, but:

* switch to `save_rows` / `save_ibis_table`
* replace ad-hoc `_should_skip_target` utilities with `NativeTargetExecutor.for_target(...).should_skip()` (or with the patterns helper if there’s file-state hashing involved)

### 8.4 Gatekeeping

Update the DAG validator or add a lightweight “lint/validator” to ensure:

* mixed-output targets use `record_from_materializations`
* saver nodes for contract outputs always provide `path_template` via `value(...)` for artifacts (enforced already at decorator level, but validator can catch drift earlier)

### 8.5 Testing and rollout gates

Add explicit tests and checks for the shared libraries before migrating additional targets:

* saver helper tags remain DAG-derivable (static `target_name`, `table_key`/`artifact_name`, `path_template`, `output_role`)
* `record_from_materializations` handles mixed outputs, missing metadata, and input_hash mismatch correctly
* path resolution uses DAG saver templates when available and respects fallback templates otherwise
* tool-target helpers do not bypass manifest-based skip checks
* `build validate` with `enforce_compute_io_purity=True` passes for each migrated target set

---

## 9) Additional “best-in-class” consolidation opportunities enabled by this library

These go beyond the minimum, but they make the system *hard to regress* and *easy to extend*:

### 9.1 “Rows-by-table” as the canonical ingest payload

Standardize ingestion functions to return:

* `dict[table_key, Rows]` (or a `RowsByTable` dataclass)

Then generate all per-table row saver nodes from that mapping using the node factory.
This makes “add a new table output” a **single-line change** (add table_key to spec and rows mapping).

### 9.2 Multi-sink targets as first-class templates

Some targets will write:

* multiple artifacts
* multiple DuckDB tables
* potentially additional sinks later (e.g., object store)

By building `record_from_materializations` and collectors to be sink-agnostic (metadata union), you can later add:

* `record_from_materializations(..., other_sink_materializations=...)`
  without rewriting target anchors.

### 9.3 Tool telemetry + structured logs as a reusable plug-in

Once tool execution is standardized in `tool_target.py`, it becomes trivial to:

* measure durations
* collect tool stdout/stderr summaries
* attach structured metadata into node results for Hamilton telemetry / UI views

(Your architecture explicitly highlights per-node telemetry as a first-class feature in the execution path.)

### 9.4 Derive `TargetResources.tools` from tool tags

Reduce drift between tool nodes and target specs by deriving tools from DAG tags:

* add a canonical tool tag key in `codeintel.core.hamilton.tags` and in `tagging.TagKey`
* apply `tag_tool(extra_tags={...})` to tool nodes with stable tool identifiers
* extend `target_spec_compiler._resources_from_tags(...)` to merge tag-derived tools when the spec does not explicitly set tools (explicit tools remain the override)

---

## 10) What you’ll end up with

After implementing this shared library and migrating representative targets:

* Target modules become **thin DAG specifications**:

  * business logic + minimal spec constants
  * *no bespoke skip logic*
  * *no bespoke record building*
  * *no repeated SaveTo boilerplate*
* Mixed-output targets become as easy to author as table-only targets.
* Support surfaces become richer (artifact path nodes), so downstream DAG code stays minimal.
* IO, orchestration, and incremental behavior are **maximally coupled to DAG-derived saver outputs and manifests**, aligning with your architecture’s core premise.

---

### 10.1 Shared library acceptance criteria

* `native/patterns` is importable and does not register as a target module.
* Mixed-output targets build `TargetRunRecord` solely from saver metadata via `record_from_materializations`.
* Tool nodes resolve output paths from saver-derived templates (with explicit fallback only when tags are missing).
* Tool tag metadata can derive `TargetResources.tools` when explicit tools are not set.
* Support modules can emit `p__*` path nodes when enabled without introducing target cycles.

---

The target template spec catalog follows.

# target template spec catalog #

## Target template spec catalog

Below is a **canonical catalog of 5 target “template specs”** that the repo can converge on, with **clear invariants**, **expected Hamilton DAG shape**, and **where each spec should live** (as reusable “pattern helpers” so every target looks the same).

The goal is: **adding a new target should be mostly “declare outputs + declare dependencies + write pure compute”**, with everything else (skip/hashing, materialization plumbing, run-record construction, artifact/table naming, and telemetry tags) derived from the DAG + spec.

---

## 1) Artifact-only template spec

### When to use

Use when a target’s *only contract outputs* are **file artifacts** and it **does not produce new tables**. Typical for:

* exporting pre-existing warehouse tables (JSONL/Parquet bundles)
* compiling registries/manifests/buildspecs
* producing deterministic serving artifacts

### Contract outputs

* `artifact.<artifact_name>` nodes (1 or many)
* No DuckDB table keys in the target’s contract outputs

### Canonical DAG shape

* `t__<target>__compute_*` (optional): produces either

  * `ArtifactWritePlan` (preferred), or
  * `str | bytes` content (written by FileArtifactSaver), or
  * `Path` (already exists, but this should be avoided unless you can guarantee determinism/location)
* `artifact_*__content` nodes (1 per artifact): **DAG-visible** IO via `SaveToObjectMetadataDecorator([FileArtifactSaver], ...)`
* `t__<target>` anchor: `record_from_file_artifact_materializations(...)` (multi) or `record_from_file_artifact_materialization(...)` (single)

### Required invariants

* **All writes go through FileArtifactSaver** (no ad-hoc `Path.write_text` inside compute).
* Artifact paths must come from **a path template** derived from env paths (build_dir/export_dir/etc.), not computed ad hoc.
* Skip/hashing must be handled consistently:

  * either via `NativeTargetExecutor.should_skip()` at the anchor boundary, or
  * via `should_skip_native_target` in compute producing `None` (then saver sees `None` and skips).

### Recommended shared helper(s)

* `native/patterns/artifact_target.py`:

  * normalize “compute → artifact plan/content → saver → record” boilerplate
  * standardize multi-artifact bundling node (e.g., `artifact__materializations()`)

---

## 2) Tables-only template spec

### When to use

Use when the target’s contract outputs are **only DuckDB tables/views**, and the target is conceptually a *derivation* from the warehouse state (even if the compute is Python-heavy).

Typical for:

* analytics metrics & profiles
* derived graph metrics/views
* “validation outputs” stored as a table (optional)

### Contract outputs

* One or more DuckDB table keys (including view keys like `graph.v_*`)
* No file artifacts

### Canonical DAG shape

Two canonical sub-variants (both count as “tables-only”):

**(A) Ibis table variant**

* `q__...` inputs are Ibis tables
* compute nodes return `ir.Table`
* materialization uses `SaveToObjectMetadataDecorator([DuckDBIbisTableSaver], ...)`

**(B) Rows variant**

* compute nodes return `tuple[tuple[object, ...], ...]` (or `None` if skipped)
* materialization uses `SaveToObjectMetadataDecorator([DuckDBRowsSaver], ...)`

Then always:

* `t__<target>` anchor → `record_from_duckdb_materializations(...)` (multi) or `record_from_duckdb_materialization(...)` (single)

### Required invariants

* **No direct `env.warehouse.materialize_*` inside compute**. The compute must return rows/Ibis expressions; materialization must be DAG-visible via saver nodes.
* **One saver node per output table/view key.** This is what makes outputs fully DAG-derived (and supports inventory checks, caching, telemetry, etc.).
* Standardized naming:

  * `t__<target>__compute` = the “root compute”
  * `<target>__<table_shortname>_rows` or `<target>__<view_shortname>` = per-output emission nodes
  * `m__<schema>__<table>` metadata nodes auto-created by saver decoration
* Validation, schema contracts, and “output_kind=view” tags should attach to the **table-producing nodes**, not the anchor.

### Recommended shared helper(s)

* `native/patterns/table_target.py`:

  * helper to declare output savers with minimal boilerplate (esp. multi-table targets)
* `materialization_records.record_from_materializations(...)`:

  * unify “record_from_duckdb_materialization(s)” into one canonical builder

---

## 3) Tool → Artifacts template spec

### When to use

Use when the target’s *only outputs* are **file artifacts produced by an external tool invocation** (ToolService/ToolRunner), and you do **not** ingest the result into DuckDB as canonical tables.

Typical for:

* building an index/package bundle from a tool
* producing reports, intermediate tool outputs, etc.

### Contract outputs

* Artifacts only

### Canonical DAG shape

* `t__<target>__run` (tag_tool): runs tool via ToolService/ToolRunner and returns a structured result:

  * `ToolStepOutput(result=ExecutionResult, outputs={artifact_name: Path, ...})`
* `artifact_<name>__content|plan` nodes: DAG-visible write via `FileArtifactSaver` (often copy/move into build dir for stability)
* `t__<target>` anchor: records artifacts with `record_from_file_artifact_materializations`

### Required invariants

* External tool execution must be **isolated** to the `__run` node(s) and must not write directly into final serving/export dirs (use intermediate tool dirs + FileArtifactSaver to “bless” outputs into canonical artifact locations).
* Tool version/provenance should be recorded via tags or run metadata fields.

### Recommended shared helper(s)

* `native/patterns/tool_target.py`:

  * standard tool invocation using `ToolStepOutput`
  * standard “tool output dir → canonical artifact path template” mapping

---

## 4) Tool → Ingest → Tables template spec

### When to use

Use when a target:

* reads from **outside** the warehouse boundary (repo filesystem scan, AST parse, coverage/typing tools, etc.), and
* produces **canonical DuckDB tables**.

This is the canonical spec for ingestion-like targets (even if the module is in `graphs/` vs `ingestion/`).

### Contract outputs

* Tables/views only
* No file artifacts (unless you explicitly want to preserve tool outputs as artifacts; then it becomes “Mixed”)

### Canonical DAG shape

* `t__<target>__scan` or `t__<target>__run` (tag_tool): does external boundary work (filesystem, tool service, parsing) and returns `ToolStepOutput` (ExecutionResult + output paths)
* `<target>__<table>_rows` nodes: extract row tuples from the result (or return `None` if skipped/failed)
* per-table saver nodes via `DuckDBRowsSaver` / `DuckDBIbisTableSaver`
* `t__<target>` anchor: uses `NativeTargetExecutor` for consistent failure/skip semantics, and `record_from_duckdb_materializations`

### Required invariants

* Tool/scan nodes must be *pure from DAG perspective* (they can read external state, but must not perform final DB writes themselves).
* Row extraction nodes must be thin and deterministic.
* Input hashing should include:

  * options hash
  * manifest index (already)
  * **file_state hash** (or equivalent) when scanning/parsing repo files

### Recommended shared helper(s)

* `native/patterns/tool_target.py`:

  * standard tool-step output using `ToolStepOutput`
  * standard `hash_options` construction (including file_state hash)
  * standard anchor boundary wrapper

---

## 5) Mixed tool + tables + artifacts template spec

### When to use

Use when a target both:

* runs an external tool and wants to **preserve tool outputs as artifacts**, and
* ingests derived results into **canonical DuckDB tables**.

This is the “best-in-class” pattern for tool-backed indexing because it’s fully reproducible and observable:

* artifacts give you “what the tool produced”
* tables give you “the canonical warehouse state derived from it”

### Contract outputs

* One or more `artifact.*`
* One or more table keys

### Canonical DAG shape

* `t__<target>__run` (tag_tool): run tool → returns tool output paths + execution metadata
* `artifact_*` saver nodes: copy/normalize outputs into canonical artifact locations
* `<target>__ingest` compute nodes: parse artifacts / transform to row payloads
* table saver nodes: materialize derived tables
* anchor `t__<target>`: record both artifacts + tables in **one record**, derived from DAG materialization metadata

### Required invariants

* Tool outputs are “blessed” into canonical artifact paths via FileArtifactSaver.
* No DB writes occur inside tool-run or ingest nodes; only via saver nodes.
* Tool run, artifact materialization, ingest, and table writes stay in a single target DAG; do not split into separate targets for the same calculation scope.
* The anchor must construct the run record *only* from:

  * `MaterializationMetadata` for artifacts/tables
  * DAG-derived row counts (not recomputed)

### Recommended shared helper(s)

* `native/patterns/mixed_target.py`

  * a single helper that enforces “run → artifacts → ingest → tables → record”
  * standard materialization bundling nodes so the anchor is minimal

---

# Template migration mapping for current native modules

This mapping is based on the modules currently loaded by `build/hamilton/native/discovery.py` (domains: ingestion, graphs, analytics, export). The intent is: **after migration, each module should conform to one template spec**; if a module spans multiple specs, **split it**.

> Legend:
> **AO** = Artifact-only
> **TO** = Tables-only
> **TA** = Tool→Artifacts
> **TT** = Tool→Ingest→Tables
> **MX** = Mixed tool + tables + artifacts

---

## Mapping table

| Domain    | Native module                          | Primary targets inside                                                                                                                   | Current output shape                         | Template it should migrate to     | Notes to converge / split                                                                                                                                                                        |
| --------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ingestion | `ingestion/extraction_targets.py`      | `ast`, `cst`, `docstrings`                                                                                                               | tables                                       | **TT**                            | Already very close; keep external parsing in tool nodes, rows via savers, records via `record_from_duckdb_materializations`.                                                                     |
| ingestion | `ingestion/ingest_targets.py`          | `modules`, `config_ingest`, `coverage_ingest`, `tests_ingest`, `typing`                                                                  | tables                                       | **TT**                            | Canonical TT module. Biggest win is further shared helpers for repeated skip/hash/record boilerplate.                                                                                            |
| ingestion | `ingestion/scip.py`                    | `scip`                                                                                                                                   | artifacts + tables                           | **MX**                            | This is your representative “MX” archetype; keep as the gold standard for tool-backed ingestion.                                                                                                 |
| graphs    | `graphs/call_graph.py`                 | `call_graph`                                                                                                                             | tables (but writes inside compute today)     | **TT**                            | Needs migration away from `env.warehouse.materialize_*` inside `__extract`; return rows/Ibis and let savers materialize. Keep “large data” optimizations via streaming saver or chunked rows.    |
| graphs    | `graphs/cfg_dfg.py`                    | `cfg`, `dfg`                                                                                                                             | tables (but writes inside compute today)     | **TT**                            | Same migration as call_graph: compute → rows → saver nodes; remove marker-only savers once real savers exist.                                                                                    |
| graphs    | `graphs/import_graph.py`               | `import_graph`                                                                                                                           | tables (but writes inside compute today)     | **TT**                            | Same TT migration; keep AST parse in tool node, return row tuples for modules/edges.                                                                                                             |
| graphs    | `graphs/graph_targets.py`              | `goids`, `symbol_uses`, `call_graph_views`, `graph_metrics`, `graph_validation`                                                          | mixed table derivations + ingest-like writes | **SPLIT** → **TT + TO**           | This module spans two templates. Split to minimize variance: (1) ingest-like: goids/symbol_uses → **TT**, (2) derived views/metrics/validation → **TO**. See recommended split below.            |
| analytics | `analytics/classification_targets.py`  | `semantic_roles`, `test_profile`                                                                                                         | tables                                       | **TO**                            | Pure compute → rows → savers. Already aligned.                                                                                                                                                   |
| analytics | `analytics/config_graph_targets.py`    | `config_data_flow`, `cfg_dfg_metrics`                                                                                                    | tables                                       | **TO**                            | Keep as TO; use standardized multi-table bundling helper.                                                                                                                                        |
| analytics | `analytics/coverage_targets.py`        | `coverage_functions`, `coverage_test_edges`, `behavioral_coverage`                                                                       | tables/views                                 | **TO**                            | It’s the archetype “TO with ibis + rows”. Great template example.                                                                                                                                |
| analytics | `analytics/dependency_targets.py`      | `external_deps`, `entrypoints`                                                                                                           | tables                                       | **TO**                            | TO but with an ordering constraint; encode dependency through DAG nodes (materialization metadata → next compute) so it’s explicit.                                                              |
| analytics | `analytics/execution_context.py`       | (support nodes)                                                                                                                          | no target                                    | **TO (support primitives)**       | This isn’t a target module. Either (a) keep as support primitives, or (b) move into generated support module / shared support package and remove from discovery to reduce “native surface area”. |
| analytics | `analytics/function_detail_targets.py` | `function_contracts`, `function_effects`                                                                                                 | tables                                       | **TO**                            | Already canonical “compute result container → rows → saver → record”.                                                                                                                            |
| analytics | `analytics/function_metrics.py`        | `function_metrics`                                                                                                                       | tables                                       | **TO**                            | Multi-table rows pattern; keep as reference for TO multi-table.                                                                                                                                  |
| analytics | `analytics/hotspots.py`                | `hotspots`                                                                                                                               | tables                                       | **TO**                            | Even though it shells out to git, output shape is tables-only. Keep it TO; if you want strict separation of “tool boundary”, you can tag the git call node as tool, but template remains TO.     |
| analytics | `analytics/metadata_targets.py`        | `data_models`, `data_model_usage`, `function_ast_features`, `profiles`                                                                   | tables                                       | **TO**                            | Large multi-target module, but all TO; optionally split by “theme” for readability (still TO).                                                                                                   |
| analytics | `analytics/metrics_targets.py`         | `function_history`, `history_timeseries`, `subsystem_graph_metrics`, `symbol_graph_metrics`, `subsystem_agreement`, `test_graph_metrics` | tables                                       | **TO**                            | All TO (rows). If you later enforce “tool boundary nodes” for git/test runtime, you still keep TO because outputs are tables.                                                                    |
| analytics | `analytics/risk_factors.py`            | `risk_factors`                                                                                                                           | tables/views                                 | **TO**                            | Archetypal TO ibis pipeline; keep as a canonical example.                                                                                                                                        |
| analytics | `analytics/subsystem_cache_targets.py` | `subsystem_caches`                                                                                                                       | tables                                       | **TO**                            | Canonical TO.                                                                                                                                                                                    |
| analytics | `analytics/subsystem_targets.py`       | `subsystems`                                                                                                                             | tables                                       | **TO**                            | Canonical TO.                                                                                                                                                                                    |
| export    | `export/export_targets.py`             | `export_jsonl`, `export_parquet`                                                                                                         | artifacts                                    | **AO**                            | Even if internal compute is tagged tool, this is “artifact-only”: no tables, only FileArtifactSaver outputs.                                                                                     |
| export    | `export/serving_artifacts.py`          | `serving_artifacts`                                                                                                                      | artifacts (plus hidden DB side effects)      | **AO** (recommended) or **SPLIT** | Best convergence is: keep this module **AO** (pure artifact compilation), and move lineage sync into a separate **TO** target (or explicitly declare lineage tables as outputs and treat as MX). |

---

## Recommended split plan for `graphs/graph_targets.py`

To eliminate the only major “multi-template module” in the discovered native set, split into **two TO/TT-conformant modules** (or 3–5 if you want ultra-granular convergence):

### Minimal split (2 modules)

1. `graphs/graph_ingestion_targets.py` → **TT**

* `goids`
* `symbol_uses`

2. `graphs/graph_derived_targets.py` → **TO**

* `call_graph_views`
* `graph_metrics`
* `graph_validation`

### Granular split (5 modules)

* `graphs/goids_targets.py` (**TT**)
* `graphs/symbol_uses_targets.py` (**TT**)
* `graphs/call_graph_views_targets.py` (**TO**, view materialization)
* `graphs/graph_metrics_targets.py` (**TO**, rows materialization)
* `graphs/graph_validation_targets.py` (**TO**, and decide whether it actually materializes `analytics.graph_validation` or is a pure check target with no table output)

This split is “minimal variance” friendly because each module becomes a single template archetype and can be generated/maintained with the same scaffolding.

---

# Small set of “repo convergence rules” to enforce templates

If you want this to stay best-in-class as you add new targets, these rules prevent drift:

1. **No warehouse writes inside compute/tool nodes**
   All DB writes go through saver decorators (`DuckDBRowsSaver`/`DuckDBIbisTableSaver`).

2. **No file writes inside compute/tool nodes**
   All final artifact writes go through `FileArtifactSaver` with canonical path templates.

3. **Anchors only “record”**
   `t__<target>` should become almost boilerplate: validate skip/fail and convert materialization metadata → `TargetRunRecord`.

4. **Every contract output must correspond to a DAG node**
   If a target claims it outputs `analytics.graph_validation`, there must be a node producing the rows/Ibis for it and a saver node.

5. **Mixed modules must be split**
   A native module should be **one template spec**. If it isn’t, split.

---

If you want, I can also generate a **“template adoption checklist”** for each spec (AO/TO/TT/TA/MX) that’s phrased as a step-by-step migration rubric you can apply module-by-module without thinking—i.e., the exact sequence of transformations needed to convert a “manual materialize + marker saver” module (like `call_graph.py`) into a fully DAG-visible TT module.

Below is a **module-by-module “template adoption checklist”** for each canonical spec (AO/TO/TT/TA/MX). It’s written as a **mechanical migration rubric**: you can apply it in the same order every time, with minimal judgment calls, and end up with a target that is **maximally DAG-derived** (contract + IO surface + orchestration + support-node exposure all come from the Hamilton graph).

I’m grounding this in your architecture conventions: stable node naming via `naming.py` (`t__`, `m__`, `d__`, `q__`, `df__`, `a__`), **target anchors** as `t__<target>` materialize nodes, and **contract outputs derived from DataSaver tags** (`output_role`, `table_key`/`artifact`, `artifact_path_template`). I also assume you’re leaning into **saver-derived outputs** for support surfaces (via `DerivedTargetOutputs`) as the steady-state.

---

## Global invariants you apply in every module before choosing AO/TO/TT/TA/MX

These are “do this or you’re not actually DAG-native”.

1. **Use stable node names via naming helpers**

* Always name:

  * targets with `t__*` (`target_node(...)`)
  * materializers with `m__*` (`materialize_node(...)`)
  * support nodes are auto-generated as `d__*`, `q__*`, `df__*`, `a__*`
* **Never invent ad-hoc prefixes** in a module. Your ability to compile/compare/spec-diff depends on stable identity.

2. **Every target must have a real DAG-native anchor**

* Implement `t__<target>` and decorate with `@codeintel_target(domain=..., target=..., spec=...)`.
* Anchor must be `node_type="materialize"` and include `domain`, `target`, and `target_spec_version == "1"`.
* Anchor output is the per-target runtime record (`TargetRunRecord`).

3. **All writes must be represented by real saver nodes, not side effects**

* Persisting tables/artifacts must occur through `SaveToObjectMetadataDecorator(...).transform_node(...)` (producing `m__*` nodes that return `MaterializationMetadata`).
* Avoid “manual write + marker saver” entirely. That pattern makes the DAG *declare* IO without *executing* IO at the boundary, and you lose reliable IO surface derivation.

4. **Contract outputs are derived from DataSaver tags**

* All “real outputs” must come from DataSaver nodes with:

  * `output_role` ∈ {"contract","internal"}
  * Exactly one identity: `{table_key}` **or** `{artifact}`
  * If artifact contract output: must include `artifact_path_template`
    This is what makes `OutputTarget` compilation truly DAG-native.

5. **Support surfaces should be derivable from saver-derived outputs**

* Support nodes can be generated from contracts or from saver-derived `DerivedTargetOutputs`.
* The “endgame” is: **support nodes derive from saver tags**, not a parallel manually-maintained contract surface.

6. **Output inventory is DAG-derived in steady state**

* Inventory and templates are derived from saver tags, not a parallel declared source of truth.
* Helpers that read inventory data should accept a fallback template/value only for targets that are not yet fully tagged.

7. **Execution model assumptions**

* Build executor runs Hamilton with `inputs={"env": env, "graph": graph}` and `final_vars=[t__* anchors]`.
  So everything you want to happen must be reachable from the anchor through the dependency graph.

---

## AO checklist — Artifact-only target

**Spec definition:** no tool step, no tables. Only produces artifacts (files) as outputs.

### AO migration rubric (mechanical)

1. **Define your spec constants (top of module)**

* `DOMAIN`, `TARGET_NAME`
* `ARTIFACTS = { artifact_name: path_template }`
* Decide which artifacts are **contract** vs **internal debug**.

2. **Create (or refactor into) pure compute nodes for each artifact**

* One compute node per artifact.
* Output should be one of the supported artifact payloads (commonly `Path | None`) and be deterministic from inputs.
* Tag it as compute and ensure it carries the `domain/target` context (so savers inherit domain).

3. **Attach the artifact saver metadata node**

* Wrap each compute node with `SaveToObjectMetadataDecorator([FileArtifactSaver], ...)`.
* Provide **static identity** using `value(...)` for:

  * `target_name=value(TARGET_NAME)`
  * `artifact_name=value(<artifact_name>)`
  * `path_template=value(<template>)`
    This is non-negotiable for contract derivation and artifact template indexing.

4. **Aggregate artifact materializations**

* Create a helper node that returns `dict[artifact_name, MaterializationMetadata]` from the saver nodes (`m__artifact__...`).

5. **Implement the anchor `t__<target>`**

* Decorate with `@codeintel_target(domain=..., target=..., spec=...)` and ensure tags are valid.
* Anchor should:

  * validate upstream status (if any deps are `TargetRunRecord`)
  * build the target record using the artifact materializations (the “file artifact materializations” record builder pattern is explicitly called out in the architecture walkthrough).
* Anchor must return `TargetRunRecord`.

6. **Delete all legacy IO and completion signals**

* Remove:

  * manual file writes from compute nodes (only allowed inside the saver or a tool node)
  * marker files
  * any manual manifest/record persistence (the run record layer should own it)

7. **Done checks**

* Contract compilation yields your artifact list and templates.
* Support nodes `a__<artifact_name>` appear when support nodes are derived and enabled.
* IO surface introspection sees artifact writes via saver tags.

### AO “advanced opt-ins”

* If artifact generation is large, return a streaming-friendly payload type (or a write-plan object) so compute stays pure and savers perform IO.
* If you need per-artifact incremental skip beyond target-level skipping, you can add an internal node that computes “artifact-specific hash” and have the compute return `None` early when unchanged (still letting the saver mark “skipped”).

---

## TO checklist — Tables-only target

**Spec definition:** no tool step, no artifacts. Only table outputs.

### TO migration rubric (mechanical)

1. **Define constants**

* `DOMAIN`, `TARGET_NAME`
* `TABLE_KEYS = [ ... ]` (fully qualified keys like `graph.call_graph_nodes`)

2. **Add one compute node per table output**
   Choose one pattern per table (don’t mix inside a single table):

* **Ibis expression** path: compute returns an ibis table expr → use `DuckDBIbisTableSaver`
* **Rows** path: compute returns `Iterable[tuple] | None` → use `DuckDBRowsSaver`

3. **Attach the table saver metadata node**

* Wrap each table compute node with `SaveToObjectMetadataDecorator([DuckDBIbisTableSaver] | [DuckDBRowsSaver], ...)`.
* Identity must be static:

  * `target_name=value(TARGET_NAME)`
  * `table_key=value(<table_key>)`
* If using rows saver, ensure schema/columns are resolvable (registry requirement is called out).

4. **Eliminate manual warehouse materialization**

* No `env.warehouse.materialize_*` inside compute nodes.
* The saver does the write boundary (and produces `MaterializationMetadata`).

5. **Aggregate table materializations**

* `dict[table_key, MaterializationMetadata]` helper node.

6. **Anchor `t__<target>`**

* Decorate and tag properly.
* Build record from table materializations (DuckDB materialization record builder pattern).
* Fail fast if any table materialization reports failure.

7. **Optional skip gating (high leverage for expensive TO)**

* Add a helper node `t__<target>__skip` (or similar) that computes skip once (via the native skip mechanism) and threads it into each compute node so they can return `None` early.
  This prevents expensive row generation when the run should skip.

8. **Done checks**

* Contract compilation derives table outputs from saver tags (no markers).
* Support nodes `d__<table_key>`, `q__<table_key>`, `df__<table_key>` are generated when enabled.
* IO registry and IO surface derivation see table writes via saver tags.

### TO “advanced opt-ins”

* Prefer Ibis expressions when possible: it keeps compute nodes pure and lets the saver push work into DuckDB efficiently.
* If a TO target generates multiple tables with shared upstream logic, create upstream helper nodes that produce intermediate ibis relations or typed frames, then derive each table expr from those helpers.

---

## TA checklist — Tool → Artifacts target

**Spec definition:** there is a tool/external step; outputs are artifacts only.

This is the template that replaces the “manual materialize + marker saver” anti-pattern for artifact-producing modules.

### TA migration rubric (mechanical)

1. **Define constants**

* `DOMAIN`, `TARGET_NAME`
* `ARTIFACTS = { artifact_name: path_template }`
* Use `ToolStepOutput` for the tool step output (result + output paths).

2. **Implement the tool node `t__<target>__run`**

* Tag it as tool (not materialize).
* It should:

  * compute skip once (native skip check pattern is explicitly in the scip walkthrough)
  * if skip → return `ToolStepOutput(result=ExecutionResult.skip(...))` (no IO)
  * else run the tool, producing outputs in canonical locations (typically under a build dir)
  * return structured paths/metadata
    This node is where “imperative orchestration” lives.

3. **Implement one artifact compute node per artifact**

* Each node depends on `t__<target>__run`.
* If `run.result.skipped` or not `run.result.success` → return `None`.
* Else return `Path` to the artifact (or another supported artifact payload).

4. **Attach `FileArtifactSaver` to each artifact compute node**

* Use `SaveToObjectMetadataDecorator([FileArtifactSaver], ...)` with:

  * `target_name=value(TARGET_NAME)`
  * `artifact_name=value(...)`
  * `path_template=value(...)`
* Ensure these are **real contract outputs**, not markers.

5. **Aggregate artifact materializations**

* `dict[artifact_name, MaterializationMetadata]` helper node.

6. **Anchor `t__<target>`**

* Decorate with `@codeintel_target(...)` and meet invariants.
* Validate tool success:

  * If tool failed, fail the target (don’t pretend outputs exist).
* Build the record from artifact materializations (pattern shown in the scip walkthrough).

7. **Delete markers and manual completion**

* Remove any “marker saver nodes” that return `None`.
* Remove any marker file writing; if an external system requires a marker file, treat it as:

  * an **internal artifact output** (output_role=`internal`) saved via `FileArtifactSaver`, or
  * a contract artifact if it’s truly part of the public IO surface.

8. **Done checks**

* Contract outputs (artifacts + templates) are derivable from saver tags.
* Support nodes `a__<artifact_name>` exist when enabled.
* IO surface introspection shows artifact writes for the target.

### TA: exact “manual materialize + marker saver” → DAG-visible TA conversion sequence

This is the “no thinking” conversion recipe when you see a legacy module doing:

* tool runs and writes files manually
* a “marker saver node” exists solely so the contract compiler thinks there are outputs

**Step-by-step transformation:**

1. **Identify what the marker is pretending to represent**

* List the “declared outputs” from marker savers (artifact names and templates).
* List the “actual outputs” written manually by the module.

2. **Replace every marker saver with a real artifact compute node**

* The compute node returns a `Path` to the manually created file (or returns the bytes/string to be written).
* It must depend on the tool run node.

3. **Wrap that compute node with a `FileArtifactSaver`**

* This makes the saver boundary authoritative, and your contract derivation becomes honest.
* If the tool already wrote the file, the compute node can return the `Path` and the saver can “adopt/copy/validate” depending on how you want to enforce destination layout.

4. **Move all “where the file lives” logic into `path_template`**

* Stop embedding output paths throughout the tool logic.
* Tool writes to a temp/run dir; saver owns “final” path resolution via template.

5. **Remove all marker file logic**

* If you still need a marker, model it as a first-class artifact output (internal or contract), not an implicit side effect.

6. **Make the anchor depend on artifact materializations**

* Anchor returns `TargetRunRecord` assembled from artifact materializations, not from “I wrote a marker so I’m done”.

7. **Delete manual manifest/run record writing**

* The target record builder path and saver metadata should drive incremental behavior; don’t maintain two sources of truth.

---

## TT checklist — Tool → Ingest → Tables target

**Spec definition:** tool step exists; outputs are tables only; tool output may be ephemeral.

### TT migration rubric (mechanical)

1. **Define constants**

* `DOMAIN`, `TARGET_NAME`
* `TABLE_KEYS = [...]`
* Define tool output using `ToolStepOutput` and an ingest payload wrapper that carries `ExecutionResult`.

2. **Tool node `t__<target>__run`**

* Perform native skip check once (like scip does).
* If skip: return `ToolStepOutput(result=ExecutionResult.skip(...))`; do not do IO.
* Else: run tool and return the raw outputs required for ingest (paths or in-memory structures).

3. **Ingest node(s)**

* Convert tool outputs into normalized row payloads or ibis expressions.
* This node should contain parsing/normalization logic, not persistence.
* Return `ExecutionResult` alongside payloads so downstream nodes can gate on `success/skipped`.

4. **One compute node per table**

* Depends on ingest output.
* Returns rows / ibis expr, or `None` if skipped/failed.

5. **Attach a table saver metadata node to each table compute**

* `DuckDBRowsSaver` or `DuckDBIbisTableSaver` with static identity:

  * `target_name=value(TARGET_NAME)`
  * `table_key=value(TABLE_KEY)`

6. **Aggregate table materializations**

* `dict[table_key, MaterializationMetadata]`.

7. **Anchor `t__<target>`**

* Validate tool result and table materializations.
* Build `TargetRunRecord` from the table materializations.

8. **Delete manual writes and marker savers**

* No `warehouse.materialize_*` in compute/ingest nodes.
* No marker savers that return `None`.

9. **Done checks**

* Contract compilation derives tables from saver tags.
* Support nodes for datasets/loaders appear when enabled.
* IO surface introspection shows table writes.

### TT “advanced opt-ins”

* If ingest is expensive and skip is common, thread the tool skip state into ingest and table compute nodes so they can return `None` early.
* If ingest uses large intermediates, persist those as **internal artifacts** (output_role=internal) only when debug is enabled; otherwise keep them ephemeral.

---

## MX checklist — Mixed tool + tables + artifacts target

**Spec definition:** tool exists; the target has both artifact outputs and table outputs (typically both are part of the “contract surface”).

This is the “scip-like” template; the architecture walkthrough explicitly frames scip as the representative end-to-end mixed target.

### MX migration rubric (mechanical)

1. **Define constants**

* `DOMAIN`, `TARGET_NAME`
* `ARTIFACTS = {artifact_name: path_template}`
* `TABLE_KEYS = [...]`
* Use `ToolStepOutput` for the tool step output (result + output paths).

2. **Tool node `t__<target>__run`**

* Single point for:

  * skip decision (native skip check)
  * tool execution orchestration
  * returning structured result

3. **Artifact compute nodes + `FileArtifactSaver`**

* One compute per artifact (depends on tool result).
* Wrap with `SaveToObjectMetadataDecorator([FileArtifactSaver], target_name=value(...), artifact_name=value(...), path_template=value(...))`.

4. **Ingest nodes**

* Parse tool outputs into normalized structures used by table compute nodes.

5. **Table compute nodes + DuckDB savers**

* One compute per table.
* Wrap with `SaveToObjectMetadataDecorator([DuckDBRowsSaver], output_name_=materialize_node(table_key), ...)` (scip pattern).

6. **Aggregate artifact materializations + table materializations**

* Two helper nodes:

  * artifacts: `dict[artifact_name, MaterializationMetadata]`
  * tables: `dict[table_key, MaterializationMetadata]`

7. **Anchor `t__<target>`**

* Decorate with `@codeintel_target(...)` and meet invariants.
* Validate:

  * tool success
  * table materializations don’t include failures
* Build `TargetRunRecord` such that both:

  * artifact refs are included
  * row_counts (from table writes) are included
    The scip anchor pattern explicitly uses “artifact materializations + row counts” to assemble the record.

8. **Delete marker savers and manual writes**

* No markers that return `None` to declare outputs.
* No direct `warehouse.materialize_*` in tool/ingest/compute nodes.

9. **Done checks**

* Contract derivation sees both:

  * tables (from saver tags)
  * artifacts + templates (from saver tags, including `artifact_path_template`)
* Support nodes include both dataset loaders and artifact refs.

### MX “advanced opt-ins”

* Add internal “debug artifacts” (output_role=internal) for tool stdout/stderr, extracted raw payloads, or row sampling—these become first-class DAG outputs without polluting the public contract.
* Push more into derived IO: ensure IO registry compilation (`compile_write_registry`) and target IO surface derivation remain accurate for mixed targets.

---

## “Definition of Done” checklist you can apply to every migrated module

Run these after each conversion, regardless of template:

1. **Graph invariants**

* Target anchor has proper domain/target/spec tags and `target_spec_version=="1"`.

2. **Contract derivation sanity**

* All contract outputs appear as saver-derived outputs:

  * `output_role="contract"`
  * identity is exactly one of `{table_key, artifact}`
  * artifacts include `artifact_path_template`

3. **Support surface**

* Support nodes appear for the outputs you expect via saver-derived outputs, and are usable by downstream targets (`d__/q__/df__/a__`).

4. **IO surface introspection**

* `derive_target_io_surface(...)` shows the writes you think the target performs (this is how you keep orchestration and “what happens” DAG-derived).

5. **Execution correctness**

* Running build executes `t__*` anchors as final vars with `inputs={"env": env, "graph": graph}` and the target’s writes happen through saver nodes, not manual side effects.

---

## Migration gate checklist (PR-ready)

Use this block verbatim in PRs that migrate a target to DAG-derived IO:

* **Settings**: no compatibility toggles; outputs/support nodes are derived from saver tags.
* **Validation**: `build validate` includes `validate_graph(..., enforce_compute_io_purity=True)`; no drift or IO-purity violations.
* **Inventory**: DAG-derived inventory contains all new outputs/templates without relying on declared inventory.
* **Hashing**: shared `*_hash_options` node is used by tool execution, savers, and record builders (no duplicated hash logic).
* **Reuse**: tool reuse is gated by `expected_input_hash`; no filesystem short-circuiting when manifest says rerun.
* **Module-State**: `core.scip_module_state` is populated for the snapshot and shard manifests regenerate from it when missing.
* **Merge Policy**: symbol_information and external_symbols merge rules are deterministic and covered by regression tests.

---

If you want, I can also produce a **“conversion playbook page”** that’s literally a one-page set of regex/find patterns (e.g., `executor_materialize`, `warehouse.materialize_*`, marker savers returning `None`, direct `save_manifest` calls) → “replace with template steps X/Y/Z” mapping, so a developer can mechanically process the repo like a migration pipeline.

# conversion playbook #

## Conversion playbook page: repo-wide mechanical find patterns → template steps

This is meant to be used **literally as a migration pipeline**: pick a native module, run the “spec classifier” greps, then apply every matching rewrite rule below (in order). The “template steps” referenced correspond to the AO/TO/TT/TA/MX adoption checklists you already have.

---

### A) 15-second template spec classifier (pick exactly one: AO / TO / TT / TA / MX)

Run these greps in the module (or whole native subtree) and classify by **features present**:

**Tool present?**

* `@tag_tool\b`
* `env\.providers\.tool_service\.`
* `ToolService`
* `asyncio\.run\(`
* `NativeTargetExecutor\.for_target\(.*\)\.should_skip\(` (or `executor\.should_skip\(\)`)

**Table outputs present?**

* `DuckDBRowsSaver|DuckDBIbisTableSaver`
* `table_key=value\(`
* `materialize_node\("?[a-z0-9_]+\.[a-z0-9_]+"?\)` (DuckDB table keys)
* `m__.*__.*:\s*MaterializationMetadata` in `t__<target>(...)` signature

**File artifacts present?**

* `FileArtifactSaver`
* `artifact_name=value\(`
* `path_template=value\(`
* `output_name_=materialize_node\("artifact\.`

**Choose spec**

* **AO**: artifacts only; no tools; no tables.
* **TO**: tables only; no tools; no file artifacts.
* **TT**: tool + tables; artifacts optional but not “first-class deliverables”.
* **TA**: tool + artifacts only; no tables.
* **MX**: tool + tables + artifacts are all first-class deliverables (or module mixes multiple output “kinds” and you want one unified pattern).

---

### B) Rewrite rules (regex/find → “replace with template steps X/Y/Z”)

Apply **every** rule that matches. If multiple specs apply, always choose the **highest**: `MX > TT > TA > TO > AO`.

---

#### Rule 1 — `executor_materialize(...)` (legacy “executor boundary” finalizer)

**FIND**

* `executor_materialize\(`

**MEANS**

* The module’s “target node” is not DAG-driven by saver metadata; it’s doing orchestration/final status outside the materialization nodes.

**REPLACE WITH**

* **TO**: `TO-5` + `TO-6`
* **TT**: `TT-6` + `TT-7`
* **TA**: `TA-5` + `TA-6`
* **MX**: `MX-6` + `MX-7`

**DO**

* Delete `executor_materialize(...)` from `t__<target>` and instead make `t__<target>` depend on `m__...` saver metadata nodes and return `record_from_*` (or the MX record builder) *only*.
* Delete any intermediate `*_execution_result` adapter nodes **unless** they are truly used as semantic outputs (they usually aren’t after migration).

---

#### Rule 2 — Manual warehouse writes inside compute (`env.warehouse.materialize_*` / `warehouse.materialize_*`)

**FIND**

* `env\.warehouse\.materialize_(rows|table|dataframe|mappings)\(`
* `warehouse\.materialize_(rows|table|dataframe|mappings)\(`

**MEANS**

* IO is happening “inside compute”, which breaks DAG visibility, caching semantics, and materialization uniformity.

**REPLACE WITH**

* **TO**: `TO-2` + `TO-4`
* **TT**: `TT-4` + `TT-5`
* **TA**: reclassify to **TT/MX** (TA targets should not write tables)
* **MX**: `MX-3` + `MX-5`

**DO**

* Refactor compute nodes to **return** the data (rows or ibis table), and let the saver (`DuckDBRowsSaver` / `DuckDBIbisTableSaver`) do the write.
* The *only* materialization should occur in saver nodes created via `@SaveToObjectMetadataDecorator(...)`.

---

#### Rule 3 — Marker saver nodes returning `None` (aka “fake outputs”)

**FIND**

* `def .*_marker\(`
* plus either:

  * `return None` (unconditional), **or**
  * docstring phrase: `used only for metadata`

**MEANS**

* The module is “declaring outputs” without connecting real computed values → saver nodes. This is the biggest “DAG invisibility” smell.

**REPLACE WITH**

* **TO**: `TO-2` + `TO-3` + `TO-4`
* **TT**: `TT-2` + `TT-3` + `TT-4` + `TT-5`
* **TA**: `TA-2` + `TA-3` + `TA-4` + `TA-5`
* **MX**: `MX-2` + `MX-3` + `MX-4` + `MX-5`

**DO**

* Delete marker nodes entirely.
* Replace with *real output nodes* per table/artifact that:

  * return rows/ibis table/path, and
  * are decorated with the saver decorator.
* Ensure `t__<target>` consumes `m__table_key` / `m__artifact_name` metadata nodes (not markers, not ad-hoc “execution results”).

---

#### Rule 4 — `MaterializeOptions` / `materialize_options(...)` living in native modules

**FIND**

* `MaterializeOptions`
* `materialize_options\(`
* `append_materialize_options\(`

**MEANS**

* Options/config are being computed in-module (or per-target), rather than being standardized in savers/templates.

**REPLACE WITH**

* **TO/TA/MX**: treat as part of `*-3` + `*-5` (saver/template centralization)

**DO**

* Remove per-target `materialize_options(...)` calls.
* Ensure savers are passed only **env/graph/target_name/table_key** and let the saver compute options consistently.

---

#### Rule 5 — Explicit “skip” logic that bypasses saver/manifest semantics

**FIND**

* `if .*should_skip\(\):`
* `_should_skip_`
* `should_skip_native_target\(`
* `options_hash_for_target\(` + `compute_input_hash\(` (paired pattern)

**MEANS**

* The module has bespoke skip logic that may diverge from saver/manifest-derived skip semantics.

**REPLACE WITH**

* **TO**: `TO-1` + `TO-3` cleanup
* **TT/TA/MX**: `TT-2` or `TA-2` or `MX-2` cleanup

**DO**

* Prefer **one** skip authority:

  * If output is persisted via saver → let saver context/manifest decide.
  * If tool run is expensive → keep skip check at `t__target__run` level, but derive it from the same input-hash mechanism the saver uses (not a separate bespoke scheme).
* Delete `_should_skip_*` helpers once the template absorbs them.

---

#### Rule 6 — `t__<target>` signature does *not* include any `m__...: MaterializationMetadata`

**FIND**

* `def t__\w+\([\s\S]*\)\s*->\s*TargetRunRecord:`
  and within that signature:

  * **no** `m__` parameters

**MEANS**

* The target record is being produced without reading the DAG’s materialization truth.

**REPLACE WITH**

* **AO/TO/TT/TA/MX**: `*-3` / `*-4` (depends on spec)

**DO**

* Change `t__<target>` to consume the saver metadata nodes:

  * tables: `m__<schema>__<table>`
  * artifacts: `m__artifact__<name>`
* Make `t__<target>` a pure “record builder” node (no IO, no tool run, no heavy compute).

---

#### Rule 7 — Per-target “summarize materializations” functions

**FIND**

* `def _summarize_.*materializations\(`
* `DuckDBMaterializationMetadata\.from_mapping\(`
* `FileArtifactMaterializationMetadata\.from_mapping\(` (or similar typed parse functions)

**MEANS**

* Each module is re-implementing status derivation, which causes drift and inconsistent failure semantics.

**REPLACE WITH**

* **TO**: `TO-6` (use centralized record_from_duckdb_materializations semantics)
* **TT**: `TT-7` (use record_from_duckdb_materializations semantics)
* **TA**: `TA-6` (use record_from_file_artifact_materializations semantics)
* **MX**: `MX-7` (use unified mixed record builder)

**DO**

* Delete the summarizer.
* Push “metadata interpretation” into:

  * `record_from_duckdb_materializations`
  * `record_from_file_artifact_materializations`
  * or the new MX unifier (`record_from_materializations` / `record_from_mixed_materializations`).

---

#### Rule 8 — “Artifact existence checks” instead of manifest/saver checks

**FIND**

* `Path\(.*\)\.exists\(\)`
* `if output_.*\.exists\(\)`
* `if .*index_path.*` or `if output_scip.exists()` style existence gating

**MEANS**

* Artifact caching semantics are file-system-based rather than manifest/DAG-based (causes partial builds + drift).

**REPLACE WITH**

* **AO**: `AO-2` + `AO-3`
* **TA**: `TA-3` + `TA-4`
* **MX**: `MX-3` + `MX-4` (remove existence checks; rely on savers + ingest nodes)

**DO**

* Artifact nodes should simply return the intended path (or `None` on failure/skip).
* Let `FileArtifactSaver` decide whether to write/skip based on manifest context.

---

#### Rule 9 — Tool run nodes returning “raw” types and scattering tool output parsing

**FIND**

* `env\.providers\.tool_service\.\w+`
* `asyncio\.run\(env\.providers\.tool_service\.\w+`
* `result\.documents` / `result\.stdout` / `result\.paths` pattern usage in multiple downstream nodes

**MEANS**

* Tool execution boundary is not stable/typed; downstream parsing is repeated.

**REPLACE WITH**

* **TT**: `TT-1` + `TT-2`
* **TA**: `TA-1` + `TA-2`
* **MX**: `MX-1` + `MX-2`

**DO**

* Make exactly one `t__<target>__run` node that returns a **typed dataclass**.
* Downstream nodes accept that dataclass and do deterministic parsing/transforms.

---

#### Rule 10 — Repeated `@SaveToObjectMetadataDecorator(...)` boilerplate blocks

**FIND**

* `@SaveToObjectMetadataDecorator\(` repeated many times in same module

**MEANS**

* Output declarations are too verbose; future targets will diverge.

**REPLACE WITH**

* **Any spec**: “template helper extraction” step (the `*-2` family)

**DO**

* Replace repeated decorator blocks with a tiny helper per output kind, e.g.:

  * `declare_table_output(table_key, target_name, domain, ...)`
  * `declare_artifact_output(artifact_name, path_template, ...)`
* (This is where your shared library extraction plan hooks in: shared `tool_target.py`, shared “output decl” helpers, etc.)

---

#### Rule 11 — Direct `save_manifest(...)` calls inside modules

**FIND**

* `save_manifest\(`

**MEANS**

* Manifest write is not centralized; likely inconsistent with run record semantics.

**REPLACE WITH**

* **Any spec**: `*-3` / `*-4` (move to record builder layer)

**DO**

* Delete direct calls.
* Only `record_from_*` (or the template’s record builder) should be allowed to commit manifest updates.

*(In your current Phase4 tree this mostly appears in helpers, but keep the rule to prevent regressions.)*

---

#### Rule 12 — Non-DAG “side effect” calls in the target module

**FIND**

* `log\.(info|warning|error)\(` (large volumes inside compute)
* `env\.gateway\.` used directly in compute nodes (outside pure reads)
* `open\(` / `write_text\(` / `json\.dump\(` in compute nodes

**MEANS**

* Side effects are not captured by Hamilton’s execution boundary and make caching/parallelization brittle.

**REPLACE WITH**

* **Any spec**: move into savers or into the tool boundary

**DO**

* IO → saver nodes only.
* Tool execution side effects → only in `t__<target>__run`.
* Compute nodes → pure transforms.

---

### C) Repo “hot matches” you can migrate mechanically right now

If you run just Rule 1–4 on native modules, the highest-payoff immediate conversions are concentrated here:

* `src/codeintel/build/hamilton/native/graphs/import_graph.py` (marker savers + manual warehouse writes + executor_materialize)
* `src/codeintel/build/hamilton/native/graphs/call_graph.py` (marker savers + manual warehouse writes + MaterializeOptions/MaterializationResult + executor_materialize)
* `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py` (marker savers + manual warehouse writes + executor_materialize)
* `src/codeintel/build/hamilton/native/graphs/graph_targets.py` (marker savers + manual warehouse writes + executor_materialize)

Those are essentially “mechanical TT/TO/TA/MX migrations” using the rules above.

---

### D) “Processing order” (so developers don’t have to think)

For each module:

1. **Classify spec** (Section A).
2. Apply **Rule 1 → Rule 12** in order.
3. Ensure `t__<target>` now:

   * depends on only **typed inputs** and `m__...` saver metadata nodes,
   * has **no IO** and **no tool execution**, and
   * returns via the appropriate **record_from_* / unified record builder**.
4. Delete:

   * marker nodes,
   * manual warehouse writes,
   * bespoke summarizers/skip helpers (if now redundant).

If you want, I can also generate a repo-specific “migration command sheet” (a single block of copy-paste `grep -R ...` commands in the exact order above, including the file paths that will match in this Phase4 snapshot), so someone can run it like a checklist without even scanning the code.
