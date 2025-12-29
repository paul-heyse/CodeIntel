Here’s the narrative pairing for **(10) “adding a new target = adding a module” + formal plugin boundaries**—what the system *becomes*, how the mental model changes, and what day-to-day extension looks like after the refactor.

---

## The core shift: targets stop being “registered objects” and become “imported code”

### Before

Even if you already rely heavily on Hamilton discovery, most systems still have some form of “target registry” vibe:

* targets are enumerated implicitly by scattered module lists,
* or there are multiple discovery paths (CLI vs serving vs export),
* or “known targets” are in one place while “actual nodes” come from another,
* and external extension is effectively “copy code into the repo,” not “install a pack.”

That creates friction and drift risk: the “set of targets” is not always *identical* across contexts unless you carefully keep module discovery consistent.

### After

A target is no longer a concept you register or configure directly.

> A target exists **iff** its module is imported into the runtime’s module set **and** it defines a properly tagged `t__*` anchor node.

So targets become an emergent property of **module import + tag semantics**.

This means:

* “add a new target” is literally “add a new Python module that defines nodes.”
* there is no other place to update.
* all target metadata (domain/spec version/outputs) remains DAG-derived.

---

## The new extension topology: core + workspace + plugins

After the change, the runtime module set is constructed deterministically from three sources:

1. **Core modules** (shipped with CodeIntel)
2. **Workspace modules** (repo-local package, e.g. `codeintel_targets/**`)
3. **Plugin packs** (pip-installed distributions discovered via entry points)

Crucially, the composition root builds one ordered list from these sources, imports them, fingerprints them, builds the driver once, and compiles the catalog once.

So no matter whether you are:

* running CLI builds,
* exporting serving artifacts,
* serving an API,
* or running plan/explain…

…you are always operating on the same module inventory.

---

## What “plugin boundaries” actually mean here

### Before

“Plugins” are often a fuzzy architecture concept: you can technically drop code somewhere, but there’s no stable ABI, no versioning story, no discovery contract, and no provenance.

### After

Plugins become a first-class packaging boundary:

* A plugin is a Python distribution that exposes **one entry point**: `codeintel.target_packs`
* That entry point resolves to a callable returning a small typed descriptor:

  * pack name
  * pack version
  * list of module import paths it contributes
  * compatibility constraints (requires_codeintel spec)
  * optional config namespace / capabilities

So plugins don’t “reach into” CodeIntel; they declare *modules* that contain Hamilton nodes. The DAG is the integration contract.

---

## Why the “SDK” matters (and what it changes for authors)

If plugin authors import internal modules, your refactors will break them. The plan introduces a minimal `codeintel.sdk` that is explicitly the only supported integration surface:

* stable tag taxonomy exports (keys and allowed values)
* stable anchor decorator wrapper (so target tags are always correct)
* stable saver wrappers (`save_to_table`, `save_to_artifact`) so outputs are declared correctly (saver-derived inventory)
* optionally stable table decorators (if you want plugin targets to inherit your canonical transforms)

This changes plugin authoring from “copy internal tagging patterns” to:

* “use SDK to define targets and outputs in a way that automatically fits catalog compilation, enforcement, and serving.”

---

## What day-to-day “adding a target” looks like after

### In the workspace (inside your repo)

You create a new file under a single canonical root, e.g.:

`src/codeintel_targets/<domain>/<target_name>.py`

In that file you define:

* one `t__*` anchor node tagged as `materialize` with `target=...`, `domain=...`
* saver nodes (`m__*`) that declare outputs (table_key / artifact name, output_role=contract)

That’s it.

On the next run:

* module resolver discovers the file
* imports the module
* driver includes the nodes
* catalog compilation sees the new anchor and outputs
* support nodes update automatically
* target appears in `codeintel targets list` and is runnable

No central registry edits. No config updates required unless you want enable/disable behaviors.

### As a pip-installed plugin pack

You do the same thing, but in a separate package:

* you ship the target module(s) in your distribution
* you expose a pack descriptor via entry point
* CodeIntel discovers it automatically at runtime composition

This makes extension feel like:

* install package
* enable pack (or default enabled)
* targets appear

---

## Provenance becomes first-class (and why it’s important)

A major hidden improvement is attribution:

Every loaded module is recorded with provenance:

* origin = core/workspace/plugin
* plugin name + version (if applicable)
* module file path and content hash

This solves a real operational problem: when validation fails due to duplicates (table_key collisions, duplicate anchor target names), error messages can say:

* “Duplicate `table_key=...` declared by plugin X module Y and workspace module Z.”

That’s the difference between “debugging the whole repo blindly” and “immediately knowing which pack caused conflict.”

It also powers:

* `codeintel plugins list`
* `codeintel targets list --show-origin`

So the system is both extensible and diagnosable.

---

## Determinism + cache correctness (why fingerprinting is mandatory)

When you allow external modules, you must ensure that:

* a plugin update invalidates cache,
* workspace edits invalidate cache,
* “same config + same code” yields same runtime fingerprint.

The plan ensures this by computing a `modules_fingerprint` that includes:

* module import strings (ordered)
* distribution versions for plugin packs
* module file content hashes (covers editable installs)
* config digest

So “what you built” is reproducibly identifiable, and caching/manifest semantics remain trustworthy across plugin changes.

---

## What gets deleted conceptually

You’re removing an entire class of bespoke “target management” code:

* no static target registry
* no YAML manifest of targets
* no per-context module lists that diverge
* no “serving has a different driver than build”
* no “export rebuilds a driver to discover views”

Everything goes through: **resolve modules → build driver → compile catalog → use runtime bundle**.

---

## The end state in one sentence

After (10), CodeIntel’s extension story becomes:

> Targets are modules, modules come from core/workspace/plugins, plugins are just packaged modules discovered via entry points, and the entire system composes a single deterministic runtime from that inventory.

That’s the cleanest plugin boundary you can get in a Hamilton-first architecture, and it sets you up to grow target packs without growing the core codebase footprint.


Below is a **repo‑concrete, breaking‑change–optimized**, **high lexical density** implementation plan for:

> **(10) Treat “adding a new target” as adding a module—then formalize plugin boundaries**

You explicitly want the “best‑in‑class” design with no backward‑compat constraints; therefore this plan:

* makes **targets purely module‑defined** (a target exists iff a `t__*` anchor node exists),
* makes module discovery and plugin discovery **deterministic, fingerprinted, and compositional**,
* formalizes an **external plugin ABI** so third‑party target packs can be installed via `pip` and discovered via Python entry points,
* and deletes any remaining “static registries” / ad hoc discovery hooks.

---

# 0) End‑state contract

## 0.1 Target definition is module definition

* A “target” exists iff a Hamilton node with canonical tags (`node_type=materialize`, `target=...`, `domain=...`) is present in the imported module set.
* Adding a target requires **only**:

  1. adding a Python module containing a `t__*` anchor (and saver nodes declaring contract outputs),
  2. ensuring the module is discoverable by the module resolver (local pack or plugin pack).

No central registry of targets, no YAML inventories, no manual “target list updates”.

## 0.2 Plugin boundary is explicit and stable

* External packages contribute modules via a **single entry point group** (e.g. `codeintel.target_packs`).
* Each entry point resolves to a callable returning a typed `TargetPack` descriptor (name/version/modules/config constraints).
* Composition root loads packs deterministically, validates compatibility, imports modules, builds driver, compiles catalog; downstream is agnostic.

## 0.3 Deterministic assembly and cache‑correctness

* Runtime fingerprint incorporates:

  * core version
  * resolved module import strings
  * **plugin distribution versions**
  * content hash of each module file (covers editable installs)
  * normalized runtime config digest
* Any plugin code change forces cache invalidation via the same fingerprint path used in cache keys.

## 0.4 Provenance is first‑class

* Every loaded module is assigned provenance:

  * `origin = "core" | "workspace" | "plugin"`
  * `plugin_name` (optional)
  * distribution metadata (name/version)
  * filesystem path
* Provenance is used to:

  * make validation errors actionable (“duplicate table_key declared by plugin X module Y”),
  * power `codeintel plugins list` and `codeintel targets list --show-origin`.

---

# 1) Define the external plugin ABI (TargetPack descriptor + loader)

### Files CREATED

## 1.1 `src/codeintel/runtime/plugins/spec.py`

Typed plugin contract used by both core and third‑party packs. Keep it minimal and declarative.

* `TargetPackModule` (frozen/slots)

  * `import_path: str`  (Python module import string; no file paths)
  * `kind: Literal["hamilton"] = "hamilton"` (future‑proof)
* `TargetPack` (frozen/slots)

  * `name: str` (globally unique; namespace recommended: `org.project.pack`)
  * `version: str` (pack version; usually distribution version)
  * `modules: tuple[TargetPackModule, ...]` (the only required contribution)
  * `requires_codeintel: str` (PEP 440 specifier; enforced)
  * `default_enabled: bool = True`
  * `config_namespace: str | None` (optional; forces namespacing)
  * `capabilities: frozenset[str]` (optional; e.g. `{"targets","schemas","semantic_views"}`)

## 1.2 `src/codeintel/runtime/plugins/loader.py`

Entry point discovery and pack materialization using `importlib.metadata`.

Core routines:

* `discover_target_packs(group: str="codeintel.target_packs") -> tuple[TargetPack, ...]`
* `load_pack(entry_point) -> TargetPack`
* `validate_pack(pack: TargetPack, *, codeintel_version: str) -> None`
* deterministic ordering: sort by `(pack.name, pack.version)` then module import path.

**Critical behavior**: treat pack loading as a pure “descriptor acquisition” step; actual module imports occur in the module resolver so import side‑effects are controlled centrally.

### Code snippet (entry point loading pattern)

```python
# loader.py (illustrative)
from importlib.metadata import entry_points, version, PackageNotFoundError
from packaging.specifiers import SpecifierSet

from .spec import TargetPack

EP_GROUP = "codeintel.target_packs"

def discover_target_packs(*, codeintel_version: str) -> tuple[TargetPack, ...]:
    eps = entry_points().select(group=EP_GROUP)
    packs = []
    for ep in eps:
        factory = ep.load()  # callable returning TargetPack
        pack = factory()
        _validate_pack(pack, codeintel_version=codeintel_version)
        packs.append(pack)
    packs.sort(key=lambda p: (p.name, p.version))
    return tuple(packs)

def _validate_pack(pack: TargetPack, *, codeintel_version: str) -> None:
    if not SpecifierSet(pack.requires_codeintel).contains(codeintel_version):
        raise RuntimeError(f"Pack {pack.name} incompatible with CodeIntel {codeintel_version}")
    if not pack.modules:
        raise RuntimeError(f"Pack {pack.name} has no modules")
```

---

# 2) Formalize a stable SDK import surface for plugin authors

Without a stable ABI, plugins will import internal modules and break constantly. Create a minimal `codeintel.sdk` that re‑exports only “safe” building blocks (tags, saver decorators, target anchor decorator, table contract decorators).

### Files CREATED

## 2.1 `src/codeintel/sdk/__init__.py`

Re-export canonical plugin API:

* `from .tags import *`
* `from .target import target_anchor`
* `from .save_to import save_to_table, save_to_artifact`
* `from .annotations import schema_output, tag, tag_output`
* `from .validation import check_output_warn, check_output_fail`
* `from .io import dataloader, datasaver, load_from, save_to`
* `from .columns import with_columns_lazy, pipe_input, pipe_output`
* optionally: `table_contract` / `pipe_clean_df` / `with_features` if you want plugin authors to use your canonical transforms

## 2.2 `src/codeintel/sdk/tags.py`

Re-export *only* tag taxonomy constants (string keys + allowed values). This is your public “semantic ABI”.

## 2.3 `src/codeintel/sdk/target.py`

Provide a stable anchor decorator that applies canonical tags and naming conventions. This prevents every pack from copying fragile tag glue.

Example:

```python
# sdk/target.py (illustrative)
from codeintel.core.hamilton import tags as ht
from codeintel.build.hamilton.tagging import tag_materialize  # internal; keep stable behind sdk

def target_anchor(*, target: str, domain: str, spec_version: str):
    def deco(fn):
        return tag_materialize(target=target, domain=domain, spec_version=spec_version)(fn)
    return deco
```

## 2.4 `src/codeintel/sdk/save_to.py`

Re-export stable saver wrappers that:

* apply `hamilton.data_saver=True`,
* apply `output_role`, `table_key`/`artifact`, sink metadata,
* and preserve your Phase (3) inventory semantics.

## 2.5 Advanced modifier wrappers (schema hints, validation, I/O, columns)

To keep plugins on a stable ABI while still using Hamilton's advanced surface,
add SDK wrappers for the function modifier families you expect authors to use.
These should remain "hint-only" in line with the inference-first plan.

### Files CREATED

* `src/codeintel/sdk/annotations.py`

  * `schema_output(...)` wrapper for `@schema.output` (stored as soft hints only)
  * `tag(...)` and `tag_output(...)` wrappers (enforce canonical key taxonomy)

* `src/codeintel/sdk/validation.py`

  * `check_output_warn(...)` wrapper for `@check_output(importance="warn")`
  * `check_output_fail(...)` wrapper for explicit opt-in failure

* `src/codeintel/sdk/io.py`

  * `dataloader(...)`, `datasaver(...)`, `load_from.<format>(...)`,
    `save_to.<format>(...)` re-exports with stable defaults

* `src/codeintel/sdk/columns.py`

  * `with_columns_lazy(...)` wrapper for `h_polars_lazyframe.with_columns`
  * optional `pipe_input(...)` / `pipe_output(...)` shims for graph-local pipelines

### Files MODIFIED

* `src/codeintel/sdk/__init__.py`

  * export the above wrappers so plugin authors never import Hamilton internals directly

---

# 3) Extend the module resolver to merge core + workspace + plugins deterministically

This is the mechanical heart of the phase: make “module set resolution” the only place that imports target modules.

### Files MODIFIED

## 3.1 `src/codeintel/runtime/module_resolver.py`

Replace any ad hoc discovery with a single deterministic pipeline:

### New data structures

* `ModuleProvenance` (frozen/slots)

  * `origin: Literal["core","workspace","plugin"]`
  * `module_import: str`
  * `file_path: str | None`
  * `plugin_name: str | None`
  * `dist_name: str | None`
  * `dist_version: str | None`
* `ResolvedModuleSet` (frozen/slots)

  * `modules: tuple[ModuleType, ...]`
  * `provenance: dict[str, ModuleProvenance]` keyed by `module.__name__`
  * `fingerprint: str` (blake3)
  * `packs: tuple[TargetPack, ...]` (loaded pack descriptors)

### Resolution pipeline

1. Resolve **core modules** (always):

   * `codeintel.build.hamilton.native.*` (plus planning nodes, support nodes module, etc.)
2. Resolve **workspace modules** (repo-local target modules):

   * choose a single canonical root, e.g. `src/codeintel_targets/**` or `src/codeintel/build/hamilton/targets/**`
   * enforce “module = file” mapping; avoid implicit imports
3. Resolve **plugin pack modules**:

   * call `discover_target_packs()` (from loader)
   * apply enable/disable filters from config
   * collect `TargetPack.modules[*].import_path`
4. Import all modules in stable order
5. Compute fingerprint:

   * core version + config digest
   * sorted module import strings
   * distribution version for plugin packs
   * file content hash per module (if `__file__` exists)

**File hash** is mandatory to cover editable installs; use `blake3(open(__file__, "rb").read())`.

### Code snippet (fingerprinting)

```python
def _module_content_hash(mod) -> str:
    p = getattr(mod, "__file__", None)
    if not p:
        return "nofile"
    data = Path(p).read_bytes()
    return blake3(data).hexdigest()

def fingerprint(modules: Sequence[ModuleType], packs: Sequence[TargetPack], cfg_digest: str) -> str:
    h = blake3()
    h.update(cfg_digest.encode())
    for p in packs:
        h.update(f"pack:{p.name}:{p.version}".encode())
    for m in sorted(modules, key=lambda x: x.__name__):
        h.update(m.__name__.encode())
        h.update(_module_content_hash(m).encode())
    return h.hexdigest()
```

## 3.2 Config-driven DAG shaping (when/resolve)

Make configuration-driven DAG shaping a first-class behavior (no bespoke
"target toggles"). Hamilton's `@config.when` and `@resolve` let packs emit
nodes conditionally without changing module inventories.

### Files MODIFIED

* `src/codeintel/runtime/compose.py`

  * pass `Builder.with_config(runtime_cfg.hamilton_config)` so config is applied
    before DAG construction
  * enforce `config_namespace` for plugin packs when set (prevents collisions)

* `src/codeintel/runtime/plugins/config.py`

  * add `hamilton_config: Mapping[str, object]` bucket for config-driven DAG rules
  * apply pack-scoped overrides under `config_namespace` (if defined)

### Definition

* “Module set = what can exist.”
* “Config = what does exist for this run.”

This keeps the graph authoritative while allowing environment-specific
structure changes.

---

# 4) Composition root consumes `ResolvedModuleSet` and publishes provenance

### Files MODIFIED

## 4.1 `src/codeintel/runtime/compose.py`

Change assembly steps:

* resolve module set once:

  * `resolved = resolve_modules(cfg, env, codeintel_version=...)`
* build driver using `resolved.modules`
* store module provenance + packs on runtime bundle:

  * `runtime.modules = resolved.modules` (optional)
  * `runtime.module_provenance = resolved.provenance`
  * `runtime.packs = resolved.packs`
  * incorporate `resolved.fingerprint` into `RuntimeBundle.fingerprint`

### Files MODIFIED

## 4.2 `src/codeintel/runtime/runtime_bundle.py`

Add:

* `packs: tuple[TargetPack, ...]`
* `module_provenance: Mapping[str, ModuleProvenance]`
* `modules_fingerprint: str`

## 4.3 Standard execution profiles (cache + structured logs)

Make caching and audit logs first-class execution profiles aligned with
Hamilton's cache semantics.

### Files MODIFIED

* `src/codeintel/runtime/compose.py`

  * set `code_version` to `modules_fingerprint`
  * expose a cache profile (opt-in caching + JSONL logs) via
    `Builder.with_cache(log_to_file=True, default_behavior="disable")`

* `src/codeintel/observability/*` or a new cache-log ingester module

  * ingest `dr.cache.logs(...)` JSONL into DuckDB tables for run audits
  * add a `cache_lineage` table for deterministic data version tracing

## 4.4 Telemetry + semantic registry compiler (UI-aligned)

Integrate the HamiltonTracker and the tag taxonomy so the UI becomes a first-class
source of truth alongside the runtime graph.

### Files MODIFIED

* `src/codeintel/runtime/compose.py`

  * optionally attach `HamiltonTracker` with a stable `dag_name`
  * include `repo`, `commit`, `run_kind`, `modules_fingerprint` as tracker tags

* `src/codeintel/serving/semantic/registry_compiler.py` (or equivalent)

  * compile `semantic_registry.json` from DAG tags (`layer="semantic"`)
  * enforce required tags (`semantic_id`, `kind`, `grain`, `schema_ref`, etc.)

---

# 5) Configuration surface for plugins (enable/disable, strictness, namespace isolation)

### Files CREATED

## 5.1 `src/codeintel/runtime/plugins/config.py`

Define:

* `PluginConfig` (frozen/slots)

  * `enabled: tuple[str, ...] | None` (allowlist; if set, only these)
  * `disabled: tuple[str, ...]` (denylist)
  * `strict: bool = True` (fail-fast on load incompatibility/import errors)
  * `namespace_enforcement: bool = True` (if pack.config_namespace required)
  * `allow_workspace_modules: bool = True` (toggle repo-local scanning)
  * `hamilton_config: Mapping[str, object] = {}` (config for @config.when/@resolve)

### Files MODIFIED

* `src/codeintel/build/config.py` or your runtime config module

  * embed `PluginConfig` under `RuntimeConfig.plugins`
  * delete any legacy “target registry” or “manual module list” config keys immediately

---

# 6) Diagnostics + CLI: make plugins and target origins visible

### Files CREATED

## 6.1 `src/codeintel/cli/handlers/plugins.py`

Commands:

* `codeintel plugins list` → shows pack name/version/modules/enabled
* `codeintel plugins info <name>` → shows module provenance, file hashes, compatibility spec

### Files MODIFIED

* `src/codeintel/cli/handlers/build.py`

  * add flags:

    * `--enable-plugin NAME` (repeatable)
    * `--disable-plugin NAME`
    * `--no-workspace-targets`
  * provide `--list-targets --show-origin`:

    * prints `target`, `domain`, `anchor_module`, `origin/plugin_name`

### Files MODIFIED

* `src/codeintel/cli/__init__.py` (or router)

  * wire new handler.

---

# 6.5 Optional dynamic execution profile (Parallelizable/Collect)

Add a configurable execution mode for high-fan-out targets that benefit from
Hamilton's dynamic task model.

### Files MODIFIED

* `src/codeintel/runtime/compose.py`

  * allow `enable_dynamic_execution(allow_experimental_mode=True)`
  * wire `with_remote_executor(...)` and `with_local_executor(...)`
  * gate behind config to avoid accidental activation

### Decision gate

* Enable only for DAGs that emit large parallelizable workloads (e.g., per-file
  ingestion or per-module analysis).

---

# 7) Validation hardening with provenance-aware error messages

You already validate duplicate table keys / artifacts / target anchors in catalog compilation and node validation. This phase adds **provenance correlation** so the error says “who declared it”.

### Files MODIFIED

* `src/codeintel/build/hamilton/validate.py`

  * accept optional `module_provenance` mapping
  * when reporting:

    * duplicate `table_key`
    * duplicate `artifact`
    * duplicate `target` anchors
  * validate semantic tag taxonomy for public outputs:

    * required: `layer`, `semantic_id`, `kind`, `entity`, `grain`, `version`
    * if `kind="table"`: `schema_ref`, `entity_keys`, `join_keys`
    * include provenance in tag validation errors
  * include module origin:

    * `module = node.originating_module` (if available) else best effort (search callable `__module__`)
    * map to provenance for plugin attribution

If Hamilton node objects expose `node.callable.__module__`, use that consistently. If not, fall back to searching loaded modules for attribute names (less ideal, but workable).

---

# 8) Workspace module boundary: codify a single canonical directory + naming contract

To make “adding a new target = adding a module” ergonomic inside the repo, enforce:

* single directory root for user-defined targets, e.g. `src/codeintel_targets/`
* module naming conventions:

  * `codeintel_targets.<domain>.<target_name>.py`
* optionally a `__init__.py` that exports nothing (avoid implicit side-effects)

### Files CREATED

* `src/codeintel_targets/__init__.py` (empty marker package)
* `src/codeintel_targets/README.md` (contract + examples)
* `src/codeintel_targets/analytics/hello_target.py` (minimal sample)

### Files MODIFIED

* `src/codeintel/runtime/module_resolver.py`

  * implement workspace scanning as:

    * discover python modules under `codeintel_targets` package
    * import them by import string, not by file execution hacks
  * compute provenance `origin="workspace"`

> This yields “drop a new file under codeintel_targets → it is discovered automatically → new target appears”.

---

# 9) Provide an example external target pack (packaging + entry points)

### Files CREATED

Under `examples/target_packs/hello_pack/`:

* `examples/target_packs/hello_pack/pyproject.toml`

  * defines entry point:

    ```toml
    [project.entry-points."codeintel.target_packs"]
    hello_pack = "hello_pack.plugin:codeintel_target_pack"
    ```
* `examples/target_packs/hello_pack/src/hello_pack/plugin.py`

  * returns `TargetPack` referencing module import paths
* `examples/target_packs/hello_pack/src/hello_pack/targets/hello.py`

  * defines `t__hello` anchor + saver nodes using `codeintel.sdk`

This is crucial: it forces you to keep the ABI honest.

---

# 10) Delete legacy “static target registry” and any non-module-based target bootstrapping

Since you requested immediate deletion, remove any mechanism that:

* enumerates targets in code
* maps target name → module list explicitly
* injects “registered targets” from config

### Files DELETED (category; exact filenames depend on your current repo)

* Any module whose primary responsibility is “target registry” or “target discovery list”
* Any JSON/YAML “targets manifest” used to declare targets
* Any “TargetGraph builder from registry” (already removed in phase 1)

**Mechanically:** after implementing module resolver + workspace + plugin packs, search for:

* `REGISTERED_TARGETS`, `TARGET_REGISTRY`, `discover_targets(`, `load_targets_manifest(`

and delete those modules and call sites.

---

# 11) Tests: enforce determinism, provenance, and plugin ABI behavior

### Files CREATED

1. `tests/plugins/test_entrypoint_pack_loading.py`

* patch `importlib.metadata.entry_points()` to return a synthetic entry point returning a `TargetPack`
* assert resolver imports its modules and records provenance
* assert deterministic ordering independent of discovery order

2. `tests/plugins/test_pack_fingerprint_invalidation.py`

* create a temp module file and import it as workspace module
* mutate file contents, re-compose runtime
* assert `modules_fingerprint` changes and thus runtime fingerprint changes (cache correctness)

3. `tests/plugins/test_duplicate_output_reports_origin.py`

* two packs both declare same `table_key`
* assert validator error includes `plugin_name` and `module_import`

4. `tests/workspace/test_add_module_adds_target.py`

* create a new file under `codeintel_targets/` in a temp package context (or simulate with importlib)
* re-resolve modules
* assert new target anchor appears in catalog

5. `tests/validation/test_semantic_tag_taxonomy.py`

* assert required semantic tags are enforced for public outputs
* assert provenance is included in tag validation errors

6. `tests/semantic_registry/test_registry_compiler.py`

* compile registry from DAG tags and validate expected output schema

7. `tests/observability/test_cache_log_ingest.py`

* emit cache JSONL logs and assert they ingest into DuckDB tables

8. `tests/runtime/test_dynamic_execution_profile.py`

* verify dynamic execution is gated by config and composes remote/local executors

---

# 11.5 Executable Implementation Checklist (new additions)

Use this checklist to implement the added Hamilton-advanced alignment items in
order. Each step lists exact file edits.

1) SDK wrappers for advanced modifiers
   - Create `src/codeintel/sdk/annotations.py` with `schema_output`, `tag`,
     `tag_output` wrappers (hint-only; no runtime enforcement).
   - Create `src/codeintel/sdk/validation.py` with `check_output_warn` and
     `check_output_fail` wrappers.
   - Create `src/codeintel/sdk/io.py` with `dataloader`, `datasaver`,
     `load_from`, `save_to` re-exports and stable defaults.
   - Create `src/codeintel/sdk/columns.py` with `with_columns_lazy` plus optional
     `pipe_input`/`pipe_output` shims.
   - Update `src/codeintel/sdk/__init__.py` to export these wrappers.

2) Config-driven DAG shaping
   - Update `src/codeintel/runtime/plugins/config.py` to add
     `hamilton_config: Mapping[str, object] = {}`.
   - Update `src/codeintel/build/config.py` (or runtime config module) to
     surface `hamilton_config` under plugins/runtime config.
   - Update `src/codeintel/runtime/compose.py` to call
     `Builder.with_config(runtime_cfg.plugins.hamilton_config)` and enforce
     `config_namespace` isolation.

3) Cache profile + JSONL audit logs
   - Update `src/codeintel/runtime/compose.py` to set
     `code_version = modules_fingerprint` and enable
     `Builder.with_cache(log_to_file=True, default_behavior="disable")` under a
     config-controlled profile.
   - Create `src/codeintel/observability/cache_log_ingest.py` to ingest cache
     JSONL into DuckDB (run_id, node, event, data_version).
   - Wire ingestion entry points (CLI or observability runner) as needed.

4) Telemetry + registry compiler
   - Update `src/codeintel/runtime/compose.py` to attach `HamiltonTracker`
     (self-hosted) with stable `dag_name` and tags
     (`repo`, `commit`, `run_kind`, `modules_fingerprint`).
   - Create `src/codeintel/serving/semantic/registry_compiler.py` to compile
     `semantic_registry.json` from DAG tags and enforce required tag schema.
   - Add a CLI entry to run the compiler (e.g., `codeintel semantic compile`).

5) Semantic tag taxonomy enforcement
   - Update `src/codeintel/build/hamilton/validate.py` to enforce required tag
     keys for semantic outputs and include module provenance in error messages.

6) Dynamic execution profile (optional)
   - Update `src/codeintel/runtime/compose.py` to gate
     `enable_dynamic_execution(allow_experimental_mode=True)` behind config.
   - Wire `with_remote_executor(...)` and `with_local_executor(...)` from config.

7) Tests for new additions
   - Add `tests/validation/test_semantic_tag_taxonomy.py`.
   - Add `tests/semantic_registry/test_registry_compiler.py`.
   - Add `tests/observability/test_cache_log_ingest.py`.
   - Add `tests/runtime/test_dynamic_execution_profile.py`.

8) Doc + rollout alignment
   - Keep `docs/hamilton_inference_first_alignment_plan.md` cross‑linked to
     the above steps (see “Plan Alignment” section).

---

# 12) File index summary (additions / modifications / deletions)

## Created

* `src/codeintel/runtime/plugins/spec.py`
* `src/codeintel/runtime/plugins/loader.py`
* `src/codeintel/runtime/plugins/config.py`
* `src/codeintel/runtime/tag_query.py` already exists from (8); reuse
* `src/codeintel/sdk/__init__.py`
* `src/codeintel/sdk/tags.py`
* `src/codeintel/sdk/target.py`
* `src/codeintel/sdk/save_to.py`
* `src/codeintel/sdk/annotations.py`
* `src/codeintel/sdk/validation.py`
* `src/codeintel/sdk/io.py`
* `src/codeintel/sdk/columns.py`
* `src/codeintel_targets/__init__.py`
* `src/codeintel_targets/README.md`
* `src/codeintel_targets/analytics/hello_target.py`
* `src/codeintel/cli/handlers/plugins.py`
* `src/codeintel/serving/semantic/registry_compiler.py`
* `src/codeintel/observability/cache_log_ingest.py`
* `examples/target_packs/hello_pack/**` (pyproject + plugin + targets)
* tests under `tests/plugins/**`, `tests/workspace/**`, `tests/validation/**`,
  `tests/semantic_registry/**`, `tests/observability/**`, `tests/runtime/**`

## Modified

* `src/codeintel/runtime/module_resolver.py` (merge core/workspace/plugin modules + fingerprint + provenance)
* `src/codeintel/runtime/compose.py` (consume ResolvedModuleSet; publish packs/provenance; cache/telemetry profiles)
* `src/codeintel/runtime/runtime_bundle.py` (store packs/provenance/modules_fingerprint)
* `src/codeintel/build/config.py` or runtime config module (add PluginConfig + hamilton config; delete legacy target list config)
* `src/codeintel/build/hamilton/validate.py` (provenance-aware diagnostics + semantic tag taxonomy)
* `src/codeintel/cli/handlers/build.py` (flags + show-origin)
* CLI router wiring

## Deleted

* Any target registry / manifest / explicit mapping infrastructure (search-driven deletion as described in §10)

---

# 13) Definition of Done (hard gates)

1. **Workspace:** dropping a new module under `codeintel_targets/` containing `t__*` makes the target appear with zero additional wiring.
2. **Plugins:** installing a pip package with an entry point in group `codeintel.target_packs` causes its modules to be imported and targets to appear.
3. **Determinism:** module resolution order and runtime fingerprint are stable for identical inputs; two runs yield identical `modules_fingerprint`.
4. **Cache correctness:** changing any workspace/plugin module file contents changes `modules_fingerprint` and thus invalidates cache keys/fingerprints.
5. **Provenance:** `codeintel plugins list` and `codeintel targets list --show-origin` clearly attribute targets/outputs to core/workspace/plugin modules.
6. **Tag taxonomy:** semantic outputs fail validation if required tags are missing or malformed.
7. **Telemetry:** registry compiler output is derived from DAG tags and matches UI contract.
8. **Caching:** cache logs are emitted in JSONL and ingestable into DuckDB with stable run IDs.
9. **Dynamic execution (optional):** when enabled, dynamic execution is gated by config and uses explicit remote/local executors.
10. **No static registries:** there is no code path where targets are enumerated outside Hamilton module import + tag semantics.

---

## Architectural payoff

This phase completes the “DAG‑derived extensibility” story: targets are no longer a bespoke concept in the orchestration layer. They are **modules**, and modules can come from:

* core distribution,
* workspace repository,
* installed plugin distributions.

That is the cleanest plugin boundary possible for a Hamilton‑driven system: the DAG is the integration contract, and a “plugin” is simply a pack that contributes DAG nodes under a stable ABI.
