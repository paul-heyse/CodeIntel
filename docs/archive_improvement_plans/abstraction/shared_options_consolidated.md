Here’s a “metadata‑level” treatment of the **shared options system** — same style as the revised metadata plan, but now for options/profiles, with more emphasis on cross‑cutting scope and how it all fits together.

This summarizes and restructures everything in your shared‑options docs (ConfigSource, PluginOptionsResolver, ProfiledConfigSource, BuildRunConfig, fast/full profiles, include_graph_metrics, scope_paths, etc.) into one coherent design, plus a concrete “next big improvement” you can implement on top.   

---

## 1. Design goals for the shared options system

The options system should:

1. **Centralize “how plugins get options”**

   * No more per‑plugin one‑off config logic.
   * Same mechanism for analytics, graphs, ingestion, export, etc. 

2. **Be metadata‑driven**

   * `CorePluginMetadata.options_model` says *what* options a plugin expects. 
   * The resolver looks that up and builds a typed object.

3. **Support policy / profiles**

   * `base` config (defaults)
   * `profile` overrides (`fast`, `full`, `ci`, …)
   * `CLI` / run‑specific overrides
     all merged in a predictable way. 

4. **Play nicely with manifests & caching**

   * Options → `options_hash` → part of `input_hash` for `PluginExecutionRecord`. 

5. **Be incrementally adoptable**

   * Start with `analytics.function_metrics` + `graphs.callgraph`.
   * Other plugins just add `options_model` + a few lines of boilerplate.

---

## 2. Core abstractions (cross‑cutting, domain‑agnostic)

### 2.1 ConfigSource – “where do plugin options come from?”

A tiny protocol in `core/plugins/execution/options.py`: 

```python
# core/plugins/execution/options.py

from typing import Any, Mapping, Protocol, runtime_checkable

@runtime_checkable
class ConfigSource(Protocol):
    """Where plugin options are loaded from (files/env/CLI/etc)."""

    def get_plugin_options(self, plugin_name: str) -> Mapping[str, Any] | None:
        """Return raw option values for a plugin, or None if not configured."""
        ...


class EmptyConfigSource:
    """ConfigSource that always returns no options.

    Used as a safe default so plugins still see valid model defaults.
    """

    def get_plugin_options(self, plugin_name: str) -> Mapping[str, Any] | None:
        return None
```

Later you plug in real config (YAML, env, etc.) and profiles on top — but plugins **only ever see** a `ConfigSource`.

---

### 2.2 PluginOptionsResolver – “given plugin + model, give me options”

Also in `core/plugins/execution/options.py`: 

```python
from dataclasses import replace
from typing import Any, Mapping, Type, TypeVar
from codeintel.core.plugins.types.metadata import CorePluginMetadata

T = TypeVar("T")

class PluginOptionsResolver:
    """Central helper to construct typed options for any plugin."""

    def __init__(self, config_source: ConfigSource | None = None) -> None:
        self._config_source = config_source or EmptyConfigSource()

    def get_options(
        self,
        plugin_metadata: CorePluginMetadata,
        model: Type[T],
        *,
        dynamic_overrides: Mapping[str, Any] | None = None,
    ) -> T:
        raw = self._config_source.get_plugin_options(plugin_metadata.name) or {}

        # Base options from config
        base = model(**raw)  # type: ignore[arg-type]

        if not dynamic_overrides:
            return base

        # Merge runtime-only overrides (AST maps, etc.)
        if hasattr(base, "__dataclass_fields__"):             # dataclass
            return replace(base, **dynamic_overrides)         # type: ignore[return-value]
        if hasattr(base, "model_copy"):                       # Pydantic v2
            return base.model_copy(update=dict(dynamic_overrides))  # type: ignore[return-value]
        if hasattr(base, "copy"):                             # Pydantic v1
            return base.copy(update=dict(dynamic_overrides))  # type: ignore[return-value]

        for k, v in dynamic_overrides.items():                # fallback
            setattr(base, k, v)
        return base
```

**Cross‑cutting behavior:**

* Works for any plugin with a `CorePluginMetadata` + `options_model`, regardless of domain.
* Cleanly separates:

  * **static** config (from files/profiles/CLI),
  * **dynamic** per‑run fields (AST caches, in‑memory sets, etc.).

---

### 2.3 PluginConfigBundle & ProfiledConfigSource – layering base / profile / CLI

Still in `options.py`: 

```python
from dataclasses import dataclass
from typing import Dict

def _merge_dicts(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if base:
        result.update(base)
    if override:
        result.update(override)
    return result


@dataclass(frozen=True)
class PluginConfigBundle:
    """Config data for all plugins for one layer (base, profile, cli)."""

    plugin_options: Mapping[str, Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "plugin_options", dict(self.plugin_options or {}))

    def get(self, plugin_name: str) -> Mapping[str, Any] | None:
        return self.plugin_options.get(plugin_name)
```

And the layered `ProfiledConfigSource`: 

```python
class ProfiledConfigSource(ConfigSource):
    """Merge base, profile, CLI plugin options into a single view."""

    def __init__(
        self,
        *,
        base: PluginConfigBundle | None = None,
        profile: PluginConfigBundle | None = None,
        cli: PluginConfigBundle | None = None,
        active_profile_name: str | None = None,
    ) -> None:
        self._base = base or PluginConfigBundle(plugin_options={})
        self._profile = profile or PluginConfigBundle(plugin_options={})
        self._cli = cli or PluginConfigBundle(plugin_options={})
        self._active_profile_name = active_profile_name

    def get_plugin_options(self, plugin_name: str) -> Mapping[str, Any] | None:
        base_raw = self._base.get(plugin_name)
        profile_raw = self._profile.get(plugin_name) if self._active_profile_name else None
        cli_raw = self._cli.get(plugin_name)

        merged = _merge_dicts(base_raw, profile_raw)
        merged = _merge_dicts(merged, cli_raw)

        return merged or None
```

**Cross‑cutting:** every plugin, regardless of domain, gets:

> `effective_options = merge(base.plugins[P], profile.plugins[P], cli.plugins[P])`

…via the same abstraction.

---

### 2.4 BuildRunConfig – how a single run describes its policy

In `build/options.py`: 

```python
from dataclasses import dataclass, field
from typing import Any, Mapping
from codeintel.core.plugins.execution.options import PluginConfigBundle

@dataclass
class BuildRunConfig:
    """Configuration for one build/run (used by build/CLI)."""

    profile: str | None = None   # "fast", "full", "ci", etc.

    base_plugin_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    profiles_plugin_options: Mapping[str, Mapping[str, Mapping[str, Any]]] = field(
        default_factory=dict
    )
    cli_plugin_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def to_profiled_config_bundles(self):
        base_bundle = PluginConfigBundle(plugin_options=self.base_plugin_options)

        profile_options: Mapping[str, Mapping[str, Any]] = {}
        if self.profile:
            profile_options = self.profiles_plugin_options.get(self.profile, {})
        profile_bundle = PluginConfigBundle(plugin_options=profile_options)

        cli_bundle = PluginConfigBundle(plugin_options=self.cli_plugin_options)
        return base_bundle, profile_bundle, cli_bundle
```

**Cross‑cutting:** this is the **run‑level policy container**. Graphs, analytics, ingestion all read the same layered view through `ProfiledConfigSource`.

---

## 3. End‑to‑end flow: from CLI to plugin options (cross‑domain)

Putting those pieces together, for *any* plugin:

1. **CLI / caller chooses a profile & overrides**

   In your CLI layer (e.g. `cli/commands/build.py`): 

   ```python
   run_config = BuildRunConfig(profile=args.profile)

   # CLI plugin overrides: --plugin-option analytics.function_metrics:include_coverage_metrics=false
   cli_plugin_options: dict[str, dict[str, Any]] = {}
   for flag in args.plugin_option or []:
       plugin_name, key, value = _parse_plugin_option_flag(flag)
       cli_plugin_options.setdefault(plugin_name, {})[key] = value

   run_config.cli_plugin_options = cli_plugin_options

   # later: set run_config.base_plugin_options / profiles_plugin_options from config files
   run_build(run_config=run_config, args=args)
   ```

2. **Build constructs a ProfiledConfigSource for the run**

   In `build/executor.py`: 

   ```python
   from codeintel.core.plugins.execution.options import ProfiledConfigSource

   base_bundle, profile_bundle, cli_bundle = run_config.to_profiled_config_bundles()

   config_source = ProfiledConfigSource(
       base=base_bundle,
       profile=profile_bundle,
       cli=cli_bundle,
       active_profile_name=run_config.profile,
   )

   for step in plan.steps:
       ctx = TargetExecutionContext(
           snapshot=step.snapshot,
           gateway=gateway,
           logger=logger,
           resources=resources,
           config_source=config_source,  # same for all plugins in this run
       )
       await step.plugin.execute(ctx)
   ```

3. **Execution context exposes config_source to every plugin**

   `TargetExecutionContext` just adds one field: 

   ```python
   # build/context.py

   from dataclasses import dataclass, field
   from codeintel.core.plugins.execution.options import ConfigSource, EmptyConfigSource

   @dataclass
   class TargetExecutionContext:
       snapshot: Snapshot
       gateway: StorageGateway
       logger: Logger
       resources: Resources

       config_source: ConfigSource = field(
           default_factory=EmptyConfigSource,
           repr=False,
       )
   ```

4. **Plugin resolves its options using metadata + resolver**

   For *any* plugin with `metadata.options_model`, the pattern is:

   ```python
   resolver = PluginOptionsResolver(config_source=ctx.config_source)
   options = resolver.get_options(self.metadata, self.metadata.options_model, dynamic_overrides={...})
   ```

   Then you pass `options` into the plugin’s core compute function.

---

## 4. Concrete cross‑cutting examples (analytics + graphs today, others later)

### 4.1 Analytics: `analytics.function_metrics`

**Options model (config + dynamic):** 

```python
# analytics/functions/config.py

from dataclasses import dataclass
from typing import Any, Mapping

@dataclass
class FunctionAnalyticsOptions:
    # Dynamic, runtime-only:
    function_ast_map: Mapping[int, Any] | None = None
    missing_function_goids: set[int] | None = None

    # Config-driven:
    include_graph_metrics: bool = True
    include_coverage_metrics: bool = True
    include_type_metrics: bool = True
```

**Metadata:** `options_model=FunctionAnalyticsOptions` on `FunctionMetricsPlugin.metadata`. 

**Plugin execution:**  

```python
class FunctionMetricsPlugin(TargetPlugin):
    metadata: ClassVar[CorePluginMetadata] = FUNCTION_METRICS_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        function_ast_map = ...
        missing_function_goids = ...

        resolver = PluginOptionsResolver(config_source=ctx.config_source)
        options = resolver.get_options(
            self.metadata,
            FunctionAnalyticsOptions,
            dynamic_overrides={
                "function_ast_map": function_ast_map,
                "missing_function_goids": missing_function_goids,
            },
        )

        cfg = FunctionAnalyticsStepConfig(snapshot=ctx.snapshot)
        counts = compute_function_metrics_and_types(ctx.gateway, cfg, options=options)
        ...
```

**Core compute respects `include_graph_metrics`:** 

```python
def compute_function_metrics_and_types(
    gateway: StorageGateway,
    cfg: FunctionAnalyticsStepConfig,
    options: FunctionAnalyticsOptions,
) -> dict[str, int]:
    functions = load_functions(gateway, cfg)

    callgraph = None
    if options.include_graph_metrics:
        callgraph = load_callgraph(gateway, cfg)

    coverage = load_coverage(gateway, cfg) if options.include_coverage_metrics else None
    types = load_types(gateway, cfg) if options.include_type_metrics else None

    metrics_rows = []
    types_rows = []

    for fn in functions:
        base = compute_base_metrics(fn, options=options, coverage=coverage, types=types)

        if options.include_graph_metrics and callgraph is not None:
            graph_metrics = compute_graph_metrics_for_function(fn, callgraph, options)
            base.update(graph_metrics)

        metrics_rows.append(base)

        if options.include_type_metrics:
            types_rows.extend(compute_type_rows_for_function(fn, types))

    write_metrics(gateway, metrics_rows)
    write_types(gateway, types_rows)
    return {"metrics_rows": len(metrics_rows), "types_rows": len(types_rows)}
```

### 4.2 Graphs: `graphs.callgraph`

**Options model:**  

```python
# graphs/plugins/builders/callgraph.py

from dataclasses import dataclass

@dataclass
class CallGraphOptions:
    scope_paths: list[str] | None = None
    include_external_calls: bool = False
    max_module_size_lines: int | None = None
    use_ast_fallback: bool = True

    include_test_files: bool = True
    max_edges_per_function: int | None = None
    skip_stdlib_calls: bool = False
```

**Metadata:** `options_model=CallGraphOptions` on `CALLGRAPH_METADATA`. 

**Plugin execution:**  

```python
class CallGraphPlugin(TargetPlugin):
    metadata: ClassVar[CorePluginMetadata] = CALLGRAPH_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        resolver = PluginOptionsResolver(config_source=ctx.config_source)
        options = resolver.get_options(self.metadata, CallGraphOptions)

        cfg = CallGraphStepConfig(snapshot=ctx.snapshot)
        gateway, repo, commit = ctx.gateway, cfg.repo, cfg.commit

        function_index = load_function_index(gateway, repo=repo, commit=commit)
        all_paths = list(function_index.paths())

        paths = self._filter_paths_by_scope(all_paths, options.scope_paths)

        ...
```

**Scope filter:** 

```python
    @staticmethod
    def _filter_paths_by_scope(
        paths: list[str],
        scope_paths: list[str] | None,
    ) -> list[str]:
        if not scope_paths:
            return paths
        prefixes = tuple(scope_paths)
        return [p for p in paths if p.startswith(prefixes)]
```

Again, same pattern: metadata → options model → resolver → compute logic.

### 4.3 This pattern generalizes to ingestion/export

For any other plugin:

1. Add an options dataclass/Pydantic model.
2. Set `options_model` in its `CorePluginMetadata`.
3. In `execute`, do:

   ```python
   resolver = PluginOptionsResolver(ctx.config_source)
   options = resolver.get_options(self.metadata, MyPluginOptions)
   ```

No domain‑specific code needed in core.

---

## 5. Profiles as first‑class policies (fast vs full, today)

On top of the shared options system, you already sketched **fast vs full** semantics. 

### 5.1 Config shape (conceptual YAML)

```yaml
plugins:
  analytics.function_metrics:
    include_graph_metrics: true
    include_coverage_metrics: true
    include_type_metrics: true

  graphs.callgraph:
    scope_paths: null   # whole repo
    include_external_calls: true
    include_test_files: true
    skip_stdlib_calls: false

profiles:
  fast:
    plugins:
      analytics.function_metrics:
        include_graph_metrics: false
        include_coverage_metrics: false

      graphs.callgraph:
        scope_paths: ["src/"]
        include_test_files: false
        include_external_calls: false
        skip_stdlib_calls: true

  full:
    plugins:
      analytics.function_metrics: {}
      graphs.callgraph: {}
```

Loader puts these into:

* `BuildRunConfig.base_plugin_options` ← `config.plugins`
* `BuildRunConfig.profiles_plugin_options` ← `config.profiles.*.plugins`

…and `ProfiledConfigSource` does the merging.

### 5.2 CLI usage

```bash
# full (default or explicit)
codeintel build --operation compute-hotspots --profile full

# fast profile
codeintel build --operation compute-hotspots --profile fast

# fast with one-off override
codeintel build \
  --operation compute-hotspots \
  --profile fast \
  --plugin-option analytics.function_metrics:include_coverage_metrics=true
```

Under the hood, plugins see:

* `FunctionAnalyticsOptions.include_graph_metrics == False` in fast profile.
* `CallGraphOptions.scope_paths == ["src/"]` in fast profile.

But **all the wiring is via the same shared options system**.

---

## 6. Cross‑cutting integration with metadata & manifests

This is where options plug into the other two big axes: metadata and execution records.  

1. **Metadata couples plugin ↔ options model**

   * `CorePluginMetadata.options_model` is the authoritative “this plugin is configured by type X” field.
   * `PluginOptionsResolver` is parameterized by that model.

2. **Options become part of the execution signature**

   * `compute_options_hash(meta.name, options)` produces `options_hash`. 
   * `options_hash` goes into `compute_input_hash(payload)` → `input_hash`.
   * `PluginExecutionRecord` stores both `options_hash` and `input_hash` for every run.

3. **Profiles are just a structured way to change options**

   * Changing profile from `full` to `fast` → different options dict → different `options_hash` → different `input_hash`.
   * So skip/re‑run semantics are consistent and transparent for both analytics and graphs.

4. **Variant field matches profile**

   * When you run with `--profile fast`, you can set `variant="fast"` on `PluginExecutionRecord`. 
   * Now a record is fully policy‑aware: `(plugin_name, repo, commit, scope_id, variant)`.

So the shared options system isn’t just about nicer config; it’s the **policy surface area** that feeds into:

* metadata,
* caching,
* manifests,
* future ExecutionEngine decisions.

---

## 7. “Next major improvement” on top of this: make profiles first‑class policies

You’re already 90% of the way to a **policy‑driven architecture** with this options system. The next big but still incremental improvement I’d recommend is:

> Introduce a **typed “ExecutionProfile” registry** that lives in `core` and describes, in code, what `fast`, `full`, `ci`, etc. *mean* for a subset of plugins — then wire that into `BuildRunConfig` / `ProfiledConfigSource`.

Concretely:

1. **Define ExecutionProfile types**

   ```python
   # core/plugins/execution/profile.py

   from dataclasses import dataclass
   from typing import Mapping, Any

   @dataclass(frozen=True)
   class ExecutionProfile:
       name: str
       description: str
       # canonical plugin overrides for this profile
       plugin_options: Mapping[str, Mapping[str, Any]]
   ```

2. **Define built‑in profiles programmatically**

   ```python
   FAST_PROFILE = ExecutionProfile(
       name="fast",
       description="Good-enough signal, much faster.",
       plugin_options={
           "analytics.function_metrics": {
               "include_graph_metrics": False,
               "include_coverage_metrics": False,
           },
           "graphs.callgraph": {
               "scope_paths": ["src/"],
               "include_test_files": False,
               "include_external_calls": False,
               "skip_stdlib_calls": True,
           },
       },
   )

   FULL_PROFILE = ExecutionProfile(
       name="full",
       description="Max signal; all heavy analytics and full graph.",
       plugin_options={},
   )

   BUILTIN_PROFILES = {
       FAST_PROFILE.name: FAST_PROFILE,
       FULL_PROFILE.name: FULL_PROFILE,
   }
   ```

3. **Feed ExecutionProfile into BuildRunConfig**

   In the CLI entrypoint, instead of letting YAML fully define profiles, you:

   ```python
   from codeintel.core.plugins.execution.profile import BUILTIN_PROFILES

   run_config = BuildRunConfig(profile=args.profile)

   profile_obj = BUILTIN_PROFILES.get(args.profile or "full")
   if profile_obj:
       run_config.profiles_plugin_options[profile_obj.name] = profile_obj.plugin_options
   ```

   (You can still allow external config to extend/override these.)

4. **Result**

   * Profiles now live in **code** as well as config.
   * They are **documented, typed, and discoverable**, not just YAML blobs.
   * They’re still applied through the same `ProfiledConfigSource` + `PluginOptionsResolver` path, so no new plumbing.

This is a very natural “next big improvement” because it:

* leverages every piece you’ve already put in place,
* deepens the **policy/preset semantics** without changing execution,
* and makes profiles something the team can reason about and evolve over time, not just some scattered config.

---

