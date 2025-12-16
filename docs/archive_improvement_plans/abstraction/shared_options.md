
# shared options system implementation plan overview #

We’ll build a **shared options system** that:

* centralizes *how* plugins get options,
* is used by `analytics.function_metrics` now,
* is ready for `graphs.callgraph` and other plugins later,
* and doesn’t require any big behavioral changes yet.

I’ll lay this out as a concrete checklist with code snippets.

---

## Step 0 – Mental model

We’re going to introduce:

1. A **ConfigSource** abstraction – “where do plugin options come from?”
2. A **PluginOptionsResolver** – “given a plugin + model, give me an options instance”.
3. A `CallGraphOptions` model and wiring for `CallGraphPlugin`.
4. Wiring for `FunctionMetricsPlugin` to use the resolver instead of hand-building options.

We’ll keep all current behavior, and for callgraph we’ll just *fetch* options but not *use* them yet.

---

## Step 1 – Core: ConfigSource + PluginOptionsResolver

Create a new module:

`core/plugins/execution/options.py`

```python
# core/plugins/execution/options.py

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Protocol, Type, TypeVar, runtime_checkable

from codeintel.core.plugins.types.metadata import CorePluginMetadata

T = TypeVar("T")


@runtime_checkable
class ConfigSource(Protocol):
    """Minimal interface for loading plugin configuration.

    Implementations can read from:
    - static config files (YAML, pyproject, etc),
    - environment variables,
    - CLI arguments,
    - snapshot-specific settings,
    - or any combination of these.

    The key idea: given a canonical plugin name, return a dict of option
    values that can be passed to the plugin's options_model.
    """

    def get_plugin_options(self, plugin_name: str) -> Mapping[str, Any] | None:
        """Return raw option values for a plugin, or None if not configured."""
        ...


class EmptyConfigSource:
    """ConfigSource that always returns no options.

    Useful as a default while wiring up the system. It ensures that
    plugins still see valid option objects with default values.
    """

    def get_plugin_options(self, plugin_name: str) -> Mapping[str, Any] | None:
        return None


class PluginOptionsResolver:
    """Helper to construct typed options objects for plugins.

    This is the central place where plugin options are:
    - fetched from configuration,
    - validated by constructing an options model,
    - and optionally merged with dynamic overrides at runtime.
    """

    def __init__(self, config_source: ConfigSource | None = None) -> None:
        self._config_source = config_source or EmptyConfigSource()

    def get_options(
        self,
        plugin_metadata: CorePluginMetadata,
        model: Type[T],
        *,
        dynamic_overrides: Mapping[str, Any] | None = None,
    ) -> T:
        """Construct an options instance for a plugin.

        Parameters
        ----------
        plugin_metadata
            The plugin's canonical metadata.
        model
            The options model class (dataclass, Pydantic model, etc.).
        dynamic_overrides
            Per-call overrides used for runtime-only fields, such as
            AST caches or in-memory maps. These are *not* taken from
            config, but from the current execution context.

        Returns
        -------
        T
            An instance of `model` populated from configuration and
            dynamic overrides.
        """
        raw = self._config_source.get_plugin_options(plugin_metadata.name) or {}

        # First, create the base options from config only.
        base = model(**raw)  # type: ignore[arg-type]

        if not dynamic_overrides:
            return base

        # Merge dynamic overrides. We support both dataclasses and
        # Pydantic-like models in a simple way.
        if hasattr(base, "__dataclass_fields__"):
            # Dataclass: use dataclasses.replace
            return replace(base, **dynamic_overrides)  # type: ignore[return-value]

        if hasattr(base, "model_copy"):
            # Pydantic v2-style
            return base.model_copy(update=dict(dynamic_overrides))  # type: ignore[return-value]

        if hasattr(base, "copy"):
            # Pydantic v1-style
            return base.copy(update=dict(dynamic_overrides))  # type: ignore[return-value]

        # Fallback: naive attribute setting
        for key, value in dynamic_overrides.items():
            setattr(base, key, value)
        return base
```

This file is:

* **self-contained**,
* domain-agnostic,
* and immediately usable from analytics and graphs.

---

## Step 2 – Wire ConfigSource into TargetExecutionContext

We want **every plugin** to have access to a `ConfigSource` via its execution context.

### 2.1 Extend TargetExecutionContext

Open `build/context.py` (or equivalent) where `TargetExecutionContext` is defined.

Add imports:

```python
# build/context.py

from codeintel.core.plugins.execution.options import ConfigSource, EmptyConfigSource
```

Extend the dataclass:

```python
@dataclass
class TargetExecutionContext:
    snapshot: Snapshot
    gateway: StorageGateway
    logger: Logger
    resources: Resources
    # ... whatever else is already here

    # New: configuration source for plugin options
    config_source: ConfigSource = field(
        default_factory=EmptyConfigSource,
        repr=False,
    )
```

### 2.2 Construct config_source in the build executor

Find where you construct `TargetExecutionContext` (likely in `build/executor.py` or similar). Right now it will look something like:

```python
ctx = TargetExecutionContext(
    snapshot=snapshot,
    gateway=gateway,
    logger=logger,
    resources=resources,
)
```

For now, we can use the simplest possible config source: **no config**, just defaults.

```python
from codeintel.core.plugins.execution.options import EmptyConfigSource

ctx = TargetExecutionContext(
    snapshot=snapshot,
    gateway=gateway,
    logger=logger,
    resources=resources,
    config_source=EmptyConfigSource(),
)
```

Later, you can replace `EmptyConfigSource()` with a real implementation that reads from your config system. For this step, using the empty source guarantees **no behavior change**.

If you already have some central config object that knows about analytics/graphs/ingest settings, you can wrap it immediately (see the “optional” note further below), but you don’t have to.

---

## Step 3 – Add CallGraphOptions and wire it into metadata

Now we define a simple options model for callgraph that we *fetch* via the resolver, but don’t yet use.

### 3.1 Define CallGraphOptions

In `graphs/plugins/builders/callgraph.py`:

Add imports:

```python
from dataclasses import dataclass
from typing import ClassVar, Mapping, Any
```

Define the options model near the top of the file:

```python
@dataclass
class CallGraphOptions:
    """Static configuration options for call graph construction.

    These are configuration-level knobs, not runtime state. They can be
    populated from configuration and policy profiles.
    """

    # Limit call graph to a subset of source paths (repo-relative).
    scope_paths: list[str] | None = None

    # Whether to include calls to library/external functions in the graph.
    include_external_calls: bool = False

    # If set, skip modules larger than this many lines.
    max_module_size_lines: int | None = None

    # Whether to fall back to AST-based analysis when LibCST parsing fails.
    use_ast_fallback: bool = True
```

### 3.2 Add options_model to CALLGRAPH_METADATA

If you’re using `CorePluginMetadata` from our previous step, update it:

```python
from codeintel.core.plugins.types.metadata import CorePluginMetadata
from .callgraph import CallGraphOptions  # or same file

CALLGRAPH_METADATA = CorePluginMetadata(
    name="graphs.callgraph",
    version="3.0.0",
    description="Build call graph nodes and edges.",
    domain="graph",
    kind="builder",
    stage="edges",
    provides=("graph.callgraph",),
    requires=("core.goids",),
    produces_tables=("graph.call_graph_nodes", "graph.call_graph_edges"),
    consumes_tables=("core.goids", "core.modules"),
    supports_incremental=False,
    scope_aware=False,
    options_model=CallGraphOptions,   # NEW
    extra={"graph_kinds": ("callgraph",)},
)
```

And ensure `CallGraphPlugin` exposes it:

```python
class CallGraphPlugin(TargetPlugin):
    plugin_name: ClassVar[str] = "callgraph"
    plugin_version: ClassVar[str] = CALLGRAPH_METADATA.version
    plugin_description: ClassVar[str] = CALLGRAPH_METADATA.description

    metadata: ClassVar[CorePluginMetadata] = CALLGRAPH_METADATA
    ...
```

Now the central “brain” knows:

* “callgraph’s options are described by `CallGraphOptions`.”

---

## Step 4 – Make FunctionMetricsPlugin use PluginOptionsResolver

Now we switch `FunctionMetricsPlugin` to use the resolver instead of hand-rolling options, **but we preserve all dynamic behavior**.

### 4.1 Import the resolver

In `analytics/plugins/functions/metrics.py`, add:

```python
from codeintel.core.plugins.execution.options import PluginOptionsResolver
from codeintel.core.plugins.types.metadata import CorePluginMetadata
```

(We’ll assume `FunctionMetricsPlugin.metadata: CorePluginMetadata` already exists.)

### 4.2 Identify dynamic vs config fields in FunctionAnalyticsOptions

Today you probably construct `FunctionAnalyticsOptions` something like:

```python
opts = FunctionAnalyticsOptions(
    function_ast_map=function_ast_map,
    missing_function_goids=missing_function_goids,
)
```

Where:

* `function_ast_map` is computed at runtime (mapping GOID → parsed AST).
* `missing_function_goids` is a runtime set of IDs where parsing failed.

Those are **dynamic**; they shouldn’t come from config.

Other fields on `FunctionAnalyticsOptions` (like flags for which metrics to include, thresholds, etc.) **are** good candidates for config.

So we’ll:

1. Let `PluginOptionsResolver` construct a “base” `FunctionAnalyticsOptions` from config.
2. Override the dynamic fields via `dynamic_overrides`.

### 4.3 Update execute to use resolver

Find `FunctionMetricsPlugin.execute`. Update it roughly like this:

```python
class FunctionMetricsPlugin(TargetPlugin):
    ...

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        # 1) Build any runtime-only data structures as before
        # (I’ll keep this high-level since your exact code may differ.)
        function_ast_map = ...           # built from snapshot / gateway
        missing_function_goids = ...     # collected during parsing

        # 2) Construct the options resolver from the context's config_source
        resolver = PluginOptionsResolver(config_source=ctx.config_source)

        # 3) Build FunctionAnalyticsOptions from config + dynamic overrides
        options = resolver.get_options(
            self.metadata,
            FunctionAnalyticsOptions,
            dynamic_overrides={
                "function_ast_map": function_ast_map,
                "missing_function_goids": missing_function_goids,
            },
        )

        # 4) Build step config as before
        cfg = FunctionAnalyticsStepConfig(snapshot=ctx.snapshot)

        # 5) Run the existing compute function with the new options object
        result_counts = compute_function_metrics_and_types(
            ctx.gateway,
            cfg,
            options=options,
        )

        return TargetResult.succeeded(
            row_counts={
                "analytics.function_metrics": result_counts.get("metrics_rows", 0),
                "analytics.function_types": result_counts.get("types_rows", 0),
            }
        )
```

**Key point:**

* `FunctionAnalyticsOptions` is still the same model.
* The underlying compute function is untouched.
* The only change is *where* the base options come from:

  * previously: hard-coded or inline defaults in the plugin;
  * now: `ConfigSource → PluginOptionsResolver → FunctionAnalyticsOptions`.

Because `config_source` is currently an `EmptyConfigSource`, this behaves identically to “no config” and just uses model defaults + dynamic overrides.

Later, when you wire a real `ConfigSource`, you’ll be able to control metrics via config (e.g., enable/disable specific metrics, thresholds, etc.).

---

## Step 5 – Make CallGraphPlugin fetch options (but not yet act on them)

We’ll mirror the pattern for callgraph, but keep behavior unchanged for now.

### 5.1 Import resolver and options model

In `graphs/plugins/builders/callgraph.py`:

```python
from codeintel.core.plugins.execution.options import PluginOptionsResolver
from codeintel.core.plugins.types.metadata import CorePluginMetadata
from .callgraph import CallGraphOptions  # if options are in same file
```

Ensure metadata & plugin are wired as in Step 3.

### 5.2 Use resolver inside execute

Find the `execute` method:

```python
class CallGraphPlugin(TargetPlugin):
    ...

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        cfg = CallGraphStepConfig(snapshot=ctx.snapshot)
        gateway, repo, commit = ctx.gateway, cfg.repo, cfg.commit
        ...
```

Update to:

```python
    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        # 1) Load static config options via the resolver
        resolver = PluginOptionsResolver(config_source=ctx.config_source)
        options = resolver.get_options(self.metadata, CallGraphOptions)

        # For phase 1, we don't change behavior yet; but we could log options:
        ctx.logger.debug(
            "CallGraphPlugin options for repo=%s commit=%s: %s",
            ctx.snapshot.repo,
            ctx.snapshot.commit,
            options,
        )

        # 2) Existing behavior continues as before
        cfg = CallGraphStepConfig(snapshot=ctx.snapshot)
        gateway, repo, commit = ctx.gateway, cfg.repo, cfg.commit

        # ... rest of current logic unchanged ...
```

This makes callgraph:

* **options-aware** (it can see `CallGraphOptions`),
* but **behavior-neutral** (you haven’t changed what it does yet).

Later, when you’re comfortable, you can start using `options.scope_paths`/`include_external_calls` in the logic, one at a time.

---

## Optional: Real ConfigSource wrapper (still small, but more power)

Right now we used `EmptyConfigSource` which always returns `{}`. That’s enough to centralize the pattern without behavior change.

If you want to start using real config quickly, you can add a simple wrapper that reads from your existing config structures.

For example, if `Snapshot` carries some kind of `config` dict:

```python
# core/plugins/execution/options.py

class SnapshotConfigSource:
    """ConfigSource backed by the current snapshot's configuration."""

    def __init__(self, snapshot: "Snapshot") -> None:
        self._snapshot = snapshot

    def get_plugin_options(self, plugin_name: str) -> Mapping[str, Any] | None:
        # Example: snapshot.config.plugins[plugin_name] is a dict of options
        plugin_configs = getattr(self._snapshot, "plugins_config", None) or {}
        return plugin_configs.get(plugin_name)
```

Then in `build/executor.py`:

```python
from codeintel.core.plugins.execution.options import SnapshotConfigSource

ctx = TargetExecutionContext(
    snapshot=snapshot,
    gateway=gateway,
    logger=logger,
    resources=resources,
    config_source=SnapshotConfigSource(snapshot),
)
```

You can grow this over time to merge CLI flags, env, etc.

---

## Step 6 – Sanity checks / tests

A few simple tests/checks to validate this change:

1. **Type-level sanity:**

   * `FunctionMetricsPlugin.metadata.options_model is FunctionAnalyticsOptions`
   * `CallGraphPlugin.metadata.options_model is CallGraphOptions`

2. **Runtime sanity:**

   * With `EmptyConfigSource`, both plugins still run successfully on your current test repos.
   * For `function_metrics`, confirm row counts are unchanged vs pre-change.

3. **Future sanity:**

   * Temporarily hack a `SnapshotConfigSource` to return a non-default value (e.g., set `include_external_calls=True` for callgraph, or toggle a metric flag for function_metrics) and log that the options object reflects it. Don’t change behavior yet, just confirm config flows correctly.

---

## Why this meaningfully moves you forward

After this change, you have:

* A **single pattern** for plugin options:

  * `CorePluginMetadata.options_model` advertises the shape.
  * `ConfigSource` defines where they come from.
  * `PluginOptionsResolver` builds them (plus runtime-only overrides).

* Both a **representative analytics plugin** and a **representative graph plugin** wired into it.

* No major behavior changes, but a ton of payoff later:

  * When you introduce policy profiles (`fast`, `full`, `ci`), they’ll plug into the same `ConfigSource`/resolver.
  * When you build the unified `ExecutionContext` / `ExecutionEngine`, you won’t need to redesign options — the backbone is already there.

If you’d like, the next small step after this could be:

* Designing a simple `ProfiledConfigSource` that merges:

  * base config,
  * profile-specific overrides (e.g., `fast` vs `full`),
  * run-specific overrides (CLI flags),
    and then using that in the build executor — which gets you very close to the “policy profile” idea we talked about earlier.


    # ProfiledConfigSource merging mutliple data features #

Nice, this is the fun “policies/profiles but lightweight” step. Let’s wire up a **ProfiledConfigSource** that sits on top of the options system you already have.

I’ll treat this as a concrete implementation plan:

* define what data we’re merging,
* implement `ProfiledConfigSource`,
* thread it into the build executor,
* show how CLI/flags can feed into it,
* and show how `FunctionMetricsPlugin` / `CallGraphPlugin` benefit without changing their code again.

I’ll keep names consistent with what we’ve already been using (`CorePluginMetadata`, `PluginOptionsResolver`, `TargetExecutionContext`).

---

## 1. What we’re merging (conceptually)

We want three layers:

1. **Base config** – long‑lived defaults, usually from files:

   * “By default, function metrics include graph metrics and coverage”.
2. **Profile-specific overrides** – “fast” vs “full” vs “ci”:

   * “In `fast`, don’t compute coverage; in `full`, do everything”.
3. **Run-specific overrides** – CLI flags or ad‑hoc settings:

   * “For this run only, set `max_module_size_lines=1000` for callgraph”.

At runtime, for a given plugin `P`, we want:

```python
effective_options = merge(
    base.plugins[P],
    profiles[active_profile].plugins[P],
    cli_overrides.plugins[P]
)
```

Where each element is a dict of option fields.

The important part: **PluginOptionsResolver stays exactly the same** – it just sees a `ConfigSource`. We’ll implement `ProfiledConfigSource` as a `ConfigSource` that does this merging.

---

## 2. Implement ProfiledConfigSource in core

Open `core/plugins/execution/options.py` where you already have:

* `ConfigSource` protocol
* `EmptyConfigSource`
* `PluginOptionsResolver`

We’ll add:

1. A tiny merge helper.
2. A data holder for profile config.
3. `ProfiledConfigSource`.

### 2.1 Merge helpers

At the top of `options.py`, add:

```python
from dataclasses import dataclass
from typing import Dict
```

Then, below `EmptyConfigSource` (or near it):

```python
def _merge_dicts(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Shallow merge two option dictionaries.

    Values in `override` take precedence over `base`. None means "no layer".
    This is sufficient for typical plugin option dicts where each key is a
    primitive or small collection. If you later need deep merge, you can
    extend this function.
    """
    result: Dict[str, Any] = {}
    if base:
        result.update(base)
    if override:
        result.update(override)
    return result
```

For plugin options, a shallow merge is usually enough (keys are e.g. `include_coverage`, `scope_paths`, etc.).

### 2.2 A simple bundle for config data

Still in `options.py`, add:

```python
@dataclass(frozen=True)
class PluginConfigBundle:
    """Configuration data for all plugins for a single "layer".

    Typically:
    - base:    long-lived defaults from files
    - profile: overrides for a given profile ("fast", "full", "ci")
    - cli:     overrides for the current run from CLI flags / env
    """

    # Mapping from plugin canonical name -> options dict
    plugin_options: Mapping[str, Mapping[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Normalize None to empty mapping
        object.__setattr__(self, "plugin_options", dict(self.plugin_options or {}))

    def get(self, plugin_name: str) -> Mapping[str, Any] | None:
        return self.plugin_options.get(plugin_name)
```

This is just a thin wrapper around a `dict[str, dict[str, Any]]`.

### 2.3 ProfiledConfigSource implementation

Now add `ProfiledConfigSource`:

```python
class ProfiledConfigSource(ConfigSource):
    """ConfigSource that merges base, profile, and run-time overrides.

    Resolution order for a given plugin_name:

        base.plugins[plugin_name]
        → profile.plugins[plugin_name]
        → cli.plugins[plugin_name]

    Later layers override earlier ones on a key-by-key basis.
    """

    def __init__(
        self,
        *,
        base: PluginConfigBundle | None = None,
        profile: PluginConfigBundle | None = None,
        cli: PluginConfigBundle | None = None,
        active_profile_name: str | None = None,
    ) -> None:
        # We keep base/profile/cli bundles even if active_profile_name is None,
        # in case you want a profile-agnostic layer later. For now profile is
        # only used when active_profile_name is not None.
        self._base = base or PluginConfigBundle(plugin_options={})
        self._profile = profile or PluginConfigBundle(plugin_options={})
        self._cli = cli or PluginConfigBundle(plugin_options={})
        self._active_profile_name = active_profile_name

    def get_plugin_options(self, plugin_name: str) -> Mapping[str, Any] | None:
        # 1) Base config
        base_raw = self._base.get(plugin_name)

        # 2) Profile layer: we could support multiple profiles here later,
        # but for now we assume we've already selected the right profile bundle.
        profile_raw = self._profile.get(plugin_name) if self._active_profile_name else None

        # 3) CLI / run overrides
        cli_raw = self._cli.get(plugin_name)

        merged = _merge_dicts(base_raw, profile_raw)
        merged = _merge_dicts(merged, cli_raw)

        return merged or None
```

Note:

* We don’t encode the full concept of multiple profile bundles here – we assume the caller passes in the `profile` bundle appropriate for the active profile. That keeps this class simple and focused.

If you prefer, you can also encode multiple profiles here, but this is enough for a first step.

---

## 3. Build-side: constructing a ProfiledConfigSource

Now we need to:

* decide where base/profile/CLI configs come from,
* construct a `ProfiledConfigSource` in the build executor,
* pass it into `TargetExecutionContext.config_source`.

This is the only step that actually “plugs it in”.

### 3.1 Represent run-level config in the build layer

Let’s define a small run-level config holder, e.g. in `build/options.py`:

```python
# build/options.py

from dataclasses import dataclass, field
from typing import Any, Mapping

from codeintel.core.plugins.execution.options import PluginConfigBundle

@dataclass
class BuildRunConfig:
    """Configuration for a single build/run.

    This captures the plugin option layers that will be merged by
    ProfiledConfigSource.
    """

    # Name of the active profile, if any ("fast", "full", "ci", etc.)
    profile: str | None = None

    # Base plugin option dictionaries
    base_plugin_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    # Profile-specific plugin options.
    #
    # Typically structured as:
    #   {
    #     "fast":  {"analytics.function_metrics": {...}, "graphs.callgraph": {...}},
    #     "full":  {...},
    #     "ci":    {...},
    #   }
    profiles_plugin_options: Mapping[str, Mapping[str, Mapping[str, Any]]] = field(
        default_factory=dict
    )

    # Run-specific (CLI) plugin overrides
    cli_plugin_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def to_profiled_config_bundles(self):
        """Convert to PluginConfigBundle instances for ProfiledConfigSource."""
        base_bundle = PluginConfigBundle(plugin_options=self.base_plugin_options)

        # Select the active profile's plugin options if present
        profile_options = {}
        if self.profile:
            profile_options = self.profiles_plugin_options.get(self.profile, {})
        profile_bundle = PluginConfigBundle(plugin_options=profile_options)

        cli_bundle = PluginConfigBundle(plugin_options=self.cli_plugin_options)

        return base_bundle, profile_bundle, cli_bundle
```

This doesn’t prescribe where the data comes from (YAML, env, etc.) – it just gives build a place to put it.

### 3.2 Parse profile & CLI overrides in the CLI layer

In your CLI command module (e.g. `cli/commands/build.py`), you likely have an argument parser.

We want to:

* add something like `--profile` (or `--mode`),
* optionally add some simple `--plugin-option` flag to test the plumbing.

Example (obviously adapt to your actual CLI framework):

```python
# cli/commands/build.py

import argparse

from codeintel.build.options import BuildRunConfig

def add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=["fast", "full", "ci"],
        default=None,
        help="Execution profile to use for plugin configuration.",
    )
    parser.add_argument(
        "--plugin-option",
        action="append",
        metavar="PLUGIN:KEY=VALUE",
        help=(
            "Override a plugin option for this run. "
            "Example: --plugin-option analytics.function_metrics:include_coverage=false"
        ),
    )

def _parse_plugin_option_flag(flag: str) -> tuple[str, str, str]:
    # "analytics.function_metrics:include_coverage=false" -> ("analytics.function_metrics", "include_coverage", "false")
    plugin_part, kv = flag.split(":", 1)
    key, value = kv.split("=", 1)
    return plugin_part, key, value

def build_main(args: argparse.Namespace) -> None:
    # 1) Create BuildRunConfig from CLI
    run_config = BuildRunConfig(profile=args.profile)

    # 2) Parse CLI plugin overrides into cli_plugin_options dict
    cli_plugin_options: dict[str, dict[str, Any]] = {}
    for flag in args.plugin_option or []:
        plugin_name, key, value_str = _parse_plugin_option_flag(flag)
        # You might want to do some type coercion here; keep it string for now.
        plugin_opts = cli_plugin_options.setdefault(plugin_name, {})
        plugin_opts[key] = value_str

    run_config.cli_plugin_options = cli_plugin_options

    # 3) TODO: load base and profile configs from config files, env, etc.
    # For example:
    # run_config.base_plugin_options = load_base_plugin_options()
    # run_config.profiles_plugin_options = load_profiles_plugin_options()

    # 4) Hand run_config down into the build executor
    from codeintel.build.executor import run_build

    run_build(run_config=run_config, args=args)
```

We’ve now:

* made profile selection explicit (`--profile`),
* allowed per-run plugin option overrides (`--plugin-option`),
* and packaged everything into `BuildRunConfig`.

### 3.3 Build executor creates ProfiledConfigSource

In `build/executor.py` (or equivalent), update the executor entrypoint.

Suppose you have something like:

```python
# build/executor.py

from codeintel.build.context import TargetExecutionContext
from codeintel.build.graph import BuildPlan
from codeintel.storage.gateway import StorageGateway

def run_build(run_config: BuildRunConfig, args: argparse.Namespace) -> None:
    plan = make_plan(...)  # your existing planning logic
    gateway = StorageGateway(...)
    logger = ...
    resources = ...

    for step in plan.steps:
        ctx = TargetExecutionContext(
            snapshot=step.snapshot,
            gateway=gateway,
            logger=logger,
            resources=resources,
            # previously: config_source=EmptyConfigSource(),
        )
        result = step.plugin.execute(ctx)
        ...
```

Update it to construct a `ProfiledConfigSource` per run and reuse it across steps:

```python
from codeintel.core.plugins.execution.options import (
    ProfiledConfigSource,
    PluginConfigBundle,
)

def run_build(run_config: BuildRunConfig, args: argparse.Namespace) -> None:
    plan = make_plan(...)
    gateway = StorageGateway(...)
    logger = ...
    resources = ...

    # 1) Convert BuildRunConfig to plugin config bundles
    base_bundle, profile_bundle, cli_bundle = run_config.to_profiled_config_bundles()

    # 2) Create a single ProfiledConfigSource for this run
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
            config_source=config_source,  # NEW: same source for all plugins in this run
        )
        result = step.plugin.execute(ctx)
        ...
```

Now:

* Every plugin executed in this build gets the same `config_source`.
* `FunctionMetricsPlugin` and `CallGraphPlugin` already use `PluginOptionsResolver(config_source=ctx.config_source)`.
* That resolver now sees:

  * base → profile overrides → CLI overrides.

You’ve just wired in profile/profile-specific/CLI layering without touching plugin code again.

---

## 4. How this looks from a plugin’s perspective

Nothing in the plugin changed. For example, `FunctionMetricsPlugin` is still:

```python
resolver = PluginOptionsResolver(config_source=ctx.config_source)
options = resolver.get_options(
    self.metadata,
    FunctionAnalyticsOptions,
    dynamic_overrides={
        "function_ast_map": function_ast_map,
        "missing_function_goids": missing_function_goids,
    },
)
```

What’s new is:

* `ctx.config_source` is now a `ProfiledConfigSource` that knows about:

  * base config,
  * an active profile,
  * CLI overrides.

So at runtime, for `analytics.function_metrics`, the resolver effectively does:

```python
raw = profiled_config_source.get_plugin_options("analytics.function_metrics")
options = FunctionAnalyticsOptions(**(raw or {}), **dynamic_overrides)
```

Same for callgraph.

---

## 5. Example configuration shapes (optional, but useful for thinking)

You don’t have to adopt this exact external config shape now, but this is how your data *could* look in a config file:

```yaml
# some_config.yml

plugins:
  analytics.function_metrics:
    include_graph_metrics: true
    include_coverage: true

  graphs.callgraph:
    include_external_calls: false
    max_module_size_lines: 2000

profiles:
  fast:
    plugins:
      analytics.function_metrics:
        include_coverage: false
      graphs.callgraph:
        max_module_size_lines: 500

  full:
    plugins:
      analytics.function_metrics:
        include_coverage: true
      graphs.callgraph:
        max_module_size_lines: 5000
```

You’d then have some loader:

```python
def load_base_plugin_options() -> dict[str, dict[str, Any]]:
    # parse config.plugins
    ...

def load_profiles_plugin_options() -> dict[str, dict[str, dict[str, Any]]]:
    # parse config.profiles[profile_name].plugins
    ...
```

and assign those into `BuildRunConfig` before calling `run_build`.

The merging logic you’ve already wired in will then just work.

---

## 6. Why this is a good next step

After this change:

* You have a **real policy/profile mechanism** in place, even if you only use it lightly at first.
* It fits perfectly with the existing `CorePluginMetadata.options_model` + `PluginOptionsResolver` design:

  * Metadata says *what* options exist.
  * ConfigSource hierarchy says *where* they come from (base, profile, CLI).
  * Resolver constructs the typed options object.
* You didn’t touch core compute logic or build planning:

  * All intermediate states are easy to test.
  * You can roll in “fast vs full vs ci” behavior gradually, plugin by plugin, as you gain confidence.

If you want, the next iteration from here could be drafting a simple “fast profile” spec for just two plugins (`analytics.function_metrics` and `graphs.callgraph`) – e.g. which options to change, what impact that has – so that profiles have a concrete, documented semantics your team can lean on.


# “fast profile” spec for just two plugins (analytics.function_metrics and graphs.callgraph) #

Perfect, let’s make “fast mode” something concrete your team can actually talk about and design around.

I’ll define:

* What **profiles** you have (at least `full` and `fast`).
* For each of the two plugins:

  * Which **options** matter,
  * What they are in `full` vs `fast`,
  * What the **functional + performance impact** is.
* A small **config sketch** (YAML-ish) showing how it would look wired into the system we just designed.

You can treat this as a spec: “this is what `fast` means for these two plugins.”

---

## 1. Profiles: semantics

We’ll define just two for now:

### `full` profile

* Goal: *max fidelity, max signal*.
* Characteristics:

  * Run all heavy analytics.
  * Build the richest graph you reasonably can.
  * Prefer completeness over runtime.

### `fast` profile

* Goal: *good-enough signal, much faster*.
* Characteristics:

  * Disable the heaviest metrics/features.
  * Skip or downsample large/expensive portions of the repo.
  * Prefer speed over completeness, but keep semantics *consistent*: outputs are still structurally the same, just less complete or approximate.

You can later add `ci`, `dev`, etc. but this is enough to start.

---

## 2. `analytics.function_metrics` – fast vs full

We’ll assume (either now or soon) that `FunctionAnalyticsOptions` has (or will have) fields that control:

* Whether to compute graph-based metrics.
* Whether to incorporate coverage.
* How deep to go in AST-based complexity.
* Whether to compute very heavy metrics (centrality, multi-pass metrics, etc.).

Here’s a proposed option set and profile mapping.

### 2.1 Option fields (conceptual)

Augment `FunctionAnalyticsOptions` (if needed) with something like:

```python
@dataclass
class FunctionAnalyticsOptions:
    # Existing dynamic fields:
    function_ast_map: dict[int, Any] | None = None
    missing_function_goids: set[int] | None = None

    # New config-driven fields:
    include_graph_metrics: bool = True
    include_coverage_metrics: bool = True
    include_type_metrics: bool = True

    compute_centrality_metrics: bool = True
    max_ast_depth_for_complexity: int | None = None  # None = unbounded
    sample_large_functions: bool = False
    large_function_loc_threshold: int = 1000
```

You don’t have to add all of these at once; this is a “menu”.

### 2.2 Profile values and impact

**Profile table:**

| Option                         | Type | `full` value | `fast` value | Functional impact                                                          | Performance impact                             |
| ------------------------------ | ---- | ------------ | ------------ | -------------------------------------------------------------------------- | ---------------------------------------------- |
| `include_graph_metrics`        | bool | `True`       | `False`      | Fast: no fan-in/out, no “caller-based” metrics.                            | Skip graph joins / traversal per function.     |
| `include_coverage_metrics`     | bool | `True`       | `False`      | Fast: hotspot metrics/noisy risk signals ignore coverage.                  | Avoid joins with coverage datasets.            |
| `include_type_metrics`         | bool | `True`       | `True`       | Same in both; keep type signal high.                                       | No change.                                     |
| `compute_centrality_metrics`   | bool | `True`       | `False`      | Fast: no betweenness/centrality metrics; callgraph risk signal simplified. | Avoid expensive multi-pass graph computations. |
| `max_ast_depth_for_complexity` | int? | `None`       | `40`         | Fast: for deeply nested functions, complexity is capped at “approximate”.  | Avoid ultra-deep AST traversals.               |
| `sample_large_functions`       | bool | `False`      | `True`       | Fast: for huge functions, may compute approximate complexity / metrics.    | Significantly faster on large codebases.       |
| `large_function_loc_threshold` | int  | `1000`       | `500`        | Fast: more functions are considered “large” → more sampling.               | Faster, especially on monolithic modules.      |

**High-level semantic summary:**

* **Full**:

  * Produces the richest metrics you can: complexity, coverage, graph-based metrics, centrality, types.
  * Suitable for nightly, deep analysis, “what should we refactor this quarter?” dashboards.

* **Fast**:

  * Produces:

    * structural metrics (LOC, basic complexity),
    * type metrics,
    * *simplified* hotspot/risk metrics (no centrality, no coverage signals).
  * Graph-based metrics are either off or minimal.
  * Results are still well-formed; no breaking of downstream consumers, just less detail.

You can gradually plug these flags into `compute_function_metrics_and_types` as you implement them.

---

## 3. `graphs.callgraph` – fast vs full

We already defined a simple `CallGraphOptions`:

```python
@dataclass
class CallGraphOptions:
    scope_paths: list[str] | None = None
    include_external_calls: bool = False
    max_module_size_lines: int | None = None
    use_ast_fallback: bool = True
```

Let’s interpret them for profiles.

### 3.1 Additional options for better control

You might want a couple more fields:

```python
@dataclass
class CallGraphOptions:
    scope_paths: list[str] | None = None
    include_external_calls: bool = False
    max_module_size_lines: int | None = None
    use_ast_fallback: bool = True

    # NEW:
    include_test_files: bool = True
    max_edges_per_function: int | None = None
    skip_stdlib_calls: bool = False
```

Again, you can adopt these incrementally.

### 3.2 Profile values and impact

**Profile table:**

| Option                   | Type      | `full` value | `fast` value         | Functional impact                                                                 | Performance impact                           |
| ------------------------ | --------- | ------------ | -------------------- | --------------------------------------------------------------------------------- | -------------------------------------------- |
| `scope_paths`            | list|None | `None`       | `["src/"]` (example) | Fast: ignores non-app code (e.g. tooling in `tools/`, test folders).              | Fewer files → fewer parses + edges.          |
| `include_external_calls` | bool      | `True`       | `False`              | Fast: edges to library/3rd-party funcs are omitted.                               | Less resolution work, fewer edges.           |
| `include_test_files`     | bool      | `True`       | `False`              | Fast: no nodes/edges for tests.                                                   | Skip test modules entirely.                  |
| `max_module_size_lines`  | int|None  | `None`       | `2000`               | Fast: skip or truncate callgraph for very large modules.                          | Avoid worst-case parsing on huge files.      |
| `max_edges_per_function` | int|None  | `None`       | `200`                | Fast: limit fan-out; drop low-signal edges beyond a cap (e.g. in big dispatchers) | Bound memory/time in pathological functions. |
| `skip_stdlib_calls`      | bool      | `False`      | `True`               | Fast: edges into `builtins`, `collections`, etc. are omitted.                     | Less name resolution / symbol lookup.        |
| `use_ast_fallback`       | bool      | `True`       | `True`               | Same: fallback keeps behavior robust.                                             | No change.                                   |

**High-level semantic summary:**

* **Full**:

  * Attempts to build a complete callgraph for the repo:

    * app + tests + tooling, all modules, all edges.
  * Good for full-program analyses (impact, global refactors, etc.).

* **Fast**:

  * Focuses on “business logic”:

    * app modules only (`scope_paths`),
    * no tests,
    * no stdlib/third-party edges,
    * call fan-out capped to avoid pathological blow-ups.
  * Still structurally valid: you can consume `graph.call_graph_nodes/edges` the same way; you just get fewer nodes/edges.

Again, plugin code changes can be incremental:

* start with `scope_paths` + `include_test_files`, then add `include_external_calls`, etc.

---

## 4. Example config: `full` vs `fast` in practice

Here’s a concrete example of what your **base config** + **profiles** could look like, assuming a top-level YAML that later gets parsed into `BuildRunConfig.base_plugin_options` and `BuildRunConfig.profiles_plugin_options`.

### 4.1 Base config (applies to all profiles unless overridden)

```yaml
# config.yml (conceptual)

plugins:
  analytics.function_metrics:
    include_graph_metrics: true
    include_coverage_metrics: true
    include_type_metrics: true
    compute_centrality_metrics: true
    # No AST caps by default
    max_ast_depth_for_complexity: null
    sample_large_functions: false
    large_function_loc_threshold: 1000

  graphs.callgraph:
    scope_paths: null              # whole repo
    include_external_calls: true
    include_test_files: true
    skip_stdlib_calls: false
    max_module_size_lines: null    # no limit
    max_edges_per_function: null   # no cap
```

This is your **`full`-equivalent** baseline.

### 4.2 Profiles section

```yaml
profiles:
  fast:
    plugins:
      analytics.function_metrics:
        include_graph_metrics: false
        include_coverage_metrics: false
        compute_centrality_metrics: false
        max_ast_depth_for_complexity: 40
        sample_large_functions: true
        large_function_loc_threshold: 500

      graphs.callgraph:
        scope_paths:
          - "src/"          # restrict to app code
        include_test_files: false
        include_external_calls: false
        skip_stdlib_calls: true
        max_module_size_lines: 2000
        max_edges_per_function: 200

  full:
    plugins:
      # Maybe you don’t even need this; base already reflects "full".
      # But you can still override here if you want:
      analytics.function_metrics: {}
      graphs.callgraph: {}
```

### 4.3 CLI usage examples

With the `ProfiledConfigSource` + `BuildRunConfig` we designed:

* **Full run** (implicitly uses base config):

  ```bash
  codeintel build --operation compute-hotspots --profile full
  ```

* **Fast run** (uses `profiles.fast` on top of base config):

  ```bash
  codeintel build --operation compute-hotspots --profile fast
  ```

* **Fast run with one-off override**:

  ```bash
  codeintel build \
    --operation compute-hotspots \
    --profile fast \
    --plugin-option analytics.function_metrics:include_coverage_metrics=true
  ```

That last command effectively says:

> “Use the fast profile, but for **this run only**, still include coverage in function metrics.”

The resolver will see the merged config:

* base → profile override `false` → CLI override `true`.

---

## 5. How your team can treat this spec

This gives you a **concrete, documented contract**:

* When someone says “run in fast mode”:

  * They know exactly which metrics/edges are sacrificed.
  * They know we’re still producing consistent, structurally valid outputs.
* When you add new metrics or callgraph features:

  * You can decide: “Is this fast-only, full-only, or both?” and add to the appropriate profile stanza.
* When you build UX on top (CLI, MCP, UI):

  * You can expose “fast vs full” as first-class modes with real semantics, not just “maybe shorter timeout.”

And all of this plugs cleanly into:

* `CorePluginMetadata.options_model`
* `ConfigSource`/`ProfiledConfigSource`
* `PluginOptionsResolver`

…so you’re evolving toward the policy/profile idea **without** needing the full unified execution engine yet.

If you want, next I can take one of these option fields (e.g. `include_graph_metrics` or `scope_paths`) and sketch the concrete changes in `compute_function_metrics_and_types` or `CallGraphPlugin.execute` to actually honor that setting in a minimal, low-risk way.


# include_graph_metrics zoom in #

Nice, this is where the profiles actually start doing something you can *see*.

I’ll do **both** examples you mentioned, but in a very minimal / low‑risk way so you can implement either one independently:

1. Make `analytics.function_metrics` actually honor `include_graph_metrics`.
2. Make `graphs.callgraph` honor `scope_paths`.

In both cases:

* Defaults preserve current behavior (`include_graph_metrics=True`, `scope_paths=None`).
* Changes are local and easy to test.

---

## 1. `analytics.function_metrics` – honoring `include_graph_metrics`

### 1.1 Extend `FunctionAnalyticsOptions` (if needed)

In `analytics/functions/config.py` (or wherever `FunctionAnalyticsOptions` lives), add the flag if it doesn’t exist yet:

```python
# analytics/functions/config.py

from dataclasses import dataclass
from typing import Any, Mapping

@dataclass
class FunctionAnalyticsOptions:
    # Existing dynamic fields
    function_ast_map: Mapping[int, Any] | None = None
    missing_function_goids: set[int] | None = None

    # New config-driven fields
    include_graph_metrics: bool = True
    include_coverage_metrics: bool = True
    include_type_metrics: bool = True

    # You can add more later (centrality, sampling, etc.)
    # compute_centrality_metrics: bool = True
    # ...
```

Default `include_graph_metrics=True` means **no change** to behavior until you flip it in config/profile.

If you already added this earlier, just verify the default is `True`.

---

### 1.2 Make `compute_function_metrics_and_types` respect the flag

Let’s assume `analytics/functions/metrics.py` currently has something like:

```python
# analytics/functions/metrics.py

def compute_function_metrics_and_types(
    gateway: StorageGateway,
    cfg: FunctionAnalyticsStepConfig,
    options: FunctionAnalyticsOptions,
) -> dict[str, int]:
    # 1) Load functions
    functions = load_functions(gateway, cfg)

    # 2) Load callgraph (today: *always* loads)
    callgraph = load_callgraph(gateway, cfg)

    # 3) Load coverage, types, etc.
    coverage = load_coverage(...)
    types = load_types(...)

    # 4) Compute metrics
    metrics_rows = []
    types_rows = []

    for fn in functions:
        base_metrics = compute_base_metrics(fn, options, coverage, types, callgraph)
        metrics_rows.append(base_metrics)
        # etc.

    # 5) Write to DuckDB and return row counts
    write_metrics(gateway, metrics_rows)
    write_types(gateway, types_rows)

    return {
        "metrics_rows": len(metrics_rows),
        "types_rows": len(types_rows),
    }
```

We’ll tweak this in two small ways:

1. Only load callgraph if `include_graph_metrics=True`.
2. Make graph metric computations conditional on that flag.

Rough patch:

```python
# analytics/functions/metrics.py

def compute_function_metrics_and_types(
    gateway: StorageGateway,
    cfg: FunctionAnalyticsStepConfig,
    options: FunctionAnalyticsOptions,
) -> dict[str, int]:
    # 1) Load functions (unchanged)
    functions = load_functions(gateway, cfg)

    # 2) Conditionally load callgraph
    callgraph = None
    if options.include_graph_metrics:
        callgraph = load_callgraph(gateway, cfg)
    # else: leave callgraph = None

    # 3) Load coverage / types as before
    coverage = load_coverage(gateway, cfg) if options.include_coverage_metrics else None
    types = load_types(gateway, cfg) if options.include_type_metrics else None

    metrics_rows: list[dict[str, Any]] = []
    types_rows: list[dict[str, Any]] = []

    for fn in functions:
        # Base metrics (LOC, AST complexity, etc.)
        base = compute_base_metrics(
            fn,
            options=options,
            coverage=coverage,
            types=types,
        )

        # Graph-based metrics only if enabled and callgraph is present
        if options.include_graph_metrics and callgraph is not None:
            graph_metrics = compute_graph_metrics_for_function(fn, callgraph, options)
            base.update(graph_metrics)

        metrics_rows.append(base)

        # Types table, unchanged
        if options.include_type_metrics:
            types_rows.extend(compute_type_rows_for_function(fn, types))

    write_metrics(gateway, metrics_rows)
    write_types(gateway, types_rows)

    return {
        "metrics_rows": len(metrics_rows),
        "types_rows": len(types_rows),
    }
```

Notes:

* `load_callgraph`, `compute_graph_metrics_for_function`, `load_coverage`, `load_types` are placeholders for your current helpers; you just wrap usage with the new flag.
* When `include_graph_metrics=False`, nothing in this function touches callgraph tables at all → cheaper query set and less CPU.

No other parts of the system need to change; downstream code still sees the same `analytics.function_metrics` table schema, just with some graph-related columns either not populated or populated with neutral values (depending on your implementation).

---

### 1.3 Make sure FunctionMetricsPlugin passes the options (it already does)

From the earlier step, your plugin should already be doing:

```python
# analytics/plugins/functions/metrics.py

from codeintel.core.plugins.execution.options import PluginOptionsResolver

class FunctionMetricsPlugin(TargetPlugin):
    metadata: ClassVar[CorePluginMetadata] = FUNCTION_METRICS_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        # dynamic runtime-only bits
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

        result_counts = compute_function_metrics_and_types(
            ctx.gateway,
            cfg,
            options=options,
        )

        return TargetResult.succeeded(
            row_counts={
                "analytics.function_metrics": result_counts.get("metrics_rows", 0),
                "analytics.function_types": result_counts.get("types_rows", 0),
            }
        )
```

Because `FunctionAnalyticsOptions.include_graph_metrics` defaults to `True`, current behavior is unchanged until you flip it via config/profile.

To test, you can:

* Run once with default config (should be same as today).
* Run once with a `fast` profile where:

  ```yaml
  profiles:
    fast:
      plugins:
        analytics.function_metrics:
          include_graph_metrics: false
  ```

…and see graph metrics no longer drive extra queries/processing.

---

## 2. `graphs.callgraph` – honoring `scope_paths`

Now we’ll make `CallGraphPlugin` respect `options.scope_paths` to filter which files are processed.

### 2.1 Confirm CallGraphOptions has `scope_paths`

We previously defined:

```python
# graphs/plugins/builders/callgraph.py

from dataclasses import dataclass

@dataclass
class CallGraphOptions:
    scope_paths: list[str] | None = None
    include_external_calls: bool = False
    max_module_size_lines: int | None = None
    use_ast_fallback: bool = True
    # plus any other fields you’ve added...
```

Default `scope_paths=None` means “whole repo” → same as today.

### 2.2 Filter `paths` in CallGraphPlugin.execute

Today, the core structure of `CallGraphPlugin.execute` looks like (simplified):

```python
class CallGraphPlugin(TargetPlugin):
    plugin_name: ClassVar[str] = "callgraph"
    ...

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        cfg = CallGraphStepConfig(snapshot=ctx.snapshot)
        gateway, repo, commit = ctx.gateway, cfg.repo, cfg.commit

        function_index = load_function_index(gateway, repo=repo, commit=commit)
        paths = function_index.paths()  # iterable of repo-relative paths

        # maybe some early return if no paths

        all_nodes: list[CallGraphNode] = []
        all_edges: list[CallGraphEdge] = []

        for path in paths:
            # normalize path, read file, parse, collect edges...
            ...

        _persist_nodes(gateway, repo, commit, all_nodes)
        _persist_edges(gateway, repo, commit, all_edges)

        return TargetResult.succeeded(
            row_counts={
                "graph.call_graph_nodes": len(all_nodes),
                "graph.call_graph_edges": len(all_edges),
            }
        )
```

We’ll:

1. Fetch options via `PluginOptionsResolver` (which you already wired).
2. Filter `paths` based on `options.scope_paths`.

Patch:

```python
from codeintel.core.plugins.execution.options import PluginOptionsResolver

class CallGraphPlugin(TargetPlugin):
    ...

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        # 1) Resolve static options
        resolver = PluginOptionsResolver(config_source=ctx.config_source)
        options = resolver.get_options(self.metadata, CallGraphOptions)

        cfg = CallGraphStepConfig(snapshot=ctx.snapshot)
        gateway, repo, commit = ctx.gateway, cfg.repo, cfg.commit

        function_index = load_function_index(gateway, repo=repo, commit=commit)
        all_paths = list(function_index.paths())

        # 2) Apply scope_paths filter, if any
        paths = self._filter_paths_by_scope(all_paths, options.scope_paths)

        if not paths:
            ctx.logger.info(
                "CallGraphPlugin: no paths selected after scope filtering "
                "for repo=%s commit=%s (scope_paths=%s)",
                repo,
                commit,
                options.scope_paths,
            )
            return TargetResult.succeeded(
                row_counts={
                    "graph.call_graph_nodes": 0,
                    "graph.call_graph_edges": 0,
                }
            )

        all_nodes: list[CallGraphNode] = []
        all_edges: list[CallGraphEdge] = []

        for path in paths:
            # existing logic, unchanged
            ...

        _persist_nodes(gateway, repo, commit, all_nodes)
        _persist_edges(gateway, repo, commit, all_edges)

        return TargetResult.succeeded(
            row_counts={
                "graph.call_graph_nodes": len(all_nodes),
                "graph.call_graph_edges": len(all_edges),
            }
        )

    @staticmethod
    def _filter_paths_by_scope(
        paths: list[str],
        scope_paths: list[str] | None,
    ) -> list[str]:
        """Filter repo-relative paths by the configured scope_paths.

        scope_paths is a list of prefixes (e.g., ["src/", "app/"]) that
        must match the beginning of the path. If scope_paths is None or
        empty, return the original list unchanged.
        """
        if not scope_paths:
            return paths

        prefixes = tuple(scope_paths)
        return [p for p in paths if p.startswith(prefixes)]
```

Notes:

* This assumes `function_index.paths()` returns repo-relative strings like `"src/foo/bar.py"`. If they’re Path objects, you’d adapt to `.as_posix()` or similar.
* Default `scope_paths=None` → `_filter_paths_by_scope` returns `paths` unchanged.

Now you can configure:

```yaml
profiles:
  fast:
    plugins:
      graphs.callgraph:
        scope_paths:
          - "src/"
          - "lib/"
```

And the callgraph will operate only on those paths in `fast` profile, but on the entire repo in `full`.

Again, no changes to downstream consumers: they still read from `graph.call_graph_nodes` and `graph.call_graph_edges` like before.

---

## 3. Why these are “minimal, low-risk” toggles

Both of these changes:

* **Defaults match current behavior**:

  * `include_graph_metrics=True` → still loads and uses callgraph in function_metrics.
  * `scope_paths=None` → still processes all paths in callgraph.

* **Scope of change is small**:

  * One compute function (`compute_function_metrics_and_types`) and one plugin (`CallGraphPlugin.execute`).
  * No changes to schemas, manifests, or build planning.

* **They plug directly into the profile/config system** you already wired:

  * For function metrics: your `fast` profile sets `include_graph_metrics: false`.
  * For callgraph: your `fast` profile sets `scope_paths: [...]`.

From here, you can iterate slowly:

* Add another flag (`include_coverage_metrics`, `max_module_size_lines`) and gate one more piece of logic,
* Or introduce a `ci` profile that’s just a small tweak of `fast` (e.g., slightly stricter `scope_paths`).
