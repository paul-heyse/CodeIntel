

# overall implementation plan and code snippets #

Below is a plan that:

* defines **one canonical metadata shape** in `core`,
* shows how **graphs, analytics, and ingestion** all plug into it, and
* gives you a very incremental way to introduce it (no big‑bang rewrite).

I’ll keep code concrete but focus on structure, not every plugin.

---

## 0. Reframe the goal

What we want:

> A **single metadata basis** that:
>
> * is defined in `core`,
> * covers **all plugin domains** (ingest, graph, analytics, export, etc.),
> * is the **only** place that encodes: name, version, kind, stage, capabilities, datasets, incremental behavior, options model, etc.

Domain-specific APIs (graph runtime, ingestion runner, analytics runtime, build) then **read from that same object**, instead of each inventing its own.

So: the “source of truth” is the **core plugin metadata**; graph/analytics/ingestion just *interpret* it.

---

## 1. Target design: a single CorePluginMetadata

First, we make the “one brain” explicit.

### 1.1 Add/clarify a core metadata type

In `core/plugins/types/metadata.py` (or extend `core/plugins/types/protocol.py` if you prefer):

```python
# core/plugins/types/metadata.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Type

# Optional: if you already have PluginKind / PluginStage enums, use those
PluginDomain = str  # "ingest" | "graph" | "analytics" | "export" | "cli" | ...
PluginKind = str    # "builder" | "metric" | "validator" | "tool" | ...
PluginStage = str   # "ingest" | "goid" | "edges" | "function" | "hotspots" | ...


@dataclass(frozen=True)
class CorePluginMetadata:
    # Identity
    name: str                        # canonical id, e.g. "analytics.function_metrics"
    version: str                     # "3.0.0"
    description: str

    # Top-level classification
    domain: PluginDomain             # "ingest" | "graph" | "analytics" | ...
    kind: PluginKind                 # "builder" | "metric" | "validator" | ...
    stage: PluginStage | None = None # domain-specific stage (e.g. "edges", "function")

    # Capabilities (cross-domain)
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()

    # Dataset IO
    produces_tables: tuple[str, ...] = ()
    consumes_tables: tuple[str, ...] = ()

    # Execution semantics
    supports_incremental: bool = False
    scope_aware: bool = False

    # Options & tuning
    options_model: Type[Any] | None = None
    resource_hints: Mapping[str, Any] = field(default_factory=dict)

    # Escape hatch for domain-specific extras (e.g. graph kinds)
    extra: Mapping[str, Any] = field(default_factory=dict)
```

Then, in `core/plugins/types/protocol.py`, instead of a separate `PluginMetadata` you either:

* **Alias**:

  ```python
  from .metadata import CorePluginMetadata as PluginMetadata
  ```

* Or, if you already have a `PluginMetadata` that’s similar, extend it to match this shape (add missing fields like `domain`, `produces_tables`, etc.).

This is the key move: **everything** (graphs, analytics, ingestion) speaks this *one* type.

### 1.2 Thin domain views (optional)

If you want nice typed wrappers per domain, they become *thin views* over this core type.

Example: `GraphPluginMetadata` wraps `CorePluginMetadata`:

```python
# graphs/core/protocol.py

from core.plugins.types.metadata import CorePluginMetadata

@dataclass(frozen=True)
class GraphPluginMetadata:
    core: CorePluginMetadata
    produces_graph_kinds: tuple[str, ...] = ()
    # You can also just put this into core.extra["graph_kinds"]
```

But the **canonical** fields (name, provides/requires, produces_tables, etc.) live in `core.CorePluginMetadata`.

---

## 2. Make plugin bases metadata-aware (all domains)

Now we spread this “core metadata” across all plugin bases, without changing behavior yet.

### 2.1 TargetPlugin (ingest + analytics)

In `build/plugin.py` (or wherever `TargetPlugin` is defined), add an optional metadata attribute:

```python
# build/plugin.py

from typing import ClassVar, Optional
from core.plugins.types.metadata import CorePluginMetadata

class TargetPlugin:
    """Base class for build/Target-style plugins."""

    # Legacy fields
    plugin_name: ClassVar[str]
    plugin_version: ClassVar[str] = "0.0.0"
    plugin_description: ClassVar[str] = ""

    # New: canonical metadata (may be None on older plugins)
    metadata: ClassVar[Optional[CorePluginMetadata]] = None
```

No behavior change. Existing plugins keep working; new ones can set `metadata`.

### 2.2 GraphPluginProtocol

In `graphs/core/protocol.py`, adjust the protocol to use CorePluginMetadata (not a totally separate type):

```python
from core.plugins.types.metadata import CorePluginMetadata

class GraphPluginProtocol(Protocol):
    @property
    def metadata(self) -> CorePluginMetadata: ...
    # or .core_metadata if you still want a GraphPluginMetadata wrapper
```

And if you still want graph-specific metadata, you wrap:

```python
@dataclass(frozen=True)
class GraphPluginMetadata:
    core: CorePluginMetadata
    produces_graph_kinds: tuple[str, ...] = ()
    # any other graph-only fields
```

The important thing: **the `.core` inside this wrapper is the shared object**. There is no duplication.

---

## 3. Attach CorePluginMetadata to one plugin in each domain

Now we actually *use* this in three representative plugins—one analytics, one graph, one ingest. That’s where it starts to become real.

I’ll show concrete but compact snippets.

### 3.1 Analytics: FunctionMetricsPlugin

`analytics/plugins/functions/metrics.py`:

```python
from core.plugins.types.metadata import CorePluginMetadata
from codeintel.analytics.functions import FunctionAnalyticsOptions

FUNCTION_METRICS_METADATA = CorePluginMetadata(
    name="analytics.function_metrics",
    version="3.0.0",
    description="Compute function complexity and type coverage metrics.",
    domain="analytics",
    kind="metric",
    stage="function",
    provides=(
        "analytics.function_metrics",
        "analytics.function_types",
    ),
    requires=("core.goids", "graph.callgraph"),
    produces_tables=(
        "analytics.function_metrics",
        "analytics.function_types",
        "analytics.function_validation",
    ),
    consumes_tables=("core.goids",),
    supports_incremental=False,         # honest for now
    scope_aware=False,
    options_model=FunctionAnalyticsOptions,
)
```

Then in the class:

```python
class FunctionMetricsPlugin(TargetPlugin):
    plugin_name: ClassVar[str] = "functions.metrics"  # legacy id
    plugin_version: ClassVar[str] = FUNCTION_METRICS_METADATA.version
    plugin_description: ClassVar[str] = FUNCTION_METRICS_METADATA.description

    metadata: ClassVar[CorePluginMetadata] = FUNCTION_METRICS_METADATA
    ...
```

### 3.2 Graph: CallGraphPlugin

`graphs/plugins/builders/callgraph.py`:

```python
from core.plugins.types.metadata import CorePluginMetadata

CALLGRAPH_METADATA = CorePluginMetadata(
    name="graphs.callgraph",                # canonical
    version="3.0.0",
    description="Build call graph nodes and edges.",
    domain="graph",
    kind="builder",
    stage="edges",
    provides=("graph.callgraph",),
    requires=("core.goids",),               # and ingest.scip_index if you want
    produces_tables=("graph.call_graph_nodes", "graph.call_graph_edges"),
    consumes_tables=("core.goids", "core.modules"),
    supports_incremental=False,             # honest for now
    scope_aware=False,
    options_model=None,
    extra={
        "graph_kinds": ("callgraph",),
    },
)
```

Then:

```python
class CallGraphPlugin(TargetPlugin):
    plugin_name: ClassVar[str] = "callgraph"    # legacy id
    plugin_version: ClassVar[str] = CALLGRAPH_METADATA.version
    plugin_description: ClassVar[str] = CALLGRAPH_METADATA.description

    metadata: ClassVar[CorePluginMetadata] = CALLGRAPH_METADATA
    ...
```

If you still like a `GraphPluginMetadata` wrapper, define:

```python
GRAPH_CALLGRAPH_METADATA = GraphPluginMetadata(
    core=CALLGRAPH_METADATA,
    produces_graph_kinds=("callgraph",),
)
```

and have graph runtime use `GRAPH_CALLGRAPH_METADATA` but always refer back to `.core` for cross-domain concerns.

### 3.3 Ingestion: ScipIngestPlugin (or RepoScanPlugin)

Pick one ingestion plugin, e.g. SCIP ingest.

`ingestion/plugins/scip_python.py` (or whatever the path is):

```python
from core.plugins.types.metadata import CorePluginMetadata

SCIP_PYTHON_METADATA = CorePluginMetadata(
    name="ingest.scip_python",
    version="1.0.0",
    description="Run scip-python to index Python modules.",
    domain="ingest",
    kind="builder",
    stage="index",
    provides=("ingest.scip_index", "core.symbols"),
    requires=("ingest.modules",),
    produces_tables=("ingest.scip_index", "core.symbols"),
    consumes_tables=("ingest.modules",),
    supports_incremental=True,          # presumably true
    scope_aware=False,
    options_model=None,                 # or your ScipIngestOptions
)
```

Then attach:

```python
class ScipPythonPlugin(TargetPlugin):
    plugin_name: ClassVar[str] = "scip_python"
    plugin_version: ClassVar[str] = SCIP_PYTHON_METADATA.version
    plugin_description: ClassVar[str] = SCIP_PYTHON_METADATA.description

    metadata: ClassVar[CorePluginMetadata] = SCIP_PYTHON_METADATA
    ...
```

Now you have **one metadata type** in core that’s concretely used by three representative plugins across all domains.

---

## 4. Make adapters & registries read the core metadata generically

Now, we start consuming `CorePluginMetadata` in a **generic** way, but only where it’s helpful and safe. This is where the “single basis” becomes obvious.

### 4.1 Graph TargetPluginAdapter: read `.metadata.core` instead of reconstructing

Before, the adapter built `GraphPluginMetadata` from `plugin.plugin_name`, `_PLUGIN_KIND_STAGE_MAP`, etc.

Now, in `graphs/core/adapters.py`, change the metadata builder to:

```python
from core.plugins.types.metadata import CorePluginMetadata
from graphs.core.protocol import GraphPluginMetadata

class TargetPluginAdapter(GraphPluginProtocol):
    def __init__(self, plugin: TargetPlugin, ...):
        self._plugin = plugin
        self._graph_metadata = self._create_graph_metadata()

    def _create_graph_metadata(self) -> GraphPluginMetadata:
        plugin = self._plugin

        # 1) If plugin has canonical core metadata, wrap it
        core_meta = getattr(plugin, "metadata", None)
        if isinstance(core_meta, CorePluginMetadata):
            graph_kinds = tuple(core_meta.extra.get("graph_kinds", ()))
            return GraphPluginMetadata(core=core_meta, produces_graph_kinds=graph_kinds)

        # 2) Fallback: synthesize a minimal core metadata from legacy fields
        plugin_name = plugin.plugin_name
        description = plugin.plugin_description or f"Plugin: {plugin_name}"

        # Determine domain, kind, stage from legacy maps
        # For now, treat all target-wrapped graph plugins as domain="graph"
        kind, stage = _PLUGIN_KIND_STAGE_MAP.get(plugin_name, ("builder", "edges"))

        core = CorePluginMetadata(
            name=plugin_name,
            version=plugin.plugin_version,
            description=description,
            domain="graph",
            kind=kind,
            stage=stage,
        )
        return GraphPluginMetadata(core=core, produces_graph_kinds=())
```

This is the **bridge**:

* For `CallGraphPlugin` (and any graph plugin you migrate), it uses the **canonical** `CorePluginMetadata`.
* For older graph plugins, it creates a “shell” metadata on the fly.

Everything in the graph runtime now gets:

* `.core.name`, `.core.provides`, `.core.produces_tables` etc.
* plus graph-only `.produces_graph_kinds`.

### 4.2 Build plugin registry: expose metadata without changing behavior

In `build/plugin_registry.py`, add a helper as before, but now explicitly typed with `CorePluginMetadata`:

```python
from core.plugins.types.metadata import CorePluginMetadata

def get_core_metadata_for_target(target_name: str) -> CorePluginMetadata | None:
    """Return CorePluginMetadata for a target's plugin, if defined."""
    plugin_cls = get_plugin_for_target(target_name)  # existing helper
    core_meta = getattr(plugin_cls, "metadata", None)
    if isinstance(core_meta, CorePluginMetadata):
        return core_meta
    return None
```

You **don’t** change how build executes plugins yet; you just:

* start using this helper in places where metadata matters:

  * logging (“run analytics.function_metrics 3.0.0”),
  * diagnostics (show required/provided capabilities for a target),
  * future planners.

For example, you might update logging in your build executor from:

```python
log.info("Running plugin %s", plugin.plugin_name)
```

to:

```python
meta = getattr(plugin, "metadata", None)
if isinstance(meta, CorePluginMetadata):
    log.info("Running plugin %s (%s)", meta.name, meta.version)
else:
    log.info("Running plugin %s", plugin.plugin_name)
```

This alone makes the **canonical name** visible everywhere, without breaking anything.

---

## 5. Why this is now obviously “one basis” (and why it helps even before the big pivot)

With these steps:

* There is exactly **one type** (`CorePluginMetadata`) that describes:

  * identity (name, version),
  * domain/kind/stage,
  * capabilities (provides/requires),
  * datasets,
  * incremental/scope behavior,
  * options model.

* That type is **owned by `core`**, not by graphs or build.

* All domains plug into it:

  * `FunctionMetricsPlugin` (analytics) → `CorePluginMetadata(domain="analytics", ...)`
  * `CallGraphPlugin` (graph) → `CorePluginMetadata(domain="graph", ..., extra={"graph_kinds": ...})`
  * `ScipPythonPlugin` (ingest) → `CorePluginMetadata(domain="ingest", ...)`

* Graph runtime’s `GraphPluginMetadata` is now a **thin wrapper**:

  * `GraphPluginMetadata.core` is the shared metadata.
  * graph-only fields are derived from `.core.extra` or passed alongside.

* Build can **gradually** start reading `CorePluginMetadata` for:

  * logging,
  * planning,
  * dataset–capability mapping,
  * without changing execution semantics.

So:

> This change is no longer just “graphs have their metadata, analytics have theirs”.
> Everything meaningful about a plugin lives in `CorePluginMetadata`.
> Graphs, analytics, ingestion just interpret / extend it.

And it’s modular:

* You can ship it in small PRs:

  1. Introduce `CorePluginMetadata` and `TargetPlugin.metadata`.
  2. Attach metadata to the three spine plugins.
  3. Update `TargetPluginAdapter` and add `get_core_metadata_for_target`.
* Every intermediate state is testable and doesn’t break existing behavior.

Once this is in, the **next steps** (later) become almost mechanical:

* The unified `ExecutionEngine` can take `CorePluginMetadata` and drive planning/execution purely off it.
* You can move more ingestion/graph/analytics plugins over by just adding `metadata` attributes, no structural rewrite needed.

If you’d like, next I can propose the *next* tiny step after this: for example, updating your analytics runtime to consume `CorePluginMetadata` for `function_metrics` (instead of hand-coded tables/dependencies), which really shows off the power of this central “metadata brain”.


# function_metrics runtime/plan info derived from CorePluginMetadata zoom in #

Totally fair read – “use metadata in two plugins” doesn’t *feel* like it’s really driving anything yet. Let’s make it do real work in analytics.

In your current repo, the *actual* orchestration for analytics lives in the **build layer** (targets + executor), and the analytics “runtime” is basically:

* the build target graph, and
* the manifest/reporting structures in `analytics.runtime`.

So for this next step we’ll:

> Make `function_metrics` the **first analytics target whose runtime/plan info is derived from `CorePluginMetadata`**, not from scattered strings.

That means:

* Build no longer hard‑codes its table list for `function_metrics`.
* Analytics manifest/plan has a helper that **reads plugin metadata** to know what the step is and what it produces.

I’ll assume we’ve already done the previous step:

* `CorePluginMetadata` exists in `core.plugins.types.metadata`.
* `FunctionMetricsPlugin.metadata` is set to a `CorePluginMetadata` instance.

If not, I’ll show those bits inline so this plan is self‑contained.

---

## 1. Make sure `FunctionMetricsPlugin` exposes `CorePluginMetadata`

First, let’s concretely wire the metadata onto the plugin (if you haven’t already).

### 1.1 Define `CorePluginMetadata` (if not already done)

In `core/plugins/types/metadata.py` (new file or extend an existing one):

```python
# core/plugins/types/metadata.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Type

PluginDomain = str    # "ingest" | "graph" | "analytics" | ...
PluginKind = str      # "builder" | "metric" | "validator" | ...
PluginStage = str     # "ingest" | "goid" | "edges" | "function" | "hotspots" | ...

@dataclass(frozen=True)
class CorePluginMetadata:
    # Identity
    name: str               # canonical id, e.g. "analytics.function_metrics"
    version: str            # "3.0.0"
    description: str

    # Domain / classification
    domain: PluginDomain    # e.g. "analytics"
    kind: PluginKind        # e.g. "metric"
    stage: PluginStage | None = None

    # Capabilities
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()

    # Dataset IO
    produces_tables: tuple[str, ...] = ()
    consumes_tables: tuple[str, ...] = ()

    # Execution semantics
    supports_incremental: bool = False
    scope_aware: bool = False

    # Options & tuning
    options_model: Type[Any] | None = None
    resource_hints: Mapping[str, Any] = field(default_factory=dict)

    # Domain-specific extras
    extra: Mapping[str, Any] = field(default_factory=dict)
```

Then in `core/plugins/types/protocol.py` you can alias:

```python
from codeintel.core.plugins.types.metadata import CorePluginMetadata as PluginMetadata
```

(If you already have a `PluginMetadata`, you can just refit it to this shape.)

### 1.2 Attach metadata to `FunctionMetricsPlugin`

In `analytics/plugins/functions/metrics.py`:

```python
from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.functions import (
    FunctionAnalyticsOptions,
    compute_function_metrics_and_types,
)
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import FunctionAnalyticsStepConfig
from codeintel.core.plugins.types.metadata import CorePluginMetadata
```

Define metadata constants above the class:

```python
FUNCTION_METRICS_METADATA = CorePluginMetadata(
    name="analytics.function_metrics",
    version="3.0.0",
    description="Compute function complexity and type coverage metrics.",
    domain="analytics",
    kind="metric",
    stage="function",
    provides=(
        "analytics.function_metrics",
        "analytics.function_types",
    ),
    requires=("core.goids",),  # later you can add "graph.callgraph" if you want
    produces_tables=(
        "analytics.function_metrics",
        "analytics.function_types",
        "analytics.function_validation",
    ),
    consumes_tables=("core.goids",),
    supports_incremental=False,          # honest today
    scope_aware=False,
    options_model=FunctionAnalyticsOptions,
)
```

Then in the class:

```python
class FunctionMetricsPlugin(TargetPlugin):
    """Compute function complexity and type coverage metrics.

    Output Tables
    -------------
    - analytics.function_metrics: Complexity and size metrics
    - analytics.function_types: Type annotation data
    """

    plugin_name: ClassVar[str] = "function_metrics"  # legacy build id
    plugin_version: ClassVar[str] = FUNCTION_METRICS_METADATA.version
    plugin_description: ClassVar[str] = FUNCTION_METRICS_METADATA.description

    # New: canonical core metadata
    metadata: ClassVar[CorePluginMetadata] = FUNCTION_METRICS_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        ...
```

At this point the plugin and metadata are wired; nothing else breaks.

---

## 2. Make the build target for `function_metrics` read from metadata

This is where the analytics *execution* (via build) actually starts consuming the metadata.

### 2.1 Add a helper to the build plugin registry

In `build/plugin_registry.py`, add a small helper:

```python
# build/plugin_registry.py

from codeintel.core.plugins.types.metadata import CorePluginMetadata

...

def get_core_metadata_for_target(target_name: str) -> CorePluginMetadata | None:
    """Return CorePluginMetadata for a target, if its plugin exposes it.

    This does not affect execution; it's a read-only view into the plugin's
    canonical contract.
    """
    plugin_cls = get_plugin_for_target(target_name)  # existing helper
    meta = getattr(plugin_cls, "metadata", None)
    if isinstance(meta, CorePluginMetadata):
        return meta
    return None
```

This works for *any* target whose plugin class has `.metadata`; right now we’ll only lean on it for `function_metrics` (and later callgraph, etc.).

### 2.2 Update `FUNCTION_METRICS_TARGET` to use metadata for tables/description

Open `build/registry.py` and find `FUNCTION_METRICS_TARGET`:

```python
FUNCTION_METRICS_TARGET = OutputTarget(
    name="function_metrics",
    module="analytics",
    plugin="function_metrics",
    tables=("analytics.function_metrics", "analytics.function_types"),
    dependencies=("goids", "ast"),
    description="Function structural metrics and type annotations.",
)
```

Change it to import the plugin and use its metadata:

```python
from codeintel.analytics.plugins.functions.metrics import FunctionMetricsPlugin

...

FUNCTION_METRICS_TARGET = OutputTarget(
    name="function_metrics",
    module="analytics",
    plugin="function_metrics",
    # Use canonical table list from CorePluginMetadata
    tables=FunctionMetricsPlugin.metadata.produces_tables,
    dependencies=("goids", "ast"),  # keep target-level deps for now
    # Stay in sync with plugin description
    description=FunctionMetricsPlugin.metadata.description,
)
```

**What changed:**

* We are no longer hand‑typing the tables in `OutputTarget`.
* The plugin’s canonical `produces_tables` and `description` are now the **single source of truth**.

**What didn’t change (yet):**

* Target dependencies (`("goids", "ast")`) are still target-level and expressed in target-space, not in capability-space. That’s fine for now; we’ll migrate that later.

### 2.3 (Optional but nice): add a metadata self-check

Still in `build/registry.py`, you can add a small sanity check that runs at import time in dev/tests (or behind a flag):

```python
def _validate_function_metrics_target() -> None:
    """Ensure FUNCTION_METRICS_TARGET agrees with plugin metadata."""
    from codeintel.core.plugins.types.metadata import CorePluginMetadata

    meta = FunctionMetricsPlugin.metadata
    if not isinstance(meta, CorePluginMetadata):
        raise RuntimeError("FunctionMetricsPlugin.metadata is not CorePluginMetadata")

    # Tables: target tables should be a subset of metadata tables
    target_tables = set(FUNCTION_METRICS_TARGET.tables)
    meta_tables = set(meta.produces_tables)
    if not target_tables.issubset(meta_tables):
        raise RuntimeError(
            f"FUNCTION_METRICS_TARGET.tables {target_tables} "
            f"not subset of metadata.produces_tables {meta_tables}"
        )

# Call validation once at module import (or behind an env flag)
_validate_function_metrics_target()
```

This is a guardrail: if someone later edits the plugin to add/remove tables and forgets to adjust the target, tests will fail.

---

## 3. Let the analytics manifest/plan see metadata for `function_metrics`

Now we make the **analytics manifest builder** use `CorePluginMetadata` when it constructs a plan. This is where the “analytics runtime consumes metadata” becomes clear.

Right now:

* `analytics/runtime/manifest.py` defines `AnalyticsPlanInfo`, `AnalyticsRunReport`, `encode_manifest(...)`, but nothing actually populates `AnalyticsPlanInfo` from real plugins.

We’ll add a very small helper there that, for a given list of targets (starting with just `["function_metrics"]`), builds an `AnalyticsPlanInfo` using metadata.

### 3.1 Extend `AnalyticsPlanInfo` construction logic

In `analytics/runtime/manifest.py`, add imports:

```python
# analytics/runtime/manifest.py

from dataclasses import dataclass, field
from typing import Iterable

from codeintel.build.plugin_registry import get_core_metadata_for_target
from codeintel.core.plugins.types.metadata import CorePluginMetadata
```

(You already import dataclasses; just extend as needed.)

Then add a helper to build a plan from a list of build targets:

```python
@dataclass(frozen=True)
class AnalyticsPlanInfo:
    ...
    plan_id: str | None = None
    ordered_steps: tuple[str, ...] = ()
    skipped_steps: tuple[AnalyticsSkippedStep, ...] = ()
    dep_graph: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


def build_plan_for_targets(
    target_names: Iterable[str],
) -> AnalyticsPlanInfo:
    """Build an AnalyticsPlanInfo from build target names using plugin metadata.

    This helper provides a metadata-backed view of the analytics plan.
    For now, it uses CorePluginMetadata for each target's plugin to
    derive step identifiers and (capability-based) dependencies.

    Parameters
    ----------
    target_names
        Names of build targets to include in the plan, in desired
        execution order (e.g., ["function_metrics"]).

    Returns
    -------
    AnalyticsPlanInfo
        Plan info populated from CorePluginMetadata where available.
    """
    ordered_steps: list[str] = []
    dep_graph: dict[str, tuple[str, ...]] = {}

    for target_name in target_names:
        meta = get_core_metadata_for_target(target_name)
        if isinstance(meta, CorePluginMetadata):
            # Use canonical plugin name as the step id
            step_id = meta.name
            ordered_steps.append(step_id)

            # For now, encode dependencies in capability space
            dep_graph[step_id] = tuple(meta.requires)
        else:
            # Fallback: just use the target name if metadata is missing
            ordered_steps.append(target_name)
            dep_graph[target_name] = ()

    # Note: plan_id could be a hash of ordered_steps+dep_graph; keep None for now
    return AnalyticsPlanInfo(
        plan_id=None,
        ordered_steps=tuple(ordered_steps),
        skipped_steps=(),
        dep_graph=dep_graph,
    )
```

So if you call:

```python
plan = build_plan_for_targets(["function_metrics"])
```

You’ll get something like:

```python
plan.ordered_steps == ("analytics.function_metrics",)
plan.dep_graph   == {"analytics.function_metrics": ("core.goids",)}
```

**Important nuance**:

* The `dep_graph` here is **capability-based** (meta.requires) not **target-based** (`("goids", "ast")`). That’s intentional; it reflects the *future* view of dependencies.
* The build target dependency graph is still authoritative for actual execution; this helper is a read-only, analytics-focused view used for manifesting and introspection.

### 3.2 (Optional) Use this helper when building AnalyticsRunReport

If you have (or later add) a place where you construct an `AnalyticsRunReport`, you can now use `build_plan_for_targets` instead of hand-assembling the plan.

For example, imagine a (future) orchestrator:

```python
from codeintel.analytics.runtime import AnalyticsRunReport
from codeintel.analytics.runtime.manifest import build_plan_for_targets

def build_function_metrics_report(repo: str, commit: str) -> AnalyticsRunReport:
    plan = build_plan_for_targets(["function_metrics"])
    # The rest of the report fields (scope, tags, records) would be filled in
    # from the actual run state and PluginExecutionRecord data.
    return AnalyticsRunReport(
        repo=repo,
        commit=commit,
        scope=AnalyticsScope(),  # unresolved for now
        plan=plan,
        tags={},
        records=(),
    )
```

Even if you don’t wire that up yet, the helper + metadata gives you:

* a **concrete, testable** way to derive analytics plan info from metadata, and
* a ready-made hook for future orchestration/serving that wants “what analytic steps did we run?” with canonical names and dependencies.

---

## 4. Why this is a good “next tiny step”

What we’ve accomplished with this step:

1. **FunctionMetricsPlugin has a single, canonical metadata object** that says:

   * “I am `analytics.function_metrics`”
   * “I produce `analytics.function_metrics` / `analytics.function_types` / `analytics.function_validation`”
   * “I require `core.goids`”

2. The **build target** for `function_metrics` now *reads* that metadata:

   * No more duplication of table names or descriptions.
   * A small validation hook can ensure things never drift.

3. The **analytics manifest layer** now has a helper that:

   * Builds `AnalyticsPlanInfo` for `function_metrics` using `CorePluginMetadata`.
   * Encodes dependencies in the *future* capability space (`core.goids`), independent of the old target names.
   * Is completely read-only and doesn’t change execution behavior.

All intermediate states are:

* **Small**: a few lines in 3 files.
* **Safe**: no change to how plugins actually run or how targets are resolved.
* **Forward-looking**: they make `CorePluginMetadata` the obvious, authoritative place to look for “what does function_metrics do?”, both in build and analytics manifest.

Once this is in, the next easy things become:

* doing the same for `hotspots` (analytics),
* and then using `CorePluginMetadata` in `build.operations` to map operations to targets by **capabilities** rather than ad‑hoc strings.

If you want, we can next map out exactly how `hotspots` would be wired the same way so that the `function_metrics → hotspots` chain is entirely metadata-described from plugins upwards.
