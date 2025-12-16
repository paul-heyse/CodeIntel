Here’s a tightened, “best‑in‑class” version of the metadata plan that assumes all the other pieces you’ve built (options, profiles, manifests) and deliberately plugs into them.

Think of this as: **one canonical metadata type in core**, plus a clear lifecycle of how it’s defined, indexed, and consumed by graphs, analytics, ingestion, config, and manifests.   

I’ll structure it as:

1. Design goals
2. Core metadata model (final form)
3. How plugins declare metadata (per domain)
4. Global metadata index (capabilities, datasets)
5. Integration with options/config (profiles, CLI)
6. Integration with manifest/hashing
7. Phased adoption checklist

---

## 1. Design goals (post‑integration)

Metadata should:

* Live in **core** and be the **single source of truth** for:

  * plugin identity, domain, kind, stage,
  * capabilities (`provides`, `requires`),
  * datasets (`produces_tables`, `consumes_tables`),
  * execution semantics (incremental, scope‑aware),
  * options model.
* Be **uniform across domains**: ingest, graph, analytics (and export/serving later). 
* Directly fuel:

  * **options resolution** (`options_model` → `PluginOptionsResolver` → `ConfigSource` / `ProfiledConfigSource`) 
  * **hashing & manifests** (`PluginExecutionRecord`, `upstream_state`, `input_hash`) 
  * **planning & dependency mapping** (capability graph).
* Be **incrementally adoptable**: legacy plugins can run without it; new/migrated ones improve behavior.

---

## 2. Core metadata model (final form)

**File:** `core/plugins/types/metadata.py`

```python
# core/plugins/types/metadata.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence, Type


class PluginDomain(str, Enum):
    INGEST = "ingest"
    GRAPH = "graph"
    ANALYTICS = "analytics"
    EXPORT = "export"
    SERVING = "serving"
    CLI = "cli"


class PluginKind(str, Enum):
    BUILDER = "builder"
    METRIC = "metric"
    VALIDATOR = "validator"
    TOOL = "tool"
    ORCHESTRATOR = "orchestrator"


class PluginStage(str, Enum):
    # You can grow this as needed
    INGEST = "ingest"
    GOIDS = "goids"
    EDGES = "edges"
    FUNCTION = "function"
    HOTSPOTS = "hotspots"
    COVERAGE = "coverage"
    INDEX = "index"
    EXPORT = "export"


@dataclass(frozen=True)
class CorePluginMetadata:
    """
    Canonical plugin metadata used across all domains.

    This type is the single source of truth for:
    - plugin identity
    - domain/kind/stage
    - capabilities
    - dataset IO
    - execution semantics
    - options model
    """

    # Identity
    name: str             # e.g. "analytics.function_metrics"
    version: str          # e.g. "3.0.0"
    description: str

    # High-level classification
    domain: PluginDomain
    kind: PluginKind
    stage: PluginStage | None = None

    # Capabilities (cross-domain semantics)
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()

    # Dataset IO (tables in DuckDB / storage)
    produces_tables: tuple[str, ...] = ()
    consumes_tables: tuple[str, ...] = ()

    # Execution semantics
    supports_incremental: bool = False
    scope_aware: bool = False   # plugin reacts to scopes (paths/modules)

    # Options & tuning
    options_model: Type[Any] | None = None
    resource_hints: Mapping[str, Any] = field(default_factory=dict)

    # Domain-specific extras (e.g., graph kinds, export formats)
    extra: Mapping[str, Any] = field(default_factory=dict)
```

Then alias this as the “official” plugin metadata type:

```python
# core/plugins/types/protocol.py

from codeintel.core.plugins.types.metadata import CorePluginMetadata as PluginMetadata
```

**Key ideas:**

* Strings for capabilities/datasets keep things simple and align with your current code (`"core.goids"`, `"graph.callgraph"`, etc.). 
* `extra` keeps domain‑specific quirks out of the core type while still centralizing the canonical bits.

---

## 3. How plugins declare metadata (across domains)

### 3.1 Base plugin classes expose `metadata`

**File:** `build/plugin.py` (for Target-based plugins)

```python
# build/plugin.py

from typing import ClassVar, Optional
from codeintel.core.plugins.types.metadata import CorePluginMetadata

class TargetPlugin:
    """Base class for build/Target-style plugins (ingest, graph, analytics)."""

    # Legacy fields (still used in some places)
    plugin_name: ClassVar[str]
    plugin_version: ClassVar[str] = "0.0.0"
    plugin_description: ClassVar[str] = ""

    # New: canonical metadata
    metadata: ClassVar[Optional[CorePluginMetadata]] = None
```

This keeps legacy usage intact but lets new/migrated plugins declare metadata.

Graph protocol (if you keep a separate interface):

```python
# graphs/core/protocol.py

from dataclasses import dataclass
from typing import Protocol
from codeintel.core.plugins.types.metadata import CorePluginMetadata

@dataclass(frozen=True)
class GraphPluginMetadata:
    core: CorePluginMetadata
    produces_graph_kinds: tuple[str, ...] = ()

class GraphPluginProtocol(Protocol):
    @property
    def metadata(self) -> CorePluginMetadata: ...
```

### 3.2 Example: analytics – `FunctionMetricsPlugin`

**File:** `analytics/plugins/functions/metrics.py`

```python
from typing import ClassVar

from codeintel.analytics.functions import FunctionAnalyticsOptions
from codeintel.build.plugin import TargetPlugin
from codeintel.core.plugins.types.metadata import (
    CorePluginMetadata,
    PluginDomain,
    PluginKind,
    PluginStage,
)

FUNCTION_METRICS_METADATA = CorePluginMetadata(
    name="analytics.function_metrics",
    version="3.0.0",
    description="Compute function complexity and type coverage metrics.",
    domain=PluginDomain.ANALYTICS,
    kind=PluginKind.METRIC,
    stage=PluginStage.FUNCTION,
    provides=(
        "analytics.function_metrics",
        "analytics.function_types",
    ),
    requires=(
        "core.goids",
        # later, if you want: "graph.callgraph",
    ),
    produces_tables=(
        "analytics.function_metrics",
        "analytics.function_types",
        "analytics.function_validation",
    ),
    consumes_tables=("core.goids",),
    supports_incremental=False,
    scope_aware=False,
    options_model=FunctionAnalyticsOptions,
)

class FunctionMetricsPlugin(TargetPlugin):
    plugin_name: ClassVar[str] = "function_metrics"  # legacy id
    plugin_version: ClassVar[str] = FUNCTION_METRICS_METADATA.version
    plugin_description: ClassVar[str] = FUNCTION_METRICS_METADATA.description

    metadata: ClassVar[CorePluginMetadata] = FUNCTION_METRICS_METADATA

    ...
```

### 3.3 Example: graphs – `CallGraphPlugin`

**File:** `graphs/plugins/builders/callgraph.py`

```python
from typing import ClassVar
from dataclasses import dataclass

from codeintel.build.plugin import TargetPlugin
from codeintel.core.plugins.types.metadata import (
    CorePluginMetadata,
    PluginDomain,
    PluginKind,
    PluginStage,
)

@dataclass
class CallGraphOptions:
    scope_paths: list[str] | None = None
    include_external_calls: bool = False
    max_module_size_lines: int | None = None
    use_ast_fallback: bool = True
    include_test_files: bool = True
    skip_stdlib_calls: bool = False

CALLGRAPH_METADATA = CorePluginMetadata(
    name="graphs.callgraph",
    version="3.0.0",
    description="Build call graph nodes and edges.",
    domain=PluginDomain.GRAPH,
    kind=PluginKind.BUILDER,
    stage=PluginStage.EDGES,
    provides=("graph.callgraph",),
    requires=("core.goids",),
    produces_tables=("graph.call_graph_nodes", "graph.call_graph_edges"),
    consumes_tables=("core.goids", "core.modules"),
    supports_incremental=False,
    scope_aware=True,
    options_model=CallGraphOptions,
    extra={
        "graph_kinds": ("callgraph",),
    },
)

class CallGraphPlugin(TargetPlugin):
    plugin_name: ClassVar[str] = "callgraph"
    plugin_version: ClassVar[str] = CALLGRAPH_METADATA.version
    plugin_description: ClassVar[str] = CALLGRAPH_METADATA.description

    metadata: ClassVar[CorePluginMetadata] = CALLGRAPH_METADATA

    ...
```

### 3.4 Example: ingestion – SCIP plugin

**File:** `ingestion/plugins/scip_python.py`

```python
from typing import ClassVar

from codeintel.build.plugin import TargetPlugin
from codeintel.core.plugins.types.metadata import (
    CorePluginMetadata,
    PluginDomain,
    PluginKind,
    PluginStage,
)

SCIP_PYTHON_METADATA = CorePluginMetadata(
    name="ingest.scip_python",
    version="1.0.0",
    description="Run scip-python to index Python modules.",
    domain=PluginDomain.INGEST,
    kind=PluginKind.BUILDER,
    stage=PluginStage.INDEX,
    provides=("ingest.scip_index", "core.symbols"),
    requires=("ingest.modules",),
    produces_tables=("ingest.scip_index", "core.symbols"),
    consumes_tables=("ingest.modules",),
    supports_incremental=True,
    scope_aware=False,
    options_model=None,  # or ScipPythonOptions later
)

class ScipPythonPlugin(TargetPlugin):
    plugin_name: ClassVar[str] = "scip_python"
    plugin_version: ClassVar[str] = SCIP_PYTHON_METADATA.version
    plugin_description: ClassVar[str] = SCIP_PYTHON_METADATA.description

    metadata: ClassVar[CorePluginMetadata] = SCIP_PYTHON_METADATA

    ...
```

Now all domains expose **the same metadata shape**, and each plugin module has **exactly one** `*_METADATA` object.

---

## 4. Global metadata index (capabilities, datasets, plugins)

Once you have a few plugins wired, you can build **indexes** that the rest of the system uses.

**File:** `core/plugins/registry.py`

```python
# core/plugins/registry.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from codeintel.core.plugins.types.metadata import CorePluginMetadata

@dataclass(frozen=True)
class PluginRegistryIndex:
    by_name: dict[str, CorePluginMetadata]
    by_capability: dict[str, CorePluginMetadata]
    by_output_table: dict[str, CorePluginMetadata]

def build_registry_index(
    all_metadata: Iterable[CorePluginMetadata],
) -> PluginRegistryIndex:
    by_name: dict[str, CorePluginMetadata] = {}
    by_capability: dict[str, CorePluginMetadata] = {}
    by_output_table: dict[str, CorePluginMetadata] = {}

    for meta in all_metadata:
        by_name[meta.name] = meta
        for cap in meta.provides:
            # Last-writer wins if multiple provide same capability
            by_capability[cap] = meta
        for table in meta.produces_tables:
            by_output_table[table] = meta

    return PluginRegistryIndex(
        by_name=by_name,
        by_capability=by_capability,
        by_output_table=by_output_table,
    )
```

You populate `all_metadata` by scanning plugin classes (or explicitly listing them for now):

```python
from codeintel.analytics.plugins.functions.metrics import FUNCTION_METRICS_METADATA
from codeintel.graphs.plugins.builders.callgraph import CALLGRAPH_METADATA
from codeintel.ingestion.plugins.scip_python import SCIP_PYTHON_METADATA

ALL_PLUGIN_METADATA = [
    FUNCTION_METRICS_METADATA,
    CALLGRAPH_METADATA,
    SCIP_PYTHON_METADATA,
    # ...more as you migrate
]

PLUGIN_REGISTRY_INDEX = build_registry_index(ALL_PLUGIN_METADATA)
```

This index powers:

* capability→provider lookups (for `upstream_state`),
* dataset→provider lookups (for debugging or planning),
* plugin name→metadata (for logging, introspection, UI).

---

## 5. Integration with options/config (profiles, CLI)

You already have:

* `ConfigSource`, `PluginOptionsResolver` and `ProfiledConfigSource` in `core/plugins/execution/options.py` 

This is how metadata ties into it concretely:

### 5.1 ConfigSource & resolver (recap with metadata)

```python
# core/plugins/execution/options.py

from codeintel.core.plugins.types.metadata import CorePluginMetadata

class PluginOptionsResolver:
    ...
    def get_options(
        self,
        plugin_metadata: CorePluginMetadata,
        model: Type[T],
        *,
        dynamic_overrides: Mapping[str, Any] | None = None,
    ) -> T:
        raw = self._config_source.get_plugin_options(plugin_metadata.name) or {}
        base = model(**raw)  # config-level defaults

        # apply dynamic_overrides (AST maps, in-memory sets, etc.)
        ...
```

So:

* `CorePluginMetadata.options_model` advertises the **shape**.
* `plugin_metadata.name` is the key into config/profile layers.
* The resolver uses both to construct a typed options object.

### 5.2 ProfiledConfigSource uses metadata names and profiles

We already defined:

* `PluginConfigBundle`, `ProfiledConfigSource`,
* `BuildRunConfig` assembling base/profile/CLI plugin options,
* `run_build` creating a `ProfiledConfigSource` and passing it into `TargetExecutionContext.config_source`. 

That makes metadata + config + profiles work together:

* `meta.name` (e.g., `"analytics.function_metrics"`) is the **canonical identity**.
* `profiles.fast.plugins[meta.name]` holds overrides.
* `ProfiledConfigSource.get_plugin_options(meta.name)` returns merged layer values.

Plugins just call:

```python
resolver = PluginOptionsResolver(ctx.config_source)
options = resolver.get_options(self.metadata, self.metadata.options_model)
```

And now options are:

* **metadata‑driven** (shape + plugin identity),
* **policy/profile aware** (via `ProfiledConfigSource`).

---

## 6. Integration with manifests & hashing

You also have:

* `PluginExecutionRecord`, `compute_options_hash`, `compute_input_hash`, `ManifestStore` in `core/plugins/execution/manifest.py` 

Here’s how metadata slots in.

### 6.1 Derive upstream_state from metadata.requires + registry

Use the capability index built from metadata:

```python
# core/plugins/execution/upstream.py

from typing import Mapping
from codeintel.core.plugins.types.metadata import CorePluginMetadata
from codeintel.core.plugins.execution.manifest import ManifestStore, PluginExecutionRecord
from codeintel.core.plugins.registry import PluginRegistryIndex

def resolve_upstream_state(
    meta: CorePluginMetadata,
    registry: PluginRegistryIndex,
    repo: str,
    commit: str,
    scope_id: str | None,
    variant: str | None,
    manifest_store: ManifestStore,
) -> dict[str, str]:
    """Map required capabilities to provider input hashes."""
    state: dict[str, str] = {}

    for required_cap in meta.requires:
        provider_meta = registry.by_capability.get(required_cap)
        if not provider_meta:
            continue

        rec = manifest_store.load_last_record(
            plugin_name=provider_meta.name,
            repo=repo,
            commit=commit,
            scope_id=scope_id,
            variant=variant,
        )
        if rec:
            state[required_cap] = rec.input_hash

    return state
```

This is the **metadata‑driven** version of “upstream_state”.

### 6.2 Build a canonical input signature using metadata

```python
# core/plugins/execution/signature.py

from typing import Mapping, Tuple
from codeintel.core.plugins.types.metadata import CorePluginMetadata
from codeintel.core.plugins.execution.manifest import (
    compute_options_hash,
    compute_input_hash,
)

def build_input_signature(
    *,
    meta: CorePluginMetadata,
    repo: str,
    commit: str,
    scope_id: str | None,
    variant: str | None,
    options: object | None,
    upstream_state: Mapping[str, str],
) -> tuple[str | None, str]:
    """Compute (options_hash, input_hash) for a plugin run."""
    options_hash = compute_options_hash(meta.name, options)

    payload = {
        "repo": repo,
        "commit": commit,
        "plugin_name": meta.name,
        "plugin_version": meta.version,
        "scope_id": scope_id,
        "variant": variant or "",
        "options_hash": options_hash or "",
        "upstream_state": dict(upstream_state),
    }
    input_hash = compute_input_hash(payload)
    return options_hash, input_hash
```

Now both graphs and analytics runtimes can:

* read `meta` from the plugin,
* use `CorePluginMetadata` to compute `upstream_state` and `input_hash`,
* and persist a `PluginExecutionRecord` with consistent semantics.

### 6.3 An integrated “prepare plugin run” helper

Bringing it together:

```python
# core/plugins/execution/run_context.py

from dataclasses import dataclass
from typing import Mapping

from codeintel.core.plugins.types.metadata import CorePluginMetadata
from codeintel.core.plugins.execution.options import ConfigSource, PluginOptionsResolver
from codeintel.core.plugins.execution.manifest import ManifestStore
from codeintel.core.plugins.registry import PluginRegistryIndex
from codeintel.core.plugins.execution.upstream import resolve_upstream_state
from codeintel.core.plugins.execution.signature import build_input_signature

@dataclass
class PluginRunContext:
    meta: CorePluginMetadata
    repo: str
    commit: str
    scope_id: str | None
    variant: str | None

    options: object | None
    options_hash: str | None
    upstream_state: dict[str, str]
    input_hash: str

def prepare_plugin_run(
    *,
    meta: CorePluginMetadata,
    repo: str,
    commit: str,
    scope_id: str | None,
    variant: str | None,
    config_source: ConfigSource,
    manifest_store: ManifestStore,
    registry_index: PluginRegistryIndex,
    options_resolver: PluginOptionsResolver | None = None,
) -> PluginRunContext:
    resolver = options_resolver or PluginOptionsResolver(config_source=config_source)

    options = None
    if meta.options_model is not None:
        options = resolver.get_options(meta, meta.options_model)

    upstream_state = resolve_upstream_state(
        meta=meta,
        registry=registry_index,
        repo=repo,
        commit=commit,
        scope_id=scope_id,
        variant=variant,
        manifest_store=manifest_store,
    )

    options_hash, input_hash = build_input_signature(
        meta=meta,
        repo=repo,
        commit=commit,
        scope_id=scope_id,
        variant=variant,
        options=options,
        upstream_state=upstream_state,
    )

    return PluginRunContext(
        meta=meta,
        repo=repo,
        commit=commit,
        scope_id=scope_id,
        variant=variant,
        options=options,
        options_hash=options_hash,
        upstream_state=upstream_state,
        input_hash=input_hash,
    )
```

Each runtime (graphs, analytics) can call this and get a unified view of:

* metadata,
* policy‑driven options,
* upstream_state,
* canonical hashes.

That’s your **“transition data manifold”** in code.

---

## 7. Phased adoption checklist (metadata‑focused)

To keep this implementable in small PRs:

### Phase 1 – Core & spine plugins

1. Add `CorePluginMetadata`, enums, and alias in `core/plugins/types`. 
2. Add `metadata: ClassVar[CorePluginMetadata | None]` to `TargetPlugin`.
3. Attach `*_METADATA` and `metadata` to:

   * `FunctionMetricsPlugin`,
   * `CallGraphPlugin`,
   * one ingestion plugin (e.g. SCIP).

### Phase 2 – Registries & basic consumption

4. Build `PluginRegistryIndex` from these three metadata objects.
5. Expose simple helpers in build/graphs:

   * `get_core_metadata_for_target(target_name)`,
   * `TargetPluginAdapter` uses `plugin.metadata` when present. 
6. Use metadata for:

   * target tables/description for `function_metrics`,
   * logging plugin name/version in build and graphs.

### Phase 3 – Options & profiles integration

7. Make sure `options_model` is set in metadata for function_metrics and callgraph.
8. Ensure `FunctionMetricsPlugin` and `CallGraphPlugin` use `PluginOptionsResolver` + `ctx.config_source`. 
9. Wire `ProfiledConfigSource` into `run_build` so metadata names plus profiles actually drive config.

### Phase 4 – Manifest & hashing integration

10. Introduce `PluginExecutionRecord`, hash helpers, `ManifestStore`. 
11. Implement `resolve_upstream_state` and `build_input_signature` using `CorePluginMetadata`.
12. Use `prepare_plugin_run` for:

    * `CallGraphPlugin` in graph runtime,
    * `FunctionMetricsPlugin` in analytics runtime,
    * recording `PluginExecutionRecord` with metadata‑driven hashes.

From there, extending to more plugins is mostly:

* copy/pasting the metadata pattern,
* adding them to `ALL_PLUGIN_METADATA`,
* letting the index, config, and manifests automatically understand them.

This revised plan is intentionally **metadata‑centric**, but it’s shaped by the integrated view: metadata is now explicitly the “brain” that config, profiles, manifests, and planning all plug into, rather than just a nicer way to store a few strings.
