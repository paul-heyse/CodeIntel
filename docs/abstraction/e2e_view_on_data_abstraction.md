Short version: we’ll take **one ingestion plugin** (I’ll call it `ingest.scip_python`), and bring it into the same world as `analytics.function_metrics` and `graphs.callgraph`:

* same **CorePluginMetadata**,
* same **options / profiles** system,
* same **PluginExecutionRecord + hashing** mechanism.

That gives you a true end‑to‑end chain: *ingest → graphs → analytics* all driven by one metadata / config / manifest “manifold”.   

I’ll walk it in four layers:

1. Add an **options model** for an ingestion plugin + wire into metadata
2. Plug ingestion into the **shared options & profiles** system
3. Plug ingestion into **PluginExecutionRecord + hashing**
4. Show the **end‑to‑end flow** across ingestion, graphs, analytics

---

## 1. Ingestion plugin options + metadata

We’ll pick a concrete ingestion plugin: the SCIP indexer for Python (naming is illustrative; adjust to your actual class / module).

### 1.1 Define `IngestScipOptions`

Create a small, config‑driven options model for this plugin, in e.g. `ingestion/plugins/scip_python_options.py`:

```python
# ingestion/plugins/scip_python_options.py

from dataclasses import dataclass
from typing import Sequence

@dataclass
class IngestScipOptions:
    """Configuration for SCIP-based indexing.

    These are configuration-level knobs: they come from config/profile/CLI
    and are stable across a run. No runtime-only state here.
    """

    # Only index changed files since the last successful run
    incremental_only: bool = True

    # Include/exclude patterns (repo-relative, glob-ish)
    include_paths: Sequence[str] | None = None   # e.g., ["src/", "lib/"]
    exclude_paths: Sequence[str] | None = None   # e.g., ["tests/", "docs/"]

    # Whether to index test files at all
    include_tests: bool = False

    # Controls concurrency / resource use
    max_workers: int | None = None              # None = default heuristic
    max_indexed_files: int | None = None        # hard cap for very large repos

    # Whether to tolerate some files failing to index
    allow_partial_failures: bool = True
```

These map very naturally to “fast vs full” semantics later (incremental only, path scope, etc.). 

### 1.2 Attach `options_model` in `CorePluginMetadata`

In the ingestion plugin module, e.g. `ingestion/plugins/scip_python.py`:

```python
# ingestion/plugins/scip_python.py

from typing import ClassVar

from codeintel.build.plugin import TargetPlugin
from codeintel.core.plugins.types.metadata import CorePluginMetadata
from ingestion.plugins.scip_python_options import IngestScipOptions

SCIP_PYTHON_METADATA = CorePluginMetadata(
    name="ingest.scip_python",
    version="1.0.0",
    description="Run scip-python to index Python modules.",
    domain="ingest",
    kind="builder",
    stage="index",

    # Capabilities in your unified vocabulary
    provides=(
        "ingest.scip_index",
        "core.symbols",
    ),
    requires=(
        "ingest.modules",      # or whatever feeds file lists
    ),

    produces_tables=(
        "ingest.scip_index",
        "core.symbols",
    ),
    consumes_tables=(
        "ingest.modules",
    ),

    supports_incremental=True,
    scope_aware=False,         # we treat repo as scope for now
    options_model=IngestScipOptions,
    extra={},
)
```

Expose on the plugin class:

```python
class ScipPythonPlugin(TargetPlugin):
    plugin_name: ClassVar[str] = "scip_python"  # legacy build id
    plugin_version: ClassVar[str] = SCIP_PYTHON_METADATA.version
    plugin_description: ClassVar[str] = SCIP_PYTHON_METADATA.description

    metadata: ClassVar[CorePluginMetadata] = SCIP_PYTHON_METADATA

    ...
```

Now ingestion is using the same **CorePluginMetadata** shape as graphs & analytics. 

---

## 2. Wire ingestion through the shared options & profile system

You already have:

* `ConfigSource` + `PluginOptionsResolver` in `core/plugins/execution/options.py` 
* `ProfiledConfigSource` + `PluginConfigBundle` and `BuildRunConfig` in the build layer 

We’ll just plug ingestion into that.

### 2.1 Use `PluginOptionsResolver` in the ingestion plugin

In `ScipPythonPlugin.execute`, change to:

```python
from codeintel.core.plugins.execution.options import PluginOptionsResolver
from ingestion.plugins.scip_python_options import IngestScipOptions

class ScipPythonPlugin(TargetPlugin):
    metadata: ClassVar[CorePluginMetadata] = SCIP_PYTHON_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        # 1) Resolve options from config/profile/CLI
        resolver = PluginOptionsResolver(config_source=ctx.config_source)
        options = resolver.get_options(self.metadata, IngestScipOptions)

        # 2) Use these options to drive ingestion
        cfg = ScipIngestStepConfig(snapshot=ctx.snapshot)  # whatever you already have
        gateway = ctx.gateway

        # Example usage
        file_list = load_ingest_modules(gateway, cfg)

        # Path filtering (include/exclude/tests)
        paths = _filter_paths(file_list.paths(), options)

        # Incremental mode: maybe restrict to changed files
        if options.incremental_only:
            paths = _filter_to_changed_paths(gateway, cfg, paths)

        # Concurrency / caps
        paths = _limit_paths(paths, options.max_indexed_files)

        row_counts = run_scip_indexer(
            gateway=gateway,
            paths=paths,
            max_workers=options.max_workers,
            allow_partial_failures=options.allow_partial_failures,
        )

        return TargetResult.succeeded(row_counts=row_counts)
```

Simple helpers:

```python
def _filter_paths(
    paths: list[str],
    options: IngestScipOptions,
) -> list[str]:
    filtered = list(paths)

    # include_paths: keep only these prefixes
    if options.include_paths:
        prefixes = tuple(options.include_paths)
        filtered = [p for p in filtered if p.startswith(prefixes)]

    # exclude_paths: drop these prefixes
    if options.exclude_paths:
        excl = tuple(options.exclude_paths)
        filtered = [p for p in filtered if not p.startswith(excl)]

    if not options.include_tests:
        filtered = [p for p in filtered if not _looks_like_test_path(p)]

    return filtered


def _limit_paths(paths: list[str], max_files: int | None) -> list[str]:
    if max_files is None:
        return paths
    return paths[:max_files]
```

Defaults match current behavior as long as:

* `incremental_only=True` matches your existing incremental behaviour (or set to `False` if current default is full reindex),
* include/exclude lists default `None` (no filtering),
* max limits `None`.

### 2.2 Declare ingestion profile semantics (fast vs full)

Extend your config (conceptually) so ingestion also has profile knobs, just like analytics + graphs. 

```yaml
# config.yml (conceptual)

plugins:
  ingest.scip_python:
    incremental_only: false
    include_paths: null          # whole repo
    exclude_paths: null
    include_tests: true
    max_workers: null
    max_indexed_files: null
    allow_partial_failures: true

profiles:
  fast:
    plugins:
      ingest.scip_python:
        incremental_only: true
        include_paths:
          - "src/"
        exclude_paths:
          - "tests/"
        include_tests: false
        max_indexed_files: 20000
        max_workers: 4
  full:
    plugins:
      ingest.scip_python: {}
```

Then your config loader populates:

* `BuildRunConfig.base_plugin_options["ingest.scip_python"]` ← `config.plugins.ingest.scip_python`
* `BuildRunConfig.profiles_plugin_options["fast"]["ingest.scip_python"]` ← `config.profiles.fast.plugins.ingest.scip_python`

…and `ProfiledConfigSource` merges base + profile + CLI overrides exactly the same way as for analytics & graphs. 

No new plumbing required: the ingestion plugin just sees a fully merged `IngestScipOptions`.

---

## 3. Bring ingestion into the centralized hashing + manifest world

You already defined:

* `PluginExecutionRecord`, `PluginStatus`, `compute_options_hash`, `compute_input_hash` in `core/plugins/execution/manifest.py`  
* `ManifestStore` (and likely a DuckDB implementation) 

Graphs + analytics are wired to use these; we’ll now make ingestion use them too, so *all three domains* share the same notion of input/options hashing and execution records.

### 3.1 Introduce a generic helper `build_plugin_execution_record`

Right now your doc shows separate helpers for graphs and analytics that are almost identical. For ingestion, it’s cleanest to factor a generic helper in `core/plugins/execution/manifest.py`:

```python
# core/plugins/execution/manifest.py

from typing import Mapping
from datetime import datetime

from codeintel.core.plugins.types.metadata import CorePluginMetadata

def build_plugin_execution_record(
    *,
    meta: CorePluginMetadata,
    repo: str,
    commit: str,
    scope_id: str | None,
    variant: str | None,
    options: object | None,
    upstream_state: Mapping[str, str],
    row_counts: Mapping[str, int],
    status: PluginStatus,
    started_at: datetime,
    finished_at: datetime,
    extra: Mapping[str, Any] | None = None,
) -> PluginExecutionRecord:
    """Construct a PluginExecutionRecord for any plugin domain.

    This unifies graphs, analytics, and ingestion around the same input
    signature and hashing rules.
    """
    options_hash = compute_options_hash(meta.name, options)

    input_payload: dict[str, object] = {
        "repo": repo,
        "commit": commit,
        "plugin_name": meta.name,
        "plugin_version": meta.version,
        "scope_id": scope_id,
        "variant": variant or "",
        "options_hash": options_hash or "",
        "upstream_state": dict(upstream_state),
    }

    input_hash = compute_input_hash(input_payload)

    return PluginExecutionRecord(
        plugin_name=meta.name,
        version=meta.version,
        repo=repo,
        commit=commit,
        scope_id=scope_id,
        variant=variant,
        status=status,
        input_hash=input_hash,
        options_hash=options_hash,
        row_counts=dict(row_counts),
        upstream_state=dict(upstream_state),
        started_at=started_at,
        finished_at=finished_at,
        extra=dict(extra or {}),
    )
```

Graphs & analytics can eventually switch to this too, but ingestion can use it immediately.

### 3.2 Ingestion orchestrator: record SCIP runs via `PluginExecutionRecord`

Wherever you orchestrate ingestion (e.g. `ingestion/runtime/orchestrator.py`), wire in manifest recording similar to graphs/analytics: 

```python
# ingestion/runtime/orchestrator.py

from datetime import datetime, timezone
from typing import Mapping

from codeintel.core.plugins.execution.manifest import (
    ManifestStore,
    PluginStatus,
    build_plugin_execution_record,
)
from codeintel.core.plugins.types.metadata import CorePluginMetadata
from ingestion.plugins.scip_python import ScipPythonPlugin

class IngestionOrchestrator:
    def __init__(self, manifest_store: ManifestStore, ...):
        self._manifest_store = manifest_store
        ...

    async def run_scip_ingest(
        self,
        *,
        ctx: TargetExecutionContext,
        repo: str,
        commit: str,
        variant: str | None,
    ) -> None:
        plugin_cls = ScipPythonPlugin
        plugin = plugin_cls()  # or however you construct it
        meta: CorePluginMetadata = plugin_cls.metadata

        # 1) Options: we already wired options via PluginOptionsResolver inside execute.
        #    If you also want options here, you can mirror that or trust the plugin.

        # 2) Upstream state: ingestion.scip_python depends on ingest.modules
        upstream_state: dict[str, str] = {}

        modules_record = self._manifest_store.load_last_record(
            plugin_name="ingest.modules",  # or actual provider plugin id
            repo=repo,
            commit=commit,
            scope_id=None,
            variant=variant,
        )
        if modules_record:
            upstream_state["ingest.modules"] = modules_record.input_hash

        # 3) Execute plugin and time it
        started_at = datetime.now(timezone.utc)
        try:
            result = await plugin.execute(ctx)
            status = PluginStatus.SUCCESS
        except Exception:
            # your existing exception/logging behavior
            status = PluginStatus.FAILED
            result = None

        finished_at = datetime.now(timezone.utc)

        row_counts = getattr(result, "row_counts", {}) if result is not None else {}

        # 4) Build & record execution entry (scope_id=None for full-repo ingest)
        record = build_plugin_execution_record(
            meta=meta,
            repo=repo,
            commit=commit,
            scope_id=None,      # ingestion is typically full-repo; refine later if needed
            variant=variant,
            options=None,       # or pass the IngestScipOptions if you have it handy
            upstream_state=upstream_state,
            row_counts=row_counts,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            extra={"kind": "ingest"},
        )
        self._manifest_store.append_record(record)
```

To capture `options_hash` properly, you can either:

* Pass the actual `IngestScipOptions` instance you used inside `execute`, or
* Move `PluginOptionsResolver` usage up into the orchestrator, pass `options` both to `plugin.execute` and to `build_plugin_execution_record`.

Conceptually:

```python
resolver = PluginOptionsResolver(config_source=ctx.config_source)
options = resolver.get_options(meta, IngestScipOptions)

result = await plugin.execute_with_options(ctx, options)  # small refactor
...
record = build_plugin_execution_record(
    meta=meta,
    ...,
    options=options,
    ...
)
```

Either way, ingestion now produces a **PluginExecutionRecord** with:

* `input_hash` computed the exact same way as graphs & analytics,
* `options_hash` from the same helper,
* `upstream_state["ingest.modules"]` recording its dependency signature.

---

## 4. End‑to‑end: ingestion → graphs → analytics under one “data manifold”

With this in place, the story across domains looks like this:

### 4.1 Profiles + options

* CLI chooses profile: `--profile fast` / `--profile full`.

* `BuildRunConfig` gets:

  * `base_plugin_options` from `config.plugins`,
  * `profiles_plugin_options` from `config.profiles[profile].plugins`,
  * `cli_plugin_options` from `--plugin-option` flags. 

* `ProfiledConfigSource` merges them and is placed on `TargetExecutionContext.config_source`.
  Every plugin (ingest, graphs, analytics) uses `PluginOptionsResolver(ctx.config_source)` + `metadata.options_model` to obtain typed options. 

So:

* `IngestScipOptions` picks up “fast ingest” vs “full ingest” knobs,
* `CallGraphOptions` picks up “fast graph” vs “full graph” settings,
* `FunctionAnalyticsOptions` picks up “fast analytics” vs “full analytics” toggles.

### 4.2 Metadata + capabilities

* `SCIP_PYTHON_METADATA.provides=("ingest.scip_index","core.symbols")`
* `CALLGRAPH_METADATA.requires=("core.goids", "ingest.scip_index")` (once you add it) 
* `FUNCTION_METRICS_METADATA.requires=("core.goids", "graph.callgraph")`

That’s the **capability manifold**: the build/engine can reason about which plugins provide what, independent of targets.

### 4.3 Hashes + manifests

For each plugin:

1. Options resolved via profiles → `options` instance
2. `options_hash = compute_options_hash(meta.name, options)`
3. `upstream_state` assembled from prior `PluginExecutionRecord`s (ingest → graph → analytics)
4. `input_hash = compute_input_hash({ repo, commit, meta.name, meta.version, scope_id, variant, options_hash, upstream_state })`  
5. Unified `PluginExecutionRecord` appended via `ManifestStore`. 

So on a fast profile run:

* `ingest.scip_python` runs with narrow include_paths, incremental_only, etc.
* `graphs.callgraph` uses those results and records `upstream_state["ingest.scip_index"] = ingest_record.input_hash`.
* `analytics.function_metrics` uses callgraph and records `upstream_state["graph.callgraph"] = callgraph_record.input_hash`.

**All three share**:

* the same metadata shape,
* the same options / profile infrastructure,
* the same hashing + manifest model.

That’s exactly the “unified data manifold” you were aiming for.

---

## 5. How to roll this out incrementally

You can land this in three small PRs:

1. **PR A – Ingestion metadata + options**

   * Add `IngestScipOptions`.
   * Add `SCIP_PYTHON_METADATA` with `options_model=IngestScipOptions`.
   * Make `ScipPythonPlugin.metadata = SCIP_PYTHON_METADATA`.
   * Use `PluginOptionsResolver` in `ScipPythonPlugin.execute`, but keep behavior identical (defaults match current behavior).

2. **PR B – Profiles + config for ingestion**

   * Extend your existing config YAML / loader to include `plugins.ingest.scip_python` entry and `profiles.fast.plugins.ingest.scip_python` overrides.
   * No plugin code changes; just exercise that options now change under `--profile fast`.

3. **PR C – Manifest integration**

   * Add `build_plugin_execution_record` helper to `core/plugins/execution/manifest.py`.
   * Wire ingestion orchestrator to build `PluginExecutionRecord` for SCIP runs using that helper.
   * (Optionally) start using ingestion’s `PluginExecutionRecord` as upstream_state in graphs.

Each step is testable, reversible, and additive—but they line up perfectly with the metadata, options, and hashing work you’ve already done for graphs + analytics.

If you’d like, the next thing we can do is pick the **exact** ingestion plugin you want (if it’s not SCIP) and adapt this skeleton to its real tables / capabilities / current behavior so it’s almost a copy‑paste into your repo.
