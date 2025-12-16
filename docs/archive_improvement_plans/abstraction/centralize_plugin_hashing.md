Love it, this is the one that makes all the skip/rehash logic feel sane.

I’ll turn Change 3 into a concrete plan with code, in four steps:

1. Add **core manifest primitives** in `core/plugins/execution/manifest.py`
2. Make **graphs runtime** use them
3. Make **analytics runtime** use them
4. Leave **build/ingestion** untouched for now

I’ll assume the earlier pieces we’ve discussed exist:

* `CorePluginMetadata` in `core/plugins/types/metadata.py`
* `PluginOptionsResolver` and `ConfigSource` in `core/plugins/execution/options.py`
* `FunctionMetricsPlugin` and `CallGraphPlugin` have `metadata: CorePluginMetadata`

If some of that isn’t in place yet, you can still use this as a target and adapt names.

---

## 1. Core: manifest primitives in `core/plugins/execution/manifest.py`

Create a new module:

```python
# core/plugins/execution/manifest.py

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, MutableMapping, Protocol

from codeintel.core.plugins.types.metadata import CorePluginMetadata
```

### 1.1 PluginStatus enum

If you don’t already have a suitable enum, define one here for graphs + analytics:

```python
from enum import Enum

class PluginStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"
```

You can later map this to any existing status enums you have; for now, this is internal to this module and the runtimes.

### 1.2 PluginExecutionRecord dataclass

```python
@dataclass(frozen=True)
class PluginExecutionRecord:
    """Canonical record of a single plugin execution.

    This is the shape both graphs and analytics runtimes should use
    when recording what happened during a run. The unified execution
    engine will later adopt this type directly.
    """

    # Identity
    plugin_name: str               # canonical name, e.g. "analytics.function_metrics"
    version: str                   # plugin version from metadata
    repo: str                      # "owner/repo"
    commit: str                    # commit SHA
    scope_id: str | None           # scope hash (paths/modules) or None
    variant: str | None            # "fast", "full", etc. (optional)

    # Status
    status: PluginStatus

    # Hashes
    input_hash: str                # hash of all logical inputs
    options_hash: str | None       # hash of options_model fields only

    # Outputs
    row_counts: Dict[str, int]     # dataset_name -> row count

    # State and timing
    upstream_state: Dict[str, str] # capability -> upstream state signature
    started_at: datetime
    finished_at: datetime

    # Misc
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.finished_at - self.started_at).total_seconds() * 1000.0
```

### 1.3 Hash helpers

```python
def compute_options_hash(plugin_name: str, options: object | None) -> str | None:
    """Compute a stable hash for a plugin's options.

    This function should be used by both graphs and analytics runtimes
    whenever they record a PluginExecutionRecord.
    """
    if options is None:
        return None

    # Try to get a dict-like view of the options
    if hasattr(options, "dict"):
        # Pydantic v1
        raw = options.dict()
    elif hasattr(options, "model_dump"):
        # Pydantic v2
        raw = options.model_dump()
    elif hasattr(options, "__dict__"):
        raw = vars(options)
    else:
        # Fallback: best-effort stringification
        raw = {"_repr": repr(options)}

    payload = {
        "plugin": plugin_name,
        "options": raw,
    }

    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def compute_input_hash(payload: Mapping[str, object]) -> str:
    """Compute a stable hash from a generic payload.

    Used as the "input signature" for a plugin execution, combining repo,
    commit, plugin identity, options hash, and upstream state.
    """
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
```

### 1.4 ManifestStore protocol

```python
class ManifestStore(Protocol):
    """Abstract interface for storing and retrieving execution records."""

    def load_last_record(
        self,
        *,
        plugin_name: str,
        repo: str,
        commit: str,
        scope_id: str | None,
        variant: str | None,
    ) -> PluginExecutionRecord | None:
        """Return the most recent record for this (plugin, repo, commit, scope, variant)."""
        ...

    def append_record(self, record: PluginExecutionRecord) -> None:
        """Persist a new PluginExecutionRecord."""
        ...
```

### 1.5 Example DuckDB-backed implementation (skeleton)

You can adapt this to wrap your existing tracking tables. The key is to **centralize** the read/write logic, even if the schema is the same as today.

```python
# core/plugins/execution/manifest.py (continued)

from codeintel.storage.gateway import StorageGateway  # adjust to your actual import

class DuckDBManifestStore(ManifestStore):
    """DuckDB-backed ManifestStore.

    This is a thin wrapper over your existing tracking tables. You can
    adapt the SQL to match your actual schema.
    """

    def __init__(self, gateway: StorageGateway) -> None:
        self._gw = gateway

    def load_last_record(
        self,
        *,
        plugin_name: str,
        repo: str,
        commit: str,
        scope_id: str | None,
        variant: str | None,
    ) -> PluginExecutionRecord | None:
        # NOTE: adapt field names & table name to your existing schema.
        sql = """
            SELECT
                plugin_name,
                version,
                repo,
                commit,
                scope_id,
                variant,
                status,
                input_hash,
                options_hash,
                row_counts,
                upstream_state,
                started_at,
                finished_at,
                extra
            FROM plugin_execution_records
            WHERE plugin_name = ?
              AND repo = ?
              AND commit = ?
              AND (scope_id IS ? OR scope_id = ?)
              AND (variant IS ? OR variant = ?)
            ORDER BY finished_at DESC
            LIMIT 1
        """
        params = (
            plugin_name,
            repo,
            commit,
            scope_id,
            scope_id,
            variant,
            variant,
        )
        rows, cols = self._gw.query(sql, params)
        if not rows:
            return None

        row = rows[0]
        data = dict(zip(cols, row))

        return PluginExecutionRecord(
            plugin_name=data["plugin_name"],
            version=data["version"],
            repo=data["repo"],
            commit=data["commit"],
            scope_id=data["scope_id"],
            variant=data["variant"],
            status=PluginStatus(data["status"]),
            input_hash=data["input_hash"],
            options_hash=data["options_hash"],
            row_counts=json.loads(data["row_counts"]),
            upstream_state=json.loads(data["upstream_state"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            finished_at=datetime.fromisoformat(data["finished_at"]),
            extra=json.loads(data["extra"] or "{}"),
        )

    def append_record(self, record: PluginExecutionRecord) -> None:
        # NOTE: adapt to your existing table schema
        sql = """
            INSERT INTO plugin_execution_records (
                plugin_name,
                version,
                repo,
                commit,
                scope_id,
                variant,
                status,
                input_hash,
                options_hash,
                row_counts,
                upstream_state,
                started_at,
                finished_at,
                extra
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            record.plugin_name,
            record.version,
            record.repo,
            record.commit,
            record.scope_id,
            record.variant,
            record.status.value,
            record.input_hash,
            record.options_hash,
            json.dumps(record.row_counts, sort_keys=True),
            json.dumps(record.upstream_state, sort_keys=True),
            record.started_at.isoformat(),
            record.finished_at.isoformat(),
            json.dumps(record.extra, sort_keys=True),
        )
        self._gw.execute(sql, params)
```

You can wire this up to your actual table or keep your existing manifest store and just map it to this `PluginExecutionRecord` shape inside its read/write methods.

---

## 2. Graphs runtime: use core hashing & records

Now let’s make `graphs/runtime/manifest.py` use these primitives.

### 2.1 Replace graph-specific record type with PluginExecutionRecord

If you currently have something like:

```python
# graphs/runtime/manifest.py

@dataclass(frozen=True)
class GraphPluginExecutionRecord:
    plugin_name: str
    version: str
    repo: str
    commit: str
    scope_id: str | None
    status: str
    input_hash: str
    row_counts: dict[str, int]
    # ...
```

You can replace it with:

```python
from codeintel.core.plugins.execution.manifest import PluginExecutionRecord, PluginStatus
```

and just use `PluginExecutionRecord` directly everywhere in this module.

If you really want a graph-specific wrapper, you can do:

```python
@dataclass(frozen=True)
class GraphPluginExecutionRecord:
    core: PluginExecutionRecord

    @property
    def plugin_name(self) -> str:
        return self.core.plugin_name
    # etc.
```

but it’s usually easier to just use `PluginExecutionRecord`.

### 2.2 Shared helpers to compute scope_id, upstream_state, and hashes

In `graphs/runtime/manifest.py` (or a nearby graph runtime module), add:

```python
from codeintel.core.plugins.execution.manifest import (
    PluginExecutionRecord,
    PluginStatus,
    compute_input_hash,
    compute_options_hash,
)
from codeintel.core.plugins.types.metadata import CorePluginMetadata
```

Then define a small helper to compute a `scope_id` consistently:

```python
import hashlib
import json

def compute_scope_id_from_paths(paths: list[str] | None) -> str | None:
    """Compute a stable scope hash for a set of repo-relative paths."""
    if not paths:
        return None
    payload = {"paths": sorted(paths)}
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
```

And a helper to compute the `PluginExecutionRecord` for a single graph plugin run:

```python
from datetime import datetime, timezone
from typing import Mapping

def build_graph_execution_record(
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
    """Construct a PluginExecutionRecord for a graph plugin run."""

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

### 2.3 Use it for callgraph (and other graph plugins)

Wherever you record completion of a graph plugin run (e.g., in a graph orchestrator), change from graph-specific hashing to:

```python
from codeintel.core.plugins.execution.manifest import DuckDBManifestStore

def run_graph_plugin(
    *,
    plugin,
    meta: CorePluginMetadata,
    repo: str,
    commit: str,
    scope_paths: list[str] | None,
    variant: str | None,
    options: object | None,
    upstream_records: Mapping[str, PluginExecutionRecord],
    manifest_store: DuckDBManifestStore,
):
    # 1) Compute scope_id and upstream_state signatures
    scope_id = compute_scope_id_from_paths(scope_paths)

    upstream_state = {
        capability: rec.input_hash
        for capability, rec in upstream_records.items()
    }

    # 2) Execute plugin and measure time
    started_at = datetime.now(timezone.utc)
    try:
        result = plugin.execute(...)  # existing call
        status = PluginStatus.SUCCESS
    except Exception:
        # handle/log exception as you already do
        status = PluginStatus.FAILED
        result = None  # or a partial result

    finished_at = datetime.now(timezone.utc)

    row_counts = getattr(result, "row_counts", {}) if result is not None else {}

    # 3) Build and persist the execution record
    record = build_graph_execution_record(
        meta=meta,
        repo=repo,
        commit=commit,
        scope_id=scope_id,
        variant=variant,
        options=options,
        upstream_state=upstream_state,
        row_counts=row_counts,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
    )
    manifest_store.append_record(record)
```

Now **all graph plugins**, including callgraph, produce the exact same `PluginExecutionRecord` shape as analytics will.

---

## 3. Analytics runtime: use the same records & hashes

Now do the analogous wiring in `analytics/runtime/manifest.py`.

### 3.1 Use PluginExecutionRecord instead of analytics-specific record type

If you currently have something like:

```python
@dataclass(frozen=True)
class AnalyticsExecutionRecord:
    plugin_name: str
    repo: str
    commit: str
    # ...
```

replace it (or wrap it) with:

```python
from codeintel.core.plugins.execution.manifest import PluginExecutionRecord, PluginStatus
```

and use `PluginExecutionRecord` directly in any analytics manifest structures (`AnalyticsRunReport`, etc.).

### 3.2 Build records via the same helpers

Add imports:

```python
from codeintel.core.plugins.execution.manifest import (
    PluginExecutionRecord,
    PluginStatus,
    compute_input_hash,
    compute_options_hash,
)
from codeintel.core.plugins.types.metadata import CorePluginMetadata
```

Then a helper that parallels `build_graph_execution_record`, but in analytics context:

```python
from datetime import datetime, timezone
from typing import Mapping

def build_analytics_execution_record(
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

### 3.3 Include upstream_state for analytics.function_metrics

For `analytics.function_metrics`, you likely have upstream dependencies like:

* `core.goids` (from `graphs.goid_builder`)
* `graph.callgraph` (from `graphs.callgraph`)
* maybe coverage / history capabilities

In the analytics orchestrator (where you run `FunctionMetricsPlugin`), you can derive `upstream_state` by looking up their `PluginExecutionRecord`s from the manifest store:

```python
def run_function_metrics(
    *,
    plugin,
    meta: CorePluginMetadata,
    repo: str,
    commit: str,
    scope_id: str | None,
    variant: str | None,
    options: object | None,
    manifest_store: ManifestStore,
):
    # 1) Build upstream_state based on required capabilities
    upstream_state: dict[str, str] = {}

    # Example: if you know which plugin (or capability) provides core.goids:
    goids_record = manifest_store.load_last_record(
        plugin_name="graphs.goid_builder",  # or use capability->plugin mapping
        repo=repo,
        commit=commit,
        scope_id=scope_id,
        variant=variant,
    )
    if goids_record:
        upstream_state["core.goids"] = goids_record.input_hash

    callgraph_record = manifest_store.load_last_record(
        plugin_name="graphs.callgraph",
        repo=repo,
        commit=commit,
        scope_id=scope_id,
        variant=variant,
    )
    if callgraph_record:
        upstream_state["graph.callgraph"] = callgraph_record.input_hash

    # 2) Execute plugin
    started_at = datetime.now(timezone.utc)
    try:
        result = plugin.execute(...)  # current code path
        status = PluginStatus.SUCCESS
    except Exception:
        status = PluginStatus.FAILED
        result = None

    finished_at = datetime.now(timezone.utc)
    row_counts = getattr(result, "row_counts", {}) if result is not None else {}

    # 3) Build and persist execution record
    record = build_analytics_execution_record(
        meta=meta,
        repo=repo,
        commit=commit,
        scope_id=scope_id,
        variant=variant,
        options=options,
        upstream_state=upstream_state,
        row_counts=row_counts,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
    )
    manifest_store.append_record(record)
```

Now:

* `analytics.function_metrics` will include:

  * its own plugin info,
  * `options_hash` for `FunctionAnalyticsOptions`,
  * and `upstream_state` signatures for `core.goids` and `graph.callgraph`.

The exact way you map capabilities → plugin names can be improved later (e.g., use `CorePluginMetadata.requires`/`provides`), but the structure is ready.

---

## 4. Build & ingestion: deliberately unchanged (for now)

Per your original plan, you **don’t** need to touch:

* `build/hashing.py` (or however you hash targets today),
* ingestion manifest logic.

They can keep their existing record types and hash functions while graphs + analytics converge.

Later, when you introduce the unified engine, you can:

* either wrap their manifests into `PluginExecutionRecord`, or
* migrate them like we just did for graphs/analytics.

For now, the important thing is:

* graphs + analytics **share**:

  * `PluginExecutionRecord`,
  * `compute_options_hash`,
  * `compute_input_hash`,
  * `ManifestStore` concept.

---

## 5. Sanity checks & debugging wins

Once implemented, you can:

1. **Assert shared behavior in tests**:

   * A graph plugin and an analytics plugin that depend on the same upstream inputs produce records whose `upstream_state` keys align.
   * Running the same plugin twice with different options → different `options_hash` and `input_hash`.

2. **Debug re-runs** by inspecting execution records:

   * If a plugin re-ran unexpectedly, compare:

     * `input_hash` vs previous,
     * `options_hash`,
     * `upstream_state` per capability,
     * `version`.
   * You’ll have a uniform story for “what changed”.

3. **Prepare for the engine**:

   * The future `ExecutionEngine` can:

     * call `ManifestStore.load_last_record(...)` to decide skip vs run,
     * rely on `PluginExecutionRecord` across domains,
     * compute input/options hashes in exactly the same way the runtimes do today.

All of this happens without changing how plugins are actually executed yet — just how you **record** and **understand** what happened, which is exactly the kind of low‑risk, high‑leverage move you were aiming for.
