# Unified Data Abstraction Architecture

> **Purpose**: This document consolidates the three planned data abstraction changes (metadata, shared options, centralized hashing) into a single, comprehensive architecture specification. It serves as the canonical reference for the transition state between the current architecture and the long-term policy-driven orchestration target.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Metadata Model](#3-core-metadata-model)
4. [Shared Options System](#4-shared-options-system)
5. [Centralized Hashing & Manifests](#5-centralized-hashing--manifests)
6. [Integration: The Plugin Run Manifold](#6-integration-the-plugin-run-manifold)
7. [Domain-Specific Integration](#7-domain-specific-integration)
   - [Analytics Plugins](#71-analytics-plugins)
   - [Graphs Plugins](#72-graphs-plugins)
   - [Ingestion Plugins](#73-ingestion-plugins)
8. [Plugin Instance Specifications](#8-plugin-instance-specifications)
9. [Module Layout](#9-module-layout)
10. [Migration Path](#10-migration-path)

---

## 1. Executive Summary

This architecture defines three orthogonal but interconnected abstractions that together form a **unified data manifold** for plugin execution:

| Abstraction | Purpose | Primary Type | Location |
|-------------|---------|--------------|----------|
| **Metadata** | Plugin identity, capabilities, dataset I/O, execution semantics | `CorePluginMetadata` | `core/plugins/types/metadata.py` |
| **Options** | Policy-driven configuration with profile layering | `ConfigSource`, `ProfiledConfigSource`, `PluginOptionsResolver` | `core/plugins/execution/options.py` |
| **Hashing** | Deterministic input signatures for caching and manifest tracking | `PluginExecutionRecord`, `compute_input_hash`, `compute_options_hash` | `core/plugins/execution/manifest.py` |

These abstractions are designed to:
- Be **domain-agnostic**: Work identically for analytics, graphs, and ingestion plugins
- Be **incrementally adoptable**: Legacy plugins continue to work; new plugins gain benefits
- **Snap together**: Metadata drives options resolution, options contribute to hashes, hashes enable skip/rerun decisions

---

## 2. Architecture Overview

### 2.1 Conceptual Model

For a single plugin execution, the three abstractions provide complementary views:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PLUGIN EXECUTION MANIFOLD                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│   │ CorePluginMeta  │───▶│ PluginOptions   │───▶│ ExecutionRecord │        │
│   │                 │    │    Resolver     │    │                 │        │
│   │ • name          │    │                 │    │ • input_hash    │        │
│   │ • domain        │    │ • ConfigSource  │    │ • options_hash  │        │
│   │ • provides      │    │ • ProfiledCS    │    │ • upstream_state│        │
│   │ • requires      │    │ • BuildRunCfg   │    │ • row_counts    │        │
│   │ • options_model │    │                 │    │ • status        │        │
│   │ • produces_tbls │    │                 │    │                 │        │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│           │                       │                       │                 │
│           │   WHAT the plugin is  │  HOW it's configured │  WHAT happened  │
│           └───────────────────────┴───────────────────────┘                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
CLI/Caller                    Build Executor                    Plugin
    │                              │                               │
    │  --profile fast              │                               │
    │  --plugin-option P:K=V       │                               │
    ├─────────────────────────────▶│                               │
    │                              │                               │
    │                   ┌──────────┴──────────┐                    │
    │                   │  BuildRunConfig     │                    │
    │                   │  • profile: "fast"  │                    │
    │                   │  • base_options     │                    │
    │                   │  • profile_options  │                    │
    │                   │  • cli_options      │                    │
    │                   └──────────┬──────────┘                    │
    │                              │                               │
    │                   ┌──────────┴──────────┐                    │
    │                   │ ProfiledConfigSource│                    │
    │                   │   merge(base,       │                    │
    │                   │     profile, cli)   │                    │
    │                   └──────────┬──────────┘                    │
    │                              │                               │
    │                              │  prepare_plugin_run()         │
    │                              ├──────────────────────────────▶│
    │                              │                               │
    │                              │  PluginRunContext             │
    │                              │  • meta                       │
    │                              │  • options                    │
    │                              │  • upstream_state             │
    │                              │  • input_hash                 │
    │                              │◀──────────────────────────────│
    │                              │                               │
    │                              │  plugin.execute(ctx)          │
    │                              ├──────────────────────────────▶│
    │                              │                               │
    │                              │  PluginExecutionRecord        │
    │                              │◀──────────────────────────────│
    │                              │                               │
```

---

## 3. Core Metadata Model

### 3.1 Type Definition

The `CorePluginMetadata` dataclass is the **single source of truth** for plugin identity, capabilities, and execution semantics. It extends the existing `PluginMetadata` from `codeintel.core.plugins.types.protocol` with explicit domain classification and options model support.

```python
# File: core/plugins/types/metadata.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Type

from codeintel.core.plugins.types.protocol import (
    PluginKind,
    PluginStage,
    PluginSeverity,
    PluginIsolation,
    PluginResourceHints,
)


class PluginDomain(str, Enum):
    """Domain classification for plugins."""
    INGEST = "ingest"
    GRAPH = "graph"
    ANALYTICS = "analytics"
    EXPORT = "export"
    SERVING = "serving"
    CLI = "cli"


@dataclass(frozen=True)
class CorePluginMetadata:
    """Canonical plugin metadata used across all domains.

    This type is the single source of truth for:
    - Plugin identity (name, version, description)
    - Domain/kind/stage classification
    - Capabilities (provides, requires)
    - Dataset I/O (produces_tables, consumes_tables)
    - Execution semantics (incremental, scope-aware)
    - Options model reference

    Attributes
    ----------
    name
        Canonical identifier, e.g., "analytics.function_metrics".
    version
        Semantic version string.
    description
        Human-readable description.
    domain
        Plugin domain (ingest, graph, analytics, export, serving, cli).
    kind
        Plugin kind (builder, metric, validation, analytics).
    stage
        Processing stage grouping.
    provides
        Capability strings this plugin provides.
    requires
        Capability strings this plugin requires.
    produces_tables
        DuckDB table keys this plugin writes.
    consumes_tables
        DuckDB table keys this plugin reads.
    supports_incremental
        Whether incremental execution is supported.
    scope_aware
        Whether the plugin reacts to scope (paths/modules).
    options_model
        Type reference to the plugin's options dataclass/Pydantic model.
    resource_hints
        Runtime resource hints for scheduling.
    extra
        Domain-specific extensions (e.g., graph_kinds for graph plugins).
    """

    # Identity
    name: str
    version: str
    description: str

    # Domain classification
    domain: PluginDomain
    kind: PluginKind
    stage: PluginStage | None = None

    # Capabilities (cross-domain semantics)
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()

    # Dataset I/O
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

### 3.2 Relationship to Existing PluginMetadata

The existing `PluginMetadata` in `core/plugins/types/protocol.py` remains the runtime protocol type. `CorePluginMetadata` is a superset that adds:

- **`domain`**: Explicit classification (not present in PluginMetadata)
- **`options_model`**: Type reference for configuration (not present in PluginMetadata)
- **`extra`**: Domain-specific extension point

Domain-specific metadata types (like `GraphPluginMetadata`) become **thin wrappers**:

```python
# File: graphs/core/protocol.py

@dataclass(frozen=True)
class GraphPluginMetadata:
    """Graph-specific metadata wrapping CorePluginMetadata."""
    
    core: CorePluginMetadata
    produces_graph_kinds: tuple[str, ...] = ()
    requires_graph_kinds: tuple[str, ...] = ()
    
    @property
    def name(self) -> str:
        return self.core.name
    
    @property
    def provides(self) -> tuple[str, ...]:
        return self.core.provides
    
    # ... delegate other properties as needed
```

### 3.3 Plugin Registry Index

A global registry provides capability-to-provider lookups:

```python
# File: core/plugins/registry/capability_index.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from codeintel.core.plugins.types.metadata import CorePluginMetadata


@dataclass(frozen=True)
class PluginRegistryIndex:
    """Index for looking up plugins by name, capability, or output table."""
    
    by_name: dict[str, CorePluginMetadata]
    by_capability: dict[str, CorePluginMetadata]
    by_output_table: dict[str, CorePluginMetadata]


def build_registry_index(
    all_metadata: Iterable[CorePluginMetadata],
) -> PluginRegistryIndex:
    """Build a registry index from plugin metadata.

    Parameters
    ----------
    all_metadata
        Iterable of CorePluginMetadata instances.

    Returns
    -------
    PluginRegistryIndex
        Index with by_name, by_capability, and by_output_table lookups.
    """
    by_name: dict[str, CorePluginMetadata] = {}
    by_capability: dict[str, CorePluginMetadata] = {}
    by_output_table: dict[str, CorePluginMetadata] = {}

    for meta in all_metadata:
        by_name[meta.name] = meta
        for cap in meta.provides:
            by_capability[cap] = meta  # Last-writer wins
        for table in meta.produces_tables:
            by_output_table[table] = meta

    return PluginRegistryIndex(
        by_name=by_name,
        by_capability=by_capability,
        by_output_table=by_output_table,
    )
```

---

## 4. Shared Options System

### 4.1 ConfigSource Protocol

The `ConfigSource` protocol defines **where** plugin options come from:

```python
# File: core/plugins/execution/options.py

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class ConfigSource(Protocol):
    """Protocol for loading plugin configuration.

    Implementations can read from config files, environment variables,
    CLI arguments, or any combination thereof.
    """

    def get_plugin_options(self, plugin_name: str) -> Mapping[str, Any] | None:
        """Return raw option values for a plugin, or None if not configured."""
        ...


class EmptyConfigSource:
    """ConfigSource that always returns no options.

    Used as a safe default so plugins see valid model defaults.
    """

    def get_plugin_options(self, plugin_name: str) -> Mapping[str, Any] | None:
        return None
```

### 4.2 PluginOptionsResolver

The resolver constructs typed options objects from metadata + configuration:

```python
# File: core/plugins/execution/options.py (continued)

from dataclasses import replace
from typing import Type, TypeVar

from codeintel.core.plugins.types.metadata import CorePluginMetadata

T = TypeVar("T")


class PluginOptionsResolver:
    """Construct typed options objects for plugins.

    This is the central mechanism for:
    1. Fetching configuration from a ConfigSource
    2. Validating via the options model from metadata
    3. Merging with dynamic runtime overrides
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
            The options model class (dataclass or Pydantic model).
        dynamic_overrides
            Per-call runtime-only overrides (not from config).

        Returns
        -------
        T
            An instance of `model` populated from configuration.
        """
        raw = self._config_source.get_plugin_options(plugin_metadata.name) or {}

        # Base options from config
        base = model(**raw)

        if not dynamic_overrides:
            return base

        # Merge dynamic overrides
        if hasattr(base, "__dataclass_fields__"):
            return replace(base, **dynamic_overrides)
        if hasattr(base, "model_copy"):  # Pydantic v2
            return base.model_copy(update=dict(dynamic_overrides))
        if hasattr(base, "copy"):  # Pydantic v1
            return base.copy(update=dict(dynamic_overrides))

        # Fallback: attribute assignment
        for key, value in dynamic_overrides.items():
            setattr(base, key, value)
        return base
```

### 4.3 ProfiledConfigSource

The layered config source implements profile-based option resolution:

```python
# File: core/plugins/execution/options.py (continued)

from dataclasses import dataclass


def _merge_dicts(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Shallow merge two option dictionaries."""
    result: dict[str, Any] = {}
    if base:
        result.update(base)
    if override:
        result.update(override)
    return result


@dataclass(frozen=True)
class PluginConfigBundle:
    """Configuration data for all plugins for a single layer."""

    plugin_options: Mapping[str, Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "plugin_options", dict(self.plugin_options or {})
        )

    def get(self, plugin_name: str) -> Mapping[str, Any] | None:
        return self.plugin_options.get(plugin_name)


class ProfiledConfigSource(ConfigSource):
    """ConfigSource that merges base, profile, and CLI overrides.

    Resolution order for plugin_name P:
        base.plugins[P] → profile.plugins[P] → cli.plugins[P]
    
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
        self._base = base or PluginConfigBundle(plugin_options={})
        self._profile = profile or PluginConfigBundle(plugin_options={})
        self._cli = cli or PluginConfigBundle(plugin_options={})
        self._active_profile_name = active_profile_name

    def get_plugin_options(self, plugin_name: str) -> Mapping[str, Any] | None:
        base_raw = self._base.get(plugin_name)
        profile_raw = (
            self._profile.get(plugin_name) if self._active_profile_name else None
        )
        cli_raw = self._cli.get(plugin_name)

        merged = _merge_dicts(base_raw, profile_raw)
        merged = _merge_dicts(merged, cli_raw)

        return merged or None
```

### 4.4 BuildRunConfig

The build-layer configuration holder:

```python
# File: build/options.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from codeintel.core.plugins.execution.options import PluginConfigBundle


@dataclass
class BuildRunConfig:
    """Configuration for a single build/run.

    Captures the plugin option layers that will be merged by ProfiledConfigSource.
    """

    profile: str | None = None
    base_plugin_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    profiles_plugin_options: Mapping[str, Mapping[str, Mapping[str, Any]]] = field(
        default_factory=dict
    )
    cli_plugin_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def to_profiled_config_bundles(
        self,
    ) -> tuple[PluginConfigBundle, PluginConfigBundle, PluginConfigBundle]:
        """Convert to PluginConfigBundle instances for ProfiledConfigSource."""
        base_bundle = PluginConfigBundle(plugin_options=self.base_plugin_options)

        profile_options: Mapping[str, Mapping[str, Any]] = {}
        if self.profile:
            profile_options = self.profiles_plugin_options.get(self.profile, {})
        profile_bundle = PluginConfigBundle(plugin_options=profile_options)

        cli_bundle = PluginConfigBundle(plugin_options=self.cli_plugin_options)
        return base_bundle, profile_bundle, cli_bundle
```

### 4.5 Execution Profile Registry

Built-in profiles as typed, discoverable objects:

```python
# File: core/plugins/execution/profiles.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ExecutionProfile:
    """Typed execution profile definition."""
    
    name: str
    description: str
    plugin_options: Mapping[str, Mapping[str, Any]]


FAST_PROFILE = ExecutionProfile(
    name="fast",
    description="Good-enough signal, much faster execution.",
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
        "ingest.scip_python": {
            "incremental_only": True,
            "include_paths": ["src/"],
            "include_tests": False,
        },
    },
)

FULL_PROFILE = ExecutionProfile(
    name="full",
    description="Maximum signal, full repository analysis.",
    plugin_options={},  # Uses all base defaults
)

BUILTIN_PROFILES: dict[str, ExecutionProfile] = {
    FAST_PROFILE.name: FAST_PROFILE,
    FULL_PROFILE.name: FULL_PROFILE,
}
```

---

## 5. Centralized Hashing & Manifests

### 5.1 Core Hashing Functions

The existing infrastructure in `core/plugins/execution/manifest.py` provides the foundation. These functions are enhanced to support the unified architecture:

```python
# File: core/plugins/execution/manifest.py (enhanced)

import hashlib
import json
from typing import Mapping


def compute_options_hash(plugin_name: str, options: object | None) -> str | None:
    """Compute a stable hash for a plugin's options.

    Parameters
    ----------
    plugin_name
        Name of the plugin.
    options
        Options value to hash (dataclass, Pydantic model, or dict).

    Returns
    -------
    str | None
        SHA-256 hash (first 16 chars) or None if no options.
    """
    if options is None:
        return None

    # Serialize options to dict
    if hasattr(options, "model_dump"):  # Pydantic v2
        raw = options.model_dump()
    elif hasattr(options, "dict"):  # Pydantic v1
        raw = options.dict()
    elif hasattr(options, "__dict__"):  # dataclass
        raw = {
            k: v
            for k, v in vars(options).items()
            if not k.startswith("_")
        }
    else:
        raw = {"_repr": repr(options)}

    payload = {"plugin": plugin_name, "options": raw}
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def compute_input_hash(payload: Mapping[str, object]) -> str:
    """Compute a stable hash from a generic payload.

    The payload typically contains:
    - repo, commit
    - plugin_name, plugin_version
    - scope_id, variant
    - options_hash
    - upstream_state

    Parameters
    ----------
    payload
        Dictionary of values to hash.

    Returns
    -------
    str
        SHA-256 hash (first 16 chars).
    """
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]
```

### 5.2 Unified PluginExecutionRecord

The canonical execution record type used across all domains:

```python
# File: core/plugins/execution/manifest.py (continued)

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class PluginStatus(str, Enum):
    """Execution status for plugins."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"


@dataclass(frozen=True)
class PluginExecutionRecord:
    """Canonical record of a single plugin execution.

    This shape is used by graphs, analytics, and ingestion runtimes
    for consistent manifest tracking and skip/rerun decisions.

    Attributes
    ----------
    plugin_name
        Canonical name, e.g., "analytics.function_metrics".
    version
        Plugin version from metadata.
    repo
        Repository identifier ("owner/repo").
    commit
        Commit SHA.
    scope_id
        Hash of scope (paths/modules) or None for whole-repo.
    variant
        Profile/variant name ("fast", "full", etc.).
    status
        Execution status.
    input_hash
        Hash of all logical inputs.
    options_hash
        Hash of options_model fields only.
    row_counts
        Dataset name → row count.
    upstream_state
        Capability → upstream provider's input_hash.
    started_at
        Execution start timestamp.
    finished_at
        Execution end timestamp.
    extra
        Additional metadata.
    """

    # Identity
    plugin_name: str
    version: str
    repo: str
    commit: str
    scope_id: str | None
    variant: str | None

    # Status
    status: PluginStatus

    # Hashes
    input_hash: str
    options_hash: str | None

    # Outputs
    row_counts: dict[str, int]

    # Dependencies & timing
    upstream_state: dict[str, str]
    started_at: datetime
    finished_at: datetime

    # Extensions
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Execution duration in milliseconds."""
        return (self.finished_at - self.started_at).total_seconds() * 1000.0
```

### 5.3 ManifestStore Protocol

Abstract interface for manifest persistence:

```python
# File: core/plugins/execution/manifest.py (continued)

from typing import Protocol


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
        """Return the most recent record for this combination."""
        ...

    def append_record(self, record: PluginExecutionRecord) -> None:
        """Persist a new PluginExecutionRecord."""
        ...
```

### 5.4 Upstream State Resolution

Derive upstream_state from metadata.requires + registry:

```python
# File: core/plugins/execution/upstream.py

from __future__ import annotations

from codeintel.core.plugins.types.metadata import CorePluginMetadata
from codeintel.core.plugins.execution.manifest import ManifestStore
from codeintel.core.plugins.registry.capability_index import PluginRegistryIndex


def resolve_upstream_state(
    meta: CorePluginMetadata,
    registry: PluginRegistryIndex,
    repo: str,
    commit: str,
    scope_id: str | None,
    variant: str | None,
    manifest_store: ManifestStore,
) -> dict[str, str]:
    """Map required capabilities to provider input hashes.

    Parameters
    ----------
    meta
        Plugin metadata with requires capabilities.
    registry
        Registry index for capability → provider lookup.
    repo
        Repository identifier.
    commit
        Commit SHA.
    scope_id
        Scope hash or None.
    variant
        Profile/variant name.
    manifest_store
        Store for loading prior execution records.

    Returns
    -------
    dict[str, str]
        Capability → provider's input_hash.
    """
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

### 5.5 Input Signature Builder

Build canonical input signatures:

```python
# File: core/plugins/execution/signature.py

from __future__ import annotations

from typing import Mapping

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
    """Compute (options_hash, input_hash) for a plugin run.

    Parameters
    ----------
    meta
        Plugin metadata.
    repo
        Repository identifier.
    commit
        Commit SHA.
    scope_id
        Scope hash or None.
    variant
        Profile/variant name.
    options
        Resolved options instance.
    upstream_state
        Capability → upstream input_hash.

    Returns
    -------
    tuple[str | None, str]
        (options_hash, input_hash).
    """
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

---

## 6. Integration: The Plugin Run Manifold

### 6.1 PluginRunContext

The unified context that combines metadata, options, and hashing:

```python
# File: core/plugins/execution/run_context.py

from __future__ import annotations

from dataclasses import dataclass

from codeintel.core.plugins.types.metadata import CorePluginMetadata
from codeintel.core.plugins.execution.options import ConfigSource, PluginOptionsResolver
from codeintel.core.plugins.execution.manifest import ManifestStore
from codeintel.core.plugins.registry.capability_index import PluginRegistryIndex
from codeintel.core.plugins.execution.upstream import resolve_upstream_state
from codeintel.core.plugins.execution.signature import build_input_signature


@dataclass
class PluginRunContext:
    """All data needed to run a single plugin invocation.

    This is the "manifold" that ties metadata, options, and hashing together
    into a single coherent view of a plugin run.
    """

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
    """Prepare a plugin run with resolved options, upstream state, and hashes.

    This is the central integration point that:
    1. Resolves options via metadata.options_model + ConfigSource
    2. Derives upstream_state via metadata.requires + ManifestStore
    3. Computes options_hash + input_hash in a standard way

    Parameters
    ----------
    meta
        Plugin metadata.
    repo
        Repository identifier.
    commit
        Commit SHA.
    scope_id
        Scope hash or None.
    variant
        Profile/variant name.
    config_source
        Source for plugin configuration.
    manifest_store
        Store for loading prior execution records.
    registry_index
        Registry for capability lookups.
    options_resolver
        Optional pre-configured resolver.

    Returns
    -------
    PluginRunContext
        Fully prepared context for plugin execution.
    """
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

### 6.2 Skip/Rerun Decision

Using the manifold for cache invalidation:

```python
# File: core/plugins/execution/skip.py

from codeintel.core.plugins.execution.run_context import PluginRunContext
from codeintel.core.plugins.execution.manifest import ManifestStore, PluginStatus


def should_skip_plugin(
    run_ctx: PluginRunContext,
    manifest_store: ManifestStore,
) -> tuple[bool, str | None]:
    """Determine if a plugin can be skipped based on unchanged inputs.

    Parameters
    ----------
    run_ctx
        Prepared plugin run context.
    manifest_store
        Store for loading prior execution records.

    Returns
    -------
    tuple[bool, str | None]
        (should_skip, reason).
    """
    last = manifest_store.load_last_record(
        plugin_name=run_ctx.meta.name,
        repo=run_ctx.repo,
        commit=run_ctx.commit,
        scope_id=run_ctx.scope_id,
        variant=run_ctx.variant,
    )

    if last is None:
        return False, None

    if last.status != PluginStatus.SUCCESS:
        return False, "prior_run_not_successful"

    if last.input_hash != run_ctx.input_hash:
        return False, "input_hash_changed"

    return True, "unchanged"
```

---

## 7. Domain-Specific Integration

### 7.1 Analytics Plugins

#### 7.1.1 Enhanced FunctionAnalyticsOptions

```python
# File: analytics/functions/config.py (enhanced)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from codeintel.analytics.parsing.ast_cache import FunctionAst
    from codeintel.analytics.parsing.validation import FunctionValidationReporter


@dataclass(frozen=True)
class FunctionAnalyticsOptions:
    """Configuration and runtime context for function analytics.

    Config-driven fields (from profiles/config):
    - include_graph_metrics: Whether to compute graph-based metrics
    - include_coverage_metrics: Whether to include coverage signals
    - include_type_metrics: Whether to compute type annotation metrics
    - compute_centrality_metrics: Whether to compute expensive centrality

    Runtime-only fields (from execution context):
    - function_ast_map: Pre-parsed AST data
    - missing_function_goids: GOIDs that failed parsing
    - validation_reporter: Reporter for validation issues
    """

    # Config-driven fields (set via profiles/config)
    include_graph_metrics: bool = True
    include_coverage_metrics: bool = True
    include_type_metrics: bool = True
    compute_centrality_metrics: bool = True
    max_ast_depth_for_complexity: int | None = None
    sample_large_functions: bool = False
    large_function_loc_threshold: int = 1000

    # Runtime-only fields (set via dynamic_overrides)
    function_ast_map: dict[int, FunctionAst] | None = None
    missing_function_goids: set[int] = field(default_factory=set)
    validation_reporter: FunctionValidationReporter | None = None

    def get_ast_map(self) -> dict[int, FunctionAst]:
        """Return the function AST map, empty if not provided."""
        return self.function_ast_map or {}

    def has_ast_data(self) -> bool:
        """Check if AST data is available."""
        return self.function_ast_map is not None
```

#### 7.1.2 FunctionMetricsPlugin with CorePluginMetadata

```python
# File: analytics/plugins/functions/metrics.py (enhanced)

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.functions import (
    FunctionAnalyticsOptions,
    compute_function_metrics_and_types,
)
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import FunctionAnalyticsStepConfig
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.execution.options import PluginOptionsResolver

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


# Canonical metadata for function_metrics
FUNCTION_METRICS_METADATA = CorePluginMetadata(
    name="analytics.function_metrics",
    version="3.0.0",
    description="Compute function complexity and type coverage metrics.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="function",
    provides=(
        "analytics.function_metrics",
        "analytics.function_types",
    ),
    requires=(
        "core.goids",
        "graph.callgraph",  # Optional, gated by include_graph_metrics
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
    """Compute function complexity and type coverage metrics.

    Output Tables
    -------------
    - analytics.function_metrics: Complexity and size metrics
    - analytics.function_types: Type annotation data
    """

    plugin_name: ClassVar[str] = "function_metrics"
    plugin_version: ClassVar[str] = FUNCTION_METRICS_METADATA.version
    plugin_description: ClassVar[str] = FUNCTION_METRICS_METADATA.description

    # Canonical metadata
    metadata: ClassVar[CorePluginMetadata] = FUNCTION_METRICS_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute function metrics computation."""
        _ = self

        # Get runtime-only data
        function_ast_map = None
        missing_function_goids: set[int] = set()

        # Resolve options via the unified system
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
        result = compute_function_metrics_and_types(ctx.gateway, cfg, options=options)

        return TargetResult.succeeded(
            row_counts={
                "analytics.function_metrics": result.get("metrics_rows", 0),
                "analytics.function_types": result.get("types_rows", 0),
            }
        )
```

### 7.2 Graphs Plugins

#### 7.2.1 CallGraphOptions

```python
# File: graphs/plugins/builders/callgraph_options.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CallGraphOptions:
    """Configuration for call graph construction.

    Config-driven fields control graph scope and resolution behavior.
    """

    # Scope control
    scope_paths: list[str] | None = None  # None = whole repo
    include_external_calls: bool = False
    include_test_files: bool = True

    # Performance controls
    max_module_size_lines: int | None = None
    max_edges_per_function: int | None = None
    skip_stdlib_calls: bool = False

    # Parsing behavior
    use_ast_fallback: bool = True
```

#### 7.2.2 CallGraphPlugin with CorePluginMetadata

```python
# File: graphs/plugins/builders/callgraph.py (enhanced)

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config import CallGraphStepConfig
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.execution.options import PluginOptionsResolver
from codeintel.graphs.plugins.builders.callgraph_options import CallGraphOptions

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


# Canonical metadata for callgraph
CALLGRAPH_METADATA = CorePluginMetadata(
    name="graphs.callgraph",
    version="3.0.0",
    description="Build call graph nodes and edges.",
    domain=PluginDomain.GRAPH,
    kind="builder",
    stage="edges",
    provides=("graph.callgraph",),
    requires=("core.goids",),
    produces_tables=(
        "graph.call_graph_nodes",
        "graph.call_graph_edges",
    ),
    consumes_tables=(
        "core.goids",
        "core.modules",
    ),
    supports_incremental=False,
    scope_aware=True,
    options_model=CallGraphOptions,
    extra={
        "graph_kinds": ("callgraph",),
    },
)


class CallGraphPlugin(TargetPlugin):
    """Build call graph nodes and edges.

    Output Tables
    -------------
    - graph.call_graph_nodes: Call graph nodes
    - graph.call_graph_edges: Call graph edges
    """

    plugin_name: ClassVar[str] = "callgraph"
    plugin_version: ClassVar[str] = CALLGRAPH_METADATA.version
    plugin_description: ClassVar[str] = CALLGRAPH_METADATA.description

    # Canonical metadata
    metadata: ClassVar[CorePluginMetadata] = CALLGRAPH_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute call graph construction."""
        _ = self

        # Resolve options
        resolver = PluginOptionsResolver(config_source=ctx.config_source)
        options = resolver.get_options(self.metadata, CallGraphOptions)

        cfg = CallGraphStepConfig(snapshot=ctx.snapshot)
        gateway, repo, commit = ctx.gateway, cfg.repo, cfg.commit

        # Load function index
        function_index = load_function_index(gateway, repo=repo, commit=commit)
        all_paths = list(function_index.paths())

        # Apply scope_paths filter
        paths = self._filter_paths_by_scope(all_paths, options)

        if not paths:
            return TargetResult.succeeded(
                row_counts={"graph.call_graph_nodes": 0, "graph.call_graph_edges": 0}
            )

        # ... rest of existing implementation ...

    @staticmethod
    def _filter_paths_by_scope(
        paths: list[str],
        options: CallGraphOptions,
    ) -> list[str]:
        """Filter paths by scope_paths and test file settings."""
        filtered = list(paths)

        if options.scope_paths:
            prefixes = tuple(options.scope_paths)
            filtered = [p for p in filtered if p.startswith(prefixes)]

        if not options.include_test_files:
            filtered = [
                p for p in filtered
                if not any(seg in p for seg in ("test_", "_test", "tests/"))
            ]

        return filtered
```

### 7.3 Ingestion Plugins

#### 7.3.1 IngestScipOptions

```python
# File: ingestion/plugins/scip_options.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class IngestScipOptions:
    """Configuration for SCIP-based indexing.

    Config-driven fields control indexing scope and behavior.
    """

    # Incremental behavior
    incremental_only: bool = True

    # Scope control
    include_paths: Sequence[str] | None = None  # None = whole repo
    exclude_paths: Sequence[str] | None = None
    include_tests: bool = False

    # Resource limits
    max_workers: int | None = None
    max_indexed_files: int | None = None

    # Error handling
    allow_partial_failures: bool = True
```

#### 7.3.2 ScipIngestPlugin with CorePluginMetadata

```python
# File: ingestion/plugins/scip_plugin.py (enhanced)

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.build.errors import ToolNotAvailableError
from codeintel.build.plugin import TargetPlugin
from codeintel.build.result import TargetResult
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.execution.options import PluginOptionsResolver
from codeintel.ingestion.plugins.scip_options import IngestScipOptions

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


# Canonical metadata for scip_ingest
SCIP_PYTHON_METADATA = CorePluginMetadata(
    name="ingest.scip_python",
    version="3.0.0",
    description="Run scip-python to index Python modules.",
    domain=PluginDomain.INGEST,
    kind="builder",
    stage="pipeline_ingestion",
    provides=(
        "ingest.scip_index",
        "core.symbols",
    ),
    requires=(
        "ingest.modules",
    ),
    produces_tables=(
        "core.scip_symbols",
        "core.goid_crosswalk",
    ),
    consumes_tables=(
        "core.modules",
    ),
    supports_incremental=True,
    scope_aware=False,
    options_model=IngestScipOptions,
)


class ScipIngestPlugin(TargetPlugin):
    """Run scip-python and persist symbols and GOID crosswalk.

    Output Tables
    -------------
    - core.scip_symbols: Symbol table
    - core.goid_crosswalk: GOID crosswalk
    """

    plugin_name: ClassVar[str] = "scip_ingest"
    plugin_version: ClassVar[str] = SCIP_PYTHON_METADATA.version
    plugin_description: ClassVar[str] = SCIP_PYTHON_METADATA.description

    # Canonical metadata
    metadata: ClassVar[CorePluginMetadata] = SCIP_PYTHON_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute SCIP indexing."""
        _ = self

        if ctx.resources.scip_indexer is None:
            raise ToolNotAvailableError(target=self.plugin_name, tool="scip-python")

        # Resolve options
        resolver = PluginOptionsResolver(config_source=ctx.config_source)
        options = resolver.get_options(self.metadata, IngestScipOptions)

        # Get module paths with filtering
        paths = get_module_paths(ctx)
        paths = self._filter_paths(paths, options)

        if options.incremental_only:
            paths = self._filter_to_changed(ctx, paths)

        # ... rest of existing implementation ...

    def _filter_paths(
        self,
        paths: list[str],
        options: IngestScipOptions,
    ) -> list[str]:
        """Apply include/exclude/test filters."""
        filtered = list(paths)

        if options.include_paths:
            prefixes = tuple(options.include_paths)
            filtered = [p for p in filtered if p.startswith(prefixes)]

        if options.exclude_paths:
            excl = tuple(options.exclude_paths)
            filtered = [p for p in filtered if not p.startswith(excl)]

        if not options.include_tests:
            filtered = [
                p for p in filtered
                if not any(seg in p for seg in ("test_", "_test", "tests/"))
            ]

        if options.max_indexed_files:
            filtered = filtered[: options.max_indexed_files]

        return filtered
```

---

## 8. Plugin Instance Specifications

### 8.1 Complete Plugin Metadata Registry

```python
# File: core/plugins/registry/all_metadata.py

from __future__ import annotations

from codeintel.analytics.plugins.functions.metrics import FUNCTION_METRICS_METADATA
from codeintel.analytics.plugins.hotspots.build import HOTSPOTS_METADATA
from codeintel.analytics.plugins.coverage.functions import COVERAGE_FUNCTIONS_METADATA
from codeintel.graphs.plugins.builders.callgraph import CALLGRAPH_METADATA
from codeintel.graphs.plugins.builders.import_graph import IMPORT_GRAPH_METADATA
from codeintel.graphs.plugins.builders.goid import GOID_BUILDER_METADATA
from codeintel.ingestion.plugins.scip_plugin import SCIP_PYTHON_METADATA
from codeintel.ingestion.plugins.repo_scan import REPO_SCAN_METADATA
from codeintel.core.plugins.types.metadata import CorePluginMetadata
from codeintel.core.plugins.registry.capability_index import (
    PluginRegistryIndex,
    build_registry_index,
)


# All metadata instances
ALL_PLUGIN_METADATA: tuple[CorePluginMetadata, ...] = (
    # Ingestion
    REPO_SCAN_METADATA,
    SCIP_PYTHON_METADATA,
    # Graphs
    GOID_BUILDER_METADATA,
    CALLGRAPH_METADATA,
    IMPORT_GRAPH_METADATA,
    # Analytics
    FUNCTION_METRICS_METADATA,
    COVERAGE_FUNCTIONS_METADATA,
    HOTSPOTS_METADATA,
    # ... add more as migrated
)


# Global registry index (built once, cached)
PLUGIN_REGISTRY_INDEX: PluginRegistryIndex = build_registry_index(ALL_PLUGIN_METADATA)
```

### 8.2 Profile Configurations (Full Specification)

```yaml
# Conceptual config schema

plugins:
  # Analytics plugins
  analytics.function_metrics:
    include_graph_metrics: true
    include_coverage_metrics: true
    include_type_metrics: true
    compute_centrality_metrics: true
    max_ast_depth_for_complexity: null
    sample_large_functions: false
    large_function_loc_threshold: 1000

  # Graph plugins
  graphs.callgraph:
    scope_paths: null
    include_external_calls: true
    include_test_files: true
    max_module_size_lines: null
    max_edges_per_function: null
    skip_stdlib_calls: false

  # Ingestion plugins
  ingest.scip_python:
    incremental_only: false
    include_paths: null
    exclude_paths: null
    include_tests: true
    max_workers: null
    max_indexed_files: null

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
        scope_paths: ["src/"]
        include_test_files: false
        include_external_calls: false
        skip_stdlib_calls: true
        max_module_size_lines: 2000
        max_edges_per_function: 200

      ingest.scip_python:
        incremental_only: true
        include_paths: ["src/"]
        include_tests: false
        max_indexed_files: 20000

  full:
    plugins:
      # Empty overrides = use base defaults
      analytics.function_metrics: {}
      graphs.callgraph: {}
      ingest.scip_python: {}

  ci:
    plugins:
      analytics.function_metrics:
        include_graph_metrics: true
        include_coverage_metrics: true
        compute_centrality_metrics: false

      graphs.callgraph:
        include_test_files: true
        max_module_size_lines: 3000

      ingest.scip_python:
        incremental_only: false
        include_tests: true
```

---

## 9. Module Layout

### 9.1 Core Modules

```
src/codeintel/core/plugins/
├── types/
│   ├── __init__.py
│   ├── metadata.py          # CorePluginMetadata, PluginDomain
│   ├── protocol.py          # Existing PluginMetadata, PluginProtocol
│   ├── result.py            # PluginResult, PluginExecutionRecord
│   └── report.py            # Report types
├── execution/
│   ├── __init__.py
│   ├── context.py           # PluginExecutionContext
│   ├── options.py           # ConfigSource, PluginOptionsResolver, ProfiledConfigSource
│   ├── manifest.py          # Hashing functions, ManifestStore
│   ├── upstream.py          # resolve_upstream_state
│   ├── signature.py         # build_input_signature
│   ├── run_context.py       # PluginRunContext, prepare_plugin_run
│   ├── skip.py              # should_skip_plugin
│   ├── profiles.py          # ExecutionProfile, BUILTIN_PROFILES
│   ├── policy.py            # BaseExecutionPolicy
│   └── executor.py          # Plugin executor
└── registry/
    ├── __init__.py
    ├── base.py              # Base registry
    ├── capability_index.py  # PluginRegistryIndex, build_registry_index
    ├── all_metadata.py      # ALL_PLUGIN_METADATA, PLUGIN_REGISTRY_INDEX
    └── sorting.py           # Dependency sorting
```

### 9.2 Domain Modules

```
# Analytics
src/codeintel/analytics/
├── functions/
│   ├── config.py            # FunctionAnalyticsOptions (enhanced)
│   ├── metrics.py           # compute_function_metrics_and_types
│   └── __init__.py
└── plugins/
    ├── functions/
    │   ├── metrics.py       # FunctionMetricsPlugin, FUNCTION_METRICS_METADATA
    │   └── __init__.py
    ├── registration.py      # ALL_PLUGINS
    └── __init__.py

# Graphs
src/codeintel/graphs/
├── core/
│   ├── protocol.py          # GraphPluginMetadata (thin wrapper)
│   └── __init__.py
└── plugins/
    ├── builders/
    │   ├── callgraph.py     # CallGraphPlugin, CALLGRAPH_METADATA
    │   ├── callgraph_options.py  # CallGraphOptions
    │   └── __init__.py
    └── __init__.py

# Ingestion
src/codeintel/ingestion/
└── plugins/
    ├── scip_plugin.py       # ScipIngestPlugin, SCIP_PYTHON_METADATA
    ├── scip_options.py      # IngestScipOptions
    └── __init__.py

# Build
src/codeintel/build/
├── options.py               # BuildRunConfig
└── executor.py              # Uses ProfiledConfigSource
```

---

## 10. Migration Path

### 10.1 Phase 1: Core Infrastructure

1. **Add `CorePluginMetadata`** to `core/plugins/types/metadata.py`
2. **Add options infrastructure** to `core/plugins/execution/options.py`:
   - `ConfigSource`, `EmptyConfigSource`
   - `PluginOptionsResolver`
   - `PluginConfigBundle`, `ProfiledConfigSource`
3. **Enhance manifest infrastructure** in `core/plugins/execution/manifest.py`:
   - Ensure `compute_options_hash` and `compute_input_hash` are generic
   - Add `PluginStatus` enum if not present
4. **Add registry index** to `core/plugins/registry/capability_index.py`

### 10.2 Phase 2: Spine Plugin Migration

Migrate three representative plugins:

1. **`analytics.function_metrics`**:
   - Enhance `FunctionAnalyticsOptions` with config-driven fields
   - Add `FUNCTION_METRICS_METADATA`
   - Update `FunctionMetricsPlugin` to use `PluginOptionsResolver`

2. **`graphs.callgraph`**:
   - Add `CallGraphOptions`
   - Add `CALLGRAPH_METADATA`
   - Update `CallGraphPlugin` to use `PluginOptionsResolver`

3. **`ingest.scip_python`**:
   - Add `IngestScipOptions`
   - Add `SCIP_PYTHON_METADATA`
   - Update `ScipIngestPlugin` to use `PluginOptionsResolver`

### 10.3 Phase 3: Build Integration

1. **Add `BuildRunConfig`** to `build/options.py`
2. **Update build executor** to:
   - Parse profile selection from CLI
   - Construct `ProfiledConfigSource`
   - Pass `config_source` in `TargetExecutionContext`
3. **Add profile definitions** in `core/plugins/execution/profiles.py`

### 10.4 Phase 4: Manifest Integration

1. **Add `resolve_upstream_state`** to `core/plugins/execution/upstream.py`
2. **Add `build_input_signature`** to `core/plugins/execution/signature.py`
3. **Add `prepare_plugin_run`** to `core/plugins/execution/run_context.py`
4. **Wire manifests into runtimes**:
   - Analytics runtime records `PluginExecutionRecord`
   - Graphs runtime records `PluginExecutionRecord`
   - Both use shared hashing functions

### 10.5 Phase 5: Incremental Rollout

For each additional plugin:
1. Define options dataclass if needed
2. Create `CorePluginMetadata` constant
3. Add `metadata: ClassVar` to plugin class
4. Add to `ALL_PLUGIN_METADATA`
5. Update execution to use `PluginOptionsResolver` (if configurable)

---

## Appendix A: Profile Impact Matrix

| Profile | Analytics Impact | Graph Impact | Ingestion Impact |
|---------|------------------|--------------|------------------|
| **full** | All metrics, all graphs, full coverage | Complete callgraph with external calls | Full repo index |
| **fast** | Skip graph/coverage metrics | Scope-limited, no tests, no stdlib | Incremental, scope-limited |
| **ci** | Graph metrics, no centrality | Include tests, moderate limits | Full index |

## Appendix B: Capability Namespace

| Prefix | Domain | Examples |
|--------|--------|----------|
| `core.*` | Cross-domain foundational | `core.goids`, `core.modules`, `core.symbols` |
| `ingest.*` | Ingestion | `ingest.scip_index`, `ingest.modules` |
| `graph.*` | Graphs | `graph.callgraph`, `graph.import_graph` |
| `analytics.*` | Analytics | `analytics.function_metrics`, `analytics.hotspots` |

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **Capability** | A named contract (e.g., `graph.callgraph`) that plugins provide/require |
| **Data Manifold** | The unified abstraction combining metadata, options, and execution records |
| **Options Model** | A dataclass/Pydantic model defining a plugin's configuration schema |
| **Profile** | A named configuration preset (e.g., `fast`, `full`) |
| **Upstream State** | Map of required capabilities to their provider's input hashes |
| **Variant** | The active profile name stored in execution records |
