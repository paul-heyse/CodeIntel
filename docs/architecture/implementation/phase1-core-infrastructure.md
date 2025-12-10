# Phase 1: Core Infrastructure Implementation Plan

> **Scope**: Add core data abstraction infrastructure without changing plugin behavior
> **Duration**: 1-2 days
> **Risk Level**: Low (additive changes only)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Task 1: CorePluginMetadata Type](#3-task-1-corepluginmetadata-type)
4. [Task 2: Options Infrastructure](#4-task-2-options-infrastructure)
5. [Task 3: Enhanced Manifest Infrastructure](#5-task-3-enhanced-manifest-infrastructure)
6. [Task 4: Capability Registry Index](#6-task-4-capability-registry-index)
7. [Task 5: Export Updates](#7-task-5-export-updates)
8. [Verification](#8-verification)
9. [Rollback Plan](#9-rollback-plan)

---

## 1. Overview

Phase 1 establishes the foundational types and utilities that subsequent phases build upon. All changes are **additive** - no existing code is modified in ways that could break behavior.

### Deliverables

| File | Action | Purpose |
|------|--------|---------|
| `core/plugins/types/metadata.py` | **NEW** | `CorePluginMetadata`, `PluginDomain` enum |
| `core/plugins/execution/options.py` | **NEW** | `ConfigSource`, `PluginOptionsResolver`, `ProfiledConfigSource` |
| `core/plugins/execution/manifest.py` | **ENHANCE** | Add `ManifestStore` protocol, upstream state utilities |
| `core/plugins/registry/capability_index.py` | **NEW** | `PluginRegistryIndex`, `build_registry_index` |
| `core/plugins/types/__init__.py` | **UPDATE** | Re-export new types |
| `core/plugins/execution/__init__.py` | **UPDATE** | Re-export new types |
| `core/plugins/registry/__init__.py` | **UPDATE** | Re-export new types |

---

## 2. Prerequisites

Before starting, verify the development environment:

```bash
# Ensure environment is bootstrapped
scripts/bootstrap.sh

# Run quality checks to establish baseline
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest tests/core/plugins/ -q
```

---

## 3. Task 1: CorePluginMetadata Type

### 3.1 Create `core/plugins/types/metadata.py`

This new module defines the canonical metadata type that extends the existing `PluginMetadata` with domain classification and options model support.

```python
# File: src/codeintel/core/plugins/types/metadata.py
"""Core plugin metadata types for unified data abstraction.

This module defines `CorePluginMetadata`, the canonical metadata type that
extends the existing `PluginMetadata` with explicit domain classification
and options model support. It serves as the single source of truth for
plugin identity, capabilities, and execution semantics.

Architecture
------------
CorePluginMetadata is designed to:
- Coexist with the existing PluginMetadata (which it imports)
- Add domain classification (PluginDomain enum)
- Add options_model reference for typed configuration
- Provide an extension point (extra) for domain-specific fields

Usage
-----
Plugins declare a CorePluginMetadata constant:

>>> FUNCTION_METRICS_METADATA = CorePluginMetadata(
...     name="analytics.function_metrics",
...     version="3.0.0",
...     description="Compute function complexity metrics.",
...     domain=PluginDomain.ANALYTICS,
...     kind="metric",
...     stage="function",
...     provides=("analytics.function_metrics",),
...     requires=("core.goids",),
...     produces_tables=("analytics.function_metrics",),
...     options_model=FunctionAnalyticsOptions,
... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


class PluginDomain(str, Enum):
    """Domain classification for plugins.

    Each plugin belongs to exactly one domain, which determines its
    execution context and runtime behavior.

    Values
    ------
    INGEST
        Ingestion plugins that process raw repository data.
    GRAPH
        Graph plugins that build or analyze code graphs.
    ANALYTICS
        Analytics plugins that compute metrics and insights.
    EXPORT
        Export plugins that produce output artifacts.
    SERVING
        Serving plugins that provide runtime APIs.
    CLI
        CLI plugins that provide command-line tools.
    """

    INGEST = "ingest"
    GRAPH = "graph"
    ANALYTICS = "analytics"
    EXPORT = "export"
    SERVING = "serving"
    CLI = "cli"


@dataclass(frozen=True)
class CorePluginMetadata:
    """Canonical plugin metadata for unified data abstraction.

    This type is the single source of truth for:
    - Plugin identity (name, version, description)
    - Domain/kind/stage classification
    - Capabilities (provides, requires)
    - Dataset I/O (produces_tables, consumes_tables)
    - Execution semantics (incremental, scope-aware)
    - Options model reference

    The existing `PluginMetadata` from `protocol.py` remains the runtime
    protocol type. `CorePluginMetadata` extends it conceptually with:
    - `domain`: Explicit plugin domain classification
    - `options_model`: Type reference for configuration schema
    - `extra`: Domain-specific extension point

    Attributes
    ----------
    name
        Canonical identifier, e.g., "analytics.function_metrics".
        Must be unique across all registered plugins.
    version
        Semantic version string (e.g., "3.0.0").
    description
        Human-readable description of what the plugin does.
    domain
        Plugin domain (ingest, graph, analytics, export, serving, cli).
    kind
        Plugin kind (builder, metric, validation, analytics, tool).
    stage
        Processing stage grouping for execution ordering.
    provides
        Capability strings this plugin provides to others.
    requires
        Capability strings this plugin requires from others.
    produces_tables
        DuckDB table keys this plugin writes to.
    consumes_tables
        DuckDB table keys this plugin reads from.
    supports_incremental
        Whether the plugin supports incremental execution.
    scope_aware
        Whether the plugin reacts to scope (paths/modules).
    options_model
        Type reference to the plugin's options dataclass/Pydantic model.
        When set, the PluginOptionsResolver can construct typed options.
    resource_hints
        Runtime resource hints for scheduling (memory, CPU, etc.).
    extra
        Domain-specific extensions (e.g., graph_kinds for graph plugins).

    Examples
    --------
    Creating metadata for an analytics plugin:

    >>> from codeintel.core.plugins.types.metadata import (
    ...     CorePluginMetadata,
    ...     PluginDomain,
    ... )
    >>> metadata = CorePluginMetadata(
    ...     name="analytics.example",
    ...     version="1.0.0",
    ...     description="Example analytics plugin.",
    ...     domain=PluginDomain.ANALYTICS,
    ...     kind="metric",
    ...     stage="function",
    ...     provides=("analytics.example",),
    ...     requires=("core.goids",),
    ...     produces_tables=("analytics.example_metrics",),
    ... )
    >>> metadata.name
    'analytics.example'
    >>> metadata.domain
    <PluginDomain.ANALYTICS: 'analytics'>
    """

    # Identity
    name: str
    version: str
    description: str

    # Domain classification
    domain: PluginDomain
    kind: str  # PluginKind from protocol.py
    stage: str | None = None  # PluginStage from protocol.py

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
    options_model: type[Any] | None = None
    resource_hints: Mapping[str, Any] = field(default_factory=dict)

    # Domain-specific extras
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        if not self.name:
            message = "Plugin name cannot be empty"
            raise ValueError(message)
        if not self.version:
            message = "Plugin version cannot be empty"
            raise ValueError(message)

    @property
    def has_options(self) -> bool:
        """Check if this plugin has a typed options model.

        Returns
        -------
        bool
            True if options_model is set.
        """
        return self.options_model is not None

    @property
    def capability_names(self) -> tuple[str, ...]:
        """Return all capability names (provides + requires).

        Returns
        -------
        tuple[str, ...]
            Combined capabilities.
        """
        return (*self.provides, *self.requires)

    @property
    def all_tables(self) -> tuple[str, ...]:
        """Return all table names (produces + consumes).

        Returns
        -------
        tuple[str, ...]
            Combined tables.
        """
        return (*self.produces_tables, *self.consumes_tables)


__all__ = [
    "CorePluginMetadata",
    "PluginDomain",
]
```

### 3.2 Test File: `tests/core/plugins/test_metadata.py`

```python
# File: tests/core/plugins/test_metadata.py
"""Tests for CorePluginMetadata."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from codeintel.core.plugins.types.metadata import (
    CorePluginMetadata,
    PluginDomain,
)


class TestPluginDomain:
    """Tests for PluginDomain enum."""

    def test_domain_values(self) -> None:
        """Verify all domain values are strings."""
        assert PluginDomain.INGEST.value == "ingest"
        assert PluginDomain.GRAPH.value == "graph"
        assert PluginDomain.ANALYTICS.value == "analytics"
        assert PluginDomain.EXPORT.value == "export"
        assert PluginDomain.SERVING.value == "serving"
        assert PluginDomain.CLI.value == "cli"

    def test_domain_is_string_enum(self) -> None:
        """Verify domains can be used as strings."""
        domain = PluginDomain.ANALYTICS
        assert f"domain={domain}" == "domain=analytics"


class TestCorePluginMetadata:
    """Tests for CorePluginMetadata."""

    def test_minimal_metadata(self) -> None:
        """Verify minimal valid metadata can be created."""
        meta = CorePluginMetadata(
            name="test.plugin",
            version="1.0.0",
            description="Test plugin.",
            domain=PluginDomain.ANALYTICS,
            kind="metric",
        )
        assert meta.name == "test.plugin"
        assert meta.version == "1.0.0"
        assert meta.domain == PluginDomain.ANALYTICS
        assert meta.kind == "metric"
        assert meta.stage is None
        assert meta.provides == ()
        assert meta.requires == ()

    def test_full_metadata(self) -> None:
        """Verify full metadata with all fields."""
        @dataclass
        class TestOptions:
            threshold: float = 0.5

        meta = CorePluginMetadata(
            name="analytics.function_metrics",
            version="3.0.0",
            description="Compute function metrics.",
            domain=PluginDomain.ANALYTICS,
            kind="metric",
            stage="function",
            provides=("analytics.function_metrics", "analytics.function_types"),
            requires=("core.goids",),
            produces_tables=("analytics.function_metrics",),
            consumes_tables=("core.goids",),
            supports_incremental=False,
            scope_aware=False,
            options_model=TestOptions,
            resource_hints={"max_memory_mb": 512},
            extra={"custom_key": "custom_value"},
        )
        assert meta.name == "analytics.function_metrics"
        assert meta.provides == ("analytics.function_metrics", "analytics.function_types")
        assert meta.requires == ("core.goids",)
        assert meta.options_model is TestOptions
        assert meta.has_options is True
        assert meta.extra["custom_key"] == "custom_value"

    def test_has_options_without_model(self) -> None:
        """Verify has_options returns False when no model set."""
        meta = CorePluginMetadata(
            name="test.plugin",
            version="1.0.0",
            description="Test.",
            domain=PluginDomain.GRAPH,
            kind="builder",
        )
        assert meta.has_options is False

    def test_capability_names_property(self) -> None:
        """Verify capability_names combines provides and requires."""
        meta = CorePluginMetadata(
            name="test.plugin",
            version="1.0.0",
            description="Test.",
            domain=PluginDomain.ANALYTICS,
            kind="metric",
            provides=("cap.a", "cap.b"),
            requires=("cap.c",),
        )
        assert meta.capability_names == ("cap.a", "cap.b", "cap.c")

    def test_all_tables_property(self) -> None:
        """Verify all_tables combines produces and consumes."""
        meta = CorePluginMetadata(
            name="test.plugin",
            version="1.0.0",
            description="Test.",
            domain=PluginDomain.ANALYTICS,
            kind="metric",
            produces_tables=("table.a",),
            consumes_tables=("table.b", "table.c"),
        )
        assert meta.all_tables == ("table.a", "table.b", "table.c")

    def test_empty_name_raises(self) -> None:
        """Verify empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            CorePluginMetadata(
                name="",
                version="1.0.0",
                description="Test.",
                domain=PluginDomain.ANALYTICS,
                kind="metric",
            )

    def test_empty_version_raises(self) -> None:
        """Verify empty version raises ValueError."""
        with pytest.raises(ValueError, match="version cannot be empty"):
            CorePluginMetadata(
                name="test.plugin",
                version="",
                description="Test.",
                domain=PluginDomain.ANALYTICS,
                kind="metric",
            )

    def test_metadata_is_frozen(self) -> None:
        """Verify metadata is immutable."""
        meta = CorePluginMetadata(
            name="test.plugin",
            version="1.0.0",
            description="Test.",
            domain=PluginDomain.ANALYTICS,
            kind="metric",
        )
        with pytest.raises(AttributeError):
            meta.name = "modified"  # type: ignore[misc]
```

---

## 4. Task 2: Options Infrastructure

### 4.1 Create `core/plugins/execution/options.py`

This module provides the shared options system for policy-driven configuration.

```python
# File: src/codeintel/core/plugins/execution/options.py
"""Shared options infrastructure for plugin configuration.

This module provides the unified options system that enables:
- Policy-driven configuration via profiles
- Type-safe options resolution via metadata
- Layered configuration merging (base → profile → CLI)

Architecture
------------
The options system consists of:

1. ConfigSource Protocol
   - Abstract interface for where options come from
   - EmptyConfigSource provides safe defaults

2. PluginOptionsResolver
   - Constructs typed options from metadata + ConfigSource
   - Supports dynamic runtime overrides

3. ProfiledConfigSource
   - Implements layered configuration merging
   - Supports base, profile, and CLI override layers

4. PluginConfigBundle
   - Data holder for a single configuration layer

Usage
-----
Basic usage with empty config (preserves defaults):

>>> from codeintel.core.plugins.execution.options import (
...     PluginOptionsResolver,
...     EmptyConfigSource,
... )
>>> resolver = PluginOptionsResolver(EmptyConfigSource())

With profiled configuration:

>>> from codeintel.core.plugins.execution.options import (
...     ProfiledConfigSource,
...     PluginConfigBundle,
... )
>>> config_source = ProfiledConfigSource(
...     base=PluginConfigBundle(plugin_options={"plugin.name": {"key": "base"}}),
...     profile=PluginConfigBundle(plugin_options={"plugin.name": {"key": "fast"}}),
...     active_profile_name="fast",
... )
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.core.plugins.types.metadata import CorePluginMetadata

T = TypeVar("T")


# =============================================================================
# ConfigSource Protocol
# =============================================================================


@runtime_checkable
class ConfigSource(Protocol):
    """Protocol for loading plugin configuration.

    Implementations can read from:
    - Static config files (YAML, TOML, etc.)
    - Environment variables
    - CLI arguments
    - Snapshot-specific settings
    - Any combination of these

    The key idea: given a canonical plugin name, return a dict of option
    values that can be passed to the plugin's options_model.

    Examples
    --------
    Implementing a custom ConfigSource:

    >>> class EnvConfigSource:
    ...     def get_plugin_options(self, plugin_name: str) -> dict[str, Any] | None:
    ...         import os
    ...         prefix = f"PLUGIN_{plugin_name.upper().replace('.', '_')}_"
    ...         options = {}
    ...         for key, value in os.environ.items():
    ...             if key.startswith(prefix):
    ...                 opt_key = key[len(prefix):].lower()
    ...                 options[opt_key] = value
    ...         return options or None
    """

    def get_plugin_options(self, plugin_name: str) -> Mapping[str, Any] | None:
        """Return raw option values for a plugin, or None if not configured.

        Parameters
        ----------
        plugin_name
            Canonical plugin name (e.g., "analytics.function_metrics").

        Returns
        -------
        Mapping[str, Any] | None
            Option key-value pairs, or None if no configuration exists.
        """
        ...


class EmptyConfigSource:
    """ConfigSource that always returns no options.

    Useful as a default while wiring up the system. It ensures that
    plugins still see valid option objects with default values from
    their options model.

    Examples
    --------
    >>> source = EmptyConfigSource()
    >>> source.get_plugin_options("any.plugin") is None
    True
    """

    def get_plugin_options(self, plugin_name: str) -> Mapping[str, Any] | None:
        """Return None for any plugin.

        Parameters
        ----------
        plugin_name
            Plugin name (ignored).

        Returns
        -------
        None
            Always returns None.
        """
        _ = plugin_name
        return None


# =============================================================================
# PluginOptionsResolver
# =============================================================================


class PluginOptionsResolver:
    """Construct typed options objects for plugins.

    This is the central mechanism for:
    1. Fetching configuration from a ConfigSource
    2. Validating via the options model from metadata
    3. Merging with dynamic runtime overrides

    The resolver separates:
    - **Static config**: Values from files/profiles/CLI (via ConfigSource)
    - **Dynamic overrides**: Per-run values like AST caches, runtime state

    Examples
    --------
    Basic usage:

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class MyOptions:
    ...     threshold: float = 0.5
    ...     enabled: bool = True
    >>> resolver = PluginOptionsResolver(EmptyConfigSource())
    >>> # Would use: resolver.get_options(meta, MyOptions)
    """

    def __init__(self, config_source: ConfigSource | None = None) -> None:
        """Initialize the resolver with a configuration source.

        Parameters
        ----------
        config_source
            Source for plugin configuration. Defaults to EmptyConfigSource.
        """
        self._config_source = config_source or EmptyConfigSource()

    @property
    def config_source(self) -> ConfigSource:
        """Return the underlying configuration source.

        Returns
        -------
        ConfigSource
            The configuration source.
        """
        return self._config_source

    def get_options(
        self,
        plugin_metadata: CorePluginMetadata,
        model: type[T],
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
            These are values computed at execution time, such as
            AST caches or in-memory maps.

        Returns
        -------
        T
            An instance of `model` populated from configuration
            and dynamic overrides.

        Examples
        --------
        >>> from dataclasses import dataclass
        >>> from codeintel.core.plugins.types.metadata import (
        ...     CorePluginMetadata,
        ...     PluginDomain,
        ... )
        >>> @dataclass
        ... class TestOptions:
        ...     value: int = 10
        >>> meta = CorePluginMetadata(
        ...     name="test.plugin",
        ...     version="1.0.0",
        ...     description="Test.",
        ...     domain=PluginDomain.ANALYTICS,
        ...     kind="metric",
        ...     options_model=TestOptions,
        ... )
        >>> resolver = PluginOptionsResolver(EmptyConfigSource())
        >>> opts = resolver.get_options(meta, TestOptions)
        >>> opts.value
        10
        """
        raw = self._config_source.get_plugin_options(plugin_metadata.name) or {}

        # Construct base options from config
        base = model(**raw)

        if not dynamic_overrides:
            return base

        # Merge dynamic overrides using appropriate strategy
        return self._merge_overrides(base, dynamic_overrides)

    def _merge_overrides(self, base: T, overrides: Mapping[str, Any]) -> T:
        """Merge dynamic overrides into base options.

        Supports dataclasses, Pydantic v1, Pydantic v2, and fallback
        attribute assignment.

        Parameters
        ----------
        base
            Base options instance.
        overrides
            Override values to apply.

        Returns
        -------
        T
            Options with overrides applied.
        """
        # Dataclass: use replace()
        if hasattr(base, "__dataclass_fields__"):
            return replace(base, **overrides)  # type: ignore[return-value]

        # Pydantic v2: use model_copy()
        if hasattr(base, "model_copy"):
            return base.model_copy(update=dict(overrides))  # type: ignore[return-value]

        # Pydantic v1: use copy()
        if hasattr(base, "copy"):
            return base.copy(update=dict(overrides))  # type: ignore[return-value]

        # Fallback: direct attribute assignment
        for key, value in overrides.items():
            setattr(base, key, value)
        return base


# =============================================================================
# Configuration Bundles and Profiled Source
# =============================================================================


def _merge_dicts(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Shallow merge two option dictionaries.

    Values in `override` take precedence over `base`. None means "no layer".

    Parameters
    ----------
    base
        Base dictionary.
    override
        Override dictionary.

    Returns
    -------
    dict[str, Any]
        Merged dictionary.
    """
    result: dict[str, Any] = {}
    if base:
        result.update(base)
    if override:
        result.update(override)
    return result


@dataclass(frozen=True)
class PluginConfigBundle:
    """Configuration data for all plugins for a single layer.

    Represents one layer of configuration (base, profile, or CLI).
    Each layer maps plugin names to their option dictionaries.

    Attributes
    ----------
    plugin_options
        Mapping from plugin canonical name to options dict.

    Examples
    --------
    >>> bundle = PluginConfigBundle(plugin_options={
    ...     "analytics.function_metrics": {"include_graph_metrics": False},
    ...     "graphs.callgraph": {"scope_paths": ["src/"]},
    ... })
    >>> bundle.get("analytics.function_metrics")
    {'include_graph_metrics': False}
    >>> bundle.get("unknown.plugin") is None
    True
    """

    plugin_options: Mapping[str, Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        """Normalize None to empty mapping."""
        object.__setattr__(
            self, "plugin_options", dict(self.plugin_options or {})
        )

    def get(self, plugin_name: str) -> Mapping[str, Any] | None:
        """Return options for a specific plugin.

        Parameters
        ----------
        plugin_name
            Canonical plugin name.

        Returns
        -------
        Mapping[str, Any] | None
            Plugin options or None if not configured.
        """
        if self.plugin_options is None:
            return None
        return self.plugin_options.get(plugin_name)


class ProfiledConfigSource(ConfigSource):
    """ConfigSource that merges base, profile, and CLI overrides.

    Implements layered configuration resolution:

        effective_options = merge(base, profile, cli)

    Where each layer is optional and later layers override earlier ones
    on a key-by-key basis (shallow merge).

    Attributes
    ----------
    base
        Long-lived defaults, typically from config files.
    profile
        Profile-specific overrides ("fast", "full", "ci").
    cli
        Run-specific overrides from CLI flags.
    active_profile_name
        Name of the active profile for logging/debugging.

    Examples
    --------
    >>> base = PluginConfigBundle(plugin_options={
    ...     "analytics.function_metrics": {
    ...         "include_graph_metrics": True,
    ...         "include_coverage_metrics": True,
    ...     },
    ... })
    >>> profile = PluginConfigBundle(plugin_options={
    ...     "analytics.function_metrics": {
    ...         "include_graph_metrics": False,
    ...     },
    ... })
    >>> source = ProfiledConfigSource(
    ...     base=base,
    ...     profile=profile,
    ...     active_profile_name="fast",
    ... )
    >>> opts = source.get_plugin_options("analytics.function_metrics")
    >>> opts["include_graph_metrics"]  # Overridden by profile
    False
    >>> opts["include_coverage_metrics"]  # From base
    True
    """

    def __init__(
        self,
        *,
        base: PluginConfigBundle | None = None,
        profile: PluginConfigBundle | None = None,
        cli: PluginConfigBundle | None = None,
        active_profile_name: str | None = None,
    ) -> None:
        """Initialize with configuration layers.

        Parameters
        ----------
        base
            Base configuration layer.
        profile
            Profile-specific configuration layer.
        cli
            CLI override configuration layer.
        active_profile_name
            Name of the active profile.
        """
        self._base = base or PluginConfigBundle(plugin_options={})
        self._profile = profile or PluginConfigBundle(plugin_options={})
        self._cli = cli or PluginConfigBundle(plugin_options={})
        self._active_profile_name = active_profile_name

    @property
    def active_profile_name(self) -> str | None:
        """Return the active profile name.

        Returns
        -------
        str | None
            Active profile name or None.
        """
        return self._active_profile_name

    def get_plugin_options(self, plugin_name: str) -> Mapping[str, Any] | None:
        """Return merged options for a plugin.

        Parameters
        ----------
        plugin_name
            Canonical plugin name.

        Returns
        -------
        Mapping[str, Any] | None
            Merged options from all layers, or None if no configuration.
        """
        # Layer 1: Base config
        base_raw = self._base.get(plugin_name)

        # Layer 2: Profile overrides (only if profile is active)
        profile_raw = (
            self._profile.get(plugin_name) if self._active_profile_name else None
        )

        # Layer 3: CLI / run overrides
        cli_raw = self._cli.get(plugin_name)

        # Merge layers
        merged = _merge_dicts(base_raw, profile_raw)
        merged = _merge_dicts(merged, cli_raw)

        return merged or None


__all__ = [
    "ConfigSource",
    "EmptyConfigSource",
    "PluginConfigBundle",
    "PluginOptionsResolver",
    "ProfiledConfigSource",
]
```

### 4.2 Test File: `tests/core/plugins/test_options.py`

```python
# File: tests/core/plugins/test_options.py
"""Tests for plugin options infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from codeintel.core.plugins.execution.options import (
    ConfigSource,
    EmptyConfigSource,
    PluginConfigBundle,
    PluginOptionsResolver,
    ProfiledConfigSource,
)
from codeintel.core.plugins.types.metadata import (
    CorePluginMetadata,
    PluginDomain,
)


@dataclass
class SampleOptions:
    """Sample options model for testing."""

    threshold: float = 0.5
    enabled: bool = True
    name: str = "default"


class DictConfigSource:
    """ConfigSource backed by a dict for testing."""

    def __init__(self, options: dict[str, dict[str, Any]]) -> None:
        self._options = options

    def get_plugin_options(self, plugin_name: str) -> dict[str, Any] | None:
        return self._options.get(plugin_name)


class TestEmptyConfigSource:
    """Tests for EmptyConfigSource."""

    def test_always_returns_none(self) -> None:
        """Verify EmptyConfigSource returns None for any plugin."""
        source = EmptyConfigSource()
        assert source.get_plugin_options("any.plugin") is None
        assert source.get_plugin_options("another.plugin") is None

    def test_implements_protocol(self) -> None:
        """Verify EmptyConfigSource implements ConfigSource."""
        source = EmptyConfigSource()
        assert isinstance(source, ConfigSource)


class TestPluginOptionsResolver:
    """Tests for PluginOptionsResolver."""

    @pytest.fixture
    def sample_metadata(self) -> CorePluginMetadata:
        """Create sample metadata for testing."""
        return CorePluginMetadata(
            name="test.plugin",
            version="1.0.0",
            description="Test plugin.",
            domain=PluginDomain.ANALYTICS,
            kind="metric",
            options_model=SampleOptions,
        )

    def test_with_empty_source_uses_defaults(
        self, sample_metadata: CorePluginMetadata
    ) -> None:
        """Verify default options are used with empty config."""
        resolver = PluginOptionsResolver(EmptyConfigSource())
        opts = resolver.get_options(sample_metadata, SampleOptions)
        assert opts.threshold == 0.5
        assert opts.enabled is True
        assert opts.name == "default"

    def test_with_config_overrides_defaults(
        self, sample_metadata: CorePluginMetadata
    ) -> None:
        """Verify config values override defaults."""
        source = DictConfigSource({
            "test.plugin": {"threshold": 0.8, "name": "custom"},
        })
        resolver = PluginOptionsResolver(source)
        opts = resolver.get_options(sample_metadata, SampleOptions)
        assert opts.threshold == 0.8
        assert opts.enabled is True  # Still default
        assert opts.name == "custom"

    def test_dynamic_overrides(
        self, sample_metadata: CorePluginMetadata
    ) -> None:
        """Verify dynamic overrides are applied."""
        source = DictConfigSource({
            "test.plugin": {"threshold": 0.8},
        })
        resolver = PluginOptionsResolver(source)
        opts = resolver.get_options(
            sample_metadata,
            SampleOptions,
            dynamic_overrides={"name": "runtime"},
        )
        assert opts.threshold == 0.8  # From config
        assert opts.name == "runtime"  # From dynamic override

    def test_config_source_property(self) -> None:
        """Verify config_source property returns the source."""
        source = EmptyConfigSource()
        resolver = PluginOptionsResolver(source)
        assert resolver.config_source is source


class TestPluginConfigBundle:
    """Tests for PluginConfigBundle."""

    def test_get_existing_plugin(self) -> None:
        """Verify get returns options for existing plugin."""
        bundle = PluginConfigBundle(plugin_options={
            "plugin.a": {"key": "value"},
        })
        assert bundle.get("plugin.a") == {"key": "value"}

    def test_get_missing_plugin(self) -> None:
        """Verify get returns None for missing plugin."""
        bundle = PluginConfigBundle(plugin_options={
            "plugin.a": {"key": "value"},
        })
        assert bundle.get("plugin.b") is None

    def test_none_plugin_options(self) -> None:
        """Verify None plugin_options is normalized to empty dict."""
        bundle = PluginConfigBundle(plugin_options=None)
        assert bundle.get("any.plugin") is None


class TestProfiledConfigSource:
    """Tests for ProfiledConfigSource."""

    def test_base_only(self) -> None:
        """Verify base layer is used when no profile."""
        base = PluginConfigBundle(plugin_options={
            "plugin.a": {"key": "base_value"},
        })
        source = ProfiledConfigSource(base=base)
        opts = source.get_plugin_options("plugin.a")
        assert opts is not None
        assert opts["key"] == "base_value"

    def test_profile_overrides_base(self) -> None:
        """Verify profile layer overrides base."""
        base = PluginConfigBundle(plugin_options={
            "plugin.a": {"key": "base", "other": "base_other"},
        })
        profile = PluginConfigBundle(plugin_options={
            "plugin.a": {"key": "profile"},
        })
        source = ProfiledConfigSource(
            base=base,
            profile=profile,
            active_profile_name="fast",
        )
        opts = source.get_plugin_options("plugin.a")
        assert opts is not None
        assert opts["key"] == "profile"  # Overridden
        assert opts["other"] == "base_other"  # From base

    def test_cli_overrides_profile(self) -> None:
        """Verify CLI layer overrides profile."""
        base = PluginConfigBundle(plugin_options={
            "plugin.a": {"key": "base"},
        })
        profile = PluginConfigBundle(plugin_options={
            "plugin.a": {"key": "profile"},
        })
        cli = PluginConfigBundle(plugin_options={
            "plugin.a": {"key": "cli"},
        })
        source = ProfiledConfigSource(
            base=base,
            profile=profile,
            cli=cli,
            active_profile_name="fast",
        )
        opts = source.get_plugin_options("plugin.a")
        assert opts is not None
        assert opts["key"] == "cli"

    def test_profile_not_applied_without_active_name(self) -> None:
        """Verify profile is not applied without active_profile_name."""
        base = PluginConfigBundle(plugin_options={
            "plugin.a": {"key": "base"},
        })
        profile = PluginConfigBundle(plugin_options={
            "plugin.a": {"key": "profile"},
        })
        source = ProfiledConfigSource(
            base=base,
            profile=profile,
            active_profile_name=None,  # No active profile
        )
        opts = source.get_plugin_options("plugin.a")
        assert opts is not None
        assert opts["key"] == "base"  # Profile not applied

    def test_missing_plugin_returns_none(self) -> None:
        """Verify None is returned for unconfigured plugins."""
        source = ProfiledConfigSource()
        assert source.get_plugin_options("unknown.plugin") is None

    def test_implements_protocol(self) -> None:
        """Verify ProfiledConfigSource implements ConfigSource."""
        source = ProfiledConfigSource()
        assert isinstance(source, ConfigSource)
```

---

## 5. Task 3: Enhanced Manifest Infrastructure

### 5.1 Add to `core/plugins/execution/manifest.py`

Add the `ManifestStore` protocol and upstream state utilities. These are **additions** to the existing file.

```python
# Add to: src/codeintel/core/plugins/execution/manifest.py
# Location: After the existing content, before __all__

# =============================================================================
# ManifestStore Protocol
# =============================================================================


class ManifestStore(Protocol):
    """Abstract interface for storing and retrieving execution records.

    Implementations persist PluginExecutionRecord instances and support
    queries for the most recent record matching specific criteria.

    This protocol enables:
    - Skip/rerun decisions based on input hash comparison
    - Upstream state resolution for dependency tracking
    - Execution history for debugging and analytics

    Examples
    --------
    Implementing a simple in-memory store:

    >>> class InMemoryManifestStore:
    ...     def __init__(self) -> None:
    ...         self._records: list[PluginExecutionRecord] = []
    ...
    ...     def load_last_record(
    ...         self,
    ...         *,
    ...         plugin_name: str,
    ...         repo: str,
    ...         commit: str,
    ...         scope_id: str | None,
    ...         variant: str | None,
    ...     ) -> PluginExecutionRecord | None:
    ...         for rec in reversed(self._records):
    ...             if (rec.plugin_name == plugin_name and
    ...                 rec.meta.get("repo") == repo and
    ...                 rec.meta.get("commit") == commit):
    ...                 return rec
    ...         return None
    ...
    ...     def append_record(self, record: PluginExecutionRecord) -> None:
    ...         self._records.append(record)
    """

    def load_last_record(
        self,
        *,
        plugin_name: str,
        repo: str,
        commit: str,
        scope_id: str | None,
        variant: str | None,
    ) -> PluginExecutionRecord | None:
        """Return the most recent record for this combination.

        Parameters
        ----------
        plugin_name
            Canonical plugin name.
        repo
            Repository identifier ("owner/repo").
        commit
            Commit SHA.
        scope_id
            Scope hash (paths/modules) or None for whole-repo.
        variant
            Profile/variant name ("fast", "full", etc.) or None.

        Returns
        -------
        PluginExecutionRecord | None
            Most recent matching record, or None if not found.
        """
        ...

    def append_record(self, record: PluginExecutionRecord) -> None:
        """Persist a new PluginExecutionRecord.

        Parameters
        ----------
        record
            Execution record to persist.
        """
        ...


# =============================================================================
# Upstream State Resolution
# =============================================================================


def compute_scope_id(paths: list[str] | None) -> str | None:
    """Compute a stable scope hash for a set of repo-relative paths.

    Parameters
    ----------
    paths
        List of repo-relative paths, or None for whole-repo scope.

    Returns
    -------
    str | None
        SHA-256 hash (first 16 chars) or None if no paths.

    Examples
    --------
    >>> compute_scope_id(["src/", "lib/"])
    '...'  # 16-char hash
    >>> compute_scope_id(None) is None
    True
    >>> compute_scope_id([]) is None
    True
    """
    if not paths:
        return None
    payload = {"paths": sorted(paths)}
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def build_upstream_state_from_records(
    required_capabilities: tuple[str, ...],
    provider_lookup: Mapping[str, str],
    manifest_store: ManifestStore,
    repo: str,
    commit: str,
    scope_id: str | None,
    variant: str | None,
) -> dict[str, str]:
    """Build upstream_state from required capabilities.

    Parameters
    ----------
    required_capabilities
        Capability strings the plugin requires.
    provider_lookup
        Mapping from capability name to provider plugin name.
    manifest_store
        Store for loading prior execution records.
    repo
        Repository identifier.
    commit
        Commit SHA.
    scope_id
        Scope hash or None.
    variant
        Profile/variant name or None.

    Returns
    -------
    dict[str, str]
        Capability → provider's input_hash.

    Examples
    --------
    >>> state = build_upstream_state_from_records(
    ...     required_capabilities=("core.goids", "graph.callgraph"),
    ...     provider_lookup={
    ...         "core.goids": "graphs.goid_builder",
    ...         "graph.callgraph": "graphs.callgraph",
    ...     },
    ...     manifest_store=store,
    ...     repo="owner/repo",
    ...     commit="abc123",
    ...     scope_id=None,
    ...     variant="fast",
    ... )
    """
    state: dict[str, str] = {}

    for cap in required_capabilities:
        provider_name = provider_lookup.get(cap)
        if not provider_name:
            continue

        rec = manifest_store.load_last_record(
            plugin_name=provider_name,
            repo=repo,
            commit=commit,
            scope_id=scope_id,
            variant=variant,
        )
        if rec and rec.meta.get("input_hash"):
            state[cap] = str(rec.meta["input_hash"])

    return state


# Update __all__ to include new exports
# Add these to the existing __all__ list:
# "ManifestStore",
# "compute_scope_id",
# "build_upstream_state_from_records",
```

### 5.2 Update `__all__` in manifest.py

Add the new exports to the existing `__all__` list:

```python
__all__ = [
    # Existing exports
    "InputHashPayload",
    "ManifestState",
    "PluginExecutionManifest",
    "build_manifest_entry",
    "compute_input_hash",
    "compute_options_hash",
    "create_skip_record",
    "is_unchanged",
    # New exports
    "ManifestStore",
    "build_upstream_state_from_records",
    "compute_scope_id",
]
```

### 5.3 Test Additions: `tests/core/plugins/test_manifest_extensions.py`

```python
# File: tests/core/plugins/test_manifest_extensions.py
"""Tests for manifest infrastructure extensions."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.core.plugins.execution.manifest import (
    build_upstream_state_from_records,
    compute_scope_id,
)
from codeintel.core.plugins.types.result import PluginExecutionRecord


class InMemoryManifestStore:
    """Simple in-memory manifest store for testing."""

    def __init__(self) -> None:
        self._records: dict[str, PluginExecutionRecord] = {}

    def load_last_record(
        self,
        *,
        plugin_name: str,
        repo: str,
        commit: str,
        scope_id: str | None,
        variant: str | None,
    ) -> PluginExecutionRecord | None:
        key = f"{plugin_name}:{repo}:{commit}:{scope_id}:{variant}"
        return self._records.get(key)

    def append_record(self, record: PluginExecutionRecord) -> None:
        repo = record.meta.get("repo", "")
        commit = record.meta.get("commit", "")
        scope_id = record.meta.get("scope_id")
        variant = record.meta.get("variant")
        key = f"{record.plugin_name}:{repo}:{commit}:{scope_id}:{variant}"
        self._records[key] = record


class TestComputeScopeId:
    """Tests for compute_scope_id."""

    def test_none_returns_none(self) -> None:
        """Verify None paths returns None."""
        assert compute_scope_id(None) is None

    def test_empty_list_returns_none(self) -> None:
        """Verify empty list returns None."""
        assert compute_scope_id([]) is None

    def test_paths_return_hash(self) -> None:
        """Verify paths return a 16-char hash."""
        result = compute_scope_id(["src/", "lib/"])
        assert result is not None
        assert len(result) == 16

    def test_order_independent(self) -> None:
        """Verify hash is order-independent."""
        hash1 = compute_scope_id(["src/", "lib/"])
        hash2 = compute_scope_id(["lib/", "src/"])
        assert hash1 == hash2

    def test_different_paths_different_hash(self) -> None:
        """Verify different paths produce different hashes."""
        hash1 = compute_scope_id(["src/"])
        hash2 = compute_scope_id(["lib/"])
        assert hash1 != hash2


class TestBuildUpstreamState:
    """Tests for build_upstream_state_from_records."""

    @pytest.fixture
    def manifest_store(self) -> InMemoryManifestStore:
        """Create a test manifest store."""
        return InMemoryManifestStore()

    @pytest.fixture
    def provider_lookup(self) -> dict[str, str]:
        """Create a test provider lookup."""
        return {
            "core.goids": "graphs.goid_builder",
            "graph.callgraph": "graphs.callgraph",
        }

    def test_empty_capabilities(
        self,
        manifest_store: InMemoryManifestStore,
        provider_lookup: dict[str, str],
    ) -> None:
        """Verify empty capabilities returns empty state."""
        state = build_upstream_state_from_records(
            required_capabilities=(),
            provider_lookup=provider_lookup,
            manifest_store=manifest_store,
            repo="owner/repo",
            commit="abc123",
            scope_id=None,
            variant=None,
        )
        assert state == {}

    def test_missing_provider(
        self,
        manifest_store: InMemoryManifestStore,
    ) -> None:
        """Verify missing provider is skipped."""
        state = build_upstream_state_from_records(
            required_capabilities=("unknown.capability",),
            provider_lookup={},  # No providers
            manifest_store=manifest_store,
            repo="owner/repo",
            commit="abc123",
            scope_id=None,
            variant=None,
        )
        assert state == {}

    def test_with_records(
        self,
        manifest_store: InMemoryManifestStore,
        provider_lookup: dict[str, str],
    ) -> None:
        """Verify state is populated from records."""
        now = datetime.now(tz=UTC)
        record = PluginExecutionRecord(
            plugin_name="graphs.goid_builder",
            status="succeeded",
            started_at=now,
            ended_at=now,
            duration_ms=100,
            meta={
                "repo": "owner/repo",
                "commit": "abc123",
                "scope_id": None,
                "variant": None,
                "input_hash": "hash123",
            },
        )
        manifest_store.append_record(record)

        state = build_upstream_state_from_records(
            required_capabilities=("core.goids",),
            provider_lookup=provider_lookup,
            manifest_store=manifest_store,
            repo="owner/repo",
            commit="abc123",
            scope_id=None,
            variant=None,
        )
        assert state == {"core.goids": "hash123"}
```

---

## 6. Task 4: Capability Registry Index

### 6.1 Create `core/plugins/registry/capability_index.py`

```python
# File: src/codeintel/core/plugins/registry/capability_index.py
"""Capability-based plugin registry index.

This module provides indexes for looking up plugins by:
- Name (canonical identifier)
- Capability (what they provide)
- Output table (what they produce)

The index is built from CorePluginMetadata instances and enables:
- Capability → provider resolution for upstream state
- Dataset → provider resolution for debugging
- Fast plugin lookups without registry scanning

Architecture
------------
The PluginRegistryIndex is a read-only, frozen data structure built once
from a collection of metadata instances. It's designed to be cached at
module level and shared across the application.

Usage
-----
>>> from codeintel.core.plugins.registry.capability_index import (
...     build_registry_index,
... )
>>> index = build_registry_index([meta1, meta2, meta3])
>>> provider = index.by_capability.get("core.goids")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.core.plugins.types.metadata import CorePluginMetadata


@dataclass(frozen=True)
class PluginRegistryIndex:
    """Index for looking up plugins by name, capability, or output table.

    This is a read-only data structure built from plugin metadata.
    All lookups are O(1) dictionary operations.

    Attributes
    ----------
    by_name
        Plugin name → metadata.
    by_capability
        Capability name → provider metadata.
        Note: If multiple plugins provide the same capability,
        the last one registered wins.
    by_output_table
        Table name → producer metadata.
        Note: If multiple plugins produce the same table,
        the last one registered wins.

    Examples
    --------
    >>> index = build_registry_index(all_metadata)
    >>> meta = index.by_name.get("analytics.function_metrics")
    >>> provider = index.by_capability.get("core.goids")
    """

    by_name: dict[str, CorePluginMetadata]
    by_capability: dict[str, CorePluginMetadata]
    by_output_table: dict[str, CorePluginMetadata]

    def get_by_name(self, name: str) -> CorePluginMetadata | None:
        """Look up plugin metadata by canonical name.

        Parameters
        ----------
        name
            Canonical plugin name.

        Returns
        -------
        CorePluginMetadata | None
            Metadata if found, None otherwise.
        """
        return self.by_name.get(name)

    def get_provider(self, capability: str) -> CorePluginMetadata | None:
        """Look up the provider of a capability.

        Parameters
        ----------
        capability
            Capability name.

        Returns
        -------
        CorePluginMetadata | None
            Provider metadata if found, None otherwise.
        """
        return self.by_capability.get(capability)

    def get_producer(self, table: str) -> CorePluginMetadata | None:
        """Look up the producer of a table.

        Parameters
        ----------
        table
            Table name.

        Returns
        -------
        CorePluginMetadata | None
            Producer metadata if found, None otherwise.
        """
        return self.by_output_table.get(table)

    def provider_lookup(self) -> dict[str, str]:
        """Build a capability → provider name lookup.

        Returns
        -------
        dict[str, str]
            Capability name → provider plugin name.
        """
        return {cap: meta.name for cap, meta in self.by_capability.items()}

    def all_capabilities(self) -> tuple[str, ...]:
        """Return all registered capabilities.

        Returns
        -------
        tuple[str, ...]
            All capability names.
        """
        return tuple(self.by_capability.keys())

    def all_tables(self) -> tuple[str, ...]:
        """Return all registered output tables.

        Returns
        -------
        tuple[str, ...]
            All table names.
        """
        return tuple(self.by_output_table.keys())


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

    Notes
    -----
    If multiple plugins provide the same capability or produce the same
    table, the last one in iteration order wins. This is intentional to
    allow overrides, but callers should ensure metadata is consistent.

    Examples
    --------
    >>> from codeintel.core.plugins.types.metadata import (
    ...     CorePluginMetadata,
    ...     PluginDomain,
    ... )
    >>> meta1 = CorePluginMetadata(
    ...     name="plugin.a",
    ...     version="1.0.0",
    ...     description="Plugin A",
    ...     domain=PluginDomain.ANALYTICS,
    ...     kind="metric",
    ...     provides=("cap.a",),
    ...     produces_tables=("table.a",),
    ... )
    >>> meta2 = CorePluginMetadata(
    ...     name="plugin.b",
    ...     version="1.0.0",
    ...     description="Plugin B",
    ...     domain=PluginDomain.ANALYTICS,
    ...     kind="metric",
    ...     provides=("cap.b",),
    ...     requires=("cap.a",),
    ... )
    >>> index = build_registry_index([meta1, meta2])
    >>> index.by_name["plugin.a"].name
    'plugin.a'
    >>> index.by_capability["cap.a"].name
    'plugin.a'
    """
    by_name: dict[str, CorePluginMetadata] = {}
    by_capability: dict[str, CorePluginMetadata] = {}
    by_output_table: dict[str, CorePluginMetadata] = {}

    for meta in all_metadata:
        by_name[meta.name] = meta

        for cap in meta.provides:
            by_capability[cap] = meta

        for table in meta.produces_tables:
            by_output_table[table] = meta

    return PluginRegistryIndex(
        by_name=by_name,
        by_capability=by_capability,
        by_output_table=by_output_table,
    )


__all__ = [
    "PluginRegistryIndex",
    "build_registry_index",
]
```

### 6.2 Test File: `tests/core/plugins/test_capability_index.py`

```python
# File: tests/core/plugins/test_capability_index.py
"""Tests for capability registry index."""

from __future__ import annotations

import pytest

from codeintel.core.plugins.registry.capability_index import (
    PluginRegistryIndex,
    build_registry_index,
)
from codeintel.core.plugins.types.metadata import (
    CorePluginMetadata,
    PluginDomain,
)


@pytest.fixture
def sample_metadata() -> list[CorePluginMetadata]:
    """Create sample metadata for testing."""
    return [
        CorePluginMetadata(
            name="plugin.a",
            version="1.0.0",
            description="Plugin A",
            domain=PluginDomain.ANALYTICS,
            kind="metric",
            provides=("cap.a", "cap.shared"),
            produces_tables=("table.a",),
        ),
        CorePluginMetadata(
            name="plugin.b",
            version="1.0.0",
            description="Plugin B",
            domain=PluginDomain.GRAPH,
            kind="builder",
            provides=("cap.b",),
            requires=("cap.a",),
            produces_tables=("table.b",),
        ),
        CorePluginMetadata(
            name="plugin.c",
            version="1.0.0",
            description="Plugin C",
            domain=PluginDomain.INGEST,
            kind="builder",
            provides=("cap.c", "cap.shared"),  # Overrides cap.shared
            produces_tables=("table.c",),
        ),
    ]


class TestBuildRegistryIndex:
    """Tests for build_registry_index."""

    def test_by_name_lookup(
        self, sample_metadata: list[CorePluginMetadata]
    ) -> None:
        """Verify by_name lookup works."""
        index = build_registry_index(sample_metadata)
        assert "plugin.a" in index.by_name
        assert "plugin.b" in index.by_name
        assert "plugin.c" in index.by_name
        assert index.by_name["plugin.a"].description == "Plugin A"

    def test_by_capability_lookup(
        self, sample_metadata: list[CorePluginMetadata]
    ) -> None:
        """Verify by_capability lookup works."""
        index = build_registry_index(sample_metadata)
        assert index.by_capability["cap.a"].name == "plugin.a"
        assert index.by_capability["cap.b"].name == "plugin.b"

    def test_capability_override(
        self, sample_metadata: list[CorePluginMetadata]
    ) -> None:
        """Verify last provider wins for shared capability."""
        index = build_registry_index(sample_metadata)
        # plugin.c provides cap.shared last
        assert index.by_capability["cap.shared"].name == "plugin.c"

    def test_by_output_table_lookup(
        self, sample_metadata: list[CorePluginMetadata]
    ) -> None:
        """Verify by_output_table lookup works."""
        index = build_registry_index(sample_metadata)
        assert index.by_output_table["table.a"].name == "plugin.a"
        assert index.by_output_table["table.b"].name == "plugin.b"

    def test_empty_metadata(self) -> None:
        """Verify empty metadata produces empty index."""
        index = build_registry_index([])
        assert index.by_name == {}
        assert index.by_capability == {}
        assert index.by_output_table == {}


class TestPluginRegistryIndex:
    """Tests for PluginRegistryIndex methods."""

    @pytest.fixture
    def index(self, sample_metadata: list[CorePluginMetadata]) -> PluginRegistryIndex:
        """Build index from sample metadata."""
        return build_registry_index(sample_metadata)

    def test_get_by_name_found(self, index: PluginRegistryIndex) -> None:
        """Verify get_by_name returns metadata when found."""
        meta = index.get_by_name("plugin.a")
        assert meta is not None
        assert meta.name == "plugin.a"

    def test_get_by_name_not_found(self, index: PluginRegistryIndex) -> None:
        """Verify get_by_name returns None when not found."""
        assert index.get_by_name("unknown") is None

    def test_get_provider_found(self, index: PluginRegistryIndex) -> None:
        """Verify get_provider returns metadata when found."""
        meta = index.get_provider("cap.a")
        assert meta is not None
        assert meta.name == "plugin.a"

    def test_get_provider_not_found(self, index: PluginRegistryIndex) -> None:
        """Verify get_provider returns None when not found."""
        assert index.get_provider("unknown.cap") is None

    def test_provider_lookup(self, index: PluginRegistryIndex) -> None:
        """Verify provider_lookup returns name mapping."""
        lookup = index.provider_lookup()
        assert lookup["cap.a"] == "plugin.a"
        assert lookup["cap.b"] == "plugin.b"

    def test_all_capabilities(self, index: PluginRegistryIndex) -> None:
        """Verify all_capabilities returns all registered capabilities."""
        caps = index.all_capabilities()
        assert "cap.a" in caps
        assert "cap.b" in caps
        assert "cap.shared" in caps

    def test_all_tables(self, index: PluginRegistryIndex) -> None:
        """Verify all_tables returns all registered tables."""
        tables = index.all_tables()
        assert "table.a" in tables
        assert "table.b" in tables
        assert "table.c" in tables
```

---

## 7. Task 5: Export Updates

### 7.1 Update `core/plugins/types/__init__.py`

Add exports for the new metadata types:

```python
# Add to: src/codeintel/core/plugins/types/__init__.py
# Add these imports:
from codeintel.core.plugins.types.metadata import (
    CorePluginMetadata,
    PluginDomain,
)

# Add to __all__:
# "CorePluginMetadata",
# "PluginDomain",
```

### 7.2 Update `core/plugins/execution/__init__.py`

Add exports for the new options types:

```python
# Add to: src/codeintel/core/plugins/execution/__init__.py
# Add these imports:
from codeintel.core.plugins.execution.options import (
    ConfigSource,
    EmptyConfigSource,
    PluginConfigBundle,
    PluginOptionsResolver,
    ProfiledConfigSource,
)

# Add to existing manifest imports:
from codeintel.core.plugins.execution.manifest import (
    # ... existing imports ...
    ManifestStore,
    build_upstream_state_from_records,
    compute_scope_id,
)

# Add to __all__:
# "ConfigSource",
# "EmptyConfigSource",
# "ManifestStore",
# "PluginConfigBundle",
# "PluginOptionsResolver",
# "ProfiledConfigSource",
# "build_upstream_state_from_records",
# "compute_scope_id",
```

### 7.3 Update `core/plugins/registry/__init__.py`

Add exports for the capability index:

```python
# Add to: src/codeintel/core/plugins/registry/__init__.py
# Add these imports:
from codeintel.core.plugins.registry.capability_index import (
    PluginRegistryIndex,
    build_registry_index,
)

# Add to __all__:
# "PluginRegistryIndex",
# "build_registry_index",
```

---

## 8. Verification

### 8.1 Run Quality Checks

```bash
# Format and lint
uv run ruff format src/codeintel/core/plugins/
uv run ruff check --fix src/codeintel/core/plugins/

# Type checking
uv run pyright src/codeintel/core/plugins/types/metadata.py
uv run pyright src/codeintel/core/plugins/execution/options.py
uv run pyright src/codeintel/core/plugins/registry/capability_index.py

# Pyrefly
uv run pyrefly check src/codeintel/core/plugins/
```

### 8.2 Run Tests

```bash
# Run new tests
uv run pytest tests/core/plugins/test_metadata.py -v
uv run pytest tests/core/plugins/test_options.py -v
uv run pytest tests/core/plugins/test_capability_index.py -v
uv run pytest tests/core/plugins/test_manifest_extensions.py -v

# Run existing tests to verify no regressions
uv run pytest tests/core/plugins/ -v
```

### 8.3 Verification Checklist

- [ ] All new files pass `ruff format` and `ruff check`
- [ ] All new files pass `pyright --strict`
- [ ] All new files pass `pyrefly check`
- [ ] All new tests pass
- [ ] All existing `tests/core/plugins/` tests pass
- [ ] New types are properly exported from package `__init__.py` files
- [ ] Import statements work: `from codeintel.core.plugins.types import CorePluginMetadata`

---

## 9. Rollback Plan

Since all Phase 1 changes are additive, rollback is straightforward:

1. **Delete new files**:
   - `src/codeintel/core/plugins/types/metadata.py`
   - `src/codeintel/core/plugins/execution/options.py`
   - `src/codeintel/core/plugins/registry/capability_index.py`

2. **Revert `__init__.py` changes**:
   - Remove new imports and exports from `types/__init__.py`
   - Remove new imports and exports from `execution/__init__.py`
   - Remove new imports and exports from `registry/__init__.py`

3. **Revert manifest.py additions**:
   - Remove `ManifestStore` protocol
   - Remove `compute_scope_id` function
   - Remove `build_upstream_state_from_records` function
   - Restore original `__all__` list

4. **Delete test files**:
   - `tests/core/plugins/test_metadata.py`
   - `tests/core/plugins/test_options.py`
   - `tests/core/plugins/test_capability_index.py`
   - `tests/core/plugins/test_manifest_extensions.py`

---

## Appendix A: File Checklist

| File | Action | Status |
|------|--------|--------|
| `src/codeintel/core/plugins/types/metadata.py` | CREATE | ⬜ |
| `src/codeintel/core/plugins/execution/options.py` | CREATE | ⬜ |
| `src/codeintel/core/plugins/registry/capability_index.py` | CREATE | ⬜ |
| `src/codeintel/core/plugins/execution/manifest.py` | MODIFY | ⬜ |
| `src/codeintel/core/plugins/types/__init__.py` | MODIFY | ⬜ |
| `src/codeintel/core/plugins/execution/__init__.py` | MODIFY | ⬜ |
| `src/codeintel/core/plugins/registry/__init__.py` | MODIFY | ⬜ |
| `tests/core/plugins/test_metadata.py` | CREATE | ⬜ |
| `tests/core/plugins/test_options.py` | CREATE | ⬜ |
| `tests/core/plugins/test_capability_index.py` | CREATE | ⬜ |
| `tests/core/plugins/test_manifest_extensions.py` | CREATE | ⬜ |

## Appendix B: Import Verification Script

After implementation, verify imports work correctly:

```python
#!/usr/bin/env python3
"""Verify Phase 1 imports work correctly."""

def verify_imports() -> None:
    """Verify all Phase 1 types can be imported."""
    # Metadata types
    from codeintel.core.plugins.types.metadata import (
        CorePluginMetadata,
        PluginDomain,
    )
    
    # Options types
    from codeintel.core.plugins.execution.options import (
        ConfigSource,
        EmptyConfigSource,
        PluginConfigBundle,
        PluginOptionsResolver,
        ProfiledConfigSource,
    )
    
    # Manifest additions
    from codeintel.core.plugins.execution.manifest import (
        ManifestStore,
        build_upstream_state_from_records,
        compute_scope_id,
    )
    
    # Capability index
    from codeintel.core.plugins.registry.capability_index import (
        PluginRegistryIndex,
        build_registry_index,
    )
    
    # Package-level exports
    from codeintel.core.plugins.types import (
        CorePluginMetadata,
        PluginDomain,
    )
    from codeintel.core.plugins.execution import (
        ConfigSource,
        PluginOptionsResolver,
        ProfiledConfigSource,
    )
    from codeintel.core.plugins.registry import (
        PluginRegistryIndex,
        build_registry_index,
    )
    
    print("✓ All Phase 1 imports verified successfully")


if __name__ == "__main__":
    verify_imports()
```

---

**Next Steps**: After Phase 1 is complete, proceed to Phase 2 (Spine Plugin Migration) to attach metadata and options to representative plugins from each domain.
