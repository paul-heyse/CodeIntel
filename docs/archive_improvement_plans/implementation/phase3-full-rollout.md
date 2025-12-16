# Phase 3: Full Rollout Implementation Plan

> **Scope**: Migrate all remaining plugins to use CorePluginMetadata and options resolution
> **Duration**: 3-5 days
> **Risk Level**: Medium (bulk changes, but following established patterns)
> **Depends On**: Phase 2 (Spine Plugin Migration)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Migration Strategy](#3-migration-strategy)
4. [Task 1: Analytics Domain Plugins](#4-task-1-analytics-domain-plugins)
5. [Task 2: Graphs Domain Plugins](#5-task-2-graphs-domain-plugins)
6. [Task 3: Ingestion Domain Plugins](#6-task-3-ingestion-domain-plugins)
7. [Task 4: Build All Metadata Registry](#7-task-4-build-all-metadata-registry)
8. [Verification](#8-verification)
9. [Rollback Plan](#9-rollback-plan)

---

## 1. Overview

Phase 3 extends the patterns established in Phase 2 to all remaining plugins across all domains. Each plugin receives:

1. **`CorePluginMetadata` constant** - Canonical metadata declaration
2. **Options model** (if configurable) - Typed configuration dataclass
3. **`PluginOptionsResolver` integration** - Profile-driven configuration
4. **`metadata` property** - Protocol-compatible facade

### Plugin Inventory

After completing inventory discovery, the plugins requiring migration are:

#### Analytics Domain
| Plugin | Current File | Priority |
|--------|--------------|----------|
| `TypeCoveragePlugin` | `analytics/plugins/types/coverage.py` | High |
| `DocstringMetricsPlugin` | `analytics/plugins/docs/docstrings.py` | Medium |
| `RiskProfilePlugin` | `analytics/plugins/risk/profile.py` | Medium |
| `DependencyGraphPlugin` | `analytics/plugins/deps/graph.py` | Medium |
| `HotspotAnalysisPlugin` | `analytics/plugins/hotspots/analysis.py` | Low |

#### Graphs Domain
| Plugin | Current File | Priority |
|--------|--------------|----------|
| `ImportGraphPlugin` | `graphs/plugins/builders/import_graph.py` | High |
| `DataFlowGraphPlugin` | `graphs/plugins/builders/dataflow.py` | Medium |
| `ControlFlowGraphPlugin` | `graphs/plugins/builders/cfg.py` | Medium |
| `SymbolUseGraphPlugin` | `graphs/plugins/builders/symbol_use.py` | Low |

#### Ingestion Domain
| Plugin | Current File | Priority |
|--------|--------------|----------|
| `ModuleIngestPlugin` | `ingestion/plugins/modules.py` | High |
| `GoidBuilderPlugin` | `ingestion/plugins/goid_builder.py` | High |
| `SourceFilesPlugin` | `ingestion/plugins/source_files.py` | Medium |
| `TreeSitterPlugin` | `ingestion/plugins/tree_sitter.py` | Low |

---

## 2. Prerequisites

Verify Phase 2 spine plugins are complete:

```bash
# Verify spine plugin metadata is exported
uv run python -c "
from codeintel.analytics.plugins.functions.metrics import FUNCTION_METRICS_METADATA
from codeintel.graphs.plugins.builders.callgraph import CALLGRAPH_METADATA
from codeintel.ingestion.plugins.scip_plugin import SCIP_INGEST_METADATA
print('Phase 2 spine plugins verified')
print(f'  Analytics: {FUNCTION_METRICS_METADATA.name}')
print(f'  Graphs: {CALLGRAPH_METADATA.name}')
print(f'  Ingestion: {SCIP_INGEST_METADATA.name}')
"

# Run spine plugin tests
uv run pytest \
    tests/analytics/plugins/test_function_metrics_metadata.py \
    tests/graphs/plugins/test_callgraph_metadata.py \
    tests/ingestion/plugins/test_scip_metadata.py \
    -v
```

---

## 3. Migration Strategy

### 3.1 Migration Template

Each plugin follows this migration pattern:

```python
# 1. Add imports
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.types.protocol import PluginMetadata

# 2. Define metadata constant
PLUGIN_NAME_METADATA = CorePluginMetadata(
    name="domain.plugin_name",
    version="X.Y.Z",
    description="...",
    domain=PluginDomain.DOMAIN,
    kind="...",
    stage="...",
    provides=(...),
    requires=(...),
    produces_tables=(...),
    consumes_tables=(...),
    options_model=PluginOptions,  # If configurable
)

# 3. Add metadata conversion helper
def _to_plugin_metadata(core: CorePluginMetadata) -> PluginMetadata:
    return PluginMetadata(
        name=core.name,
        version=core.version,
        description=core.description,
        kind=core.kind,
        stage=core.stage or "default",
        provides=core.provides,
        requires=core.requires,
        produces_tables=core.produces_tables,
    )

# 4. Update plugin class
class MyPlugin(TargetPlugin):
    _core_metadata: ClassVar[CorePluginMetadata] = PLUGIN_NAME_METADATA

    def __init__(self, *, options_resolver: PluginOptionsResolver | None = None) -> None:
        self._options_resolver = options_resolver

    @property
    def metadata(self) -> PluginMetadata:
        return _to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        return self._core_metadata

    def resolve_options(self, *, dynamic_overrides: Mapping[str, Any] | None = None) -> Options:
        # ... resolve options pattern from spine plugins ...
```

### 3.2 Options Model Pattern

For plugins with configurable behavior:

```python
@dataclass(frozen=True)
class PluginNameOptions:
    """Configuration options for [plugin name].

    Config-Driven Fields
    --------------------
    field1 : type
        Description. Profile impact.
    field2 : type
        Description. Profile impact.

    Dynamic Fields
    --------------
    runtime_field : type
        Set at execution time.
    """
    # Config-driven fields
    field1: type = default_value
    field2: type = default_value

    # Dynamic fields (runtime only)
    runtime_field: type | None = None
```

---

## 4. Task 1: Analytics Domain Plugins

### 4.1 TypeCoveragePlugin Migration

```python
# File: src/codeintel/analytics/plugins/types/options.py
"""Type coverage plugin options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TypeCoverageOptions:
    """Configuration options for type coverage analysis.

    Attributes
    ----------
    scope_paths : list[str] | None
        If set, only analyze files within these paths.
    include_private : bool
        Whether to include private functions in coverage.
    strictness : str
        Type checking strictness ("strict", "standard", "lenient").
    """

    scope_paths: list[str] | None = None
    include_private: bool = True
    strictness: str = "standard"


__all__ = ["TypeCoverageOptions"]
```

```python
# File: src/codeintel/analytics/plugins/types/coverage.py (modification)
# Add at module level after imports:

from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.analytics.plugins.types.options import TypeCoverageOptions

TYPE_COVERAGE_METADATA = CorePluginMetadata(
    name="analytics.type_coverage",
    version="2.0.0",
    description="Compute type annotation coverage metrics.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="function",
    provides=("analytics.type_coverage",),
    requires=("core.goids", "analytics.function_types"),
    produces_tables=("analytics.type_coverage",),
    consumes_tables=("core.goids", "analytics.function_types"),
    options_model=TypeCoverageOptions,
)

# Add to class:
# - _core_metadata class variable
# - __init__ with options_resolver
# - metadata property
# - core_metadata property
# - resolve_options method
```

### 4.2 Remaining Analytics Plugins

Apply the same pattern to:

- `DocstringMetricsPlugin`
- `RiskProfilePlugin`
- `DependencyGraphPlugin`
- `HotspotAnalysisPlugin`

**Metadata naming conventions:**

| Plugin | Metadata Name |
|--------|---------------|
| `DocstringMetricsPlugin` | `analytics.docstring_metrics` |
| `RiskProfilePlugin` | `analytics.risk_profile` |
| `DependencyGraphPlugin` | `analytics.dependency_graph` |
| `HotspotAnalysisPlugin` | `analytics.hotspot_analysis` |

---

## 5. Task 2: Graphs Domain Plugins

### 5.1 ImportGraphPlugin Migration

```python
# File: src/codeintel/graphs/plugins/builders/import_graph_options.py
"""Import graph plugin options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImportGraphOptions:
    """Configuration options for import graph construction.

    Attributes
    ----------
    scope_paths : list[str] | None
        If set, only process files within these paths.
    include_stdlib : bool
        Whether to include stdlib imports in the graph.
    include_third_party : bool
        Whether to include third-party imports.
    resolve_dynamic : bool
        Whether to attempt resolution of dynamic imports.
    """

    scope_paths: list[str] | None = None
    include_stdlib: bool = False
    include_third_party: bool = True
    resolve_dynamic: bool = False


__all__ = ["ImportGraphOptions"]
```

```python
# Metadata constant for import graph
IMPORT_GRAPH_METADATA = CorePluginMetadata(
    name="graphs.import_graph",
    version="2.0.0",
    description="Build module import graph.",
    domain=PluginDomain.GRAPH,
    kind="builder",
    stage="edges",
    provides=("graph.import_graph",),
    requires=("core.modules",),
    produces_tables=(
        "graph.import_graph_nodes",
        "graph.import_graph_edges",
    ),
    consumes_tables=("core.modules",),
    scope_aware=True,
    options_model=ImportGraphOptions,
    extra={"graph_kinds": ("import_graph",)},
)
```

### 5.2 Remaining Graphs Plugins

Apply the same pattern to:

- `DataFlowGraphPlugin`
- `ControlFlowGraphPlugin`
- `SymbolUseGraphPlugin`

**Metadata naming conventions:**

| Plugin | Metadata Name |
|--------|---------------|
| `DataFlowGraphPlugin` | `graphs.dataflow` |
| `ControlFlowGraphPlugin` | `graphs.cfg` |
| `SymbolUseGraphPlugin` | `graphs.symbol_use` |

---

## 6. Task 3: Ingestion Domain Plugins

### 6.1 ModuleIngestPlugin Migration

```python
# File: src/codeintel/ingestion/plugins/modules_options.py
"""Module ingest plugin options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleIngestOptions:
    """Configuration options for module ingestion.

    Attributes
    ----------
    scope_paths : list[str] | None
        If set, only ingest modules within these paths.
    include_tests : bool
        Whether to include test modules.
    include_generated : bool
        Whether to include generated files.
    max_file_size_kb : int
        Maximum file size to ingest.
    """

    scope_paths: list[str] | None = None
    include_tests: bool = True
    include_generated: bool = False
    max_file_size_kb: int = 1024


__all__ = ["ModuleIngestOptions"]
```

```python
# Metadata constant for module ingest
MODULE_INGEST_METADATA = CorePluginMetadata(
    name="ingest.modules",
    version="2.0.0",
    description="Discover and ingest Python modules.",
    domain=PluginDomain.INGEST,
    kind="builder",
    stage="goid",
    provides=("core.modules",),
    requires=(),  # First in chain
    produces_tables=("core.modules",),
    consumes_tables=(),
    scope_aware=True,
    options_model=ModuleIngestOptions,
)
```

### 6.2 GoidBuilderPlugin Migration

```python
# File: src/codeintel/ingestion/plugins/goid_options.py
"""GOID builder plugin options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoidBuilderOptions:
    """Configuration options for GOID construction.

    Attributes
    ----------
    scope_paths : list[str] | None
        If set, only build GOIDs for files within these paths.
    include_private : bool
        Whether to include private symbols.
    extract_docstrings : bool
        Whether to extract and store docstrings.
    """

    scope_paths: list[str] | None = None
    include_private: bool = True
    extract_docstrings: bool = True


__all__ = ["GoidBuilderOptions"]
```

```python
# Metadata constant for GOID builder
GOID_BUILDER_METADATA = CorePluginMetadata(
    name="ingest.goid_builder",
    version="2.0.0",
    description="Build global object identifiers for code symbols.",
    domain=PluginDomain.INGEST,
    kind="builder",
    stage="goid",
    provides=("core.goids",),
    requires=("core.modules",),
    produces_tables=("core.goids",),
    consumes_tables=("core.modules",),
    scope_aware=True,
    options_model=GoidBuilderOptions,
)
```

### 6.3 Remaining Ingestion Plugins

Apply the same pattern to:

- `SourceFilesPlugin`
- `TreeSitterPlugin`

---

## 7. Task 4: Build All Metadata Registry

### 7.1 Create Central Metadata Index

```python
# File: src/codeintel/core/plugins/registry/all_metadata.py
"""Central registry of all plugin metadata.

This module provides a single source of truth for all plugin metadata
across all domains, enabling:
- Global capability resolution
- Dataset dependency tracking
- Plugin discovery and documentation
"""

from __future__ import annotations

from codeintel.core.plugins.registry.capability_index import (
    PluginRegistryIndex,
    build_registry_index,
)
from codeintel.core.plugins.types.metadata import CorePluginMetadata

# =============================================================================
# Import all metadata constants
# =============================================================================

# Analytics domain
from codeintel.analytics.plugins.functions.metrics import FUNCTION_METRICS_METADATA

# from codeintel.analytics.plugins.types.coverage import TYPE_COVERAGE_METADATA
# from codeintel.analytics.plugins.docs.docstrings import DOCSTRING_METRICS_METADATA
# from codeintel.analytics.plugins.risk.profile import RISK_PROFILE_METADATA
# from codeintel.analytics.plugins.deps.graph import DEPENDENCY_GRAPH_METADATA
# from codeintel.analytics.plugins.hotspots.analysis import HOTSPOT_ANALYSIS_METADATA

# Graphs domain
from codeintel.graphs.plugins.builders.callgraph import CALLGRAPH_METADATA

# from codeintel.graphs.plugins.builders.import_graph import IMPORT_GRAPH_METADATA
# from codeintel.graphs.plugins.builders.dataflow import DATAFLOW_METADATA
# from codeintel.graphs.plugins.builders.cfg import CFG_METADATA
# from codeintel.graphs.plugins.builders.symbol_use import SYMBOL_USE_METADATA

# Ingestion domain
from codeintel.ingestion.plugins.scip_plugin import SCIP_INGEST_METADATA

# from codeintel.ingestion.plugins.modules import MODULE_INGEST_METADATA
# from codeintel.ingestion.plugins.goid_builder import GOID_BUILDER_METADATA
# from codeintel.ingestion.plugins.source_files import SOURCE_FILES_METADATA
# from codeintel.ingestion.plugins.tree_sitter import TREE_SITTER_METADATA

# =============================================================================
# All Metadata Collection
# =============================================================================

# Note: Uncomment imports above as plugins are migrated

ALL_PLUGIN_METADATA: tuple[CorePluginMetadata, ...] = (
    # Analytics
    FUNCTION_METRICS_METADATA,
    # TYPE_COVERAGE_METADATA,
    # DOCSTRING_METRICS_METADATA,
    # RISK_PROFILE_METADATA,
    # DEPENDENCY_GRAPH_METADATA,
    # HOTSPOT_ANALYSIS_METADATA,
    # Graphs
    CALLGRAPH_METADATA,
    # IMPORT_GRAPH_METADATA,
    # DATAFLOW_METADATA,
    # CFG_METADATA,
    # SYMBOL_USE_METADATA,
    # Ingestion
    SCIP_INGEST_METADATA,
    # MODULE_INGEST_METADATA,
    # GOID_BUILDER_METADATA,
    # SOURCE_FILES_METADATA,
    # TREE_SITTER_METADATA,
)

# =============================================================================
# Global Index (lazy initialization)
# =============================================================================

_GLOBAL_INDEX: PluginRegistryIndex | None = None


def get_global_registry_index() -> PluginRegistryIndex:
    """Return the global plugin registry index.

    The index is built lazily on first access and cached.

    Returns
    -------
    PluginRegistryIndex
        Global index with by_name, by_capability, by_output_table lookups.
    """
    global _GLOBAL_INDEX  # noqa: PLW0603
    if _GLOBAL_INDEX is None:
        _GLOBAL_INDEX = build_registry_index(ALL_PLUGIN_METADATA)
    return _GLOBAL_INDEX


def get_provider_lookup() -> dict[str, str]:
    """Return capability → provider name lookup.

    Returns
    -------
    dict[str, str]
        Mapping of capability name to provider plugin name.
    """
    return get_global_registry_index().provider_lookup()


__all__ = [
    "ALL_PLUGIN_METADATA",
    "get_global_registry_index",
    "get_provider_lookup",
]
```

### 7.2 Test File: `tests/core/plugins/test_all_metadata.py`

```python
# File: tests/core/plugins/test_all_metadata.py
"""Tests for all_metadata registry."""

from __future__ import annotations

import pytest

from codeintel.core.plugins.registry.all_metadata import (
    ALL_PLUGIN_METADATA,
    get_global_registry_index,
    get_provider_lookup,
)


class TestAllPluginMetadata:
    """Tests for ALL_PLUGIN_METADATA collection."""

    def test_contains_spine_plugins(self) -> None:
        """Verify spine plugins are in the collection."""
        names = {m.name for m in ALL_PLUGIN_METADATA}
        assert "analytics.function_metrics" in names
        assert "graphs.callgraph" in names
        assert "ingest.scip_python" in names

    def test_all_metadata_has_required_fields(self) -> None:
        """Verify all metadata has required fields."""
        for meta in ALL_PLUGIN_METADATA:
            assert meta.name
            assert meta.version
            assert meta.description
            assert meta.domain
            assert meta.kind

    def test_no_duplicate_names(self) -> None:
        """Verify no duplicate plugin names."""
        names = [m.name for m in ALL_PLUGIN_METADATA]
        assert len(names) == len(set(names)), "Duplicate plugin names found"


class TestGlobalRegistryIndex:
    """Tests for get_global_registry_index."""

    def test_index_contains_all_plugins(self) -> None:
        """Verify index contains all plugins."""
        index = get_global_registry_index()
        for meta in ALL_PLUGIN_METADATA:
            assert index.get_by_name(meta.name) is not None

    def test_capabilities_are_indexed(self) -> None:
        """Verify capabilities are properly indexed."""
        index = get_global_registry_index()
        # Function metrics provides analytics.function_metrics
        provider = index.get_provider("analytics.function_metrics")
        assert provider is not None
        assert provider.name == "analytics.function_metrics"

    def test_tables_are_indexed(self) -> None:
        """Verify tables are properly indexed."""
        index = get_global_registry_index()
        # Callgraph produces graph.call_graph_edges
        producer = index.get_producer("graph.call_graph_edges")
        assert producer is not None
        assert producer.name == "graphs.callgraph"


class TestProviderLookup:
    """Tests for get_provider_lookup."""

    def test_returns_mapping(self) -> None:
        """Verify provider lookup returns capability → name mapping."""
        lookup = get_provider_lookup()
        assert isinstance(lookup, dict)
        assert "analytics.function_metrics" in lookup
        assert lookup["analytics.function_metrics"] == "analytics.function_metrics"
```

---

## 8. Verification

### 8.1 Run Quality Checks

```bash
# Format and lint all domain plugins
uv run ruff format src/codeintel/analytics/plugins/ src/codeintel/graphs/plugins/ src/codeintel/ingestion/plugins/
uv run ruff check --fix src/codeintel/analytics/ src/codeintel/graphs/ src/codeintel/ingestion/

# Type checking
uv run pyright src/codeintel/analytics/plugins/ src/codeintel/graphs/plugins/ src/codeintel/ingestion/plugins/

# Pyrefly
uv run pyrefly check src/codeintel/
```

### 8.2 Run Tests

```bash
# Run all plugin metadata tests
uv run pytest tests/analytics/plugins/ tests/graphs/plugins/ tests/ingestion/plugins/ -v -k metadata

# Run registry tests
uv run pytest tests/core/plugins/test_all_metadata.py -v

# Run full plugin test suite
uv run pytest tests/analytics/ tests/graphs/ tests/ingestion/ -v
```

### 8.3 Verification Checklist

- [ ] All plugins have `_core_metadata` class variable
- [ ] All plugins have `metadata` property returning `PluginMetadata`
- [ ] All plugins have `core_metadata` property returning `CorePluginMetadata`
- [ ] All configurable plugins have options models
- [ ] All plugins are in `ALL_PLUGIN_METADATA` tuple
- [ ] Global registry index builds successfully
- [ ] All capabilities are indexed
- [ ] All tables are indexed
- [ ] No duplicate plugin names

---

## 9. Rollback Plan

Phase 3 changes are backward-compatible. To rollback:

1. **Revert plugin files** to their Phase 2 state
2. **Delete new options files** created in Phase 3
3. **Revert `all_metadata.py`** to Phase 2 state (only spine plugins)
4. **Delete Phase 3 test files**

---

## Appendix A: Plugin Migration Checklist Template

For each plugin migration, complete this checklist:

```markdown
## Plugin: [plugin_name]

### Pre-Migration
- [ ] Read current plugin implementation
- [ ] Identify configurable options
- [ ] Identify capabilities (provides/requires)
- [ ] Identify tables (produces/consumes)

### Migration
- [ ] Create options model (if needed)
- [ ] Define `CorePluginMetadata` constant
- [ ] Add `_core_metadata` class variable
- [ ] Add `__init__` with `options_resolver` parameter
- [ ] Add `metadata` property
- [ ] Add `core_metadata` property
- [ ] Add `resolve_options` method
- [ ] Update execute to use resolved options
- [ ] Export metadata constant in `__all__`

### Post-Migration
- [ ] Add to `ALL_PLUGIN_METADATA`
- [ ] Create metadata test file
- [ ] Run quality checks
- [ ] Run existing tests
- [ ] Run new tests
```

---

**Next Steps**: After Phase 3 is complete, proceed to Phase 4 (Profile Integration) to wire up the ProfiledConfigSource and define fast/full profiles.
