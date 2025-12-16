# Phase 2: Spine Plugin Migration Implementation Plan

> **Scope**: Attach CorePluginMetadata and PluginOptionsResolver to representative plugins
> **Duration**: 2-3 days
> **Risk Level**: Low-Medium (extends existing plugins without changing behavior)
> **Depends On**: Phase 1 (Core Infrastructure)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Task 1: Enhanced Options Models](#3-task-1-enhanced-options-models)
4. [Task 2: Analytics Plugin Migration](#4-task-2-analytics-plugin-migration)
5. [Task 3: Graphs Plugin Migration](#5-task-3-graphs-plugin-migration)
6. [Task 4: Ingestion Plugin Migration](#6-task-4-ingestion-plugin-migration)
7. [Task 5: Plugin Run Context Integration](#7-task-5-plugin-run-context-integration)
8. [Verification](#8-verification)
9. [Rollback Plan](#9-rollback-plan)

---

## 1. Overview

Phase 2 migrates three "spine" plugins—one from each major domain—to use the new unified data abstraction infrastructure. These serve as reference implementations for the remaining plugins.

### Target Plugins

| Domain | Plugin | Current File |
|--------|--------|--------------|
| Analytics | `FunctionMetricsPlugin` | `analytics/plugins/functions/metrics.py` |
| Graphs | `CallGraphPlugin` | `graphs/plugins/builders/callgraph.py` |
| Ingestion | `ScipIngestPlugin` | `ingestion/plugins/scip_plugin.py` |

### Migration Pattern

Each plugin migration follows the same pattern:

1. **Enhance Options Model**: Add config-driven fields (profile-sensitive settings)
2. **Create Metadata Constant**: Define `CorePluginMetadata` with full specification
3. **Add Options Resolution**: Use `PluginOptionsResolver` to get typed options
4. **Wire Metadata Property**: Return `CorePluginMetadata` as `PluginMetadata` facade
5. **Add Tests**: Verify metadata, options resolution, and existing behavior

### Key Principle: No Behavior Changes

All changes in this phase are **structural only**. Plugin execution behavior must remain identical. The new metadata and options infrastructure runs alongside existing code.

---

## 2. Prerequisites

Verify Phase 1 infrastructure is complete:

```bash
# Verify Phase 1 types are available
uv run python -c "
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.execution.options import PluginOptionsResolver, EmptyConfigSource
from codeintel.core.plugins.registry.capability_index import build_registry_index
print('Phase 1 infrastructure verified')
"

# Run Phase 1 tests
uv run pytest tests/core/plugins/test_metadata.py tests/core/plugins/test_options.py -v
```

---

## 3. Task 1: Enhanced Options Models

### 3.1 Enhance `FunctionAnalyticsOptions`

Add profile-driven configuration fields to the existing options model.

```python
# File: src/codeintel/analytics/functions/config.py
# Replace the existing FunctionAnalyticsOptions with this enhanced version

"""Configuration helpers for function analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.analytics.parsing.ast_cache import FunctionAst
    from codeintel.analytics.parsing.models import ParsedModule, SourceSpan
    from codeintel.analytics.parsing.validation import FunctionValidationReporter
    from codeintel.config import FunctionAnalyticsStepConfig


@dataclass(frozen=True)
class ProcessContext:
    """Shared context for building analytics rows."""

    cfg: FunctionAnalyticsStepConfig
    now: datetime


@dataclass(frozen=True)
class FunctionAnalyticsOptions:
    """Configuration options for function analytics computation.

    This dataclass serves as the typed options model for the function_metrics
    plugin. It contains both:
    - Config-driven fields (settable via profiles/config files)
    - Dynamic fields (set at runtime, e.g., AST caches)

    Config-Driven Fields
    --------------------
    These can be set via configuration files or profile overrides:

    include_graph_metrics : bool
        Whether to compute graph-derived metrics (PageRank, centrality).
        Set to False in "fast" profile to skip expensive graph queries.
    include_coverage_metrics : bool
        Whether to join coverage data for functions.
        Set to False in "fast" profile if coverage not available.
    complexity_threshold : int
        Maximum cyclomatic complexity before flagging as too complex.
    type_strictness : str
        Type checking strictness level ("strict", "standard", "lenient").
    scope_paths : list[str] | None
        If set, only process functions in these paths.

    Dynamic Fields
    --------------
    These are set at execution time, not from configuration:

    validation_reporter : FunctionValidationReporter | None
        Optional reporter for validation issues.
    function_ast_map : dict[int, FunctionAst] | None
        Pre-built AST map from AstProvider.
    missing_function_goids : set[int]
        GOIDs that could not be parsed.

    Examples
    --------
    Default options (full computation):

    >>> opts = FunctionAnalyticsOptions()
    >>> opts.include_graph_metrics
    True
    >>> opts.include_coverage_metrics
    True

    Fast profile options:

    >>> opts = FunctionAnalyticsOptions(
    ...     include_graph_metrics=False,
    ...     include_coverage_metrics=False,
    ... )
    """

    # === Config-driven fields (from profiles/config) ===
    include_graph_metrics: bool = True
    include_coverage_metrics: bool = True
    complexity_threshold: int = 10
    type_strictness: str = "standard"
    scope_paths: list[str] | None = None

    # === Dynamic fields (set at runtime) ===
    validation_reporter: FunctionValidationReporter | None = None
    function_ast_map: dict[int, FunctionAst] | None = None
    missing_function_goids: set[int] = field(default_factory=set)

    def get_ast_map(self) -> dict[int, FunctionAst]:
        """Return the function AST map.

        Returns
        -------
        dict[int, FunctionAst]
            The AST map, empty if not provided.
        """
        if self.function_ast_map is not None:
            return self.function_ast_map
        return {}

    def get_missing_goids(self) -> set[int]:
        """Return the set of missing GOIDs.

        Returns
        -------
        set[int]
            The missing GOIDs set.
        """
        return self.missing_function_goids

    def has_ast_data(self) -> bool:
        """Check if AST data is available.

        Returns
        -------
        bool
            True if AST data is available.
        """
        return self.function_ast_map is not None

    def should_compute_graph_metrics(self) -> bool:
        """Check if graph metrics should be computed.

        Returns
        -------
        bool
            True if graph metrics are enabled.
        """
        return self.include_graph_metrics

    def should_compute_coverage_metrics(self) -> bool:
        """Check if coverage metrics should be computed.

        Returns
        -------
        bool
            True if coverage metrics are enabled.
        """
        return self.include_coverage_metrics


@dataclass
class ProcessState:
    """Mutable state shared across per-file processing."""

    cfg: FunctionAnalyticsStepConfig
    cache: dict[str, ParsedModule | None]
    span_index: dict[int, SourceSpan]
    reporter: FunctionValidationReporter
    ctx: ProcessContext
```

### 3.2 Create `CallGraphOptions`

Create a new options model for the call graph plugin.

```python
# File: src/codeintel/graphs/plugins/builders/callgraph_options.py
"""Call graph plugin options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CallGraphOptions:
    """Configuration options for call graph construction.

    This dataclass serves as the typed options model for the callgraph
    plugin. It contains profile-driven configuration fields.

    Attributes
    ----------
    scope_paths : list[str] | None
        If set, only process files within these paths.
        Enables scoped execution for large repositories.
    max_edges_per_file : int
        Maximum number of edges to collect per file.
        Prevents runaway processing on pathological files.
    include_external_calls : bool
        Whether to include calls to external (stdlib/third-party) functions.
        Set to False in "fast" profile to reduce edge count.
    resolve_imports : bool
        Whether to resolve import aliases for better callee matching.
        Set to False in "fast" profile for speed.
    use_libcst : bool
        Whether to prefer LibCST over AST for parsing.
        LibCST provides better positions but is slower.

    Examples
    --------
    Default options (full analysis):

    >>> opts = CallGraphOptions()
    >>> opts.include_external_calls
    True
    >>> opts.resolve_imports
    True

    Fast profile options:

    >>> opts = CallGraphOptions(
    ...     include_external_calls=False,
    ...     resolve_imports=False,
    ...     use_libcst=False,
    ... )
    """

    # === Scope control ===
    scope_paths: list[str] | None = None

    # === Processing limits ===
    max_edges_per_file: int = 10000

    # === Analysis depth controls ===
    include_external_calls: bool = True
    resolve_imports: bool = True
    use_libcst: bool = True


__all__ = ["CallGraphOptions"]
```

### 3.3 Create `ScipIngestOptions`

Create a new options model for the SCIP ingest plugin.

```python
# File: src/codeintel/ingestion/plugins/scip_options.py
"""SCIP ingest plugin options."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScipIngestOptions:
    """Configuration options for SCIP indexing.

    This dataclass serves as the typed options model for the scip_ingest
    plugin. It contains profile-driven configuration fields.

    Attributes
    ----------
    scope_paths : list[str] | None
        If set, only index files within these paths.
        Enables scoped execution for large repositories.
    include_references : bool
        Whether to include reference information.
        Set to False in "fast" profile to reduce index size.
    include_implementations : bool
        Whether to include implementation relationships.
        Set to False in "fast" profile for speed.
    max_file_size_kb : int
        Maximum file size to index (in KB).
        Files larger than this are skipped.
    timeout_seconds : int
        Maximum time for the scip-python process.
    scip_output_dir : Path | None
        Directory for SCIP output files (runtime override).

    Examples
    --------
    Default options (full indexing):

    >>> opts = ScipIngestOptions()
    >>> opts.include_references
    True
    >>> opts.include_implementations
    True

    Fast profile options:

    >>> opts = ScipIngestOptions(
    ...     include_references=False,
    ...     include_implementations=False,
    ...     timeout_seconds=120,
    ... )
    """

    # === Scope control ===
    scope_paths: list[str] | None = None

    # === Index content controls ===
    include_references: bool = True
    include_implementations: bool = True

    # === Processing limits ===
    max_file_size_kb: int = 1024
    timeout_seconds: int = 300

    # === Runtime overrides (not from config) ===
    scip_output_dir: Path | None = None

    def should_include_references(self) -> bool:
        """Check if references should be included.

        Returns
        -------
        bool
            True if references are enabled.
        """
        return self.include_references

    def should_include_implementations(self) -> bool:
        """Check if implementations should be included.

        Returns
        -------
        bool
            True if implementations are enabled.
        """
        return self.include_implementations


__all__ = ["ScipIngestOptions"]
```

---

## 4. Task 2: Analytics Plugin Migration

### 4.1 Update `FunctionMetricsPlugin`

```python
# File: src/codeintel/analytics/plugins/functions/metrics.py
"""Function metrics plugin.

This plugin computes function complexity and type coverage metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from codeintel.analytics.functions import (
    FunctionAnalyticsOptions,
    compute_function_metrics_and_types,
)
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import FunctionAnalyticsStepConfig
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.types.protocol import PluginMetadata

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.execution.options import PluginOptionsResolver

# =============================================================================
# Metadata Constants
# =============================================================================

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
    requires=("core.goids",),
    produces_tables=(
        "analytics.function_metrics",
        "analytics.function_types",
    ),
    consumes_tables=("core.goids",),
    supports_incremental=False,
    scope_aware=False,
    options_model=FunctionAnalyticsOptions,
    resource_hints={"max_memory_mb": 512},
)


def _to_plugin_metadata(core: CorePluginMetadata) -> PluginMetadata:
    """Convert CorePluginMetadata to PluginMetadata for protocol compliance.

    Parameters
    ----------
    core
        Core metadata instance.

    Returns
    -------
    PluginMetadata
        Protocol-compatible metadata.
    """
    return PluginMetadata(
        name=core.name,
        version=core.version,
        description=core.description,
        kind=core.kind,
        stage=core.stage or "function",
        provides=core.provides,
        requires=core.requires,
        produces_tables=core.produces_tables,
    )


# =============================================================================
# Plugin Implementation
# =============================================================================


class FunctionMetricsPlugin(TargetPlugin):
    """Compute function complexity and type coverage metrics.

    This plugin uses the unified data abstraction infrastructure:
    - CorePluginMetadata for capability/dataset declaration
    - PluginOptionsResolver for profile-driven configuration

    Output Tables
    -------------
    - analytics.function_metrics: Complexity and size metrics
    - analytics.function_types: Type annotation data
    """

    plugin_name: ClassVar[str] = "function_metrics"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Compute function complexity and type coverage metrics."

    # New: Reference to core metadata
    _core_metadata: ClassVar[CorePluginMetadata] = FUNCTION_METRICS_METADATA

    def __init__(
        self,
        *,
        options_resolver: PluginOptionsResolver | None = None,
    ) -> None:
        """Initialize plugin with optional options resolver.

        Parameters
        ----------
        options_resolver
            Resolver for typed configuration options. If None, uses
            default values from FunctionAnalyticsOptions.
        """
        self._options_resolver = options_resolver

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata.

        Returns
        -------
        PluginMetadata
            Metadata describing the plugin.
        """
        return _to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return full core metadata.

        Returns
        -------
        CorePluginMetadata
            Full core metadata with domain/options model.
        """
        return self._core_metadata

    def resolve_options(
        self,
        *,
        dynamic_overrides: Mapping[str, Any] | None = None,
    ) -> FunctionAnalyticsOptions:
        """Resolve typed options from configuration.

        Parameters
        ----------
        dynamic_overrides
            Runtime-only overrides (AST maps, etc.).

        Returns
        -------
        FunctionAnalyticsOptions
            Resolved options instance.
        """
        if self._options_resolver is None:
            # No resolver: use defaults + dynamic overrides
            if dynamic_overrides:
                return FunctionAnalyticsOptions(**dynamic_overrides)
            return FunctionAnalyticsOptions()

        return self._options_resolver.get_options(
            self._core_metadata,
            FunctionAnalyticsOptions,
            dynamic_overrides=dynamic_overrides,
        )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute function metrics computation.

        Parameters
        ----------
        ctx
            Execution context providing gateway and parameters.

        Returns
        -------
        TargetResult
            Success result with row counts.
        """
        # Get AST data from catalog if available
        function_ast_map = None
        missing_function_goids: set[int] = set()

        # Resolve options with dynamic overrides
        opts = self.resolve_options(
            dynamic_overrides={
                "function_ast_map": function_ast_map,
                "missing_function_goids": missing_function_goids,
            }
        )

        # Build config from parameters
        cfg = FunctionAnalyticsStepConfig(
            snapshot=ctx.snapshot,
        )

        result = compute_function_metrics_and_types(ctx.gateway, cfg, options=opts)

        return TargetResult.succeeded(
            row_counts={
                "analytics.function_metrics": result.get("metrics_rows", 0),
                "analytics.function_types": result.get("types_rows", 0),
            }
        )


__all__ = [
    "FUNCTION_METRICS_METADATA",
    "FunctionMetricsPlugin",
]
```

### 4.2 Test File: `tests/analytics/plugins/test_function_metrics_metadata.py`

```python
# File: tests/analytics/plugins/test_function_metrics_metadata.py
"""Tests for FunctionMetricsPlugin metadata and options integration."""

from __future__ import annotations

from typing import Any

import pytest

from codeintel.analytics.functions.config import FunctionAnalyticsOptions
from codeintel.analytics.plugins.functions.metrics import (
    FUNCTION_METRICS_METADATA,
    FunctionMetricsPlugin,
)
from codeintel.core.plugins.execution.options import (
    ConfigSource,
    EmptyConfigSource,
    PluginOptionsResolver,
)
from codeintel.core.plugins.types.metadata import PluginDomain


class DictConfigSource(ConfigSource):
    """Test config source backed by a dict."""

    def __init__(self, options: dict[str, dict[str, Any]]) -> None:
        self._options = options

    def get_plugin_options(self, plugin_name: str) -> dict[str, Any] | None:
        return self._options.get(plugin_name)


class TestFunctionMetricsMetadata:
    """Tests for FUNCTION_METRICS_METADATA constant."""

    def test_metadata_name(self) -> None:
        """Verify metadata name is canonical."""
        assert FUNCTION_METRICS_METADATA.name == "analytics.function_metrics"

    def test_metadata_domain(self) -> None:
        """Verify metadata domain is analytics."""
        assert FUNCTION_METRICS_METADATA.domain == PluginDomain.ANALYTICS

    def test_metadata_kind_and_stage(self) -> None:
        """Verify kind and stage."""
        assert FUNCTION_METRICS_METADATA.kind == "metric"
        assert FUNCTION_METRICS_METADATA.stage == "function"

    def test_metadata_capabilities(self) -> None:
        """Verify provides and requires."""
        assert "analytics.function_metrics" in FUNCTION_METRICS_METADATA.provides
        assert "analytics.function_types" in FUNCTION_METRICS_METADATA.provides
        assert "core.goids" in FUNCTION_METRICS_METADATA.requires

    def test_metadata_tables(self) -> None:
        """Verify produces_tables."""
        assert "analytics.function_metrics" in FUNCTION_METRICS_METADATA.produces_tables
        assert "analytics.function_types" in FUNCTION_METRICS_METADATA.produces_tables
        assert "core.goids" in FUNCTION_METRICS_METADATA.consumes_tables

    def test_metadata_has_options_model(self) -> None:
        """Verify options_model is set."""
        assert FUNCTION_METRICS_METADATA.has_options
        assert FUNCTION_METRICS_METADATA.options_model is FunctionAnalyticsOptions


class TestFunctionMetricsPluginOptionsIntegration:
    """Tests for options resolution integration."""

    def test_default_options_without_resolver(self) -> None:
        """Verify default options when no resolver provided."""
        plugin = FunctionMetricsPlugin()
        opts = plugin.resolve_options()
        assert opts.include_graph_metrics is True
        assert opts.include_coverage_metrics is True

    def test_options_with_empty_resolver(self) -> None:
        """Verify options with empty config source."""
        resolver = PluginOptionsResolver(EmptyConfigSource())
        plugin = FunctionMetricsPlugin(options_resolver=resolver)
        opts = plugin.resolve_options()
        assert opts.include_graph_metrics is True  # Default value

    def test_options_with_config_override(self) -> None:
        """Verify config values override defaults."""
        source = DictConfigSource({
            "analytics.function_metrics": {
                "include_graph_metrics": False,
                "complexity_threshold": 15,
            },
        })
        resolver = PluginOptionsResolver(source)
        plugin = FunctionMetricsPlugin(options_resolver=resolver)
        opts = plugin.resolve_options()
        assert opts.include_graph_metrics is False
        assert opts.complexity_threshold == 15
        assert opts.include_coverage_metrics is True  # Still default

    def test_dynamic_overrides(self) -> None:
        """Verify dynamic overrides are applied."""
        plugin = FunctionMetricsPlugin()
        ast_map = {123: object()}  # Fake AST map
        opts = plugin.resolve_options(
            dynamic_overrides={"function_ast_map": ast_map}
        )
        assert opts.function_ast_map is ast_map


class TestFunctionMetricsPluginMetadataProperty:
    """Tests for plugin.metadata property."""

    def test_metadata_property_returns_plugin_metadata(self) -> None:
        """Verify metadata property returns PluginMetadata type."""
        plugin = FunctionMetricsPlugin()
        meta = plugin.metadata
        assert meta.name == "analytics.function_metrics"
        assert meta.version == "3.0.0"
        assert meta.kind == "metric"

    def test_core_metadata_property(self) -> None:
        """Verify core_metadata returns CorePluginMetadata."""
        plugin = FunctionMetricsPlugin()
        core = plugin.core_metadata
        assert core is FUNCTION_METRICS_METADATA
        assert core.domain == PluginDomain.ANALYTICS
```

---

## 5. Task 3: Graphs Plugin Migration

### 5.1 Update `CallGraphPlugin`

```python
# File: src/codeintel/graphs/plugins/builders/callgraph.py
"""Call graph builder plugin.

This module provides the call graph builder as a build target plugin.

Architecture
------------
The call graph plugin performs the following steps:

1. Load function spans from `core.goids` to build FunctionSpanIndex
2. Build global callee map (qualname -> GOID) for resolution
3. Build module GOID map (path -> module GOID) for SCIP fallback
4. For each Python file with functions:
   - Build local callee map from function index
   - Collect import aliases from the file
   - Create EdgeResolutionContext with all lookup maps
   - Parse file and collect call edges via LibCST (or AST fallback)
5. Persist deduplicated edges to graph.call_graph_edges
6. Persist nodes to graph.call_graph_nodes
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import libcst as cst

from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config import CallGraphStepConfig
from codeintel.config.datasets import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    call_graph_edge_to_tuple,
    call_graph_node_to_tuple,
)
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.types.protocol import PluginMetadata
from codeintel.graphs.adapters.callgraph_persistence import dedupe_edge_rows
from codeintel.graphs.catalog import (
    FunctionSpanIndex,
    load_function_index,
)
from codeintel.graphs.compute.callgraph import (
    EdgeResolutionContext,
    collect_aliases,
    collect_edges_ast,
    collect_edges_cst,
)
from codeintel.graphs.plugins.builders.callgraph_options import CallGraphOptions
from codeintel.ingestion.adapters import IngestStorageService
from codeintel.ingestion.infrastructure.paths import normalize_rel_path

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.execution.options import PluginOptionsResolver
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

# =============================================================================
# Metadata Constants
# =============================================================================

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
    scope_aware=True,  # Reacts to scope_paths
    options_model=CallGraphOptions,
    resource_hints={"max_memory_mb": 1024},
    extra={"graph_kinds": ("call_graph",)},
)


def _to_plugin_metadata(core: CorePluginMetadata) -> PluginMetadata:
    """Convert CorePluginMetadata to PluginMetadata for protocol compliance."""
    return PluginMetadata(
        name=core.name,
        version=core.version,
        description=core.description,
        kind=core.kind,
        stage=core.stage or "edges",
        provides=core.provides,
        requires=core.requires,
        produces_tables=core.produces_tables,
    )


# =============================================================================
# Helper Functions (unchanged from original)
# =============================================================================


def _log_repo_state(gateway: StorageGateway, repo: str, commit: str) -> None:
    """Log current module/GOID counts to aid validation diagnostics."""
    con = gateway.con
    try:
        modules = con.execute(
            "SELECT COUNT(*) FROM core.modules WHERE repo = ? AND commit = ?",
            [repo, commit],
        ).fetchone()
        goids = con.execute(
            "SELECT COUNT(*) FROM core.goids WHERE repo = ? AND commit = ?",
            [repo, commit],
        ).fetchone()
        module_goids = con.execute(
            "SELECT COUNT(*) FROM core.goids WHERE repo = ? AND commit = ? AND kind = 'module'",
            [repo, commit],
        ).fetchone()
        log.info(
            "call_graph_builder repo_state modules=%d goids=%d (module_kind=%d)",
            modules[0] if modules else 0,
            goids[0] if goids else 0,
            module_goids[0] if module_goids else 0,
        )
    except Exception:  # noqa: BLE001
        log.debug("call_graph_builder: Could not query repo state")


def _build_global_callee_lookup(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, int]:
    """Build a lookup mapping qualnames to function GOIDs."""
    con = gateway.con
    try:
        rows = con.execute(
            """
            SELECT qualname, goid_h128
            FROM core.goids
            WHERE repo = ? AND commit = ? AND kind IN ('function', 'method')
            """,
            [repo, commit],
        ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}
    except Exception:  # noqa: BLE001
        return {}


def _build_def_goids_by_path(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, int]:
    """Build lookup of module GOIDs by path."""
    con = gateway.con
    try:
        rows = con.execute(
            """
            SELECT rel_path, goid_h128
            FROM core.goids
            WHERE repo = ? AND commit = ? AND kind = 'module'
            """,
            [repo, commit],
        ).fetchall()
        return {normalize_rel_path(str(row[0])): int(row[1]) for row in rows}
    except Exception:  # noqa: BLE001
        return {}


def _get_source_root(gateway: StorageGateway, repo: str, commit: str) -> Path | None:
    """Retrieve source root from core.snapshots."""
    con = gateway.con
    try:
        row = con.execute(
            "SELECT source_root FROM core.snapshots WHERE repo = ? AND commit = ?",
            [repo, commit],
        ).fetchone()
        if row and row[0]:
            return Path(row[0])
    except Exception as e:  # noqa: BLE001
        log.debug("callgraph: Could not get source root: %s", e)
    return None


def _collect_edges_for_file(
    rel_path: str,
    file_path: Path,
    context: EdgeResolutionContext,
    *,
    use_libcst: bool = True,
) -> list[CallGraphEdgeRow]:
    """Collect call edges for a single Python file."""
    if not file_path.exists():
        return []

    try:
        source = file_path.read_text(encoding="utf8")
    except (OSError, UnicodeDecodeError):
        return []

    # Try LibCST first (more accurate positions) if enabled
    if use_libcst:
        try:
            module = cst.parse_module(source)
            return collect_edges_cst(rel_path, module, context)
        except cst.ParserSyntaxError:
            pass

    # Fall back to AST
    return collect_edges_ast(rel_path, file_path, context)


def _build_nodes_from_goids(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> list[CallGraphNodeRow]:
    """Build call graph node rows from function GOIDs."""
    con = gateway.con
    try:
        rows = con.execute(
            """
            SELECT
                goid_h128,
                COALESCE(language, 'python') AS language,
                kind,
                rel_path
            FROM core.goids
            WHERE repo = ? AND commit = ? AND kind IN ('function', 'method')
            """,
            [repo, commit],
        ).fetchall()
    except Exception as e:  # noqa: BLE001
        log.debug("callgraph: Could not build nodes from GOIDs: %s", e)
        return []

    return [
        CallGraphNodeRow(
            goid_h128=int(goid_h128),
            language=str(language) if language else "python",
            kind=str(kind),
            arity=0,
            is_public=True,
            rel_path=str(rel_path),
        )
        for goid_h128, language, kind, rel_path in rows
    ]


def _persist_nodes(
    gateway: StorageGateway,
    nodes: list[CallGraphNodeRow],
    repo: str,
    commit: str,
) -> int:
    """Persist call graph nodes."""
    if not nodes:
        return 0

    storage = IngestStorageService.from_gateway(gateway)
    storage.run_batch(
        "graph.call_graph_nodes",
        [call_graph_node_to_tuple(node) for node in nodes],
        delete_params=[repo, commit],
        scope="call_graph_nodes",
    )
    return len(nodes)


def _persist_edges(
    gateway: StorageGateway,
    edges: list[CallGraphEdgeRow],
    repo: str,
    commit: str,
) -> int:
    """Persist call graph edges after deduplication."""
    if not edges:
        return 0

    unique_edges = dedupe_edge_rows(edges)

    serialized: list[CallGraphEdgeRow] = []
    for edge in unique_edges:
        evidence = edge["evidence_json"]
        if isinstance(evidence, dict):
            serialized.append({**edge, "evidence_json": json.dumps(evidence)})
        else:
            serialized.append(edge)

    storage = IngestStorageService.from_gateway(gateway)
    storage.run_batch(
        "graph.call_graph_edges",
        [call_graph_edge_to_tuple(e) for e in serialized],
        delete_params=[repo, commit],
        scope="call_graph_edges",
    )
    return len(serialized)


@dataclass(frozen=True)
class _EdgeCollectionContext:
    """Context for edge collection across files."""

    function_index: FunctionSpanIndex
    global_callees: dict[str, int]
    def_goids_by_path: dict[str, int]
    source_root: Path
    repo: str
    commit: str
    use_libcst: bool = True
    resolve_imports: bool = True


def _filter_paths_by_scope(
    paths: list[str],
    scope_paths: list[str] | None,
) -> list[str]:
    """Filter paths by scope if scope_paths is set.

    Parameters
    ----------
    paths
        All available paths.
    scope_paths
        Scope filter paths, or None for no filtering.

    Returns
    -------
    list[str]
        Filtered paths.
    """
    if not scope_paths:
        return paths

    scope_prefixes = tuple(scope_paths)
    return [p for p in paths if p.startswith(scope_prefixes)]


def _collect_all_edges(
    paths: list[str],
    ctx: _EdgeCollectionContext,
) -> list[CallGraphEdgeRow]:
    """Collect edges from all files with functions."""
    all_edges: list[CallGraphEdgeRow] = []

    for rel_path in paths:
        local_callees = ctx.function_index.local_name_map(rel_path)
        file_path = ctx.source_root / rel_path
        import_aliases: dict[str, str] = {}
        if ctx.resolve_imports and file_path.exists():
            with contextlib.suppress(OSError, UnicodeDecodeError, cst.ParserSyntaxError):
                import_aliases = collect_aliases(
                    cst.parse_module(file_path.read_text(encoding="utf8"))
                )

        context = EdgeResolutionContext(
            repo=ctx.repo,
            commit=ctx.commit,
            function_index=ctx.function_index,
            local_callees=local_callees,
            global_callees=ctx.global_callees,
            import_aliases=import_aliases,
            scip_candidates_by_use_path={},
            def_goids_by_path=ctx.def_goids_by_path,
        )
        all_edges.extend(
            _collect_edges_for_file(rel_path, file_path, context, use_libcst=ctx.use_libcst)
        )

    return all_edges


# =============================================================================
# Plugin Implementation
# =============================================================================


class CallGraphPlugin(TargetPlugin):
    """Build call graph nodes and edges.

    This plugin uses the unified data abstraction infrastructure:
    - CorePluginMetadata for capability/dataset declaration
    - PluginOptionsResolver for profile-driven configuration

    Outputs
    -------
    - graph.call_graph_nodes: Call graph nodes
    - graph.call_graph_edges: Call graph edges
    """

    plugin_name: ClassVar[str] = "callgraph"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Build call graph nodes and edges."

    _core_metadata: ClassVar[CorePluginMetadata] = CALLGRAPH_METADATA

    def __init__(
        self,
        *,
        options_resolver: PluginOptionsResolver | None = None,
    ) -> None:
        """Initialize plugin with optional options resolver.

        Parameters
        ----------
        options_resolver
            Resolver for typed configuration options.
        """
        self._options_resolver = options_resolver

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return _to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return full core metadata."""
        return self._core_metadata

    def resolve_options(
        self,
        *,
        dynamic_overrides: Mapping[str, Any] | None = None,
    ) -> CallGraphOptions:
        """Resolve typed options from configuration.

        Parameters
        ----------
        dynamic_overrides
            Runtime-only overrides.

        Returns
        -------
        CallGraphOptions
            Resolved options instance.
        """
        if self._options_resolver is None:
            if dynamic_overrides:
                return CallGraphOptions(**dynamic_overrides)
            return CallGraphOptions()

        return self._options_resolver.get_options(
            self._core_metadata,
            CallGraphOptions,
            dynamic_overrides=dynamic_overrides,
        )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute call graph construction."""
        # Resolve options
        opts = self.resolve_options()

        cfg = CallGraphStepConfig(snapshot=ctx.snapshot)
        gateway, repo, commit = ctx.gateway, cfg.repo, cfg.commit

        try:
            _log_repo_state(gateway, repo, commit)

            # Load function index and get paths
            function_index = load_function_index(gateway, repo=repo, commit=commit)
            paths = function_index.paths()

            # Apply scope filtering if configured
            paths = _filter_paths_by_scope(paths, opts.scope_paths)

            if not paths:
                log.info("callgraph: No functions found, skipping")
                return TargetResult.succeeded(
                    row_counts={"graph.call_graph_nodes": 0, "graph.call_graph_edges": 0}
                )

            # Build lookup maps
            global_callees = _build_global_callee_lookup(gateway, repo, commit)
            def_goids = _build_def_goids_by_path(gateway, repo, commit)
            source_root = (
                ctx.snapshot.repo_root or _get_source_root(gateway, repo, commit) or Path.cwd()
            )

            # Collect and persist edges with options-driven behavior
            collection_ctx = _EdgeCollectionContext(
                function_index=function_index,
                global_callees=global_callees,
                def_goids_by_path=def_goids,
                source_root=source_root,
                repo=repo,
                commit=commit,
                use_libcst=opts.use_libcst,
                resolve_imports=opts.resolve_imports,
            )
            edges = _collect_all_edges(paths, collection_ctx)
            log.info("callgraph: Collected %d edges from %d files", len(edges), len(paths))

            # Build and persist nodes
            node_count = _persist_nodes(
                gateway, _build_nodes_from_goids(gateway, repo, commit), repo, commit
            )
            edge_count = _persist_edges(gateway, edges, repo, commit)

            log.info("callgraph: Persisted %d nodes, %d edges", node_count, edge_count)
            return TargetResult.succeeded(
                row_counts={
                    "graph.call_graph_nodes": node_count,
                    "graph.call_graph_edges": edge_count,
                }
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Call graph build failed: {e}")


__all__ = [
    "CALLGRAPH_METADATA",
    "CallGraphPlugin",
]
```

### 5.2 Test File: `tests/graphs/plugins/test_callgraph_metadata.py`

```python
# File: tests/graphs/plugins/test_callgraph_metadata.py
"""Tests for CallGraphPlugin metadata and options integration."""

from __future__ import annotations

from typing import Any

import pytest

from codeintel.core.plugins.execution.options import (
    ConfigSource,
    EmptyConfigSource,
    PluginOptionsResolver,
)
from codeintel.core.plugins.types.metadata import PluginDomain
from codeintel.graphs.plugins.builders.callgraph import (
    CALLGRAPH_METADATA,
    CallGraphPlugin,
)
from codeintel.graphs.plugins.builders.callgraph_options import CallGraphOptions


class DictConfigSource(ConfigSource):
    """Test config source backed by a dict."""

    def __init__(self, options: dict[str, dict[str, Any]]) -> None:
        self._options = options

    def get_plugin_options(self, plugin_name: str) -> dict[str, Any] | None:
        return self._options.get(plugin_name)


class TestCallGraphMetadata:
    """Tests for CALLGRAPH_METADATA constant."""

    def test_metadata_name(self) -> None:
        """Verify metadata name is canonical."""
        assert CALLGRAPH_METADATA.name == "graphs.callgraph"

    def test_metadata_domain(self) -> None:
        """Verify metadata domain is graph."""
        assert CALLGRAPH_METADATA.domain == PluginDomain.GRAPH

    def test_metadata_kind_and_stage(self) -> None:
        """Verify kind and stage."""
        assert CALLGRAPH_METADATA.kind == "builder"
        assert CALLGRAPH_METADATA.stage == "edges"

    def test_metadata_capabilities(self) -> None:
        """Verify provides and requires."""
        assert "graph.callgraph" in CALLGRAPH_METADATA.provides
        assert "core.goids" in CALLGRAPH_METADATA.requires

    def test_metadata_tables(self) -> None:
        """Verify produces_tables."""
        assert "graph.call_graph_nodes" in CALLGRAPH_METADATA.produces_tables
        assert "graph.call_graph_edges" in CALLGRAPH_METADATA.produces_tables

    def test_metadata_is_scope_aware(self) -> None:
        """Verify plugin is marked as scope-aware."""
        assert CALLGRAPH_METADATA.scope_aware is True

    def test_metadata_extra_graph_kinds(self) -> None:
        """Verify extra.graph_kinds is set."""
        assert CALLGRAPH_METADATA.extra.get("graph_kinds") == ("call_graph",)


class TestCallGraphPluginOptionsIntegration:
    """Tests for options resolution integration."""

    def test_default_options_without_resolver(self) -> None:
        """Verify default options when no resolver provided."""
        plugin = CallGraphPlugin()
        opts = plugin.resolve_options()
        assert opts.use_libcst is True
        assert opts.resolve_imports is True
        assert opts.include_external_calls is True

    def test_options_with_fast_profile(self) -> None:
        """Verify fast profile options."""
        source = DictConfigSource({
            "graphs.callgraph": {
                "use_libcst": False,
                "resolve_imports": False,
                "include_external_calls": False,
            },
        })
        resolver = PluginOptionsResolver(source)
        plugin = CallGraphPlugin(options_resolver=resolver)
        opts = plugin.resolve_options()
        assert opts.use_libcst is False
        assert opts.resolve_imports is False
        assert opts.include_external_calls is False

    def test_scope_paths_filtering(self) -> None:
        """Verify scope_paths is passed through options."""
        source = DictConfigSource({
            "graphs.callgraph": {
                "scope_paths": ["src/", "lib/"],
            },
        })
        resolver = PluginOptionsResolver(source)
        plugin = CallGraphPlugin(options_resolver=resolver)
        opts = plugin.resolve_options()
        assert opts.scope_paths == ["src/", "lib/"]


class TestCallGraphPluginMetadataProperty:
    """Tests for plugin.metadata property."""

    def test_metadata_property_returns_plugin_metadata(self) -> None:
        """Verify metadata property returns PluginMetadata type."""
        plugin = CallGraphPlugin()
        meta = plugin.metadata
        assert meta.name == "graphs.callgraph"
        assert meta.version == "3.0.0"

    def test_core_metadata_property(self) -> None:
        """Verify core_metadata returns CorePluginMetadata."""
        plugin = CallGraphPlugin()
        core = plugin.core_metadata
        assert core is CALLGRAPH_METADATA
```

---

## 6. Task 4: Ingestion Plugin Migration

### 6.1 Update `ScipIngestPlugin`

```python
# File: src/codeintel/ingestion/plugins/scip_plugin.py
"""SCIP ingest plugin.

This module provides `ScipIngestPlugin` that runs scip-python
and persists symbols and GOID crosswalk.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from codeintel.build.errors import ToolNotAvailableError
from codeintel.build.plugin import TargetPlugin
from codeintel.build.result import TargetResult
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.types.protocol import PluginMetadata
from codeintel.ingestion.adapters import BuildToolAdapter, DuckDBStorageAdapter
from codeintel.ingestion.compute.scip_ingest import ScipIngestConfig, ScipIngestStep
from codeintel.ingestion.plugins.helpers import get_module_paths, paths_to_modules
from codeintel.ingestion.plugins.scip_options import ScipIngestOptions

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.execution.options import PluginOptionsResolver

log = logging.getLogger(__name__)

# =============================================================================
# Metadata Constants
# =============================================================================

SCIP_INGEST_METADATA = CorePluginMetadata(
    name="ingest.scip_python",
    version="3.0.0",
    description="Run scip-python and persist symbols and GOID crosswalk.",
    domain=PluginDomain.INGEST,
    kind="builder",
    stage="goid",
    provides=(
        "core.scip_symbols",
        "core.goid_crosswalk",
    ),
    requires=("core.modules",),
    produces_tables=(
        "core.scip_symbols",
        "core.goid_crosswalk",
    ),
    consumes_tables=("core.modules",),
    supports_incremental=False,
    scope_aware=True,
    options_model=ScipIngestOptions,
    resource_hints={
        "max_memory_mb": 2048,
        "requires_tools": ["scip-python"],
    },
)


def _to_plugin_metadata(core: CorePluginMetadata) -> PluginMetadata:
    """Convert CorePluginMetadata to PluginMetadata for protocol compliance."""
    return PluginMetadata(
        name=core.name,
        version=core.version,
        description=core.description,
        kind=core.kind,
        stage=core.stage or "goid",
        provides=core.provides,
        requires=core.requires,
        produces_tables=core.produces_tables,
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_row_counts(ctx: TargetExecutionContext) -> dict[str, int]:
    """Compute row counts for output tables."""
    row_counts: dict[str, int] = {}
    for table_key in ctx.contract.table_keys:
        try:
            count = ctx.gateway.con.execute(
                f"SELECT COUNT(*) FROM {table_key} "  # noqa: S608
                f"WHERE repo = ? AND commit = ?",
                [ctx.repo, ctx.commit],
            ).fetchone()
            row_counts[table_key] = int(count[0]) if count else 0
        except (RuntimeError, OSError):
            row_counts[table_key] = 0
    return row_counts


def _filter_paths(
    paths: list[str],
    scope_paths: list[str] | None,
) -> list[str]:
    """Filter paths by scope if scope_paths is set.

    Parameters
    ----------
    paths
        All available paths.
    scope_paths
        Scope filter paths, or None for no filtering.

    Returns
    -------
    list[str]
        Filtered paths.
    """
    if not scope_paths:
        return paths

    scope_prefixes = tuple(scope_paths)
    return [p for p in paths if p.startswith(scope_prefixes)]


# =============================================================================
# Plugin Implementation
# =============================================================================


class ScipIngestPlugin(TargetPlugin):
    """Run scip-python and persist symbols and GOID crosswalk.

    This plugin uses the unified data abstraction infrastructure:
    - CorePluginMetadata for capability/dataset declaration
    - PluginOptionsResolver for profile-driven configuration

    Outputs
    -------
    - index.scip: SCIP index file
    - core.scip_symbols: Symbol table
    - core.goid_crosswalk: GOID crosswalk
    """

    plugin_name: ClassVar[str] = "scip_ingest"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Run scip-python and persist symbols and GOID crosswalk."

    _core_metadata: ClassVar[CorePluginMetadata] = SCIP_INGEST_METADATA

    def __init__(
        self,
        *,
        options_resolver: PluginOptionsResolver | None = None,
    ) -> None:
        """Initialize plugin with optional options resolver.

        Parameters
        ----------
        options_resolver
            Resolver for typed configuration options.
        """
        self._options_resolver = options_resolver

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return _to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return full core metadata."""
        return self._core_metadata

    def resolve_options(
        self,
        *,
        dynamic_overrides: Mapping[str, Any] | None = None,
    ) -> ScipIngestOptions:
        """Resolve typed options from configuration.

        Parameters
        ----------
        dynamic_overrides
            Runtime-only overrides.

        Returns
        -------
        ScipIngestOptions
            Resolved options instance.
        """
        if self._options_resolver is None:
            if dynamic_overrides:
                return ScipIngestOptions(**dynamic_overrides)
            return ScipIngestOptions()

        return self._options_resolver.get_options(
            self._core_metadata,
            ScipIngestOptions,
            dynamic_overrides=dynamic_overrides,
        )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute SCIP indexing.

        Parameters
        ----------
        ctx
            Execution context with resources and parameters.

        Returns
        -------
        TargetResult
            Success result with row counts.

        Raises
        ------
        ToolNotAvailableError
            When the scip-python tool is not available.
        """
        # Resolve options
        opts = self.resolve_options(
            dynamic_overrides={"scip_output_dir": ctx.scip_dir}
        )

        # Check tool availability
        if ctx.resources.scip_indexer is None:
            raise ToolNotAvailableError(target=self.plugin_name, tool="scip-python")

        # Get module paths and apply scope filtering
        paths = get_module_paths(ctx)
        paths = _filter_paths(paths, opts.scope_paths)
        modules = paths_to_modules(paths, ctx.repo_root)

        # Create adapters using build protocols
        storage = DuckDBStorageAdapter(ctx.gateway)
        tool = BuildToolAdapter(scip_indexer=ctx.resources.scip_indexer)

        # Create config
        scip_dir = opts.scip_output_dir or ctx.scip_dir
        config = ScipIngestConfig(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
            output_scip=scip_dir / "index.scip",
            output_json=scip_dir / "index.json",
        )

        # Execute step
        step = ScipIngestStep(storage=storage, tools=tool)
        result = await step.execute_async(modules, config)

        if not result.success:
            errors = "; ".join(result.errors) if result.errors else "Unknown error"
            return TargetResult.failed(f"SCIP ingest failed: {errors}")

        # Compute row counts
        row_counts = _compute_row_counts(ctx)
        return TargetResult.succeeded(
            row_counts=row_counts,
            artifacts_written=["index.scip", "index.json"],
        )


__all__ = [
    "SCIP_INGEST_METADATA",
    "ScipIngestPlugin",
    "get_module_paths",
    "paths_to_modules",
]
```

### 6.2 Test File: `tests/ingestion/plugins/test_scip_metadata.py`

```python
# File: tests/ingestion/plugins/test_scip_metadata.py
"""Tests for ScipIngestPlugin metadata and options integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from codeintel.core.plugins.execution.options import (
    ConfigSource,
    EmptyConfigSource,
    PluginOptionsResolver,
)
from codeintel.core.plugins.types.metadata import PluginDomain
from codeintel.ingestion.plugins.scip_options import ScipIngestOptions
from codeintel.ingestion.plugins.scip_plugin import (
    SCIP_INGEST_METADATA,
    ScipIngestPlugin,
)


class DictConfigSource(ConfigSource):
    """Test config source backed by a dict."""

    def __init__(self, options: dict[str, dict[str, Any]]) -> None:
        self._options = options

    def get_plugin_options(self, plugin_name: str) -> dict[str, Any] | None:
        return self._options.get(plugin_name)


class TestScipIngestMetadata:
    """Tests for SCIP_INGEST_METADATA constant."""

    def test_metadata_name(self) -> None:
        """Verify metadata name is canonical."""
        assert SCIP_INGEST_METADATA.name == "ingest.scip_python"

    def test_metadata_domain(self) -> None:
        """Verify metadata domain is ingest."""
        assert SCIP_INGEST_METADATA.domain == PluginDomain.INGEST

    def test_metadata_kind_and_stage(self) -> None:
        """Verify kind and stage."""
        assert SCIP_INGEST_METADATA.kind == "builder"
        assert SCIP_INGEST_METADATA.stage == "goid"

    def test_metadata_capabilities(self) -> None:
        """Verify provides and requires."""
        assert "core.scip_symbols" in SCIP_INGEST_METADATA.provides
        assert "core.goid_crosswalk" in SCIP_INGEST_METADATA.provides
        assert "core.modules" in SCIP_INGEST_METADATA.requires

    def test_metadata_tables(self) -> None:
        """Verify produces_tables."""
        assert "core.scip_symbols" in SCIP_INGEST_METADATA.produces_tables
        assert "core.goid_crosswalk" in SCIP_INGEST_METADATA.produces_tables

    def test_metadata_is_scope_aware(self) -> None:
        """Verify plugin is marked as scope-aware."""
        assert SCIP_INGEST_METADATA.scope_aware is True

    def test_metadata_resource_hints(self) -> None:
        """Verify resource hints include tool requirement."""
        hints = SCIP_INGEST_METADATA.resource_hints
        assert hints.get("requires_tools") == ["scip-python"]


class TestScipIngestPluginOptionsIntegration:
    """Tests for options resolution integration."""

    def test_default_options_without_resolver(self) -> None:
        """Verify default options when no resolver provided."""
        plugin = ScipIngestPlugin()
        opts = plugin.resolve_options()
        assert opts.include_references is True
        assert opts.include_implementations is True
        assert opts.timeout_seconds == 300

    def test_options_with_fast_profile(self) -> None:
        """Verify fast profile options."""
        source = DictConfigSource({
            "ingest.scip_python": {
                "include_references": False,
                "include_implementations": False,
                "timeout_seconds": 120,
            },
        })
        resolver = PluginOptionsResolver(source)
        plugin = ScipIngestPlugin(options_resolver=resolver)
        opts = plugin.resolve_options()
        assert opts.include_references is False
        assert opts.include_implementations is False
        assert opts.timeout_seconds == 120

    def test_dynamic_overrides(self) -> None:
        """Verify dynamic overrides are applied."""
        plugin = ScipIngestPlugin()
        scip_dir = Path("/tmp/scip")
        opts = plugin.resolve_options(
            dynamic_overrides={"scip_output_dir": scip_dir}
        )
        assert opts.scip_output_dir == scip_dir


class TestScipIngestPluginMetadataProperty:
    """Tests for plugin.metadata property."""

    def test_metadata_property_returns_plugin_metadata(self) -> None:
        """Verify metadata property returns PluginMetadata type."""
        plugin = ScipIngestPlugin()
        meta = plugin.metadata
        assert meta.name == "ingest.scip_python"
        assert meta.version == "3.0.0"

    def test_core_metadata_property(self) -> None:
        """Verify core_metadata returns CorePluginMetadata."""
        plugin = ScipIngestPlugin()
        core = plugin.core_metadata
        assert core is SCIP_INGEST_METADATA
```

---

## 7. Task 5: Plugin Run Context Integration

### 7.1 Create `PluginRunContext` Helper

This module ties together metadata, options, and hashing for prepare/skip logic.

```python
# File: src/codeintel/core/plugins/execution/run_context.py
"""Plugin run context for unified execution preparation.

This module provides `PluginRunContext` and `prepare_plugin_run` to tie
together metadata, options, and upstream state for plugin execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codeintel.core.plugins.execution.manifest import (
    compute_input_hash,
    compute_options_hash,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.core.plugins.execution.options import PluginOptionsResolver
    from codeintel.core.plugins.types.metadata import CorePluginMetadata


@dataclass(frozen=True)
class PluginRunContext:
    """Context for a single plugin execution.

    This dataclass encapsulates all data needed for a plugin invocation:
    - Plugin metadata
    - Resolved options
    - Upstream state (capability → provider input hash)
    - Computed hashes (options_hash, input_hash)

    Attributes
    ----------
    metadata
        Plugin's CorePluginMetadata.
    options
        Resolved options instance.
    upstream_state
        Mapping of capability name → provider's input hash.
    options_hash
        SHA-256 hash of serialized options (16 chars).
    input_hash
        SHA-256 hash of all logical inputs (16 chars).

    Examples
    --------
    >>> ctx = prepare_plugin_run(
    ...     metadata=FUNCTION_METRICS_METADATA,
    ...     resolver=resolver,
    ...     upstream_state={"core.goids": "hash123"},
    ... )
    >>> ctx.options_hash
    '...'
    """

    metadata: CorePluginMetadata
    options: Any
    upstream_state: Mapping[str, str]
    options_hash: str
    input_hash: str

    @property
    def plugin_name(self) -> str:
        """Return the plugin's canonical name.

        Returns
        -------
        str
            Plugin name from metadata.
        """
        return self.metadata.name

    @property
    def plugin_version(self) -> str:
        """Return the plugin's version.

        Returns
        -------
        str
            Version from metadata.
        """
        return self.metadata.version


def prepare_plugin_run(
    metadata: CorePluginMetadata,
    resolver: PluginOptionsResolver,
    upstream_state: Mapping[str, str],
    *,
    dynamic_overrides: Mapping[str, Any] | None = None,
) -> PluginRunContext:
    """Prepare context for a plugin run.

    This function:
    1. Resolves options from the resolver + dynamic overrides
    2. Computes options_hash from serializable option values
    3. Computes input_hash from options_hash + upstream_state

    Parameters
    ----------
    metadata
        Plugin's CorePluginMetadata.
    resolver
        Options resolver for configuration.
    upstream_state
        Mapping of capability → provider input hash.
    dynamic_overrides
        Runtime-only overrides for options.

    Returns
    -------
    PluginRunContext
        Context ready for execution or skip check.

    Examples
    --------
    >>> from codeintel.core.plugins.execution.options import (
    ...     PluginOptionsResolver,
    ...     EmptyConfigSource,
    ... )
    >>> resolver = PluginOptionsResolver(EmptyConfigSource())
    >>> ctx = prepare_plugin_run(
    ...     metadata=FUNCTION_METRICS_METADATA,
    ...     resolver=resolver,
    ...     upstream_state={"core.goids": "abc123"},
    ... )
    """
    if metadata.options_model is None:
        options = None
        options_hash = compute_options_hash({})
    else:
        options = resolver.get_options(
            metadata,
            metadata.options_model,
            dynamic_overrides=dynamic_overrides,
        )
        # Extract serializable fields only
        options_dict = _extract_serializable_options(options)
        options_hash = compute_options_hash(options_dict)

    input_hash = compute_input_hash(
        options_hash=options_hash,
        upstream_state=dict(upstream_state),
    )

    return PluginRunContext(
        metadata=metadata,
        options=options,
        upstream_state=upstream_state,
        options_hash=options_hash,
        input_hash=input_hash,
    )


def _extract_serializable_options(options: Any) -> dict[str, Any]:
    """Extract serializable fields from an options object.

    Parameters
    ----------
    options
        Options instance (dataclass or Pydantic model).

    Returns
    -------
    dict[str, Any]
        Dictionary of serializable fields only.
    """
    if hasattr(options, "__dataclass_fields__"):
        # Dataclass: filter out non-serializable fields
        import dataclasses

        result = {}
        for field_info in dataclasses.fields(options):
            value = getattr(options, field_info.name)
            if _is_serializable(value):
                result[field_info.name] = value
        return result

    if hasattr(options, "model_dump"):
        # Pydantic v2
        return options.model_dump(exclude_none=True)

    if hasattr(options, "dict"):
        # Pydantic v1
        return options.dict(exclude_none=True)

    # Fallback: try __dict__
    return {k: v for k, v in vars(options).items() if _is_serializable(v)}


def _is_serializable(value: Any) -> bool:
    """Check if a value is JSON-serializable.

    Parameters
    ----------
    value
        Value to check.

    Returns
    -------
    bool
        True if value is serializable.
    """
    if value is None:
        return True
    if isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_serializable(v) for v in value)
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and _is_serializable(v) for k, v in value.items()
        )
    return False


__all__ = [
    "PluginRunContext",
    "prepare_plugin_run",
]
```

### 7.2 Test File: `tests/core/plugins/test_run_context.py`

```python
# File: tests/core/plugins/test_run_context.py
"""Tests for PluginRunContext and prepare_plugin_run."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from codeintel.core.plugins.execution.options import (
    EmptyConfigSource,
    PluginOptionsResolver,
)
from codeintel.core.plugins.execution.run_context import (
    PluginRunContext,
    prepare_plugin_run,
)
from codeintel.core.plugins.types.metadata import (
    CorePluginMetadata,
    PluginDomain,
)


@dataclass(frozen=True)
class TestOptions:
    """Test options model."""

    threshold: float = 0.5
    enabled: bool = True


@pytest.fixture
def sample_metadata() -> CorePluginMetadata:
    """Create sample metadata for testing."""
    return CorePluginMetadata(
        name="test.plugin",
        version="1.0.0",
        description="Test plugin.",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
        provides=("test.output",),
        requires=("test.input",),
        options_model=TestOptions,
    )


@pytest.fixture
def resolver() -> PluginOptionsResolver:
    """Create empty options resolver."""
    return PluginOptionsResolver(EmptyConfigSource())


class TestPreparePluginRun:
    """Tests for prepare_plugin_run."""

    def test_creates_context_with_defaults(
        self,
        sample_metadata: CorePluginMetadata,
        resolver: PluginOptionsResolver,
    ) -> None:
        """Verify context is created with default options."""
        ctx = prepare_plugin_run(
            metadata=sample_metadata,
            resolver=resolver,
            upstream_state={},
        )
        assert ctx.metadata is sample_metadata
        assert isinstance(ctx.options, TestOptions)
        assert ctx.options.threshold == 0.5

    def test_computes_hashes(
        self,
        sample_metadata: CorePluginMetadata,
        resolver: PluginOptionsResolver,
    ) -> None:
        """Verify hashes are computed."""
        ctx = prepare_plugin_run(
            metadata=sample_metadata,
            resolver=resolver,
            upstream_state={"test.input": "upstream123"},
        )
        assert len(ctx.options_hash) == 16
        assert len(ctx.input_hash) == 16

    def test_different_upstream_produces_different_hash(
        self,
        sample_metadata: CorePluginMetadata,
        resolver: PluginOptionsResolver,
    ) -> None:
        """Verify different upstream state produces different input hash."""
        ctx1 = prepare_plugin_run(
            metadata=sample_metadata,
            resolver=resolver,
            upstream_state={"test.input": "upstream1"},
        )
        ctx2 = prepare_plugin_run(
            metadata=sample_metadata,
            resolver=resolver,
            upstream_state={"test.input": "upstream2"},
        )
        assert ctx1.options_hash == ctx2.options_hash  # Same options
        assert ctx1.input_hash != ctx2.input_hash  # Different upstream

    def test_plugin_name_property(
        self,
        sample_metadata: CorePluginMetadata,
        resolver: PluginOptionsResolver,
    ) -> None:
        """Verify plugin_name property."""
        ctx = prepare_plugin_run(
            metadata=sample_metadata,
            resolver=resolver,
            upstream_state={},
        )
        assert ctx.plugin_name == "test.plugin"

    def test_metadata_without_options_model(
        self,
        resolver: PluginOptionsResolver,
    ) -> None:
        """Verify context works without options model."""
        meta = CorePluginMetadata(
            name="test.no_options",
            version="1.0.0",
            description="Test.",
            domain=PluginDomain.ANALYTICS,
            kind="metric",
            options_model=None,
        )
        ctx = prepare_plugin_run(
            metadata=meta,
            resolver=resolver,
            upstream_state={},
        )
        assert ctx.options is None
        assert len(ctx.options_hash) == 16
```

---

## 8. Verification

### 8.1 Run Quality Checks

```bash
# Format and lint all modified files
uv run ruff format \
    src/codeintel/analytics/functions/config.py \
    src/codeintel/analytics/plugins/functions/metrics.py \
    src/codeintel/graphs/plugins/builders/callgraph.py \
    src/codeintel/graphs/plugins/builders/callgraph_options.py \
    src/codeintel/ingestion/plugins/scip_plugin.py \
    src/codeintel/ingestion/plugins/scip_options.py \
    src/codeintel/core/plugins/execution/run_context.py

uv run ruff check --fix \
    src/codeintel/analytics/ \
    src/codeintel/graphs/plugins/builders/ \
    src/codeintel/ingestion/plugins/ \
    src/codeintel/core/plugins/execution/run_context.py

# Type checking
uv run pyright \
    src/codeintel/analytics/functions/config.py \
    src/codeintel/analytics/plugins/functions/metrics.py \
    src/codeintel/graphs/plugins/builders/callgraph.py \
    src/codeintel/graphs/plugins/builders/callgraph_options.py \
    src/codeintel/ingestion/plugins/scip_plugin.py \
    src/codeintel/ingestion/plugins/scip_options.py \
    src/codeintel/core/plugins/execution/run_context.py

# Pyrefly
uv run pyrefly check \
    src/codeintel/analytics/plugins/functions/ \
    src/codeintel/graphs/plugins/builders/ \
    src/codeintel/ingestion/plugins/
```

### 8.2 Run Tests

```bash
# Run Phase 2 tests
uv run pytest tests/analytics/plugins/test_function_metrics_metadata.py -v
uv run pytest tests/graphs/plugins/test_callgraph_metadata.py -v
uv run pytest tests/ingestion/plugins/test_scip_metadata.py -v
uv run pytest tests/core/plugins/test_run_context.py -v

# Run existing plugin tests to verify no regression
uv run pytest tests/analytics/plugins/ tests/graphs/plugins/ tests/ingestion/plugins/ -v
```

### 8.3 Verification Checklist

- [ ] All modified files pass `ruff format` and `ruff check`
- [ ] All modified files pass `pyright --strict`
- [ ] All modified files pass `pyrefly check`
- [ ] All new tests pass
- [ ] All existing plugin tests pass (no regression)
- [ ] `FunctionMetricsPlugin` exports `FUNCTION_METRICS_METADATA`
- [ ] `CallGraphPlugin` exports `CALLGRAPH_METADATA`
- [ ] `ScipIngestPlugin` exports `SCIP_INGEST_METADATA`
- [ ] Each plugin's `metadata` property returns compatible `PluginMetadata`

---

## 9. Rollback Plan

Phase 2 modifications are backward-compatible. To rollback:

1. **Revert plugin files** to their previous versions
2. **Delete new options files**:
   - `src/codeintel/graphs/plugins/builders/callgraph_options.py`
   - `src/codeintel/ingestion/plugins/scip_options.py`
3. **Delete run_context module**:
   - `src/codeintel/core/plugins/execution/run_context.py`
4. **Delete test files**:
   - `tests/analytics/plugins/test_function_metrics_metadata.py`
   - `tests/graphs/plugins/test_callgraph_metadata.py`
   - `tests/ingestion/plugins/test_scip_metadata.py`
   - `tests/core/plugins/test_run_context.py`

---

## Appendix A: File Checklist

| File | Action | Status |
|------|--------|--------|
| `src/codeintel/analytics/functions/config.py` | MODIFY | ⬜ |
| `src/codeintel/analytics/plugins/functions/metrics.py` | MODIFY | ⬜ |
| `src/codeintel/graphs/plugins/builders/callgraph.py` | MODIFY | ⬜ |
| `src/codeintel/graphs/plugins/builders/callgraph_options.py` | CREATE | ⬜ |
| `src/codeintel/ingestion/plugins/scip_plugin.py` | MODIFY | ⬜ |
| `src/codeintel/ingestion/plugins/scip_options.py` | CREATE | ⬜ |
| `src/codeintel/core/plugins/execution/run_context.py` | CREATE | ⬜ |
| `tests/analytics/plugins/test_function_metrics_metadata.py` | CREATE | ⬜ |
| `tests/graphs/plugins/test_callgraph_metadata.py` | CREATE | ⬜ |
| `tests/ingestion/plugins/test_scip_metadata.py` | CREATE | ⬜ |
| `tests/core/plugins/test_run_context.py` | CREATE | ⬜ |

---

**Next Steps**: After Phase 2 is complete, proceed to Phase 3 (Full Rollout) to migrate all remaining plugins using the spine plugins as templates.
