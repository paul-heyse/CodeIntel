# Legacy Decommissioning Plan — Phase 2

**Status:** Completed  
**Created:** December 13, 2024  
**Completed:** December 13, 2024  
**Predecessor:** [Legacy Decommissioning Summary](./Legacy_Decommissioning_Summary.md)

---

## Executive Summary

Phase 2 decommissioning targets **residual legacy infrastructure** remaining after the Hamilton migration. This includes orphaned analytics registry code, unused ingestion plugins, broken build plugin catalogs, and legacy plugin constraint extraction. The goal is to consolidate on the build registry (`codeintel.build.plugin_registry`) as the single source of truth.

**Estimated Removals:**
- **5+ source files** deleted
- **3+ test files** deleted or modified  
- **~800+ lines** of dead code removed
- **2+ broken import paths** fixed

---

## Table of Contents

1. [Scope Validation](#scope-validation)
2. [Work Packages](#work-packages)
   - [WP-1: Analytics Core Removal](#wp-1-analytics-core-removal)
   - [WP-2: Plugin Catalog Consolidation](#wp-2-plugin-catalog-consolidation)
   - [WP-3: ModuleIngestPlugin Removal](#wp-3-moduleingestplugin-removal)
   - [WP-4: Build Plugins Consolidation](#wp-4-build-plugins-consolidation)
   - [WP-5: Plugin Constraints Rewrite](#wp-5-plugin-constraints-rewrite)
3. [Dependency Graph](#dependency-graph)
4. [Execution Order](#execution-order)
5. [Verification Checklist](#verification-checklist)
6. [Rollback Strategy](#rollback-strategy)
7. [Post-Decommissioning Cleanup](#post-decommissioning-cleanup)

---

## Scope Validation

### Validated Candidates for Removal

| File/Directory | Status | Reason | Dependencies |
|----------------|--------|--------|--------------|
| `src/codeintel/analytics/core/` | ✅ Confirmed | Only contains stub `__init__.py` (9 lines) | `plugin_constraints.py` tries to import from here |
| `src/codeintel/analytics/graphs/plugin_catalog.py` | ✅ Confirmed | Uses legacy `ALL_PLUGINS` from `registration.py` | Scripts, tests import from here |
| `src/codeintel/analytics/plugins/registration.py` | ⚠️ Conditional | Used by `plugin_catalog.py` and `__init__.py` | Migrate consumers first |
| `src/codeintel/ingestion/plugins/modules_plugin.py` | ✅ Confirmed | Build registry uses `RepoScanPlugin` for "modules" target | Only self-referential |
| `src/codeintel/build/plugins.py` | ⚠️ Conditional | `PLUGIN_CATALOG` referenced but undefined | Tests use `PluginCatalog` class |

### Files Requiring Modification

| File | Required Changes |
|------|------------------|
| `src/codeintel/config/datasets/plugin_constraints.py` | Remove dead import of `codeintel.analytics.core.registry` |
| `src/codeintel/config/datasets/schema_registry.py` | Remove dead import of `codeintel.build.plugins.PLUGIN_CATALOG` |
| `src/codeintel/analytics/plugins/__init__.py` | Remove `ALL_PLUGINS` and singleton exports from `registration.py` |
| `src/codeintel/ingestion/plugins/__init__.py` | Remove `ModuleIngestPlugin` export |
| `scripts/render_graph_plugin_catalog.py` | Rewrite to use build registry or delete |
| `tests/analytics/core/test_plugin_registration.py` | Delete or migrate to new pattern |
| `tests/analytics/test_graph_plugin_catalog.py` | Delete or rewrite |
| `tests/build/test_plugin_registry_plugins.py` | Remove imports from `codeintel.build.plugins` |

---

## Work Packages

### WP-1: Analytics Core Removal

**Goal:** Remove the empty `src/codeintel/analytics/core/` directory and fix broken imports.

#### Current State

```
src/codeintel/analytics/core/
├── __init__.py  (9 lines - stub only)
└── __pycache__/
```

The `__init__.py` contains:
```python
"""Analytics core module (minimal stub).

The legacy analytics plugin infrastructure has been removed.
Analytics plugins now implement TargetPlugin from codeintel.build.plugin.
"""

from __future__ import annotations

__all__: list[str] = []
```

#### Broken Import Chain

```
config/datasets/plugin_constraints.py
  └── imports codeintel.analytics.core.registry  ← DOES NOT EXIST
        └── tries to access ANALYTICS_REGISTRY    ← FAILS
```

#### Actions

| Step | Action | Files |
|------|--------|-------|
| 1.1 | Fix broken import in `plugin_constraints.py` | `src/codeintel/config/datasets/plugin_constraints.py` |
| 1.2 | Delete analytics/core directory | `src/codeintel/analytics/core/` |
| 1.3 | Delete associated test directory | `tests/analytics/core/` |
| 1.4 | Update AGENTS.md references | `AGENTS.md` |

#### Code Changes

**`src/codeintel/config/datasets/plugin_constraints.py`** — Rewrite `_get_plugin_catalog()`:

```python
# BEFORE (broken):
def _get_plugin_catalog() -> object | None:
    try:
        analytics_registry = importlib.import_module("codeintel.analytics.core.registry")
        catalog = analytics_registry.ANALYTICS_REGISTRY
    except ImportError:
        log.debug("Analytics registry not available for plugin constraint extraction")
        return None
    return _PluginCatalogHolder.get(lambda: catalog)

# AFTER (use build registry):
def _get_plugin_catalog() -> object | None:
    """Load plugin metadata from build registry.
    
    Returns
    -------
    object | None
        Iterable of plugins with core_metadata, or None if unavailable.
    """
    existing = _PluginCatalogHolder.get_or_none()
    if existing is not None:
        return existing

    try:
        from codeintel.build.plugin_registry import get_all_plugins
        plugins = get_all_plugins()
        # Wrap dict values as iterable for compatibility
        return _PluginCatalogHolder.get(lambda: _PluginIterableAdapter(plugins))
    except ImportError:
        log.debug("Build plugin registry not available for plugin constraint extraction")
        return None


class _PluginIterableAdapter:
    """Adapter to iterate plugin instances from registry."""
    
    def __init__(self, plugins: dict[str, type]) -> None:
        self._plugins = plugins
    
    def all(self) -> list[object]:
        return [cls() for cls in self._plugins.values()]
```

#### Files to Delete

- `src/codeintel/analytics/core/__init__.py`
- `src/codeintel/analytics/core/` (directory)
- `tests/analytics/core/__init__.py`
- `tests/analytics/core/test_plugin_registration.py`
- `tests/analytics/core/` (directory)

---

### WP-2: Plugin Catalog Consolidation

**Goal:** Retire `analytics.graphs.plugin_catalog` and `analytics.plugins.registration` in favor of build registry.

#### Current State

The legacy analytics plugin catalog uses a static `ALL_PLUGINS` tuple:

```python
# analytics/plugins/registration.py
ALL_PLUGINS = (
    FUNCTION_METRICS_PLUGIN,
    FUNCTION_AST_FEATURES_PLUGIN,
    # ... 25 plugin instances
)
```

This is consumed by:
- `analytics/graphs/plugin_catalog.py` — generates JSON/Markdown catalogs
- `analytics/plugins/__init__.py` — exports plugin singletons
- `tests/analytics/core/test_plugin_registration.py` — validates ALL_PLUGINS

#### Problem

The build registry (`codeintel.build.plugin_registry`) is the canonical source:

```python
# build/plugin_registry.py
_PLUGIN_DEFINITIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("codeintel.analytics.plugins.hotspots.build", "HotspotsPlugin", ("hotspots",)),
    ("codeintel.analytics.plugins.functions.metrics", "FunctionMetricsPlugin", ("function_metrics",)),
    # ... all 40+ plugin definitions
)
```

Maintaining two registries causes drift and confusion.

#### Decision Matrix

| Option | Description | Recommended |
|--------|-------------|-------------|
| A | Delete `plugin_catalog.py` and `registration.py` entirely | ❌ Breaks scripts |
| B | Rewrite `plugin_catalog.py` to use build registry | ✅ **Yes** |
| C | Keep both, document "use build registry" | ❌ Technical debt |

#### Actions

| Step | Action | Files |
|------|--------|-------|
| 2.1 | Rewrite `plugin_catalog.py` to use `get_all_plugins()` | `src/codeintel/analytics/graphs/plugin_catalog.py` |
| 2.2 | Update `render_graph_plugin_catalog.py` import path | `scripts/render_graph_plugin_catalog.py` |
| 2.3 | Delete `registration.py` | `src/codeintel/analytics/plugins/registration.py` |
| 2.4 | Update `analytics/plugins/__init__.py` | Remove ALL_PLUGINS and singleton re-exports |
| 2.5 | Update tests | `tests/analytics/test_graph_plugin_catalog.py` |

#### Code Changes

**`src/codeintel/analytics/graphs/plugin_catalog.py`** — Rewrite to use build registry:

```python
"""Plugin catalog generation using the build registry.

This module generates documentation catalogs from the unified
build registry (codeintel.build.plugin_registry).
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from codeintel.build.plugin_registry import get_all_plugins

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


def _compute_version_hash(name: str, version: str) -> str:
    """Compute a hash of version-relevant metadata."""
    raw = f"{name}:{version}"
    return hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]


def build_plugin_catalog() -> dict[str, Any]:
    """Build a JSON-serializable catalog of all registered plugins.

    Uses the build registry as the single source of truth.

    Returns
    -------
    dict[str, Any]
        Catalog dict with 'plugins' key containing plugin metadata.
    """
    plugins: dict[str, dict[str, Any]] = {}
    registry = get_all_plugins()

    for target_name, plugin_class in registry.items():
        plugin = plugin_class()
        plugins[plugin.plugin_name] = {
            "name": plugin.plugin_name,
            "description": getattr(plugin, "plugin_description", ""),
            "version": getattr(plugin, "plugin_version", "1.0.0"),
            "version_hash": _compute_version_hash(
                plugin.plugin_name,
                getattr(plugin, "plugin_version", "1.0.0"),
            ),
            "target": target_name,
            "stage": "analytics",
            "enabled_by_default": True,
            "depends_on": [],
            "provides": [],
            "requires": [],
        }

    return {"plugins": plugins, "count": len(plugins)}
```

**`scripts/render_graph_plugin_catalog.py`** — Fix import path:

```python
# BEFORE (broken path):
from codeintel.analytics.graphs.catalog import (

# AFTER (correct path):
from codeintel.analytics.graphs.plugin_catalog import (
```

#### Files to Delete

- `src/codeintel/analytics/plugins/registration.py`

#### Files to Modify

- `src/codeintel/analytics/plugins/__init__.py` — Remove ALL_PLUGINS exports and singleton re-exports
- `tests/analytics/test_graph_plugin_catalog.py` — Update assertions

---

### WP-3: ModuleIngestPlugin Removal

**Goal:** Remove the unused `ModuleIngestPlugin` since `RepoScanPlugin` handles the "modules" target.

#### Current State

Build registry maps "modules" to `RepoScanPlugin`:

```python
# build/plugin_registry.py line 37-40
(
    "codeintel.ingestion.plugins.repo_scan",
    "RepoScanPlugin",
    ("modules",),
),
```

But `ModuleIngestPlugin` still exists in:
- `src/codeintel/ingestion/plugins/modules_plugin.py` (237 lines)
- Exported from `src/codeintel/ingestion/plugins/__init__.py`

#### Actions

| Step | Action | Files |
|------|--------|-------|
| 3.1 | Delete `modules_plugin.py` | `src/codeintel/ingestion/plugins/modules_plugin.py` |
| 3.2 | Remove export from `__init__.py` | `src/codeintel/ingestion/plugins/__init__.py` |

#### Code Changes

**`src/codeintel/ingestion/plugins/__init__.py`**:

```python
# BEFORE:
from codeintel.ingestion.plugins.modules_plugin import ModuleIngestPlugin
# ...
__all__ = [
    # ...
    "ModuleIngestPlugin",
    # ...
]

# AFTER:
# Remove ModuleIngestPlugin import and __all__ entry
```

#### Files to Delete

- `src/codeintel/ingestion/plugins/modules_plugin.py`

---

### WP-4: Build Plugins Consolidation

**Goal:** Clarify the role of `build/plugins.py` and fix broken `PLUGIN_CATALOG` reference.

#### Current State

`build/plugins.py` defines:
- `TargetPlugin` protocol (also in `build/plugin.py`)
- `PluginCatalog` class (decorator-based registry)
- `register_plugin`, `get_plugin`, `all_plugins` functions
- **No `PLUGIN_CATALOG` constant** (despite being imported by `schema_registry.py`)

`config/datasets/schema_registry.py` tries to load `PLUGIN_CATALOG`:

```python
def _load_plugin_catalog() -> object | None:
    spec = importlib.util.find_spec("codeintel.build.plugins")
    if spec is None:
        return None
    plugins_module = importlib.import_module("codeintel.build.plugins")
    return getattr(plugins_module, "PLUGIN_CATALOG", None)  # ← Returns None always
```

#### Decision Matrix

| Option | Description | Recommended |
|--------|-------------|-------------|
| A | Delete `build/plugins.py` entirely | ❌ Breaks tests |
| B | Keep `PluginCatalog` for decorator pattern, fix imports | ⚠️ Partial |
| C | Consolidate: move decorator pattern to `plugin_registry.py` | ✅ **Yes** |

#### Actions

| Step | Action | Files |
|------|--------|-------|
| 4.1 | Move `PluginCatalog` class to `plugin_registry.py` (if keeping) | `src/codeintel/build/plugin_registry.py` |
| 4.2 | Fix `schema_registry.py` to use build registry | `src/codeintel/config/datasets/schema_registry.py` |
| 4.3 | Update tests to import from `plugin_registry` | `tests/build/test_plugin_registry_plugins.py` |
| 4.4 | Delete `build/plugins.py` | `src/codeintel/build/plugins.py` |

#### Code Changes

**`src/codeintel/config/datasets/schema_registry.py`** — Rewrite `producers_of`/`consumers_of`:

```python
# BEFORE (broken):
def _load_plugin_catalog() -> object | None:
    spec = importlib.util.find_spec("codeintel.build.plugins")
    if spec is None:
        return None
    plugins_module = importlib.import_module("codeintel.build.plugins")
    return getattr(plugins_module, "PLUGIN_CATALOG", None)

# AFTER (working):
def _get_plugin_metadata() -> dict[str, object]:
    """Load plugin metadata from build registry.
    
    Returns
    -------
    dict[str, object]
        Mapping of target names to plugin instances.
    """
    try:
        from codeintel.build.plugin_registry import get_all_plugins
        registry = get_all_plugins()
        return {name: cls() for name, cls in registry.items()}
    except ImportError:
        return {}

@staticmethod
def producers_of(table_key: str) -> list[str]:
    """Find plugins that produce the given dataset."""
    plugins = _get_plugin_metadata()
    result: list[str] = []
    for name, plugin in plugins.items():
        if hasattr(plugin, "core_metadata"):
            produces = getattr(plugin.core_metadata, "produces_tables", None)
            if produces and table_key in produces:
                result.append(name)
    return result

@staticmethod  
def consumers_of(table_key: str) -> list[str]:
    """Find plugins that consume the given dataset."""
    plugins = _get_plugin_metadata()
    result: list[str] = []
    for name, plugin in plugins.items():
        if hasattr(plugin, "core_metadata"):
            consumes = getattr(plugin.core_metadata, "consumes_tables", None)
            if consumes and table_key in consumes:
                result.append(name)
    return result
```

**`tests/build/test_plugin_registry_plugins.py`** — Update imports:

```python
# BEFORE:
from codeintel.build.plugins import (
    PluginCatalog,
    TargetPlugin,
    all_plugins,
    get_plugin,
)
from codeintel.build.plugins import (
    register_plugin as decorator_register_plugin,
)

# AFTER:
from codeintel.build.plugin import TargetPlugin
from codeintel.build.plugin_registry import (
    PluginRegistryStore,
    get_all_plugins,
    get_plugin_for_target,
    register_plugin,
)
# If PluginCatalog pattern is needed, move to plugin_registry.py or remove tests
```

#### Files to Delete

- `src/codeintel/build/plugins.py` (242 lines)

---

### WP-5: Plugin Constraints Rewrite

**Goal:** Make `plugin_constraints.py` work with the build registry.

#### Current State

The module attempts to:
1. Import `codeintel.analytics.core.registry` (doesn't exist)
2. Access `ANALYTICS_REGISTRY.all()` (fails)
3. Extract `core_metadata.produces_tables` / `consumes_tables`

#### Actions

| Step | Action | Files |
|------|--------|-------|
| 5.1 | Rewrite `_get_plugin_catalog()` to use build registry | See WP-1 |
| 5.2 | Update `get_producer_plugins()` and `get_consumer_plugins()` | `src/codeintel/config/datasets/plugin_constraints.py` |
| 5.3 | Add tests for constraint extraction | New test file |

#### Code Changes

Complete rewrite of `plugin_constraints.py`:

```python
"""Plugin-based constraint extraction for dataset schema introspection.

This module extracts constraints from plugin metadata using the build
registry as the single source of truth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.config.datasets.constraints import Constraint, ConstraintKind, ConstraintSet

if TYPE_CHECKING:
    from codeintel.core.plugins.types.metadata import CorePluginMetadata

__all__ = [
    "PluginTableRelation",
    "extract_constraints_from_plugins",
    "get_consumer_plugins",
    "get_producer_plugins",
    "get_table_plugin_relations",
]

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_all_plugins_metadata() -> dict[str, CorePluginMetadata]:
    """Load all plugin metadata from the build registry.
    
    Returns
    -------
    dict[str, CorePluginMetadata]
        Mapping of target name to core metadata.
    """
    try:
        from codeintel.build.plugin_registry import get_all_plugins
    except ImportError:
        log.debug("Build plugin registry not available")
        return {}
    
    result: dict[str, CorePluginMetadata] = {}
    for target_name, plugin_class in get_all_plugins().items():
        try:
            plugin = plugin_class()
            if hasattr(plugin, "core_metadata"):
                result[target_name] = plugin.core_metadata
        except Exception:
            log.debug("Failed to get metadata for %s", target_name)
    return result


@dataclass(frozen=True)
class PluginTableRelation:
    """A relationship between a plugin and a table."""
    
    plugin_name: str
    plugin_version: str
    table_key: str
    relation_type: str
    domain: str
    
    @property
    def is_producer(self) -> bool:
        return self.relation_type == "produces"
    
    @property
    def is_consumer(self) -> bool:
        return self.relation_type == "consumes"


def get_producer_plugins(table_key: str) -> list[CorePluginMetadata]:
    """Find plugins that produce the given dataset."""
    result: list[CorePluginMetadata] = []
    for meta in _get_all_plugins_metadata().values():
        produces = getattr(meta, "produces_tables", None)
        if produces and table_key in produces:
            result.append(meta)
    return result


def get_consumer_plugins(table_key: str) -> list[CorePluginMetadata]:
    """Find plugins that consume the given dataset."""
    result: list[CorePluginMetadata] = []
    for meta in _get_all_plugins_metadata().values():
        consumes = getattr(meta, "consumes_tables", None)
        if consumes and table_key in consumes:
            result.append(meta)
    return result


# ... rest of the module (get_table_plugin_relations, extract_constraints_from_plugins, etc.)
```

---

## Dependency Graph

```
WP-5 (Plugin Constraints) ─────────┐
                                   │
WP-1 (Analytics Core Removal) ─────┼───► WP-2 (Plugin Catalog)
                                   │           │
                                   │           ▼
WP-4 (Build Plugins) ──────────────┘     WP-3 (ModuleIngestPlugin)
```

**Execution Order:**
1. **WP-1** and **WP-4** can run in parallel (independent)
2. **WP-5** depends on WP-1 (shares `plugin_constraints.py` fixes)
3. **WP-2** depends on WP-1 (needs analytics.core gone first)
4. **WP-3** is independent (can run anytime)

---

## Execution Order

### Phase 2A: Foundation (Independent)

| Order | Work Package | Est. Time | Risk |
|-------|--------------|-----------|------|
| 1 | WP-1: Analytics Core Removal | 30 min | Low |
| 1 | WP-4: Build Plugins Consolidation | 45 min | Medium |
| 1 | WP-3: ModuleIngestPlugin Removal | 15 min | Low |

### Phase 2B: Consolidation (Dependent)

| Order | Work Package | Est. Time | Risk |
|-------|--------------|-----------|------|
| 2 | WP-5: Plugin Constraints Rewrite | 30 min | Medium |
| 3 | WP-2: Plugin Catalog Consolidation | 45 min | Medium |

---

## Verification Checklist

### Pre-Implementation

- [ ] Backup current state (git stash or branch)
- [ ] Run full test suite to establish baseline
- [ ] Document current import graph

### Post-Implementation

- [ ] All quality checks pass:
  ```bash
  uv run ruff check --fix
  uv run pyright --warnings --pythonversion=3.13
  uv run pyrefly check
  uv run pytest -q
  ```

- [ ] No remaining references to deleted modules:
  ```bash
  grep -r "analytics.core" src/ tests/ --include="*.py" | grep -v "__pycache__"
  grep -r "analytics.plugins.registration" src/ tests/ --include="*.py" | grep -v "__pycache__"
  grep -r "ModuleIngestPlugin" src/ tests/ --include="*.py" | grep -v "__pycache__"
  grep -r "build.plugins" src/ tests/ --include="*.py" | grep -v "__pycache__"
  ```

- [ ] Build registry is the single source of truth:
  ```python
  from codeintel.build.plugin_registry import get_all_plugins
  plugins = get_all_plugins()
  assert len(plugins) >= 40  # All plugins registered
  ```

- [ ] CLI commands still work:
  ```bash
  codeintel build run --targets modules --dry-run
  codeintel graph targets-list
  ```

---

## Rollback Strategy

Each work package is designed for independent rollback:

| WP | Rollback Action |
|----|-----------------|
| WP-1 | Restore `analytics/core/` from git |
| WP-2 | Restore `registration.py` and `plugin_catalog.py` imports |
| WP-3 | Restore `modules_plugin.py` and `__init__.py` export |
| WP-4 | Restore `build/plugins.py` and test imports |
| WP-5 | Revert `plugin_constraints.py` changes |

---

## Post-Decommissioning Cleanup

### AGENTS.md Updates

The following AGENTS.md sections reference deleted/moved modules:

```markdown
# Lines 1087-1092 reference codeintel.analytics.core.*
from codeintel.analytics.core.base import ConfiguredTableWriterPlugin
from codeintel.analytics.core.execution_context import PluginExecutionContext
from codeintel.analytics.core.plugin_protocol import PluginStage
```

**Action:** Update to reference `codeintel.build.plugin` and `codeintel.build.context`.

### Documentation Updates

- [ ] Update `docs/plugin_catalog.md` if regenerated
- [ ] Update any architecture diagrams referencing analytics.core
- [ ] Add migration notes to `docs/Hamilton_integration/`

### Future Considerations

1. **Test Coverage**: New tests needed for:
   - Plugin constraint extraction from build registry
   - Schema registry `producers_of`/`consumers_of` methods
   
2. **Dataset Flow CLI**: The `codeintel dataset flow` command should work after these changes:
   ```bash
   codeintel dataset flow analytics.function_metrics
   # Should show producers/consumers from build registry
   ```

3. **Plugin Metadata Enrichment**: Consider adding `produces_tables`/`consumes_tables` to all `TargetPlugin` definitions in the build registry for complete data lineage.

---

## Summary of Deletions

| Category | Files Deleted | Lines Removed |
|----------|---------------|---------------|
| Analytics Core | 2 files (+ test dir) | ~50 |
| Plugin Catalog | 1 file | ~230 |
| Registration | 1 file | ~130 |
| ModuleIngestPlugin | 1 file | ~240 |
| Build Plugins | 1 file | ~240 |
| **Total** | **6+ files** | **~890 lines** |

---

## References

- [Legacy Decommissioning Summary](./Legacy_Decommissioning_Summary.md) — Phase 1 completion
- [Phase 3 & 4 Implementation Plan](./Phase3_4_Aligned_Implementation_Plan.md) — Hamilton architecture
- [Hamilton Phase 4](./Hamilton_apache_phase4.md) — Asset catalog specs
- Build Registry: `src/codeintel/build/plugin_registry.py`
- Target Graph: `src/codeintel/build/registry.py`

