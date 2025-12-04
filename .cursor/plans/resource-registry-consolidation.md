# Resource Registry Consolidation Plan

## Overview

Consolidate graph-specific `ResourceContainer` into the unified `ResourceRegistry` in core, eliminating 7 `# type: ignore` suppressions and removing the dual-resource system in graph contexts.

---

## Phase 1: Enhance Core ResourceRegistry with Factory Support

### 1.1 Add Factory Registration to `core/resources/registry.py`

Add the following to `ResourceRegistry`:

```python
# New instance variable in __init__
self._factories: dict[str, Callable[[], object]] = {}

# New methods
def register_factory(self, name: str, factory: Callable[[], object]) -> None:
    """Register a factory for lazy provider creation."""
    
def _resolve_factory(self, name: str) -> bool:
    """Resolve a factory if not yet instantiated."""

@property
def registered_names(self) -> tuple[str, ...]:
    """Get all registered resource names including factories."""
```

### 1.2 Update Existing Methods to Support Factories

- `has_by_name(name)` - check factories too
- `get_by_name(name)` - resolve factory first
- `require_by_name(name)` - resolve factory first
- `clear()` - also clear factories

### 1.3 Add Convenience Methods

```python
def register_provider(self, provider: object) -> None:
    """Register using RESOURCE_NAME attribute."""
    
def cleanup(self) -> None:
    """Invalidate all and clear (including factories)."""
```

---

## Phase 2: Migrate Graph Context to Unified Registry

### 2.1 Update `graphs/core/context.py`

**Current dual-system:**
```python
@dataclass
class GraphPluginExecutionContext(PluginExecutionContext):
    graph_resources: ResourceContainer = field(default_factory=ResourceContainer)
```

**Migrate to:**
- Remove `graph_resources` field
- Use inherited `resources: ResourceRegistry` from base class
- Update `require()` and `require_graphs()` to use unified registry

### 2.2 Update `GraphPluginExecutionContextBuilder`

- Remove `_graph_resources` field
- Remove `with_graph_resources()` method  
- Update `register_graph_resource()` to use unified registry
- Update `build_graph_context()` to not pass `graph_resources`

---

## Phase 3: Update Graph Plugin Infrastructure

### 3.1 Files to Update

| File | Changes |
|------|---------|
| `graphs/recipes/executor.py` | Use `ResourceRegistry` instead of `ResourceContainer` |
| `graphs/runtime/executor.py` | Use `ResourceRegistry` instead of `ResourceContainer` |
| `graphs/plugins/runner.py` | Use `ResourceRegistry` instead of `ResourceContainer` |
| `graphs/plugins/builders/goid.py` | Use `ResourceRegistry` instead of `ResourceContainer` |
| `graphs/__init__.py` | Update exports |
| `graphs/resources/__init__.py` | Update exports |

---

## Phase 4: Remove ResourceContainer

### 4.1 Delete or Deprecate `graphs/resources/container.py`

**Option A (Recommended):** Make `ResourceContainer` a type alias:
```python
# graphs/resources/container.py
from codeintel.core.resources import ResourceRegistry
ResourceContainer = ResourceRegistry  # Backward compatibility alias
```

**Option B:** Delete the file entirely after updating all imports.

---

## Phase 5: Cleanup Empty Module

### 5.1 Remove `analytics/plugins/graphs/__init__.py`

The module is empty and no longer exports anything. Verify no imports and delete.

---

## Files Modified Summary

| File | Action |
|------|--------|
| `core/resources/registry.py` | MODIFY - Add factory support |
| `core/resources/__init__.py` | MODIFY - Export new methods |
| `graphs/core/context.py` | MODIFY - Remove dual-system |
| `graphs/recipes/executor.py` | MODIFY - Use ResourceRegistry |
| `graphs/runtime/executor.py` | MODIFY - Use ResourceRegistry |
| `graphs/plugins/runner.py` | MODIFY - Use ResourceRegistry |
| `graphs/plugins/builders/goid.py` | MODIFY - Use ResourceRegistry |
| `graphs/__init__.py` | MODIFY - Update exports |
| `graphs/resources/__init__.py` | MODIFY - Update exports |
| `graphs/resources/container.py` | MODIFY - Make type alias |
| `analytics/plugins/graphs/__init__.py` | DELETE |

---

## Quality Gates

- Zero `# type: ignore` suppressions in modified files
- Zero `# noqa` suppressions in modified files
- All ruff checks pass
- All pyright checks pass (strict mode)
- All pyrefly checks pass
- Import tests verify backward compatibility

---

## Risk Assessment

- **Low Risk:** Factory support is additive
- **Medium Risk:** Removing `graph_resources` field changes context structure
- **Mitigation:** Keep `ResourceContainer` as alias for transition period

---

## Implementation Todos

1. Add factory support to core/resources/registry.py
2. Add registered_names property and cleanup method to registry
3. Update graphs/core/context.py to remove dual-system
4. Update graph plugin infrastructure files
5. Make ResourceContainer a type alias
6. Delete empty analytics/plugins/graphs/__init__.py
7. Run comprehensive quality checks

