# Graphs Package Cleanup Plan

> **Generated:** 2025-12-13  
> **Updated:** 2025-12-13 (Phases 1-3 completed)  
> **Package:** `codeintel.graphs`  
> **Status:** Phases 1-3 Complete, Phase 4 Ready for Review

## Executive Summary

The `graphs` package has undergone significant cleanup with Phases 1-3 now complete:

**Completed:**
- ~~2 empty directories deleted~~ ✅
- ~~9 unused backward-compatibility aliases removed~~ ✅
- ~~1 deprecated stub module removed~~ ✅
- ~~1 unused protocol removed~~ ✅

**Remaining Opportunities (Phase 4+):**
- Catalog layer consolidation (3 overlapping data classes)
- Ports layer simplification (3 single-implementation ports)
- Data loading function consolidation
- Compute layer documentation improvements

---

## Table of Contents

1. [Completed Work](#1-completed-work)
2. [Catalog Layer Consolidation](#2-catalog-layer-consolidation)
3. [Data Class Unification](#3-data-class-unification)
4. [Ports Layer Simplification](#4-ports-layer-simplification)
5. [Compute Layer Opportunities](#5-compute-layer-opportunities)
6. [Implementation Checklist](#6-implementation-checklist)

---

## 1. Completed Work

### Phase 1: Dead Directories and Aliases ✅

**Completed 2025-12-13**

- Deleted empty directories: `core/`, `runtime/`
- Removed 9 unused backward-compatibility aliases:
  - 4 from `graphs/engine/views.py`
  - 4 from `graphs/compute/callgraph/resolution.py`
  - 1 from `graphs/compute/callgraph/collection.py`

### Phase 2: Deprecated Adapters Removal ✅

**Completed 2025-12-13**

- Deleted `src/codeintel/graphs/adapters/` directory
- Updated `graphs/__init__.py` to remove adapters import and export
- Updated module docstring to remove adapters references

### Phase 3: ParsingPort Protocol Removal ✅

**Completed 2025-12-13**

- Removed unused `ParsingPort` protocol from `graphs/ports/parsing.py`
- Kept data classes: `ParsedFunction`, `ParsedModule`, `ParseError`, `ParseResult`
- Updated `graphs/ports/__init__.py` exports

---

## 2. Catalog Layer Consolidation

### Status: 🟡 Medium Priority (Recommended for Phase 4)

Deep analysis reveals significant overlap between catalog-related classes that creates cognitive overhead and maintenance burden.

### Current Architecture

```
graphs/catalog.py
├── FunctionSpan (frozen dataclass)
├── FunctionSpanIndex (lookup structure)
├── FunctionMeta (frozen dataclass, adds URN)
├── FunctionCatalog (main catalog class)
├── FunctionCatalogProvider (Protocol)
└── FunctionCatalogService (service wrapper)

graphs/resources/catalog.py
└── CatalogResource (resource provider wrapper)

graphs/ports/catalog.py
├── FunctionSpanData (frozen dataclass, duplicates FunctionSpan + URN)
└── CatalogPort (Protocol)
```

### Identified Redundancies

#### 2.1 Three Overlapping Span Data Classes

| Class | Location | Fields | Used By |
|-------|----------|--------|---------|
| `FunctionSpan` | `catalog.py` | goid, rel_path, qualname, start_line, end_line | FunctionSpanIndex, FunctionCatalog |
| `FunctionMeta` | `catalog.py` | goid, urn, rel_path, qualname, start_line, end_line | FunctionCatalog.functions_by_path |
| `FunctionSpanData` | `ports/catalog.py` | goid, rel_path, qualname, start_line, end_line, urn, local_name (property) | CatalogResource, CatalogPort |

**Issue:** `FunctionSpanData` is essentially `FunctionSpan` + `urn` + `local_name` property. `FunctionMeta` is `FunctionSpan` + `urn`. All three represent the same concept.

**Recommendation:** Unify into a single `FunctionSpan` class with optional `urn` field:

```python
@dataclass(frozen=True)
class FunctionSpan:
    """Unified function span representation."""
    goid: int
    rel_path: str
    qualname: str
    start_line: int
    end_line: int
    urn: str | None = None  # Optional, populated when available
    
    @property
    def local_name(self) -> str:
        return self.qualname.rsplit(".", maxsplit=1)[-1]
```

#### 2.2 Dual Service/Resource Classes

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `FunctionCatalogService` | Analytics-facing wrapper | catalog(), local_name_map(), urn_for_goid(), lookup_goid() |
| `CatalogResource` | Graph plugin wrapper | function_spans, spans_for_path(), local_name_map(), lookup_goid() |

**Issue:** Both wrap `FunctionCatalog` with nearly identical methods. `CatalogResource` converts to `FunctionSpanData`; `FunctionCatalogService` uses `FunctionSpan` directly.

**Recommendation:** Merge into single `CatalogService` that implements both `FunctionCatalogProvider` and resource protocols:

```python
@dataclass
class CatalogService:
    """Unified catalog access for graphs and analytics."""
    
    RESOURCE_NAME: ClassVar[str] = "catalog"
    _catalog: FunctionCatalog
    
    # ResourceProvider protocol
    def get(self) -> CatalogService: return self
    def invalidate(self) -> None: ...
    
    # FunctionCatalogProvider protocol  
    def catalog(self) -> FunctionCatalog: ...
    def urn_for_goid(self, goid: int) -> str | None: ...
    
    # Unified span access
    @property
    def function_spans(self) -> Sequence[FunctionSpan]: ...
    def spans_for_path(self, rel_path: str) -> Sequence[FunctionSpan]: ...
```

### Migration Impact

| Consumer | Current Import | After Consolidation |
|----------|----------------|---------------------|
| Analytics plugins | `FunctionCatalogService` | `CatalogService` |
| Graph builders | `CatalogResource` | `CatalogService` |
| Tests | Both | `CatalogService` |

**Estimated Effort:** 2-3 hours
**Risk:** Low (internal refactoring, no API changes)

---

## 3. Data Class Unification

### Status: 🟢 Lower Priority (Part of Catalog Consolidation)

### Current State

The `graphs/catalog.py` module defines three loading functions with overlapping logic:

```python
def load_function_spans(gateway, *, repo, commit) -> list[FunctionSpan]:
    """Load spans without URN."""
    
def load_function_index(gateway, *, repo, commit) -> FunctionSpanIndex:
    """Load spans and wrap in index."""
    
def load_function_catalog(gateway, *, repo, commit) -> FunctionCatalog:
    """Load spans with URN and module mapping."""
```

### Recommendation

Consolidate into a single loader that returns the full catalog (most common use case):

```python
def load_catalog(gateway: StorageGateway, *, repo: str, commit: str) -> FunctionCatalog:
    """Load function catalog with spans, URNs, and module mapping."""
    ...

# Convenience accessors if needed
def load_span_index(gateway, *, repo, commit) -> FunctionSpanIndex:
    """Load catalog and return just the span index."""
    return load_catalog(gateway, repo=repo, commit=commit).function_index
```

**Benefits:**
- Single source of truth for loading logic
- Reduces code duplication
- Simplifies testing

---

## 4. Ports Layer Simplification

### Status: 🟢 Lower Priority (Architectural)

### Current State (Post-Phase 3)

| Port | Location | Implementation | Used By |
|------|----------|----------------|---------|
| `StoragePort` | `ports/storage.py` | `StorageResource` | Internal only |
| `CatalogPort` | `ports/catalog.py` | `CatalogResource` | Internal only |
| `EnginePort` | `ports/engine.py` | `GraphResource` | Internal only |

### Analysis

1. **Each port has exactly one implementation**
2. **Ports are used internally only** - no external consumers
3. **No DI-based testing** - tests don't swap implementations
4. **Data classes in ports are used** - `FunctionSpanData`, `GraphData`, `QueryResult`, etc.

### Options

#### Option A: Keep Ports (Status Quo)
- **Pros:** Follows hexagonal architecture, ready for future flexibility
- **Cons:** Over-engineered for current single-implementation usage

#### Option B: Inline Protocols into Resources
- **Pros:** Simpler codebase, fewer files
- **Cons:** Loses architectural clarity

#### Option C: Consolidate Port/Resource Pairs
- Move `StoragePort` methods into `StorageResource`
- Move `CatalogPort` methods into `CatalogResource` (or unified `CatalogService`)
- Move `EnginePort` methods into `GraphResource`
- Keep data classes in ports (they're used as DTOs)

### Recommendation

**Option C** with catalog consolidation from Section 2. The ports layer adds value for data classes but the protocol/resource split is unnecessary overhead.

**Post-consolidation structure:**
```
graphs/ports/
├── __init__.py          # Export data classes
├── catalog.py           # FunctionSpanData (if not unified)
├── engine.py            # GraphData  
├── parsing.py           # ParsedFunction, ParsedModule, etc.
└── storage.py           # QueryResult, BatchResult

graphs/resources/
├── __init__.py
├── catalog.py           # CatalogService (unified)
├── graphs.py            # GraphResource (with EnginePort inlined)
└── storage.py           # StorageResource (with StoragePort inlined)
```

---

## 5. Compute Layer Opportunities

### Status: 🟢 Lower Priority (Documentation/Polish)

The compute layer is well-structured with clean separation of concerns. Minor improvements identified:

### 5.1 Callgraph Module Organization

**Current:**
```
compute/callgraph/
├── __init__.py       # Re-exports everything
├── collection.py     # Edge collection visitors (CST/AST)
├── persistence.py    # dedupe_edge_rows, default_edge_key
├── resolution.py     # Callee resolution logic
└── types.py          # CallEdge, ResolutionResult, contexts
```

**Observation:** Module is well-organized. The `persistence.py` module is minimal (80 lines) after cleanup - could be merged into `collection.py` if desired.

### 5.2 Metrics Module

**Current structure is good:**
```
compute/metrics/
├── bipartite.py      # Bipartite graph metrics
├── centrality.py     # PageRank, betweenness, etc.
├── cfg.py            # Control flow metrics
├── community.py      # Community detection
├── components.py     # SCC, connected components
├── coupling.py       # Coupling metrics
├── dfg.py            # Data flow metrics
├── paths.py          # Path-related metrics
├── statistics.py     # Graph statistics
└── structural.py     # Clustering, triangles, etc.
```

**Note:** This is properly used by `analytics.compute.graphs` which wraps these pure functions - correct layered architecture.

### 5.3 Documentation Opportunity

The `compute/__init__.py` docstring references outdated module names. Update to reflect current structure:

```python
"""Pure stateless computation layer for graph operations.

Subpackages
-----------
callgraph/
    Call edge collection, resolution, and persistence utilities
metrics/
    Graph metric computations (centrality, community, structural, etc.)

Modules
-------
cfg
    Control-flow graph construction
dfg  
    Data-flow graph construction
goid
    GOID hash computation and URN building
imports
    Import relationship analysis
symbols
    Symbol use analysis
"""
```

---

## 6. Implementation Checklist

### Completed Phases ✅

#### Phase 1: Dead Directories and Aliases ✅
- [x] Delete `src/codeintel/graphs/core/` directory
- [x] Delete `src/codeintel/graphs/runtime/` directory
- [x] Remove 4 aliases from `graphs/engine/views.py`
- [x] Remove 4 aliases from `graphs/compute/callgraph/resolution.py`
- [x] Remove 1 alias from `graphs/compute/callgraph/collection.py`

#### Phase 2: Deprecated Module Removal ✅
- [x] Remove `src/codeintel/graphs/adapters/` directory
- [x] Update `src/codeintel/graphs/__init__.py` to remove adapters

#### Phase 3: Protocol Simplification ✅
- [x] Remove `ParsingPort` protocol from `graphs/ports/parsing.py`
- [x] Update `graphs/ports/__init__.py` exports

### Remaining Phases

#### Phase 4: Catalog Layer Consolidation (Recommended Next)
- [ ] Unify `FunctionSpan`, `FunctionMeta`, `FunctionSpanData` into single class
- [ ] Merge `FunctionCatalogService` and `CatalogResource` into `CatalogService`
- [ ] Update all consumers (analytics, graphs, tests)
- [ ] Consolidate loading functions
- [ ] Run full test suite

#### Phase 5: Ports Layer Simplification (Optional)
- [ ] Inline `StoragePort` into `StorageResource`
- [ ] Inline `EnginePort` into `GraphResource`
- [ ] Remove empty protocol files
- [ ] Update imports

#### Phase 6: Polish (Optional)
- [ ] Update `compute/__init__.py` docstring
- [ ] Consider merging `persistence.py` into `collection.py`
- [ ] Add module-level examples to key modules

---

## Verification Commands

After implementing changes, run:

```bash
# Type checking
uv run pyright --warnings --pythonversion=3.13
uv run pyrefly check

# Linting
uv run ruff check --fix

# Full test suite
uv run pytest -q

# Verify no dead code introduced
uv run vulture src/codeintel/graphs --min-confidence 90
```

---

## Related Documents

- [BUILD_CLEANUP_PLAN.md](./BUILD_CLEANUP_PLAN.md)
- [BUILD_CONSOLIDATION_PLAN.md](./BUILD_CONSOLIDATION_PLAN.md)
- [BUILD_REFINEMENT_PLAN.md](./BUILD_REFINEMENT_PLAN.md)
