# Ingestion-Core Consolidation Analysis

## Executive Summary

The ingestion package has significant architectural overlap with core. While `ingestion/resources/` already re-exports from `codeintel.core.resources`, there are substantial opportunities to leverage core base classes, particularly in:

1. **Execution Context** - Near-duplicate implementations
2. **Scratch Store** - Identical implementations
3. **Result Types** - Similar but divergent structures
4. **ValidationResult** - Duplicate implementations
5. **Resource Hints** - Near-identical structures
6. **Traits** - Partial overlap

---

## Detailed Analysis

### 1. Scratch Store - HIGHEST PRIORITY (Identical Duplication)

| Feature | core.PluginScratch | ingestion.IngestRuntimeScratch |
|---------|-------------------|-------------------------------|
| `declare(key, value)` | ✓ | ✓ |
| `consume(key, default)` | ✓ | ✓ |
| `has(key)` | ✓ | ✓ |
| `register_cleanup(callback)` | ✓ | ✓ |
| `cleanup()` | ✓ | ✓ |
| `keys()` | ✓ | ✓ |
| `__len__()` | ✓ | ✓ |

**Recommendation**: Delete `IngestRuntimeScratch`, use `PluginScratch` from core.

**Files to modify**:
- `ingestion/plugins/protocol.py` - Remove `IngestRuntimeScratch`, import from core
- `ingestion/core/execution_context.py` - Use `PluginScratch`
- Update all ingestion tests

---

### 2. ValidationResult - HIGH PRIORITY (Identical Duplication)

Both `core.plugins.protocol.ValidationResult` and `ingestion.core.base.ValidationResult` have:
- `valid: bool`
- `errors: tuple[str, ...]`
- `success()` factory
- `failure(errors)` factory

Core version also has `warnings` field.

**Recommendation**: Delete ingestion's `ValidationResult`, use core's version.

**Files to modify**:
- `ingestion/core/base.py` - Import from core
- Update usages throughout ingestion

---

### 3. Resource Hints - HIGH PRIORITY (Near-identical)

| Field | core.PluginResourceHints | ingestion.IngestResourceHints |
|-------|-------------------------|------------------------------|
| `max_runtime_ms` | ✓ | ✓ |
| `max_memory_mb` | ✓ (named differently) | `memory_mb_hint` |
| `cpu_intensive` | ✓ | ✓ |
| `io_intensive` | ✓ | ✓ |
| `requires_gpu` | ✓ | - |
| `priority` | ✓ | - |

**Design Options**:

**Option A: Use Core's ResourceHints Directly**
- Rename `memory_mb_hint` to `max_memory_mb` in ingestion
- Ignore unused fields (`requires_gpu`, `priority`)

**Option B: Create IngestionResourceHints as Subclass**
- Inherit from core, add/override as needed

**Recommendation**: Option A - the extra fields in core don't hurt, and using a single type simplifies the system.

---

### 4. Execution Context - MEDIUM PRIORITY (Significant Overlap)

**Current core.PluginExecutionContext fields**:
```python
gateway: StorageGateway
snapshot: SnapshotRef
run_id: str | None
resources: ResourceRegistry
configs: ConfigProvider
scratch: PluginScratch
paths: BuildPaths | None
options: dict | None
plugin_name: str | None
extra: dict
run_context: RunContext | None
```

**Current ingestion.IngestExecutionContext fields**:
```python
gateway: StorageGateway
snapshot: SnapshotRef
paths: BuildPaths  # Required vs optional
code_profile: ScanProfile  # Ingestion-specific
config_profile: ScanProfile  # Ingestion-specific
tools: ToolsConfig  # Ingestion-specific
resources: ResourceRegistry
scratch: IngestRuntimeScratch
configs: ConfigRegistry  # Different type
plugin_name: str | None
run_id: str | None
run_context: RunContext | None
_plugin_start_times: dict  # Ingestion-specific
_plugin_durations: dict  # Ingestion-specific
```

**Design Options**:

**Option A: Full Inheritance (Recommended)**
```python
@dataclass
class IngestExecutionContext(PluginExecutionContext):
    """Execution context for ingestion plugins."""
    
    # Ingestion-specific fields
    code_profile: ScanProfile
    config_profile: ScanProfile
    tools: ToolsConfig = field(default_factory=ToolsConfig.default)
    _plugin_start_times: dict[str, float] = field(default_factory=dict)
    _plugin_durations: dict[str, float] = field(default_factory=dict)
    
    # Ingestion uses ConfigRegistry, not ConfigProvider
    configs: ConfigRegistry = field(default_factory=ConfigRegistry)  # Override type
```

**Challenges**:
- `configs` has different type (`ConfigProvider` vs `ConfigRegistry`)
- `paths` is optional in core, required in ingestion
- Need to handle the type difference for `configs`

**Option B: Composition**
```python
@dataclass
class IngestExecutionContext:
    """Execution context for ingestion plugins."""
    
    _base: PluginExecutionContext
    code_profile: ScanProfile
    config_profile: ScanProfile
    tools: ToolsConfig
    
    @property
    def gateway(self) -> StorageGateway:
        return self._base.gateway
    # ... delegate all common properties
```

**Challenges**:
- More boilerplate
- Less clean than inheritance

**Recommendation**: Option A with careful handling of the `configs` type difference. We may need to:
1. Make `ConfigProvider` and `ConfigRegistry` share a protocol
2. Or use a union type
3. Or make ingestion use `ConfigProvider` with adapter

---

### 5. Result Types - MEDIUM PRIORITY (Similar but Different)

**Core PluginResult**:
```python
success: bool
message: str | None
error: str | None
error_kind: str | None
row_counts: Mapping[str, int] | None
meta: dict[str, Any]
started_at: datetime | None
ended_at: datetime | None
duration_ms: float | None
```

**Ingestion IngestPluginResult**:
```python
success: bool
row_counts: Mapping[str, int] | None
error: str | None
error_kind: str | None
skipped: bool
skip_reason: str | None
artifacts: Mapping[str, Path] | None
input_hash: str | None
options_hash: str | None
```

**Design Options**:

**Option A: Unified Result Type**
Add all fields to a single type (with sensible defaults):
```python
@dataclass(frozen=True)
class PluginResult:
    success: bool = True
    message: str | None = None
    error: str | None = None
    error_kind: str | None = None
    row_counts: Mapping[str, int] | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_ms: float | None = None
    skipped: bool = False
    skip_reason: str | None = None
    artifacts: Mapping[str, Path] | None = None
    input_hash: str | None = None
    options_hash: str | None = None
```

**Option B: Base + Subclass**
```python
# In core
@dataclass(frozen=True)
class BasePluginResult:
    success: bool = True
    error: str | None = None
    error_kind: str | None = None
    row_counts: Mapping[str, int] | None = None

# In ingestion
@dataclass(frozen=True)
class IngestPluginResult(BasePluginResult):
    skipped: bool = False
    skip_reason: str | None = None
    artifacts: Mapping[str, Path] | None = None
    # ...
```

**Recommendation**: Option B - keeps domain-specific concerns separate while sharing common base.

---

### 6. Traits - LOW PRIORITY (Partial Overlap)

**Already in core.plugins.traits**:
- `IsolatedPlugin`
- `RetryablePlugin`
- `ProgressReportingPlugin`
- `CacheAwarePlugin`

**In ingestion.core.traits (domain-specific)**:
- `IncrementalIngestPlugin` - ingestion-specific (compute_input_hash, is_unchanged)
- `ToolAwarePlugin` - could be generalized
- `TrackerAwarePlugin` - ingestion-specific

**Recommendation**: Keep most ingestion traits in ingestion since they're domain-specific. Consider moving `ToolAwarePlugin` to core if useful for other domains.

---

## Implementation Plan

### Phase 1: Low-Risk Quick Wins

**1.1. Consolidate PluginScratch**
- Remove `IngestRuntimeScratch` from `ingestion/plugins/protocol.py`
- Update `ingestion/core/execution_context.py` to use `PluginScratch` from core
- Update imports throughout ingestion

**1.2. Consolidate ValidationResult**
- Remove `ValidationResult` from `ingestion/core/base.py`
- Import from `codeintel.core.plugins.protocol`
- Update imports in base plugin classes

**1.3. Consolidate ResourceHints**
- Remove `IngestResourceHints` from `ingestion/plugins/protocol.py`
- Use `PluginResourceHints` from core
- Update field references (`memory_mb_hint` → `max_memory_mb`)

### Phase 2: Execution Context Unification

**2.1. Analyze Config Type Difference**
- `ConfigProvider` (core) vs `ConfigRegistry` (ingestion)
- Create shared protocol or adapter

**2.2. Make IngestExecutionContext Extend PluginExecutionContext**
- Requires careful handling of `configs` type
- Keep ingestion-specific fields

### Phase 3: Result Type Consolidation (Optional)

**3.1. Create Base Result in Core**
- Extract common fields
- Keep factory methods

**3.2. Make IngestPluginResult Extend Base**
- Add ingestion-specific fields

---

## Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| PluginScratch consolidation | Low | Direct drop-in replacement |
| ValidationResult consolidation | Low | Same interface |
| ResourceHints consolidation | Low | Minor field rename |
| ExecutionContext inheritance | Medium | Config type difference needs resolution |
| Result type consolidation | Medium | Factory methods may need adjustment |

---

## Files Summary

### Files to Modify

| File | Phase | Changes |
|------|-------|---------|
| `ingestion/plugins/protocol.py` | 1 | Remove IngestRuntimeScratch, IngestResourceHints |
| `ingestion/core/base.py` | 1 | Remove ValidationResult, import from core |
| `ingestion/core/execution_context.py` | 1+2 | Use PluginScratch, extend base context |
| `ingestion/plugins/registry.py` | 1 | Update imports |
| `ingestion/runtime/executor.py` | 1 | Update scratch type |
| `ingestion/recipes/executor.py` | 1 | Update scratch type |
| Various test files | All | Update imports |

### Files to Create

None required - all changes are consolidations.

### Files to Delete

None - just removing duplicate classes from existing files.

---

## Quality Gates

- Zero `# type: ignore` suppressions
- Zero `# noqa` suppressions
- All ruff checks pass
- All pyright checks pass (strict mode)
- All pyrefly checks pass
- Existing ingestion tests pass
- Import tests verify backward compatibility

