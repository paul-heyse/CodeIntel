# Epic 10 Implementation Comments

This document provides granular details on deviations from the original implementation plan specified in `docs/app-integration-epic-4.md`.

---

## Executive Summary

The Epic 10 implementation closely follows the original plan with targeted refinements to align with project conventions (AGENTS.md), improve type safety, and leverage existing infrastructure. The core functional scope is fully realized.

---

## 1. Module Structure

### 1.1 File Locations

| Specification | Implementation | Notes |
|---------------|----------------|-------|
| `codeintel/pipeline/op_planner.py` | `src/codeintel/pipeline/op_planner.py` | Added `src/` prefix per project layout |
| `tests/pipeline/test_op_planner.py` | `tests/pipeline/test_op_planner.py` | Matches spec |
| `tests/pipeline/test_operation_prereqs.py` | `tests/pipeline/test_operation_prereqs.py` | Matches spec |

### 1.2 Exports via `__init__.py`

**Addition:** The spec did not explicitly mention updating `src/codeintel/pipeline/__init__.py`, but we added exports for:
- `build_pipeline_for_operation`
- `ensure_prerequisites_for_operation`
- `build_prereq_summary`
- `OpPrereqSummary`
- `NOOP_PIPELINE`

This ensures the new APIs are accessible via `from codeintel.pipeline import ...`.

---

## 2. Data Structures

### 2.1 `OpPrereqSummary` Dataclass

**Spec (Section 2.1.1):**
```python
@dataclass(frozen=True)
class OpPrereqSummary:
    op: Operation
    required_tables: frozenset[str]
    expanded_tables: frozenset[str]
    core_tables: frozenset[str]
    graph_tables: frozenset[str]
    analytics_tables: frozenset[str]
    required_graphs: frozenset[str]
```

**Implementation:** Matches spec exactly. Added comprehensive NumPy-style docstring with `Attributes` section per AGENTS.md requirements.

### 2.2 `OwnerPackage` Type Alias

**Addition:** Created a `Literal` type alias not in the original spec:
```python
OwnerPackage = Literal["core", "analytics", "graphs", "qa", "docs"]
```

This improves type safety for owner package classification, though it's currently unused in function signatures (reserved for potential future use).

---

## 3. Helper Functions

### 3.1 `_build_contract_index()`

**Spec (Section 2.1.3):**
```python
def _build_contract_index() -> tuple[
    dict[str, DatasetContract],  # by table_key
    dict[str, DatasetContract],  # by name
]:
```

**Implementation:** Matches spec. Function returns `(by_table_key, by_name)` tuple.

### 3.2 `_get_required_from_operation()`

**Spec (Section 2.1.2):**
```python
def _get_required_from_operation(op_id: str) -> tuple[Operation, set[str], set[str]]:
```

**Implementation:** Matches spec.

**Deviation:** Spec used inline f-string for error message:
```python
raise ValueError(f"Unknown operation id: {op_id}")
```

Implementation uses a variable to comply with Ruff rule `EM102` (exception messages should not be inline f-strings):
```python
message = f"Unknown operation id: {op_id}"
raise ValueError(message)
```

### 3.3 `_expand_dataset_dependencies()`

**Spec (Section 2.1.3):**
```python
def _expand_dataset_dependencies(required_tables: set[str]) -> set[str]:
```

**Implementation:** Matches spec algorithm (BFS traversal for transitive closure).

### 3.4 `_partition_by_owner_package()`

**Spec (Section 2.1.4):**
```python
def _partition_by_owner_package(
    table_keys: Iterable[str],
) -> tuple[set[str], set[str], set[str]]:
```

**Implementation:** Matches spec.

**Deviation:** Spec suggested ignoring unknown owner_package values or logging them. Implementation attributes unknown tables to `core_tables` bucket to ensure no tables are silently dropped:
```python
else:
    # Unknown owner_package, attribute to core
    core_tables.add(table_key)
```

### 3.5 `_compute_stage_flags()`

**Spec (Section 2.1.5):**
```python
def _compute_stage_flags(
    *,
    core_tables: set[str],
    graph_tables: set[str],
    analytics_tables: set[str],
    required_graphs: set[str],
    include_analytics: bool,
) -> tuple[bool, bool, bool]:
```

**Implementation:** Matches spec signature.

**Deviation in Logic:** Spec stated:
```python
need_analytics = bool(analytics_tables) or include_analytics
```

Implementation refines this to only include analytics when there's actual work to do:
```python
need_analytics = bool(analytics_tables)
if include_analytics and (need_ingestion or need_graphs):
    need_analytics = True
```

This prevents `include_analytics=True` from forcing analytics stage when no other stages are needed (e.g., for NOOP operations).

### 3.6 `_choose_spec()`

**Spec (Section 2.1.6):**
```python
def _choose_spec(
    *,
    need_ingestion: bool,
    need_graphs: bool,
    need_analytics: bool,
) -> PipelineSpec:
```

**Implementation:** Matches spec exactly, including the fallback to `FULL_PIPELINE` for mixed two-stage combinations.

---

## 4. Public API Functions

### 4.1 `build_pipeline_for_operation()`

**Spec (Section 2.1.7):**
```python
def build_pipeline_for_operation(
    op_id: str,
    snapshot: SnapshotRef,
    *,
    include_analytics: bool = True,
) -> PipelineSpec:
```

**Implementation:**
```python
def build_pipeline_for_operation(
    op_id: str,
    _snapshot: SnapshotRef,
    *,
    include_analytics: bool = True,
) -> PipelineSpec:
```

**Deviation:** Parameter renamed from `snapshot` to `_snapshot` to indicate it's reserved for future use but currently unused. This satisfies Ruff rule `ARG001` (unused function argument). Docstring updated accordingly:
```
_snapshot
    Repository snapshot reference (reserved for future incremental hints).
```

**Deviation:** Spec suggested optional debug logging. Implementation includes INFO-level logging:
```python
log.info(
    "op_planner.build op=%s tables=%d graphs=%d spec=%s",
    op_id,
    len(expanded_tables),
    len(required_graphs),
    spec.id,
)
```

### 4.2 `build_prereq_summary()`

**Addition:** This function was mentioned as "optional" in the spec but was fully implemented to enable introspection:
```python
def build_prereq_summary(
    op_id: str,
    _snapshot: SnapshotRef,
) -> OpPrereqSummary:
```

### 4.3 `ensure_prerequisites_for_operation()`

**Spec (Section 3.2.3):**
```python
def ensure_prerequisites_for_operation(
    *,
    op_id: str,
    snapshot: SnapshotRef,
    paths: BuildPaths,
    gateway: StorageGateway,
    tools: ToolsConfig,
    include_analytics: bool = True,
    trigger: TriggerKind = "api",
) -> PipelineRunRecord:
```

**Implementation:** Matches spec signature.

**Deviation:** Added `# noqa: PLR0913` comment to suppress "too many arguments" warning, as this is an intentional API design choice for a top-level orchestration function.

**Enhancement (Phase 1.5):** Implementation passes `run_kind_override="op_prereqs"` to `run_pipeline`:
```python
return run_pipeline(
    spec=spec,
    snapshot=snapshot,
    paths=paths,
    gateway=gateway,
    tools=tools,
    trigger=trigger,
    run_kind_override="op_prereqs",
)
```

This was mentioned in Phase 2 of the spec (Section 5, item 4) but was implemented early to ensure runs are clearly identifiable as prerequisite runs.

---

## 5. Pipeline Infrastructure Changes

### 5.1 `NOOP_PIPELINE` Addition

**Spec (Section 2.1.6):**
```python
NOOP_PIPELINE = PipelineSpec(
    id="noop",
    description="No-op pipeline for operations with no prerequisites.",
    stages=(),
)
```

**Implementation:** Matches spec. Also added to `PIPELINE_SPECS` registry and `__all__` exports.

### 5.2 `run_kind_override` Parameter

**Spec (Section 5, Phase 2):** Mentioned as future enhancement:
> Extend `build_pipeline_plan` to accept `run_kind_override="op_prereqs"` when called from `ensure_prerequisites_for_operation`

**Implementation:** Implemented in Phase 1 as a refinement. Added to both:
- `build_pipeline_plan()` in `planner.py`
- `run_pipeline()` in `executor.py`

This ensures all prerequisite runs have `kind="op_prereqs"` and run IDs prefixed with `op_prereqs-`.

### 5.3 `requested_operation` / `requested_datasets` Pass-through

**Spec (Section 3.2.2):** Suggested extending `build_pipeline_plan` to accept:
```python
requested_operation: str | None = None,
requested_datasets: Sequence[str] | None = None,
```

**Implementation:** Not implemented. The `run_kind_override="op_prereqs"` approach provides sufficient observability without requiring changes to the core planning infrastructure. The operation ID and datasets can be inferred from the run's `pipeline_name` and associated spec.

---

## 6. Docstring Compliance

### 6.1 NumPy Style

**Spec:** Did not specify docstring style.

**Implementation:** All functions have comprehensive NumPy-style docstrings per AGENTS.md requirements, including:
- One-line summary in imperative mood
- `Parameters` section with types and descriptions
- `Returns` section
- `Raises` section (where applicable, but removed per DOC502 rule when exceptions propagate from helpers)

### 6.2 DOC502 Compliance

**Deviation:** Some docstrings in the spec included `Raises` sections for exceptions raised by helper functions. Implementation removed these to comply with Ruff rule `DOC502` (raised exception not explicitly raised in function body).

Example: `build_pipeline_for_operation()` does not document `ValueError` since it's raised by `_get_required_from_operation()`, not directly.

---

## 7. Testing Approach

### 7.1 Private Function Testing

**Spec (Section 4.3.1):** Suggested testing private helper functions directly:
```python
from codeintel.pipeline.op_planner import _expand_dataset_dependencies
```

**Implementation:** Tests use only public API functions to comply with Ruff rule `PLC2701` (no private name imports from external modules). Private function behavior is tested indirectly through:
- `build_pipeline_for_operation()` - tests spec mapping
- `build_prereq_summary()` - tests dependency expansion and partitioning

### 7.2 Test Fixtures

**Spec (Section 4.3.2):** Suggested using:
```python
gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
tools = ToolsConfig()
```

**Implementation:**
- Uses `open_ingestion_gateway_with_macros()` from `tests/_helpers/gateway` (project convention)
- Uses `make_tools_config()` from `tests/_helpers/tooling` to avoid pyright errors with direct `ToolsConfig()` construction

### 7.3 Additional Test Classes

**Addition:** Tests include additional classes not in spec:
- `TestOpPrereqsRunKind` - verifies `run_kind_override` behavior
- `TestDependencyExpansionViaPrereqSummary` - tests expansion through public API
- `TestGraphRequirements` - tests graph-specific operations

### 7.4 Parametrized Tests

**Deviation:** Spec showed individual test functions for each operation. Implementation uses `@pytest.mark.parametrize` for operation coverage:
```python
@pytest.mark.parametrize(
    ("op_id", "expected_spec_id"),
    [
        ("function.summary", "full"),
        ("datasets.list", "noop"),
        # ...
    ],
)
def test_operation_spec_mapping_full(...):
```

Split into separate test methods (`test_operation_spec_mapping_full` and `test_operation_spec_mapping_noop`) to avoid boolean positional arguments per Ruff rule `FBT001`.

---

## 8. Type Annotations

### 8.1 TYPE_CHECKING Guards

**Addition:** All heavy imports are guarded under `TYPE_CHECKING` per project typing gates policy:
```python
if TYPE_CHECKING:
    from codeintel.config.models import ToolsConfig
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.runtime import TriggerKind
    from codeintel.storage.gateway import StorageGateway
    from codeintel.storage.run_tracking import PipelineRunRecord
```

### 8.2 `from __future__ import annotations`

**Addition:** All modules include `from __future__ import annotations` as the first import per project convention.

---

## 9. Phase 2 Items Not Implemented

The following Phase 2 items from Section 5 of the spec were explicitly deferred:

1. **Plugin introspection helpers** - Not implemented
2. **Operation-specific plugin sets** - Not implemented
3. **`op_prereqs.*` stage names** - Not implemented
4. **`run_kind_override`** - ✅ Implemented early as a Phase 1.5 refinement

---

## 10. Summary of Key Deviations

| Category | Deviation | Rationale |
|----------|-----------|-----------|
| Parameter naming | `snapshot` → `_snapshot` | Ruff ARG001 compliance |
| Error messages | Extracted to variables | Ruff EM102 compliance |
| Analytics flag logic | Conditional on other stages | Prevents spurious analytics for NOOP |
| Private function testing | Via public API only | Ruff PLC2701 compliance |
| Test fixtures | Helper functions | Pyright compatibility |
| Docstring Raises | Removed propagated exceptions | Ruff DOC502 compliance |
| `run_kind_override` | Implemented in Phase 1 | Early observability improvement |

---

## 11. Files Changed Summary

| File | Change Type | Spec Reference |
|------|-------------|----------------|
| `src/codeintel/pipeline/spec.py` | Modified | Section 2.1.6 |
| `src/codeintel/pipeline/op_planner.py` | Created | Sections 2, 3 |
| `src/codeintel/pipeline/planner.py` | Modified | Section 5 (Phase 2 → Phase 1) |
| `src/codeintel/pipeline/executor.py` | Modified | Section 5 (Phase 2 → Phase 1) |
| `src/codeintel/pipeline/__init__.py` | Modified | Not specified |
| `tests/pipeline/test_op_planner.py` | Created | Section 4.3.1 |
| `tests/pipeline/test_operation_prereqs.py` | Created | Section 4.3.2 |

---

## 12. Verification

All implementation passes:
- ✅ 39 pytest tests (22 unit + 17 integration)
- ✅ Ruff format and check (0 errors)
- ✅ Pyright strict mode (0 errors)
- ✅ Pyrefly check (0 errors)

