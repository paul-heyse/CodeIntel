# Build System Refinement Plan

## Overview

This document outlines opportunities to consolidate shared functionality and refine the build system towards a best-in-class implementation.

---

## Part 1: Plugin Migration to MetadataPlugin (42 plugins)

### Problem

All 42 plugins manually implement boilerplate that `MetadataPlugin` already provides:

```python
# Current pattern (repeated 42 times):
class MyPlugin(TargetPlugin):
    plugin_name: ClassVar[str] = "my_plugin"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "..."
    _core_metadata: ClassVar[CorePluginMetadata] = MY_METADATA

    @property
    def metadata(self) -> PluginMetadata:
        return to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        return self._core_metadata
```

### Solution

Migrate to `MetadataPlugin` which provides all this automatically:

```python
# New pattern:
class MyPlugin(MetadataPlugin):
    _core_metadata: ClassVar[CorePluginMetadata] = MY_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        ...
```

### Files to Update

All 42 plugin files in:
- `plugins/analytics/**/*.py` (24 files)
- `plugins/ingestion/**/*.py` (12 files)
- `plugins/graphs/**/*.py` (6 files)

### Migration Script Approach

```python
# Pattern to find and replace:
# 1. Change base class: TargetPlugin → MetadataPlugin
# 2. Remove ClassVar declarations for plugin_name, plugin_version, plugin_description
# 3. Remove @property def metadata() and @property def core_metadata()
# 4. Add __init__ method if options_resolver pattern is used
```

---

## Part 2: Row Count Computation Consolidation

### Problem

8+ identical `_compute_row_counts` implementations scattered across plugins with minor variations.

### Solution

Create a shared utility in `build/plugins/_helpers.py`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

def compute_row_counts(
    ctx: TargetExecutionContext,
    table_keys: tuple[str, ...] | None = None,
) -> dict[str, int]:
    """Compute row counts for output tables in the current snapshot.

    Parameters
    ----------
    ctx
        Execution context with gateway and snapshot.
    table_keys
        Specific table keys to count. If None, uses ctx.contract.table_keys.

    Returns
    -------
    dict[str, int]
        Mapping of table key to row count.
    """
    from codeintel.storage.gateway.protocol import DuckDBCatalogException
    from codeintel.storage.ibis_types import filter_by, ibis_bool
    from ibis.common.exceptions import TableNotFound

    keys = table_keys or ctx.contract.table_keys
    row_counts: dict[str, int] = {}

    for table_key in keys:
        try:
            table = ctx.gateway.ibis.table(table_key)
            count_expr = filter_by(
                table,
                ibis_bool(table.repo == ctx.repo),
                ibis_bool(table.commit == ctx.commit),
            ).count()
            row_counts[table_key] = int(cast("SupportsInt", count_expr.execute()))
        except (RuntimeError, OSError, DuckDBCatalogException, TableNotFound):
            row_counts[table_key] = 0

    return row_counts
```

### Files to Update

| File | Current Function |
|------|-----------------|
| `plugins/analytics/hotspots/build.py` | `_compute_row_counts` |
| `plugins/analytics/cfg_dfg/metrics.py` | inline counting |
| `plugins/analytics/symbol_graph_metrics/compute.py` | inline counting |
| `plugins/analytics/tests/graph_metrics.py` | inline counting |
| `plugins/ingestion/scip_plugin.py` | `_compute_row_counts` |
| `plugins/ingestion/repo_scan.py` | `_compute_row_counts` |
| `plugins/graphs/metrics/core.py` | inline counting |

---

## Part 3: Options Resolver Pattern Consolidation

### Problem

Many plugins have identical `resolve_options` methods:

```python
def resolve_options(
    self,
    *,
    dynamic_overrides: Mapping[str, Any] | None = None,
) -> MyOptionsType:
    if self._options_resolver is None:
        if dynamic_overrides:
            return MyOptionsType(**dynamic_overrides)
        return MyOptionsType()
    return self._options_resolver.get_options(
        self._core_metadata,
        MyOptionsType,
        dynamic_overrides=dynamic_overrides,
    )
```

### Solution

Add `resolve_options` to `MetadataPlugin` as a generic helper:

```python
class MetadataPlugin(TargetPlugin, ABC, Generic[TOptions]):
    """Enhanced plugin base with automatic metadata and options handling."""

    _core_metadata: ClassVar[CorePluginMetadata]
    _options_type: ClassVar[type[TOptions]]  # Add this
    _options_resolver: PluginOptionsResolver | None

    def resolve_options(
        self,
        *,
        dynamic_overrides: Mapping[str, Any] | None = None,
    ) -> TOptions:
        """Resolve typed options from configuration.

        Returns
        -------
        TOptions
            Resolved options instance.
        """
        if self._options_resolver is None:
            if dynamic_overrides:
                return self._options_type(**dynamic_overrides)
            return self._options_type()
        return self._options_resolver.get_options(
            self._core_metadata,
            self._options_type,
            dynamic_overrides=dynamic_overrides,
        )
```

---

## Part 4: Context Hierarchy Simplification

### Problem

Multiple overlapping context types:
- `MaterializationContext` (deprecated, but still used)
- `ArtifactMaterializationContext` (separate from BuildContext)
- `MaterializationContextProtocol` (protocol for compatibility)

### Solution

1. Remove `MaterializationContext` entirely (already deprecated)
2. Add artifact support directly to `BuildContext`:

```python
@dataclass(frozen=True)
class BuildContext:
    # ... existing fields ...

    def materialize_table(self, table_key: str, expr: ir.Table) -> DatasetRef:
        """Materialize an Ibis expression to DuckDB."""
        from codeintel.build.hamilton.native.materializer import materialize_table
        return materialize_table(self, table_key, expr)

    def materialize_artifact(
        self,
        name: str,
        content: bytes | str,
        path: Path,
        artifact_type: str = "file",
    ) -> ArtifactRef:
        """Materialize a file artifact."""
        from codeintel.build.hamilton.native.artifact_materializer import (
            materialize_artifact,
            ArtifactMaterializationContext,
            ArtifactMaterializationSpec,
        )
        ctx = ArtifactMaterializationContext.from_build_context(self)
        spec = ArtifactMaterializationSpec(
            artifact_name=name,
            artifact_type=artifact_type,
            content=content,
            output_path=path,
        )
        return materialize_artifact(ctx, spec)
```

---

## Part 5: Native Target Pattern Improvements

### Problem

Native targets still have some repetitive patterns even after `NativeTargetExecutor`:
- Artifact handling requires manual record creation
- Export targets have duplicated JSON/Parquet patterns

### Solution

#### 5.1 Add artifact support to NativeTargetExecutor

```python
class NativeTargetExecutor:
    # ... existing code ...

    def execute_with_artifacts(
        self,
        compute_fn: Callable[[], tuple[dict[str, int], tuple[ArtifactRef, ...]]],
    ) -> TargetRunRecord:
        """Execute with artifact support."""
        start = time.perf_counter()
        try:
            row_counts, artifacts = compute_fn()
        except Exception as exc:
            return self.fail(exc)

        duration_ms = (time.perf_counter() - start) * 1000
        run = NativeRunInfo(
            input_hash=self.input_hash,
            options_hash=self.options_hash,
            duration_ms=duration_ms,
            row_counts=row_counts,
        )
        record = create_run_record(
            self.target,
            "succeeded",
            self.input_hash,
            env=self.env,
            run=run,
        )

        # Add artifacts to record
        record = TargetRunRecord(
            target=record.target,
            plugin_name=record.plugin_name,
            status=record.status,
            input_hash=record.input_hash,
            options_hash=record.options_hash,
            duration_ms=record.duration_ms,
            row_counts=record.row_counts,
            error=record.error,
            datasets=record.datasets,
            artifacts=artifacts,
        )

        save_manifest(self.env, record)
        return record
```

#### 5.2 Create ExportTargetMixin for common export patterns

```python
class ExportTargetMixin:
    """Shared utilities for export targets."""

    @staticmethod
    def export_to_jsonl(
        data: list[dict[str, Any]],
        output_path: Path,
        *,
        include_metadata: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, int]:
        """Export data to JSONL format.

        Returns
        -------
        tuple[str, int]
            Tuple of (content_string, line_count).
        """
        import json

        lines: list[str] = []
        if include_metadata and metadata:
            lines.append(json.dumps({"_metadata": metadata}, ensure_ascii=False))

        for record in data:
            lines.append(json.dumps(record, ensure_ascii=False))

        return "\n".join(lines) + "\n", len(lines)

    @staticmethod
    def export_to_parquet(
        df: pd.DataFrame,
        output_path: Path,
    ) -> bytes:
        """Export DataFrame to Parquet format.

        Returns
        -------
        bytes
            Parquet file content.
        """
        import io

        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, engine="pyarrow")
        return buffer.getvalue()
```

---

## Part 6: Error Handling Improvements

### Problem

Some code still uses generic `ValueError`/`RuntimeError` instead of structured errors.

### Solution

Audit and replace remaining generic exceptions with structured error types:

```python
# Replace:
raise ValueError(f"Target '{name}' not found")

# With:
from codeintel.build.errors import TargetNotFoundError
raise TargetNotFoundError(name, available=list(registry))
```

Key locations to audit:
- `unified_registry.py` - use `TargetNotFoundError`, `RegistryValidationError`
- `context.py` - use `GatewayNotAvailableError`
- `materializer.py` - use `SchemaValidationError`

---

## Part 7: Import Organization

### Problem

Many modules have inconsistent import patterns:
- Some use TYPE_CHECKING guards, some don't
- Lazy imports (`# noqa: PLC0415`) are scattered
- Some circular import workarounds are complex

### Solution

Establish clear patterns:

1. **Heavy imports** (numpy, pandas, ibis): Always in TYPE_CHECKING
2. **Circular import prevention**: Use lazy imports in `__init__.py`
3. **Protocol types**: Import from canonical facade modules

Create a facade for common type imports:

```python
# build/typing.py
"""Type imports for build system modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.build.context_base import BuildContext, ExecutionContext, PathResolver
    from codeintel.build.contracts import OutputContract, ArtifactSpec, TableSchema
    from codeintel.build.result import TargetResult
    from codeintel.build.plugin import TargetPlugin, MetadataPlugin
    from codeintel.build.targets import OutputTarget, TargetGraph
    from codeintel.build.errors import BuildError

__all__ = [
    "ArtifactSpec",
    "BuildContext",
    "BuildError",
    "ExecutionContext",
    "MetadataPlugin",
    "OutputContract",
    "OutputTarget",
    "PathResolver",
    "TableSchema",
    "TargetExecutionContext",
    "TargetGraph",
    "TargetPlugin",
    "TargetResult",
]
```

---

## Implementation Priority

| Priority | Part | Effort | Impact |
|----------|------|--------|--------|
| 1 | Part 2: Row Count Helper | Low | High - Immediate DRY win |
| 2 | Part 1: Plugin Migration | Medium | High - Major boilerplate reduction |
| 3 | Part 3: Options Resolver | Low | Medium - Further DRY |
| 4 | Part 5.1: Executor Artifacts | Low | Medium - Simplifies export targets |
| 5 | Part 6: Error Handling | Low | Medium - Better diagnostics |
| 6 | Part 4: Context Simplification | Medium | Medium - Cleaner API |
| 7 | Part 5.2: Export Mixin | Low | Low - Nice-to-have |
| 8 | Part 7: Import Organization | Medium | Low - Maintainability |

---

## Validation Checklist

After each phase:

```bash
# Lint and format
uv run ruff check --fix src/codeintel/build/
uv run ruff format src/codeintel/build/

# Type checking
uv run pyright src/codeintel/build/
uv run pyrefly check src/codeintel/build/

# Registry validation
uv run python -c "
from codeintel.build.unified_registry import get_unified_registry
reg = get_unified_registry()
errors = reg.validate()
print(f'Registry: {len(reg)} targets, errors={errors}')
"

# Tests
uv run pytest tests/build/ -q
```

---

## Summary

This refinement plan addresses:

1. **42 plugins** with duplicated metadata boilerplate
2. **8+ implementations** of row count computation
3. **Multiple overlapping** context types
4. **Scattered** options resolver patterns
5. **Duplicated** export patterns in native targets
6. **Inconsistent** error handling
7. **Unorganized** import patterns

Total estimated effort: ~2-3 days for full implementation

The highest-impact items (Parts 1-3) would eliminate ~60% of the boilerplate across the plugin system.

