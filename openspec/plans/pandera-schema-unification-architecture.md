# Pandera Schema Unification Architecture

> **Status:** Draft  
> **Author:** AI Architect  
> **Created:** 2025-01-09  
> **Related:** `plans/pandera_to_schema.md`, `plans/ibis-pandera-type-safety-plan.md`

## Executive Summary

This architecture describes a transition from **explicitly-defined, scattered row models** to a **unified, constraint-driven schema layer** where Pandera DataFrameSchemas become the single source of truth (SSOT). The goal is to:

1. **Reduce duplication** by deriving row models, serializers, and validation from one canonical schema
2. **Enable introspection** so LLMs and tooling can trace data flow from producer to consumer
3. **Prepare for logic-framework evolution** where calculation dependencies drive behavior implicitly

---

## 1. Current State Analysis

### 1.1 Existing Schema Infrastructure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CURRENT SCHEMA LANDSCAPE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  primitives.py            contracts.py           pandera_schemas.py        │
│  ┌───────────────┐       ┌───────────────┐      ┌───────────────────┐      │
│  │ TableSchema   │──────▶│DatasetContract│─────▶│DATASET_SCHEMAS    │      │
│  │ Column        │       │RowBinding     │      │(Pandera)          │      │
│  │ CompositeSchema│      │               │      │                   │      │
│  └───────────────┘       └───────────────┘      └───────────────────┘      │
│         │                       │                       │                   │
│         ▼                       ▼                       ▼                   │
│  DDL Generation          TypedDict Models         Validation Checks         │
│                                                                             │
│  config/datasets/rows/           storage/gateway/rows/                      │
│  ┌───────────────────┐          ┌───────────────────────┐                   │
│  │ analytics.py      │          │ analytics.py (DUPE!)  │                   │
│  │ core.py           │          │ core.py               │                   │
│  │ graph.py          │          │ graph.py              │                   │
│  │ profiles.py       │          │                       │                   │
│  │ test.py           │          │                       │                   │
│  └───────────────────┘          └───────────────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Files and Their Roles

| File | Purpose | Current Role | Future Role |
|------|---------|--------------|-------------|
| `config/datasets/primitives.py` | DDL-level Column, TableSchema, fragments | SSOT for DDL | Derived from or validated against Pandera |
| `config/datasets/contracts.py` | DatasetContract registry with metadata | Central contract store | Enhanced with unified DatasetSchema |
| `storage/pandera_schemas.py` | Pandera DataFrameSchema registry | Validation-only | **SSOT for all schema information** |
| `config/datasets/rows/*.py` | TypedDict row models (manual) | Manual maintenance | **Auto-generated** from Pandera |
| `storage/gateway/rows/*.py` | Generated row models for inserts | Separate generation | **Unified** with config row models |

### 1.3 Problem Statement

1. **Duplication**: Row models are defined in multiple places (`config/datasets/rows/` and `storage/gateway/rows/`)
2. **Drift Risk**: Manual TypedDicts can drift from TableSchema definitions
3. **Scattered Validation**: Column checks exist in `pandera_schemas.py` but aren't universally applied
4. **Missing Linkage**: Plugins declare `produces_tables` but there's no automatic schema enforcement
5. **No Introspection**: Can't easily answer "what columns does plugin X produce?"

---

## 2. Target Architecture

### 2.1 Layered Schema Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      UNIFIED SCHEMA ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌──────────────────────────────┐                         │
│                    │      DatasetSchema           │ ◀── THE NEW SSOT        │
│                    │  (Wraps Pandera + Metadata)  │                         │
│                    └──────────────────────────────┘                         │
│                              │                                              │
│           ┌──────────────────┼──────────────────┐                          │
│           ▼                  ▼                  ▼                          │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                  │
│   │ Row Model     │  │ DDL Generator │  │ JSON Schema   │                  │
│   │ (TypedDict)   │  │ (TableSchema) │  │ (Export/Docs) │                  │
│   └───────────────┘  └───────────────┘  └───────────────┘                  │
│           │                  │                  │                          │
│           ▼                  ▼                  ▼                          │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                  │
│   │ Plugin I/O    │  │ DuckDB Tables │  │ API Contracts │                  │
│   │ Validation    │  │               │  │               │                  │
│   └───────────────┘  └───────────────┘  └───────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 DatasetSchema: The Unified Abstraction

```python
# config/datasets/schema.py (NEW)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal

from pandera import DataFrameSchema

if TYPE_CHECKING:
    from collections.abc import Mapping
    from codeintel.config.datasets.primitives import TableSchema, CompositeSchema


@dataclass(frozen=True)
class DatasetSchema:
    """Unified schema abstraction for all datasets.

    This is THE single source of truth for dataset structure. Everything else
    (row models, DDL, JSON schemas) derives from this.

    Parameters
    ----------
    name
        Fully qualified table name (e.g., "analytics.function_metrics").
    pandera_schema
        Pandera DataFrameSchema defining structure and constraints.
    row_model
        Auto-generated TypedDict for row-level typing (populated lazily).
    ddl_schema
        TableSchema for DDL generation (derived from Pandera).
    metadata
        Additional dataset metadata (ownership, SLA, dependencies).
    composition
        Optional CompositeSchema for profile datasets.
    """

    name: str
    pandera_schema: DataFrameSchema
    row_model: type[Any] | None = None
    ddl_schema: TableSchema | None = None
    metadata: DatasetMetadata = field(default_factory=lambda: DatasetMetadata())
    composition: CompositeSchema | None = None

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate DataFrame against Pandera schema.

        Parameters
        ----------
        df
            DataFrame to validate.

        Returns
        -------
        pd.DataFrame
            Validated (and coerced) DataFrame.
        """
        return self.pandera_schema.validate(df, lazy=True)

    def column_names(self) -> tuple[str, ...]:
        """Return ordered column names.

        Returns
        -------
        tuple[str, ...]
            Column names in definition order.
        """
        return tuple(self.pandera_schema.columns.keys())

    def json_schema(self) -> dict[str, Any]:
        """Generate JSON Schema 2020-12 from Pandera schema.

        Returns
        -------
        dict[str, Any]
            JSON Schema representation.
        """
        from codeintel.storage.pandera_schemas import pandera_to_json_schema
        return pandera_to_json_schema(self.pandera_schema)

    def get_row_model(self) -> type[Any]:
        """Return or generate the TypedDict row model.

        Returns
        -------
        type[Any]
            TypedDict class for this dataset's rows.
        """
        if self.row_model is not None:
            return self.row_model
        from codeintel.config.datasets.row_factory import typed_dict_from_pandera
        return typed_dict_from_pandera(
            f"{_to_class_name(self.name)}Row",
            self.pandera_schema,
        )


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata for dataset governance and operations.

    Parameters
    ----------
    description
        Human-readable description.
    owner
        Team or individual owner.
    family
        Dataset family (core, analytics, graph, docs).
    freshness_sla
        Expected refresh frequency.
    retention_policy
        Data retention policy.
    upstream_dependencies
        Other datasets this one depends on.
    downstream_consumers
        Datasets that consume this one (computed lazily).
    tags
        Classification tags.
    deprecated
        Whether this dataset is deprecated.
    deprecation_message
        Migration guidance if deprecated.
    """

    description: str | None = None
    owner: str | None = None
    family: str | None = None
    freshness_sla: str | None = None
    retention_policy: str | None = None
    upstream_dependencies: tuple[str, ...] = ()
    downstream_consumers: tuple[str, ...] = ()
    tags: frozenset[str] = frozenset()
    deprecated: bool = False
    deprecation_message: str | None = None
```

### 2.3 Schema Registry and Builder

```python
# config/datasets/schema_registry.py (NEW)

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.config.datasets.schema import DatasetSchema

if TYPE_CHECKING:
    from pandera import DataFrameSchema


class DatasetSchemaRegistry:
    """Global registry for all DatasetSchema instances.

    This registry is the authoritative source for dataset schemas.
    It integrates with existing infrastructure to provide backward
    compatibility while enabling the new unified architecture.
    """

    _schemas: dict[str, DatasetSchema]
    _initialized: bool

    def __init__(self) -> None:
        self._schemas = {}
        self._initialized = False

    def initialize(self) -> None:
        """Build schemas from existing contracts and Pandera definitions.

        This method bridges the current infrastructure with the new
        unified schema layer.
        """
        if self._initialized:
            return

        from codeintel.config.datasets.schema_builder import build_all_schemas
        self._schemas = build_all_schemas()
        self._initialized = True

    def get(self, table_key: str) -> DatasetSchema | None:
        """Retrieve a DatasetSchema by table key.

        Parameters
        ----------
        table_key
            Fully qualified table name.

        Returns
        -------
        DatasetSchema | None
            Schema if registered, None otherwise.
        """
        self.initialize()
        return self._schemas.get(table_key)

    def require(self, table_key: str) -> DatasetSchema:
        """Retrieve a DatasetSchema or raise if missing.

        Parameters
        ----------
        table_key
            Fully qualified table name.

        Returns
        -------
        DatasetSchema
            The registered schema.

        Raises
        ------
        KeyError
            If no schema is registered for the table key.
        """
        schema = self.get(table_key)
        if schema is None:
            msg = f"No DatasetSchema registered for '{table_key}'"
            raise KeyError(msg)
        return schema

    def all(self) -> dict[str, DatasetSchema]:
        """Return all registered schemas.

        Returns
        -------
        dict[str, DatasetSchema]
            All schemas keyed by table name.
        """
        self.initialize()
        return dict(self._schemas)

    def producers_of(self, table_key: str) -> list[str]:
        """Find plugins that produce this dataset.

        Parameters
        ----------
        table_key
            Dataset to find producers for.

        Returns
        -------
        list[str]
            Plugin names that produce this dataset.
        """
        from codeintel.build.plugins import PLUGIN_CATALOG
        return [
            p.plugin_name
            for p in PLUGIN_CATALOG.all()
            if hasattr(p, 'core_metadata')
            and table_key in (p.core_metadata.produces_tables or ())
        ]

    def consumers_of(self, table_key: str) -> list[str]:
        """Find plugins that consume this dataset.

        Parameters
        ----------
        table_key
            Dataset to find consumers for.

        Returns
        -------
        list[str]
            Plugin names that consume this dataset.
        """
        from codeintel.build.plugins import PLUGIN_CATALOG
        return [
            p.plugin_name
            for p in PLUGIN_CATALOG.all()
            if hasattr(p, 'core_metadata')
            and table_key in (p.core_metadata.consumes_tables or ())
        ]


# Global singleton
SCHEMA_REGISTRY = DatasetSchemaRegistry()


def get_schema(table_key: str) -> DatasetSchema | None:
    """Convenience function to get a schema from the global registry.

    Parameters
    ----------
    table_key
        Fully qualified table name.

    Returns
    -------
    DatasetSchema | None
        Schema if registered.
    """
    return SCHEMA_REGISTRY.get(table_key)
```

### 2.4 Row Model Factory

```python
# config/datasets/row_factory.py (NEW)

from __future__ import annotations

from typing import Any, TypedDict

from pandera import DataFrameSchema
from pandera.engines.pandas_engine import PandasDtype


def _pandera_dtype_to_python(dtype: PandasDtype) -> type[Any]:
    """Map Pandera dtype to Python type for TypedDict.

    Parameters
    ----------
    dtype
        Pandera column dtype.

    Returns
    -------
    type[Any]
        Corresponding Python type.
    """
    import datetime
    from pandas import Int64Dtype, Float64Dtype, StringDtype, BooleanDtype

    dtype_str = str(dtype).lower()

    # Handle pandas nullable dtypes
    if isinstance(dtype, Int64Dtype) or 'int' in dtype_str:
        return int
    if isinstance(dtype, Float64Dtype) or 'float' in dtype_str or 'double' in dtype_str:
        return float
    if isinstance(dtype, BooleanDtype) or 'bool' in dtype_str:
        return bool
    if 'datetime' in dtype_str:
        return datetime.datetime

    # Default to string
    return str


def typed_dict_from_pandera(
    name: str,
    schema: DataFrameSchema,
    *,
    nullable_as_optional: bool = True,
) -> type[TypedDict]:
    """Generate a TypedDict from a Pandera DataFrameSchema.

    This enables automatic derivation of row types from the
    canonical Pandera schema, eliminating manual TypedDict
    maintenance.

    Parameters
    ----------
    name
        Name for the generated TypedDict class.
    schema
        Pandera DataFrameSchema to derive from.
    nullable_as_optional
        If True, nullable columns become `T | None`.

    Returns
    -------
    type[TypedDict]
        Generated TypedDict class.

    Examples
    --------
    >>> schema = DataFrameSchema({
    ...     "repo": Column(str),
    ...     "loc": Column(int, nullable=True),
    ... })
    >>> RowModel = typed_dict_from_pandera("MyRow", schema)
    >>> # RowModel is a TypedDict with repo: str, loc: int | None
    """
    annotations: dict[str, Any] = {}

    for col_name, column in schema.columns.items():
        py_type = _pandera_dtype_to_python(column.dtype)

        if nullable_as_optional and column.nullable:
            # Use union with None for nullable columns
            annotations[col_name] = py_type | None
        else:
            annotations[col_name] = py_type

    # Create TypedDict using functional form
    return TypedDict(name, annotations, total=True)


def row_serializer_from_pandera(
    schema: DataFrameSchema,
) -> Callable[[Mapping[str, Any]], tuple[Any, ...]]:
    """Generate a row serializer from Pandera schema.

    The serializer converts a row dict to a tuple in column order,
    suitable for database INSERT operations.

    Parameters
    ----------
    schema
        Pandera DataFrameSchema defining column order.

    Returns
    -------
    Callable[[Mapping[str, Any]], tuple[Any, ...]]
        Serializer function.
    """
    columns = tuple(schema.columns.keys())

    def serialize(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(row[col] for col in columns)

    return serialize
```

---

## 3. Constraint Aggregation Layer

### 3.1 Purpose

The **Constraint Aggregation Layer** collects all constraints that implicitly define a dataset's structure:

- Column types and nullability (from DuckDB DDL)
- Pandera column checks (e.g., non-negative, ratio bounds)
- Primary key constraints
- Cross-column checks (e.g., `covered_lines <= executable_lines`)
- Foreign key relationships (from plugin metadata)
- Computation dependencies (from plugin DAG)

### 3.2 ConstraintSet Model

```python
# config/datasets/constraints.py (NEW)

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ConstraintKind(Enum):
    """Classification of constraint types."""
    TYPE = "type"           # Column type constraint
    NULLABILITY = "null"    # Nullable/required constraint
    RANGE = "range"         # Numeric range (min/max)
    PATTERN = "pattern"     # String pattern/regex
    UNIQUENESS = "unique"   # Uniqueness constraint
    FOREIGN_KEY = "fk"      # References another dataset
    CROSS_COLUMN = "cross"  # Multi-column check
    COMPUTATION = "compute" # Derived from calculation


@dataclass(frozen=True)
class Constraint:
    """A single constraint on a column or table.

    Parameters
    ----------
    kind
        Type of constraint.
    column
        Column name (None for table-level constraints).
    expression
        Human-readable constraint expression.
    check_fn
        Optional callable for runtime validation.
    source
        Where this constraint was inferred from.
    """

    kind: ConstraintKind
    column: str | None
    expression: str
    check_fn: Callable[[Any], bool] | None = None
    source: str = "manual"


@dataclass
class ConstraintSet:
    """Aggregated constraints for a dataset.

    This collects constraints from multiple sources to provide
    a complete picture of what defines a dataset's structure.

    Parameters
    ----------
    table_key
        Fully qualified table name.
    constraints
        List of all constraints.
    inferred_from
        Sources from which constraints were inferred.
    """

    table_key: str
    constraints: list[Constraint] = field(default_factory=list)
    inferred_from: set[str] = field(default_factory=set)

    def add(self, constraint: Constraint) -> None:
        """Add a constraint to the set.

        Parameters
        ----------
        constraint
            Constraint to add.
        """
        self.constraints.append(constraint)
        if constraint.source:
            self.inferred_from.add(constraint.source)

    def for_column(self, column: str) -> list[Constraint]:
        """Get constraints for a specific column.

        Parameters
        ----------
        column
            Column name.

        Returns
        -------
        list[Constraint]
            Constraints applying to this column.
        """
        return [c for c in self.constraints if c.column == column]

    def table_level(self) -> list[Constraint]:
        """Get table-level constraints.

        Returns
        -------
        list[Constraint]
            Constraints that span multiple columns.
        """
        return [c for c in self.constraints if c.column is None]


def extract_constraints_from_pandera(
    table_key: str,
    schema: DataFrameSchema,
) -> ConstraintSet:
    """Extract ConstraintSet from a Pandera schema.

    Parameters
    ----------
    table_key
        Dataset identifier.
    schema
        Pandera DataFrameSchema.

    Returns
    -------
    ConstraintSet
        Extracted constraints.
    """
    cs = ConstraintSet(table_key=table_key)

    for col_name, column in schema.columns.items():
        # Type constraint
        cs.add(Constraint(
            kind=ConstraintKind.TYPE,
            column=col_name,
            expression=f"{col_name}: {column.dtype}",
            source="pandera.column.dtype",
        ))

        # Nullability
        cs.add(Constraint(
            kind=ConstraintKind.NULLABILITY,
            column=col_name,
            expression=f"{col_name} {'nullable' if column.nullable else 'required'}",
            source="pandera.column.nullable",
        ))

        # Column checks
        if column.checks:
            for check in column.checks:
                # Try to extract range constraints
                check_str = str(check)
                if ">= 0" in check_str or "(s >= 0)" in check_str:
                    cs.add(Constraint(
                        kind=ConstraintKind.RANGE,
                        column=col_name,
                        expression=f"{col_name} >= 0",
                        source="pandera.check.non_negative",
                    ))
                elif ">= 1" in check_str:
                    cs.add(Constraint(
                        kind=ConstraintKind.RANGE,
                        column=col_name,
                        expression=f"{col_name} >= 1",
                        source="pandera.check.positive",
                    ))
                # TODO: Extract ratio constraints, etc.

    # Table-level checks
    if schema.checks:
        for check in schema.checks:
            cs.add(Constraint(
                kind=ConstraintKind.CROSS_COLUMN,
                column=None,
                expression=str(check),
                source="pandera.dataframe_check",
            ))

    return cs
```

### 3.3 Constraint Introspection Service

```python
# config/datasets/introspection.py (NEW)

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.config.datasets.constraints import ConstraintSet
    from codeintel.config.datasets.schema import DatasetSchema


@dataclass
class DatasetIntrospection:
    """Complete introspection of a dataset for LLM/tooling consumption.

    This aggregates all metadata about a dataset into a single,
    queryable structure suitable for code intelligence.

    Parameters
    ----------
    schema
        The unified DatasetSchema.
    constraints
        Aggregated constraints from all sources.
    producers
        Plugins that produce this dataset.
    consumers
        Plugins that consume this dataset.
    upstream
        Datasets this one depends on.
    downstream
        Datasets that depend on this one.
    """

    schema: DatasetSchema
    constraints: ConstraintSet
    producers: list[str]
    consumers: list[str]
    upstream: list[str]
    downstream: list[str]

    def summary_for_llm(self) -> str:
        """Generate a human/LLM-readable summary.

        Returns
        -------
        str
            Markdown summary of the dataset.
        """
        lines = [
            f"# Dataset: {self.schema.name}",
            "",
            f"**Description:** {self.schema.metadata.description or 'No description'}",
            f"**Owner:** {self.schema.metadata.owner or 'Unassigned'}",
            "",
            "## Columns",
            "",
        ]

        for col_name in self.schema.column_names():
            col_constraints = self.constraints.for_column(col_name)
            constraint_strs = [c.expression for c in col_constraints]
            lines.append(f"- `{col_name}`: {', '.join(constraint_strs)}")

        lines.extend([
            "",
            "## Data Flow",
            "",
            f"**Produced by:** {', '.join(self.producers) or 'Unknown'}",
            f"**Consumed by:** {', '.join(self.consumers) or 'None'}",
            f"**Depends on:** {', '.join(self.upstream) or 'None'}",
        ])

        return "\n".join(lines)


def introspect_dataset(table_key: str) -> DatasetIntrospection:
    """Build complete introspection for a dataset.

    Parameters
    ----------
    table_key
        Fully qualified table name.

    Returns
    -------
    DatasetIntrospection
        Complete dataset introspection.
    """
    from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY
    from codeintel.config.datasets.constraints import extract_constraints_from_pandera

    schema = SCHEMA_REGISTRY.require(table_key)
    constraints = extract_constraints_from_pandera(table_key, schema.pandera_schema)

    return DatasetIntrospection(
        schema=schema,
        constraints=constraints,
        producers=SCHEMA_REGISTRY.producers_of(table_key),
        consumers=SCHEMA_REGISTRY.consumers_of(table_key),
        upstream=list(schema.metadata.upstream_dependencies),
        downstream=list(schema.metadata.downstream_consumers),
    )
```

---

## 4. Integration Points

### 4.1 Plugin Integration

Plugins should use `DatasetSchema` for validation at write time:

```python
# Example: Enhanced plugin with schema validation

class FunctionMetricsPlugin(TargetPlugin):
    """Compute function metrics with schema-aware validation."""

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        # Compute rows
        rows = self._compute_rows(ctx)
        df = pd.DataFrame(rows)

        # Validate against canonical schema
        from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY

        metrics_schema = SCHEMA_REGISTRY.require("analytics.function_metrics")
        validated_df = metrics_schema.validate(df)

        # Write validated data
        ctx.write_table("analytics.function_metrics", validated_df)

        return TargetResult.succeeded(
            row_counts={"analytics.function_metrics": len(validated_df)}
        )
```

### 4.2 Adapter Integration

Analytics adapters should derive row types from schemas:

```python
# analytics/adapters/functions.py

from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY


class FunctionMetricsAdapter:
    """Adapter for analytics.function_metrics persistence."""

    @property
    def row_model(self) -> type:
        """Return the canonical row model."""
        schema = SCHEMA_REGISTRY.require("analytics.function_metrics")
        return schema.get_row_model()

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate DataFrame before persistence."""
        schema = SCHEMA_REGISTRY.require("analytics.function_metrics")
        return schema.validate(df)
```

### 4.3 Build Context Integration

The `TargetExecutionContext` should expose schema-aware helpers:

```python
# build/context.py enhancement

class TargetExecutionContext:
    """Execution context with schema-aware write methods."""

    def write_validated_table(
        self,
        table_key: str,
        df: pd.DataFrame,
        *,
        strict: bool = True,
    ) -> int:
        """Write DataFrame with automatic schema validation.

        Parameters
        ----------
        table_key
            Target table name.
        df
            DataFrame to write.
        strict
            Raise on validation failure if True.

        Returns
        -------
        int
            Number of rows written.
        """
        from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY

        schema = SCHEMA_REGISTRY.get(table_key)
        if schema is not None:
            try:
                df = schema.validate(df)
            except Exception as e:
                if strict:
                    raise
                self.log.warning("Validation failed for %s: %s", table_key, e)

        return self.write_table(table_key, df)
```

---

## 5. Migration Strategy

### 5.1 Phase 1: Foundation (Week 1-2)

**Goal:** Establish unified schema infrastructure without breaking existing code.

1. **Create new modules:**
   - `config/datasets/schema.py` — `DatasetSchema` dataclass
   - `config/datasets/schema_registry.py` — `DatasetSchemaRegistry`
   - `config/datasets/row_factory.py` — `typed_dict_from_pandera`
   - `config/datasets/schema_builder.py` — Bridge from existing contracts

2. **Build schemas from existing infrastructure:**
   - Read from `DATASET_CONTRACTS`
   - Use existing `pandera_schemas.py` for Pandera schemas
   - Validate that generated row models match existing TypedDicts

3. **Add tests:**
   - Test that all existing datasets have valid `DatasetSchema`
   - Test that generated row models match manual ones
   - Test constraint extraction

### 5.2 Phase 2: Adoption (Week 3-4)

**Goal:** Start using unified schemas in new code.

1. **Update analytics plugins:**
   - Import `SCHEMA_REGISTRY` for validation
   - Use `schema.validate()` before writes
   - Update plugin tests to verify schema compliance

2. **Update adapters:**
   - Replace direct `pandera_schemas.DATASET_SCHEMAS` access
   - Use `SCHEMA_REGISTRY.require()` for type-safe access

3. **Add introspection CLI:**
   - `codeintel dataset info <table_key>` — Show schema details
   - `codeintel dataset constraints <table_key>` — Show constraint summary
   - `codeintel dataset flow <table_key>` — Show producer/consumer graph

### 5.3 Phase 3: Consolidation (Week 5-6)

**Goal:** Remove duplicate row models and establish Pandera as SSOT.

1. **Migrate row models:**
   - For each dataset, replace manual TypedDict with generated model
   - Keep backward-compatible exports until all consumers updated
   - Remove `storage/gateway/rows/` duplicates

2. **Update contracts:**
   - `DatasetContract.row_binding` now references schema-generated model
   - Remove manual `RowBinding` definitions

3. **Documentation:**
   - Update AGENTS.md with new schema patterns
   - Document how to add new datasets using the unified approach

### 5.4 Phase 4: Logic Framework Preparation (Week 7-8)

**Goal:** Enable constraint-driven behavior derivation.

1. **Implement ConstraintSet aggregation:**
   - Extract constraints from Pandera schemas
   - Extract constraints from plugin DAG (produces/consumes)
   - Enable querying "what defines this column?"

2. **Add dependency inference:**
   - Auto-discover `upstream_dependencies` from plugin metadata
   - Auto-discover `downstream_consumers` from plugin metadata

3. **Expose to LLMs:**
   - JSON export of all constraint sets
   - Introspection API for agent consumption

---

## 6. Success Criteria

### 6.1 Quantitative Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Duplicate row model files | 2 locations | 1 location |
| Manual TypedDict definitions | ~50 | 0 (all generated) |
| Datasets with Pandera validation | ~60% | 100% |
| Schema drift incidents | Unknown | 0 (tested) |

### 6.2 Qualitative Goals

1. **Single Source of Truth:** Any question about a dataset's structure can be answered by looking at one place (`DatasetSchema`).

2. **Introspection:** LLMs can query "what columns does X produce?" and "what constraints apply to Y?" programmatically.

3. **Consistency:** Row models, DDL, and validation are guaranteed to match because they derive from the same source.

4. **Extensibility:** Adding a new dataset requires defining one Pandera schema; everything else is generated.

---

## 7. Appendix: Relation to Long-Term Vision

This architecture is a **stepping stone** toward implicit, constraint-driven behavior:

```
Current State                     This Architecture                 Future State
┌───────────────┐                 ┌───────────────────┐             ┌───────────────────┐
│ Explicit      │                 │ Unified Schema    │             │ Logic Framework   │
│ Row Models    │  ──────────▶    │ (Pandera SSOT)    │ ──────────▶ │ (Constraint       │
│ TypedDicts    │                 │ + Constraints     │             │  Propagation)     │
└───────────────┘                 └───────────────────┘             └───────────────────┘

• Manual definitions              • Generated from schema           • Schemas derived from
• Scattered across files          • Central registry                  calculation logic
• Drift risk                      • Introspectable                  • Behavior implicit
                                 • Aggregated constraints            from dependencies
```

The **Constraint Aggregation Layer** is the key enabler: once we have all constraints in one queryable structure, we can start inferring behavior (e.g., "this column must exist because plugin X writes it and plugin Y reads it") rather than declaring it explicitly.

---

## 8. References

- `plans/pandera_to_schema.md` — Original proposal for Pandera unification
- `plans/ibis-pandera-type-safety-plan.md` — Related type safety improvements
- `AGENTS.md` — Agent operating protocol (testing charter, quality gates)
- `src/codeintel/storage/pandera_schemas.py` — Existing Pandera infrastructure
- `src/codeintel/config/datasets/contracts.py` — Existing contract infrastructure
