# Ibis + Pandera Integration Guide

This guide explains how to use Ibis for data access and Pandera for schema validation in the CodeIntel codebase.

## Overview

The CodeIntel data layer uses three key technologies:

1. **DuckDB** - Physical storage engine (OLAP database)
2. **Ibis** - Query interface for DataFrame-like access with SQL generation
3. **Pandera** - Schema validation and data contracts

```
┌──────────────────────────────────────────┐
│     Application Layer                     │
│  (Serving, Analytics, CLI)               │
├──────────────────────────────────────────┤
│     Ibis Query Layer                     │
│  (Type-safe DataFrame expressions)       │
├──────────────────────────────────────────┤
│     Pandera Validation Layer             │
│  (Schema contracts, data quality)        │
├──────────────────────────────────────────┤
│     DuckDB Storage                       │
│  (Physical tables and views)             │
└──────────────────────────────────────────┘
```

## Reading Data with Ibis

### Basic Usage

Access Ibis tables through the `StorageGateway.ibis` property:

```python
from codeintel.storage.gateway import StorageGateway

def read_functions(gateway: StorageGateway, repo: str, commit: str) -> pd.DataFrame:
    # Get Ibis table expression
    tbl = gateway.ibis.table("analytics.function_metrics")
    
    # Build filter expression
    expr = tbl.filter(
        (tbl.repo == repo) & (tbl.commit == commit)
    ).order_by("qualname")
    
    # Execute and return DataFrame
    return pd.DataFrame(expr.execute())
```

### Repository Pattern with Fallbacks

Use Ibis with SQL fallback for compatibility:

```python
from ibis.common.exceptions import IbisError
from codeintel.storage.repositories.base import BaseRepository

class MyRepository(BaseRepository):
    def get_metrics(self, goid: int) -> RowDict | None:
        def ibis_query():
            tbl = self._ibis_table("analytics.function_metrics")
            return tbl.filter(tbl.function_goid_h128 == goid)
        
        sql = """
            SELECT * FROM analytics.function_metrics
            WHERE function_goid_h128 = ?
        """
        return self._ibis_one_with_fallback(
            ibis_query, sql, [goid],
            table_key="analytics.function_metrics"
        )
```

## Schema Validation with Pandera

### Schema Registry

All table schemas are registered in `pandera_schemas.py`:

```python
from codeintel.storage.pandera_schemas import (
    DATASET_SCHEMAS,
    get_dataset_schema,
    validate_dataset_df,
)

# Get schema for a table
schema = get_dataset_schema("analytics.function_metrics")

# Validate a DataFrame
validated_df = validate_dataset_df("analytics.function_metrics", df)
```

### Write Path Validation

Always validate data before writing to the database:

```python
from codeintel.storage.pandera_schemas import validate_dataset_df

def persist_data(gateway: StorageGateway, rows: list[dict]) -> int:
    df = pd.DataFrame(rows)
    validated = validate_dataset_df("analytics.my_table", df)
    
    # Convert NaN to None for SQL compatibility
    records = validated.where(pd.notna(validated), None).to_dict(orient="records")
    
    # Insert validated records
    storage.insert_rows("analytics.my_table", records)
    return len(records)
```

### Validation Result Pattern

Use `ValidationResult` for detailed error handling:

```python
from codeintel.storage.pandera_schemas import validate_with_result

result = validate_with_result("analytics.my_table", df, strict=True)
if result.success:
    process_data(result.validated_df)
else:
    log.error("Validation failed: %d errors", result.error_count)
    for error in result.errors:
        log.error("  - %s", error)
```

## Ibis Views

The system defines several Ibis-backed views for common queries:

```python
from codeintel.storage.views.ibis_views import create_all_ibis_views

# Create all views during initialization
create_all_ibis_views(gateway)
```

### Available Views

| View | Description |
|------|-------------|
| `analytics.v_function_summary` | Function metrics with type coverage |
| `analytics.v_function_hotspots` | Functions ranked by hotspot score |
| `graph.v_call_graph_degree` | Call graph in/out degree |
| `graph.v_import_graph_degree` | Import graph in/out degree |
| `core.v_goid_crosswalk_join` | GOID with crosswalk data |
| `docs.v_function_summary` | Enriched function view |
| `docs.v_subsystem_summary` | Subsystem overview |
| `docs.v_subsystem_profile` | Subsystem with graph metrics |
| `docs.v_subsystem_coverage` | Subsystem test coverage |
| `docs.v_file_summary` | File-level summary |
| `docs.v_module_architecture` | Module architectural view |

## Column Checks

Pandera schemas include semantic checks:

### Non-Negative Checks
- `goid_h128`, `loc`, `complexity`, `fan_in`, `fan_out`
- Line counts, duration, row counts

### Positive Checks (>= 1)
- `start_line`, `end_line`, `lineno`
- Any 1-indexed values

### Ratio Checks (0.0 to 1.0)
- `coverage_ratio`, `param_typed_ratio`
- `file_typed_ratio`, `confidence`

### Cross-Column Checks
- `end_line >= start_line`
- `covered_lines <= executable_lines`
- `failing_test_count <= test_count`

## JSON Schema Export

Generate JSON Schema for API documentation:

```python
from codeintel.storage.pandera_schemas import (
    pandera_to_json_schema,
    dataset_json_schema,
)

# Get JSON Schema for a dataset
json_schema = dataset_json_schema("analytics.function_metrics")

# Convert with options
schema = pandera_to_json_schema(
    df_schema,
    include_constraints=True,
    include_metadata=True
)
```

## Best Practices

### 1. Prefer Ibis Over Raw SQL

```python
# Good: Type-safe, composable
tbl = gateway.ibis.table("core.goids")
expr = tbl.filter(tbl.kind == "function").limit(100)

# Avoid: String-based SQL
sql = "SELECT * FROM core.goids WHERE kind = 'function' LIMIT 100"
```

### 2. Always Validate Write Paths

```python
# Good: Validate before insert
validated = validate_dataset_df(table_key, df)
insert_rows(validated.to_dict(orient="records"))

# Avoid: Direct insert without validation
insert_rows(df.to_dict(orient="records"))
```

### 3. Use SQL Fallbacks for Compatibility

```python
def query_data(self):
    try:
        return self._execute_ibis_query()
    except IbisError:
        log.debug("Falling back to SQL")
        return self._execute_sql_query()
```

### 4. Handle Nulls Correctly

```python
# Convert NaN to None for SQL
records = df.where(pd.notna(df), None).to_dict(orient="records")
```

## Testing

### Property-Based Tests

Use Hypothesis with Pandera strategies:

```python
from hypothesis import given, settings
from hypothesis import strategies as st

@given(
    loc=st.integers(min_value=0, max_value=10000),
    complexity=st.integers(min_value=0, max_value=100),
)
@settings(max_examples=50)
def test_non_negative_metrics(loc: int, complexity: int):
    df = pd.DataFrame({"loc": [loc], "cyclomatic_complexity": [complexity]})
    result = validate_dataset_df("analytics.function_metrics", df)
    assert len(result) == 1
```

### View Consistency Tests

Verify Ibis views match expected schemas:

```python
def test_function_summary_view_schema():
    schema = get_dataset_schema("analytics.v_function_summary")
    assert schema is not None
    assert "function_goid_h128" in schema.columns
```

## Troubleshooting

### IbisError on Missing Table

```python
try:
    tbl = gateway.ibis.table("missing.table")
except IbisError as e:
    log.warning("Table not found: %s", e)
```

### Pandera Validation Errors

```python
from pandera.errors import SchemaErrors

try:
    validated = schema.validate(df, lazy=True)
except SchemaErrors as e:
    for failure in e.failure_cases.itertuples():
        log.error("Column %s: %s", failure.column, failure.check)
```

### Type Coercion Issues

Pandera schemas have `coerce=True` by default. If you encounter type issues:

```python
# Explicit type conversion before validation
df["goid_h128"] = pd.to_numeric(df["goid_h128"], errors="coerce")
validated = validate_dataset_df(table_key, df)
```
