# Advanced DuckDB & SQLGlot Features: Enhancement Opportunities

**Generated**: 2024-12-17  
**Status**: Analysis Document  
**Scope**: `/home/paul/CodeIntel/src/codeintel/storage`

---

## Executive Summary

This document identifies advanced DuckDB and SQLGlot functionality that the CodeIntel storage layer is not currently leveraging. The analysis reveals significant opportunities to:

1. **Streamline code** by replacing manual SQL with native APIs
2. **Boost performance** through zero-copy Arrow integration and vectorized UDFs
3. **Enhance robustness** via query optimization and better type safety
4. **Improve extensibility** through proper extension management and custom types
5. **Increase maintainability** by centralizing complex patterns into reusable abstractions

---

## Current Implementation Summary

### What's Working Well

The storage layer demonstrates solid architectural patterns:

- **SQLGlot-based DDL generation** via `DuckDBPolicyBackend` for type-safe SQL
- **Ibis integration** for composable query building
- **Gateway pattern** with clear protocol definitions (`MinimalGateway`, `StorageGateway`)
- **Snapshot-scoped operations** for safe data management
- **Schema provider** for centralized table definitions

### Current DuckDB API Usage

| Feature | Usage Level | Files |
|---------|-------------|-------|
| `con.execute()` | Heavy | 16+ files |
| `ColumnExpression/ConstantExpression` | Limited | `warehouse.py`, `queries/safe.py` |
| Relational API (`.filter()`) | Limited | `warehouse.py`, `queries/safe.py` |
| FTS Extension | Single use | `serving/search_index.py` |
| `executemany()` | Light | `metadata/bootstrap.py`, `tracking/*.py` |

### Current SQLGlot Usage

| Feature | Usage Level | Location |
|---------|-------------|----------|
| Expression building | Heavy | `duckdb_policy_backend.py` |
| `parse_one()` | Light | DDL construction |
| Dialect specification | Heavy | `DUCKDB_DIALECT` constant |
| AST modification | None | — |
| Query optimization | None | — |

---

## Gap Analysis: DuckDB Advanced Features

### 1. DuckDB Relational API (Method Chaining)

**Current State**: The codebase uses basic `relation.filter()` but doesn't leverage the full chainable API.

**Missed Opportunities**:

```python
# Current pattern (warehouse.py:182-186)
relation = self.gateway.con.table(f"{schema}.{name}")
relation = relation.filter(
    (ColumnExpression("repo") == ConstantExpression(repo))
    & (ColumnExpression("commit") == ConstantExpression(commit))
)
row = relation.count("*").fetchone()
```

**Enhanced Pattern**:

```python
# Using full relational API
result = (
    self.gateway.con.table(f"{schema}.{name}")
    .filter(f"repo = '{repo}' AND commit = '{commit}'")
    .project("function_goid_h128", "loc", "complexity")
    .aggregate("avg(loc) as avg_loc, max(complexity) as max_complexity")
    .order("avg_loc DESC")
    .limit(100)
    .fetchdf()  # Zero-copy to DataFrame
)
```

**Benefits**:
- Lazy evaluation until terminal operation
- More readable query construction
- Eliminates intermediate SQL strings
- Direct DataFrame/Arrow output

**Implementation Priority**: HIGH - Impacts query performance across repositories

---

### 2. DuckDBPyType for Complex Types

**Current State**: Only basic column types (VARCHAR, INTEGER, BOOLEAN, etc.) are used in schema definitions.

**Missed Opportunities**:

```python
from duckdb.typing import DuckDBPyType

# For storing function parameters, return types, etc.
param_list_type = DuckDBPyType(list[str])  # LIST(VARCHAR)

# For storing structured metadata
metadata_struct = DuckDBPyType({
    'source_file': str,
    'line_start': int,
    'line_end': int,
    'annotations': list[str]
})  # STRUCT(source_file VARCHAR, line_start INTEGER, ...)

# For key-value metrics
metrics_map = DuckDBPyType({str: float})  # MAP(VARCHAR, DOUBLE)
```

**Use Cases in CodeIntel**:
- **`analytics.function_metrics`**: Store complexity breakdown as STRUCT instead of flattened columns
- **`graph.call_graph_edges`**: Store edge metadata as MAP for flexible attributes
- **`core.goid_symbols`**: Store symbol attributes as STRUCT

**Benefits**:
- More natural data modeling
- Better query expressiveness with nested access
- Reduced JOIN operations for related data
- Type-safe nested data access

**Implementation Priority**: MEDIUM - Schema changes required

---

### 3. Expression API (Beyond Filter)

**Current State**: Only `ColumnExpression` and `ConstantExpression` are used for filtering.

**Missed Opportunities**:

```python
from duckdb import FunctionExpression, CaseExpression, StarExpression

# Complex aggregations
complexity_bucket = CaseExpression(
    ColumnExpression("cyclomatic_complexity") <= 5, "low",
    ColumnExpression("cyclomatic_complexity") <= 10, "medium",
    default="high"
)

# Function calls in expressions
normalized_path = FunctionExpression(
    "regexp_replace",
    ColumnExpression("file_path"),
    ConstantExpression(r"^src/"),
    ConstantExpression("")
)

# Dynamic column selection
metrics_query = (
    con.table("analytics.function_metrics")
    .project(StarExpression())  # All columns
    .filter(complexity_bucket.alias("risk_level") == "high")
)
```

**Benefits**:
- Eliminates SQL string building for complex expressions
- Type-safe function invocations
- Composable expression building

**Implementation Priority**: MEDIUM - Improves query safety

---

### 4. User-Defined Functions (UDFs)

**Current State**: No UDFs are registered. All transformations happen in Python before/after queries.

**Missed Opportunities**:

```python
import duckdb
import pyarrow as pa

# Scalar UDF for GOID hashing
def compute_goid_hash(file_path: str, symbol_name: str, line: int) -> str:
    import hashlib
    content = f"{file_path}:{symbol_name}:{line}"
    return hashlib.blake2b(content.encode(), digest_size=16).hexdigest()

con.create_function("goid_hash", compute_goid_hash)

# Vectorized UDF for performance-critical operations (using Arrow)
def vectorized_complexity_score(
    loc: pa.Array, 
    cyclomatic: pa.Array
) -> pa.Array:
    # Process entire columns at once
    import pyarrow.compute as pc
    return pc.divide(cyclomatic, pc.add(loc, 1))

con.create_function(
    "complexity_score",
    vectorized_complexity_score,
    parameters=[duckdb.typing.BIGINT, duckdb.typing.BIGINT],
    return_type=duckdb.typing.DOUBLE,
    type="arrow"  # Vectorized execution
)
```

**Use Cases**:
- GOID computation directly in SQL
- Risk score calculation
- Path normalization
- Custom aggregations for graph metrics

**Benefits**:
- Reduce Python-SQL round trips
- Vectorized execution for large datasets
- Reusable business logic in queries

**Implementation Priority**: HIGH - Significant performance gains possible

---

### 5. Arrow/DataFrame Integration

**Current State**: Limited integration. Results are fetched as tuples and manually converted.

**Current Pattern** (from `repositories/base.py`):
```python
def _ibis_to_df(self, expr: it.Table) -> pd.DataFrame:
    return expr.execute()  # Goes through Ibis, not direct
```

**Missed Opportunities**:

```python
# Zero-copy Arrow table
arrow_table = con.table("analytics.function_metrics").fetch_arrow_table()

# Direct Polars integration
polars_df = con.table("analytics.function_metrics").pl()

# Direct Pandas with controlled memory
pandas_df = con.table("analytics.function_metrics").df()

# Streaming results for large datasets
for batch in con.table("large_table").fetch_arrow_batches():
    process_batch(batch)

# In-memory data as source (replacement scan)
import pandas as pd
temp_df = pd.DataFrame({"ids": [1, 2, 3]})
result = con.execute("""
    SELECT t.* FROM analytics.function_metrics t
    JOIN temp_df ON t.function_id = temp_df.ids
""").fetchall()  # temp_df is auto-registered
```

**Benefits**:
- Zero-copy data transfer
- Native Polars/Pandas support
- Streaming for memory efficiency
- In-memory joins without temp tables

**Implementation Priority**: HIGH - Major performance improvement

---

### 6. Extension Ecosystem

**Current State**: Only FTS extension is used. Extensions loaded from env var.

**Missing Extensions**:

| Extension | Use Case | Benefit |
|-----------|----------|---------|
| `httpfs` | Remote file access | Read Parquet/CSV from S3/GCS |
| `json` | JSON parsing | Native JSON column operations |
| `parquet` | Direct Parquet read | Skip Python intermediaries |
| `delta` | Delta Lake | Time-travel queries |
| `iceberg` | Iceberg tables | Schema evolution |
| `postgres/mysql` | External DBs | Federated queries |

**Enhanced Pattern**:

```python
# Load extensions programmatically
def ensure_extensions(con: DuckDBPyConnection) -> None:
    """Ensure all required extensions are loaded."""
    extensions = ["httpfs", "json", "parquet"]
    for ext in extensions:
        try:
            con.execute(f"LOAD {ext}")
        except duckdb.Error:
            con.execute(f"INSTALL {ext}")
            con.execute(f"LOAD {ext}")

# Use httpfs for remote data
con.execute("SET s3_region = 'us-west-2'")
result = con.execute("""
    SELECT * FROM parquet_scan('s3://bucket/analytics/*.parquet')
    WHERE repo = ?
""", [repo]).fetchdf()
```

**Implementation Priority**: MEDIUM - Enables new capabilities

---

### 7. Query Profiling & Performance

**Current State**: Profiling enabled via env var, but not programmatically controlled.

**Missed Opportunities**:

```python
from duckdb import ExplainType

# Query explanation for optimization
def explain_query(con: DuckDBPyConnection, query: str) -> dict:
    """Get query plan and statistics."""
    plan = con.execute(query).explain(ExplainType.ANALYZE_JSON)
    return {
        "plan": plan,
        "estimated_cardinality": ...,
        "actual_cardinality": ...,
    }

# Progress tracking for long queries
con.execute("PRAGMA enable_progress_bar")
con.execute("PRAGMA enable_progress_bar_print=true")

# Adaptive execution tuning
def configure_for_workload(con: DuckDBPyConnection, workload: str) -> None:
    if workload == "analytics":
        con.execute("SET threads = 8")
        con.execute("SET memory_limit = '4GB'")
    elif workload == "serving":
        con.execute("SET threads = 2")
        con.execute("SET memory_limit = '512MB'")
```

**Benefits**:
- Query optimization insights
- Workload-specific tuning
- Better debugging capabilities

**Implementation Priority**: LOW - Nice to have for observability

---

## Gap Analysis: SQLGlot Advanced Features

### 1. Query Optimization

**Current State**: SQLGlot is only used for AST building, not optimization.

**Missed Opportunity**:

```python
import sqlglot
from sqlglot import optimize

# Schema-aware optimization
schema = {
    "analytics.function_metrics": {
        "function_goid_h128": "VARCHAR",
        "repo": "VARCHAR",
        "commit": "VARCHAR",
        "loc": "INTEGER",
        "complexity": "INTEGER",
    }
}

raw_query = """
SELECT * FROM analytics.function_metrics
WHERE repo = 'org/repo' AND 1=1
ORDER BY loc
"""

optimized = optimize(raw_query, schema=schema, dialect="duckdb")
# Removes redundant "1=1", may reorder joins, etc.
```

**Benefits**:
- Automatic query simplification
- Predicate pushdown
- Join reordering
- Redundant condition elimination

**Implementation Priority**: MEDIUM - Improves generated query quality

---

### 2. AST Transformations

**Current State**: No AST transformations are used.

**Missed Opportunities**:

```python
import sqlglot
from sqlglot import exp

def add_snapshot_filter(query: str, repo: str, commit: str) -> str:
    """Automatically add snapshot filtering to all tables."""
    ast = sqlglot.parse_one(query, dialect="duckdb")
    
    def transform(node):
        if isinstance(node, exp.Table):
            # Add snapshot filter to each table reference
            alias = node.alias or node.name
            filter_cond = f"{alias}.repo = '{repo}' AND {alias}.commit = '{commit}'"
            # ... construct WHERE clause addition
        return node
    
    return ast.transform(transform).sql(dialect="duckdb")

# Use case: Ensure all ad-hoc queries are snapshot-scoped
user_query = "SELECT * FROM analytics.function_metrics"
safe_query = add_snapshot_filter(user_query, "org/repo", "abc123")
```

**Benefits**:
- Centralized query policy enforcement
- Automatic audit logging injection
- Schema migration transformations

**Implementation Priority**: HIGH - Improves query safety

---

### 3. Query Semantic Diff

**Current State**: Not used.

**Missed Opportunity**:

```python
from sqlglot import diff

def validate_migration_equivalence(old_query: str, new_query: str) -> bool:
    """Verify that a refactored query is semantically equivalent."""
    changes = diff(
        sqlglot.parse_one(old_query),
        sqlglot.parse_one(new_query)
    )
    
    # Only allow structural changes, not semantic
    semantic_changes = [c for c in changes if c.is_semantic]
    return len(semantic_changes) == 0
```

**Use Case**: View definition migrations, query refactoring validation

**Implementation Priority**: LOW - Useful for CI/testing

---

### 4. Embedded SQL Executor

**Current State**: All tests require actual DuckDB connections.

**Missed Opportunity**:

```python
from sqlglot.executor import execute

# Test query logic without database
tables = {
    "analytics.function_metrics": [
        {"function_goid_h128": "abc", "repo": "org/repo", "loc": 100},
        {"function_goid_h128": "def", "repo": "org/repo", "loc": 200},
    ]
}

result = execute(
    "SELECT repo, SUM(loc) FROM analytics.function_metrics GROUP BY repo",
    tables=tables
)
# Returns: [{"repo": "org/repo", "col1": 300}]
```

**Benefits**:
- Faster unit tests (no DB setup)
- Query logic validation in isolation
- Test data fixtures without persistence

**Implementation Priority**: MEDIUM - Improves test speed

---

## Implementation Recommendations

### Phase 1: Quick Wins (1-2 weeks)

1. **Arrow Integration** in `IbisGateway`:
   - Add `fetch_arrow_table()` method
   - Add `fetch_arrow_batches()` for streaming
   - Update repositories to use direct DataFrame methods

2. **UDF Registration** in `DuckDBSession`:
   - Create `register_standard_udfs()` method
   - Implement GOID hash UDF
   - Implement complexity score UDF

3. **Relational API** in `Warehouse`:
   - Replace manual filter construction with chainable API
   - Add `project()` support for column selection

### Phase 2: Structural Improvements (2-4 weeks)

4. **SQLGlot Query Transformer**:
   - Create `QueryTransformer` class
   - Implement automatic snapshot filtering
   - Add query optimization pass

5. **Extension Manager**:
   - Centralize extension loading
   - Add extension dependency resolution
   - Support optional extensions

6. **Complex Types**:
   - Add STRUCT support to schema provider
   - Update `_column_type_to_sqlglot()` for complex types
   - Create migration for nested data columns

### Phase 3: Advanced Features (4-8 weeks)

7. **Embedded SQL Executor for Tests**:
   - Create test harness using SQLGlot executor
   - Add fixtures for common table data
   - Speed up query logic tests

8. **Query Profiling Integration**:
   - Add `ProfiledQuery` context manager
   - Integrate with observability (OpenTelemetry)
   - Create profiling report generator

---

## Architectural Recommendations

### 1. Create `DuckDBFeatures` Facade

```python
@dataclass
class DuckDBFeatures:
    """Unified access to advanced DuckDB capabilities."""
    
    con: DuckDBPyConnection
    
    def register_udf(self, name: str, func: Callable, ...) -> None: ...
    def ensure_extensions(self, extensions: list[str]) -> None: ...
    def optimize_query(self, query: str, schema: dict) -> str: ...
    def explain_analyze(self, query: str) -> QueryPlan: ...
    def stream_results(self, query: str, batch_size: int) -> Iterator[pa.RecordBatch]: ...
```

### 2. Enhance `IbisGateway` with Arrow Methods

```python
class IbisGateway:
    def fetch_arrow(self, table_key: str, **filters) -> pa.Table: ...
    def fetch_arrow_batches(self, table_key: str, batch_size: int) -> Iterator[pa.RecordBatch]: ...
    def write_arrow(self, table_key: str, table: pa.Table) -> WriteResult: ...
```

### 3. Create `QueryPolicy` for Transformations

```python
class QueryPolicy:
    """Policy-based query transformation."""
    
    def enforce_snapshot_scope(self, query: str, snapshot: SnapshotRef) -> str: ...
    def add_audit_columns(self, query: str) -> str: ...
    def optimize(self, query: str) -> str: ...
```

---

## Conclusion

The CodeIntel storage layer has a solid foundation but is leaving significant DuckDB and SQLGlot capabilities unused. The highest-impact improvements are:

1. **Arrow integration** for zero-copy data transfer (performance)
2. **UDFs** for in-database computation (performance, reduced complexity)
3. **SQLGlot transformations** for query safety (robustness)
4. **Relational API adoption** for cleaner query building (maintainability)

These changes align with the project's principles of type safety, performance, and maintainability while reducing code complexity.

---

## References

- DuckDB Python API: https://duckdb.org/docs/api/python/overview
- DuckDB Relational API: https://duckdb.org/docs/api/python/relational_api
- SQLGlot Documentation: https://sqlglot.com/sqlglot.html
- CodeIntel AGENTS.md: Ibis 11 Patterns, Bulk Operations guidelines

