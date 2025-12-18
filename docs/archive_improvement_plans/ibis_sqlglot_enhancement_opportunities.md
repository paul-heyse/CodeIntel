# Ibis & SQLGlot Enhancement Opportunities

> **Purpose**: Identify advanced Ibis v11 and SQLGlot v28 functionality not currently leveraged in the CodeIntel storage layer, and propose targeted enhancements to improve robustness, maintainability, and functionality.

---

## Executive Summary

The CodeIntel storage layer already implements a solid foundation using Ibis for query building and SQLGlot for DDL/DML generation. However, several advanced capabilities remain untapped that could significantly streamline the codebase while boosting type safety, observability, and maintainability.

**High-Impact Opportunities (Ranked by Value/Effort)**:

1. **Ibis Schema ↔ SQLGlot DDL Round-Trips** — Eliminate manual column def building
2. **Typed Parameterization** — Replace ad-hoc parameter handling with `ibis.param()`
3. **SQLGlot AST Access** — Enable lineage, diffing, and query fingerprinting
4. **Query Optimization & Canonicalization** — Standardize SQL for caching/testing
5. **Streaming Results** — Add batched Arrow export for large result sets

---

## 1. Schema ↔ SQLGlot DDL Round-Trips

### Current State

`duckdb_policy_backend.py` manually builds SQLGlot column definitions:

```python
def _build_column_def(col_name: str, col_type: str, *, nullable: bool) -> exp.ColumnDef:
    constraints: list[exp.Expression] = []
    if not nullable:
        constraints.append(exp.NotNullColumnConstraint())

    return exp.ColumnDef(
        this=exp.to_identifier(col_name),
        kind=_column_type_to_sqlglot(col_type),
        constraints=[exp.ColumnConstraint(kind=c) for c in constraints] if constraints else None,
    )
```

This requires maintaining `_column_type_to_sqlglot()` mapping and manually handling nullability constraints.

### Advanced Feature Not Leveraged

**Ibis provides bidirectional schema conversion**:

```python
# Ibis Schema → SQLGlot ColumnDef list
sch = ibis.schema({"a": "int64", "b": "!string"})  # ! = not nullable
cols = sch.to_sqlglot_column_defs(dialect="duckdb")

# SQLGlot schema → Ibis Schema
ibis.Schema.from_sqlglot(sqlglot_schema_expr, dialect="duckdb")
```

### Proposed Enhancement

```python
# In duckdb_policy_backend.py - replace _build_column_def chain

def _build_create_table_from_schema(
    table_schema: TableSchema,
    *,
    if_not_exists: bool = False,
) -> exp.Create:
    """Build CREATE TABLE using Ibis schema → SQLGlot round-trip."""
    # Build Ibis schema from our TableSchema contract
    ibis_schema_dict = {
        col.name: f"{'!' if not col.nullable else ''}{col.type.lower()}"
        for col in table_schema.columns
    }
    ibis_sch = ibis.schema(ibis_schema_dict)
    
    # Let Ibis generate dialect-correct column defs
    col_defs = ibis_sch.to_sqlglot_column_defs(dialect=DUCKDB_DIALECT)
    
    # Add primary key if present
    if table_schema.primary_key:
        col_defs.append(_build_primary_key_constraint(table_schema.primary_key))
    
    return exp.Create(
        this=exp.Schema(
            this=exp.Table(
                this=exp.to_identifier(table_schema.name),
                db=exp.to_identifier(table_schema.schema),
            ),
            expressions=col_defs,
        ),
        kind="TABLE",
        exists=if_not_exists,
    )
```

### Benefits

- **Eliminate manual type mapping**: Remove `_column_type_to_sqlglot()` and its maintenance burden
- **Dialect correctness guaranteed**: Ibis handles DuckDB type nuances (TIMESTAMPTZ, DECIMAL precision)
- **Nullability via Ibis conventions**: `!` prefix is cleaner than manual constraint building
- **Future-proofing**: New types automatically supported via Ibis upgrades

---

## 2. Typed Parameterization with `ibis.param()`

### Current State

The semantic query builder (`query_builder.py`) uses ad-hoc literal binding:

```python
def _predicate_eq(col: it.Value, value: object) -> it.BooleanValue:
    return col == ibis.literal(value)
```

And views use raw value injection:

```python
# In ibis_views.py
small_loc = loc_col <= ibis.literal(CALLGRAPH_LOC_SMALL)
```

### Advanced Feature Not Leveraged

**Ibis deferred scalar parameters** enable "prepared statement style" templates:

```python
import ibis

# Define typed parameter
min_score = ibis.param("float64")

# Build template expression (compile once)
template = (
    t.filter(t.risk_score >= min_score)
    .filter(t.repo == ibis.param("string").name("repo"))
)

# Execute with bindings (bind many times)
result = template.execute(params={min_score: 0.8, ...})

# Compile to SQL with params resolved
sql = template.compile(params={min_score: 0.8}, pretty=True)
```

### Proposed Enhancement

Create a `QueryTemplate` pattern for the semantic layer:

```python
# In serving/semantic/templates.py

from dataclasses import dataclass
from typing import Any
import ibis
import ibis.expr.types as it

@dataclass(frozen=True)
class SemanticParam:
    """Typed deferred parameter for semantic queries."""
    name: str
    dtype: str
    required: bool = True
    default: Any = None
    
    def make_expr(self) -> it.Scalar:
        return ibis.param(self.dtype).name(self.name)


class SemanticTemplate:
    """Reusable typed query template."""
    
    def __init__(self, name: str) -> None:
        self.name = name
        self._params: dict[str, SemanticParam] = {}
        self._param_exprs: dict[str, it.Scalar] = {}
    
    def param(
        self,
        name: str,
        dtype: str,
        *,
        required: bool = True,
        default: Any = None,
    ) -> "SemanticTemplate":
        spec = SemanticParam(name=name, dtype=dtype, required=required, default=default)
        self._params[name] = spec
        self._param_exprs[name] = spec.make_expr()
        return self
    
    def compile_with_params(
        self,
        expr: it.Table,
        bindings: dict[str, Any],
    ) -> str:
        """Compile to SQL with type-checked parameter binding."""
        param_map = {}
        for name, spec in self._params.items():
            value = bindings.get(name, spec.default)
            if value is None and spec.required:
                raise ValueError(f"Required parameter {name} not provided")
            param_map[self._param_exprs[name]] = value
        return expr.compile(params=param_map, pretty=True)
```

### Benefits

- **Type safety at definition time**: Params have declared Ibis types
- **SQL template caching**: Expression shape is stable; only values change
- **Audit-friendly SQL generation**: `compile(params=..., pretty=True)` yields deterministic SQL
- **No SQL injection surface**: Values are never string-interpolated

---

## 3. SQLGlot AST Access via `con.compiler.to_sqlglot()`

### Current State

SQL generation uses `ibis.to_sql()` which returns a string:

```python
# In ibis_adapter.py
select_sql = ibis.to_sql(expr, dialect=DUCKDB_DIALECT)
```

This discards the intermediate AST, losing opportunities for analysis and transformation.

### Advanced Feature Not Leveraged

**Ibis exposes the SQLGlot AST** before stringification:

```python
# Get SQLGlot Expression directly
sg_expr = con.compiler.to_sqlglot(expr)

# Now you can:
# 1. Extract metadata (tables, columns referenced)
# 2. Apply transforms (inject filters, normalize identifiers)
# 3. Compute semantic diffs
# 4. Generate lineage graphs
# 5. Then render to SQL
sql = sg_expr.sql(dialect="duckdb", pretty=True)
```

### Proposed Enhancement

Add AST access utilities to `IbisGateway`:

```python
# In ibis_adapter.py

class IbisGateway:
    # ... existing code ...
    
    def to_sqlglot(self, expr: it.Expr) -> exp.Expression:
        """Return SQLGlot AST for an Ibis expression.
        
        Use for:
        - Column lineage extraction
        - Query fingerprinting/canonicalization
        - Semantic diffing between query versions
        - AST-level transforms (filter injection, identifier normalization)
        """
        return self.con.compiler.to_sqlglot(expr)
    
    def extract_referenced_tables(self, expr: it.Table) -> set[str]:
        """Extract all tables referenced in an expression."""
        sg_expr = self.to_sqlglot(expr)
        return {
            f"{t.db}.{t.this.name}" if t.db else t.this.name
            for t in sg_expr.find_all(exp.Table)
        }
    
    def extract_referenced_columns(self, expr: it.Table) -> set[tuple[str, str]]:
        """Extract (table, column) pairs from expression."""
        sg_expr = self.to_sqlglot(expr)
        return {
            (col.table or "", col.name)
            for col in sg_expr.find_all(exp.Column)
        }
```

### Benefits

- **Query observability**: Know exactly which tables/columns a query touches
- **Access control integration**: Validate queries against allowlists at AST level
- **Lineage extraction**: Feed into OpenLineage or custom lineage systems
- **AST-based testing**: Compare query structure, not brittle SQL strings

---

## 4. Query Optimization & Canonicalization

### Current State

No query canonicalization or fingerprinting is performed. Each compilation produces slightly different SQL formatting depending on expression construction order.

### Advanced Feature Not Leveraged

**SQLGlot's optimizer produces canonical ASTs**:

```python
from sqlglot import optimizer

# Optimize and canonicalize
canonical = optimizer.optimize(
    sg_expr,
    dialect="duckdb",
    schema=schema_dict,  # optional but improves rewrites
)

# Now canonical.sql() gives deterministic output
fingerprint = hashlib.sha256(canonical.sql(dialect="duckdb").encode()).hexdigest()[:16]
```

### Proposed Enhancement

Add canonicalization for query caching and golden tests:

```python
# In storage/helpers/query_fingerprint.py

import hashlib
from sqlglot import optimizer, exp

def canonicalize_query(sg_expr: exp.Expression, *, schema: dict | None = None) -> exp.Expression:
    """Return canonical form of a SQLGlot expression.
    
    Useful for:
    - Query fingerprinting (cache keys)
    - Golden SQL tests (stable output)
    - Semantic comparison (ignore formatting differences)
    """
    return optimizer.optimize(
        sg_expr,
        dialect="duckdb",
        schema=schema or {},
        rules=(
            optimizer.qualify.qualify,
            optimizer.normalize.normalize,
        ),
    )


def query_fingerprint(sg_expr: exp.Expression, *, schema: dict | None = None) -> str:
    """Generate a stable fingerprint for a query.
    
    Two semantically equivalent queries produce the same fingerprint.
    """
    canonical = canonicalize_query(sg_expr, schema=schema)
    sql = canonical.sql(dialect="duckdb", pretty=False)
    return hashlib.sha256(sql.encode()).hexdigest()[:16]
```

### Benefits

- **Query-level caching**: Use fingerprints as cache keys for expensive computations
- **Golden test stability**: Canonical SQL doesn't change with Ibis/SQLGlot minor versions
- **Deduplication**: Detect semantically equivalent queries built differently

---

## 5. Semantic Query Diffing

### Current State

No semantic comparison of queries exists. Changes to view definitions are compared as SQL strings.

### Advanced Feature Not Leveraged

**SQLGlot provides semantic diff**:

```python
from sqlglot import diff, parse_one

old = parse_one("SELECT a, b FROM t WHERE x > 1")
new = parse_one("SELECT a, c FROM t WHERE x > 2")

changes = diff(old, new)
# Returns: [Keep(a), Remove(b), Insert(c), Update(Literal(1) -> Literal(2))]
```

### Proposed Enhancement

Add semantic diff for view evolution tracking:

```python
# In storage/views/diff.py

from sqlglot import diff, exp
from dataclasses import dataclass
from enum import Enum


class ChangeType(Enum):
    KEEP = "keep"
    INSERT = "insert"
    REMOVE = "remove"
    UPDATE = "update"
    MOVE = "move"


@dataclass(frozen=True)
class QueryChange:
    change_type: ChangeType
    node_type: str
    old_value: str | None
    new_value: str | None


def diff_queries(
    old_expr: exp.Expression,
    new_expr: exp.Expression,
) -> list[QueryChange]:
    """Compute semantic diff between two queries."""
    raw_diff = diff(old_expr, new_expr)
    return [
        QueryChange(
            change_type=ChangeType(change.__class__.__name__.lower()),
            node_type=change.expression.__class__.__name__,
            old_value=getattr(change, "source", change.expression).sql() if hasattr(change, "source") else None,
            new_value=change.expression.sql(),
        )
        for change in raw_diff
        if not isinstance(change, diff.Keep)
    ]


def summarize_view_changes(
    old_view_sql: str,
    new_view_sql: str,
) -> dict[str, list[str]]:
    """Summarize changes between view versions for changelog."""
    from sqlglot import parse_one
    
    changes = diff_queries(
        parse_one(old_view_sql, dialect="duckdb"),
        parse_one(new_view_sql, dialect="duckdb"),
    )
    
    return {
        "columns_added": [c.new_value for c in changes if c.change_type == ChangeType.INSERT and c.node_type == "Column"],
        "columns_removed": [c.old_value for c in changes if c.change_type == ChangeType.REMOVE and c.node_type == "Column"],
        "filters_changed": [c for c in changes if c.node_type in ("Where", "EQ", "GT", "LT")],
    }
```

### Benefits

- **Change detection**: Know exactly what changed in a view definition
- **Breaking change warnings**: Detect column removals, type changes
- **Automated changelogs**: Generate meaningful descriptions of view evolution

---

## 6. Large IN-List Handling via Memtable + Join

### Current State

The semantic query builder handles `IN` filters with literal lists:

```python
if op == "in":
    if not isinstance(value, list):
        msg = "IN operator requires list value"
        raise QueryBuilderError(msg)
    values = [ibis.literal(v) for v in value]
    return col_expr.isin(values)
```

This generates `column IN (?, ?, ?, ...)` which can exceed driver placeholder limits for large lists.

### Advanced Feature Not Leveraged

**Memtable + semi-join pattern** handles arbitrarily large lists:

```python
# Stage list as temp table
ids_table = ibis.memtable({"id": large_id_list})
con.create_table("_temp_ids", ids_table, temp=True, overwrite=True)

# Use semi-join instead of IN
result = (
    base_table
    .semi_join(con.table("_temp_ids"), base_table.id == con.table("_temp_ids").id)
)
```

### Proposed Enhancement

Add list parameter handling to semantic queries:

```python
# In serving/semantic/query_builder.py

IN_LIST_THRESHOLD = 100  # Use memtable pattern above this size


def _build_predicate_in(
    *,
    table: it.Table,
    col_expr: it.Value,
    values: list,
    con: DuckDBBackend,
    instance_id: str,
) -> it.BooleanValue:
    """Build IN predicate, using memtable for large lists."""
    if len(values) <= IN_LIST_THRESHOLD:
        return col_expr.isin([ibis.literal(v) for v in values])
    
    # Stage large list as temp table
    col_name = col_expr.get_name()
    temp_name = f"_qt_in_{col_name}_{instance_id}"
    
    mt = ibis.memtable({col_name: list(set(values))})  # dedupe
    con.create_table(temp_name, mt, temp=True, overwrite=True)
    
    # Return semi-join predicate (caller applies to table)
    return table.inner_join(
        con.table(temp_name),
        col_expr == con.table(temp_name)[col_name],
    ).select(table)  # Preserve original columns
```

### Benefits

- **No placeholder limits**: Works with millions of IDs
- **Better query plans**: DuckDB can hash-join instead of OR-chain
- **Deduplication built-in**: Prevents duplicate results

---

## 7. Streaming Results with Arrow Batches

### Current State

Large result sets are loaded entirely into memory:

```python
# In kernel.py
df_pd = expr.to_pandas()
return df_pd.to_dict(orient="records")
```

### Advanced Feature Not Leveraged

**Ibis supports streaming via RecordBatchReader**:

```python
# Streaming export
batches = con.to_pyarrow_batches(expr, chunk_size=10_000)
for batch in batches:
    process_batch(batch)  # Memory-efficient processing
```

### Proposed Enhancement

Add streaming export to the serving layer:

```python
# In serving/http/streaming.py

from collections.abc import AsyncIterator
import pyarrow as pa
from fastapi.responses import StreamingResponse


async def stream_query_results(
    expr: it.Table,
    con: DuckDBBackend,
    *,
    chunk_size: int = 10_000,
    format: str = "jsonl",
) -> AsyncIterator[bytes]:
    """Stream query results in chunks."""
    batches = con.to_pyarrow_batches(expr, chunk_size=chunk_size)
    
    for batch in batches:
        if format == "jsonl":
            df = batch.to_pandas()
            for _, row in df.iterrows():
                yield (row.to_json() + "\n").encode()
        elif format == "arrow":
            sink = pa.BufferOutputStream()
            writer = pa.ipc.new_stream(sink, batch.schema)
            writer.write_batch(batch)
            writer.close()
            yield sink.getvalue().to_pybytes()
```

### Benefits

- **Constant memory usage**: Process million-row results without OOM
- **Faster time-to-first-byte**: Client receives data immediately
- **Backpressure support**: Natural flow control via iteration

---

## 8. DuckDB Extension Bootstrap at Connect Time

### Current State

Extensions are loaded ad-hoc:

```python
# Scattered across codebase
con.execute("INSTALL httpfs; LOAD httpfs;")
```

### Advanced Feature Not Leveraged

**Ibis connect accepts `extensions=` parameter**:

```python
con = ibis.duckdb.connect(
    database="local.ddb",
    extensions=["httpfs", "json", "parquet"],  # Auto install+load
    threads=8,
    memory_limit="4GB",
    temp_directory="/tmp/duckdb",
)
```

### Proposed Enhancement

Centralize extension management in gateway initialization:

```python
# In storage/gateway/factory.py

REQUIRED_EXTENSIONS = ["json", "parquet"]
OPTIONAL_EXTENSIONS = ["httpfs", "spatial", "delta"]


def create_duckdb_connection(
    config: StorageConfig,
) -> DuckDBPyConnection:
    """Create configured DuckDB connection with extensions."""
    extensions = REQUIRED_EXTENSIONS.copy()
    
    if config.enable_cloud_access:
        extensions.append("httpfs")
    if config.enable_spatial:
        extensions.append("spatial")
    
    return ibis.duckdb.connect(
        database=str(config.database_path),
        read_only=config.read_only,
        extensions=extensions,
        threads=config.threads,
        memory_limit=config.memory_limit,
        temp_directory=str(config.temp_directory),
    )
```

### Benefits

- **Deterministic startup**: All extensions loaded before first query
- **Configuration-driven**: Enable/disable features via config
- **Fail-fast**: Missing extensions error at connect, not mid-query

---

## 9. Cross-Dialect SQL Migration

### Current State

Raw SQL strings are assumed to be DuckDB dialect. No migration path for SQL from other sources.

### Advanced Feature Not Leveraged

**Ibis/SQLGlot transpile SQL between dialects**:

```python
# Parse MySQL syntax, execute on DuckDB
result = con.sql(
    "SELECT DATE_FORMAT(ts, '%Y-%m') as month FROM events",
    dialect="mysql"  # SQLGlot transpiles to DuckDB
)
```

### Proposed Enhancement

Add dialect-aware SQL ingestion:

```python
# In storage/helpers/sql_compat.py

from sqlglot import transpile

SUPPORTED_SOURCE_DIALECTS = {"mysql", "postgres", "sqlite", "spark"}


def migrate_sql_to_duckdb(
    sql: str,
    *,
    source_dialect: str,
) -> str:
    """Transpile SQL from another dialect to DuckDB."""
    if source_dialect not in SUPPORTED_SOURCE_DIALECTS:
        raise ValueError(f"Unsupported source dialect: {source_dialect}")
    
    return transpile(
        sql,
        read=source_dialect,
        write="duckdb",
        pretty=True,
    )[0]


def execute_cross_dialect_sql(
    con: DuckDBBackend,
    sql: str,
    *,
    source_dialect: str | None = None,
) -> it.Table:
    """Execute SQL, optionally from another dialect."""
    if source_dialect:
        sql = migrate_sql_to_duckdb(sql, source_dialect=source_dialect)
    return con.sql(sql)
```

### Benefits

- **Migration support**: Bring existing SQL assets into CodeIntel
- **Multi-source compatibility**: Accept queries from various BI tools
- **Gradual modernization**: Run legacy SQL while migrating to Ibis

---

## 10. Column-Level Lineage Extraction

### Current State

No column-level lineage is tracked. We know which views exist but not which columns flow where.

### Advanced Feature Not Leveraged

**SQLGlot provides lineage graph construction**:

```python
from sqlglot.lineage import lineage

# Build lineage for a specific output column
result = lineage(
    "output_col",
    sql="SELECT a + b AS output_col FROM t",
    dialect="duckdb",
)

# result.source contains input columns that flow to output_col
```

### Proposed Enhancement

Add lineage extraction for documentation and impact analysis:

```python
# In storage/views/lineage.py

from sqlglot.lineage import lineage
from sqlglot import parse_one
from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnLineage:
    output_column: str
    source_columns: frozenset[tuple[str, str]]  # (table, column)
    transformations: tuple[str, ...]  # function names applied


def extract_view_lineage(
    view_sql: str,
    output_columns: list[str] | None = None,
) -> dict[str, ColumnLineage]:
    """Extract column-level lineage for a view definition."""
    parsed = parse_one(view_sql, dialect="duckdb")
    
    if output_columns is None:
        # Get all output columns from SELECT
        output_columns = [
            col.alias_or_name
            for col in parsed.find_all(exp.Alias, exp.Column)
            if col.parent and isinstance(col.parent, exp.Select)
        ]
    
    lineage_map = {}
    for col in output_columns:
        try:
            result = lineage(col, sql=view_sql, dialect="duckdb")
            sources = frozenset(
                (node.source.alias_or_name, node.name)
                for node in result.walk()
                if hasattr(node, "source") and hasattr(node, "name")
            )
            lineage_map[col] = ColumnLineage(
                output_column=col,
                source_columns=sources,
                transformations=tuple(),  # Could extract function names
            )
        except Exception:
            # Some columns may not have traceable lineage
            pass
    
    return lineage_map
```

### Benefits

- **Impact analysis**: Know which views break when a source column changes
- **Documentation**: Auto-generate "where does this data come from?" docs
- **Data governance**: Track PII flow through the system

---

## Implementation Priorities

### Phase 1: Foundation (Week 1-2)

1. **Schema round-trips** (§1) — Immediate code reduction
2. **SQLGlot AST access** (§3) — Enables all subsequent features
3. **Extension bootstrap** (§8) — Stabilizes environment

### Phase 2: Query Intelligence (Week 3-4)

4. **Typed parameterization** (§2) — Better semantic layer
5. **Query canonicalization** (§4) — Cache key generation
6. **Large IN-list handling** (§6) — Production robustness

### Phase 3: Observability (Week 5-6)

7. **Semantic diffing** (§5) — View evolution tracking
8. **Lineage extraction** (§10) — Documentation & governance
9. **Streaming results** (§7) — Large result handling

### Phase 4: Migration Support (Week 7+)

10. **Cross-dialect SQL** (§9) — External SQL integration

---

## References

- [Ibis v11 Release Notes](https://ibis-project.org/release_notes)
- [Ibis DuckDB Backend](https://ibis-project.org/backends/duckdb)
- [Ibis Schema Reference](https://ibis-project.org/reference/schemas)
- [SQLGlot Documentation](https://sqlglot.com/)
- [SQLGlot Lineage API](https://sqlglot.com/sqlglot/lineage.html)
- [DuckDB Python DB-API](https://duckdb.org/docs/stable/clients/python/dbapi.html)

