# Holistic Data Operations Enhancement Plan

> **Purpose**: Unified assessment of DuckDB, Ibis, and SQLGlot enhancement opportunities across the CodeIntel storage, build, and serving layers—identifying synergies, integration points, and a cohesive implementation roadmap.

**Generated**: 2024-12-17  
**Status**: Strategic Assessment  
**Scope**: Storage layer and integrations with build/serving

---

## Executive Summary

The CodeIntel data operations stack is built on three powerful technologies—**DuckDB** (execution), **Ibis** (query building), and **SQLGlot** (SQL generation/analysis)—but currently uses only a fraction of their combined capabilities. This document presents a **holistic enhancement strategy** that leverages synergies between these tools to achieve:

1. **40-60% reduction in boilerplate** through schema/DDL automation
2. **Type-safe query composition** via Ibis parameterization and SQLGlot AST access
3. **Zero-copy data interchange** using Arrow throughout the pipeline
4. **Query observability** through lineage extraction, fingerprinting, and profiling
5. **Extensibility** via centralized extension/UDF management

### Strategic Insight: The Three-Layer Integration Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                          │
│    (Serving: HTTP APIs, Semantic Search, MCP)                      │
│         ↓ Ibis expressions + typed params                          │
├─────────────────────────────────────────────────────────────────────┤
│                      QUERY INTELLIGENCE LAYER                       │
│    Ibis v11 ↔ SQLGlot v28 bidirectional compilation                │
│    • Schema round-trips     • AST transforms      • Lineage        │
│    • Parameterization       • Fingerprinting      • Optimization   │
│         ↓ SQLGlot expressions                                       │
├─────────────────────────────────────────────────────────────────────┤
│                        EXECUTION LAYER                             │
│    DuckDB (Connection, Relational API, Extensions, UDFs)           │
│    • Complex types (STRUCT, LIST, MAP)                             │
│    • Arrow integration (zero-copy)                                 │
│    • Vectorized UDFs                                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part I: Integrated Gap Analysis

### Current Architecture Summary

| Component | Current Usage | Untapped Potential |
|-----------|---------------|-------------------|
| **DuckDB** | `execute()`, basic Relational API | Complex types, Arrow streaming, UDFs, profiling |
| **Ibis** | Query building, `to_sql()` | Schema round-trips, `ibis.param()`, AST access |
| **SQLGlot** | DDL generation in `DuckDBPolicyBackend` | Optimization, transforms, lineage, diffing |

### Key Integration Gaps

1. **Schema Management**: Manual type mapping in `_column_type_to_sqlglot()` when Ibis provides `Schema.to_sqlglot_column_defs()`
2. **Query Compilation**: Using `ibis.to_sql()` (string) when `con.compiler.to_sqlglot()` (AST) enables analysis
3. **Parameterization**: Ad-hoc `ibis.literal()` when `ibis.param()` provides type-safe templates
4. **Data Transfer**: Pandas intermediaries when Arrow provides zero-copy
5. **Query Analysis**: No fingerprinting, lineage, or semantic diffing

---

## Part II: Enhancement Opportunities (Integrated)

### 1. Unified Schema Management Pipeline

**Current Pain Point**: Three separate schema representations (Python dataclasses, TableSchema, SQLGlot column defs) that must be kept in sync manually.

**Integrated Solution**: Single-source schema definition flowing through all layers.

```python
# PROPOSED: Unified schema flow
from dataclasses import dataclass
import ibis
from sqlglot import exp

@dataclass(frozen=True)
class UnifiedSchema:
    """Single source of truth for table schemas."""
    
    name: str
    schema: str
    ibis_schema: ibis.Schema
    primary_key: tuple[str, ...] = ()
    
    @classmethod
    def from_columns(
        cls,
        name: str,
        schema: str,
        columns: dict[str, str],  # {"col": "int64", "col2": "!string"} (! = not nullable)
        primary_key: tuple[str, ...] = (),
    ) -> "UnifiedSchema":
        """Create from Ibis schema notation."""
        return cls(
            name=name,
            schema=schema,
            ibis_schema=ibis.schema(columns),
            primary_key=primary_key,
        )
    
    def to_sqlglot_create_table(self, *, if_not_exists: bool = True) -> exp.Create:
        """Generate SQLGlot CREATE TABLE using Ibis round-trip."""
        col_defs = self.ibis_schema.to_sqlglot_column_defs(dialect="duckdb")
        
        if self.primary_key:
            pk_constraint = exp.PrimaryKey(
                expressions=[exp.to_identifier(c) for c in self.primary_key]
            )
            col_defs.append(pk_constraint)
        
        return exp.Create(
            this=exp.Schema(
                this=exp.Table(
                    this=exp.to_identifier(self.name),
                    db=exp.to_identifier(self.schema),
                ),
                expressions=col_defs,
            ),
            kind="TABLE",
            exists=if_not_exists,
        )
    
    def to_pandera_schema(self) -> "pa.DataFrameSchema":
        """Generate Pandera validation schema."""
        # Leverage existing schema registry integration
        ...
```

**Impact**: 
- Eliminates `_column_type_to_sqlglot()` (~50 lines)
- Eliminates `_build_column_def()` chain (~30 lines)
- Type correctness guaranteed by Ibis

**Files Affected**: `duckdb_policy_backend.py`, `storage/schema/`, `build/hamilton/contracts/`

---

### 2. SQLGlot AST Access Layer

**Current Pain Point**: SQL is generated as strings, losing opportunities for analysis, transformation, and caching.

**Integrated Solution**: Central AST access point in `IbisGateway`.

```python
# PROPOSED: Add to IbisGateway (ibis_adapter.py)

from sqlglot import exp, optimizer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Set

class IbisGateway:
    # ... existing code ...
    
    def to_sqlglot(self, expr: it.Expr) -> exp.Expression:
        """Return SQLGlot AST for an Ibis expression.
        
        Parameters
        ----------
        expr
            Any Ibis expression (Table, Scalar, etc.)
        
        Returns
        -------
        exp.Expression
            SQLGlot AST that can be analyzed, transformed, or rendered.
        
        Notes
        -----
        Use cases:
        - Column lineage extraction
        - Query fingerprinting for caching
        - Semantic diffing between query versions
        - AST-level transforms (filter injection, identifier normalization)
        
        Examples
        --------
        >>> sg_expr = gateway.ibis.to_sqlglot(expr)
        >>> tables = {t.name for t in sg_expr.find_all(exp.Table)}
        >>> sql = sg_expr.sql(dialect="duckdb", pretty=True)
        """
        return self.con.compiler.to_sqlglot(expr)
    
    def extract_table_lineage(self, expr: it.Table) -> set[str]:
        """Extract all tables referenced by an expression."""
        sg_expr = self.to_sqlglot(expr)
        return {
            f"{t.db}.{t.this.name}" if t.db else t.this.name
            for t in sg_expr.find_all(exp.Table)
        }
    
    def extract_column_lineage(self, expr: it.Table) -> set[tuple[str, str]]:
        """Extract (table, column) pairs from expression."""
        sg_expr = self.to_sqlglot(expr)
        return {
            (col.table or "", col.name)
            for col in sg_expr.find_all(exp.Column)
        }
    
    def canonicalize(self, expr: it.Expr, *, schema: dict[str, dict[str, str]] | None = None) -> exp.Expression:
        """Return canonical SQLGlot AST for fingerprinting/caching."""
        sg_expr = self.to_sqlglot(expr)
        return optimizer.optimize(
            sg_expr,
            dialect="duckdb",
            schema=schema or {},
            rules=(
                optimizer.qualify.qualify,
                optimizer.normalize.normalize,
            ),
        )
    
    def query_fingerprint(self, expr: it.Expr, *, schema: dict[str, dict[str, str]] | None = None) -> str:
        """Generate stable fingerprint for query caching."""
        import hashlib
        canonical = self.canonicalize(expr, schema=schema)
        sql = canonical.sql(dialect="duckdb", pretty=False)
        return hashlib.sha256(sql.encode()).hexdigest()[:16]
```

**Impact**:
- Enables query-level caching in serving layer
- Powers automated lineage for documentation
- Foundation for semantic view diffing

**Integration Points**:
- `serving/semantic/kernel.py`: Use fingerprints for response caching
- `build/contracts.py`: Extract lineage for dependency tracking
- `storage/views/`: Enable semantic diff for view evolution

---

### 3. Typed Parameterization System

**Current Pain Point**: Parameters injected via `ibis.literal()` or string formatting, losing type safety and caching potential.

**Integrated Solution**: Ibis `param()` based template system.

```python
# PROPOSED: serving/semantic/templates.py

from dataclasses import dataclass, field
from typing import Any
import ibis
import ibis.expr.types as it

@dataclass(frozen=True)
class QueryParam:
    """Typed deferred parameter specification."""
    
    name: str
    dtype: str  # Ibis type string (e.g., "string", "int64", "float64")
    required: bool = True
    default: Any = None
    description: str = ""
    
    def make_expr(self) -> it.Scalar:
        """Create Ibis scalar parameter expression."""
        return ibis.param(self.dtype).name(self.name)


@dataclass
class QueryTemplate:
    """Reusable typed query template with parameter binding."""
    
    name: str
    description: str = ""
    _params: dict[str, QueryParam] = field(default_factory=dict)
    _param_exprs: dict[str, it.Scalar] = field(default_factory=dict)
    
    def param(
        self,
        name: str,
        dtype: str,
        *,
        required: bool = True,
        default: Any = None,
        description: str = "",
    ) -> "QueryTemplate":
        """Add a typed parameter to the template."""
        spec = QueryParam(
            name=name,
            dtype=dtype,
            required=required,
            default=default,
            description=description,
        )
        self._params[name] = spec
        self._param_exprs[name] = spec.make_expr()
        return self
    
    def get_param_expr(self, name: str) -> it.Scalar:
        """Get the Ibis parameter expression for use in query building."""
        return self._param_exprs[name]
    
    def bind(self, **bindings: Any) -> dict[it.Scalar, Any]:
        """Validate and bind parameters for execution."""
        param_map: dict[it.Scalar, Any] = {}
        
        for name, spec in self._params.items():
            value = bindings.get(name, spec.default)
            
            if value is None and spec.required:
                msg = f"Required parameter '{name}' not provided"
                raise ValueError(msg)
            
            if value is not None:
                param_map[self._param_exprs[name]] = value
        
        return param_map
    
    def compile_sql(
        self,
        expr: it.Table,
        bindings: dict[str, Any],
        *,
        pretty: bool = True,
    ) -> str:
        """Compile to SQL with bound parameters."""
        param_map = self.bind(**bindings)
        return expr.compile(params=param_map, pretty=pretty)
    
    def execute(
        self,
        expr: it.Table,
        bindings: dict[str, Any],
    ) -> Any:
        """Execute with bound parameters."""
        param_map = self.bind(**bindings)
        return expr.execute(params=param_map)


# Usage example in semantic query builder:
def build_semantic_query_template() -> tuple[QueryTemplate, it.Table]:
    """Build reusable semantic search template."""
    template = (
        QueryTemplate("semantic_search", "Vector similarity search")
        .param("repo", "string", required=True, description="Repository identifier")
        .param("min_score", "float64", default=0.5, description="Minimum similarity score")
        .param("limit", "int64", default=100, description="Maximum results")
    )
    
    # Build expression using parameters
    t = con.table("vectors.embeddings")
    expr = (
        t.filter(t.repo == template.get_param_expr("repo"))
        .filter(t.score >= template.get_param_expr("min_score"))
        .limit(template.get_param_expr("limit"))
    )
    
    return template, expr
```

**Impact**:
- Type-safe parameter binding
- SQL injection prevention (values never interpolated)
- Query plan caching (same SQL shape)

**Integration Points**:
- `serving/semantic/query_builder.py`: Replace literal injection
- `build/exports/exprs.py`: Template-based export queries

---

### 4. Arrow-Native Data Pipeline

**Current Pain Point**: Data converted to Pandas, losing zero-copy benefits and adding memory overhead.

**Integrated Solution**: Arrow-first data flow with streaming support.

```python
# PROPOSED: Add to IbisGateway

from collections.abc import Iterator
import pyarrow as pa

class IbisGateway:
    # ... existing code ...
    
    def fetch_arrow(self, expr: it.Table) -> pa.Table:
        """Execute expression and return Arrow Table (zero-copy when possible)."""
        return expr.to_pyarrow()
    
    def fetch_arrow_batches(
        self,
        expr: it.Table,
        *,
        chunk_size: int = 10_000,
    ) -> Iterator[pa.RecordBatch]:
        """Stream results as Arrow batches for memory-efficient processing.
        
        Parameters
        ----------
        expr
            Ibis table expression to execute.
        chunk_size
            Number of rows per batch.
        
        Yields
        ------
        pa.RecordBatch
            Arrow record batches for streaming processing.
        
        Notes
        -----
        Use for large result sets to maintain constant memory usage.
        Particularly useful for:
        - Streaming HTTP responses (JSONL, Arrow IPC)
        - Incremental processing pipelines
        - Memory-constrained environments
        """
        return expr.to_pyarrow_batches(chunk_size=chunk_size)
    
    def write_arrow(
        self,
        table_key: str,
        arrow_table: pa.Table,
        *,
        on_conflict: OnConflict | None = None,
    ) -> WriteResult:
        """Write Arrow Table directly (zero-copy when possible)."""
        schema, name = table_key.split(".", 1)
        
        # Convert Arrow to DuckDB-friendly format
        # DuckDB can read Arrow natively via replacement scan
        temp_name = f"_arrow_{name}_{id(arrow_table)}"
        self._gateway.con.register(temp_name, arrow_table)
        
        try:
            # Use INSERT...SELECT from registered Arrow table
            expr = self.con.table(temp_name)
            return self._write_ibis_expression(table_key, expr, on_conflict=on_conflict)
        finally:
            self._gateway.con.unregister(temp_name)


# PROPOSED: Streaming HTTP response in serving layer

async def stream_query_results(
    expr: it.Table,
    con: ibis.backends.duckdb.Backend,
    *,
    chunk_size: int = 10_000,
    format: str = "jsonl",
) -> AsyncIterator[bytes]:
    """Stream query results as HTTP response chunks.
    
    Parameters
    ----------
    expr
        Ibis expression to stream.
    con
        DuckDB backend connection.
    chunk_size
        Rows per batch.
    format
        Output format: 'jsonl', 'arrow', or 'csv'.
    
    Yields
    ------
    bytes
        Encoded response chunks.
    """
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
        elif format == "csv":
            yield batch.to_pandas().to_csv(index=False, header=False).encode()
```

**Impact**:
- Constant memory for large results
- Faster time-to-first-byte
- Native integration with ML frameworks (PyTorch, TensorFlow)

**Integration Points**:
- `serving/http/`: Streaming endpoints
- `build/hamilton/io/`: Arrow-based data loaders
- `storage/repositories/`: Arrow-first read methods

---

### 5. DuckDB Complex Types for Nested Data

**Current Pain Point**: Flattened schemas lose semantic structure; metadata stored as JSON strings parsed at runtime.

**Integrated Solution**: Native DuckDB complex types (STRUCT, LIST, MAP).

```python
# PROPOSED: Complex type support in schema definitions

from duckdb.typing import DuckDBPyType

# Define complex types for code intelligence data
FUNCTION_PARAMS_TYPE = DuckDBPyType(list[{
    "name": str,
    "type": str,
    "default": str | None,
    "position": int,
}])

SYMBOL_LOCATION_TYPE = DuckDBPyType({
    "file_path": str,
    "line_start": int,
    "line_end": int,
    "column_start": int,
    "column_end": int,
})

EDGE_METADATA_TYPE = DuckDBPyType({str: str})  # MAP for flexible attributes


# Example schema using complex types
ENHANCED_FUNCTION_METRICS_SCHEMA = UnifiedSchema.from_columns(
    name="function_metrics_v2",
    schema="analytics",
    columns={
        "function_goid_h128": "!string",
        "repo": "!string",
        "commit": "!string",
        "name": "!string",
        "qualified_name": "!string",
        # Complex nested types (new)
        "location": SYMBOL_LOCATION_TYPE,  # STRUCT
        "parameters": FUNCTION_PARAMS_TYPE,  # LIST<STRUCT>
        "annotations": DuckDBPyType(list[str]),  # LIST<VARCHAR>
        "metrics": DuckDBPyType({str: float}),  # MAP<VARCHAR, DOUBLE>
        # Scalars
        "loc": "!int64",
        "cyclomatic_complexity": "!int64",
    },
    primary_key=("function_goid_h128", "repo", "commit"),
)


# Query nested data naturally
def query_complex_function(gateway: StorageGateway, repo: str, commit: str):
    """Query function with nested data access."""
    t = gateway.ibis.table("analytics.function_metrics_v2")
    
    return (
        t.filter(t.repo == repo)
        .filter(t.commit == commit)
        # Access nested fields
        .mutate(
            file_path=t.location["file_path"],
            line_start=t.location["line_start"],
            param_count=t.parameters.length(),
            has_default=t.parameters.map(lambda p: p["default"].notnull()).any(),
        )
        .select("name", "file_path", "line_start", "param_count", "has_default")
    )
```

**Impact**:
- More natural data modeling (matches code structure)
- Fewer JOINs for related data
- Better query expressiveness

**Migration Consideration**: Requires schema evolution; implement as v2 tables alongside existing.

---

### 6. Centralized Extension & UDF Management

**Current Pain Point**: Extensions loaded ad-hoc; no standard UDF registration.

**Integrated Solution**: Gateway-level extension and UDF bootstrap.

```python
# PROPOSED: storage/gateway/extensions.py

from dataclasses import dataclass, field
from typing import Callable, Any
import duckdb

@dataclass
class ExtensionSpec:
    """Specification for a DuckDB extension."""
    
    name: str
    required: bool = True
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class UDFSpec:
    """Specification for a Python UDF."""
    
    name: str
    func: Callable
    parameters: list[str] | None = None  # None = infer from annotations
    return_type: str | None = None
    vectorized: bool = False  # type='arrow' if True
    null_handling: str = "NULL"  # "NULL" or "special"
    exception_handling: str = "throw"  # "throw" or "return_null"


# Standard extensions for CodeIntel
REQUIRED_EXTENSIONS: list[ExtensionSpec] = [
    ExtensionSpec("json"),
    ExtensionSpec("parquet"),
    ExtensionSpec("fts"),  # Full-text search
]

OPTIONAL_EXTENSIONS: list[ExtensionSpec] = [
    ExtensionSpec("httpfs", required=False, config={"s3_region": "us-west-2"}),
    ExtensionSpec("spatial", required=False),
    ExtensionSpec("delta", required=False),
]


def ensure_extensions(
    con: duckdb.DuckDBPyConnection,
    *,
    enable_cloud: bool = False,
    enable_spatial: bool = False,
) -> list[str]:
    """Load required extensions and optionally enable optional ones.
    
    Returns
    -------
    list[str]
        Names of loaded extensions.
    """
    loaded: list[str] = []
    
    # Required extensions
    for ext in REQUIRED_EXTENSIONS:
        _load_extension(con, ext)
        loaded.append(ext.name)
    
    # Optional extensions based on flags
    if enable_cloud:
        httpfs = next(e for e in OPTIONAL_EXTENSIONS if e.name == "httpfs")
        _load_extension(con, httpfs)
        loaded.append("httpfs")
    
    if enable_spatial:
        spatial = next(e for e in OPTIONAL_EXTENSIONS if e.name == "spatial")
        _load_extension(con, spatial)
        loaded.append("spatial")
    
    return loaded


def _load_extension(con: duckdb.DuckDBPyConnection, spec: ExtensionSpec) -> None:
    """Load a single extension with config."""
    try:
        con.execute(f"LOAD {spec.name}")
    except duckdb.Error:
        if spec.required:
            con.execute(f"INSTALL {spec.name}")
            con.execute(f"LOAD {spec.name}")
        else:
            return
    
    # Apply extension config
    for key, value in spec.config.items():
        con.execute(f"SET {key} = '{value}'")


# Standard UDFs for CodeIntel
def goid_hash(file_path: str, symbol_name: str, line: int) -> str:
    """Compute GOID hash for a symbol."""
    import hashlib
    content = f"{file_path}:{symbol_name}:{line}"
    return hashlib.blake2b(content.encode(), digest_size=16).hexdigest()


def complexity_score(loc: int, cyclomatic: int) -> float:
    """Compute normalized complexity score."""
    if loc <= 0:
        return 0.0
    return cyclomatic / (loc + 1)


STANDARD_UDFS: list[UDFSpec] = [
    UDFSpec("goid_hash", goid_hash),
    UDFSpec("complexity_score", complexity_score),
]


def register_standard_udfs(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Register standard CodeIntel UDFs."""
    registered: list[str] = []
    
    for udf in STANDARD_UDFS:
        params = udf.parameters
        ret = udf.return_type
        udf_type = "arrow" if udf.vectorized else "native"
        
        con.create_function(
            udf.name,
            udf.func,
            parameters=params,
            return_type=ret,
            type=udf_type,
            null_handling=udf.null_handling,
            exception_handling=udf.exception_handling,
        )
        registered.append(udf.name)
    
    return registered
```

**Impact**:
- Deterministic extension loading at startup
- Configuration-driven capability enablement
- Reusable SQL functions for common operations

---

### 7. Query Policy Enforcement via AST Transforms

**Current Pain Point**: Snapshot filtering (`repo`, `commit`) manually added to each query.

**Integrated Solution**: Automatic AST transformation for policy enforcement.

```python
# PROPOSED: storage/helpers/query_policy.py

from sqlglot import exp, parse_one
from dataclasses import dataclass

@dataclass(frozen=True)
class SnapshotRef:
    """Reference to a specific data snapshot."""
    repo: str
    commit: str


def inject_snapshot_filter(
    sg_expr: exp.Expression,
    snapshot: SnapshotRef,
    *,
    tables: set[str] | None = None,
) -> exp.Expression:
    """Inject snapshot filtering into all table references.
    
    Parameters
    ----------
    sg_expr
        SQLGlot expression to transform.
    snapshot
        Snapshot reference (repo, commit) to filter by.
    tables
        Optional set of table names to filter. If None, filter all tables.
    
    Returns
    -------
    exp.Expression
        Transformed expression with snapshot filters.
    
    Notes
    -----
    This is the key to centralized snapshot scoping. Instead of manually
    adding repo/commit filters everywhere, we transform the AST after
    Ibis compilation.
    """
    def transform_node(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Select):
            return node
        
        # Find all tables in FROM clause
        from_clause = node.find(exp.From)
        if not from_clause:
            return node
        
        # Build filter conditions for each table
        conditions: list[exp.Expression] = []
        for table in from_clause.find_all(exp.Table):
            table_name = table.this.name if hasattr(table.this, "name") else str(table.this)
            
            if tables and table_name not in tables:
                continue
            
            alias = table.alias or table_name
            conditions.extend([
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("repo"), table=exp.to_identifier(alias)),
                    expression=exp.Literal.string(snapshot.repo),
                ),
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("commit"), table=exp.to_identifier(alias)),
                    expression=exp.Literal.string(snapshot.commit),
                ),
            ])
        
        if not conditions:
            return node
        
        # Combine with existing WHERE clause
        combined = conditions[0]
        for cond in conditions[1:]:
            combined = exp.And(this=combined, expression=cond)
        
        existing_where = node.args.get("where")
        if existing_where:
            combined = exp.And(this=existing_where.this, expression=combined)
        
        return node.copy()
        # Note: Full implementation would modify the WHERE clause properly
    
    return sg_expr.transform(transform_node)


class QueryPolicy:
    """Policy-based query transformation and validation."""
    
    def __init__(self, snapshot: SnapshotRef | None = None):
        self.snapshot = snapshot
        self._snapshot_scoped_tables: set[str] = {
            "analytics.function_metrics",
            "analytics.test_catalog",
            "graph.call_graph_edges",
            "graph.import_graph_edges",
            # Add all snapshot-scoped tables
        }
    
    def enforce(self, sg_expr: exp.Expression) -> exp.Expression:
        """Apply all policy transformations."""
        result = sg_expr
        
        if self.snapshot:
            result = inject_snapshot_filter(
                result,
                self.snapshot,
                tables=self._snapshot_scoped_tables,
            )
        
        return result
    
    def validate_tables(self, sg_expr: exp.Expression) -> list[str]:
        """Validate that query only accesses allowed tables."""
        tables = {t.this.name for t in sg_expr.find_all(exp.Table)}
        # Could implement allowlist checking here
        return list(tables)
```

**Impact**:
- Centralized snapshot scoping
- Eliminates repetitive filter code
- Foundation for access control integration

---

### 8. Semantic Query Diffing for View Evolution

**Current Pain Point**: View changes compared as SQL strings, losing semantic understanding.

**Integrated Solution**: SQLGlot-based semantic diff.

```python
# PROPOSED: storage/views/diff.py

from sqlglot import diff, exp, parse_one
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
    """Represents a semantic change between two queries."""
    
    change_type: ChangeType
    node_type: str
    description: str
    old_sql: str | None = None
    new_sql: str | None = None


def diff_view_definitions(
    old_sql: str,
    new_sql: str,
    *,
    dialect: str = "duckdb",
) -> list[QueryChange]:
    """Compute semantic diff between two view definitions.
    
    Parameters
    ----------
    old_sql
        Previous view SQL definition.
    new_sql
        New view SQL definition.
    dialect
        SQL dialect for parsing.
    
    Returns
    -------
    list[QueryChange]
        List of semantic changes between versions.
    """
    old_expr = parse_one(old_sql, dialect=dialect)
    new_expr = parse_one(new_sql, dialect=dialect)
    
    raw_diff = diff(old_expr, new_expr)
    
    changes: list[QueryChange] = []
    for change in raw_diff:
        change_name = change.__class__.__name__.lower()
        
        if change_name == "keep":
            continue  # Skip unchanged nodes
        
        node_type = change.expression.__class__.__name__
        
        changes.append(QueryChange(
            change_type=ChangeType(change_name),
            node_type=node_type,
            description=_describe_change(change),
            old_sql=getattr(change, "source", change.expression).sql() if hasattr(change, "source") else None,
            new_sql=change.expression.sql(),
        ))
    
    return changes


def _describe_change(change) -> str:
    """Generate human-readable description of a change."""
    node = change.expression
    change_type = change.__class__.__name__
    
    if isinstance(node, exp.Column):
        return f"{change_type} column '{node.name}'"
    if isinstance(node, exp.Table):
        return f"{change_type} table reference '{node.name}'"
    if isinstance(node, (exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE)):
        return f"{change_type} filter condition"
    if isinstance(node, exp.Alias):
        return f"{change_type} alias '{node.alias}'"
    
    return f"{change_type} {node.__class__.__name__}"


def summarize_view_evolution(
    old_sql: str,
    new_sql: str,
) -> dict[str, list[str]]:
    """Summarize view changes for changelog/release notes.
    
    Returns
    -------
    dict
        Summary with keys: columns_added, columns_removed, filters_changed, etc.
    """
    changes = diff_view_definitions(old_sql, new_sql)
    
    summary: dict[str, list[str]] = {
        "columns_added": [],
        "columns_removed": [],
        "filters_changed": [],
        "tables_added": [],
        "tables_removed": [],
        "other_changes": [],
    }
    
    for change in changes:
        if change.node_type == "Column":
            if change.change_type == ChangeType.INSERT:
                summary["columns_added"].append(change.new_sql or "unknown")
            elif change.change_type == ChangeType.REMOVE:
                summary["columns_removed"].append(change.old_sql or "unknown")
        elif change.node_type == "Table":
            if change.change_type == ChangeType.INSERT:
                summary["tables_added"].append(change.new_sql or "unknown")
            elif change.change_type == ChangeType.REMOVE:
                summary["tables_removed"].append(change.old_sql or "unknown")
        elif change.node_type in ("EQ", "GT", "LT", "Where"):
            summary["filters_changed"].append(change.description)
        else:
            summary["other_changes"].append(change.description)
    
    return {k: v for k, v in summary.items() if v}  # Remove empty lists
```

**Impact**:
- Automated changelog generation for view updates
- Breaking change detection (column removal)
- Schema evolution tracking

---

### 9. Large IN-List Handling via Memtable

**Current Pain Point**: Large `IN (...)` lists can exceed placeholder limits or generate slow query plans.

**Integrated Solution**: Automatic memtable + semi-join pattern.

```python
# PROPOSED: Add to serving/semantic/query_builder.py

IN_LIST_THRESHOLD = 100  # Use memtable pattern above this size


def build_in_predicate(
    table: it.Table,
    col_expr: it.Value,
    values: list,
    *,
    con: ibis.backends.duckdb.Backend,
    instance_id: str,
) -> it.Table:
    """Build efficient IN predicate, using memtable for large lists.
    
    Parameters
    ----------
    table
        Base Ibis table expression.
    col_expr
        Column to filter.
    values
        List of values for IN clause.
    con
        DuckDB backend connection.
    instance_id
        Unique identifier for temp table naming.
    
    Returns
    -------
    it.Table
        Filtered table expression.
    
    Notes
    -----
    For small lists (<=100 values), uses standard IN clause.
    For large lists, creates a temporary memtable and uses semi-join.
    This avoids placeholder limits and generates better query plans.
    """
    if len(values) <= IN_LIST_THRESHOLD:
        # Standard IN clause for small lists
        return table.filter(col_expr.isin([ibis.literal(v) for v in values]))
    
    # Large list: use memtable + semi-join
    col_name = col_expr.get_name()
    temp_name = f"_in_list_{col_name}_{instance_id}"
    
    # Create memtable with deduplicated values
    unique_values = list(set(values))
    mt = ibis.memtable({col_name: unique_values})
    
    # Materialize as temp table
    con.create_table(temp_name, mt, temp=True, overwrite=True)
    temp_table = con.table(temp_name)
    
    # Semi-join (keeps only matching rows from left table)
    return table.semi_join(
        temp_table,
        col_expr == temp_table[col_name],
    )
```

**Impact**:
- No placeholder limits
- Better query plans (hash join vs OR chain)
- Automatic deduplication

---

### 10. Query Profiling Integration

**Current Pain Point**: Query performance is opaque; debugging slow queries requires manual EXPLAIN.

**Integrated Solution**: Integrated profiling with OpenTelemetry.

```python
# PROPOSED: storage/helpers/profiling.py

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
import time
import json

@dataclass
class QueryProfile:
    """Profile data for a single query execution."""
    
    sql: str
    duration_ms: float
    rows_returned: int
    plan: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""
    
    def to_otel_attributes(self) -> dict[str, Any]:
        """Convert to OpenTelemetry span attributes."""
        return {
            "db.statement": self.sql[:1000],  # Truncate for safety
            "db.operation.duration_ms": self.duration_ms,
            "db.operation.rows_returned": self.rows_returned,
            "codeintel.query.fingerprint": self.fingerprint,
        }


@contextmanager
def profile_query(
    con: "DuckDBPyConnection",
    *,
    capture_plan: bool = False,
):
    """Context manager for query profiling.
    
    Parameters
    ----------
    con
        DuckDB connection.
    capture_plan
        Whether to capture EXPLAIN ANALYZE output.
    
    Yields
    ------
    QueryProfile
        Profile object populated after query execution.
    
    Examples
    --------
    >>> with profile_query(con, capture_plan=True) as profile:
    ...     result = con.execute("SELECT * FROM large_table").fetchall()
    >>> print(f"Query took {profile.duration_ms}ms")
    """
    profile = QueryProfile(sql="", duration_ms=0, rows_returned=0)
    
    if capture_plan:
        con.execute("SET enable_profiling = 'json'")
        con.execute("SET profiling_mode = 'detailed'")
    
    start = time.perf_counter()
    
    try:
        yield profile
    finally:
        end = time.perf_counter()
        profile.duration_ms = (end - start) * 1000
        
        if capture_plan:
            try:
                plan_result = con.execute("SELECT * FROM duckdb_profiles()").fetchone()
                if plan_result:
                    profile.plan = json.loads(plan_result[0])
            except Exception:
                pass
            finally:
                con.execute("SET enable_profiling = 'none'")
```

**Impact**:
- Query performance visibility
- Integration with observability stack
- Debugging aid for slow queries

---

## Part III: Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
**Goal**: Establish core infrastructure for all subsequent enhancements.

| Task | Files | Effort | Dependencies |
|------|-------|--------|--------------|
| Unified Schema Pipeline | `storage/schema/`, `duckdb_policy_backend.py` | 3d | None |
| SQLGlot AST Access in IbisGateway | `ibis_adapter.py` | 2d | None |
| Extension Bootstrap at Connect | `gateway/factory.py` | 1d | None |
| Query Fingerprinting | `storage/helpers/` | 1d | AST Access |

**Success Criteria**:
- Schema created via Ibis round-trip matches existing DDL
- `gateway.ibis.to_sqlglot(expr)` returns valid AST
- All tests pass with new extension loading

### Phase 2: Query Intelligence (Weeks 3-4)
**Goal**: Type-safe parameterization and query caching.

| Task | Files | Effort | Dependencies |
|------|-------|--------|--------------|
| QueryTemplate System | `serving/semantic/templates.py` | 3d | Phase 1 |
| Migrate Semantic Queries | `serving/semantic/query_builder.py` | 2d | QueryTemplate |
| Query Fingerprint Caching | `serving/semantic/kernel.py` | 2d | Fingerprinting |
| Large IN-List Handling | `serving/semantic/query_builder.py` | 1d | None |

**Success Criteria**:
- Semantic queries use typed parameters
- Cache hit rate observable
- No placeholder limit errors

### Phase 3: Data Pipeline (Weeks 5-6)
**Goal**: Arrow-native data flow and complex types.

| Task | Files | Effort | Dependencies |
|------|-------|--------|--------------|
| Arrow Methods in IbisGateway | `ibis_adapter.py` | 2d | Phase 1 |
| Streaming HTTP Responses | `serving/http/` | 3d | Arrow Methods |
| Complex Type Schema Definitions | `storage/schema/` | 3d | Unified Schema |
| Standard UDF Registration | `gateway/extensions.py` | 2d | Extension Bootstrap |

**Success Criteria**:
- Large queries stream without OOM
- Complex types queryable via Ibis
- UDFs available in SQL

### Phase 4: Observability & Governance (Weeks 7-8)
**Goal**: Query analysis, lineage, and policy enforcement.

| Task | Files | Effort | Dependencies |
|------|-------|--------|--------------|
| Query Policy Enforcement | `storage/helpers/query_policy.py` | 3d | AST Access |
| Semantic View Diffing | `storage/views/diff.py` | 2d | None |
| Column Lineage Extraction | `ibis_adapter.py` | 2d | AST Access |
| Query Profiling Integration | `storage/helpers/profiling.py` | 2d | None |

**Success Criteria**:
- Snapshot filtering automatic via policy
- View changes generate semantic diff
- Lineage extractable for any query

---

## Part IV: Risk Assessment & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Ibis/SQLGlot API drift | Breaking changes | Medium | Version pin, golden tests |
| Schema migration complexity | Data loss | Low | v2 tables, gradual migration |
| Arrow memory pressure | OOM in serving | Medium | Streaming + chunk size tuning |
| UDF performance | Query slowdown | Low | Vectorized UDFs, selective use |
| Query transformation bugs | Wrong results | Medium | Comprehensive test suite |

### Testing Strategy

```python
# Golden SQL tests for API stability
def test_schema_roundtrip_produces_same_ddl():
    """Ensure Ibis schema round-trip matches manual DDL."""
    expected = "CREATE TABLE analytics.function_metrics (..."
    actual = UnifiedSchema.from_columns(...).to_sqlglot_create_table().sql()
    assert normalize_sql(actual) == normalize_sql(expected)

def test_ast_transform_preserves_semantics():
    """Ensure AST transforms don't change query results."""
    original = build_query(...)
    transformed = policy.enforce(gateway.to_sqlglot(original))
    
    # Both should return same results (modulo filtering)
    assert set(execute(original)) >= set(execute(transformed))

def test_parameter_binding_type_safety():
    """Ensure parameters are correctly typed."""
    template = QueryTemplate(...).param("score", "float64")
    
    with pytest.raises(TypeError):
        template.bind(score="not a float")
```

---

## Part V: Summary

### High-Value Opportunities (Implement First)

1. **Schema Round-Trips** — Immediate 50+ line reduction, type-safe DDL
2. **SQLGlot AST Access** — Foundation for all query intelligence
3. **Typed Parameterization** — Type safety + caching + injection prevention
4. **Arrow Data Pipeline** — Zero-copy, streaming, memory efficiency
5. **Extension Bootstrap** — Deterministic startup, fail-fast

### Medium-Value Opportunities (Phase 2-3)

6. **Query Policy Enforcement** — Centralized snapshot scoping
7. **Complex Types** — Better data modeling (schema migration required)
8. **Standard UDFs** — Reusable SQL functions
9. **Large IN-List Handling** — Production robustness
10. **Query Profiling** — Observability integration

### Future Opportunities (Phase 4+)

11. **Semantic View Diffing** — Automated changelogs
12. **Column Lineage** — Data governance, impact analysis
13. **Cross-Dialect SQL Migration** — External SQL integration
14. **Embedded SQL Executor (Testing)** — Faster unit tests

---

## References

- [Ibis v11 Release Notes](https://ibis-project.org/release_notes)
- [Ibis DuckDB Backend](https://ibis-project.org/backends/duckdb)
- [Ibis Schema Reference](https://ibis-project.org/reference/schemas)
- [SQLGlot Documentation](https://sqlglot.com/)
- [SQLGlot Lineage API](https://sqlglot.com/sqlglot/lineage.html)
- [DuckDB Python API](https://duckdb.org/docs/api/python/overview)
- [DuckDB Relational API](https://duckdb.org/docs/api/python/relational_api)
- [DuckDB Types API](https://duckdb.org/docs/api/python/types)
- [DuckDB Expression API](https://duckdb.org/docs/api/python/expression)
- CodeIntel AGENTS.md: Ibis 11 Patterns, Bulk Operations guidelines

---

*This document consolidates findings from the separate DuckDB/SQLGlot and Ibis/SQLGlot assessments into a unified enhancement strategy. Implementation should proceed in phases, with each phase building on the previous to minimize risk and maximize value.*

