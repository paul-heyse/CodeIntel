# Holistic Data Operations Enhancement Plan

> **Purpose**: Unified assessment of DuckDB, Ibis, and SQLGlot enhancement opportunities across the CodeIntel storage, build, and serving layers—identifying synergies, integration points, and a cohesive implementation roadmap.

**Generated**: 2024-12-17  
**Updated**: 2025-12-17 (repo review + implementation reality check)  
**Status**: Strategic Assessment (Revised)  
**Scope**: Storage + build + serving data operations

---

## Executive Summary

The CodeIntel data operations stack is built on three powerful technologies—**DuckDB** (execution), **Ibis** (query building), and **SQLGlot** (SQL generation/analysis)—but currently uses only a fraction of their combined capabilities. This document presents a **holistic enhancement strategy** that leverages synergies between these tools to achieve:

1. **40-60% reduction in boilerplate** through schema/DDL automation
2. **Type-safe query composition** via Ibis parameterization and a first-class SQLGlot AST hook
3. **Zero-copy interchange + true streaming exports** with Arrow end-to-end (no buffering)
4. **Operational observability** (build-time profiling artifacts + serving-side instrumentation)
5. **Deterministic capability bootstrap** via centralized extension/secret/session management

### Repo Review Addendum (2025-12-17): Immediate Priorities

The repo already implements several pieces of this plan (schema contracts, view dependency ordering, warehouse profiling),
and it also reveals a few high-impact gaps that should be pulled forward:

- **Exports buffer in memory**: “streaming” endpoints collect full `list(...)` payloads and use `fetchall()` in serving paths.
- **Upsert-from-expression fallback**: `IbisGateway` upsert from an Ibis expression materializes to pandas as a fallback.
- **DDL typing is hand-maintained**: `_column_type_to_sqlglot()` and `_build_column_def()` are a recurring maintenance tax.
- **Extension loading is split**: env-based installs at connect-time plus ad-hoc loads (FTS) rather than one policy surface.
- **Typed parameterization not standardized**: no `ibis.param()` usage; most dynamic inputs are `ibis.literal()` or `.isin()`.

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
| **DuckDB** | DDL/mutations via policy backend; warehouse profiling artifacts; selective extension load | True Arrow batch streaming in serving; prepared-statement patterns; unified extension/secret policy |
| **Ibis** | Query building + compile to SQL strings; snapshot filtering in warehouse reads | Typed scalar params (`ibis.param`); Arrow batch streaming (`to_pyarrow_batches`); param-aware SQLGlot compilation |
| **SQLGlot** | DDL generation; view dependency extraction | Canonicalization for fingerprints; lineage/diff integrated with build/asset tracking; perimeter validation for SQL strings |

### Key Integration Gaps

1. **Schema Management**: Manual type mapping in `_column_type_to_sqlglot()` when Ibis provides `Schema.to_sqlglot_column_defs()`
2. **Query Compilation**: String-first SQL generation (and parse round-trips) where a param-aware SQLGlot AST hook is possible
3. **Parameterization**: No standardized `ibis.param()` templates; dynamic inputs reduce cacheability and complicate typing
4. **Export/Streaming**: “Streaming” APIs buffer results (list + fetchall) instead of Arrow-batch streaming
5. **Write Paths**: Upsert-from-expression falls back to pandas materialization rather than staged temp tables + `INSERT..SELECT`
6. **Temp Object Hygiene**: No shared lifecycle management for staged memtables/temp tables (risk of leaks in long-lived sessions)
7. **Governance Hooks**: Diff/lineage/fingerprinting concepts exist but are not wired into build tracking or serving caching

---

## Part II: Enhancement Opportunities (Integrated)

### 1. Unified Schema Management Pipeline

**Current Pain Point**: The repo already has a canonical schema language (`TableSchema`), but DDL typing is still
hand-maintained (`_column_type_to_sqlglot()` / `_build_column_def()`), which creates drift risk and slows evolution.

**Revised Integrated Solution**: Keep `TableSchema` as the single source of truth and add a *single* bridge:
`TableSchema → ibis.Schema → SQLGlot ColumnDef` using `Schema.to_sqlglot_column_defs(dialect="duckdb")`.

This preserves the contract-first architecture (Pandera generation, schema drift checks, dataset registry) while
removing the need to maintain a parallel DDL type mapping.

```python
# PROPOSED: src/codeintel/storage/schema/ibis_roundtrip.py
import ibis
from sqlglot import exp

from codeintel.core.schemas.primitives import TableSchema
from codeintel.storage.constants import DUCKDB_DIALECT


def ibis_schema_from_table_schema(table: TableSchema) -> ibis.Schema:
    """Convert TableSchema to an ibis.Schema (including nullability)."""
    type_map: dict[str, str] = {
        "BOOLEAN": "boolean",
        "INTEGER": "int32",
        "BIGINT": "int64",
        "DOUBLE": "float64",
        "VARCHAR": "string",
        "JSON": "json",
        "TIMESTAMP": "timestamp",
        "TIMESTAMPTZ": "timestamp",
        "DECIMAL": "decimal",
        "DECIMAL(38,0)": "decimal(38,0)",
    }

    cols: dict[str, str] = {}
    for col in table.columns:
        dtype = type_map[col.type]
        cols[col.name] = f"!{dtype}" if not col.nullable else dtype
    return ibis.schema(cols)


def create_table_ast(table: TableSchema, *, if_not_exists: bool = True) -> exp.Create:
    """Build SQLGlot CREATE TABLE from TableSchema via Ibis round-trip."""
    ibis_schema = ibis_schema_from_table_schema(table)
    col_defs = ibis_schema.to_sqlglot_column_defs(dialect=DUCKDB_DIALECT)

    if table.primary_key:
        col_defs.append(
            exp.PrimaryKey(expressions=[exp.to_identifier(c) for c in table.primary_key])
        )

    return exp.Create(
        this=exp.Schema(
            this=exp.Table(this=exp.to_identifier(table.name), db=exp.to_identifier(table.schema)),
            expressions=col_defs,
        ),
        kind="TABLE",
        exists=if_not_exists,
    )
```

**Impact**
- Removes hand-maintained DDL type mapping (less drift, faster schema evolution).
- Keeps `TableSchema` canonical across build + storage + serving validation.
- Sets up a clean future path for complex types (requires extending `ColumnType` + Pandera mapping first).

**Files Affected**
- `src/codeintel/storage/duckdb_policy_backend.py` (swap manual ColumnDef building for round-trip helpers)
- `src/codeintel/storage/schema/` (add the round-trip bridge as a single reusable utility)

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
    
    def to_sqlglot(
        self,
        expr: it.Expr,
        *,
        params: dict[it.Scalar, object] | None = None,
        limit: int | None = None,
    ) -> exp.Expression:
        """Return SQLGlot AST for an Ibis expression.
        
        Parameters
        ----------
        expr
            Any Ibis expression (Table, Scalar, etc.)
        params
            Optional scalar parameter bindings. Required when `expr` contains `ibis.param(...)`.
        limit
            Optional limit to apply at compile time.
        
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
        return self.con.compiler.to_sqlglot(expr, params=params, limit=limit)
    
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
    
    def canonicalize(
        self,
        expr: it.Expr,
        *,
        params: dict[it.Scalar, object] | None = None,
        schema: dict[str, dict[str, str]] | None = None,
    ) -> exp.Expression:
        """Return canonical SQLGlot AST for fingerprinting/caching."""
        sg_expr = self.to_sqlglot(expr, params=params)
        return optimizer.optimize(
            sg_expr,
            dialect="duckdb",
            schema=schema or {},
            rules=(
                optimizer.qualify.qualify,
                optimizer.normalize.normalize,
            ),
        )
    
    def query_fingerprint(
        self,
        expr: it.Expr,
        *,
        params: dict[it.Scalar, object] | None = None,
        schema: dict[str, dict[str, str]] | None = None,
    ) -> str:
        """Generate stable fingerprint for query caching."""
        import hashlib
        canonical = self.canonicalize(expr, params=params, schema=schema)
        sql = canonical.sql(dialect="duckdb", pretty=False)
        return hashlib.sha256(sql.encode()).hexdigest()[:16]
```

**Impact**:
- Enables query-level caching in serving layer
- Powers automated lineage for documentation
- Foundation for semantic view diffing

**Integration Points**:
- `src/codeintel/serving/semantic/kernel.py`: Use fingerprints for response caching
- `src/codeintel/build/contracts.py`: Extract lineage for dependency tracking
- `src/codeintel/storage/views/`: Enable semantic diff for view evolution

---

### 3. Typed Parameterization System

**Current Pain Point**: Parameters injected via `ibis.literal()` or string formatting, losing type safety and caching potential.

**Integrated Solution**: Ibis `param()` based template system.

```python
# PROPOSED: src/codeintel/serving/semantic/templates.py

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
        limit: int | None = None,
        pretty: bool = True,
    ) -> str:
        """Compile to SQL with bound parameters."""
        param_map = self.bind(**bindings)
        return expr.compile(params=param_map, limit=limit, pretty=pretty)
    
    def execute(
        self,
        expr: it.Table,
        bindings: dict[str, Any],
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> Any:
        """Execute with bound parameters."""
        param_map = self.bind(**bindings)
        if limit is None and offset:
            msg = "offset requires limit to be set"
            raise ValueError(msg)
        limited = expr.limit(limit, offset=offset) if limit is not None else expr
        return limited.execute(params=param_map)


# Usage example in semantic query builder:
def build_semantic_query_template() -> tuple[QueryTemplate, it.Table]:
    """Build reusable semantic search template."""
    template = (
        QueryTemplate("semantic_search", "Vector similarity search")
        .param("repo", "string", required=True, description="Repository identifier")
        .param("min_score", "float64", default=0.5, description="Minimum similarity score")
    )
    
    # Build a reusable expression template using scalar parameters.
    # Apply pagination (limit/offset) out-of-band at compile/execute time to keep the template stable.
    t = con.table("vectors.embeddings")
    expr = (
        t.filter(t.repo == template.get_param_expr("repo"))
        .filter(t.score >= template.get_param_expr("min_score"))
    )
    
    return template, expr
```

**Impact**:
- Type-safe parameter binding
- SQL injection prevention (values never interpolated)
- Enables stable query templates at the Ibis IR level; input values flow via `params` for hashing/caching

**Integration Points**:
- `src/codeintel/serving/semantic/query_builder.py`: Replace literal injection
- `src/codeintel/build/exports/exprs.py`: Template-based export queries

---

### 4. Arrow-Native Data Pipeline + True Streaming Exports

**Current Pain Point**: “Streaming” APIs frequently buffer results (e.g., `list(...)` materialization, `fetchall()`), and
export implementations often round-trip through pandas. This risks OOM on large views and delays time-to-first-byte.

**Integrated Solution**: Arrow-first data flow with batch streaming end-to-end:
- Use `to_pyarrow_batches()` for constant-memory iteration.
- Build NDJSON streaming responses without collecting full result sets; Arrow/Parquet via streaming writer or spool-to-disk.
- Prefer Arrow registration/replacement scans for bulk writes (avoid Python tuple loops when large).

```python
# PROPOSED: Add to IbisGateway

import json
import pyarrow as pa
import pyarrow.csv as pa_csv

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
    ) -> pa.RecordBatchReader:
        """Stream results as Arrow batches for memory-efficient processing.
        
        Parameters
        ----------
        expr
            Ibis table expression to execute.
        chunk_size
            Number of rows per batch.
        
        Yields
        ------
        pa.RecordBatchReader
            Arrow RecordBatchReader for streaming processing.
        
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
    *,
    chunk_size: int = 10_000,
    output_format: str = "jsonl",
) -> AsyncIterator[bytes]:
    """Stream query results as HTTP response chunks.
    
    Parameters
    ----------
    expr
        Ibis expression to stream.
    chunk_size
        Rows per batch.
    output_format
        Output format: 'jsonl' or 'csv'. (Arrow/Parquet are handled separately.)
    
    Yields
    ------
    bytes
        Encoded response chunks.
    """
    reader = expr.to_pyarrow_batches(chunk_size=chunk_size)

    for batch in reader:
        if output_format == "jsonl":
            # Avoid buffering the full resultset; emit per batch.
            for row in batch.to_pylist():
                yield (json.dumps(row, default=str) + "\n").encode("utf-8")
        elif output_format == "csv":
            # CSV is row-oriented; keep the batch size small to limit memory.
            table = pa.Table.from_batches([batch])
            sink = pa.BufferOutputStream()
            pa_csv.write_csv(table, sink)
            yield sink.getvalue().to_pybytes()
```

**Impact**:
- Constant memory for large results
- Faster time-to-first-byte
- Native integration with ML frameworks (PyTorch, TensorFlow)

**Integration Points**:
- `src/codeintel/serving/http/routes/v1/export.py`: Remove list buffering; stream generator output directly
- `src/codeintel/serving/semantic/kernel.py`: Avoid `fetchall()`; expose Arrow-batch export primitives
- `src/codeintel/build/hamilton/io/`: Optional Arrow-based loaders/exports where pandas becomes a bottleneck

---

### 5. DuckDB Complex Types for Nested Data (Defer / v2 Experiment)

**Current Pain Point**: Flattened schemas lose semantic structure; some fields are stored as JSON strings and parsed at runtime.

**Repo Review Adjustment**: Complex types are high-upside but *high-blast-radius* today because the schema contract language
(`ColumnType` + Pandera mapping) currently models a small scalar set. Complex types should be introduced only after:

1. DDL round-trips are stabilized (Opportunity 1),
2. streaming exports are fixed (Opportunity 4), and
3. the contract language is extended end-to-end (schema hashing + Pandera + row bindings).

**Experimental Path**: Pilot 1–2 v2 tables (side-by-side) using DuckDB Types API helpers.

```python
# PROPOSED (experimental): complex types via DuckDB Types API
import duckdb

# STRUCT
symbol_location_type = duckdb.struct_type(
    {
        "file_path": duckdb.sqltypes.VARCHAR,
        "line_start": duckdb.sqltypes.INTEGER,
        "line_end": duckdb.sqltypes.INTEGER,
        "column_start": duckdb.sqltypes.INTEGER,
        "column_end": duckdb.sqltypes.INTEGER,
    }
)

# LIST<STRUCT>
function_param_type = duckdb.struct_type(
    {
        "name": duckdb.sqltypes.VARCHAR,
        "type": duckdb.sqltypes.VARCHAR,
        "default": duckdb.sqltypes.VARCHAR,
        "position": duckdb.sqltypes.INTEGER,
    }
)
function_params_type = duckdb.list_type(function_param_type)

# MAP<VARCHAR, VARCHAR>
edge_metadata_type = duckdb.map_type(duckdb.sqltypes.VARCHAR, duckdb.sqltypes.VARCHAR)
```

**Impact (when ready)**
- Better modeling of code-intelligence entities (nested/structured data)
- Fewer joins and less JSON parsing in query paths
- Better alignment with Arrow-native interchange

**Migration Consideration**: Treat as v2 schemas/tables; do not block core operational fixes.

---

### 6. Centralized Session, Extension, and Secret Management (Revise)

**Current Pain Point**: Extension and session initialization is spread across env-driven connect-time behavior and
feature-local code paths (e.g., FTS loads its own extension). This creates inconsistent behavior between build vs serving,
and it risks performing `INSTALL` in environments where network access or write permissions are undesirable.

**Repo Review Adjustment**: The repo already has the correct seams:
- env-based extension list loading at connect-time, and
- a session wrapper (`DuckDBSession`) that can own init SQL.

The enhancement should consolidate on *one* lifecycle owner: the session/connection layer.

**Revised Integrated Solution**: Promote a single “capabilities bootstrap” surface in the session layer:

- **Serving/read-only**: `LOAD` only, fail-fast if missing (no implicit installs).
- **Build/write**: allow `INSTALL` + `LOAD` when explicitly enabled.
- **Secrets/init SQL**: use a single init pipeline (e.g., `CODEINTEL_DUCKDB_INIT_SQL`) for `SET/PRAGMA/CREATE SECRET/ATTACH`.

```python
# PROPOSED: storage/backend/duckdb_session.py (capability bootstrap sketch)
#
# Policy:
# - allow_install=False in serving (read-only)
# - allow_install=True only in build / explicit admin paths

def bootstrap_capabilities(
    con: "DuckDBPyConnection",
    *,
    extensions: list[str],
    allow_install: bool,
) -> list[str]:
    loaded: list[str] = []
    for name in extensions:
        try:
            con.execute(f"LOAD {name}")
        except Exception:
            if not allow_install:
                raise
            con.execute(f"INSTALL {name}")
            con.execute(f"LOAD {name}")
        loaded.append(name)
    return loaded
```

**UDF Registry (Optional)**: Only introduce a standard UDF catalog if there is repeated SQL-level demand. Today, the repo
has near-zero registered UDF usage, so start with extension/session unification first.

**Impact**
- Deterministic “capabilities on startup” for build + serving
- Clear `LOAD` vs `INSTALL` policy (safe defaults in serving)
- One place to wire secrets/session configuration for cloud access

---

### 7. Snapshot Scoping & Query Governance (Ibis-first; AST optional)

**Current Pain Point**: Snapshot scoping (`repo`, `commit`) is applied inconsistently: some paths rely on the warehouse
helper, while many repository/query paths still manually add filters.

**Repo Review Adjustment**: The repo already implements an Ibis-layer snapshot filter in the warehouse read path
(`Warehouse.read(...)` only scopes when both `repo` and `commit` exist). This is the safest place to start because it
preserves Ibis typing and avoids SQL rewriting edge cases.

**Revised Integrated Solution**

1. **Ibis-first scoping (near-term)**  
   Provide a snapshot-aware table accessor for repository code so most queries no longer repeat `repo/commit` filters.
   (E.g., BaseRepository can route through `Warehouse.read(...)` or a shared `scoped_table(...)` helper.)

2. **SQLGlot governance hooks (later; boundary-only)**  
   Use SQLGlot AST inspection/transforms only at boundaries where SQL strings exist (e.g., `Table.sql(...)`,
   `Backend.sql(...)`, or any “raw SQL” ingress), primarily for:
   - **table allowlists** (what does this SQL touch?)
   - **schema/column minimization** (what does this query select?)
   - **safety policies** (disallow writes, forbid unbounded scans in serving, etc.)

```python
# PROPOSED: minimal perimeter validation using SQLGlot (allowlist example)
from sqlglot import exp, parse_one

def referenced_tables(sql: str, *, dialect: str = "duckdb") -> set[str]:
    root = parse_one(sql, dialect=dialect)
    return {f\"{t.db}.{t.name}\" if t.db else t.name for t in root.find_all(exp.Table)}
```

**Impact**
- Removes repetitive snapshot filter boilerplate in repository/query code
- Preserves Ibis typing and avoids brittle AST rewrite semantics
- Still enables SQLGlot-based governance where SQL strings are unavoidable

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

**Repo Review Adjustment**: Treat semantic SQL diffs as a *build artifact*, not a standalone utility. The repo already
compiles view SQL for materialization ordering; persist compiled SQL per build/run and compute semantic diffs between
successive versions to power changelogs and breaking-change detection.

**Impact**:
- Automated changelog generation for view updates
- Breaking change detection (column removal)
- Schema/view evolution tracking integrated with asset/build tracking

---

### 9. Large IN-List Handling via Memtable

**Current Pain Point**: Large `IN (...)` lists can exceed placeholder limits or generate slow query plans.

**Repo Review Adjustment**: This should be a reusable primitive with explicit lifecycle management. Creating temp tables
inside “pure” query builder code makes cleanup difficult and risks leaking temp objects in long-lived serving processes.

**Integrated Solution**: A 2–3 tier strategy with staging at execution time:

1. **Small lists**: use `.isin(...)`
2. **Large lists**: stage values into a temp table and use a semi-join
3. **Raw SQL paths (optional)**: prefer `col = ANY(?)` where DuckDB supports array parameters (reduces temp tables)

```python
# PROPOSED: serving/semantic/in_list.py (execution-time staging)

from contextlib import contextmanager
from collections.abc import Iterator

import ibis
import ibis.expr.types as it

IN_LIST_THRESHOLD = 100  # Use memtable pattern above this size


@contextmanager
def staged_values_table(
    con: ibis.backends.duckdb.Backend,
    *,
    column_name: str,
    values: list[object],
    instance_id: str,
) -> Iterator[it.Table]:
    """Stage list values into a temp table and ensure cleanup."""
    temp_name = f"__in_{column_name}_{instance_id}"
    mt = ibis.memtable({column_name: values})
    con.create_table(temp_name, mt, temp=True, overwrite=True)
    try:
        yield con.table(temp_name)
    finally:
        con.drop_table(temp_name, force=True)


def apply_in_filter(
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
    Staging must happen at execution time (so the temp table exists when executed) and must be cleaned up.
    """
    if len(values) <= IN_LIST_THRESHOLD:
        # Standard IN clause for small lists
        return table.filter(col_expr.isin([ibis.literal(v) for v in values]))

    col_name = col_expr.get_name()
    unique_values = list(dict.fromkeys(values))  # stable dedupe

    with staged_values_table(
        con,
        column_name=col_name,
        values=unique_values,
        instance_id=instance_id,
    ) as staged:
        return table.semi_join(staged, col_expr == staged[col_name])
```

**Impact**:
- No placeholder limits
- Better query plans (hash join vs OR chain)
- Automatic deduplication + temp table hygiene

---

### 10. Query Profiling & Observability (Build + Serving)

**Current Pain Point**: Query performance instrumentation is inconsistent across layers. Build/materialization may capture
profiling artifacts, while serving/query paths often rely on ad-hoc `EXPLAIN` and buffered result extraction.

**Repo Review Adjustment**: The repo already supports DuckDB JSON profiling artifacts in the warehouse materialization
path. The enhancement should extend and standardize this rather than introducing a parallel profiling subsystem.

**Integrated Solution**

1. **Build-time (warehouse)**  
   Keep DuckDB JSON profiling artifacts as the canonical “deep profile” output (enabled via a profiling output dir).

2. **Serving-time (semantic/query/export)**  
   Add lightweight instrumentation by default:
   - duration (ms)
   - row counts (when cheaply available)
   - query hash (inputs) + schema hash (manifest) for correlation

   Gate heavier capture (EXPLAIN ANALYZE / DuckDB profiles) behind configuration and/or slow-query thresholds.

```python
# PROPOSED: lightweight query telemetry (serving)

from dataclasses import dataclass
from time import perf_counter

@dataclass(frozen=True)
class QueryTelemetry:
    sql: str
    duration_ms: float
    query_hash: str | None = None
    schema_hash: str | None = None
    rows_returned: int | None = None


def timed_execute(con: "DuckDBPyConnection", *, sql: str, params: list[object] | None = None):
    start = perf_counter()
    rel = con.execute(sql, params) if params is not None else con.execute(sql)
    duration_ms = (perf_counter() - start) * 1000
    return rel, QueryTelemetry(sql=sql, duration_ms=duration_ms)
```

**Impact**
- Consistent observability signals across build + serving
- Faster diagnosis of slow queries and export bottlenecks
- Clear separation between “cheap telemetry” and “expensive profiling”

---

### 11. Eliminate Pandas Fallbacks in Write Paths (Upsert + Bulk)

**Current Pain Point**: Some write paths are “safe but expensive”:
- upsert-from-expression can fall back to `expr.to_pandas()` materialization,
- large DataFrame writes may normalize into Python tuples (slow, high overhead),
- temp object staging (for joins or writes) is not consistently cleaned up.

**Integrated Solution**: Add two fast lanes while keeping existing small-write behavior:

1. **Upsert from expression without pandas**  
   Stage the expression result into DuckDB (temp table or registered Arrow) and run:
   `INSERT ... ON CONFLICT ... SELECT ...` (same semantics, no full DataFrame materialization).

2. **Bulk writes via Arrow/replacement scans**  
   For large DataFrames/Arrow tables, prefer `register(...)` + `INSERT ... SELECT` over Python tuple loops.

**Impact**
- Removes a major OOM risk in write-heavy pipelines
- Improves throughput for large inserts/upserts
- Aligns with Arrow-first pipeline goals

---

### 12. Contract-Driven Filter Typing and Query Validation (Serving)

**Current Pain Point**: Serving validates *identifiers* (allowed columns) but not *operator compatibility*.
This allows invalid queries to reach DuckDB (e.g., string operators on numeric columns) and complicates caching and
client ergonomics.

**Integrated Solution**: Use the schema inventory (column name → type) to validate filter specs:

- string ops (`contains`, `startswith`, `ilike`) → `VARCHAR`
- numeric comparisons (`lt/lte/gt/gte`) → numeric/temporal only
- `in` → element type check + list size strategy (Opportunity 9)
- enforce nullable semantics where needed (e.g., optional filters)

**Impact**
- Fewer runtime query errors
- Better client feedback (fast fail with clear messages)
- Stronger guarantees for query hashing/fingerprinting in exports and MCP responses

---

### 13. Compiler Upgrade Gates (SQLGlot/Ibis)

**Current Pain Point**: SQLGlot minor releases can be backwards-incompatible, and Ibis uses SQLGlot as its compilation
substrate. This means dependency bumps can silently change compiled SQL, query plans, and even semantics.

**Integrated Solution**: Treat SQLGlot/Ibis bumps like compiler upgrades:

- pin versions deliberately (and bump with a small “compiler upgrade” PR),
- add golden SQL snapshots for representative expressions,
- add AST-shape checks and/or SQLGlot semantic diffs to explain changes,
- include execution validation on DuckDB for critical query paths.

**Impact**
- Predictable upgrades with clear diffs
- Lower risk of “silent” query behavior changes

---

## Part III: Implementation Roadmap

This roadmap is ordered by “repo reality” impact: first remove known OOM/buffering paths and pandas fallbacks, then
standardize query inputs and DDL generation, and finally add governance/observability and upgrade gates.

### Phase 1: Operational Hardening (Weeks 1-2)
**Goal**: Make serving exports truly streaming and remove pandas/materialization fallbacks in write paths.

| Task | Files | Effort | Dependencies |
|------|-------|--------|--------------|
| Stream `export_rows` without `fetchall()` | `src/codeintel/serving/semantic/kernel.py` | 1-2d | None |
| NDJSON export without `list(...)` buffering | `src/codeintel/serving/http/routes/v1/export.py`, `src/codeintel/serving/http/streaming.py` | 1d | export_rows streaming |
| Arrow/Parquet exports without row lists (spool-to-disk or streaming writer) | `src/codeintel/serving/http/routes/v1/export.py` | 2-3d | export_rows streaming |
| Remove `list(kernel.export_rows(...))` buffering in MCP | `src/codeintel/serving/mcp/app.py`, `src/codeintel/serving/mcp/resources.py` | 0.5-1d | export_rows streaming |
| Upsert-from-expression without pandas fallback | `src/codeintel/storage/ibis_adapter.py` | 2-3d | None |
| Bulk writes via Arrow/replacement scans | `src/codeintel/storage/ibis_adapter.py` | 1-2d | Upsert fast lane |
| Centralize `LOAD` vs `INSTALL` policy (+ init SQL) | `src/codeintel/storage/gateway/connection.py`, `src/codeintel/storage/backend/duckdb_session.py` | 1-2d | None |

**Success Criteria**:
- NDJSON export starts streaming immediately and does not materialize `rows: list[...]`.
- `SemanticQueryKernel.export_rows()` does not call `fetchall()` and maintains constant memory for large exports.
- Upsert-from-expression does not materialize to pandas for large inputs and preserves current semantics.
- Serving environments do not perform `INSTALL` implicitly (explicit opt-in only).

### Phase 2: Safe Query Inputs + Schema Automation (Weeks 3-4)
**Goal**: Make query inputs typed/validated and reduce DDL drift/boilerplate.

| Task | Files | Effort | Dependencies |
|------|-------|--------|--------------|
| TableSchema → Ibis → SQLGlot DDL bridge | `src/codeintel/storage/duckdb_policy_backend.py`, `src/codeintel/storage/schema/ibis_roundtrip.py` (new) | 2-3d | None |
| Param-aware SQLGlot AST hook (`to_sqlglot`) | `src/codeintel/storage/ibis_adapter.py` | 1-2d | None |
| Typed QueryTemplate + param binding | `src/codeintel/serving/semantic/query_builder.py`, `src/codeintel/serving/semantic/templates.py` (new) | 2-3d | None |
| Contract-driven filter typing/validation | `src/codeintel/serving/semantic/models.py`, `src/codeintel/serving/semantic/query_builder.py` | 1-2d | QueryTemplate |
| Large IN-list staging with cleanup | `src/codeintel/serving/semantic/in_list.py` (new), `src/codeintel/serving/semantic/query_builder.py` | 1-2d | Filter validation |

**Success Criteria**:
- Dynamic inputs flow via validated specs + `params` (no SQL string interpolation; fewer ad-hoc `ibis.literal(...)` paths).
- DDL generation no longer depends on hand-maintained `_column_type_to_sqlglot()` mappings.
- Large IN-list queries succeed without placeholder limits and leave no temp tables behind.

### Phase 3: Caching, Telemetry, and Build Artifacts (Weeks 5-6)
**Goal**: Make query execution observable and make upgrades predictable.

| Task | Files | Effort | Dependencies |
|------|-------|--------|--------------|
| Query fingerprinting integrated in serving | `src/codeintel/serving/semantic/kernel.py`, `src/codeintel/serving/mcp/response_models.py` | 1-2d | AST hook |
| Lightweight serving telemetry (duration/rows/query hash) | `src/codeintel/serving/semantic/kernel.py` | 1d | None |
| Semantic SQL diffs as build artifacts | `src/codeintel/storage/views/diff.py` (new), `src/codeintel/storage/views/dependencies.py` | 2-3d | None |
| Compiler upgrade gates (golden SQL + execution validation) | `tests/` (new) | 2-3d | Phase 2 |

**Success Criteria**:
- Fingerprints are stable across requests and incorporate schema hash + param bindings.
- Serving emits cheap latency/rowcount telemetry by default; deep profiling remains opt-in.
- View SQL changes produce a semantic diff artifact per build/run.
- SQLGlot/Ibis upgrades are gated by golden snapshots + execution checks.

### Phase 4: Governance + Deferred Experiments (Weeks 7-8+)
**Goal**: Strengthen governance at SQL boundaries and explore complex types only after contracts support them.

| Task | Files | Effort | Dependencies |
|------|-------|--------|--------------|
| Snapshot-aware accessors for repository queries | `src/codeintel/storage/warehouse.py`, repository base classes | 2-3d | Phase 2 |
| SQL perimeter validation for raw SQL ingress | `src/codeintel/storage/queries/safe.py` | 1-2d | AST hook |
| DuckDB complex types pilot (v2 tables only) | `src/codeintel/core/schemas/`, `src/codeintel/storage/duckdb_policy_backend.py` | 1-2w | Contract extensions |

**Success Criteria**:
- Most query paths stop hand-applying `repo/commit` filters.
- Raw SQL ingress is validated against allowlists/safety policies.
- Complex types are introduced only in isolated v2 tables with full contract support.

---

## Part IV: Risk Assessment & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Serving export buffering regressions | OOM / latency | Medium | Add streaming-focused tests; avoid `list(...)`/`fetchall()` in export paths |
| Client disconnect/backpressure | Resource leaks | Medium | Cancellation-aware streaming; context-managed cleanup for temp tables/registrations |
| `LOAD` vs `INSTALL` misconfiguration | Missing features in prod | Medium | Explicit config flags; startup health check; serving defaults to `LOAD` only |
| Upsert fast-lane semantic drift | Incorrect writes | Low-Med | Golden write-path tests comparing old/new; staged rollout; keep legacy path for small writes |
| Cache correctness | Wrong results served | Medium | Fingerprint includes schema hash + params + query shape; disable cache for non-deterministic queries |
| SQLGlot/Ibis compiler drift | Breaking changes | Medium | Compiler upgrade gates (golden SQL + execution validation) |
| Complex types blast radius | Data/model incompatibility | Low | Defer; v2 tables only; extend contracts + validators first |

### Testing Strategy

- **Streaming/export tests**
  - Ensure HTTP endpoints return `StreamingResponse` backed by generators (no pre-materialization).
  - Cover NDJSON, Arrow IPC, and Parquet paths; assert no `list(kernel.export_rows(...))`.
- **Write-path tests**
  - Upsert-from-expression executes without pandas materialization for large inputs and preserves conflict semantics.
  - Bulk Arrow writes use replacement scans and maintain schema/ordering guarantees.
- **DDL/schema tests**
  - TableSchema → Ibis schema → SQLGlot DDL round-trips produce stable, parseable SQL for DuckDB.
- **Filter validation tests**
  - Contract-driven operator/type compatibility errors are raised before execution.
  - IN-list staging creates and cleans temp tables even on exceptions.
- **Compiler upgrade gates**
  - Golden SQL snapshots for representative expressions + view compiles.
  - Execution validation in DuckDB for critical query paths to catch semantic changes.

---

## Part V: Summary

### Implement First (Repo-Driven Priorities)

1. **True Streaming Exports (NDJSON/Arrow)** — remove `list(...)` + `fetchall()` buffering (Opportunity 4)
2. **Eliminate Pandas Write Fallbacks** — upsert-from-expression + bulk writes via Arrow/replacement scans (Opportunity 11)
3. **Centralize Session Bootstrap** — deterministic `LOAD`/`INSTALL` + init SQL + secrets policy (Opportunity 6)
4. **Typed Query Inputs** — `ibis.param` templates + contract-driven filter validation (Opportunities 3, 12, 9)
5. **Schema Round-Trips for DDL** — keep `TableSchema` canonical; generate ColumnDefs via Ibis (Opportunity 1)

### Implement Next (Query Intelligence + Build Artifacts)

6. **SQLGlot AST Access + Fingerprints** — param-aware AST hook enabling caching/lineage (Opportunity 2)
7. **Serving Telemetry + Profiling Alignment** — cheap metrics by default; reuse warehouse profiling artifacts (Opportunity 10)
8. **Semantic View Diff Artifacts** — build-time compiled SQL diffs for changelogs/breaking changes (Opportunity 8)
9. **Compiler Upgrade Gates** — golden SQL + execution validation for SQLGlot/Ibis bumps (Opportunity 13)
10. **Snapshot Scoping Governance** — Ibis-first scoping + boundary validation for raw SQL (Opportunity 7)

### Defer / Experiment

11. **DuckDB Complex Types** — only after contract language + validators support them end-to-end (Opportunity 5)

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
