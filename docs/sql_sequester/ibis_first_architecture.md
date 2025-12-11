# Ibis-First Database Architecture

## Executive Summary

This document specifies the architecture for CodeIntel's **Ibis-First Database Layer**: a comprehensive abstraction that makes Ibis the sole programmatic interface for all DuckDB interactions. The goal is to eliminate raw SQL strings from application code entirely, replacing them with a typed, composable, and portable query algebra that seamlessly translates between Python data structures and database operations.

The architecture rests on three pillars:

1. **Dataset Contracts as Source of Truth** — All table structures, column types, constraints, and validation rules derive from declarative Python contracts, not hand-written DDL.

2. **Ibis as the Universal Query Language** — Every read, filter, join, aggregation, and transformation uses the Ibis expression API; SQL becomes an internal implementation detail.

3. **Policy Backend for Non-Query Operations** — DDL, mutations, and escape hatches are centralized in a single module that uses SQLGlot to generate SQL, isolating the "messy" database operations from all other code.

This design eliminates the cognitive overhead of context-switching between Python and SQL, prevents SQL injection vulnerabilities, enables cross-database portability, and creates a foundation where **data models drive schemas** rather than schemas constraining data models.

---

## Table of Contents

1. [Design Philosophy & Goals](#1-design-philosophy--goals)
2. [Architectural Layers](#2-architectural-layers)
3. [The Gateway & Ibis Adapter](#3-the-gateway--ibis-adapter)
4. [Dataset Contracts: The Schema Source of Truth](#4-dataset-contracts-the-schema-source-of-truth)
5. [The DuckDB Policy Backend](#5-the-duckdb-policy-backend)
6. [Ibis Expression Patterns for Queries](#6-ibis-expression-patterns-for-queries)
7. [UDF Registry: Extending Ibis for DuckDB-Specific Functions](#7-udf-registry-extending-ibis-for-duckdb-specific-functions)
8. [View Management via Ibis IR](#8-view-management-via-ibis-ir)
9. [Pandera Integration for Runtime Validation](#9-pandera-integration-for-runtime-validation)
10. [Migration Strategy](#10-migration-strategy)
11. [Configuration & Environment](#11-configuration--environment)
12. [Ibis 11 Compatibility Guidelines](#12-ibis-11-compatibility-guidelines)
13. [Anti-Patterns & What to Avoid](#13-anti-patterns--what-to-avoid)
14. [Appendix: SQLGlot Expression Reference](#appendix-sqlglot-expression-reference)

---

## 1. Design Philosophy & Goals

### 1.1 The Problem with Raw SQL in Application Code

The current codebase exhibits a common anti-pattern: SQL strings scattered throughout analytics, ingestion, and serving modules. This creates several problems:

**Cognitive overhead.** Developers must mentally switch between Python semantics and SQL semantics, tracking identifier quoting, parameter placeholders, join syntax, and dialect-specific quirks.

**Type safety gaps.** Raw SQL strings bypass Python's type system entirely. A typo in a column name becomes a runtime error, often discovered only after data flows through multiple pipeline stages.

**Testing complexity.** SQL strings resist unit testing. You either mock the entire database connection (which tells you nothing about query correctness) or spin up a real database (which is slow and fragile).

**Maintenance burden.** When a table schema changes, you must hunt down every SQL string that references the affected columns. There is no compiler to help you.

**Security surface.** While parameterized queries mitigate SQL injection, the presence of SQL strings anywhere in the codebase creates opportunities for mistakes, especially when composing queries dynamically.

### 1.2 The Ibis-First Vision

The Ibis-First architecture addresses these problems by making **Ibis expressions the canonical representation of all database queries**:

- **Type-checked at construction time.** Ibis expressions are Python objects with known types. Column names are attributes; filters are boolean expressions; aggregations are method calls. Static analysis tools (pyright, pyrefly) can catch many errors before execution.

- **Backend-agnostic.** The same Ibis expression can target DuckDB today and Snowflake tomorrow. While we currently use only DuckDB, this portability is insurance against future architecture changes.

- **Composable.** Ibis expressions are immutable and composable. You can build query fragments as reusable functions, combine them, and only execute at the boundary.

- **Inspectable.** You can always call `expr.to_sql(dialect="duckdb")` to see exactly what SQL will be generated. This aids debugging without requiring SQL authorship.

### 1.3 The Data Model Drives the Schema

A key architectural goal is **inverting the traditional relationship between schemas and data models**:

| Traditional Approach | Ibis-First Approach |
|---------------------|---------------------|
| Write DDL manually | Define `TableSchema` in Python |
| Generate type stubs from DDL | Generate DDL from `TableSchema` |
| SQL strings reference column names | Column names come from contracts |
| Schema changes require DDL + code updates | Schema changes propagate automatically |

This inversion means your Python data models are the **single source of truth**. The database schema is a *derived artifact*, regenerable at any time from the contracts.

### 1.4 Boundaries and Escape Hatches

No abstraction is complete. The architecture explicitly designates where SQL is permitted:

- **Never in application code** (analytics, ingestion, serving, CLI, graphs)
- **Only in the storage layer**, specifically:
  - `DuckDBPolicyBackend` for DDL and mutations
  - `ibis_builtins.py` for UDF wrappers
  - View builders for Ibis-to-view registration
  - Tests (where raw SQL may verify behavior)

This boundary is **enforced by architecture tests** that scan the codebase and fail if SQL constructs appear outside approved modules.

---

## 2. Architectural Layers

The Ibis-First architecture organizes into four distinct layers, each with clear responsibilities and dependencies:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                                │
│  (analytics, ingestion, serving, CLI, graphs)                          │
│                                                                         │
│  • Consumes repositories and gateways                                  │
│  • Builds Ibis expressions for queries                                 │
│  • Never imports duckdb or constructs SQL                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        REPOSITORY LAYER                                 │
│  (storage/repositories/*.py)                                           │
│                                                                         │
│  • Provides domain-oriented query methods                              │
│  • Implements queries using Ibis expressions                           │
│  • Returns typed Python objects (dataclasses, TypedDicts, DataFrames)  │
│  • May call Pandera validation before returning                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        GATEWAY LAYER                                    │
│  (storage/gateway/*.py, storage/ibis_adapter.py)                       │
│                                                                         │
│  • StorageGateway protocol defines the contract                        │
│  • IbisGateway wraps DuckDB connection with Ibis backend               │
│  • Provides table access: gateway.ibis.table("schema.name")            │
│  • Exposes raw Ibis backend for advanced operations                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        POLICY LAYER                                     │
│  (storage/duckdb_policy_backend.py, storage/schema/ddl.py)             │
│                                                                         │
│  • DuckDBPolicyBackend owns all non-Ibis SQL                           │
│  • Uses SQLGlot to generate DDL, DELETE, MERGE statements              │
│  • Executes via Ibis backend's raw_sql                                 │
│  • Only module permitted to construct SQL strings                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        CONTRACT LAYER                                   │
│  (config/datasets/*.py, storage/views/ibis_registry.py)                │
│                                                                         │
│  • TableSchema defines columns, types, constraints, indexes            │
│  • DatasetContract bundles schema with metadata                        │
│  • VIEW_BUILDERS maps view names to Ibis expression builders           │
│  • Pandera schemas derive from TableSchema                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Information Flow

**Downward (configuration):**
1. `TableSchema` objects define the logical structure of tables
2. `DuckDBPolicyBackend` generates DDL from these schemas via SQLGlot
3. `IbisGateway` provides typed access to the resulting tables
4. Repositories expose domain queries using Ibis expressions
5. Application code calls repository methods and works with Python objects

**Upward (data):**
1. Application code produces Python objects (dataclasses, dicts, DataFrames)
2. Repositories may validate these against Pandera schemas
3. Mutations flow through `DuckDBPolicyBackend` methods
4. Data lands in DuckDB tables matching the `TableSchema` contracts

### 2.2 Dependency Rules

To maintain architectural integrity, the following import rules apply:

| Module | Can Import | Cannot Import |
|--------|------------|---------------|
| Application code | Repositories, `gateway.ibis`, Ibis expressions | `duckdb`, `sqlglot`, raw SQL |
| Repositories | `gateway.ibis`, Ibis, Pandera | `duckdb.execute`, raw SQL |
| `IbisGateway` | `ibis`, `duckdb` | `sqlglot` |
| `DuckDBPolicyBackend` | `sqlglot`, `gateway.ibis.con`, contracts | — |
| Contracts | Primitive types only | Runtime code |

These rules are enforced by an architecture test (`tests/architecture/test_ibis_only_queries.py`) that scans imports and method calls.

---

## 3. The Gateway & Ibis Adapter

The gateway layer provides the bridge between Python application code and the DuckDB database. The key innovation is exposing an **Ibis-backed interface** as the primary API, while hiding the raw DuckDB connection.

### 3.1 The StorageGateway Protocol

The `StorageGateway` protocol (defined in `storage/gateway/protocol.py`) establishes the contract that all gateway implementations must satisfy:

```python
from typing import Protocol, TYPE_CHECKING
from collections.abc import Sequence

if TYPE_CHECKING:
    from codeintel.storage.ibis_adapter import IbisGateway
    from duckdb import DuckDBPyConnection

class StorageGateway(Protocol):
    """Protocol for database access throughout the application.
    
    The gateway provides both low-level connection access (for the policy
    backend) and high-level Ibis access (for all application queries).
    """
    
    @property
    def con(self) -> "DuckDBPyConnection":
        """Raw DuckDB connection — use only in storage layer."""
        ...
    
    @property
    def ibis(self) -> "IbisGateway":
        """Ibis-backed gateway for all query building.
        
        This is the primary interface for application code. Queries should
        be expressed as Ibis expressions, not SQL strings.
        """
        ...
    
    @property
    def repo(self) -> str:
        """Current repository context (e.g., 'owner/repo')."""
        ...
    
    @property
    def commit(self) -> str:
        """Current commit SHA context."""
        ...
```

The protocol deliberately exposes `con` for the policy backend's use while making `ibis` the recommended interface. Application code should never access `con` directly.

### 3.2 The IbisGateway Wrapper

`IbisGateway` (defined in `storage/ibis_adapter.py`) wraps a `StorageGateway` and provides convenient access to Ibis functionality:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ibis
from ibis.expr.types import Table as IbisTable

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import StorageGateway


@dataclass(frozen=True)
class IbisGateway:
    """Ibis-backed interface to the DuckDB database.
    
    This class wraps a StorageGateway and provides:
    
    - Access to the Ibis DuckDB backend via `.con`
    - Table access via `.table("schema.name")`
    - Convenience methods for common operations
    
    All application queries should flow through this interface.
    """
    
    gateway: "StorageGateway"
    _backend: ibis.backends.duckdb.Backend | None = None
    
    @property
    def con(self) -> ibis.backends.duckdb.Backend:
        """Return the Ibis DuckDB backend.
        
        The backend is created lazily on first access and cached.
        """
        if self._backend is None:
            # Ibis 11 pattern: from_connection, not connect(con=...)
            object.__setattr__(
                self, 
                "_backend", 
                ibis.duckdb.from_connection(self.gateway.con)
            )
        return self._backend
    
    def table(self, qualified_name: str) -> IbisTable:
        """Access a table by qualified name (e.g., 'analytics.function_metrics').
        
        This method handles the Ibis 11 API change that requires the database
        parameter for qualified table access.
        """
        if "." in qualified_name:
            schema, name = qualified_name.split(".", 1)
            return self.con.table(name, database=schema)
        return self.con.table(qualified_name)
```

### 3.3 Ibis 11 API Considerations

The Ibis 11 release introduced breaking changes in how qualified table names are handled. The `IbisGateway.table()` method abstracts these changes:

**Before Ibis 11:**
```python
# This worked but is now deprecated
table = con.table("analytics.function_metrics")
```

**Ibis 11 and later:**
```python
# Must use the database parameter
table = con.table("function_metrics", database="analytics")
```

The `IbisGateway.table()` method accepts either form and translates to the correct Ibis 11 API internally. This shields application code from API changes.

### 3.4 Gateway Initialization

Gateways are constructed at application startup and threaded through the dependency injection system:

```python
from codeintel.storage.gateway.connection import create_gateway

# In CLI entry points, plugin execution, etc.
gateway = create_gateway(
    db_path=config.db_path,
    repo=config.repo,
    commit=config.commit,
)

# The gateway is then passed to repositories, plugins, etc.
```

The `IbisGateway` is constructed lazily when `gateway.ibis` is first accessed, ensuring no Ibis overhead for code paths that don't use it.

---

## 4. Dataset Contracts: The Schema Source of Truth

Dataset contracts are declarative Python objects that define the structure of every table in the database. These contracts drive:

- DDL generation (via SQLGlot)
- Pandera validation schemas
- IDE autocompletion and type checking
- Documentation generation

### 4.1 The TableSchema Model

A `TableSchema` captures everything needed to generate DDL for a table:

```python
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Column:
    """A single column in a table schema.
    
    Parameters
    ----------
    name
        Column name (case-insensitive in DuckDB).
    type
        DuckDB type as a string (e.g., 'VARCHAR', 'BIGINT', 'DECIMAL(38,0)').
    nullable
        Whether the column accepts NULL values.
    default
        Optional default expression (as SQL string).
    """
    name: str
    type: str
    nullable: bool = True
    default: str | None = None


@dataclass(frozen=True)
class Index:
    """A secondary index on a table.
    
    Parameters
    ----------
    name
        Index name (must be unique within the database).
    columns
        Columns to include in the index.
    unique
        Whether the index enforces uniqueness.
    """
    name: str
    columns: tuple[str, ...]
    unique: bool = False


@dataclass(frozen=True)
class TableSchema:
    """Complete schema for a database table.
    
    Parameters
    ----------
    schema
        DuckDB schema name (e.g., 'analytics', 'core', 'graph').
    name
        Table name within the schema.
    columns
        Ordered sequence of column definitions.
    primary_key
        Optional tuple of column names forming the primary key.
    indexes
        Secondary indexes on the table.
    """
    schema: str
    name: str
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...] | None = None
    indexes: tuple[Index, ...] = field(default_factory=tuple)
    
    @property
    def qualified_name(self) -> str:
        """Full table name as 'schema.table'."""
        return f"{self.schema}.{self.name}"
```

### 4.2 The DatasetContract Model

A `DatasetContract` bundles a schema with additional metadata:

```python
@dataclass(frozen=True)
class DatasetContract:
    """A complete contract for a dataset (table or view).
    
    Parameters
    ----------
    table_key
        Unique identifier in 'schema.name' format.
    description
        Human-readable description of the dataset's purpose.
    schema
        TableSchema for tables; None for views (which are defined by Ibis IR).
    row_type
        Optional TypedDict or dataclass defining the row structure.
    """
    table_key: str
    description: str
    schema: TableSchema | None
    row_type: type | None = None
```

### 4.3 Contract Registry

All contracts are registered in a central registry that the policy backend consumes:

```python
# config/datasets/__init__.py

from codeintel.config.datasets.analytics import ANALYTICS_CONTRACTS
from codeintel.config.datasets.core import CORE_CONTRACTS
from codeintel.config.datasets.graph import GRAPH_CONTRACTS
from codeintel.config.datasets.docs import DOCS_CONTRACTS

ALL_CONTRACTS: tuple[DatasetContract, ...] = (
    *ANALYTICS_CONTRACTS,
    *CORE_CONTRACTS,
    *GRAPH_CONTRACTS,
    *DOCS_CONTRACTS,
)

def get_dataset_contracts_by_table_key() -> dict[str, DatasetContract]:
    """Return all contracts indexed by table_key."""
    return {c.table_key: c for c in ALL_CONTRACTS}
```

### 4.4 Example Contract Definition

Here's a complete example for the `analytics.function_metrics` table:

```python
# config/datasets/analytics.py

from codeintel.config.datasets.primitives import (
    Column, DatasetContract, Index, TableSchema
)

FUNCTION_METRICS_SCHEMA = TableSchema(
    schema="analytics",
    name="function_metrics",
    columns=(
        Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),
        Column("module", "VARCHAR", nullable=True),
        Column("qualname", "VARCHAR", nullable=False),
        Column("loc", "INTEGER", nullable=False),
        Column("cyclomatic_complexity", "INTEGER", nullable=False),
        Column("cognitive_complexity", "INTEGER", nullable=False),
        Column("parameter_count", "INTEGER", nullable=False),
        Column("return_count", "INTEGER", nullable=False),
        Column("halstead_volume", "DOUBLE", nullable=True),
        Column("maintainability_index", "DOUBLE", nullable=True),
        Column("computed_at", "TIMESTAMP", nullable=False),
    ),
    primary_key=("function_goid_h128", "repo", "commit"),
    indexes=(
        Index(
            name="idx_function_metrics_repo_commit",
            columns=("repo", "commit"),
            unique=False,
        ),
        Index(
            name="idx_function_metrics_qualname",
            columns=("qualname",),
            unique=False,
        ),
    ),
)

FUNCTION_METRICS_CONTRACT = DatasetContract(
    table_key="analytics.function_metrics",
    description="Per-function static analysis metrics including complexity and maintainability scores.",
    schema=FUNCTION_METRICS_SCHEMA,
    row_type=FunctionMetricsRow,  # TypedDict defined in rows/analytics.py
)
```

### 4.5 Benefits of Contract-Driven Schemas

**Single source of truth.** There is exactly one place where the `function_metrics` table structure is defined. DDL, validation, type hints, and documentation all derive from this definition.

**Refactoring safety.** When you rename a column in the contract, pyright/pyrefly will flag every reference to the old name. When you add a required column, the type system will demand you provide values for it.

**Automated documentation.** The contract's `description` field and column types can be extracted to generate schema documentation automatically.

**Cross-layer consistency.** The `row_type` ensures that the Python objects flowing through repositories match the database structure exactly.

---

## 5. The DuckDB Policy Backend

The `DuckDBPolicyBackend` is the **single choke point** for all non-Ibis SQL operations. It centralizes DDL, mutations, and escape hatches, using SQLGlot to generate SQL rather than hand-crafting strings.

### 5.1 Design Principles

**Isolation.** All SQL generation is confined to this module. Application code never sees SQL; it sees semantic methods like `clear_metrics_for_snapshot()` or `ensure_table_exists()`.

**Safety.** SQLGlot generates SQL from structured expression trees, eliminating string concatenation errors and ensuring proper identifier quoting.

**Auditability.** Because all SQL flows through a single point, you can log, trace, or validate every database mutation.

**Extensibility.** New database operations are added by defining new methods on the policy backend, not by scattering SQL strings throughout the codebase.

### 5.2 Core Structure

```python
# storage/duckdb_policy_backend.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import sqlglot
from sqlglot import expressions as exp

if TYPE_CHECKING:
    from codeintel.config.datasets import TableSchema
    from codeintel.storage.gateway.protocol import StorageGateway


# Known schemas that must exist
SCHEMAS: tuple[str, ...] = ("build", "core", "graph", "analytics", "docs", "metadata")

# Tables that should not be auto-created (e.g., views handled separately)
TABLE_CREATION_DENYLIST: frozenset[str] = frozenset({"docs.v_validation_summary"})


@dataclass
class DuckDBPolicyBackend:
    """Centralized policy layer for DuckDB-specific operations.
    
    This module owns all SQL that cannot be expressed as Ibis expressions:
    
    - Schema and table DDL (CREATE SCHEMA, CREATE TABLE, CREATE INDEX)
    - Mutation operations (DELETE for snapshot scoping)
    - Future: MERGE/UPSERT patterns
    
    All application code should call semantic methods on this class rather
    than constructing SQL strings.
    """
    
    gateway: StorageGateway
    
    @property
    def backend(self):
        """Return the Ibis DuckDB backend for raw_sql execution."""
        return self.gateway.ibis.con
    
    def _run(self, expr: exp.Expression) -> None:
        """Execute a SQLGlot expression against the database."""
        sql = expr.sql(dialect="duckdb")
        self.backend.raw_sql(sql)
    
    def _run_many(self, exprs: Iterable[exp.Expression]) -> None:
        """Execute multiple SQLGlot expressions in sequence."""
        for e in exprs:
            self._run(e)
```

### 5.3 Schema Creation Methods

The policy backend generates DDL from `TableSchema` objects using SQLGlot's expression builders:

```python
    def create_schema_if_not_exists(self, schema_name: str) -> None:
        """CREATE SCHEMA IF NOT EXISTS.
        
        Parameters
        ----------
        schema_name
            Name of the schema to create (e.g., 'analytics').
        """
        # SQLGlot doesn't have ergonomic schema builders, so we parse
        stmt = sqlglot.parse_one(
            f"CREATE SCHEMA IF NOT EXISTS {sqlglot.to_identifier(schema_name).sql()}",
            dialect="duckdb",
        )
        self._run(stmt)
    
    def create_table_from_schema(
        self,
        table: TableSchema,
        *,
        drop_existing: bool = False,
        if_not_exists: bool = False,
    ) -> None:
        """Create a table from a TableSchema definition.
        
        Parameters
        ----------
        table
            TableSchema defining the table structure.
        drop_existing
            If True, DROP TABLE IF EXISTS before creating.
        if_not_exists
            If True, use CREATE TABLE IF NOT EXISTS.
        """
        exprs = self._create_table_exprs(table, drop_existing, if_not_exists)
        self._run_many(exprs)
    
    def _create_table_exprs(
        self,
        table: TableSchema,
        drop_existing: bool,
        if_not_exists: bool,
    ) -> list[exp.Expression]:
        """Build SQLGlot expressions for table creation."""
        table_ref = exp.Table(
            this=exp.to_identifier(table.name),
            db=exp.to_identifier(table.schema),
        )
        
        # Build column definitions
        cols: list[exp.Expression] = [
            self._column_def_expr(col) for col in table.columns
        ]
        
        # Add PRIMARY KEY constraint if defined
        if table.primary_key:
            pk = exp.PrimaryKey(
                expressions=[
                    exp.Column(this=exp.to_identifier(name))
                    for name in table.primary_key
                ]
            )
            cols.append(pk)
        
        create = exp.Create(
            this=table_ref,
            kind="TABLE",
            expression=exp.Schema(expressions=cols),
        )
        if if_not_exists:
            create.set("exists", True)
        
        result: list[exp.Expression] = []
        if drop_existing:
            drop = exp.Drop(this=table_ref, kind="TABLE")
            drop.set("exists", True)  # IF EXISTS
            result.append(drop)
        
        result.append(create)
        return result
    
    def _column_def_expr(self, col: Column) -> exp.ColumnDef:
        """Convert a Column to a SQLGlot ColumnDef."""
        dtype = exp.DataType.build(col.type)
        col_def = exp.ColumnDef(
            this=exp.to_identifier(col.name),
            kind=dtype,
        )
        constraints: list[exp.Expression] = []
        if not col.nullable:
            constraints.append(exp.NotNullColumnConstraint())
        if col.default is not None:
            constraints.append(
                exp.DefaultColumnConstraint(
                    this=sqlglot.parse_one(col.default, dialect="duckdb")
                )
            )
        if constraints:
            col_def.set("constraints", constraints)
        return col_def
```

### 5.4 Index Creation Methods

```python
    def create_indexes_from_schema(self, table: TableSchema) -> None:
        """Create all secondary indexes defined on a TableSchema."""
        for index in table.indexes:
            table_ref = exp.Table(
                this=exp.to_identifier(table.name),
                db=exp.to_identifier(table.schema),
            )
            
            create = exp.Create(
                this=exp.to_identifier(index.name),
                kind="INDEX",
                expression=exp.Schema(
                    expressions=[
                        exp.Column(this=exp.to_identifier(col))
                        for col in index.columns
                    ]
                ),
            )
            create.set("exists", True)  # IF NOT EXISTS
            create.set("on", table_ref)
            
            if index.unique:
                create.set("unique", True)
            
            self._run(create)
```

### 5.5 Snapshot-Scoped Deletion Methods

A critical pattern in analytics pipelines is clearing existing data for a specific (repo, commit) before inserting fresh results:

```python
    def _delete_repo_commit(
        self,
        *,
        schema: str,
        table: str,
        repo: str,
        commit: str,
    ) -> None:
        """DELETE FROM schema.table WHERE repo = ? AND commit = ?.
        
        Uses SQLGlot to generate properly quoted SQL with literal values.
        """
        tbl = exp.Table(
            this=exp.to_identifier(table),
            db=exp.to_identifier(schema),
        )
        
        condition = exp.and_(
            exp.EQ(
                exp.Column(this=exp.to_identifier("repo")),
                exp.Literal.string(repo),
            ),
            exp.EQ(
                exp.Column(this=exp.to_identifier("commit")),
                exp.Literal.string(commit),
            ),
        )
        
        delete_expr = exp.Delete(
            this=tbl,
            where=condition,
        )
        
        self._run(delete_expr)
    
    def clear_cfg_metrics(self, *, repo: str, commit: str) -> None:
        """Clear all CFG metrics tables for a snapshot.
        
        Tables cleared:
        - analytics.cfg_function_metrics
        - analytics.cfg_block_metrics
        - analytics.cfg_function_metrics_ext
        """
        for table in ("cfg_function_metrics", "cfg_block_metrics", "cfg_function_metrics_ext"):
            self._delete_repo_commit(schema="analytics", table=table, repo=repo, commit=commit)
    
    def clear_dfg_metrics(self, *, repo: str, commit: str) -> None:
        """Clear all DFG metrics tables for a snapshot."""
        for table in ("dfg_function_metrics", "dfg_block_metrics", "dfg_function_metrics_ext"):
            self._delete_repo_commit(schema="analytics", table=table, repo=repo, commit=commit)
```

### 5.6 Bulk Schema Application

The `ensure_all_schemas` method replaces the old hand-written DDL:

```python
    def ensure_all_schemas(
        self,
        *,
        drop_existing: bool,
        extra_ddl: Iterable[exp.Expression] | None = None,
    ) -> None:
        """Create all known schemas, tables, indexes, and views.
        
        This is the main bootstrap method called at connection initialization.
        It derives all DDL from dataset contracts, ensuring that Python
        definitions are the source of truth.
        
        Parameters
        ----------
        drop_existing
            If True, DROP TABLE IF EXISTS before CREATE (destructive mode).
            If False, CREATE TABLE IF NOT EXISTS (additive mode).
        extra_ddl
            Additional SQLGlot expressions to execute after table creation.
        """
        from codeintel.config.datasets import get_dataset_contracts_by_table_key
        
        # 1) Create logical schemas
        for schema_name in SCHEMAS:
            self.create_schema_if_not_exists(schema_name)
        
        # 2) Create tables and indexes
        contracts = get_dataset_contracts_by_table_key()
        for table_key, contract in contracts.items():
            if contract.schema is None:
                # Views are handled by ensure_all_views
                continue
            if table_key in TABLE_CREATION_DENYLIST:
                continue
            
            self.create_table_from_schema(
                contract.schema,
                drop_existing=drop_existing,
                if_not_exists=not drop_existing,
            )
            self.create_indexes_from_schema(contract.schema)
        
        # 3) Extra one-off DDL
        if extra_ddl:
            self._run_many(extra_ddl)
        
        # 4) Create views
        self.ensure_all_views(
            overwrite=drop_existing,
            strict=True,
        )
```

---

## 6. Ibis Expression Patterns for Queries

With the infrastructure in place, application code can express all queries using Ibis's fluent API. This section documents the canonical patterns for common query types.

### 6.1 Basic Table Access

```python
def get_modules(gateway: StorageGateway, repo: str, commit: str) -> list[dict]:
    """Load all modules for a snapshot."""
    t = gateway.ibis.table("core.modules")
    
    expr = (
        t.filter((t.repo == repo) & (t.commit == commit))
        .order_by(t.path)
    )
    
    df = expr.to_pandas()
    return df.to_dict("records")
```

Key observations:
- The table is accessed via `gateway.ibis.table()`, which handles Ibis 11 qualified name syntax
- Filtering uses Python operators (`==`, `&`) that Ibis translates to SQL predicates
- The query is executed only when `to_pandas()` is called
- Results are converted to Python dicts at the boundary

### 6.2 Joins and Aggregations

```python
def get_function_summary(gateway: StorageGateway, repo: str, commit: str) -> pd.DataFrame:
    """Build a summary joining functions with their metrics."""
    goids = gateway.ibis.table("core.goids")
    metrics = gateway.ibis.table("analytics.function_metrics")
    
    expr = (
        goids.left_join(
            metrics,
            (goids.goid_h128 == metrics.function_goid_h128)
            & (goids.repo == metrics.repo)
            & (goids.commit == metrics.commit),
        )
        .filter(
            (goids.repo == repo)
            & (goids.commit == commit)
            & (goids.kind.isin(["function", "method"]))
        )
        .select(
            goids.goid_h128,
            goids.qualname,
            goids.rel_path,
            metrics.loc,
            metrics.cyclomatic_complexity,
            metrics.maintainability_index,
        )
        .order_by(goids.qualname)
    )
    
    return expr.to_pandas()
```

### 6.3 Aggregations with Group By

```python
def get_complexity_by_module(gateway: StorageGateway, repo: str, commit: str) -> pd.DataFrame:
    """Calculate average complexity per module."""
    t = gateway.ibis.table("analytics.function_metrics")
    
    expr = (
        t.filter((t.repo == repo) & (t.commit == commit))
        .group_by(t.module)
        .aggregate(
            avg_complexity=t.cyclomatic_complexity.mean(),
            max_complexity=t.cyclomatic_complexity.max(),
            function_count=t.function_goid_h128.count(),
        )
        .order_by(ibis.desc("avg_complexity"))
    )
    
    return expr.to_pandas()
```

### 6.4 Window Functions

```python
def rank_functions_by_complexity(gateway: StorageGateway, repo: str, commit: str) -> pd.DataFrame:
    """Rank functions by complexity within each module."""
    t = gateway.ibis.table("analytics.function_metrics")
    
    window = ibis.window(group_by=t.module, order_by=ibis.desc(t.cyclomatic_complexity))
    
    expr = (
        t.filter((t.repo == repo) & (t.commit == commit))
        .mutate(
            complexity_rank=ibis.row_number().over(window),
            pct_rank=ibis.percent_rank().over(window),
        )
        .filter(ibis._.complexity_rank <= 10)  # Top 10 per module
        .order_by(t.module, "complexity_rank")
    )
    
    return expr.to_pandas()
```

### 6.5 Case Expressions (Ibis 11 Syntax)

Ibis 11 replaced the older `case().when().else_().end()` pattern with `ibis.cases()`:

```python
def categorize_function_size(gateway: StorageGateway, repo: str, commit: str) -> pd.DataFrame:
    """Categorize functions by lines of code."""
    t = gateway.ibis.table("analytics.function_metrics")
    
    # Ibis 11 pattern: ibis.cases()
    size_bucket = ibis.cases(
        (t.loc <= 10, "tiny"),
        (t.loc <= 50, "small"),
        (t.loc <= 200, "medium"),
        (t.loc <= 500, "large"),
        else_="huge",
    )
    
    expr = (
        t.filter((t.repo == repo) & (t.commit == commit))
        .mutate(size_category=size_bucket)
        .group_by("size_category")
        .aggregate(count=t.function_goid_h128.count())
    )
    
    return expr.to_pandas()
```

**Important:** The older `ibis.case().when(...).when(...).else_(...).end()` syntax has been removed in Ibis 11. Always use `ibis.cases()` with positional condition/value tuples.

### 6.6 Introspecting Generated SQL

For debugging and verification, you can always see the SQL that Ibis will generate:

```python
expr = gateway.ibis.table("core.modules").filter(...)

# Print the generated SQL
print(expr.to_sql(dialect="duckdb"))

# Or compile without specifying dialect (uses backend default)
print(ibis.to_sql(expr))
```

This is invaluable during development and debugging, but the generated SQL should never be copied into application code.

---

## 7. UDF Registry: Extending Ibis for DuckDB-Specific Functions

While Ibis covers most SQL operations, DuckDB has functions that Ibis doesn't expose natively. The UDF registry provides a centralized place to wrap these functions.

### 7.1 The Builtin UDF Pattern

Ibis's `@udf.scalar.builtin` decorator creates a type-safe wrapper around a database function:

```python
# storage/ibis_builtins.py

from __future__ import annotations

import ibis
from ibis import udf
import ibis.expr.datatypes as dt


@udf.scalar.builtin
def list_cosine_similarity(a: list[float], b: list[float]) -> float:
    """DuckDB's list_cosine_similarity for LIST columns.
    
    Calculates the cosine similarity between two list columns containing
    float values. This is commonly used for embedding similarity search.
    
    The function body is ignored; Ibis compiles calls to
    `LIST_COSINE_SIMILARITY(a, b)` in the generated SQL.
    """
    ...


@udf.scalar.builtin
def array_cosine_similarity(a: dt.Array[float, 384], b: dt.Array[float, 384]) -> float:
    """DuckDB's array_cosine_similarity for fixed-size ARRAY columns.
    
    More efficient than list_cosine_similarity when the array size is
    known at schema definition time.
    """
    ...


@udf.scalar.builtin(name="jaro_winkler_similarity")
def jaro_winkler(a: str, b: str) -> float:
    """DuckDB's Jaro-Winkler string similarity function.
    
    Returns a similarity score between 0 and 1, where 1 indicates
    identical strings.
    """
    ...


@udf.scalar.builtin
def levenshtein(a: str, b: str) -> int:
    """DuckDB's Levenshtein edit distance function.
    
    Returns the minimum number of single-character edits needed
    to transform string a into string b.
    """
    ...


# Aggregate UDFs
@udf.agg.builtin
def median(x: float) -> float:
    """DuckDB's median aggregate function."""
    ...


@udf.agg.builtin
def mode(x: str) -> str:
    """DuckDB's mode aggregate function (most frequent value)."""
    ...


__all__ = [
    "list_cosine_similarity",
    "array_cosine_similarity",
    "jaro_winkler",
    "levenshtein",
    "median",
    "mode",
]
```

### 7.2 Using Builtin UDFs in Queries

Once defined, these UDFs are used like any other Ibis expression:

```python
from codeintel.storage.ibis_builtins import jaro_winkler, median

def find_similar_functions(gateway: StorageGateway, query_name: str) -> pd.DataFrame:
    """Find functions with names similar to the query."""
    t = gateway.ibis.table("core.goids")
    
    expr = (
        t.filter(t.kind.isin(["function", "method"]))
        .mutate(
            similarity=jaro_winkler(t.name, query_name)
        )
        .filter(ibis._.similarity >= 0.8)
        .order_by(ibis.desc("similarity"))
        .limit(20)
    )
    
    return expr.to_pandas()


def get_median_complexity(gateway: StorageGateway, repo: str, commit: str) -> float:
    """Calculate median cyclomatic complexity."""
    t = gateway.ibis.table("analytics.function_metrics")
    
    expr = (
        t.filter((t.repo == repo) & (t.commit == commit))
        .aggregate(median_cc=median(t.cyclomatic_complexity.cast("float")))
    )
    
    result = expr.to_pandas()
    return result["median_cc"].iloc[0]
```

### 7.3 Guidelines for Adding UDFs

**Add UDFs only when necessary.** Ibis already covers most SQL functionality. Before adding a UDF, check if Ibis has an equivalent method.

**Document the DuckDB function.** Include a docstring explaining what the underlying DuckDB function does and when to use it.

**Consider portability.** UDFs are DuckDB-specific. If you later need to run against Snowflake, you'll need equivalent UDF definitions for that backend.

**Keep UDFs centralized.** All UDFs go in `storage/ibis_builtins.py`. Don't scatter `@udf` definitions throughout the codebase.

---

## 8. View Management via Ibis IR

Database views are **query IR** stored in the catalog. Unlike tables (which have fixed schemas), views are defined by their underlying query. In the Ibis-First architecture, views are defined as **Ibis expressions** and registered centrally.

### 8.1 The View Registry

Views are registered in a central registry that maps dataset keys to builder functions:

```python
# storage/views/ibis_registry.py

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from ibis.expr.types import Table as IbisTable

from codeintel.storage.ibis_adapter import IbisGateway


class ViewBuilder(Protocol):
    """Protocol for view builder functions."""
    def __call__(self, ibis_gateway: IbisGateway) -> IbisTable: ...


# Registry of view builders
VIEW_BUILDERS: dict[str, ViewBuilder] = {}


def register_view(table_key: str):
    """Decorator to register a view builder function."""
    def decorator(func: ViewBuilder) -> ViewBuilder:
        VIEW_BUILDERS[table_key] = func
        return func
    return decorator
```

### 8.2 Defining Views

Each view is defined as a function that takes an `IbisGateway` and returns an Ibis table expression:

```python
# storage/views/docs_views.py

from codeintel.storage.views.ibis_registry import register_view


@register_view("docs.v_function_summary")
def build_docs_v_function_summary(ibis_gateway: IbisGateway) -> IbisTable:
    """Build the docs.v_function_summary view.
    
    Joins function metrics with module information to provide a
    comprehensive function summary for documentation generation.
    """
    con = ibis_gateway.con
    
    modules = con.table("modules", database="core")
    metrics = con.table("function_metrics", database="analytics")
    goids = con.table("goids", database="core")
    
    expr = (
        goids
        .filter(goids.kind.isin(["function", "method"]))
        .left_join(
            modules,
            (modules.repo == goids.repo)
            & (modules.commit == goids.commit)
            & (modules.path == goids.rel_path),
        )
        .left_join(
            metrics,
            (metrics.repo == goids.repo)
            & (metrics.commit == goids.commit)
            & (metrics.function_goid_h128 == goids.goid_h128),
        )
        .select(
            goids.repo,
            goids.commit,
            goids.goid_h128,
            goids.qualname,
            goids.rel_path,
            modules.module,
            metrics.loc,
            metrics.cyclomatic_complexity,
            metrics.cognitive_complexity,
            metrics.maintainability_index,
            metrics.computed_at,
        )
    )
    
    return expr


@register_view("docs.v_high_complexity_functions")
def build_docs_v_high_complexity_functions(ibis_gateway: IbisGateway) -> IbisTable:
    """View of functions exceeding complexity thresholds."""
    con = ibis_gateway.con
    
    metrics = con.table("function_metrics", database="analytics")
    
    expr = (
        metrics
        .filter(
            (metrics.cyclomatic_complexity > 10)
            | (metrics.cognitive_complexity > 15)
        )
        .order_by(ibis.desc(metrics.cyclomatic_complexity))
    )
    
    return expr
```

### 8.3 View Materialization

The policy backend materializes all registered views:

```python
# In DuckDBPolicyBackend

def ensure_all_views(
    self,
    *,
    overwrite: bool = True,
    strict: bool = True,
) -> None:
    """Materialize all registered Ibis views.
    
    Parameters
    ----------
    overwrite
        If True, CREATE OR REPLACE VIEW (destructive).
        If False, CREATE VIEW IF NOT EXISTS (additive).
    strict
        If True, raise if a view contract has no registered builder.
    """
    from codeintel.config.datasets import get_dataset_contracts_by_table_key
    from codeintel.storage.views.ibis_registry import VIEW_BUILDERS
    
    contracts = get_dataset_contracts_by_table_key()
    
    for table_key, contract in contracts.items():
        if contract.schema is not None:
            # It's a table, not a view
            continue
        
        builder = VIEW_BUILDERS.get(table_key)
        if builder is None:
            if strict:
                raise KeyError(f"No view builder registered for '{table_key}'")
            continue
        
        expr = builder(self.gateway.ibis)
        self._create_or_replace_view(
            table_key=table_key,
            expr=expr,
            overwrite=overwrite,
        )

def _create_or_replace_view(
    self,
    *,
    table_key: str,
    expr: IbisTable,
    overwrite: bool,
) -> None:
    """Create a view from an Ibis expression."""
    schema, name = table_key.split(".", 1)
    
    # Ibis 11 pattern: use database parameter
    self.backend.create_view(
        name,
        expr,
        database=schema,
        overwrite=overwrite,
    )
```

### 8.4 Benefits of Ibis-Based Views

**Type safety.** The view definition uses the same Ibis expressions as queries, so column references are validated.

**Composability.** View builders can call other Ibis code, reusing query fragments.

**Testability.** You can unit test view builders by constructing them and inspecting `expr.to_sql()`.

**Documentation.** View builders have docstrings that explain their purpose, unlike raw CREATE VIEW SQL.

---

## 9. Pandera Integration for Runtime Validation

Dataset contracts define **static schemas**. Pandera extends this with **runtime validation** of actual data against those schemas.

### 9.1 Deriving Pandera Schemas from TableSchema

A mapping layer translates `TableSchema` definitions to Pandera `DataFrameSchema` objects:

```python
# storage/pandera_schemas.py

from __future__ import annotations

import pandera as pa

from codeintel.config.datasets import TableSchema, get_dataset_contracts_by_table_key


# Mapping from DuckDB types to Pandera types
_DUCKDB_TO_PANDERA: dict[str, type] = {
    "BOOLEAN": pa.Bool,
    "TINYINT": pa.Int8,
    "SMALLINT": pa.Int16,
    "INTEGER": pa.Int32,
    "BIGINT": pa.Int64,
    "HUGEINT": pa.Int64,  # Pandera doesn't have Int128
    "REAL": pa.Float32,
    "FLOAT": pa.Float32,
    "DOUBLE": pa.Float64,
    "DECIMAL": pa.Float64,  # Approximate
    "VARCHAR": pa.String,
    "TEXT": pa.String,
    "TIMESTAMP": pa.Timestamp,
    "TIMESTAMPTZ": pa.Timestamp,
    "DATE": pa.Date,
    "TIME": pa.Time,
    "JSON": pa.String,  # JSON stored as string
}


def _parse_type(duckdb_type: str) -> type:
    """Parse a DuckDB type string to a Pandera type."""
    # Handle parameterized types like DECIMAL(38,0)
    base_type = duckdb_type.split("(")[0].upper()
    return _DUCKDB_TO_PANDERA.get(base_type, pa.Object)


def pandera_schema_from_table(table: TableSchema) -> pa.DataFrameSchema:
    """Create a Pandera DataFrameSchema from a TableSchema."""
    cols: dict[str, pa.Column] = {}
    
    for col in table.columns:
        ptype = _parse_type(col.type)
        cols[col.name] = pa.Column(
            ptype,
            nullable=col.nullable,
            coerce=True,  # Allow type coercion
        )
    
    return pa.DataFrameSchema(
        cols,
        strict=True,  # No unexpected columns
        coerce=True,
    )


# Pre-build schemas for all contracts
_PANDERA_SCHEMAS: dict[str, pa.DataFrameSchema] = {}


def get_pandera_schema(table_key: str) -> pa.DataFrameSchema:
    """Get or create a Pandera schema for a dataset."""
    if table_key not in _PANDERA_SCHEMAS:
        contracts = get_dataset_contracts_by_table_key()
        if table_key not in contracts:
            raise KeyError(f"Unknown dataset: {table_key}")
        contract = contracts[table_key]
        if contract.schema is None:
            raise ValueError(f"Dataset '{table_key}' is a view with no TableSchema")
        _PANDERA_SCHEMAS[table_key] = pandera_schema_from_table(contract.schema)
    return _PANDERA_SCHEMAS[table_key]
```

### 9.2 Using Validation in Repositories

Repositories can validate data before returning it to callers:

```python
# storage/repositories/functions.py

from codeintel.storage.pandera_schemas import get_pandera_schema


class FunctionRepository:
    """Repository for function-related queries."""
    
    def __init__(self, gateway: StorageGateway):
        self.gateway = gateway
    
    def get_function_metrics(
        self,
        repo: str,
        commit: str,
        *,
        validate: bool = True,
    ) -> pd.DataFrame:
        """Load function metrics for a snapshot.
        
        Parameters
        ----------
        repo
            Repository identifier.
        commit
            Commit SHA.
        validate
            If True, validate results against the Pandera schema.
            Raises pandera.errors.SchemaError on validation failure.
        """
        t = self.gateway.ibis.table("analytics.function_metrics")
        
        expr = (
            t.filter((t.repo == repo) & (t.commit == commit))
            .order_by(t.qualname)
        )
        
        df = expr.to_pandas()
        
        if validate:
            schema = get_pandera_schema("analytics.function_metrics")
            df = schema.validate(df)
        
        return df
```

### 9.3 Validation at Write Time

For data entering the database, validation can catch errors before they're persisted:

```python
def store_function_metrics(
    gateway: StorageGateway,
    metrics: pd.DataFrame,
) -> None:
    """Persist function metrics to the database.
    
    The DataFrame is validated against the schema before insertion.
    """
    schema = get_pandera_schema("analytics.function_metrics")
    validated = schema.validate(metrics)
    
    # Use Ibis to insert (or the policy backend for bulk ops)
    # ...
```

### 9.4 Validation Modes

Pandera supports different validation modes that can be selected based on context:

- **Eager validation (default):** Raises on first error. Good for production.
- **Lazy validation:** Collects all errors before raising. Good for debugging.
- **Drop invalid rows:** Returns only valid rows. Good for resilient ingestion.

```python
# Lazy validation for debugging
try:
    df = schema.validate(df, lazy=True)
except pa.errors.SchemaErrors as err:
    print(err.failure_cases)  # DataFrame of all failures

# Drop invalid rows for ingestion
schema_lenient = schema.to_schema().update(drop_invalid_rows=True)
df_clean = schema_lenient.validate(df, lazy=True)
```

---

## 10. Migration Strategy

Migrating to the Ibis-First architecture is an incremental process. This section outlines the phases and provides guidance for each.

### 10.1 Phase 1: Infrastructure Foundation

**Goal:** Establish the core infrastructure without disrupting existing code.

**Steps:**

1. **Implement `IbisGateway`** in `storage/ibis_adapter.py`
   - Create the wrapper class
   - Ensure Ibis 11 compatibility

2. **Update `StorageGateway` protocol** to expose `ibis` property
   - Modify `storage/gateway/protocol.py`
   - Update `DuckDBGateway` to construct `IbisGateway` lazily

3. **Create `DuckDBPolicyBackend`** skeleton
   - Implement basic DDL methods
   - Wire into schema bootstrap

4. **Add architecture test**
   - Scan for `gateway.con.execute` usage
   - Initially in warn mode (track but don't fail)

**Duration:** 1-2 sprints

### 10.2 Phase 2: Repository Migration

**Goal:** Convert all repository read operations to Ibis.

**Steps:**

1. **Identify all repositories** (`storage/repositories/*.py`)
   - Inventory current SQL queries
   - Prioritize by usage frequency

2. **Convert repositories one at a time**
   - Replace `gateway.con.execute(...)` with Ibis expressions
   - Keep return types unchanged (backward compatible)
   - Add unit tests verifying Ibis output matches SQL output

3. **Add Pandera validation** to high-value repositories
   - Start with `function_metrics`, `modules`, `goids`
   - Validate in tests; optionally in production

**Duration:** 2-3 sprints (can be parallelized across developers)

### 10.3 Phase 3: Analytics and Ingestion Migration

**Goal:** Convert analytics and ingestion plugins to use Ibis for reads.

**Steps:**

1. **Migrate analytics plugins**
   - Replace `gateway.con.execute(...)` with `gateway.ibis.table(...)`
   - Route deletions through `DuckDBPolicyBackend.clear_*_metrics()`
   - Keep `executemany` inserts for now (bulk insert is complex)

2. **Migrate ingestion plugins**
   - Convert table reads to Ibis
   - Route schema operations through policy backend

3. **Enable architecture test as hard failure**
   - Any new SQL in application code fails CI

**Duration:** 2-3 sprints

### 10.4 Phase 4: DDL and View Consolidation

**Goal:** Make dataset contracts the sole source of DDL.

**Steps:**

1. **Delete `TABLE_DDL` and `INDEX_DDL` constants**
   - Policy backend generates from contracts

2. **Migrate view definitions** to view registry
   - Convert each `CREATE VIEW` to an Ibis builder
   - Register in `VIEW_BUILDERS`

3. **Delete legacy DDL files**
   - Remove `storage/schema/ddl.py` content (keep thin wrapper)
   - Remove hand-written SQL strings

**Duration:** 1-2 sprints

### 10.5 Phase 5: Bulk Operations and Cleanup ✅ COMPLETE

**Goal:** Handle remaining edge cases and polish.

**Status:** Completed on 2025-12-11

**Completed Steps:**

1. ✅ **Implemented `DuckDBPolicyBackend.bulk_insert()`**
   - Uses SQLGlot to generate INSERT statements
   - Accepts table_key, rows, and optional columns
   - Falls back to TableSchema for column discovery

2. ✅ **Implemented `DuckDBPolicyBackend.upsert()`**
   - Uses DuckDB's `INSERT ... ON CONFLICT` syntax
   - Supports conflict detection and selective updates
   - Generated via SQLGlot with identifier validation

3. ✅ **Deprecated legacy SQL modules**
   - `ingestion/infrastructure/safe_sql.py` - module-level deprecation warning
   - `storage/sql/primitives.py` - deprecated `SafeTable`, `SafeColumn`, `QueryBuilder`
   - Updated docstrings to redirect to `DuckDBPolicyBackend`

4. ✅ **Migrated representative `executemany` usages**
   - `analytics/profiles/writer_guard.py` - added `PolicyWriterConfig`
   - `analytics/adapters/dependencies.py` - both adapters now use `bulk_insert()`
   - `analytics/graphs/graph_stats.py` - uses policy backend

5. ✅ **Added architecture test**
   - `test_executemany_centralized_in_policy_backend()` detects direct `executemany`
   - Allowlist for storage layer and pending migration files

6. ✅ **Updated AGENTS.md**
   - Added "Bulk Operations (DuckDBPolicyBackend)" section
   - Documented `bulk_insert()`, `upsert()`, and deprecated patterns

7. ✅ **Migrated `DuckDBManifestStore` to Ibis-first**
   - Changed constructor to accept `StorageGateway` instead of `DuckDBPyConnection`
   - Rewrote `load_last_record()` to use Ibis expressions
   - Rewrote `append_record()` to use `DuckDBPolicyBackend.bulk_insert()`
   - Updated tests to use `GatewayFactory` pattern

8. ✅ **Fixed `_build_insert()` SQLGlot generation**
   - SQLGlot requires `exp.Schema` wrapper for INSERT column names
   - Without this, columns were rendered as `?` placeholders

**Duration:** Completed in 1 sprint

### 10.6 Post-Migration: Final Architecture Patterns

With Phase 5 complete, the Ibis-First architecture is fully realized. The final patterns are:

**Query Patterns:**
- All queries use Ibis expressions via `gateway.ibis.table()`
- Views are defined as Ibis expression builders in `VIEW_BUILDERS` registry
- Repository layer provides domain-oriented query methods

**Mutation Patterns:**
- Bulk inserts: `DuckDBPolicyBackend.bulk_insert(table_key, rows, columns=...)`
- Upserts: `DuckDBPolicyBackend.upsert(table_key, rows, columns=..., conflict_columns=...)`
- Deletions: `DuckDBPolicyBackend.delete_for_snapshot(table_key, repo=..., commit=...)`

**DDL Patterns:**
- Schema creation: `DuckDBPolicyBackend.ensure_all_schemas()`
- View creation: `DuckDBPolicyBackend.ensure_all_views()` using Ibis registry
- All DDL generated via SQLGlot from `TableSchema` contracts

**Deprecated (do not use in new code):**
- `con.execute()` / `con.executemany()` outside storage layer
- `SafeTable`, `SafeColumn`, `QueryBuilder` from `storage/sql/primitives.py`
- `ingestion/infrastructure/safe_sql.py` module

---

## 11. Configuration & Environment

### 11.1 Ibis Configuration

Ibis behavior is controlled through global options:

```python
import ibis

# Disable interactive mode in library code (default)
ibis.options.interactive = False

# Enable for notebook/REPL development
# ibis.options.interactive = True

# Set default backend if needed (we use explicit connections)
# ibis.set_backend(gateway.ibis.con)
```

### 11.2 Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CODEINTEL_DB_PATH` | Path to DuckDB file | `:memory:` |
| `PANDERA_VALIDATION_DEPTH` | Validation mode: `SCHEMA_ONLY`, `DATA_ONLY`, `SCHEMA_AND_DATA` | `SCHEMA_AND_DATA` |

### 11.3 Test Configuration

Tests should use isolated DuckDB instances:

```python
@pytest.fixture
def test_gateway(tmp_path):
    """Create a test gateway with fresh database."""
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    gateway = create_gateway_for_connection(con, repo="test/repo", commit="abc123")
    
    # Bootstrap schema
    policy = DuckDBPolicyBackend(gateway)
    policy.ensure_all_schemas(drop_existing=True)
    
    yield gateway
    
    con.close()
```

---

## 12. Ibis 11 Compatibility Guidelines

Ibis 11 introduced breaking changes. Follow these guidelines to ensure compatibility:

### 12.1 Connection Creation

```python
# ✅ Correct (Ibis 11)
con = ibis.duckdb.from_connection(duckdb_connection)

# ❌ Deprecated (pre-Ibis 11)
con = ibis.duckdb.connect(con=duckdb_connection)
```

### 12.2 Table Access

```python
# ✅ Correct (Ibis 11) - use database parameter
table = con.table("function_metrics", database="analytics")

# ❌ Will fail (Ibis 11)
table = con.table("analytics.function_metrics")

# ✅ Use IbisGateway.table() which handles splitting
table = gateway.ibis.table("analytics.function_metrics")
```

### 12.3 Case Expressions

```python
# ✅ Correct (Ibis 11) - ibis.cases() builder
bucket = ibis.cases(
    (loc <= 50, "small"),
    (loc <= 200, "medium"),
    else_="large",
)

# ❌ Removed (Ibis 11)
bucket = ibis.case().when(loc <= 50, "small").when(loc <= 200, "medium").else_("large").end()
```

### 12.4 View Creation

```python
# ✅ Correct (Ibis 11) - use database parameter
con.create_view("v_summary", expr, database="analytics", overwrite=True)

# ❌ Will create in wrong schema (Ibis 11)
con.create_view("analytics.v_summary", expr, overwrite=True)
```

### 12.5 Deprecated Methods

| Deprecated | Replacement |
|------------|-------------|
| `StringValue.to_date()` | `StringValue.as_date()` |
| `StringValue.to_timestamp()` | `StringValue.as_timestamp()` |
| `IntegerValue.to_interval()` | `IntegerValue.as_interval()` |
| `IntegerValue.to_timestamp()` | `IntegerValue.as_timestamp()` |
| `Struct.destructure()` | `Table.unpack()` |

---

## 13. Anti-Patterns & What to Avoid

### 13.1 SQL Strings in Application Code

```python
# ❌ Never do this
rows = gateway.con.execute(
    "SELECT * FROM analytics.function_metrics WHERE repo = ?",
    [repo],
).fetchall()

# ✅ Use Ibis instead
t = gateway.ibis.table("analytics.function_metrics")
df = t.filter(t.repo == repo).to_pandas()
```

### 13.2 Mixing Ibis and Raw SQL

```python
# ❌ Don't construct SQL and pass to Ibis
sql_fragment = f"WHERE repo = '{repo}'"  # SQL injection risk!
# ...

# ✅ Use Ibis expressions throughout
t.filter(t.repo == repo)  # Safe and typed
```

### 13.3 Executing Inside Library Functions

```python
# ❌ Executing deep in a library breaks composability
def calculate_metric(gateway: StorageGateway) -> float:
    t = gateway.ibis.table("analytics.function_metrics")
    expr = t.aggregate(avg=t.loc.mean())
    return expr.to_pandas()["avg"].iloc[0]  # Executes!

# ✅ Return the expression; let caller decide when to execute
def build_metric_query(gateway: StorageGateway) -> IbisTable:
    t = gateway.ibis.table("analytics.function_metrics")
    return t.aggregate(avg=t.loc.mean())

# Caller executes at the boundary
result = build_metric_query(gateway).to_pandas()
```

### 13.4 Hardcoding Column Names

```python
# ❌ Hardcoded column names can get out of sync
expr = t.select("function_goid_h128", "qualname", "loc")

# ✅ Reference columns as attributes for type checking
expr = t.select(t.function_goid_h128, t.qualname, t.loc)
```

### 13.5 Ignoring Generated SQL During Debugging

```python
# ❌ Writing complex expressions without verifying SQL
expr = (
    t1.join(t2, ...).join(t3, ...)
    .filter(...).group_by(...).aggregate(...)
)
# "I hope this is right!"

# ✅ Always check generated SQL during development
print(expr.to_sql(dialect="duckdb"))
# Verify it matches expectations before committing
```

---

## Appendix: SQLGlot Expression Reference

This appendix provides quick reference for common SQLGlot patterns used in the policy backend.

### A.1 Table References

```python
# Simple table reference
table = exp.Table(this=exp.to_identifier("function_metrics"))

# Qualified table reference
table = exp.Table(
    this=exp.to_identifier("function_metrics"),
    db=exp.to_identifier("analytics"),
)
```

### A.2 Column Definitions

```python
# Basic column
col = exp.ColumnDef(
    this=exp.to_identifier("user_id"),
    kind=exp.DataType.build("BIGINT"),
)

# Column with NOT NULL
col = exp.ColumnDef(
    this=exp.to_identifier("user_id"),
    kind=exp.DataType.build("BIGINT"),
    constraints=[exp.NotNullColumnConstraint()],
)

# Column with DEFAULT
col = exp.ColumnDef(
    this=exp.to_identifier("created_at"),
    kind=exp.DataType.build("TIMESTAMP"),
    constraints=[
        exp.DefaultColumnConstraint(
            this=sqlglot.parse_one("now()", dialect="duckdb")
        )
    ],
)
```

### A.3 CREATE Statements

```python
# CREATE SCHEMA
create_schema = sqlglot.parse_one(
    "CREATE SCHEMA IF NOT EXISTS analytics",
    dialect="duckdb",
)

# CREATE TABLE
create_table = exp.Create(
    this=exp.Table(this=exp.to_identifier("users"), db=exp.to_identifier("core")),
    kind="TABLE",
    expression=exp.Schema(expressions=[col1, col2, pk_constraint]),
    exists=True,  # IF NOT EXISTS
)

# CREATE INDEX
create_index = exp.Create(
    this=exp.to_identifier("idx_users_email"),
    kind="INDEX",
    expression=exp.Schema(
        expressions=[exp.Column(this=exp.to_identifier("email"))]
    ),
    on=exp.Table(this=exp.to_identifier("users"), db=exp.to_identifier("core")),
    exists=True,  # IF NOT EXISTS
    unique=True,  # UNIQUE INDEX
)
```

### A.4 DELETE Statements

```python
# DELETE with WHERE
delete = exp.Delete(
    this=exp.Table(this=exp.to_identifier("metrics"), db=exp.to_identifier("analytics")),
    where=exp.and_(
        exp.EQ(exp.Column(this=exp.to_identifier("repo")), exp.Literal.string("owner/repo")),
        exp.EQ(exp.Column(this=exp.to_identifier("commit")), exp.Literal.string("abc123")),
    ),
)
```

### A.5 DROP Statements

```python
# DROP TABLE IF EXISTS
drop = exp.Drop(
    this=exp.Table(this=exp.to_identifier("old_table"), db=exp.to_identifier("core")),
    kind="TABLE",
    exists=True,  # IF EXISTS
)

# DROP SCHEMA CASCADE
drop_schema = exp.Drop(
    this=exp.to_identifier("deprecated_schema"),
    kind="SCHEMA",
    exists=True,
    cascade=True,
)
```

### A.6 Generating SQL

```python
# Generate DuckDB-dialect SQL
sql = expr.sql(dialect="duckdb")

# With pretty printing
sql = expr.sql(dialect="duckdb", pretty=True)
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-12-11 | — | Initial architecture document |

---

## References

1. [Ibis Project Documentation](https://ibis-project.org/)
2. [DuckDB SQL Reference](https://duckdb.org/docs/sql/introduction)
3. [SQLGlot Documentation](https://sqlglot.com/)
4. [Pandera Documentation](https://pandera.readthedocs.io/)
5. [AGENTS.md — CodeIntel Operating Protocol](../AGENTS.md)
