
You’re absolutely right that this is the moment to “tie a bow on it” and simplify the whole DB story. Given what you’ve built already (ibis_adapter, dataset contracts, storage/sql primitives, macros, etc.), you’re actually very close to the architecture you described. 

Below is a detailed but *finite* implementation plan to get to:

1. **Ibis as canonical query language**
2. **Ibis built-ins + UDFs for coverage**
3. **Ibis `.sql` / `Backend.raw_sql` as the only escape hatches**
4. **SQLGlot as the generator for the “messy” SQL**
5. **Dataset contracts + Pandera/Ibis schemas as the driver for DDL + validation**

I’ll structure this as a series of concrete epics/patch-sets you can hand to Cursor.

---

## 0. Ground truth you already have

From your current repo:

* Dataset contracts are fully modularized under `config/config/datasets/*` with `TableSchema`, `DatasetContract`, etc. 
* You already have:

  * `storage/ibis_adapter.py` to wrap a `StorageGateway` DuckDB connection in an Ibis backend. 
  * Ibis-based docs views in `storage/views/ibis_views.py` that call `gateway.ibis.con` and `con.create_view("analytics.v_function_summary", summary, overwrite=True)`. 
* Schema + DDL are generated from `TableSchema` in `storage/schema/ddl.py`. 
* Macro/bootstrap logic lives in `storage/metadata/bootstrap.py` and `storage/macros/*`. 
* You have SQL-builder primitives in `storage/sql/primitives.py` and dataset-driven INSERT/DELETE helpers in `config/config/datasets/sql.py`. 

So this plan is mostly *re-wiring*, not invention.

---

## 1. Make Ibis the canonical DB handle

### 1.1 Finalize the `gateway.ibis` property

You already have `IbisGateway` in `storage/ibis_adapter.py` that wraps a `StorageGateway`. 

**Patch: `StorageGateway` protocol**

* In `storage/gateway/protocol.py`, add:

```python
from typing import Protocol, TYPE_CHECKING
import duckdb

if TYPE_CHECKING:
    from codeintel.storage.ibis_adapter import IbisGateway

class StorageGateway(Protocol):
    ...
    @property
    def con(self) -> DuckDBConnection: ...
    def execute(self, sql: str, params: Sequence[object] | None = None) -> DuckDBConnection: ...
    def table(self, name: str) -> DuckDBRelation: ...

    # NEW
    @property
    def ibis(self) -> "IbisGateway":
        """Ibis-backed view of this gateway for all query building."""
        ...
```

**Patch: `DuckDBGateway` implementation**

* In `storage/gateway/accessors.py`, where `DuckDBGateway` is defined, add:

```python
from dataclasses import dataclass, field
from codeintel.storage.ibis_adapter import IbisGateway

@dataclass(frozen=True)
class DuckDBGateway:
    ...
    con: DuckDBConnection
    ibis: IbisGateway = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ibis", IbisGateway(self))
        self.analytics = AnalyticsTables(self.con)
        self.build = BuildTracking(self.con)
        self.core = CoreTables(self.con)
        self.docs = DocsViews(self.con)
        self.graph = GraphTables(self.con)
        self.runs = PipelineRunTracking(self.con)
```

Now **everywhere** you currently have a `StorageGateway` you also have:

* `gateway.ibis.con` → Ibis DuckDB backend
* `gateway.ibis.table("core.modules")` → `it.Table`

This satisfies (1) in terms of plumbing.

---

## 2. Enforce “Ibis for all queries” and localize SQL to storage/policy

### 2.1 Add a guardrail test: no `gateway.con.execute` outside storage/tests

Add an architecture test (e.g. `tests/architecture/test_ibis_only_queries.py`) that:

* Scans `src/codeintel/**.py`
* Fails if it sees `.gateway.con.execute(` or `gateway.con.execute(` outside:

  * `storage/`
  * `tests/`

This gives you an automated TODO list (analytics, ingestion, graphs, etc.) for converting to Ibis.

### 2.2 Convert repositories to Ibis

Repositories in `storage/repositories/*.py` are the main “read/query” interface serving analytics/serving/etc.

**Pattern:**

* Before (representative):

```python
rows = self.con.execute(
    "SELECT * FROM docs.v_function_summary WHERE repo = ? AND commit = ?",
    [self.repo, self.commit],
).fetchall()
```

* After:

```python
import ibis
from ibis import _

def list_function_summary(self) -> list[dict[str, object]]:
    t = self._gateway.ibis.table("docs.v_function_summary")
    expr = (
        t.filter((t.repo == self.repo) & (t.commit == self.commit))
        .order_by(t.qualname)
    )
    df = expr.to_pandas()
    return df.to_dict("records")
```

Guidelines:

* All new methods in repositories must be expressed as Ibis expressions.
* When you touch an existing SQL-based method, convert it to Ibis.
* You’re free to still use `RowBinding`/typed rows by:

  * `df = expr.to_pandas()`
  * Pandera validation (later step)
  * mapping rows into dataclass/TypedDict if needed.

This achieves “Ibis as canonical query language” at the repository boundary.

### 2.3 Convert ingestion / analytics / graphs away from direct SQL

You’ve got ~30 `gateway.con.execute` uses in analytics, ~22 in ingestion, ~15 in graphs. 

For each:

1. **If it’s pulling data that a repository could expose**, hoist to a repo method.

   Example in ingestion plugin:

   * Before:

     ```python
     rows = ctx.gateway.con.execute(
         "SELECT path FROM core.modules WHERE repo = ? AND commit = ?",
         [ctx.repo, ctx.commit],
     ).fetchall()
     ```

   * After (either):

     a) Add `list_module_paths(repo, commit)` in `storage/repositories/modules.py` (Ibis), or

     b) Inline Ibis:

     ```python
     t = ctx.gateway.ibis.table("core.modules")
     expr = t.filter((t.repo == ctx.repo) & (t.commit == ctx.commit)).select("path")
     rows = expr.execute()
     ```

2. **If it’s one-off metrics/validation** (graphs/analytics), consider a small Ibis helper:

   * Example (cfg/dfg analytics):

     ```python
     t = gateway.ibis.table("graph.call_graph_edges")
     expr = (
         t.filter((t.repo == repo) & (t.commit == commit))
         .group_by("caller_goid_h128")
         .aggregate(out_degree=t.callee_goid_h128.nunique())
     )
     df = expr.to_pandas()
     ```

Target: after this pass, **no app layer uses `gateway.con.execute`**; they either:

* Use Ibis directly.
* Or call repositories that are themselves Ibis-based.

---

## 3. Ibis built-ins + UDFs for function coverage

Create a central module:

* `storage/ibis_builtins.py`

Example structure:

```python
# storage/ibis_builtins.py

from __future__ import annotations
import ibis
from ibis import udf
import ibis.expr.datatypes as dt

# Scalar builtin wrappers

@udf.scalar.builtin
def list_cosine_similarity(x, y) -> dt.float64:
    """DuckDB list_cosine_similarity for vector columns."""
    ...

@udf.scalar.builtin
def array_cosine_similarity(x, y) -> dt.float64:
    """DuckDB array_cosine_similarity for fixed-size arrays."""
    ...

# Aggregate builtin wrappers (if needed)
@udf.agg.builtin
def median(x) -> dt.float64:
    ...

__all__ = [
    "list_cosine_similarity",
    "array_cosine_similarity",
    "median",
]
```

Conventions:

* Any time you need a DuckDB function that Ibis doesn’t expose yet, add a wrapper here and use it from Ibis code.
* Don’t sprinkle `@udf` definitions across the tree; keep them in this one module.

This covers item (2) in your list.

---

## 4. Create a `duckdb_policy_backend` for all remaining “messy” SQL

You want *one* place that is allowed to emit or receive “non-Ibis” SQL.

Create:

* `storage/duckdb_policy_backend.py`

Core responsibilities:

1. Schema DDL (tables & indexes) – migrating from `storage/schema/ddl.py` into SQLGlot/Ibis forms.
2. Metadata tables & ingest macros DDL – from `storage/metadata/bootstrap.py` and `storage/macros/registration.py`.
3. Any future MERGE/UPSERT patterns.

### 4.1 Shape of the policy backend

```python
# storage/duckdb_policy_backend.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import sqlglot
from sqlglot import expressions as exp

from codeintel.config.datasets import get_dataset_contracts_by_table_key, TableSchema
from codeintel.storage.gateway.protocol import StorageGateway


@dataclass
class DuckDBPolicyBackend:
    """Centralized policy for DuckDB-specific operations.

    All non-Ibis SQL (DDL, PRAGMAs, MERGE, INDEXES) must go through here.
    """

    gateway: StorageGateway

    @property
    def backend(self):
        # Ibis backend
        return self.gateway.ibis.con

    def raw_sql(self, sql: str, params: list[object] | None = None) -> None:
        """Run raw SQL via Ibis backend."""
        if params:
            self.backend.raw_sql(sql, params=params)
        else:
            self.backend.raw_sql(sql)

    # --- DDL from TableSchema via SQLGlot ---

    def create_schema_if_not_exists(self, schema: str) -> None:
        stmt = (
            sqlglot
            .parse_one(f"CREATE SCHEMA IF NOT EXISTS {schema}", dialect="duckdb")
            .sql(dialect="duckdb")
        )
        self.raw_sql(stmt)

    def create_table_from_schema(self, schema: TableSchema, if_not_exists: bool = False) -> None:
        # Convert TableSchema → SQLGlot CREATE TABLE
        columns = [
            exp.ColumnDef(
                this=exp.to_identifier(col.name),
                kind=exp.DataType.build(col.type),
                not_null=not col.nullable,
            )
            for col in schema.columns
        ]
        if schema.primary_key:
            pk = exp.PrimaryKey(
                this=exp.Identifier(this="PRIMARY KEY"),
                expressions=[exp.to_identifier(c) for c in schema.primary_key],
            )
            columns.append(pk)

        create = exp.Create(
            this=exp.Table(this=exp.to_identifier(schema.name), db=schema.schema),
            kind="TABLE",
            expression=exp.Schema(expressions=columns),
            exists=if_not_exists,
        )

        sql = create.sql(dialect="duckdb")
        self.raw_sql(sql)

    def create_index_from_schema(self, schema: TableSchema) -> None:
        for index in schema.indexes:
            create = exp.Create(
                this=exp.to_identifier(index.name),
                kind="INDEX",
                exists=True,  # IF NOT EXISTS
                expression=exp.Schema(
                    expressions=[
                        exp.Column(this=exp.to_identifier(col)) for col in index.columns
                    ]
                ),
                on=exp.Table(this=exp.to_identifier(schema.name), db=schema.schema),
            )
            if index.unique:
                create.set("unique", True)
            sql = create.sql(dialect="duckdb")
            self.raw_sql(sql)
```

Then:

### 4.2 Replace `storage/schema/ddl.py` with policy backend

* Instead of precomputed `TABLE_DDL`, `INDEX_DDL`, etc., rewrite:

```python
# storage/schema/ddl.py

from codeintel.config.datasets import get_dataset_contracts_by_table_key
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend

SCHEMAS = ("build", "core", "graph", "analytics", "docs")

def apply_all_schemas(gateway: StorageGateway, *, if_not_exists: bool = False) -> None:
    policy = DuckDBPolicyBackend(gateway)
    for schema_name in SCHEMAS:
        policy.create_schema_if_not_exists(schema_name)
    for contract in get_dataset_contracts_by_table_key().values():
        if contract.schema is None:
            continue
        policy.create_table_from_schema(contract.schema, if_not_exists=if_not_exists)
        policy.create_index_from_schema(contract.schema)
```

* Adjust `gateway/connection.py` to call `apply_all_schemas(gateway, if_not_exists=config.apply_schema_mode)` instead of using raw DuckDB connection.

This moves DDL into “SQLGlot + Ibis raw_sql”, satisfying (3) and (4).

### 4.3 Move ingest macros & metadata bootstrap into policy backend

You probably don’t want to delete macros **today**, but you can:

* Keep macro definitions (names, coverage checks) in `storage/metadata/bootstrap.py`.
* Move actual `CREATE MACRO ...` strings into:

  * `DuckDBPolicyBackend.ensure_ingest_macros()` that:

    * builds the macro DDL with SQLGlot or your existing string templates,
    * calls `raw_sql` for each.

Then `storage/macros/registration.py` becomes a very thin shim calling `DuckDBPolicyBackend`.

---

## 5. Wire SQLGlot into dataset contracts for DDL & special DML

You already use `TableSchema` for DDL. Now, you’re just changing its generator from “f-strings” to SQLGlot.

We already sketched `create_table_from_schema`. Similarly, you can:

* Implement `upsert_from_contract(dataset: DatasetContract, spec: UpsertSpec) -> str` using SQLGlot.
* Replace constant `UPDATE` strings in `config/config/datasets/sql.py` (e.g., `TEST_CATALOG_UPDATE_GOIDS`, `GOID_CROSSWALK_UPDATE_SCIP`) with SQLGlot-generated statements.

Example for `GOID_CROSSWALK_UPDATE_SCIP`:

* Before:

```python
GOID_CROSSWALK_UPDATE_SCIP: Final[str] = (
    "UPDATE core.goid_crosswalk SET scip_symbol = ? WHERE goid = ? AND repo = ? AND commit = ?"
)
```

* After: a generator in `duckdb_policy_backend.py`:

```python
def build_goid_crosswalk_update_sql(self) -> str:
    table = exp.Table(this=exp.to_identifier("goid_crosswalk"), db="core")
    update = (
        exp.Update(this=table)
        .set(
            expressions=[
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("scip_symbol")),
                    expression=exp.Parameter(this="scip_symbol"),
                )
            ]
        )
        .where(
            exp.and_(
                exp.EQ(exp.Column(this=exp.to_identifier("goid")), exp.Parameter(this="goid")),
                exp.EQ(exp.Column(this=exp.to_identifier("repo")), exp.Parameter(this="repo")),
                exp.EQ(exp.Column(this=exp.to_identifier("commit")), exp.Parameter(this="commit")),
            )
        )
    )
    return update.sql(dialect="duckdb")
```

Call it once and cache, or call where you need it. No more hand-concatenated SQL.

---

## 6. Dataset contracts + Pandera/Ibis validation

You already have:

* `TableSchema` with column types and nullability. 
* `config/config/datasets/rows/*.py` TypedDicts + conversions.

Add a new module:

* `storage/pandera_schemas.py`

### 6.1 Mapping `TableSchema` → Pandera `DataFrameSchema`

Example:

```python
# storage/pandera_schemas.py

from __future__ import annotations

import pandera as pa
from pandera.typing import DataFrame
from codeintel.config.datasets import TableSchema, get_dataset_contracts_by_table_key

_DUCKDB_TO_PANDERA = {
    "BOOLEAN": pa.Bool,
    "INTEGER": pa.Int32,
    "BIGINT": pa.Int64,
    "DOUBLE": pa.Float64,
    "DECIMAL": pa.Float64,
    "DECIMAL(38,0)": pa.Int64,
    "VARCHAR": pa.String,
    "JSON": pa.String,  # or pa.Object if you want raw JSON
    "TIMESTAMP": pa.Timestamp,
    "TIMESTAMPTZ": pa.Timestamp,  # timezone-aware if needed
}

def pandera_schema_from_table(table: TableSchema) -> pa.DataFrameSchema:
    cols: dict[str, pa.Column] = {}
    for col in table.columns:
        ptype = _DUCKDB_TO_PANDERA[col.type]
        cols[col.name] = pa.Column(ptype, nullable=col.nullable)
    return pa.DataFrameSchema(cols)

_PANDERA_SCHEMAS: dict[str, pa.DataFrameSchema] = {
    contract.table_key: pandera_schema_from_table(contract.schema)
    for contract in get_dataset_contracts_by_table_key().values()
    if contract.schema is not None
}

def get_pandera_schema(table_key: str) -> pa.DataFrameSchema:
    return _PANDERA_SCHEMAS[table_key]
```

### 6.2 Using validation at boundaries

* In repositories: after `expr.to_pandas()`, validate:

```python
from codeintel.storage.pandera_schemas import get_pandera_schema

schema = get_pandera_schema("analytics.function_metrics")
df = expr.to_pandas()
validated = schema.validate(df)
return validated.to_dict("records")
```

* In analytics/graphs where you construct intermediate datasets that should match a contract, validate them before writing to DuckDB.

You can be selective:

* Use Pandera on key datasets (e.g. `core.modules`, `analytics.function_metrics`) at least in tests.
* Skip for extremely hot loops until you measure overhead.

---

## 7. Clean-up & enforcement

Finally, once the new pieces are in:

1. **Delete/retire**:

   * `ingestion/infrastructure/safe_sql.py` (no longer needed when all queries are Ibis). 
   * Most of `storage/sql/primitives.py` except what’s still used by the policy backend (or reimplement those bits in SQLGlot and retire primitives entirely).

2. **Update docs**:

   * Extend `ibis_unification.md` (or add `IBIS_SQLGLOT_PANDERA_CONVENTIONS.md`) with:

     * “No `duckdb` imports outside `storage/` and tests.”
     * “No `gateway.con.execute` outside `storage/` and tests.”
     * “All queries must be Ibis; only policy backend may call `raw_sql`.”
     * “All DDL/UPSERT SQL is generated via SQLGlot from dataset contracts.”

3. **Run a repo-wide search** after each wave:

   * `duckdb.connect` outside storage.
   * `.execute("SELECT` / `.execute("DELETE` outside storage.
   * Direct SQL strings referencing schema.table names outside config/storage/metadata.

---

If you’d like, I can next zoom in on *one* vertical slice (e.g. `analytics/cfg_dfg/*` + `storage/repositories/graphs.py`) and write a PR-style diff that:

* Replaces their `gateway.con.execute` queries with Ibis expressions,
* Adds any needed built-in UDF wrappers,
* And drops down to the policy backend only for the one or two DDL-ish operations they need.


# example implementation #

Gotcha, let’s make this nice and concrete.

Below is a **PR-style vertical slice** for:

* `analytics/cfg_dfg/dfg_core.py`
* `analytics/cfg_dfg/cfg_core.py`
* `storage/repositories/graphs.py`

The goals for this slice:

1. **All reads go through Ibis**, not `gateway.con.execute`.
2. The callgraph repo uses Ibis instead of `fetch_all_dicts`.
3. We *show* how to use the Ibis adapter (`IbisGateway`) from “normal” analytics & storage code.
4. We leave the DDL / `ensure_schema` / INSERT blobs alone in this slice (they’ll move under the DuckDB policy backend in the broader refactor).

This is meant as a template you can reuse for the rest of the codebase.

---

## 1) `analytics/cfg_dfg/dfg_core.py` – DFG reads via Ibis

### 1.1 Imports: wire in the Ibis adapter

```diff
diff --git a/analytics/cfg_dfg/dfg_core.py b/analytics/cfg_dfg/dfg_core.py
--- a/analytics/cfg_dfg/dfg_core.py
+++ b/analytics/cfg_dfg/dfg_core.py
@@ -17,7 +17,8 @@ from codeintel.analytics.compute.graphs import (
     dfg_path_lengths,
     normalize_decimal_id,
 )
-from codeintel.analytics.runtime.context import GraphContext
-from codeintel.storage.gateway import DuckDBError, StorageGateway
+from codeintel.analytics.runtime.context import GraphContext
+from codeintel.storage.gateway import DuckDBError, StorageGateway
+from codeintel.storage.ibis_adapter import IbisGateway
```

### 1.2 `load_dfg_edges` – replace raw SQL with Ibis

```diff
@@ -97,22 +98,37 @@ def load_dfg_edges(
     gateway: StorageGateway, _repo: str, _commit: str
 ) -> dict[int, list[tuple[int, int, str, str, bool, str]]]:
     """
     Load DFG edges grouped by function GOID.
@@ -105,17 +112,32 @@ def load_dfg_edges(
     Returns
     -------
     dict[int, list[tuple[int, int, str, str, bool, str]]]
         Mapping of GOID -> edge tuples.
     """
-    edges_by_fn: dict[int, list[tuple[int, int, str, str, bool, str]]] = defaultdict(list)
-    try:
-        rows: Iterable[tuple[int, int, int, str, str, bool, str]] = gateway.con.execute(
-            """
-            SELECT function_goid_h128, src_block_id, dst_block_id,
-                   src_var, dst_var, via_phi, use_kind
-            FROM graph.dfg_edges
-            """
-        ).fetchall()
-    except DuckDBError:
-        return edges_by_fn
+    edges_by_fn: dict[int, list[tuple[int, int, str, str, bool, str]]] = defaultdict(list)
+
+    # Use Ibis to load the edge rows rather than issuing raw SQL
+    ibis = IbisGateway(gateway)
+    try:
+        edges_df = (
+            ibis.table("graph.dfg_edges")
+            .select(
+                "function_goid_h128",
+                "src_block_id",
+                "dst_block_id",
+                "src_var",
+                "dst_var",
+                "via_phi",
+                "use_kind",
+            )
+            .to_pandas()
+        )
+        rows = cast(
+            "Iterable[tuple[int, int, int, str, str, bool, str]]",
+            edges_df.itertuples(index=False, name=None),
+        )
+    except DuckDBError:
+        # If the table doesn't exist yet (e.g., DFG plugin never ran),
+        # just return an empty mapping.
+        return edges_by_fn
@@ -119,7 +141,7 @@ def load_dfg_edges(
-    for fn, src_id, dst_id, src_sym, dst_sym, via_phi, use_kind in rows:
+    for fn, src_id, dst_id, src_sym, dst_sym, via_phi, use_kind in rows:
         src_idx = _parse_block_idx(src_id)
         dst_idx = _parse_block_idx(dst_id)
         if src_idx is None or dst_idx is None:
             continue
         edges_by_fn[int(fn)].append(
```

### 1.3 `dfg_function_metadata` – reuse the same Ibis join pattern

```diff
@@ -329,18 +351,35 @@ def dfg_function_metadata(
     gateway: StorageGateway, repo: str, commit: str
 ) -> dict[int, tuple[str, str | None, str | None]]:
@@ -343,16 +382,32 @@ def dfg_function_metadata(
-    rows: Iterable[tuple[object, str, str | None, str | None]] = gateway.con.execute(
-        """
-        SELECT g.goid_h128,
-               g.rel_path,
-               m.module,
-               g.qualname
-        FROM core.goids g
-        LEFT JOIN core.modules m
-          ON m.path = g.rel_path
-        WHERE g.repo = ? AND g.commit = ?
-          AND g.kind IN ('function', 'method')
-        """,
-        [repo, commit],
-    ).fetchall()
+    ibis = IbisGateway(gateway)
+    goids = ibis.table("core.goids")
+    modules = ibis.table("core.modules")
+
+    expr = (
+        goids.left_join(modules, modules.path == goids.rel_path)
+        .filter(
+            (goids.repo == repo)
+            & (goids.commit == commit)
+            & (goids.kind.isin(["function", "method"]))
+        )
+        .select(
+            goids.goid_h128,
+            goids.rel_path,
+            modules.module,
+            goids.qualname,
+        )
+    )
+
+    rows_df = expr.to_pandas()
+    rows = cast(
+        "Iterable[tuple[object, str, str | None, str | None]]",
+        rows_df.itertuples(index=False, name=None),
+    )
@@ -357,7 +412,7 @@ def dfg_function_metadata(
-    for goid_raw, rel_path, module, qualname in rows:
+    for goid_raw, rel_path, module, qualname in rows:
         goid = normalize_decimal_id(goid_raw)
         if goid is None:
             continue
         result[int(goid)] = (rel_path, module, qualname)
```

That’s the full DFG slice: **no more `gateway.con.execute`** for edge/metadata reads.

---

## 2) `analytics/cfg_dfg/cfg_core.py` – CFG reads via Ibis

### 2.1 Imports: add Ibis adapter

```diff
diff --git a/analytics/cfg_dfg/cfg_core.py b/analytics/cfg_dfg/cfg_core.py
--- a/analytics/cfg_dfg/cfg_core.py
+++ b/analytics/cfg_dfg/cfg_core.py
@@ -15,8 +15,9 @@ from codeintel.analytics.compute.graphs import (
     cfg_longest_path_length,
     cfg_reachable_nodes,
     normalize_decimal_id,
 )
 from codeintel.analytics.runtime.context import GraphContext
-from codeintel.storage.gateway import DuckDBError, StorageGateway
+from codeintel.storage.gateway import DuckDBError, StorageGateway
+from codeintel.storage.ibis_adapter import IbisGateway
```

### 2.2 `load_cfg_blocks` – replace both block + edge queries

```diff
@@ -103,18 +104,40 @@ def load_cfg_blocks(
     """
     Load CFG blocks and edges grouped by function GOID.
@@ -114,26 +137,51 @@ def load_cfg_blocks(
-    blocks_by_fn: dict[int, list[tuple[int, str, int, int]]] = defaultdict(list)
-    edges_by_fn: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
-
-    try:
-        block_rows: Iterable[tuple[int, int, str, int, int]] = gateway.con.execute(
-            """
-            SELECT function_goid_h128, block_idx, kind, in_degree, out_degree
-            FROM graph.cfg_blocks
-            """
-        ).fetchall()
-    except DuckDBError:
-        return blocks_by_fn, edges_by_fn
-    for fn, idx, kind, in_deg, out_deg in block_rows:
-        blocks_by_fn[int(fn)].append((int(idx), str(kind), int(in_deg), int(out_deg)))
-
-    try:
-        edge_rows: Iterable[tuple[int, int, int, str]] = gateway.con.execute(
-            """
-            SELECT function_goid_h128, src_block_id, dst_block_id, edge_kind
-            FROM graph.cfg_edges
-            """
-        ).fetchall()
-    except DuckDBError:
-        return blocks_by_fn, edges_by_fn
+    blocks_by_fn: dict[int, list[tuple[int, str, int, int]]] = defaultdict(list)
+    edges_by_fn: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
+
+    ibis = IbisGateway(gateway)
+
+    # Load block rows via Ibis
+    try:
+        block_df = (
+            ibis.table("graph.cfg_blocks")
+            .select(
+                "function_goid_h128",
+                "block_idx",
+                "kind",
+                "in_degree",
+                "out_degree",
+            )
+            .to_pandas()
+        )
+        block_rows = cast(
+            "Iterable[tuple[int, int, str, int, int]]",
+            block_df.itertuples(index=False, name=None),
+        )
+    except DuckDBError:
+        return blocks_by_fn, edges_by_fn
+
+    for fn, idx, kind, in_deg, out_deg in block_rows:
+        blocks_by_fn[int(fn)].append((int(idx), str(kind), int(in_deg), int(out_deg)))
+
+    # Load edge rows via Ibis
+    try:
+        edge_df = (
+            ibis.table("graph.cfg_edges")
+            .select(
+                "function_goid_h128",
+                "src_block_id",
+                "dst_block_id",
+                "edge_kind",
+            )
+            .to_pandas()
+        )
+        edge_rows = cast(
+            "Iterable[tuple[int, int, int, str]]",
+            edge_df.itertuples(index=False, name=None),
+        )
+    except DuckDBError:
+        return blocks_by_fn, edges_by_fn
@@ -137,7 +185,7 @@ def load_cfg_blocks(
-    for fn, src_id, dst_id, edge_type in edge_rows:
+    for fn, src_id, dst_id, edge_type in edge_rows:
         src_idx = parse_block_idx(src_id) if src_id is not None else None
         dst_idx = parse_block_idx(dst_id) if dst_id is not None else None
         if src_idx is None or dst_idx is None:
             continue
         edges_by_fn[int(fn)].append((src_idx, dst_idx, str(edge_type)))
```

### 2.3 `function_metadata` – same pattern as `dfg_function_metadata`

```diff
@@ -416,8 +416,8 @@ def function_metadata(
     gateway: StorageGateway, repo: str, commit: str
 ) -> dict[int, tuple[str, str | None, str | None]]:
@@ -427,16 +427,32 @@ def function_metadata(
-    rows: Iterable[tuple[object, str, str | None, str | None]] = gateway.con.execute(
-        """
-        SELECT g.goid_h128,
-               g.rel_path,
-               m.module,
-               g.qualname
-        FROM core.goids g
-        LEFT JOIN core.modules m
-          ON m.path = g.rel_path
-        WHERE g.repo = ? AND g.commit = ?
-          AND g.kind IN ('function', 'method')
-        """,
-        [repo, commit],
-    ).fetchall()
+    ibis = IbisGateway(gateway)
+    goids = ibis.table("core.goids")
+    modules = ibis.table("core.modules")
+
+    expr = (
+        goids.left_join(modules, modules.path == goids.rel_path)
+        .filter(
+            (goids.repo == repo)
+            & (goids.commit == commit)
+            & (goids.kind.isin(["function", "method"]))
+        )
+        .select(
+            goids.goid_h128,
+            goids.rel_path,
+            modules.module,
+            goids.qualname,
+        )
+    )
+
+    rows_df = expr.to_pandas()
+    rows = cast(
+        "Iterable[tuple[object, str, str | None, str | None]]",
+        rows_df.itertuples(index=False, name=None),
+    )
@@ -441,7 +457,7 @@ def function_metadata(
-    for goid_raw, rel_path, module, qualname in rows:
+    for goid_raw, rel_path, module, qualname in rows:
         goid = normalize_decimal_id(goid_raw)
         if goid is None:
             continue
         result[int(goid)] = (rel_path, module, qualname)
```

Now **all CFG/DFG reads** in this slice are Ibis-based.

> DDL-ish operations (`ensure_schema`, the big `INSERT INTO analytics.*`) are intentionally left as-is here; they’ll be routed through your `DuckDBPolicyBackend` in the broader DDL/SQLGlot refactor.

---

## 3) `storage/repositories/graphs.py` – callgraph repo via Ibis

Here we convert the two “neighbors” methods from raw SQL + `fetch_all_dicts` to Ibis API.

### 3.1 Imports: add IbisGateway

```diff
diff --git a/storage/repositories/graphs.py b/storage/repositories/graphs.py
--- a/storage/repositories/graphs.py
+++ b/storage/repositories/graphs.py
@@ -3,8 +3,10 @@ from __future__ import annotations
 
 from dataclasses import dataclass
 
-from codeintel.storage.repositories.base import BaseRepository, RowDict, fetch_all_dicts
+from codeintel.storage.repositories.base import BaseRepository, RowDict
+from codeintel.storage.ibis_adapter import IbisGateway
```

(We drop `fetch_all_dicts` because we don’t use it anymore in this repo.)

### 3.2 `get_outgoing_callgraph_neighbors` – Ibis over `docs.v_call_graph_enriched`

```diff
@@ class GraphRepository(BaseRepository):
     def get_outgoing_callgraph_neighbors(
         self, caller_goid_h128: int, *, limit: int
     ) -> list[RowDict]:
         """
         Return outgoing call edges for a caller GOID.
@@
-        sql = """
-            SELECT *
-            FROM docs.v_call_graph_enriched
-            WHERE caller_goid_h128 = ?
-              AND caller_repo = ?
-              AND caller_commit = ?
-            ORDER BY callee_qualname
-            LIMIT ?
-        """
-        return fetch_all_dicts(self.con, sql, [caller_goid_h128, self.repo, self.commit, limit])
+        ibis = IbisGateway(self.gateway)
+        t = ibis.table("docs.v_call_graph_enriched")
+
+        expr = (
+            t.filter(
+                (t.caller_goid_h128 == caller_goid_h128)
+                & (t.caller_repo == self.repo)
+                & (t.caller_commit == self.commit)
+            )
+            .order_by(t.callee_qualname)
+            .limit(limit)
+        )
+
+        df = expr.to_pandas()
+        return df.to_dict("records")
```

### 3.3 `get_incoming_callgraph_neighbors` – symmetric Ibis query

```diff
@@
     def get_incoming_callgraph_neighbors(
         self, callee_goid_h128: int, *, limit: int
     ) -> list[RowDict]:
@@
-        sql = """
-            SELECT *
-            FROM docs.v_call_graph_enriched
-            WHERE callee_goid_h128 = ?
-              AND callee_repo = ?
-              AND callee_commit = ?
-            ORDER BY caller_qualname
-            LIMIT ?
-        """
-        return fetch_all_dicts(self.con, sql, [callee_goid_h128, self.repo, self.commit, limit])
+        ibis = IbisGateway(self.gateway)
+        t = ibis.table("docs.v_call_graph_enriched")
+
+        expr = (
+            t.filter(
+                (t.callee_goid_h128 == callee_goid_h128)
+                & (t.callee_repo == self.repo)
+                & (t.callee_commit == self.commit)
+            )
+            .order_by(t.caller_qualname)
+            .limit(limit)
+        )
+
+        df = expr.to_pandas()
+        return df.to_dict("records")
```

That’s the entire callgraph repo rewritten to Ibis-based reads.

---

## 4) Notes on UDFs and policy backend in this slice

* This particular slice **doesn’t need any builtin UDFs** yet (all operations are simple selects/filters/joins), so we didn’t introduce any `@udf.scalar.builtin`. When you start doing vector ops, geospatial, or DuckDB-only scalar functions, those all go into `storage/ibis_builtins.py` as discussed.

* For DDL-ish operations in `materialize.py` (`ensure_schema`, big `INSERT INTO`, the `DELETE FROM analytics.* WHERE repo = ? AND commit = ?`), the pattern you’ll want is:

  * Move all schema creation + truncation logic into `DuckDBPolicyBackend` (SQLGlot + `backend.raw_sql`).
  * From `compute_cfg_metrics` / `compute_dfg_metrics`, call semantic methods like:

    * `policy.ensure_cfg_dfg_metric_tables()`
    * `policy.clear_cfg_metrics(repo, commit)`
    * `policy.clear_dfg_metrics(repo, commit)`

  …but that’s a slightly larger patch; this answer is focused on the read/query side template you asked for.

If you’d like, next step I can take `analytics/cfg_dfg/materialize.py` and show **exactly** how to:

* replace the three `DELETE FROM analytics.*` with small policy backend methods, and
* outline a minimal `DuckDBPolicyBackend` stub that you can grow into the full SQLGlot-driven DDL/UPSERT engine.


# policy backend illustration #

Perfect, let’s wire this through.

I’ll do exactly what you asked on one vertical slice:

* **analytics/cfg_dfg/materialize.py**
* **new storage/duckdb_policy_backend.py**

Goal:

* No more `DELETE FROM analytics.*` in analytics.
* All “clear metrics for repo+commit” behavior moves into a **small policy backend** that uses SQLGlot + `backend.raw_sql`, giving you the skeleton for the future DDL/UPSERT engine.

---

## 1. New: `storage/duckdb_policy_backend.py`

This is the minimal-but-real stub:

* Owns **DuckDB-specific** operations that don’t (yet) live as pure Ibis.
* For this slice: only “clear metrics tables for (repo, commit)”.
* Uses **SQLGlot** to generate `DELETE` statements (no hand-built SQL strings).
* Executes via **Ibis backend’s** `raw_sql` (not direct DuckDB).

```python
# storage/duckdb_policy_backend.py

"""DuckDB policy backend for non-Ibis operations.

This module centralizes all DuckDB-specific SQL that is not (yet) expressed
as an Ibis expression. In particular, it owns:

- Truncation / DELETE operations for analytics tables.
- (Later) DDL from dataset contracts via SQLGlot.
- (Later) MERGE/UPSERT patterns.

All application and analytics code should call small, semantic methods on
`DuckDBPolicyBackend` instead of issuing raw DELETE/DDL SQL directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import expressions as exp

from codeintel.storage.gateway.protocol import StorageGateway


@dataclass
class DuckDBPolicyBackend:
    """Centralized DuckDB-specific policy layer.

    Parameters
    ----------
    gateway:
        Storage gateway providing access to the underlying DuckDB connection
        and Ibis DuckDB backend.
    """

    gateway: StorageGateway

    # --- Low-level helpers -------------------------------------------------

    @property
    def backend(self):
        """Return the Ibis DuckDB backend used for executing raw SQL."""
        # `gateway.ibis.con` is an ibis.backends.duckdb.Backend
        return self.gateway.ibis.con

    def _delete_repo_commit(
        self,
        *,
        schema: str,
        table: str,
        repo: str,
        commit: str,
    ) -> None:
        """DELETE FROM schema.table WHERE repo = '<repo>' AND commit = '<commit>'.

        Uses SQLGlot to generate the SQL string instead of manual string
        concatenation, so the table/schema identifiers and literals are
        properly quoted for DuckDB.
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

        sql = delete_expr.sql(dialect="duckdb")
        # We intentionally do not expose raw_sql anywhere else.
        self.backend.raw_sql(sql)

    # --- High-level semantic methods used by analytics ---------------------

    def clear_cfg_metrics(self, *, repo: str, commit: str) -> None:
        """Clear all CFG metrics rows for a (repo, commit) scope.

        Tables cleared:
        - analytics.cfg_function_metrics
        - analytics.cfg_block_metrics
        - analytics.cfg_function_metrics_ext
        """
        for table in (
            "cfg_function_metrics",
            "cfg_block_metrics",
            "cfg_function_metrics_ext",
        ):
            self._delete_repo_commit(
                schema="analytics",
                table=table,
                repo=repo,
                commit=commit,
            )

    def clear_dfg_metrics(self, *, repo: str, commit: str) -> None:
        """Clear all DFG metrics rows for a (repo, commit) scope.

        Tables cleared:
        - analytics.dfg_function_metrics
        - analytics.dfg_block_metrics
        - analytics.dfg_function_metrics_ext
        """
        for table in (
            "dfg_function_metrics",
            "dfg_block_metrics",
            "dfg_function_metrics_ext",
        ):
            self._delete_repo_commit(
                schema="analytics",
                table=table,
                repo=repo,
                commit=commit,
            )

    # --- Placeholders / TODOs for the full engine --------------------------
    #
    # def ensure_all_tables(self) -> None:
    #     """Create all tables/indexes from dataset contracts via SQLGlot."""
    #     ...
    #
    # def upsert_from_spec(self, spec: UpsertSpec) -> None:
    #     """Run a MERGE/UPSERT expressed as a structured spec."""
    #     ...
```

This gives you a **single choke point** for “DELETE analytics metrics” today, and a place to grow DDL/UPSERT later.

---

## 2. Patch `analytics/cfg_dfg/materialize.py`

Now we make `compute_cfg_metrics` and `compute_dfg_metrics` call the policy backend instead of sprinkling DELETEs inline.

### 2.1 Import the policy backend

At the top of `analytics/cfg_dfg/materialize.py`, extend imports:

```diff
 from codeintel.analytics.cfg_dfg.dfg_core import (
@@
 )
 from codeintel.analytics.runtime.context import GraphContextSpec, resolve_graph_context
-from codeintel.storage.gateway import StorageGateway
+from codeintel.storage.gateway import StorageGateway
+from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
 from codeintel.storage.sql.builder import ensure_schema
```

### 2.2 Wire `DuckDBPolicyBackend` into `compute_cfg_metrics`

We only need to instantiate the policy once per call and then use it where the deletes used to be.

```diff
 def compute_cfg_metrics(
     gateway: StorageGateway,
     *,
     repo: str,
     commit: str,
 ) -> None:
     """Populate cfg_function_metrics and cfg_block_metrics tables."""
-    con = gateway.con
-    ensure_schema(con, "analytics.cfg_function_metrics")
-    ensure_schema(con, "analytics.cfg_block_metrics")
-    ensure_schema(con, "analytics.cfg_function_metrics_ext")
+    con = gateway.con
+    policy = DuckDBPolicyBackend(gateway)
+    ensure_schema(con, "analytics.cfg_function_metrics")
+    ensure_schema(con, "analytics.cfg_block_metrics")
+    ensure_schema(con, "analytics.cfg_function_metrics_ext")
@@
-    for fn_goid, meta in metadata.items():
-        rows = cfg_rows_for_fn(fn_goid, meta, inputs)
-        if rows is None:
-            continue
-        fn_rows.append(rows.fn_row)
-        fn_ext_rows.append(rows.ext_row)
-        block_rows.extend(rows.block_rows)
-
-    con.execute(
-        "DELETE FROM analytics.cfg_function_metrics WHERE repo = ? AND commit = ?",
-        [repo, commit],
-    )
-    con.execute(
-        "DELETE FROM analytics.cfg_function_metrics_ext WHERE repo = ? AND commit = ?",
-        [repo, commit],
-    )
-    con.execute(
-        "DELETE FROM analytics.cfg_block_metrics WHERE repo = ? AND commit = ?",
-        [repo, commit],
-    )
+    for fn_goid, meta in metadata.items():
+        rows = cfg_rows_for_fn(fn_goid, meta, inputs)
+        if rows is None:
+            continue
+        fn_rows.append(rows.fn_row)
+        fn_ext_rows.append(rows.ext_row)
+        block_rows.extend(rows.block_rows)
+
+    # Clear existing CFG metrics rows for this (repo, commit) using the
+    # centralized DuckDB policy backend instead of issuing inline DELETEs.
+    policy.clear_cfg_metrics(repo=repo, commit=commit)
@@
     if fn_rows:
         con.executemany(
             """
             INSERT INTO analytics.cfg_function_metrics (
                 function_goid_h128, repo, commit, rel_path, module, qualname,
                 cfg_block_count, cfg_edge_count, cfg_has_cycles, cfg_scc_count,
             ...
```

Behavior is identical (same three tables cleared), but:

* The function no longer knows *how* tables are cleared,
* It’s now “tell the policy backend to clear CFG metrics for this snapshot.”

### 2.3 Wire `DuckDBPolicyBackend` into `compute_dfg_metrics`

Same pattern for DFG:

```diff
 def compute_dfg_metrics(
     gateway: StorageGateway,
     *,
     repo: str,
     commit: str,
 ) -> None:
     """Populate dfg_function_metrics and dfg_block_metrics tables."""
-    con = gateway.con
-    ensure_schema(con, "analytics.dfg_function_metrics")
-    ensure_schema(con, "analytics.dfg_block_metrics")
-    ensure_schema(con, "analytics.dfg_function_metrics_ext")
+    con = gateway.con
+    policy = DuckDBPolicyBackend(gateway)
+    ensure_schema(con, "analytics.dfg_function_metrics")
+    ensure_schema(con, "analytics.dfg_block_metrics")
+    ensure_schema(con, "analytics.dfg_function_metrics_ext")
@@
-    for fn_goid, meta in metadata.items():
-        ctx = build_dfg_context(
-            DfgInputs(
-                fn_goid=fn_goid,
-                meta=meta,
-                edges=edges_by_fn.get(fn_goid, []),
-                repo=repo,
-                commit=commit,
-                now=resolved_now,
-                graph_ctx=metrics_ctx,
-            )
-        )
-        if ctx is None:
-            continue
-
-        fn_rows.append(dfg_fn_row(ctx))
-        fn_ext_rows.append(dfg_ext_row(ctx))
-        block_rows.extend(dfg_block_rows(ctx))
-
-    con.execute(
-        "DELETE FROM analytics.dfg_function_metrics WHERE repo = ? AND commit = ?",
-        [repo, commit],
-    )
-    con.execute(
-        "DELETE FROM analytics.dfg_block_metrics WHERE repo = ? AND commit = ?",
-        [repo, commit],
-    )
-    con.execute(
-        "DELETE FROM analytics.dfg_function_metrics_ext WHERE repo = ? AND commit = ?",
-        [repo, commit],
-    )
+    for fn_goid, meta in metadata.items():
+        ctx = build_dfg_context(
+            DfgInputs(
+                fn_goid=fn_goid,
+                meta=meta,
+                edges=edges_by_fn.get(fn_goid, []),
+                repo=repo,
+                commit=commit,
+                now=resolved_now,
+                graph_ctx=metrics_ctx,
+            )
+        )
+        if ctx is None:
+            continue
+
+        fn_rows.append(dfg_fn_row(ctx))
+        fn_ext_rows.append(dfg_ext_row(ctx))
+        block_rows.extend(dfg_block_rows(ctx))
+
+    # Clear existing DFG metrics rows for this (repo, commit) using the
+    # centralized DuckDB policy backend instead of inline DELETEs.
+    policy.clear_dfg_metrics(repo=repo, commit=commit)
@@
     if fn_rows:
         con.executemany(
             """
             INSERT INTO analytics.dfg_function_metrics (
                 function_goid_h128, repo, commit, rel_path, module, qualname,
                 ...
```

Again, same semantics, but deletes are now “conceptual operations” on the policy backend.

---

## 3. Optional: tiny tests for the policy backend

If you want to verify this slice end-to-end, you can add a small test in `tests/storage/test_duckdb_policy_backend.py`:

```python
from __future__ import annotations

import duckdb

from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.gateway.connection import create_gateway_for_connection


def test_clear_cfg_metrics_clears_all_tables(tmp_path) -> None:
    # Simple in-memory DB for the test
    con = duckdb.connect(str(tmp_path / "test.duckdb"))
    gateway = create_gateway_for_connection(con)  # whatever factory you use

    # Seed 1 row in each table
    con.execute(
        "CREATE SCHEMA IF NOT EXISTS analytics;"
        "CREATE TABLE analytics.cfg_function_metrics (repo TEXT, commit TEXT);"
        "CREATE TABLE analytics.cfg_block_metrics (repo TEXT, commit TEXT);"
        "CREATE TABLE analytics.cfg_function_metrics_ext (repo TEXT, commit TEXT);"
    )
    for name in (
        "analytics.cfg_function_metrics",
        "analytics.cfg_block_metrics",
        "analytics.cfg_function_metrics_ext",
    ):
        con.execute(f"INSERT INTO {name} VALUES ('r', 'c')")

    policy = DuckDBPolicyBackend(gateway)
    policy.clear_cfg_metrics(repo="r", commit="c")

    for name in (
        "analytics.cfg_function_metrics",
        "analytics.cfg_block_metrics",
        "analytics.cfg_function_metrics_ext",
    ):
        assert con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] == 0
```

(Adjust the gateway factory call to whatever you actually use.)

This isn’t strictly necessary for the design, but it gives you confidence that:

* SQLGlot is generating valid DuckDB DELETE statements.
* `backend.raw_sql` is wired correctly.

---

## 4. How this scales out

This slice is intentionally small but demonstrates the pattern you want everywhere:

* **Analytics code** now talks in conceptual operations:

  * “ensure schemas exist” (for now via `ensure_schema`)
  * “clear metrics for this snapshot” (`policy.clear_*_metrics`)
  * “insert rows”
* **The policy backend** is the only place where:

  * SQLGlot is used to build non-query SQL,
  * `backend.raw_sql` is invoked.

From here, you can:

* Add more methods to `DuckDBPolicyBackend`:

  * `ensure_all_schemas_from_contracts()`
  * `ensure_indexes_for_table(table_key)`
  * `run_upsert(spec: UpsertSpec)`
* Gradually move other DDL/DDL-ish code (schema/ddl.py, macros, metadata bootstrap) into this same policy layer, all backed by dataset contracts + SQLGlot.

If you’d like, next step I can sketch the **DDL half**: a `create_table_from_contract` and `create_index_from_contract` in `DuckDBPolicyBackend` using your `config.dataset_contract.TableSchema`, so you can start deleting the hand-rolled DDL from `storage/schema/ddl.py`.


# DDL integration #

Love it, let’s finish the “DDL half” properly.

I’ll give you:

1. New DDL methods on **`DuckDBPolicyBackend`**:

   * `create_table_from_schema(table: TableSchema, …)`
   * `create_indexes_from_schema(table: TableSchema)`
   * `ensure_all_schemas(...)` that replaces `TABLE_DDL` / `INDEX_DDL` in `storage/schema/ddl.py`
2. A thin wrapper in **`storage/schema/ddl.py`** that now just delegates to the policy backend, so you can start deleting the hand-rolled DDL.

I’ll keep it concrete but you can absolutely tweak details (e.g. how strict `drop_existing` is) as you wire it in.

---

## 1. Extend `DuckDBPolicyBackend` with DDL helpers

We’ll build on the `DuckDBPolicyBackend` you already have (with `clear_cfg_metrics` / `clear_dfg_metrics`), and add DDL-ish helpers using **SQLGlot** + your `TableSchema` from `config.config.datasets.primitives`.

```python
# storage/duckdb_policy_backend.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import sqlglot
from sqlglot import expressions as exp

from codeintel.config.datasets import (
    TableSchema,
    get_dataset_contracts_by_table_key,
)
from codeintel.storage.gateway.protocol import StorageGateway


# These match your existing ddl.py behavior
SCHEMAS: tuple[str, ...] = ("core", "graph", "analytics", "docs", "build", "metadata")

# This comes from storage/schema/ddl.py; we preserve it here
_TABLE_CREATION_DENYLIST: set[str] = {"docs.v_validation_summary"}


@dataclass
class DuckDBPolicyBackend:
    """Centralized DuckDB-specific policy layer.

    - Owns non-Ibis SQL (DDL, DELETE, MERGE, etc.)
    - Uses SQLGlot to build SQL strings.
    - Executes via the Ibis DuckDB backend's raw_sql.

    All application code should call these small semantic methods instead of
    constructing SQL.
    """

    gateway: StorageGateway

    # ------------------------------------------------------------------ #
    # Low-level helpers
    # ------------------------------------------------------------------ #

    @property
    def backend(self):
        """Return the Ibis DuckDB backend for this gateway."""
        # `gateway.ibis.con` is an ibis.backends.duckdb.Backend
        return self.gateway.ibis.con

    def _run(self, expr: exp.Expression) -> None:
        """Compile a SQLGlot expression to DuckDB SQL and execute."""
        sql = expr.sql(dialect="duckdb")
        self.backend.raw_sql(sql)

    def _run_many(self, exprs: Iterable[exp.Expression]) -> None:
        for e in exprs:
            self._run(e)

    # ------------------------------------------------------------------ #
    # CREATE SCHEMA
    # ------------------------------------------------------------------ #

    def create_schema_if_not_exists(self, schema_name: str) -> None:
        """CREATE SCHEMA IF NOT EXISTS <schema_name>."""
        # SQLGlot doesn't have a highly ergonomic SCHEMA builder,
        # so we parse a minimal statement and re-emit it.
        stmt = sqlglot.parse_one(
            f"CREATE SCHEMA IF NOT EXISTS {sqlglot.to_identifier(schema_name)}",
            dialect="duckdb",
        )
        self._run(stmt)

    # ------------------------------------------------------------------ #
    # CREATE TABLE / INDEX from TableSchema
    # ------------------------------------------------------------------ #

    def _column_def_expr(self, col) -> exp.ColumnDef:
        """Convert config.datasets.primitives.Column -> SQLGlot ColumnDef."""
        dtype = exp.DataType.build(col.type)  # e.g. "INTEGER", "DECIMAL(38,0)", ...
        col_def = exp.ColumnDef(
            this=exp.to_identifier(col.name),
            kind=dtype,
        )
        constraints: list[exp.Expression] = []
        if not col.nullable:
            constraints.append(exp.NotNull())
        if constraints:
            col_def.set("constraints", constraints)
        return col_def

    def _primary_key_exprs(self, table: TableSchema) -> list[exp.Expression]:
        """Build PRIMARY KEY(...) constraint if present."""
        if not table.primary_key:
            return []
        # PRIMARY KEY (col1, col2, ...)
        pk = exp.PrimaryKey(
            expressions=[
                exp.Column(this=exp.to_identifier(name))
                for name in table.primary_key
            ]
        )
        return [pk]

    def _create_table_exprs(
        self,
        table: TableSchema,
        *,
        drop_existing: bool,
        if_not_exists: bool,
    ) -> list[exp.Expression]:
        """Build [DROP TABLE IF EXISTS] + CREATE TABLE expressions."""
        table_ref = exp.Table(
            this=exp.to_identifier(table.name),
            db=exp.to_identifier(table.schema),
        )

        # Columns + optional PRIMARY KEY constraint
        cols: list[exp.Expression] = [
            self._column_def_expr(col) for col in table.columns
        ]
        cols.extend(self._primary_key_exprs(table))
        schema_expr = exp.Schema(expressions=cols)

        create = exp.Create(
            this=table_ref,
            kind="TABLE",
            expression=schema_expr,
        )
        if if_not_exists:
            create.set("exists", True)  # IF NOT EXISTS

        exprs: list[exp.Expression] = []
        if drop_existing:
            drop = exp.Drop(this=table_ref, kind="TABLE")
            drop.set("exists", True)  # IF EXISTS
            exprs.append(drop)

        exprs.append(create)
        return exprs

    def create_table_from_schema(
        self,
        table: TableSchema,
        *,
        drop_existing: bool = False,
        if_not_exists: bool = False,
    ) -> None:
        """Create a DuckDB table from a TableSchema via SQLGlot.

        Parameters
        ----------
        table
            TableSchema describing the DuckDB table.
        drop_existing
            If True, emit a DROP TABLE IF EXISTS before CREATE TABLE.
        if_not_exists
            If True, emit CREATE TABLE IF NOT EXISTS instead of CREATE TABLE.
        """
        exprs = self._create_table_exprs(
            table,
            drop_existing=drop_existing,
            if_not_exists=if_not_exists,
        )
        self._run_many(exprs)

    def create_indexes_from_schema(self, table: TableSchema) -> None:
        """Create all secondary indexes defined on a TableSchema."""
        for index in table.indexes:
            table_ref = exp.Table(
                this=exp.to_identifier(table.name),
                db=exp.to_identifier(table.schema),
            )
            # CREATE [UNIQUE] INDEX IF NOT EXISTS name ON schema.table (cols…)
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
            if index.unique:
                create.set("unique", True)

            # SQLGlot models the "ON table(...)" as the "on" arg
            create.set("on", table_ref)

            self._run(create)

    # ------------------------------------------------------------------ #
    # Bulk: apply all known schemas + indexes
    # ------------------------------------------------------------------ #

    def ensure_all_schemas(
        self,
        *,
        drop_existing: bool,
        extra_ddl: Iterable[exp.Expression] | None = None,
    ) -> None:
        """Create all known schemas/tables/indexes from dataset contracts.

        This replaces TABLE_DDL / INDEX_DDL in storage/schema/ddl.py.

        Parameters
        ----------
        drop_existing
            - True: destructive mode, DROP TABLE IF EXISTS + CREATE TABLE
            - False: additive mode, CREATE TABLE IF NOT EXISTS
        extra_ddl
            Optional list of extra SQLGlot expressions to run afterwards
            (e.g., one-off migration DDL).
        """
        # 1) Ensure logical schemas exist
        for schema_name in SCHEMAS:
            self.create_schema_if_not_exists(schema_name)

        # 2) Create all tables + indexes from dataset contracts
        contracts = get_dataset_contracts_by_table_key()
        for table_key, contract in contracts.items():
            table_schema = contract.schema
            if table_schema is None:
                # Views (e.g. docs.v_function_summary) are handled elsewhere
                continue
            if table_key in _TABLE_CREATION_DENYLIST:
                continue

            self.create_table_from_schema(
                table_schema,
                drop_existing=drop_existing,
                if_not_exists=not drop_existing,
            )
            self.create_indexes_from_schema(table_schema)

        # 3) Any extra one-off DDL (rare)
        if extra_ddl:
            self._run_many(extra_ddl)
```

### What this gives you

* A single place where **DDL is derived from `TableSchema` → SQLGlot → DuckDB**.
* No more hand-built `TABLE_DDL`, `INDEX_DDL`, etc. in `storage/schema/ddl.py`.
* Two execution modes:

  * `drop_existing=True` for the old “apply_all_schemas (destructive)” behavior.
  * `drop_existing=False` for `ensure_schemas_preserve` behavior.

You can expand this later with:

* `ensure_all_views_from_contracts(...)`
* `upsert_from_spec(...)`

using the same SQLGlot primitives.

---

## 2. Thin wrapper in `storage/schema/ddl.py`

Now you can simplify `storage/schema/ddl.py` so it’s basically a shim over `DuckDBPolicyBackend`.

You *can* keep the old functions around for now but have them delegate to the new policy methods.

Here’s one way to do it (keeping your existing signatures, but using the policy backend under the hood).

```python
# storage/schema/ddl.py

"""Schema bootstrap helpers thinly wrapping DuckDBPolicyBackend.

This replaces the old TABLE_DDL / INDEX_DDL string constants with
SQLGlot-based DDL generation inside DuckDBPolicyBackend.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from duckdb import DuckDBPyConnection

from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import StorageGateway


def _policy_from_connection(con: DuckDBPyConnection) -> DuckDBPolicyBackend:
    """Temporary helper to build a policy backend from a bare DuckDB connection.

    In the long term, you'll call DuckDBPolicyBackend with a real StorageGateway
    (which already has ibis wiring). For now, this wraps the connection just
    enough to satisfy the protocol.
    """
    import ibis

    # Build an Ibis DuckDB backend bound to the existing connection
    backend = ibis.duckdb.connect(connection=con)

    class _EphemeralGateway:
        def __init__(self, con, backend):
            self.con = con
            self.ibis = type("IbisWrapper", (), {"con": backend})()

    gateway = _EphemeralGateway(con, backend)
    return DuckDBPolicyBackend(gateway=gateway)  # type: ignore[arg-type]


def apply_all_schemas(
    con: DuckDBPyConnection,
    extra_ddl: Iterable[str] | None = None,
) -> None:
    """Create all known tables and indexes (destructive).

    This matches the previous behavior:

    - Ensures logical schemas exist.
    - DROP TABLE IF EXISTS + CREATE TABLE for every dataset table.
    - CREATE INDEX IF NOT EXISTS for every secondary index.
    - Executes any extra DDL strings at the end.

    Implementation is now delegated to DuckDBPolicyBackend with
    SQLGlot-based DDL generation.
    """
    policy = _policy_from_connection(con)
    policy.ensure_all_schemas(drop_existing=True, extra_ddl=None)

    # Preserve extra_ddl hook for now: parse & run via SQLGlot
    if extra_ddl:
        import sqlglot

        exprs = [sqlglot.parse_one(stmt, dialect="duckdb") for stmt in extra_ddl]
        policy._run_many(exprs)


def ensure_schemas_preserve(
    con: DuckDBPyConnection,
    extra_ddl: Iterable[str] | None = None,
) -> None:
    """Ensure schemas/tables/indexes exist without dropping existing data.

    - CREATE SCHEMA IF NOT EXISTS
    - CREATE TABLE IF NOT EXISTS
    - CREATE [UNIQUE] INDEX IF NOT EXISTS

    This replaces TABLE_DDL_IF_NOT_EXISTS and INDEX_DDL.
    """
    policy = _policy_from_connection(con)
    policy.ensure_all_schemas(drop_existing=False, extra_ddl=None)

    if extra_ddl:
        import sqlglot

        exprs = [sqlglot.parse_one(stmt, dialect="duckdb") for stmt in extra_ddl]
        policy._run_many(exprs)
```

A few notes:

* `_policy_from_connection` is just a **bridge** until you’ve fully re-centered everything on `StorageGateway` + `gateway.ibis`. Once you have a gateway at connection time, you can delete this helper and call `DuckDBPolicyBackend(gateway)` directly.
* Extra `extra_ddl` is still accepted; now it goes through SQLGlot parsing so you continue to have a single execution path (`backend.raw_sql`).

At this point, you can delete:

* `_build_table_ddl`
* `_build_table_ddl_if_not_exists`
* `_build_index_ddl`
* `TABLE_DDL`, `TABLE_DDL_IF_NOT_EXISTS`, `INDEX_DDL`
* The old `create_schemas` function (its behavior is folded into `ensure_all_schemas`).

And your DDL story is now:

> Dataset contracts → `TableSchema` → SQLGlot → DuckDB via `backend.raw_sql`, all through `DuckDBPolicyBackend`.

---

If you’d like, the next natural extension is:

* `ensure_all_views_from_contracts()` using **Ibis** for view creation (since views *are* query IR), so you unify:

  * Tables & indexes = `TableSchema` → SQLGlot → policy backend.
  * Views = Ibis `Expr` → `backend.create_view(...)`.

That would fully close the loop on “dataset_contract + Ibis/SQLGlot drive *all* physical schema.”


# Dataset_contract + Ibis/SQLGlot drive all physical schema. #

Nice, let’s close the loop and make views first-class citizens of the “contracts → IR → DB” story.

I’ll give you:

1. A **view registry** for Ibis IR (`storage/views/ibis_registry.py`)
2. An `ensure_all_views` method on **`DuckDBPolicyBackend`**
3. A small update to your **schema bootstrap** so “apply schemas” = *tables via SQLGlot* + *views via Ibis*

This is all pattern/code you can drop into your repo and then incrementally fill out with your existing view definitions.

---

## 1. Centralize Ibis view definitions in a registry

Right now, you already have Ibis-style views in something like `storage/views/ibis_views.py` (e.g. `docs.v_function_summary` built via Ibis). The first step is to **make those definitions discoverable and keyed by table_key**.

### 1.1 New module: `storage/views/ibis_registry.py`

This is your canonical “view IR directory”:

```python
# storage/views/ibis_registry.py

"""Registry of Ibis-based view definitions.

Each entry maps a dataset table_key (e.g. "docs.v_function_summary")
to a builder function that takes an Ibis gateway and returns a
TableExpr representing the view definition.

DuckDBPolicyBackend will use this registry to materialize views.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Dict, Protocol

import ibis
from ibis.expr.types import Table as IbisTable

# You already have IbisGateway; re-use that
from codeintel.storage.ibis_adapter import IbisGateway


class ViewBuilder(Protocol):
    def __call__(self, ibis_gateway: IbisGateway) -> IbisTable: ...


# ---------------------------------------------------------------------------
# Example view definitions
# (You’ll replace / extend these with your real ones.)
# ---------------------------------------------------------------------------

def build_docs_v_function_summary(ibis_gateway: IbisGateway) -> IbisTable:
    """docs.v_function_summary — example Ibis view.

    In your real code, this should match whatever you currently have
    in storage/views/ibis_views.py for this view.
    """
    con = ibis_gateway.con  # ibis DuckDB backend

    modules = con.table("core.modules")
    metrics = con.table("analytics.function_metrics")  # example

    expr = (
        metrics.left_join(
            modules,
            (modules.repo == metrics.repo)
            & (modules.commit == metrics.commit)
            & (modules.rel_path == metrics.rel_path),
        )
        .select(
            metrics.repo,
            metrics.commit,
            metrics.rel_path,
            modules.module,
            metrics.qualname,
            metrics.cfg_block_count,
            metrics.cfg_edge_count,
        )
    )

    return expr


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

VIEW_BUILDERS: Dict[str, ViewBuilder] = {
    # table_key -> builder
    # This table_key *must* match the dataset_contract.table_key
    # and the physical name `schema.view_name` in DuckDB.
    "docs.v_function_summary": build_docs_v_function_summary,

    # Add additional views here:
    # "docs.v_ide_hints": build_docs_v_ide_hints,
    # "analytics.v_function_risks": build_analytics_v_function_risks,
}
```

Key points:

* **Keyed by dataset table_key** (`schema.view_name` string).
* Each builder returns an **Ibis TableExpr** representing the view.
* This isolates all view logic into one module, so both humans and LLMs know where to look.

You’ll go back and move your existing Ibis view definitions (currently in `storage/views/*.py`) into this registry file (or import them here).

---

## 2. Add `ensure_all_views` to `DuckDBPolicyBackend`

Now we extend the policy backend so it knows how to:

* Look at dataset contracts,
* Identify which contracts correspond to views,
* Call the matching builder from `VIEW_BUILDERS`,
* Create/replace the view in DuckDB using Ibis.

### 2.1 Extend the policy backend

In `storage/duckdb_policy_backend.py`, add:

```python
from codeintel.storage.views.ibis_registry import VIEW_BUILDERS, ViewBuilder
from ibis.expr.types import Table as IbisTable
```

Then add the ensure-views method(s):

```python
    # ------------------------------------------------------------------ #
    # Views: dataset_contracts + Ibis IR -> DuckDB views
    # ------------------------------------------------------------------ #

    def _create_or_replace_view(
        self,
        *,
        table_key: str,
        expr: IbisTable,
        overwrite: bool,
    ) -> None:
        """Create or replace a view from an Ibis expression.

        Parameters
        ----------
        table_key
            Fully-qualified name, e.g. "docs.v_function_summary".
        expr
            Ibis table expression representing the view definition.
        overwrite
            If True, CREATE OR REPLACE VIEW.
            If False, CREATE VIEW IF NOT EXISTS (best effort).
        """
        backend = self.backend  # ibis DuckDB backend

        schema, name = table_key.split(".", 1)
        full_name = f"{schema}.{name}"

        # Ibis DuckDB backend typically exposes create_view(name, obj, overwrite=True)
        # If your version differs, adjust this call accordingly.
        backend.create_view(
            full_name,
            expr,
            overwrite=overwrite,
        )

    def ensure_all_views(
        self,
        *,
        overwrite: bool = True,
        strict: bool = True,
    ) -> None:
        """Materialize all known Ibis views from dataset contracts.

        - Finds all dataset contracts that represent views (no TableSchema).
        - Looks up a corresponding Ibis view builder in VIEW_BUILDERS.
        - Calls `create_view` on the Ibis DuckDB backend.

        Parameters
        ----------
        overwrite
            If True, use CREATE OR REPLACE VIEW semantics.
            If False, use CREATE VIEW IF NOT EXISTS semantics.
        strict
            If True, raise if a view contract has no registered builder.
            If False, silently skip unregistered view contracts.
        """
        contracts = get_dataset_contracts_by_table_key()

        for table_key, contract in contracts.items():
            table_schema = contract.schema
            if table_schema is not None:
                # It’s a table, not a view; handled by ensure_all_schemas
                continue

            # This dataset is a view; we require a builder for it.
            builder: ViewBuilder | None = VIEW_BUILDERS.get(table_key)
            if builder is None:
                if strict:
                    raise KeyError(
                        f"No Ibis view builder registered for view dataset '{table_key}'"
                    )
                # Non-strict mode: log and skip.
                # (You can wire in logging here if you want.)
                continue

            expr = builder(self.gateway.ibis)
            self._create_or_replace_view(
                table_key=table_key,
                expr=expr,
                overwrite=overwrite,
            )
```

Notes:

* We’re assuming your **view dataset contracts** use `schema is None` (or some other convention) to indicate “this is a view, not a base table.” If your contracts have an explicit `is_view` flag instead, just switch the condition accordingly.
* `strict=True` is *exactly* the kind of invariant you’ll like: it forces you to have a view builder for every view contract, and will scream if you forget to wire one.

---

## 3. Wire views into schema bootstrap

Now we modify your schema bootstrap code so that whenever we “apply all schemas”, we get:

* **Tables + indexes** from dataset contracts via SQLGlot (already covered), and
* **Views** from view contracts via this Ibis registry.

### 3.1 Update `ensure_all_schemas` to call `ensure_all_views`

In `DuckDBPolicyBackend.ensure_all_schemas` (the DDL method we wrote earlier), we add a call at the end:

```python
    def ensure_all_schemas(
        self,
        *,
        drop_existing: bool,
        extra_ddl: Iterable[exp.Expression] | None = None,
    ) -> None:
        """Create all known schemas/tables/indexes from dataset contracts."""
        # 1) Ensure logical schemas exist
        for schema_name in SCHEMAS:
            self.create_schema_if_not_exists(schema_name)

        # 2) Create all tables + indexes from dataset contracts
        contracts = get_dataset_contracts_by_table_key()
        for table_key, contract in contracts.items():
            table_schema = contract.schema
            if table_schema is None:
                # View dataset; handled separately in ensure_all_views
                continue
            if table_key in _TABLE_CREATION_DENYLIST:
                continue

            self.create_table_from_schema(
                table_schema,
                drop_existing=drop_existing,
                if_not_exists=not drop_existing,
            )
            self.create_indexes_from_schema(table_schema)

        # 3) Any extra one-off DDL
        if extra_ddl:
            self._run_many(extra_ddl)

        # 4) Create all views from Ibis view registry
        #    - Overwrite views when in destructive mode (drop_existing=True).
        #    - Otherwise, only create views that are missing.
        self.ensure_all_views(
            overwrite=drop_existing,
            strict=True,
        )
```

Semantics:

* “Destructive” apply (`drop_existing=True`):

  * Drop+create tables.
  * Recreate indexes.
  * **CREATE OR REPLACE VIEW** for all registered views.
* “Preserve” apply (`drop_existing=False`):

  * CREATE TABLE IF NOT EXISTS.
  * CREATE INDEX IF NOT EXISTS.
  * **CREATE VIEW IF NOT EXISTS** for all registered views.

### 3.2 `storage/schema/ddl.py` remains a thin shim

You already have `apply_all_schemas` and `ensure_schemas_preserve` delegating to `DuckDBPolicyBackend.ensure_all_schemas`. With the change above, those now implicitly also materialize views.

For clarity, you can update docstrings in `storage/schema/ddl.py`:

```python
def apply_all_schemas(
    con: DuckDBPyConnection,
    extra_ddl: Iterable[str] | None = None,
) -> None:
    """Create all known tables, indexes, and views (destructive)."""
    ...
```

and

```python
def ensure_schemas_preserve(
    con: DuckDBPyConnection,
    extra_ddl: Iterable[str] | None = None,
) -> None:
    """Ensure tables, indexes, and views exist without dropping data."""
    ...
```

---

## 4. How to migrate your existing views into this pattern

You already have Ibis view definitions scattered under `storage/views/` (docs views, analytics views, perhaps IDE views). Port them like this:

1. **Pick a view** (e.g. `docs.v_ide_hints`).

2. Move its Ibis expression builder into `ibis_registry.py`:

   ```python
   def build_docs_v_ide_hints(ibis_gateway: IbisGateway) -> IbisTable:
       con = ibis_gateway.con
       modules = con.table("core.modules")
       # whatever logic you already had
       expr = (
           modules
           .select(
               modules.repo,
               modules.commit,
               modules.rel_path,
               modules.module,
               # …
           )
       )
       return expr
   ```

3. Add it to `VIEW_BUILDERS`:

   ```python
   VIEW_BUILDERS = {
       "docs.v_function_summary": build_docs_v_function_summary,
       "docs.v_ide_hints": build_docs_v_ide_hints,
       # ...
   }
   ```

4. Ensure there is a **dataset_contract** entry with `table_key="docs.v_ide_hints"` and `schema=None` (or `is_view=True`), so `ensure_all_views` sees it.

5. Delete any old “ad-hoc view creation” code, e.g.:

   * `storage/views/docs_views.py` that previously did `con.execute("CREATE VIEW ...")` or `con.create_view(...)`.
   * Replace it with a call to `apply_all_schemas` / `ensure_schemas_preserve` so tables **and** views are created consistently.

---

## 5. End-state mental model

Once you’ve wired this in across your dataset universe, your schema pipeline is:

* **Inputs:**

  * `config.dataset_contract` (TableSchema + view datasets),
  * `storage/views/ibis_registry.VIEW_BUILDERS`.

* **Engine:**

  * `DuckDBPolicyBackend`:

    * Tables + indexes: `TableSchema` → SQLGlot → `backend.raw_sql`.
    * Views: dataset contracts (view) + `VIEW_BUILDERS` → Ibis `TableExpr` → `backend.create_view`.

* **Bootstrap entrypoints:**

  * `storage/schema/ddl.apply_all_schemas(con)` (destructive)
  * `storage/schema/ddl.ensure_schemas_preserve(con)` (additive)

No other part of the codebase is allowed to:

* Execute DDL directly,
* Know about view DDL shapes,
* Or build SQL strings for tables/views.

That’s exactly the “dataset_contract + Pandera/Ibis/SQLGlot drive all physical schema” world you were aiming for.

If you want to go one step further after this, the natural extension is:

* For each **contract**, generate a **Pandera DataFrameSchema** (we sketched this earlier) and:

  * validate Ibis-materialized data against it at the edges (repositories, output datasets),
  * giving you end-to-end “contract → schema → physical DDL → runtime validation” for both tables *and* views.
