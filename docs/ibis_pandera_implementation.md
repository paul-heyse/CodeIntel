
# ibis plus pandera implementation overview #

Nice, this is a fun pivot. Let me lay out a structured, “drop-in over time” plan that makes:

* **Ibis** the *standard query / data access* interface to DuckDB, and
* **Pandera** the *standard schema / validation* layer for tabular data,

all while respecting your current storage / serving architecture.

I’ll focus on:

* foundational plumbing (storage & serving),
* representative slices in analytics/ingestion,
* and some **advanced patterns** that really pay off in your architecture.

---

## 0. High-level design: where Ibis & Pandera “live”

Given your current stack:

* **DuckDB** sits behind `StorageGateway` & repositories.
* **Dataset contracts** live in `config.config.datasets.*` and `storage.gateway.rows.*`.
* **Serving** uses DuckDB via `DuckDBQueryService` + `DatasetQueryLayer` etc.
* **Analytics & graphs** are mostly Python compute + raw SQL or duckdb relations.

The new stance:

1. **All read-side access** to relational data should *prefer* Ibis expressions over raw SQL or DuckDBRelation, especially in:

   * `storage.repository` methods,
   * `serving.backend.*` query layers,
   * any new analytics views / slice queries.

2. **All schema definitions & validations** for table-like data should go through Pandera where feasible:

   * dataset contracts → Pandera `DataFrameSchema` / `DataFrameModel`,
   * analytics outputs (e.g. `analytics.function_metrics`) validated at boundaries,
   * tests and CI using the same schema definitions.

3. **DuckDB remains the physical storage engine**; Ibis is the semantic query front-end, Pandera is the semantic schema front-end.

---

## 1. Ibis foundation: add an Ibis façade on top of StorageGateway

### 1.1 Add a core Ibis adapter module

Create a new module:

`storage/storage/ibis_adapter.py`

Purpose: take your existing `StorageGateway` (DuckDB connection + table accessors) and expose an Ibis connection tied to it.

```python
# storage/storage/ibis_adapter.py

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import ibis
import ibis.expr.types as it

from codeintel.storage.gateway.protocol import StorageGateway

if TYPE_CHECKING:
    from ibis.backends.duckdb import Backend as DuckDBBackend


class IbisGateway:
    """Ibis-backed view over a StorageGateway.

    This wraps a StorageGateway's DuckDB connection in an Ibis backend,
    and provides convenience methods for retrieving tables/views as Ibis
    expressions. This becomes the standard entrypoint for query building.
    """

    def __init__(self, gateway: StorageGateway) -> None:
        self._gateway = gateway

    @cached_property
    def con(self) -> DuckDBBackend:
        """Return an Ibis backend bound to the gateway's DuckDB connection."""
        # Ibis supports connecting to an existing DuckDB connection:
        # ibis.duckdb.connect(con=duckdb.connect(...)) :contentReference[oaicite:0]{index=0}
        return ibis.duckdb.connect(con=self._gateway.con)

    def table(self, table_name: str) -> it.Table:
        """Return an Ibis table expression for a fully qualified table name."""
        return self.con.table(table_name)

    def view(self, view_name: str) -> it.Table:
        """Alias for table(); semantically for views."""
        return self.table(view_name)

    def sql(self, raw_sql: str) -> it.Table:
        """Run raw SQL and get an Ibis table expression."""
        return self.con.sql(raw_sql)
```

### 1.2 Wire it into StorageGateway

You already have `StorageGateway` protocol with `.con` and `.table(name) -> DuckDBRelation`. We don’t want to break that; just **add** an Ibis lens.

In `storage/storage/gateway/protocol.py`, extend the protocol:

```python
# storage/storage/gateway/protocol.py

from typing import Protocol, TYPE_CHECKING
import duckdb

if TYPE_CHECKING:
    import ibis
    from codeintel.storage.ibis_adapter import IbisGateway

class StorageGateway(Protocol):
    ...
    @property
    def con(self) -> DuckDBConnection: ...

    def table(self, name: str) -> DuckDBRelation: ...

    # NEW: ibis view
    @property
    def ibis(self) -> "IbisGateway": ...
```

In the main gateway factory (likely `storage/storage/gateway/factory.py` or `connection.py`), add:

```python
# storage/storage/gateway/__init__.py (or wherever you create concrete gateways)

from codeintel.storage.ibis_adapter import IbisGateway

class DuckDBStorageGateway:
    def __init__(self, con: DuckDBConnection, ...) -> None:
        self._con = con
        self._ibis = IbisGateway(self)

    @property
    def con(self) -> DuckDBConnection:
        return self._con

    @property
    def ibis(self) -> IbisGateway:
        return self._ibis

    def table(self, name: str) -> DuckDBRelation:
        return self._con.table(name)
```

Result:

* Anywhere you have a `StorageGateway` you now also have an **Ibis backend** at `gateway.ibis.con` and a table accessor at `gateway.ibis.table("schema.table")`.

---

## 2. Pandera foundation: dataset contracts → Pandera schemas

You already have:

* `config/config/datasets/rows/*.py` – TypedDict row shapes + serializers.
* `storage/gateway/rows/*.py` – dataclasses for row insertion.

Let’s build a Pandera schema registry on top.

### 2.1 Create a Pandera schema module for dataset contracts

Add:

`storage/storage/pandera_schemas.py`

```python
# storage/storage/pandera_schemas.py

from __future__ import annotations

from typing import Any, Mapping

import pandera.pandas as pa
from pandera import DataFrameSchema, Column, Check

from codeintel.config.datasets.rows.analytics import FunctionMetricsRow, FunctionTypesRow
from codeintel.config.datasets.rows.graph import CallGraphNodeRow, CallGraphEdgeRow
# ... import other TypedDicts as needed

SchemaMap = Mapping[str, DataFrameSchema]

# Helper to map simple Python types to Pandera columns
def _col(type_: type[Any], **kwargs: Any) -> Column:
    return Column(type_, **kwargs)


FUNCTION_METRICS_SCHEMA = DataFrameSchema(
    {
        "repo": _col(str),
        "commit": _col(str),
        "rel_path": _col(str),
        "function_goid_h128": _col(int),
        "loc": _col(int, Check.ge(0)),
        "cyclomatic_complexity": _col(int, Check.ge(0)),
        # ... continue mapping from FunctionMetricsRow fields
    },
    strict=True,
    coerce=True,
)

FUNCTION_TYPES_SCHEMA = DataFrameSchema(
    {
        "repo": _col(str),
        "commit": _col(str),
        "rel_path": _col(str),
        "function_goid_h128": _col(int),
        "param_name": _col(str),
        "annotation": _col(str),
        # ...
    },
    strict=True,
    coerce=True,
)

CALLGRAPH_NODES_SCHEMA = DataFrameSchema(
    {
        "repo": _col(str),
        "commit": _col(str),
        "function_goid_h128": _col(int),
        "rel_path": _col(str),
        "qualname": _col(str),
        "kind": _col(str),
        # ...
    },
    strict=True,
    coerce=True,
)

CALLGRAPH_EDGES_SCHEMA = DataFrameSchema(
    {
        "repo": _col(str),
        "commit": _col(str),
        "caller_goid_h128": _col(int),
        "callee_goid_h128": _col(int),
        # ...
    },
    strict=True,
    coerce=True,
)


DATASET_SCHEMAS: dict[str, DataFrameSchema] = {
    "analytics.function_metrics": FUNCTION_METRICS_SCHEMA,
    "analytics.function_types": FUNCTION_TYPES_SCHEMA,
    "graph.call_graph_nodes": CALLGRAPH_NODES_SCHEMA,
    "graph.call_graph_edges": CALLGRAPH_EDGES_SCHEMA,
    # ... add more as you go
}
```

You can bootstrap these schemas by:

* generating them using `pandera.infer_schema(df)` from sample data, then pasting+refining. ([pandera.readthedocs.io][1])

Later you can automate mapping from TypedDicts or dataset contracts; for now, representative ones (analytics, graph) are sufficient to set the pattern.

### 2.2 Add a small validation façade

In `storage/storage/pandera_schemas.py`, add:

```python
import pandas as pd

def validate_dataset_df(table_key: str, df: pd.DataFrame) -> pd.DataFrame:
    schema = DATASET_SCHEMAS.get(table_key)
    if not schema:
        return df  # grace: no schema registered yet
    return schema.validate(df)
```

This becomes the **standard gate** for Pandera validation at dataset boundaries.

---

## 3. Use Ibis & Pandera in storage & repositories (representative changes)

Now that you have `gateway.ibis` and dataset schemas, let’s show a few key patterns.

### 3.1 Storage repository: add Ibis-based read methods

Take `storage/storage/repositories/datasets.py`:

```python
@dataclass(frozen=True)
class DatasetReadRepository(BaseRepository):
    def read_dataset_rows(self, table_key: str, *, limit: int, offset: int) -> list[RowDict]:
        sql = "SELECT * FROM metadata.dataset_rows(?, ?, ?)"
        return fetch_all_dicts(self.con, sql, [table_key, limit, offset])
```

Add a **parallel Ibis-based method**:

```python
import pandas as pd
from codeintel.storage.pandera_schemas import validate_dataset_df

@dataclass(frozen=True)
class DatasetReadRepository(BaseRepository):
    ...

    def read_dataset_ibis(
        self,
        table_key: str,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> pd.DataFrame:
        """Read dataset rows as a Pandas DataFrame via Ibis, validated by Pandera."""
        # Under the hood, metadata.dataset_rows is a view; use Ibis to query it.
        t = self.gateway.ibis.table("metadata.dataset_rows")  # or appropriate view
        expr = t.filter(t.table_key == table_key)
        if offset is not None:
            expr = expr.offset(offset)
        if limit is not None:
            expr = expr.limit(limit)

        df = expr.execute()  # Ibis compiles to DuckDB and returns a DataFrame :contentReference[oaicite:2]{index=2}
        return validate_dataset_df(table_key, df)
```

From here on:

* internal consumers that want schema-valid DataFrames can use `read_dataset_ibis`,
* serving/analytics can use either:

  * `gateway.ibis.table("schema.table")` for expression pipelines, or
  * repository methods that already encapsulate common queries.

### 3.2 Storage views using Ibis expressions (representative)

Your `storage/views/*.py` modules currently define named views via raw SQL. For future new views, you can define them via Ibis instead, e.g.:

```python
# storage/storage/views/function_views.py (new style for new views)

import ibis
from codeintel.storage.gateway.protocol import StorageGateway

def create_function_summary_view(gateway: StorageGateway) -> None:
    ibis_con = gateway.ibis.con
    fm = ibis_con.table("analytics.function_metrics")
    ft = ibis_con.table("analytics.function_types")

    # Example: join metrics + types to create a summary view
    joined = fm.left_join(
        ft,
        ["repo", "commit", "rel_path", "function_goid_h128"],
    )

    summary = (
        joined.group_by(
            "repo", "commit", "rel_path", "function_goid_h128", "qualname"
        )
        .aggregate(
            loc=joined.loc.max(),
            complexity=joined.cyclomatic_complexity.max(),
            param_count=joined.param_name.nunique(),
        )
    )

    # Register as a DuckDB view via Ibis
    ibis_con.create_view("analytics.v_function_summary", summary, overwrite=True)
```

This pattern:

* keeps complex joins & aggregations in Ibis (safer than hand-written SQL, easier to refactor),
* **still materializes** to a DuckDB view for backwards compatibility,
* and benefits from Pandera if you later validate the resulting view.

---

## 4. Serving integration: Ibis as the primary query engine

Next, make serving’s query layers prefer Ibis expressions for dynamic queries (filters, projections, pagination), with Pandera as a guardrail when you surface data out.

### 4.1 DatasetQueryLayer: Ibis-based query path

Look at `serving/serving/backend/dataset_backend.py`: it likely constructs SQL strings and uses DuckDB directly.

Add an Ibis-based method alongside or under the hood.

Example:

```python
# serving/serving/backend/dataset_backend.py

import pandas as pd
import ibis.expr.types as it

from codeintel.storage.pandera_schemas import validate_dataset_df

@dataclass
class DatasetQueryLayer:
    context: DuckDBQueryContext
    repositories: Repositories  # includes gateway / dataset repos

    def query_dataset(
        self,
        table: str,
        *,
        filters: Mapping[str, object] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Query an arbitrary dataset via Ibis, validated by Pandera."""
        ibis_con = self.repositories.gateway.ibis.con
        expr = ibis_con.table(table)  # e.g., "analytics.function_metrics"

        if filters:
            for col, value in filters.items():
                expr = expr.filter(expr[col] == value)

        if columns:
            expr = expr[columns]

        if offset is not None:
            expr = expr.offset(offset)
        if limit is not None:
            expr = expr.limit(limit)

        df = expr.execute()
        return validate_dataset_df(table, df)
```

Now HTTP/MCP handlers can call `query_dataset` and get:

* Ibis-powered query semantics,
* Pandera-validated results.

More advanced: for `serving/serving/backend/function_backend.py`, you can rewrite multi-table queries with Ibis and still re-use `validate_dataset_df` on the final DataFrame.

---

## 5. Analytics / graphs: representative advanced uses

Ibis & Pandera can also cleanly slot into analytics & graph computations where data gets “wide” or where you currently hand-write SQL.

### 5.1 Ibis for analytic derived tables

Example: Instead of writing SQL to produce a `v_hotspots` view, define it in `analytics/analytics/ibis_views.py`:

```python
# analytics/analytics/ibis_views.py

import ibis

from codeintel.storage.gateway.protocol import StorageGateway

def create_hotspots_view(gateway: StorageGateway) -> None:
    con = gateway.ibis.con
    fm = con.table("analytics.function_metrics")
    churn = con.table("history.git_churn")

    joined = fm.inner_join(
        churn,
        ["repo", "rel_path", "commit", "function_goid_h128"],
    )

    # Example scoring expression
    hotspots_expr = joined.mutate(
        hotspot_score=(
            joined.cyclomatic_complexity * joined.change_count
        )
    )

    con.create_view("analytics.v_function_hotspots", hotspots_expr, overwrite=True)
```

Now your function hotspots plugin can:

* either operate against `analytics.hotspots` table, or
* use this Ibis view as a higher-level derived asset.

### 5.2 Pandera as a contract for analytics output

In `analytics/functions/metrics.py`, after computing `metrics_rows` and `types_rows`, you can:

* build DataFrames,
* validate them via the schemas from `pandera_schemas`,
* then insert.

Example:

```python
import pandas as pd
from codeintel.storage.pandera_schemas import (
    FUNCTION_METRICS_SCHEMA,
    FUNCTION_TYPES_SCHEMA,
)

def compute_function_metrics_and_types(
    gateway: StorageGateway,
    cfg: FunctionAnalyticsStepConfig,
    options: FunctionAnalyticsOptions,
) -> dict[str, int]:
    ...
    metrics_df = pd.DataFrame(metrics_rows)
    types_df = pd.DataFrame(types_rows)

    metrics_df = FUNCTION_METRICS_SCHEMA.validate(metrics_df)
    types_df = FUNCTION_TYPES_SCHEMA.validate(types_df)

    # Insert rows
    gateway.analytics.insert_function_metrics(metrics_df.itertuples(index=False, name=None))
    gateway.analytics.insert_function_types(types_df.itertuples(index=False, name=None))

    return {
        "metrics_rows": len(metrics_df),
        "types_rows": len(types_df),
    }
```

For big tables, you can:

* validate on a **sample** in production (full validation only in tests),
* or apply Pandera at the edges (e.g., on smaller intermediate DataFrames).

---

## 6. Advanced functionality: where Ibis + Pandera really shine in your architecture

Here are a few “extra mile” features that make this integration powerful for your long-term architecture.

### 6.1 Deriving schemas from Ibis + Pandera for dataset contracts

* Ibis can introspect table schema (`ibis_con.table("schema.table").schema()`) and return an `ibis.Schema` object. ([DuckDB][2])
* Pandera can infer DataFrameSchema from sample data. ([pandera.readthedocs.io][1])

You can build tooling that:

1. For a new dataset (or changed table), materialize a small sample to pandas via Ibis.
2. `schema = pa.infer_schema(df)` to get a draft Pandera schema.
3. Persist that schema for:

   * dataset contract docs,
   * JSON Schema generation for serving / LLMs,
   * generating DuckDB DDL.

This gives you a **single schema source**: dataset contract ←→ Pandera ←→ Ibis ←→ DuckDB.

### 6.2 Property-based tests with Hypothesis + Pandera

Pandera integrates well with Hypothesis to generate valid DataFrames from schemas. ([pandera.readthedocs.io][3])

You can:

* define schemas for key analytics tables (`function_metrics`, `hotspots`, `call_graph_edges`),
* derive **strategies** that generate random-but-valid DataFrames,
* run pipeline functions (plugins) on them and assert invariants, e.g.:

```python
from pandera import check_io
import hypothesis.strategies as st
from pandera.tools import dataframe_strategy

@check_io(input_schema=FUNCTION_METRICS_SCHEMA, output_schema=HOTSPOTS_SCHEMA)
def compute_hotspots_from_metrics(df: pd.DataFrame) -> pd.DataFrame:
    ...

@given(df=dataframe_strategy(FUNCTION_METRICS_SCHEMA))
def test_hotspot_invariants(df):
    hotspots = compute_hotspots_from_metrics(df)
    # assert e.g. hotspot_score >= 0, etc.
```

This strengthens your **logic + rule based execution** by testing rules at the schema level, not just at the row level.

### 6.3 Using Ibis to express cross-cutting “compositive” views for serving

For your “compositive views built from plugins and calc recipes” goal, Ibis is a natural:

* each plugin’s outputs are physically stored in DuckDB,
* composite calc recipes become Ibis expressions that join/aggregate across plugin outputs,
* the code that defines those expressions **is the recipe**.

You can even generate an internal “calc view graph” from Ibis expressions (or metadata attached to them) and treat it similarly to your plugin DAG.

---

## 7. Migration strategy: representative, not exhaustive

You don’t need to ibis-ify and pandera-ify everything at once. A sensible staged approach:

1. **Stage 1 – Foundations:**

   * Add `IbisGateway` and `StorageGateway.ibis`.
   * Create `pandera_schemas.py` with a handful of key schemas (e.g., `function_metrics`, `function_types`, `callgraph`).
   * Add Ibis + Pandera read path in `DatasetReadRepository` and `DatasetQueryLayer` as shown.

2. **Stage 2 – Analytics/Graphs:**

   * Update `FunctionMetricsPlugin` & its compute to:

     * use Pandera schema at the output boundary,
     * optionally use Ibis for any heavy SQL-like operations.
   * Update `CallGraphPlugin` or a graph view to:

     * use Ibis expressions for any post-processing views,
     * use Pandera for sample validation of key tables.

3. **Stage 3 – Serving:**

   * Convert a few HTTP routes (e.g., function metrics + hotspots) to:

     * build queries via `gateway.ibis`,
     * validate DataFrames via Pandera before serializing to transport models.

4. **Stage 4 – Tooling & schemas:**

   * Add small internal tools to:

     * infer Pandera schemas from data via Ibis + Pandera,
     * generate JSON Schema from Pandera for serving / docs,
     * generate DDL for DuckDB from dataset contracts + Pandera.

At each stage, you’re following the same pattern:

* Ibis defines how you **express and compose data** from DuckDB.
* Pandera defines which **shapes and invariants are legal**.
* Your own metadata + manifests still define **what plugins exist**, **how they compose**, and **which datasets they produce**.

That’s a good place to be for the “fully abstracted, policy-driven, logic-based pipeline” you’re aiming at.

If you’d like, we can next pick one dataset (e.g., `analytics.function_metrics`) and go all the way through:

* dataset contract → Pandera schema → Ibis-based view → JSON Schema for serving → simple property-based tests — so you have a fully fleshed out “reference implementation” of the pattern.

[1]: https://pandera.readthedocs.io/en/latest/schema_inference.html?utm_source=chatgpt.com "Schema Inference and Persistence - pandera documentation"
[2]: https://duckdb.org/docs/stable/guides/python/ibis.html?utm_source=chatgpt.com "Integration with Ibis"
[3]: https://pandera.readthedocs.io/en/stable/dataframe_schemas.html?utm_source=chatgpt.com "DataFrame Schemas - pandera documentation"


# map of where to implement #

Gotcha, let’s make this super explicit and “LLM-programmer-friendly”.

Below is:

* A **conceptual map** of where Ibis & Pandera should be used in your CodeIntel architecture.
* Then a **concrete checklist** of file/path patterns and what to do in each.
* Organized so an implementation agent can literally walk through and apply the patterns.

I’ll separate **Pandera** (schemas & validation) and **Ibis** (query & compositive views), then show cross-cutting rules.

---

## 1. Global rules (high level)

### 1.1 Where Pandera should appear

Pandera is the standard for **table-shaped data** (DataFrames) at boundaries:

1. **Every logical dataset in `config.config.datasets`**

   * Any table/view with a `DatasetContract` / `RowBinding`:

     * `core.*`
     * `graph.*`
     * `analytics.*`
     * `profiles.*`
     * `test.*`
   * **Rule:** Each such dataset should have a **Pandera DataFrameSchema** keyed by its dataset key (e.g., `"analytics.function_metrics"`, `"graph.call_graph_edges"`).

2. **Every plugin that writes to a dataset**

   * Ingestion plugins (`ingestion.plugins.*`) writing `ingest.*`, `core.*`.
   * Graph plugins (`graphs.plugins.*`) writing `graph.*`, `core.*`.
   * Analytics plugins (`analytics.plugins.*`) writing `analytics.*`, `profiles.*`.
   * **Rule:** Right before inserting rows into DuckDB, build a DataFrame and validate with the corresponding Pandera schema.

3. **Every serving API that returns DataFrames / row lists**

   * HTTP endpoints in `serving/serving/http/*`.
   * MCP tools in `serving/serving/mcp/*`.
   * **Rule:** Any backend that builds a DataFrame or row dict list from DB should:

     * fetch via Ibis,
     * validate via Pandera,
     * then map to transport models / JSON.

4. **Dataset-level validations & tests**

   * `storage/storage/validation/*`
   * `analytics/analytics/testing/*`
   * **Rule:** Use Pandera schemas as the source of truth for shape checks and for Hypothesis strategies in property-based tests.

---

### 1.2 Where Ibis should appear

Ibis is the standard for **querying and composing** relational data:

1. **All new and refactored multi-table queries**

   * Storage views (`storage/storage/views/*.py`).
   * Analytics derived tables / views (e.g., hotspots, graph metrics).
   * Serving backends that join multiple tables.
   * **Rule:** If you’re writing a non-trivial SELECT/JOIN/AGGREGATE, write it as an **Ibis expression** over `gateway.ibis.con`, not ad-hoc SQL.

2. **View definitions and “calc recipes”**

   * Any `v_*` views or “compositive” views (summaries, hotspots, profiles).
   * **Rule:** Define these as Ibis expressions, then `create_view("schema.v_name", expr, overwrite=True)`.

3. **Repository-level query helpers**

   * `storage/storage/repositories/*.py` where you currently call `con.execute` or compose SQL strings.
   * **Rule:** Add Ibis-based methods in repositories that:

     * build an Ibis expression,
     * optionally execute to a DataFrame,
     * optionally validate with Pandera.

4. **Serving query layers**

   * `serving/serving/backend/*.py` (dataset_backend, function_backend, subsystem_backend, etc.).
   * **Rule:** Use Ibis to build filtered/paginated views (user filters, search, etc.) instead of string-concat SQL.

---

## 2. Pandera: instance types + explicit path checklist

### 2.1 Dataset contracts → Pandera schemas (one per dataset)

**Instance type:**

* Every dataset key in `config.config.datasets.contracts.DATASET_CONTRACTS`

  * e.g. `core.goids`, `core.goid_crosswalk`,
  * `graph.call_graph_nodes`, `graph.call_graph_edges`, CFG/DFG tables,
  * `analytics.function_metrics`, `analytics.function_types`, `analytics.function_hotspots`, `analytics.function_validation`,
  * `profiles.*`, `test.*`.

**Where to implement:**

* File(s):

  * `storage/storage/pandera_schemas.py` (or a small package: `storage/storage/pandera_schemas/__init__.py`, `analytics.py`, `graphs.py`, `core.py`)

**What to do:**

* For each dataset in contracts:

  * Define a `DataFrameSchema` matching its columns and basic invariants.
  * Register in a global dict:

    ```python
    DATASET_SCHEMAS: dict[str, DataFrameSchema] = {
        "analytics.function_metrics": FUNCTION_METRICS_SCHEMA,
        "graph.call_graph_edges": CALLGRAPH_EDGES_SCHEMA,
        "core.goids": CORE_GOIDS_SCHEMA,
        ...
    }
    ```

* Implement a canonical validator:

  ```python
  def validate_dataset_df(table_key: str, df: pd.DataFrame) -> pd.DataFrame:
      schema = DATASET_SCHEMAS.get(table_key)
      if schema is None:
          return df
      return schema.validate(df)
  ```

**LLM programmer checklist:**

* [ ] For each dataset contract in `config/config/datasets/contracts.py`, create a Pandera schema.
* [ ] Make sure the schema key is exactly the dataset key used elsewhere (`schema.table` or logical key).
* [ ] Add non-negative checks, allowed value lists, nullability constraints where appropriate.

---

### 2.2 Plugin write paths → Pandera validate-before-insert

**Instance type:**

* Any code paths that write rows to DuckDB for a logical dataset.

**Specific code locations:**

* **Ingestion:**

  * `ingestion/ingestion/plugins/*.py` (e.g., SCIP ingest, module ingest).
  * `ingestion/ingestion/adapters/duckdb_storage.py` (bulk insert helpers).

* **Graphs:**

  * `graphs/graphs/compute/*.py` (cfg, dfg, imports, symbols, goid, callgraph),
  * `graphs/graphs/adapters/callgraph_persistence.py` and `duckdb_storage.py`,
  * `graphs/graphs/plugins/*.py` (graph plugin wrappers).

* **Analytics:**

  * `analytics/analytics/functions/metrics.py` & friends (`function_metrics`, `function_types`, `function_validation`),
  * `analytics/analytics/graphs/*.py` (graph metrics tables),
  * `analytics/analytics/history/*.py` (git history tables),
  * analytics plugins under `analytics/plugins/*`.

* **Export:**

  * `export/export/runner.py`, `export/export/backends/*.py` for any staging tables.

**What to do (pattern):**

1. At the point where row data is “complete” for a dataset, **build a DataFrame**:

   ```python
   import pandas as pd
   from codeintel.storage.pandera_schemas import validate_dataset_df

   rows = [...]  # list[dict] or list[row models]
   df = pd.DataFrame(rows)
   df = validate_dataset_df("analytics.function_metrics", df)
   ```

2. Use DataFrame → tuple conversion for insert (where appropriate):

   ```python
   tuples = list(df.itertuples(index=False, name=None))
   gateway.insert_rows("analytics.function_metrics", tuples)
   ```

3. For performance-critical cases, you can:

   * validate on a **sample** in production,
   * validate fully in tests / dev.

**LLM programmer checklist:**

* [ ] Find every plugin / compute function that writes to a dataset (look for `insert_*`, `macro_insert_rows`, `INSERT INTO`, etc.).
* [ ] Before writing, wrap rows into a DataFrame, validate via Pandera for that dataset key, then insert.
* [ ] Respect invariants: if Pandera schema fails, either:

  * raise explicit errors with context, or
  * log and fail plugin gracefully.

---

### 2.3 Serving responses → Pandera on outbound data

**Instance type:**

* Any backend that reads from DuckDB to return **tabular data** to:

  * HTTP,
  * MCP tools,
  * CLI “table” display.

**Specific code locations:**

* `serving/serving/backend/dataset_backend.py`
* `serving/serving/backend/function_backend.py`
* `serving/serving/backend/subsystem_backend.py`
* `serving/serving/backend/profile_backend.py`
* `serving/serving/backend/datasets.py`
* Any future graph/analytics-specific backends.

**What to do (pattern):**

1. Replace raw SQL / direct DuckDB relation reads with:

   ```python
   expr = gateway.ibis.table("analytics.function_metrics")  # or view
   # apply filters/sorting/pagination using Ibis
   df = expr.execute()
   df = validate_dataset_df("analytics.function_metrics", df)
   ```

2. Then map the validated DataFrame to Pydantic models or JSON dicts for the transport layer.

**LLM programmer checklist:**

* [ ] For each backend that returns a dataset (metrics, callgraphs, subsystems, profiles):

  * [ ] Fetch via Ibis,
  * [ ] validate using Pandera,
  * [ ] then convert to the domain/transport models.
* [ ] Add explicit error handling/logging when validation fails.

---

### 2.4 Tests & validation modules

**Instance type:**

* Data quality checks, property-based tests, and conformance tests.

**Specific code locations:**

* `storage/storage/validation/*.py`
* `analytics/analytics/testing/*.py`
* `tests/*` (top-level tests folder, not included here but in your repo)

**What to do:**

* Use Pandera schemas to:

  * drive simple conformance checks (row counts, missing columns),
  * generate Hypothesis strategies for property-based tests.

**LLM programmer checklist:**

* [ ] In validation modules, import `DATASET_SCHEMAS` and call `schema.validate(df)` instead of hand-rolled checks where possible.
* [ ] For property tests, use `pandera.hypothesis.dataframe_strategy(schema)` to generate valid test inputs.

---

## 3. Ibis: instance types + explicit path checklist

### 3.1 View definitions (`v_*` views and complex SQL views)

**Instance type:**

* Any view defined in SQL under `storage/storage/views/*.py` or `storage/storage/sql/*`.
* Any analytics/graph “compositive” view you’re adding.

**Specific code locations:**

* `storage/storage/views/function_views.py`
* `storage/storage/views/graph_views.py`
* `storage/storage/views/module_views.py`
* `storage/storage/views/data_model_views.py`
* `storage/storage/views/subsystem_views.py`
* `storage/storage/views/test_views.py`
* Future/other views in analytics/graphs (e.g. `analytics/analytics/ibis_views.py`, `graphs/graphs/ibis_views.py` you add).

**What to do (pattern):**

1. For each view, replace raw SQL string with Ibis expression:

   ```python
   con = gateway.ibis.con
   fm = con.table("analytics.function_metrics")
   # build expression...
   expr = fm[["repo", "commit", "function_goid_h128", "loc"]]  # example
   con.create_view("analytics.v_function_summary", expr, overwrite=True)
   ```

2. If the view is also used as a “dataset”, consider adding a Pandera schema for it.

**LLM programmer checklist:**

* [ ] For each view function in `storage/storage/views/*.py`, add or replace with an Ibis expression that ends with `create_view(...)`.
* [ ] Do not write new ad-hoc view SQL; always go through Ibis.

---

### 3.2 Repository queries (`storage/storage/repositories/*.py`)

**Instance type:**

* Read operations that build queries for:

  * tests, functions, modules, graphs, subsystems, datasets.

**Specific code locations:**

* `storage/storage/repositories/datasets.py`
* `storage/storage/repositories/functions.py`
* `storage/storage/repositories/graphs.py`
* `storage/storage/repositories/modules.py`
* `storage/storage/repositories/subsystems.py`
* `storage/storage/repositories/tests.py`

**What to do (pattern):**

* Add Ibis-based methods alongside existing methods (and progressively migrate):

  ```python
  class FunctionRepository(BaseRepository):
      ...

      def list_functions_ibis(self, repo: str, commit: str) -> pd.DataFrame:
          con = self.gateway.ibis.con
          fm = con.table("analytics.function_metrics")
          expr = fm.filter((fm.repo == repo) & (fm.commit == commit))
          df = expr.execute()
          return validate_dataset_df("analytics.function_metrics", df)
  ```

**LLM programmer checklist:**

* [ ] For each repository that currently composes SQL strings, add `*_ibis` methods that:

  * [ ] build Ibis expressions via `gateway.ibis.con.table`,
  * [ ] execute to DataFrame,
  * [ ] validate via Pandera (where appropriate).

---

### 3.3 Serving backends (`serving/serving/backend/*.py`)

**Instance type:**

* Backend logic that currently calls:

  * `query_api`, raw queries, or DuckDB directly.

**Specific files:**

* `serving/serving/backend/dataset_backend.py`
* `serving/serving/backend/function_backend.py`
* `serving/serving/backend/subsystem_backend.py`
* `serving/serving/backend/profile_backend.py`
* `serving/serving/backend/datasets.py`
* `serving/serving/backend/query_api.py`
* `serving/serving/backend/duckdb_service.py`

**What to do:**

* Replace ad-hoc SQL with Ibis query expressions:

  ```python
  con = self.context.gateway.ibis.con
  t = con.table("graph.call_graph_edges")
  expr = t.filter(
      (t.repo == repo) &
      (t.commit == commit) &
      (t.caller_goid_h128 == goid)
  ).limit(limit).offset(offset)
  df = expr.execute()
  df = validate_dataset_df("graph.call_graph_edges", df)
  ```

**LLM programmer checklist:**

* [ ] For each backend method that reads from DB:

  * [ ] use `gateway.ibis.con.table("schema.table")` to construct queries,
  * [ ] apply filters/joins via Ibis,
  * [ ] execute to DataFrame and Pandera-validate for the dataset key.

---

### 3.4 Analytics & graphs compute (optional but valuable)

**Instance type:**

* Analytical operations that currently:

  * aggregate data with Python loops,
  * manually group records.

**Specific files:**

* Analytics:

  * `analytics/analytics/graphs/*.py` (graph metrics, ext metrics, config metrics),
  * `analytics/analytics/profiles/*.py` (subsystem profiles, module profiles),
  * `analytics/analytics/history/*.py` (git churn, history tables).

* Graphs:

  * `graphs/graphs/catalog.py` (graph lookups),
  * `graphs/graphs/resources/storage.py`,
  * any graph statistic aggregation.

**What to do:**

* Where feasible, rewrite heavy aggregations as Ibis expressions and let DuckDB do the work.

**LLM programmer checklist:**

* [ ] For heavy aggregations over large tables (metrics, graphs, history):

  * [ ] prefer an Ibis expression (`group_by`, `aggregate`, etc.) over manual Python loops.
  * [ ] Use Pandera at the *input* and *output* boundaries, not necessarily on every intermediate.

---

## 4. Cross-cutting: JSON Schema export for serving & tools

For any dataset that:

* is exposed directly via serving, or
* is used in LLM tools / MCP,

we should:

1. Export JSON Schema from Pandera.

   ```python
   from codeintel.storage.pandera_schemas import DATASET_SCHEMAS
   from codeintel.serving.schema_export import pandera_to_json_schema

   FUNCTION_METRICS_JSON_SCHEMA = pandera_to_json_schema(DATASET_SCHEMAS["analytics.function_metrics"])
   CALLGRAPH_EDGES_JSON_SCHEMA = pandera_to_json_schema(DATASET_SCHEMAS["graph.call_graph_edges"])
   ```

2. Use JSON Schema in:

   * OpenAPI spec,
   * MCP tool definitions,
   * any RAG/LLM doc that wants to know row shapes.

**LLM programmer checklist:**

* [ ] For each dataset that is directly surfaced externally (HTTP, MCP, export):

  * [ ] Ensure a Pandera schema exists,
  * [ ] Ensure we have a corresponding JSON Schema via the adapter,
  * [ ] Wire that schema into the relevant API/tool definitions.

---

## 5. Cross-cutting: tests & property-based tests

Final category of “instances where these should be deployed”:

* For **“golden” datasets** (`analytics.function_metrics`, `graph.call_graph_edges`, `core.goids`, `analytics.function_hotspots`, etc.):

  * [ ] Add Pandera-backed property tests (Hypothesis) in `tests/analytics/*` and `tests/graphs/*` that check domain invariants using generated DataFrames.

Example invariants to encode:

* function_metrics: non-negative complexity/LOC, monotone risk functions, etc.
* call_graph_edges: degree view consistency, no broken repo/commit consistency, etc.

---

If you hand this list to an LLM programmer and say:

> “For each category, implement the patterns described; don’t create new SQL directly; always add/update schemas when new datasets are introduced,”

they’ll have a clear, explicit map of **where** and **how** Ibis & Pandera should be deployed throughout the repo.


# analytics.function_metrics example #

wesome, let’s make `analytics.function_metrics` the “golden path” that shows how all the pieces fit together.

We’ll walk it end-to-end:

> **Dataset contract → Pandera schema → Ibis-based view → JSON Schema for serving → property-based tests**

I’ll keep this concrete and code-level, but still a *template* so you can replicate for other datasets.

---

## 1. Dataset contract (single source of truth)

Assumptions for `analytics.function_metrics` (adjust to your real schema):

* Table: `analytics.function_metrics`
* Rough fields:

  * `repo: str`
  * `commit: str`
  * `rel_path: str`
  * `function_goid_h128: int`
  * `qualname: str`
  * `language: str`
  * `loc: int`
  * `cyclomatic_complexity: int`
  * `fan_in: int`
  * `fan_out: int`
  * `has_type_hints: bool`
  * `coverage_ratio: float | None`

### 1.1 Contract entry in config

You likely already have something like `config/config/datasets/contracts.py` (or equivalent) defining table schemas. We’ll treat that as the canonical declaration:

```python
# config/config/datasets/contracts.py (simplified example)

from dataclasses import dataclass

@dataclass(frozen=True)
class ColumnDef:
    name: str
    duckdb_type: str
    nullable: bool = False

@dataclass(frozen=True)
class TableContract:
    table_name: str
    columns: tuple[ColumnDef, ...]

FUNCTION_METRICS_CONTRACT = TableContract(
    table_name="analytics.function_metrics",
    columns=(
        ColumnDef("repo", "TEXT"),
        ColumnDef("commit", "TEXT"),
        ColumnDef("rel_path", "TEXT"),
        ColumnDef("function_goid_h128", "UBIGINT"),
        ColumnDef("qualname", "TEXT"),
        ColumnDef("language", "TEXT"),
        ColumnDef("loc", "INTEGER"),
        ColumnDef("cyclomatic_complexity", "INTEGER"),
        ColumnDef("fan_in", "INTEGER"),
        ColumnDef("fan_out", "INTEGER"),
        ColumnDef("has_type_hints", "BOOLEAN"),
        ColumnDef("coverage_ratio", "DOUBLE", nullable=True),
    ),
)

DATASET_CONTRACTS = {
    "analytics.function_metrics": FUNCTION_METRICS_CONTRACT,
    # ...
}
```

We’ll now **mirror** this in Pandera and Ibis, not re-invent it.

---

## 2. Pandera schema (contract ↔ DataFrame)

We create a Pandera schema that:

* aligns 1-for-1 with the table contract,
* enforces basic invariants (non-negative metrics, nullable fields, etc.),
* sits in a central registry.

### 2.1 Pandera schema definition

```python
# storage/storage/pandera_schemas.py

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema

FUNCTION_METRICS_SCHEMA = DataFrameSchema(
    {
        "repo": Column(str),
        "commit": Column(str),
        "rel_path": Column(str),
        "function_goid_h128": Column(int, Check.ge(0)),
        "qualname": Column(str),
        "language": Column(str),
        "loc": Column(int, Check.ge(0)),
        "cyclomatic_complexity": Column(int, Check.ge(0)),
        "fan_in": Column(int, Check.ge(0)),
        "fan_out": Column(int, Check.ge(0)),
        "has_type_hints": Column(bool),
        "coverage_ratio": Column(float, nullable=True),
    },
    strict=True,
    coerce=True,
)

DATASET_SCHEMAS: dict[str, DataFrameSchema] = {
    "analytics.function_metrics": FUNCTION_METRICS_SCHEMA,
    # ... add other datasets here over time
}

def validate_dataset_df(table_key: str, df: pd.DataFrame) -> pd.DataFrame:
    """Validate a DataFrame against the registered schema for table_key."""
    schema = DATASET_SCHEMAS.get(table_key)
    if schema is None:
        # Graceful: no schema registered → no validation yet
        return df
    return schema.validate(df)
```

> This is now the **canonical DataFrame contract** for `analytics.function_metrics`.

### 2.2 Use Pandera at the write boundary of the plugin

In `analytics/functions/metrics.py` (or equivalent), where you currently compute `metrics_rows` and write them into DuckDB:

```python
# analytics/analytics/functions/metrics.py

import pandas as pd
from codeintel.storage.pandera_schemas import FUNCTION_METRICS_SCHEMA

def compute_function_metrics_and_types(
    gateway: StorageGateway,
    cfg: FunctionAnalyticsStepConfig,
    options: FunctionAnalyticsOptions,
) -> dict[str, int]:
    # ... your existing logic building a list[dict] or list[rows]
    metrics_rows: list[dict[str, Any]] = []

    for fn in functions:
        # compute metrics dict per function
        metrics_rows.append(
            {
                "repo": cfg.repo,
                "commit": cfg.commit,
                "rel_path": fn.rel_path,
                "function_goid_h128": fn.goid,
                "qualname": fn.qualname,
                "language": fn.language,
                "loc": fn.loc,
                "cyclomatic_complexity": fn.complexity,
                "fan_in": fn.fan_in,
                "fan_out": fn.fan_out,
                "has_type_hints": fn.has_type_hints,
                "coverage_ratio": fn.coverage_ratio,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = FUNCTION_METRICS_SCHEMA.validate(metrics_df)

    # Insert into DuckDB
    gateway.analytics.insert_function_metrics_df(metrics_df)

    return {
        "metrics_rows": len(metrics_df),
        # you can still produce types_rows/etc.
    }
```

And in `gateway.analytics.insert_function_metrics_df`, you can either:

* use DuckDB’s `con.register` + `INSERT INTO`, or
* write to a Parquet temp file then `COPY`, etc.

---

## 3. Ibis-based view: derived view on top of function_metrics

Now we define a **canonical Ibis view** for a higher-level calc, e.g. `analytics.v_function_summary`.

### 3.1 Ibis view definition

```python
# analytics/analytics/ibis_views.py

from __future__ import annotations

import ibis
import ibis.expr.types as it

from codeintel.storage.gateway.protocol import StorageGateway

def create_function_summary_view(gateway: StorageGateway) -> None:
    """Create or replace analytics.v_function_summary as an Ibis-defined view.

    This view could, for example, normalize language, bucket complexity, etc.
    """
    con = gateway.ibis.con
    fm: it.Table = con.table("analytics.function_metrics")

    # Example: add derived columns
    summary_expr = fm.mutate(
        complexity_bucket=(
            ibis.case()
            .when(fm.cyclomatic_complexity <= 5, "low")
            .when(fm.cyclomatic_complexity <= 10, "medium")
            .else_("high")
            .end()
        ),
        loc_bucket=(
            ibis.case()
            .when(fm.loc <= 50, "small")
            .when(fm.loc <= 200, "medium")
            .else_("large")
            .end()
        ),
        # e.g. combine coverage + complexity into an initial risk score
        risk_score=(
            fm.cyclomatic_complexity * (1 - fm.coverage_ratio.fillna(0.0))
        ),
    )

    con.create_view(
        "analytics.v_function_summary",
        summary_expr,
        overwrite=True,
    )
```

Call this once at startup or in a “build views” step:

```python
# storage/storage/views/__init__.py
def create_all_views(gateway: StorageGateway) -> None:
    from codeintel.analytics.ibis_views import create_function_summary_view
    # ... other view creators
    create_function_summary_view(gateway)
```

Now:

* `analytics.v_function_summary` is defined via Ibis,
* depends directly on `analytics.function_metrics`,
* is both type-safe and easy to refactor.

You can also define a Pandera schema for `v_function_summary` if you want.

---

## 4. JSON Schema for serving / OpenAPI / LLMs

We want to:

* export the **shape** of `analytics.function_metrics` for:

  * serving domain models,
  * OpenAPI docs,
  * LLM-facing JSON schema for tools/RAG.

Pandera already has a concept of JSON Schema for schemas (and if it’s missing pieces, we can build a small adapter).

### 4.1 Simple Pandera → JSON Schema adapter

Add a utility:

```python
# serving/serving/schema_export.py

from __future__ import annotations

from typing import Any, Mapping

from pandera import DataFrameSchema
from codeintel.storage.pandera_schemas import FUNCTION_METRICS_SCHEMA

_PANDERA_TYPE_MAP = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
}

def pandera_to_json_schema(df_schema: DataFrameSchema) -> Mapping[str, Any]:
    """Convert a Pandera DataFrameSchema to a JSON Schema-like dict.

    This is sufficient for OpenAPI and LLM-facing docs. You can extend it with
    field descriptions, pattern constraints, etc.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, col in df_schema.columns.items():
        field_type = _PANDERA_TYPE_MAP.get(col.dtype, "string")
        prop: dict[str, Any] = {"type": field_type}
        if not col.nullable:
            required.append(name)
        properties[name] = prop

    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    return schema

FUNCTION_METRICS_JSON_SCHEMA = pandera_to_json_schema(FUNCTION_METRICS_SCHEMA)
```

You can then:

* embed `FUNCTION_METRICS_JSON_SCHEMA` into OpenAPI,
* or attach it to your LLM tool / MCP spec so it knows what the table rows look like.

### 4.2 Using JSON Schema in serving

For example, in `serving/domain_models.py` or wherever you define API models:

```python
# serving/serving/backend/schemas.py

from codeintel.serving.schema_export import FUNCTION_METRICS_JSON_SCHEMA

# You might not need to store the whole schema here, but you can expose it:
FUNCTION_METRICS_RESPONSE_SCHEMA = FUNCTION_METRICS_JSON_SCHEMA
```

For LLM tool definitions (MCP), you can plug this JSON schema into the tool argument or result schemas so the LLM knows exactly what’s coming back.

---

## 5. Property-based tests (Hypothesis + Pandera) for regression safety

Now we use Pandera’s schema to power **property-based tests** that guard your analytic logic.

### 5.1 Strategy from Pandera schema

Pandera has built-in integration with Hypothesis to create strategies from a schema. A representative pattern:

```python
# tests/analytics/test_function_metrics_properties.py

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
from pandera import hypothesis as pah

from codeintel.storage.pandera_schemas import FUNCTION_METRICS_SCHEMA
from codeintel.analytics.functions.metrics import compute_function_hotspots  # example

# Generate random valid DataFrames based on FunctionMetrics schema
function_metrics_strategy = pah.dataframe_strategy(FUNCTION_METRICS_SCHEMA)


@settings(max_examples=20)
@given(df=function_metrics_strategy)
def test_hotspots_never_negative(df: pd.DataFrame) -> None:
    """Hotspot scores should never be negative for any legal input."""
    hotspots_df = compute_function_hotspots(df)  # pure function version, or adaptation

    assert (hotspots_df["hotspot_score"] >= 0).all()


@settings(max_examples=20)
@given(df=function_metrics_strategy)
def test_hotspots_respects_complexity_monotonicity(df: pd.DataFrame) -> None:
    """For fixed coverage, higher complexity should never produce a lower risk score."""
    # Pick a single repo/commit/path group to reduce noise
    if df.empty:
        pytest.skip("empty input")

    # You'd implement a more careful check here; this is illustrative
    hotspots_df = compute_function_hotspots(df)

    group = hotspots_df.groupby(
        ["repo", "commit", "rel_path"],
        group_keys=False,
    )

    def check_group(g: pd.DataFrame) -> bool:
        g_sorted = g.sort_values("cyclomatic_complexity")
        # naive monotone check:
        return (g_sorted["risk_score"].diff().fillna(0) >= -1e-8).all()

    assert group.apply(check_group).all()
```

This pattern:

* uses the **same Pandera schema** that enforces your dataset contract to generate realistic inputs,
* tests deeper invariants about your analytics logic,
* gives you stronger confidence that changing metrics or formulas won’t break core invariants.

You might not run these in every CI run (Hypothesis tests can be heavier), but as a nightly/regression suite they’re extremely valuable.

---

## 6. Putting it together: where does it sit in the architecture?

For `analytics.function_metrics`, after following the above:

* **Dataset contract (config)** – table name + column definitions drive:

  * DuckDB DDL,
  * dataset introspection,
  * documentation.

* **Pandera schema** – `FUNCTION_METRICS_SCHEMA`:

  * enforces shape & invariants at write time,
  * generates input strategies for property-based tests,
  * is the canonical DataFrame contract for that table.

* **Ibis view** – `analytics.v_function_summary`:

  * is defined as a pure expression over the table,
  * can be changed/refactored safely,
  * can be used by serving/analytics as a higher-level data asset.

* **JSON Schema** – exported from Pandera:

  * used by serving endpoints and/or LLM tools to describe rows,
  * keeps your schema in sync across storage, compute, and interfaces.

* **Property-based tests**:

  * ensure that any change to the metrics compute or derived views still satisfies invariants for any legal input DataFrame,
  * catch subtle regressions that example-based tests might miss.

This is exactly the “fully fleshed out reference implementation” you can now treat as the standard for other key datasets (e.g. `analytics.function_hotspots`, `graph.call_graph_edges`, `core.goids`, etc.) and repeat with minimal variation.

If you want, we can next do the same for a second dataset (say, `graph.call_graph_edges`) to show how schemas + Ibis + JSON Schema + tests look for *graph* domain data, so you have one reference for analytics and one for graph.


# graph.call_graph_edges example #

Love this as a follow-on — having **one analytics** and **one graph** dataset as fully worked references will make it way easier to keep the rest consistent.

Let’s do the same end-to-end treatment for:

> **`graph.call_graph_edges`**

We’ll mirror the structure from `analytics.function_metrics`:

1. Dataset contract → table shape & semantics
2. Pandera schema → DataFrame contract & invariants
3. Ibis-based view(s) → higher-level, compositive graph views
4. JSON Schema export → serving / tools / docs
5. Property-based tests → guard graph invariants

This will give you a “graph-domain canonical example” to sit alongside the analytics one.

---

## 1. Dataset contract – `graph.call_graph_edges`

Let’s assume a representative schema for `graph.call_graph_edges` (you can adjust to your actual contract):

* `repo: TEXT`
* `commit: TEXT`
* `caller_goid_h128: UBIGINT`
* `callee_goid_h128: UBIGINT`
* `caller_rel_path: TEXT`
* `callee_rel_path: TEXT`
* `caller_qualname: TEXT`
* `callee_qualname: TEXT`
* `call_kind: TEXT` (e.g., `"direct"`, `"indirect"`, `"dynamic"`)
* `call_site_line: INTEGER` (1-based)
* `call_site_col: INTEGER` (1-based)
* `evidence_json: TEXT` (structured info about the call site, optional)

### 1.1 Contract entry in config

As with `function_metrics`, you’d have something like:

```python
# config/config/datasets/contracts.py (excerpt)

from dataclasses import dataclass

@dataclass(frozen=True)
class ColumnDef:
    name: str
    duckdb_type: str
    nullable: bool = False

@dataclass(frozen=True)
class TableContract:
    table_name: str
    columns: tuple[ColumnDef, ...]

CALLGRAPH_EDGES_CONTRACT = TableContract(
    table_name="graph.call_graph_edges",
    columns=(
        ColumnDef("repo", "TEXT"),
        ColumnDef("commit", "TEXT"),
        ColumnDef("caller_goid_h128", "UBIGINT"),
        ColumnDef("callee_goid_h128", "UBIGINT"),
        ColumnDef("caller_rel_path", "TEXT"),
        ColumnDef("callee_rel_path", "TEXT"),
        ColumnDef("caller_qualname", "TEXT"),
        ColumnDef("callee_qualname", "TEXT"),
        ColumnDef("call_kind", "TEXT"),
        ColumnDef("call_site_line", "INTEGER"),
        ColumnDef("call_site_col", "INTEGER"),
        ColumnDef("evidence_json", "TEXT", nullable=True),
    ),
)

DATASET_CONTRACTS["graph.call_graph_edges"] = CALLGRAPH_EDGES_CONTRACT
```

This remains your **SSOT** for DuckDB schema; Pandera & Ibis just mirror it.

---

## 2. Pandera schema – graph edge contract & invariants

We define a Pandera schema that:

* matches the table contract,
* encodes useful graph invariants:

  * non-negative / positive line/col,
  * caller/callee GOIDs ≥ 0,
  * optional rule against self-loops (or at least flags them),
  * `call_kind` constrained to known values (if you want).

### 2.1 Schema definition

```python
# storage/storage/pandera_schemas.py (add alongside function_metrics)

from __future__ import annotations

import pandera as pa
from pandera import Column, Check, DataFrameSchema

CALLGRAPH_EDGES_SCHEMA = DataFrameSchema(
    {
        "repo": Column(str),
        "commit": Column(str),
        "caller_goid_h128": Column(int, Check.ge(0)),
        "callee_goid_h128": Column(int, Check.ge(0)),
        "caller_rel_path": Column(str),
        "callee_rel_path": Column(str),
        "caller_qualname": Column(str),
        "callee_qualname": Column(str),
        "call_kind": Column(str),
        "call_site_line": Column(int, Check.ge(1)),
        "call_site_col": Column(int, Check.ge(1)),
        "evidence_json": Column(str, nullable=True),
    },
    strict=True,
    coerce=True,
    checks=[
        # Optional: forbid self-loops (caller == callee)
        Check(lambda df: (df["caller_goid_h128"] != df["callee_goid_h128"]).all(), 
              error="Self-loop edges (caller == callee) are not allowed."),
    ],
)

DATASET_SCHEMAS["graph.call_graph_edges"] = CALLGRAPH_EDGES_SCHEMA
```

If you *do* allow self-loops, drop that check or change it to a weaker invariant (e.g., warnings only via a separate check).

### 2.2 Use Pandera at write boundary (CallGraphPlugin)

In your callgraph builder (`graphs/plugins/builders/callgraph.py` or similar), after you’ve computed a collection of edge rows, validate them:

```python
# graphs/plugins/builders/callgraph.py

import pandas as pd
from codeintel.storage.pandera_schemas import CALLGRAPH_EDGES_SCHEMA

class CallGraphPlugin(TargetPlugin):
    ...

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        ...
        edge_rows: list[dict[str, Any]] = []

        for path in paths:
            # existing logic collecting edges
            for edge in edges_for_path:
                edge_rows.append(
                    {
                        "repo": repo,
                        "commit": commit,
                        "caller_goid_h128": edge.caller_goid_h128,
                        "callee_goid_h128": edge.callee_goid_h128,
                        "caller_rel_path": edge.caller_rel_path,
                        "callee_rel_path": edge.callee_rel_path,
                        "caller_qualname": edge.caller_qualname,
                        "callee_qualname": edge.callee_qualname,
                        "call_kind": edge.kind,
                        "call_site_line": edge.call_site_line,
                        "call_site_col": edge.call_site_col,
                        "evidence_json": edge.evidence_json,
                    }
                )

        edges_df = pd.DataFrame(edge_rows)
        edges_df = CALLGRAPH_EDGES_SCHEMA.validate(edges_df)

        # Insert into DuckDB via your gateway
        ctx.gateway.graphs.insert_callgraph_edges_df(edges_df)

        return TargetResult.succeeded(
            row_counts={
                "graph.call_graph_edges": len(edges_df),
            }
        )
```

Insertion function (`insert_callgraph_edges_df`) can be a thin wrapper that converts DataFrame rows to tuples for DuckDB or uses `con.register` and `INSERT INTO`.

---

## 3. Ibis-based view – compositive graph views

We’ll define an Ibis view that:

* summarises in/out degree per function,
* optionally exposes per-function edge counts by kind.

Call it `graph.v_callgraph_degree`.

### 3.1 Ibis view definition

```python
# graphs/graphs/ibis_views.py

from __future__ import annotations

import ibis
import ibis.expr.types as it
from codeintel.storage.gateway.protocol import StorageGateway

def create_callgraph_degree_view(gateway: StorageGateway) -> None:
    """Create/replace graph.v_callgraph_degree as an Ibis-defined view.

    This aggregates callgraph edges to derive basic graph degree metrics:
    - out_degree: number of calls outgoing from a function
    - in_degree: number of calls incoming to a function
    """

    con = gateway.ibis.con
    edges: it.Table = con.table("graph.call_graph_edges")

    # Out-degree: group by caller
    out_deg = (
        edges.group_by(
            "repo",
            "commit",
            "caller_goid_h128",
            "caller_rel_path",
            "caller_qualname",
        )
        .aggregate(out_degree=edges.callee_goid_h128.count())
        .relabel(
            {
                "caller_goid_h128": "function_goid_h128",
                "caller_rel_path": "rel_path",
                "caller_qualname": "qualname",
            }
        )
    )

    # In-degree: group by callee
    in_deg = (
        edges.group_by(
            "repo",
            "commit",
            "callee_goid_h128",
            "callee_rel_path",
            "callee_qualname",
        )
        .aggregate(in_degree=edges.caller_goid_h128.count())
        .relabel(
            {
                "callee_goid_h128": "function_goid_h128",
                "callee_rel_path": "rel_path",
                "callee_qualname": "qualname",
            }
        )
    )

    # Full degree: outer join on (repo, commit, function_goid_h128)
    joined = out_deg.outer_join(
        in_deg,
        [
            out_deg.repo == in_deg.repo,
            out_deg.commit == in_deg.commit,
            out_deg.function_goid_h128 == in_deg.function_goid_h128,
        ],
    )

    degree_expr = joined.mutate(
        in_degree=joined.in_degree.fillna(0),
        out_degree=joined.out_degree.fillna(0),
    )[
        joined.repo,
        joined.commit,
        joined.function_goid_h128,
        joined.rel_path,
        joined.qualname,
        joined.out_degree,
        joined.in_degree,
    ]

    con.create_view(
        "graph.v_callgraph_degree",
        degree_expr,
        overwrite=True,
    )
```

You can call this in your “create all views” function (e.g., at bootstrap or as part of a build step).

### 3.2 Optional: view for edge kind distribution

You can also create `graph.v_callgraph_edge_kinds` summarizing call kinds per function pair. Same pattern: group `graph.call_graph_edges` by (caller, callee, call_kind), aggregate counts.

---

## 4. JSON Schema for serving / tools

We now export the Pandera schema for `graph.call_graph_edges` into JSON Schema so:

* serving endpoints can advertise it in OpenAPI,
* LLM tool specs can refer to it for tool argument/response schemas,
* internal docs can show field shapes.

### 4.1 Extend the Pandera→JSON Schema adapter

We already had `pandera_to_json_schema` for `function_metrics`. We can re-use it for graph:

```python
# serving/serving/schema_export.py

from codeintel.storage.pandera_schemas import (
    FUNCTION_METRICS_SCHEMA,
    CALLGRAPH_EDGES_SCHEMA,
)
from pandera import DataFrameSchema

_PANDERA_TYPE_MAP = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
}

def pandera_to_json_schema(df_schema: DataFrameSchema) -> dict:
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, col in df_schema.columns.items():
        # Pandera dtype might be a pandas-compatible type, adapt as needed:
        py_type = col.dtype  # may be better mapped using .type or .type.alias
        field_type = _PANDERA_TYPE_MAP.get(py_type, "string")

        prop: dict[str, Any] = {"type": field_type}
        if not col.nullable:
            required.append(name)
        properties[name] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }

FUNCTION_METRICS_JSON_SCHEMA = pandera_to_json_schema(FUNCTION_METRICS_SCHEMA)
CALLGRAPH_EDGES_JSON_SCHEMA = pandera_to_json_schema(CALLGRAPH_EDGES_SCHEMA)
```

Now `CALLGRAPH_EDGES_JSON_SCHEMA` is ready to use.

### 4.2 Use in serving / LLM specs

For example, for an HTTP endpoint that returns edges:

```python
# serving/serving/backend/callgraph_backend.py

from dataclasses import dataclass
import pandas as pd

from codeintel.storage.gateway.protocol import StorageGateway
from codeintel.storage.pandera_schemas import CALLGRAPH_EDGES_SCHEMA

@dataclass
class CallGraphBackend:
    gateway: StorageGateway

    def list_edges_for_function(self, repo: str, commit: str, goid: int) -> pd.DataFrame:
        con = self.gateway.ibis.con
        edges = con.table("graph.call_graph_edges")

        expr = edges.filter(
            (edges.repo == repo) &
            (edges.commit == commit) &
            ((edges.caller_goid_h128 == goid) | (edges.callee_goid_h128 == goid))
        )

        df = expr.execute()
        return CALLGRAPH_EDGES_SCHEMA.validate(df)
```

For LLM tools (e.g. MCP), you’d plug `CALLGRAPH_EDGES_JSON_SCHEMA` into the tool result schema so the model knows exactly what columns exist and of what type.

---

## 5. Property-based tests – graph invariants

Here’s where Pandera shines for graph data.

We can use:

* the same `CALLGRAPH_EDGES_SCHEMA` to **generate valid edge tables**, and
* test invariants about callgraph behavior and our derived views.

### 5.1 Strategy from Pandera schema

```python
# tests/graphs/test_callgraph_edges_properties.py

from __future__ import annotations

import pandas as pd
from hypothesis import given, settings
from pandera import hypothesis as pah

from codeintel.storage.pandera_schemas import CALLGRAPH_EDGES_SCHEMA
from codeintel.graphs.ibis_views import create_callgraph_degree_view
from codeintel.storage.gateway.testing import make_test_gateway  # imaginary helper

# Strategy: random valid DataFrames consistent with CALLGRAPH_EDGES_SCHEMA
edges_strategy = pah.dataframe_strategy(CALLGRAPH_EDGES_SCHEMA)
```

### 5.2 Invariant 1: degree view must be self-consistent

For any valid edges table:

* Out-degree per node = count of rows where node appears as caller.
* In-degree per node = count of rows where node appears as callee.

We can test that your Ibis degree view matches those semantics.

```python
@settings(max_examples=20)
@given(df=edges_strategy)
def test_callgraph_degree_view_consistency(df: pd.DataFrame) -> None:
    if df.empty:
        return  # nothing to check

    # Set up an in-memory DuckDB + gateway with df registered as call_graph_edges
    gateway = make_test_gateway()
    con = gateway.ibis.con

    # Load df into DuckDB for Ibis to see
    con.create_table("graph.call_graph_edges", df, overwrite=True)

    # Build the view
    create_callgraph_degree_view(gateway)

    # Query view into a DataFrame
    degree_expr = con.table("graph.v_callgraph_degree")
    degree_df = degree_expr.execute()

    # 1) Manually compute degrees from the original df
    # out-degree: count edges per caller
    manual_out = (
        df.groupby(["repo", "commit", "caller_goid_h128"])
        .size()
        .reset_index(name="out_degree")
        .rename(columns={"caller_goid_h128": "function_goid_h128"})
    )

    # in-degree: count edges per callee
    manual_in = (
        df.groupby(["repo", "commit", "callee_goid_h128"])
        .size()
        .reset_index(name="in_degree")
        .rename(columns={"callee_goid_h128": "function_goid_h128"})
    )

    # Merge manual degrees and fill NAs with 0
    merged_manual = manual_out.merge(
        manual_in,
        on=["repo", "commit", "function_goid_h128"],
        how="outer",
    ).fillna({"out_degree": 0, "in_degree": 0})

    # 2) Join with degree_df and compare values
    merged = merged_manual.merge(
        degree_df,
        on=["repo", "commit", "function_goid_h128"],
        how="outer",
        suffixes=("_manual", "_view"),
    ).fillna({"out_degree_manual": 0, "in_degree_manual": 0,
              "out_degree_view": 0, "in_degree_view": 0})

    assert (merged["out_degree_manual"] == merged["out_degree_view"]).all()
    assert (merged["in_degree_manual"] == merged["in_degree_view"]).all()
```

This test:

* generates *valid* call_graph_edges tables with Hypothesis,
* materializes them into DuckDB,
* builds `v_callgraph_degree` via Ibis,
* checks that the view matches the straightforward groupby semantics.

If someone changes the view definition or introduces a bug, this test will catch it.

### 5.3 Invariant 2: no self-loops (if desired)

If you enforce `caller_goid_h128 != callee_goid_h128`, Pandera’s schema-level check already guards this — no extra test needed. If instead you want to *allow* self-loops but want your plugin to **enforce cleaning or tagging them**, you can write a property test that:

* generates valid edges where self-loops are allowed (drop that schema check),
* runs your plugin logic,
* asserts your conventions (e.g., “self-loops are always labeled call_kind='self'” or “self-loops must be removed”).

---

## 6. Summary: graph reference pattern

After doing this for `graph.call_graph_edges`, you now have:

* **Dataset contract** (`CALLGRAPH_EDGES_CONTRACT`): defines the DuckDB table and its fields.
* **Pandera schema** (`CALLGRAPH_EDGES_SCHEMA`):

  * enforces the contract & graph invariants at write time,
  * powers Hypothesis strategies for property-based tests.
* **Ibis view** (`graph.v_callgraph_degree`):

  * expresses a compositive graph view in a high-level, refactorable way.
* **JSON Schema** (`CALLGRAPH_EDGES_JSON_SCHEMA`):

  * documents the shape of the table for serving and tools.
* **Property-based tests**:

  * verify view semantics and graph invariants for any legal edge dataset.

This is the graph-domain analogue of the `analytics.function_metrics` reference we built before. Together, they form a **two-sample “standard”** for all future datasets: one analytics, one graph.

From here, extending to other graph datasets (`graph.call_graph_nodes`, `graph.cfg_edges`, `graph.dfg_edges`) or analytics (`analytics.function_hotspots`, `analytics.subsystem_profiles`) is mostly mechanical:

* add a TableContract → Pandera schema → Ibis view(s) → JSON Schema → Hypothesis tests.

If you’d like, we can next pick one particularly “tricky” dataset (e.g. `core.goids` or `analytics.function_hotspots`) and design a slightly richer set of invariants & views (e.g., “GOIDs must be stable across commits”, “hotspot_score is derived from function_metrics + git_churn in a specific way”) and encode those in Pandera + tests — giving you a reference for cross-table invariants as well.


# tricky example: core.goids #

Perfect dataset to “go deep” on. Let’s treat `core.goids` (and its sibling `core.goid_crosswalk`) as **the tricky, canonical example** for:

* rich per-table invariants,
* cross-table invariants,
* Ibis views for QA + querying,
* Pandera schemas + property tests.

I’ll walk through:

1. What `core.goids` / `core.goid_crosswalk` mean in your system
2. Pandera schemas with **hard** invariants
3. Ibis views for **cross-table** invariants & QA
4. Cross-table checks & tests (including “stability” semantics)
5. Property-based test examples

Everything is “representative but realistic”: you can implement it almost directly and tweak as needed.

---

## 1. Semantics recap: `core.goids` & `core.goid_crosswalk`

From your code:

* `GoidDescriptor` ➜ `compute_goid(descriptor)` ➜ `goid_h128: int` and `build_urn(descriptor)` ➜ `urn: str`.

* `GoidRow` (for `core.goids`) fields (from `graphs/compute/goid.py`):

  ```python
  class GoidRow:
      goid_h128: int                  # 128-bit hash, but truncated to DECIMAL_38_MAX
      urn: str                        # "goid:{repo}/{rel_path}#{lang}:{kind}:{qualname}?s=..."
      repo: str
      commit: str
      rel_path: str
      language: str
      kind: str
      qualname: str
      start_line: int
      end_line: int | None
      created_at: datetime
  ```

* `GoidCrosswalkRow` (for `core.goid_crosswalk`) fields:

  ```python
  class GoidCrosswalkRow:
      repo: str
      commit: str
      goid: str             # GOID URN
      lang: str
      module_path: str
      file_path: str
      start_line: int
      end_line: int | None
      scip_symbol: str | None
      ast_qualname: str
      cst_node_id: str | None
      chunk_id: str | None
      symbol_id: str | None
      updated_at: datetime
  ```

Important semantics:

* `goid_h128` is computed from **descriptor including commit** (payload includes repo, commit, language, rel_path, kind, qualname, start/end).
* `urn` **does not include commit** (only repo, rel_path, language, kind, qualname, span); thus URN is **snapshot-independent identity** at the code level, while `goid_h128` is snapshot-specific.
* `core.goid_crosswalk` links URNs to concrete analysis artifacts across commits: SCIP symbol, AST qualname, CST node id, etc.

So the interesting invariants are:

* per-row structural invariants (types, ranges),
* per-table uniqueness & URN format invariants,
* cross-table invariants (goids ↔ crosswalk),
* cross-commit “stability” of URN identity.

---

## 2. Pandera schemas for `core.goids` & `core.goid_crosswalk`

We’ll create **two schemas** in `storage/storage/pandera_schemas.py`.

### 2.1 `CORE_GOIDS_SCHEMA`

```python
# storage/storage/pandera_schemas.py

from __future__ import annotations

import pandera as pa
from pandera import Column, Check, DataFrameSchema

DECIMAL_38_MAX = 10**38 - 1

CORE_GOIDS_SCHEMA = DataFrameSchema(
    {
        "goid_h128": Column(
            int,
            [
                Check.ge(0),
                Check.lt(DECIMAL_38_MAX),
            ],
        ),
        "urn": Column(
            str,
            # Must start with "goid:" and contain "?" and "s="
            checks=[
                Check(lambda s: s.str.startswith("goid:"), element_wise=True),
                Check(lambda s: s.str.contains(r"\?s=\d+", regex=True), element_wise=True),
            ],
        ),
        "repo": Column(str),
        "commit": Column(str),
        "rel_path": Column(str),
        "language": Column(str),
        "kind": Column(str),
        "qualname": Column(str),
        "start_line": Column(int, Check.ge(1)),
        "end_line": Column(int, nullable=True),
        "created_at": Column("datetime64[ns]"),
    },
    strict=True,
    coerce=True,
    checks=[
        # No duplicates of (repo, commit, goid_h128)
        Check(
            lambda df: ~df.duplicated(subset=["repo", "commit", "goid_h128"]).any(),
            error="Duplicate (repo, commit, goid_h128) in core.goids",
        ),
        # No duplicates of (repo, commit, urn)
        Check(
            lambda df: ~df.duplicated(subset=["repo", "commit", "urn"]).any(),
            error="Duplicate (repo, commit, urn) in core.goids",
        ),
        # Optional: end_line >= start_line when present
        Check(
            lambda df: df["end_line"].isna()
            | (df["end_line"] >= df["start_line"]),
            error="end_line must be >= start_line when present",
        ),
    ],
)
```

### 2.2 `CORE_GOID_CROSSWALK_SCHEMA`

```python
CORE_GOID_CROSSWALK_SCHEMA = DataFrameSchema(
    {
        "repo": Column(str),
        "commit": Column(str),
        "goid": Column(
            str,
            checks=[
                Check(lambda s: s.str.startswith("goid:"), element_wise=True),
            ],
        ),
        "lang": Column(str),
        "module_path": Column(str),
        "file_path": Column(str),
        "start_line": Column(int, Check.ge(1)),
        "end_line": Column(int, nullable=True),
        "scip_symbol": Column(str, nullable=True),
        "ast_qualname": Column(str),
        "cst_node_id": Column(str, nullable=True),
        "chunk_id": Column(str, nullable=True),
        "symbol_id": Column(str, nullable=True),
        "updated_at": Column("datetime64[ns]"),
    },
    strict=True,
    coerce=True,
    checks=[
        # end_line >= start_line when present
        Check(
            lambda df: df["end_line"].isna()
            | (df["end_line"] >= df["start_line"]),
            error="end_line must be >= start_line when present",
        ),
        # OPTIONAL: no duplicate (repo, commit, goid, file_path, start_line)
        Check(
            lambda df: ~df.duplicated(
                subset=["repo", "commit", "goid", "file_path", "start_line"]
            ).any(),
            error="Duplicate crosswalk mapping for same goid & location",
        ),
    ],
)
```

Register them:

```python
DATASET_SCHEMAS["core.goids"] = CORE_GOIDS_SCHEMA
DATASET_SCHEMAS["core.goid_crosswalk"] = CORE_GOID_CROSSWALK_SCHEMA
```

And use `validate_dataset_df("core.goids", df)` / `validate_dataset_df("core.goid_crosswalk", df)` at write boundaries.

---

## 3. Ibis views for cross-table invariants & QA

We’ll define two representative Ibis views:

1. `core.v_goid_crosswalk_join` – canonical join between `core.goids` and `core.goid_crosswalk`.
2. `core.v_goid_crosswalk_mismatches` – QA view that surfaces mismatches (missing goids, inconsistent fields).

### 3.1 `core.v_goid_crosswalk_join`

```python
# core/core/ibis_views.py  (or storage/views/core_views.py)

from __future__ import annotations

import ibis
import ibis.expr.types as it
from codeintel.storage.gateway.protocol import StorageGateway

def create_goid_crosswalk_views(gateway: StorageGateway) -> None:
    con = gateway.ibis.con

    goids: it.Table = con.table("core.goids")
    xwalk: it.Table = con.table("core.goid_crosswalk")

    # Join on repo, commit, and URN (xwalk.goid is URN)
    joined = goids.inner_join(
        xwalk,
        [
            goids.repo == xwalk.repo,
            goids.commit == xwalk.commit,
            goids.urn == xwalk.goid,
        ],
    )

    goid_crosswalk = joined[
        goids.repo.name("repo"),
        goids.commit.name("commit"),
        goids.goid_h128,
        goids.urn,
        goids.rel_path,
        goids.language,
        goids.kind,
        goids.qualname,
        goids.start_line,
        goids.end_line,
        xwalk.lang.name("crosswalk_lang"),
        xwalk.file_path,
        xwalk.module_path,
        xwalk.ast_qualname,
        xwalk.scip_symbol,
        xwalk.updated_at,
    ]

    con.create_view("core.v_goid_crosswalk_join", goid_crosswalk, overwrite=True)
```

### 3.2 `core.v_goid_crosswalk_mismatches`

A view to highlight various mismatch categories:

```python
def create_goid_crosswalk_mismatches_view(gateway: StorageGateway) -> None:
    con = gateway.ibis.con
    join = con.table("core.v_goid_crosswalk_join")

    mismatches = join.filter(
        (join.language != join.crosswalk_lang)
        | (join.rel_path != join.file_path)
        | (join.qualname != join.ast_qualname)
    )

    con.create_view("core.v_goid_crosswalk_mismatches", mismatches, overwrite=True)
```

Now you can:

* Do routine QA via `SELECT * FROM core.v_goid_crosswalk_mismatches` (should be empty or small).
* Use this in tests (assert row_count is small/zero for known-good repos).

---

## 4. Cross-table invariants & tests (core of the “tricky” dataset)

Here are **richer invariants** you can encode in tests using Ibis / Pandera:

### 4.1 Invariant A: Every crosswalk row must have a matching goid row

For any `(repo, commit, goid)` in `core.goid_crosswalk`:

* There must exist a `core.goids` row with:

  * same `repo`,
  * same `commit`,
  * `urn == goid`.

**Ibis QA query:**

```python
def crosswalk_missing_goids(gateway: StorageGateway) -> pd.DataFrame:
    con = gateway.ibis.con
    goids = con.table("core.goids")
    xwalk = con.table("core.goid_crosswalk")

    missing = xwalk.left_join(
        goids,
        [
            xwalk.repo == goids.repo,
            xwalk.commit == goids.commit,
            xwalk.goid == goids.urn,
        ],
    ).filter(goids.urn.isnull())

    return missing.execute()
```

**Test:**

```python
def test_core_goid_crosswalk_has_matching_goids(gateway: StorageGateway) -> None:
    df_missing = crosswalk_missing_goids(gateway)
    assert df_missing.empty, "Found crosswalk rows with no matching core.goids row"
```

### 4.2 Invariant B: URN must be consistent with fields

We know `build_urn`:

```python
base = (
    f"goid:{descriptor.repo}/{descriptor.rel_path}"
    f"#{descriptor.language}:{descriptor.kind}:{descriptor.qualname}"
)
if descriptor.end_line is None:
    return f"{base}?s={descriptor.start_line}"
return f"{base}?s={descriptor.start_line}&e={descriptor.end_line}"
```

We can reconstruct `descriptor` from a row and verify URN:

```python
# tests/core/test_goids_urn_roundtrip.py

from codeintel.graphs.compute.goid import GoidDescriptor, build_urn

def test_core_goids_urn_roundtrip(gateway: StorageGateway) -> None:
    con = gateway.ibis.con
    goids = con.table("core.goids").limit(1000)  # sample for speed
    df = goids.execute()

    for row in df.itertuples(index=False):
        desc = GoidDescriptor(
            repo=row.repo,
            commit=row.commit,
            language=row.language,
            rel_path=row.rel_path,
            kind=row.kind,
            qualname=row.qualname,
            start_line=row.start_line,
            end_line=row.end_line,
        )
        expected_urn = build_urn(desc)
        assert row.urn == expected_urn
```

This ensures no future change to URN format or URN-building code silently breaks the DB.

### 4.3 Invariant C: GOID hash consistent with descriptor (optional but strong)

Since `compute_goid` uses the same descriptor used to build URN, you can also test:

```python
from codeintel.graphs.compute.goid import compute_goid

def test_core_goids_hash_roundtrip(gateway: StorageGateway) -> None:
    con = gateway.ibis.con
    goids = con.table("core.goids").limit(1000)
    df = goids.execute()

    for row in df.itertuples(index=False):
        desc = GoidDescriptor(
            repo=row.repo,
            commit=row.commit,  # note: commit included here
            language=row.language,
            rel_path=row.rel_path,
            kind=row.kind,
            qualname=row.qualname,
            start_line=row.start_line,
            end_line=row.end_line,
        )
        expected_goid = compute_goid(desc)
        assert row.goid_h128 == expected_goid
```

This ensures GOIDs in DB are derived exactly from the descriptor.

### 4.4 Invariant D (soft): Cross-commit URN stability

Because URN doesn’t include commit, you can assert:

* For a given `(repo, urn)` that appears in multiple commits, the **AST qualname and module path** from crosswalk rows are consistent (or at least changes are flagged).

**Ibis-like test:**

```python
def test_core_goid_urn_stability_across_commits(gateway: StorageGateway) -> None:
    con = gateway.ibis.con
    xwalk = con.table("core.goid_crosswalk")

    # Only URNs that appear in more than one commit
    multi_commit = (
        xwalk.group_by(["repo", "goid"])
        .aggregate(commit_count=xwalk.commit.nunique())
        .filter(lambda t: t.commit_count > 1)
    )

    # Join back to get all rows for these URNs
    mc_rows = multi_commit.join(
        xwalk,
        ["repo", "goid"],
    ).execute()

    if mc_rows.empty:
        return

    # Pandas check: for each (repo, goid), is ast_qualname consistent?
    grouped = mc_rows.groupby(["repo", "goid"])
    for _, group in grouped:
        # if they differ, that might be a rename or refactor; warn, don't necessarily fail
        if group["ast_qualname"].nunique() > 1:
            # depending on how strict you want to be:
            # assert False, f"URN {goid} changed ast_qualname across commits"
            # or just log/track
            pass
```

This is more of a **QA / monitoring invariant** than a hard enforcement, but it documents your expectations about URN stability across commits.

---

## 5. Property-based testing for `core.goids` + crosswalk

Finally, we can use Pandera + Hypothesis to generate **synthetic datasets** for goids and crosswalk and test invariants over them.

### 5.1 Single-table: `CORE_GOIDS_SCHEMA` strategy

```python
# tests/core/test_core_goids_pandera_properties.py

from __future__ import annotations

import pandas as pd
from hypothesis import given, settings
from pandera import hypothesis as pah

from codeintel.storage.pandera_schemas import CORE_GOIDS_SCHEMA

goids_strategy = pah.dataframe_strategy(CORE_GOIDS_SCHEMA)

@settings(max_examples=20)
@given(df=goids_strategy)
def test_core_goids_hash_range_property(df: pd.DataFrame) -> None:
    # This is already enforced by schema, but we can double-check logically:
    assert (df["goid_h128"] >= 0).all()
    assert (df["goid_h128"] < 10**38).all()
```

### 5.2 Cross-table: goids & crosswalk together

Pandera is DataFrame-centric, so for cross-table properties we’ll typically:

* Generate one DataFrame via strategy for each schema,
* Seed them so they can logically join, OR
* Use real DB data in integration tests.

For a simple synthetic example, you could generate:

* a `goids_df` from `CORE_GOIDS_SCHEMA`,
* then derive a `xwalk_df` by taking a subset of columns and adding crosswalk fields.

```python
# tests/core/test_goids_crosswalk_properties.py

from __future__ import annotations

import pandas as pd
from hypothesis import given, settings
from pandera import hypothesis as pah

from codeintel.storage.pandera_schemas import CORE_GOIDS_SCHEMA, CORE_GOID_CROSSWALK_SCHEMA

goids_strategy = pah.dataframe_strategy(CORE_GOIDS_SCHEMA)

@settings(max_examples=10)
@given(goids_df=goids_strategy)
def test_core_goids_crosswalk_roundtrip(goids_df: pd.DataFrame) -> None:
    if goids_df.empty:
        return

    # Build a synthetic crosswalk from goids_df
    # One simple mapping: copy URN -> goid, language, rel_path -> file_path, etc.
    xwalk_df = pd.DataFrame(
        {
            "repo": goids_df["repo"],
            "commit": goids_df["commit"],
            "goid": goids_df["urn"],
            "lang": goids_df["language"],
            "module_path": goids_df["rel_path"].apply(lambda p: p.replace("/", ".").removesuffix(".py")),
            "file_path": goids_df["rel_path"],
            "start_line": goids_df["start_line"],
            "end_line": goids_df["end_line"],
            "scip_symbol": None,
            "ast_qualname": goids_df["qualname"],
            "cst_node_id": None,
            "chunk_id": None,
            "symbol_id": None,
            "updated_at": goids_df["created_at"],
        }
    )

    # Validate that this synthetic crosswalk conforms to its schema
    from codeintel.storage.pandera_schemas import CORE_GOID_CROSSWALK_SCHEMA

    CORE_GOID_CROSSWALK_SCHEMA.validate(xwalk_df)

    # Cross-invariant: goid URN is present in goids_df for each row
    urn_set = set(goids_df["urn"])
    assert set(xwalk_df["goid"]).issubset(urn_set)
```

This doesn’t test the “real” ingest pipeline, but it gives you:

* a “constructor pattern” for valid + consistent goids & crosswalk rows,
* a template for how to build more sophisticated cross-table property tests later.

---

## 6. What this gives you for `core.goids` as a “tricky dataset”

With the above in place, you have:

* **Rich Pandera schemas** for:

  * structural invariants (types, ranges),
  * uniqueness constraints,
  * simple URN pattern checks.

* **Ibis views** that:

  * combine goids & crosswalk in one canonical join,
  * surface mismatches as a view (`core.v_goid_crosswalk_mismatches`).

* **Cross-table tests** that:

  * verify every crosswalk row has a matching goid row,
  * ensure URN and goid hash are consistent with the descriptor fields,
  * optionally monitor URN stability across commits.

* **Property-based tests** that:

  * treat the schema as the generator of valid data,
  * test invariants for any legal input, not just a few examples.

This makes `core.goids` + `core.goid_crosswalk` your **reference standard** for tricky, identity-heavy datasets: it demonstrates how to encode deep invariants across both tables and time (commits), and gives the AI programmer a clear pattern to follow for other complex datasets (like `analytics.function_hotspots`, history tables, and cross-dataset views).

If you want, we can next sketch a similar “rich invariants” treatment for `analytics.function_hotspots` that ties back explicitly to `analytics.function_metrics` + `history.git_churn` and encodes “risk = f(complexity, churn, coverage)” and invariants about that relationship.


