Here’s a deep, FAISS‑style technical tour of **sqlglot** tuned for someone building LLM + data/SQL systems. I’ll mirror the “what’s actually inside” structure you used for FAISS. 

---

## 0. Mental model & quick facts

At a high level, **sqlglot** is:

* A **hand‑written tokenizer + recursive‑descent parser** that builds a **dialect‑agnostic AST** of SQL. ([SQLGlot][1])
* A **transpiler** that can read and write **~30+ dialects** (DuckDB, Presto/Trino, Spark/Databricks, Snowflake, BigQuery, etc.). ([GitHub][2])
* A rule‑based **optimizer** that operates *directly on the AST* (not on a separate logical plan), with a configurable pipeline of rewrite rules. ([SQLGlot][3])
* A small **SQL engine** (planner + Python execution backend) that can run SQL on in‑memory Python tables (dicts / lists of rows) and passes all TPC‑H queries (for small data). ([SQLGlot][1])
* A **lineage & metadata** engine (column‑level, including subqueries, pivots, CTEs) built on AST + scope annotations. ([SQLGlot][4])

The core design choice: *everything stays as SQL AST as long as possible* and almost all analysis/optimizations happen there. That’s what makes it ideal glue for LLM‑driven SQL tools: you can parse → analyze → rewrite → re‑emit SQL in any dialect with very little loss.

---

## 1. Core user‑level operations (what you actually call)

At the top level you mostly interact with five things:

| Category           | Functions / classes                                            | What it gives you                                        |
| ------------------ | -------------------------------------------------------------- | -------------------------------------------------------- |
| Parse              | `parse`, `parse_one`, `maybe_parse`                            | SQL → AST (`Expression` tree)                            |
| Transpile & format | `transpile`, `Expression.sql()`                                | AST → SQL in target dialect, with pretty‑printing        |
| Optimize           | `optimizer.optimize`, optimizer rules                          | Canonical / simplified / pushed‑down SQL AST             |
| Execute            | `executor.execute`, `PythonExecutor`                           | Run SQL against Python tables (for tests, CI, mock data) |
| Lineage / metadata | `lineage.lineage`, AST traversal (`find_all`, `walk`, `Scope`) | Column‑level lineage, table usage, etc.                  |

Basic shapes:

```python
from sqlglot import parse_one, transpile, exp
from sqlglot.optimizer import optimize
from sqlglot.lineage import lineage

# Parse in source dialect
ast = parse_one("SELECT * FROM foo", read="spark")

# Transpile to Snowflake
sql_snowflake = transpile("SELECT * FROM foo", read="spark", write="snowflake")[0]

# Optimize for canonical form (e.g. for equivalence / linting)
optimized = optimize(ast, dialect="spark")

# Column lineage
line = lineage("total_revenue", sql="SELECT total_revenue FROM ...", dialect="snowflake")
```

---

## 2. Parsing & AST: how sqlglot represents SQL

### 2.1 Tokenizer & parser

* SQLGlot implements a **handwritten tokenizer** that converts SQL text into a stream of tokens with metadata (line / column, comments, etc.). ([SQLGlot][1])
* It then uses a **recursive‑descent parser** configurable per dialect. The parser consumes tokens sequentially and builds a single **AST root node** (usually a `Query`/`Select` tree). ([SQLGlot][1])

Key aspects for LLM tooling:

* **Dialect‑aware parsing**: `parse_one(sql, read="spark")` parses using the Spark dialect; if you omit `read`, it uses the “SQLGlot dialect”, a superset used as a common intermediate. ([GitHub][2])
* The parser aims to **accept “real‑world messy SQL”** (including warehouse‑specific quirks) and normalize it into a consistent AST.

### 2.2 Expression tree (AST model)

Every node is a subclass of `sqlglot.expressions.Expression` (often imported as `exp`). ([SQLGlot][5])

Important properties/methods you’ll actually use:

* **Structure / arguments**

  * `expression.args` is a dict of named child slots (`{"this": ..., "expressions": [...], "joins": [...]}` etc).
  * Many nodes have shortcuts: `expression.this`, `expression.expressions`, `expression.alias_or_name`, etc.
* **Traversal**

  * `expression.find_all(exp.Column)` – yield all column nodes in the subtree.
  * `expression.walk()` – DFS over the tree.
* **Mutation / rewriting**

  * `expression.transform(fn)` – apply a visitor function to every node (returns a new tree by default).
* **Round‑trip**

  * `expression.sql(dialect="snowflake", pretty=True)` – emit SQL text in a given dialect.

Example: extracting columns & tables from arbitrary SQL:

```python
from sqlglot import parse_one, exp

ast = parse_one("SELECT a + 1 AS x FROM db.foo f JOIN bar b ON f.id = b.id")

tables = {t.name for t in ast.find_all(exp.Table)}
cols   = {c.alias_or_name for c in ast.find_all(exp.Column)}
```

This AST‑centric model is also what powers lineage, canonicalization, and your own rewrite passes.

### 2.3 AST builder / “Python SQL” DSL

The same `Expression` types also act as a **builder DSL**:

```python
from sqlglot import exp

query = (
    exp.select(exp.alias_("a", "x"))
       .from_("foo", alias="f")
       .join("bar", on="f.id = bar.id")
       .where("f.a > 1")
)

print(query.sql(pretty=True))
```

This gives you:

* A way to **programmatically build SQL safely**, without string concatenation.
* An easy target for **LLM‑to‑AST generation**: you can have the LLM emit Python code that calls `exp.select`, etc., then let sqlglot turn that into concrete dialect SQL.

The **AST primer** linked from the README is the canonical guide to these builders and their argument semantics. ([GitHub][2])

---

## 3. Dialects & transpiler

### 3.1 Dialect abstraction

Sqlglot’s central abstraction is a `Dialect`:

* A dialect bundles **Tokenizer**, **Parser**, and **Generator** behaviors for a specific engine (e.g. Spark, Snowflake, BigQuery, DuckDB). ([GitHub][2])
* Each dialect can override:

  * Token rules (reserved words, operators, comments).
  * Parse productions for ambiguous constructs (e.g. `QUALIFY`, `LIMIT` vs `TOP`).
  * **Generator** methods that turn specific `Expression` types into target‑dialect SQL.

This lets you:

* **Parse in dialect A** (`read="spark"`) and **emit dialect B** (`write="snowflake"`) while preserving semantics as much as possible.
* Maintain one **canonical AST** and re‑target it across engines.

### 3.2 `transpile` and formatting

The main convenience API:

```python
from sqlglot import transpile

sqls = transpile(
    sql=query,
    read="snowflake",
    write="bigquery",
    pretty=True,
)
print(sqls[0])
```

* Returns a **list of SQL strings** because your input may contain multiple statements. ([Medium][6])
* Handles:

  * Identifier quoting differences (`"foo"` vs `foo` vs `[foo]`).
  * Function name / signature rewrites (e.g., `DATE_TRUNC`, JSON functions).
  * Type differences (e.g., `VARIANT` vs `JSON`).
  * Syntax sugar (QUALIFY, lateral views, etc.) via transforms and generator overrides. ([Tobiko Data][7])

For pure formatting in a single dialect you can either:

* Use `Expression.sql(pretty=True)` directly; or
* `transpile(sql, write="snowflake", pretty=True)[0]` with `read=write` if you want to normalize/format warehouse SQL.

### 3.3 Generator transforms & `transforms.preprocess`

For hard dialect mismatches, `sqlglot.transforms.preprocess` lets you define **pre‑generation transforms**:

```python
from sqlglot import transforms, exp

def my_custom_transform(expression: exp.Expression) -> exp.Expression:
    # e.g., rewrite some JSON path function into a generic canonical form
    ...

MyDialect.Generator.TRANSFORMS[exp.JSONExtract] = transforms.preprocess([my_custom_transform])
```

This:

* Chains arbitrary `Expression -> Expression` functions before calling the generator for that expression type. ([SQLGlot][8])
* Is heavily used by SQLMesh and related tooling to handle tricky differences like array/JSON semantics across engines. ([Tobiko Data][7])

---

## 4. Optimizer: AST‑level canonicalization & rewrites

### 4.1 The optimizer pipeline

The top‑level API:

```python
from sqlglot.optimizer import optimize

optimized = optimize(
    expression_or_sql,
    schema=my_schema,        # optional
    dialect="spark",
    db="analytics",
    catalog="prod",
    rules=None,              # or custom sequence
)
```

Under the hood, `optimize`:

1. Ensures `schema` is a `Schema` object (supports `{table: {col: type}}`, `{db: {table: ...}}`, or `{catalog: {db: ...}}`). ([SQLGlot][3])
2. Parses the input to an AST if it’s a string (`exp.maybe_parse`). ([SQLGlot][3])
3. Runs a **sequence of rules** (by default `RULES`) over the AST, in order. Rules include: ([SQLGlot][3])

   * `qualify` – fully qualify tables, alias them, expand stars.
   * `pushdown_projections` – eliminate unused columns and push projections down.
   * `normalize` – normalize predicates, e.g. convert to CNF.
   * `unnest_subqueries` – convert certain correlated subqueries into joins.
   * `pushdown_predicates` – push filters as far down as possible.
   * `optimize_joins` – simplify/canonicalize join conditions.
   * `eliminate_subqueries`, `merge_subqueries`, `eliminate_joins`, `eliminate_ctes` – structural simplifications.
   * `quote_identifiers` – consistent quoting.
   * `annotate_types` – infer types based on schema and function definitions.
   * `canonicalize` – canonical ordering/structure for equivalence testing.
   * `simplify` – boolean & arithmetic simplification (`1+1 → 2`, etc).

The key point: this is **AST‑in, AST‑out**. You get a semantically equivalent but simpler/canonical SQL tree.

> The docs explicitly warn *“Do not remove `qualify` from the sequence of rules unless you know what you’re doing!”* because many later rules assume fully qualified tables/columns. ([SQLGlot][3])

### 4.2 Why AST‑level optimization is unusual

Most engines do:

> SQL → AST → logical plan → optimize plan → physical plan → execute.

SQLGlot instead does:

> SQL → AST → **optimize AST directly** → optional logical plan → execute.

Motivation (from the engine write‑up): ([SQLGlot][1])

* Easier to **debug** and **validate** optimizations when both input and output are SQL ASTs; you can print both as SQL and diff them.
* You can apply rules **a la carte** for tooling use‑cases (e.g. just run `normalize` and `simplify` to get canonical SQL for equivalence).
* You get a canonical SQL representation that makes **semantic equivalence** checks practical (critical for text‑to‑SQL evaluation, metamorphic testing, etc.).

Physical optimizations (join ordering, cost‑based decisions) are intentionally *not* handled here; those are seen as engine‑level concerns. ([SQLGlot][1])

### 4.3 Best practices for advanced usage

For expert usage, treat `optimize` and its rules as a **scripting toolkit**:

* For **canonicalization / equivalence** (e.g., comparing LLM SQL to reference):

  * Run a subset of rules like `qualify`, `normalize`, `simplify`, `canonicalize`.
  * Emit SQL and compare normalized forms instead of raw strings.

* For **query linting**:

  * Use `qualify` first, then pattern‑match problem shapes on the AST (e.g., Cartesian joins, `SELECT *` on large tables).

* For **house‑style rewrites**:

  * Compose your own rules using `Expression.transform` and feed them into `rules=`.

* For **schema‑aware safety checks** (e.g., for LLM‑generated SQL):

  * Provide a `Schema` and rely on `annotate_types` to catch type mismatches before hitting the warehouse. ([SQLGlot][3])

---

## 5. Execution engine: planning & running SQL on Python data

The `sqlglot.executor` module implements a small **logical planner + Python execution backend**. ([SQLGlot][1])

### 5.1 Top‑level `execute` API

```python
from sqlglot.executor import execute

result_table = execute(
    sql="SELECT a, SUM(b) AS s FROM t GROUP BY a",
    schema={"t": {"a": "int", "b": "int"}},  # or omitted, see below
    read="snowflake",
    tables={"t": [{"a": 1, "b": 2}, {"a": 1, "b": 3}]},
)
```

Flow (simplified): ([SQLGlot][1])

1. **Schema inference** if you don’t pass one:

   * It flattens the `tables` mapping, inspects sample values, and calls `annotate_types(exp.convert(value))` to infer SQL types from Python values. ([SQLGlot][1])
2. `optimize(sql, schema=..., dialect=read)` to get an optimized AST.
3. Build a simple **logical plan** (DAG) with nodes like Scan, Sort, Aggregate, Join. ([SQLGlot][1])
4. `PythonExecutor` walks the plan and:

   * Uses a “Python SQL dialect” to turn expressions into Python code (`scope["x"] + 1`) and compiles them via `compile`. ([SQLGlot][9])
   * Applies operations over an in‑memory columnar `Table` abstraction.

Notes:

* The engine is deliberately not aimed at big data; it’s optimized for **low overhead** and **small test datasets** (CI, unit tests for SQL pipelines, local dev). ([SQLGlot][1])
* The planner is intentionally simple (Scan, Sort, Set, Aggregate, Join) to be easy to reason about and potentially replace with a higher‑performance backend (e.g. numpy/Polars/Arrow) later.

For LLM‑driven systems this is invaluable for:

* **Offline validation of generated SQL** against small synthetic datasets.
* **Unit tests** for SQL transformations (including cross‑engine compatibility) without provisioning the target warehouse.

---

## 6. Lineage & scope: column‑level graphs

### 6.1 `lineage.lineage` API

`sqlglot.lineage.lineage` builds a **column‑level lineage graph** from SQL. ([SQLGlot][4])

```python
from sqlglot.lineage import lineage

node = lineage(
    column="total_revenue",
    sql="SELECT total_revenue FROM sales_summary",
    schema=my_schema,
    dialect="snowflake",
)

for n in node.walk():
    print(n.name, n.expression.sql())
```

Important details: ([SQLGlot][4])

* `Node` is a frozen dataclass: `(name, expression, source, downstream, source_name, reference_node_name)`.

* `lineage(...)` pipeline:

  1. `maybe_parse(sql, dialect)` → AST.
  2. `normalize_identifiers` on the requested column (handles quoting, casing).
  3. Optional `exp.expand` to inline named query sources (if `sources` mapping is provided).
  4. `qualify.qualify(...)` with schema → ensures all columns/tables are unambiguous.
  5. `build_scope(expression)` → a set of `Scope` objects representing each SELECT, CTE, derived table, etc.
  6. `to_node` recursively walks scopes:

     * Locates the relevant SELECT item.
     * Traces downstream columns through:

       * Subqueries and derived tables.
       * Set operations (`UNION`, etc.).
       * Pivots (special logic to map pivoted columns back to underlying expressions).
     * Emits `Node.downstream` edges for each contributing column.

* `Node.to_html()` renders an interactive **vis.js graph** of the lineage (nice for tooling). ([SQLGlot][4])

Toby’s LinkedIn post emphasizes that SQLGlot’s lineage is **AST‑based with metadata**, *not* logical‑plan‑based, specifically so that the same machinery can be used for dialect‑agnostic rewriting and then re‑emitting SQL. ([LinkedIn][10])

### 6.2 Real‑world lineage usage

Downstream tools using SQLGlot lineage:

* dbt column lineage extractors and data‑quality tools (e.g. Elementary + SQLGlot) use this to produce column‑level lineage graphs inside dbt projects. ([Medium][11])
* Recce’s column‑lineage tooling uses SQLGlot’s AST + lineage to resolve column dependencies across models. ([Recce Blog][12])

LLM‑specific angle:

* You can **interpret LLM‑generated SQL** to see *which source columns feed a given output column*, or to limit the surface area of data quality checks per query.
* You can also combine lineage with query logs to analyze **column usage** over time for schema refactoring. ([Medium][13])

---

## 7. Advanced AST plumbing: transforms, visitors, and builders

### 7.1 Expression traversal & transforms

Patterns you’ll use constantly:

```python
from sqlglot import parse_one, exp

ast = parse_one("SELECT a, b FROM x WHERE a + 1 > 2")

# Visitor: collect functions
funcs = [f for f in ast.find_all(exp.Func)]

# Transformer: rewrite COUNT(*) to COUNT(1)
def rewrite_count_star(node):
    if isinstance(node, exp.Count) and node.expressions and node.expressions[0].is_star:
        return node.copy().replace(node.expressions[0], exp.Literal.number(1))
    return node

rewritten = ast.transform(rewrite_count_star)
```

Because all optimizer rules are themselves **Expression → Expression** functions, you can:

* Use them directly (e.g. call `eliminate_ctes(ast)` yourself).
* Compose them with your own domain‑specific rules.

Medium posts show using individual optimizer rules (`eliminate_ctes`, `eliminate_joins`, `eliminate_subqueries`) to simplify complex queries for analysis. ([Medium][6])

### 7.2 AST diff & canonical SQL

The README advertises **AST introspection and diff** features: you can compare two ASTs (or canonical SQL forms) to check if they’re semantically equivalent up to the current optimizer rules. ([GitHub][2])

This is particularly useful for:

* **Text‑to‑SQL evaluation**: compare LLM SQL against reference via canonicalization, rather than strict string match.
* **Metamorphic testing** (e.g. rewriting queries but expecting same plan/behavior).

---

## 8. Integrations & ecosystem

Sqlglot is increasingly the **“Calcite of Python”** – a shared SQL front‑end used by lots of projects. Examples: ([SQLGlot][1])

* **Ibis**: uses the **Python SQL expression builder** + optimizer/planner to convert SQL into dataframe operations and back. ([SQLGlot][1])
* **SQLMesh**: built around SQLGlot for parsing, optimizing, and transpiling SQL models across engines; uses transforms and type helpers (`sqlglot.expressions.DataType`) to map pandas/Arrow dtypes to warehouse types. ([Tobiko Data][7])
* **SQLFrame**: implements the **PySpark DataFrame API without Spark**, emitting SQL via SQLGlot; lets you run PySpark‑style code directly on databases. ([Reddit][14])
* **Splink**, **Quokka**, **mysql‑mimic**: use SQLGlot to parse/optimize/transpile queries in their own engines or protocol implementations. ([SQLGlot][1])
* **dbt ecosystem tools**: dbt column lineage extractors and Recce use SQLGlot lineage for fine‑grained dependency graphs. ([Reddit][15])
* **Lumen / Panel**: expose `to_sql(expression)` helpers built on SQLGlot to convert expressions to SQL. ([Lumen][16])

For an LLM/AI stack, this means:

* You can treat SQLGlot as the **single source of truth for SQL handling** and hook into the same AST across:

  * LLM SQL generation / validation
  * ETL/dataframe code generation (via SQLFrame/Ibis)
  * Lineage and quality checks
  * Cross‑warehouse migration and dev‑prod parity.

---

## 9. Best practices for expert use (especially with LLMs)

### 9.1 Dialect handling & safety

* **Always set `read=`** when parsing warehouse SQL, or you may hit subtle failures because the default “sqlglot dialect” is a superset with different rules. ([GitHub][2])
* For LLM‑generated SQL:

  1. `parse_one(sql, read="your_warehouse")` inside a try/except → produce structured parse errors for the user/model.
  2. Run a conservative subset of optimizer rules (`qualify`, `annotate_types`, `simplify`) to catch type and ambiguity issues early.
  3. Optionally re‑emit SQL with `.sql(pretty=True)` to normalize formatting before sending to your warehouse.

### 9.2 Canonicalization for evaluation & caching

* Use `optimize(..., rules=[qualify, normalize, canonicalize, simplify, ...])` to get **canonical SQL** that:

  * Normalizes predicate structure (CNF).
  * Standardizes identifiers and aliases.
  * Simplifies constant expressions. ([SQLGlot][1])

You can then:

* Compute a digest (e.g. hash) of canonical SQL for:

  * **Result caching** across semantically equivalent queries.
  * **Text‑to‑SQL evaluation** without being sensitive to superficial differences.

### 9.3 Building domain‑specific transforms

For LLM‑heavy workflows, domain‑specific transforms are gold:

* **Guardrails**: e.g. “no `DELETE`/`UPDATE` allowed”, “no cross‑schema access”:

  * Parse SQL → traverse AST → reject or rewrite disallowed operations.
* **Automatic hints / rewrites**:

  * Add `QUALIFY` or additional filters when queries are too broad.
  * Expand macros (e.g. `LATEST_PARTITION(table)` → actual partition filter).

Use `Expression.transform` plus the optimizer’s existing rules as building blocks, or wrap them via `transforms.preprocess` if you need to hook into dialect‑specific generation. ([SQLGlot][8])

### 9.4 Using the Python engine in CI & REPLs

For each LLM‑generated query:

1. Build a **tiny synthetic dataset** (Python dict of tables).
2. Run `executor.execute` with `read` set to the warehouse dialect.
3. Compare results against a reference engine or expected shape.

This catches **semantic mismatches** (wrong joins, wrong filters) without involving the actual warehouse. ([SQLGlot][1])

---

## 10. Example “recipes” you can drop into a project

### 10.1 LLM SQL validator + canonicalizer

```python
from sqlglot import parse_one
from sqlglot.optimizer import optimize
from sqlglot.expressions import Expression

def validate_and_canonicalize(sql: str, dialect: str, schema: dict | None = None) -> Expression:
    try:
        ast = parse_one(sql, read=dialect)
    except Exception as e:
        raise ValueError(f"ParseError: {e}") from e

    # Minimal, safe rules
    from sqlglot.optimizer import normalize, qualify, simplify, canonicalize

    rules = [qualify, normalize, simplify, canonicalize]
    return optimize(ast, schema=schema, dialect=dialect, rules=rules)
```

Use this to:

* Return structured “what went wrong” messages to the LLM.
* Store canonical AST / SQL for downstream evaluation.

### 10.2 “Explain this query” via lineage

```python
from sqlglot.lineage import lineage

def column_sources(sql: str, column: str, dialect: str, schema: dict | None = None):
    root = lineage(column, sql=sql, dialect=dialect, schema=schema)
    leaves = [n for n in root.walk() if not n.downstream]
    return [leaf.name for leaf in leaves]
```

This tells you **which base columns** feed an output column; plug it into an explainer UI or as context to an LLM.

### 10.3 PySpark‑like DataFrame API using SQLGlot (micro‑SQLFrame)

```python
from sqlglot import exp

class S:
    def __init__(self, expr: exp.Expression):
        self.expr = expr

    @classmethod
    def table(cls, name):
        return cls(exp.select("*").from_(name))

    def select(self, *cols):
        return S(self.expr.select(*[exp.column(c) for c in cols], append=False))

    def where(self, predicate):
        return S(self.expr.where(predicate))

    def sql(self, **opts):
        return self.expr.sql(**opts)

df = S.table("orders").select("customer_id").where("total > 100")
print(df.sql(pretty=True))
```

This is essentially what projects like SQLFrame do at scale: build a DataFrame API on top of SQLGlot and let the generator handle the SQL. ([Reddit][14])

---

If you’d like, next step we can zoom into **specific internals**, e.g.:

* The `Scope` system used by both optimizer and lineage.
* How to implement a **custom dialect** for a non‑standard engine.
* How to wire SQLGlot into a full LLM text‑to‑SQL evaluation harness (canonicalization, diff, runtime checks).

[1]: https://sqlglot.com/sqlglot/executor.html "sqlglot.executor API documentation"
[2]: https://github.com/tobymao/sqlglot "GitHub - tobymao/sqlglot: Python SQL Parser and Transpiler"
[3]: https://sqlglot.com/sqlglot/optimizer/optimizer.html "sqlglot.optimizer.optimizer API documentation"
[4]: https://sqlglot.com/sqlglot/lineage.html "sqlglot.lineage API documentation"
[5]: https://sqlglot.com/sqlglot/expressions.html?utm_source=chatgpt.com "sqlglot.expressions API documentation"
[6]: https://medium.com/%40suryaiyer95/navigating-sql-complexity-with-sqlglot-a-game-changer-in-data-analytics-73c813adc281?utm_source=chatgpt.com "Navigating SQL Complexity with SQLGlot: A Game- ..."
[7]: https://tobikodata.com/blog/transpiling-sql1?utm_source=chatgpt.com "Transpiling SQL #1: JSON Paths"
[8]: https://sqlglot.com/sqlglot/transforms.html?utm_source=chatgpt.com "sqlglot.transforms API documentation"
[9]: https://sqlglot.com/sqlglot/executor/python.html?utm_source=chatgpt.com "sqlglot.executor.python API documentation"
[10]: https://www.linkedin.com/posts/toby-mao_the-way-sqlglot-computes-column-level-lineage-activity-7166683013311852547-on21?utm_source=chatgpt.com "Tobias (Toby) Mao's Post"
[11]: https://medium.com/%40sendoamoronta/linaje-de-columna-y-calidad-de-datos-en-dbt-con-sqlglot-elementary-36f0c80b8421?utm_source=chatgpt.com "Column-Level Lineage and Data Quality in dbt with ..."
[12]: https://blog.reccehq.com/column-level-lineage-internals?utm_source=chatgpt.com "Column-Level Lineage Approach - Recce | Blog"
[13]: https://medium.com/%40pizzini.alessandro/monitoring-column-usage-with-sqlglot-a-journey-towards-data-warehouse-optimization-299667aa44dc?utm_source=chatgpt.com "Monitoring Column Usage with SQLGlot: A Journey ..."
[14]: https://www.reddit.com/r/dataengineering/comments/1cxaeh0/open_source_turning_pyspark_into_a_universal/?utm_source=chatgpt.com "[Open Source] Turning PySpark into a Universal ..."
[15]: https://www.reddit.com/r/dataengineering/comments/1g3zquw/introducing_the_dbt_column_lineage_extractor_a/?utm_source=chatgpt.com "A Lightweight Tool for dbt Column Lineage : r ..."
[16]: https://lumen.holoviz.org/reference/transform/SQLOverride.html?utm_source=chatgpt.com "SQLOverride type: None — Lumen 0.10.1 documentation"
