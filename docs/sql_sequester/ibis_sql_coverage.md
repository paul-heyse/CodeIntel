Below is a practical, end‑to‑end guide to **what Ibis gives you for SQL function coverage**, focusing on:

* **Built‑in expression APIs** (the “normal” Ibis methods you call), and
* **UDFs** (builtin + Python/vectorized) as escape hatches when the built‑ins don’t cover a SQL function.

I’m loosely mirroring the structure of your FAISS wheel writeup: quick mental model → major user‑level operations → detailed function families → UDF patterns + recipes. 

---

## 1. Mental model: how Ibis covers SQL functions

### 1.1 Expressions, operations, backends

* In Ibis, *everything* you write is an **expression tree** – a graph of typed nodes (operations) over tables and literals.
* Each node is an `Operation` (from `ibis.expr.operations`), such as `Add`, `Substring`, `GeoContains`, `Lag`, or `Sum`. ([Ibis][1])
* Each backend (DuckDB, BigQuery, Snowflake, Postgres, Spark, etc.) has a **compiler** that maps these operations to its SQL dialect or local execution (pandas/Polars/DataFusion). ([Ibis][2])
* Ibis aims to be a **full‑featured replacement for SQL `SELECT`** queries: projection, filtering, joins, aggregation, windowing, expressions, etc. ([Ibis][3])

When you call `col.abs()` or `col.str.contains("foo")`, you aren’t calling a Python function on data; you’re creating an **`Operation` node** (`Abs`, `StringContains`) that later compiles to whatever the backend’s SQL equivalent is (`ABS(col)`, `col LIKE '%foo%'`, `POSITION`, or `REGEXP_CONTAINS`, depending on backend). ([Ibis][4])

### 1.2 “Built‑ins” vs UDFs

* **Built‑ins** = methods & top‑level functions in the **Expression API**:

  * Table expressions, numeric/boolean, strings, temporal, collections, JSON, geospatial, and misc types. ([Ibis][5])
* **UDFs** = the `ibis.udf` family that lets you define:

  * **Scalar UDFs**: `@ibis.udf.scalar.builtin`, `@ibis.udf.scalar.pandas`, `@ibis.udf.scalar.pyarrow`, `@ibis.udf.scalar.python`. ([Ibis][6])
  * **Aggregate UDFs (experimental)**: `@ibis.udf.agg.builtin`. ([Ibis][7])

The key idea for **SQL function coverage**:

1. Prefer **canonical expression APIs** → portable, tested, type‑checked.
2. If a database has a function that Ibis doesn’t expose, use a **builtin UDF decorator** to call it by name.
3. If you need logic that *no SQL function* has, use **pandas/pyarrow/python UDFs** on backends that support them.
4. For truly arbitrary SQL (complex CTE/DDL/queries), use `Table.sql`, `Backend.sql`, or `Backend.raw_sql` and compose back into Ibis. ([Ibis][8])

Finally, there’s an **Operation Support Matrix** which shows exactly which operations each backend implements, across ~20 backends (19 SQL). ([Ibis][9])

---

## 2. Major user‑level SQL operations (what Ibis covers natively)

At the “SELECT statement” level, Ibis covers essentially all relational operations you’d expect from SQL:

| SQL concept                    | Ibis pattern (table expressions)                        | Notes                                                                   |
| ------------------------------ | ------------------------------------------------------- | ----------------------------------------------------------------------- |
| `FROM`, `SELECT`               | `t = con.table("foo"); t[cols]` or `t.select(...)`      | Projection & renaming. ([Ibis][10])                                     |
| `WHERE`                        | `t.filter(predicate)`                                   | Boolean expressions built from value methods. ([Ibis][11])              |
| `GROUP BY` + aggregates        | `t.group_by(keys).aggregate(metrics)`                   | Maps to `GROUP BY` + `agg()` operations (Sum, Mean, etc.). ([Ibis][10]) |
| `HAVING`                       | `t.group_by(...).aggregate(...).filter(having_expr)`    | Post‑aggregation filter.                                                |
| `ORDER BY`, `LIMIT`            | `t.order_by(keys).limit(n)`                             | Straightforward mapping. ([Ibis][11])                                   |
| Joins (`INNER`/`LEFT`/etc.)    | `t1.join(t2, on=..., how="left")`                       | Many join types; compiled to vendor dialect. ([Ibis][10])               |
| UNION/INTERSECT/EXCEPT         | `t1.union(t2)`, `t1.intersect(t2)`, `t1.difference(t2)` | Set operations. ([Ibis][10])                                            |
| Window functions (`OVER(...)`) | `expr.over(window)` / `ibis.window(...)`                | LEAD, LAG, NTILE, cumulative windows, etc. ([Ibis][11])                 |

So at the **relational** level, Ibis is essentially a full‑coverage SELECT builder; the remaining question is **value‑level function coverage**, which is where built‑ins + UDFs come in.

---

## 3. Built‑in function families (by data type)

Below is a **compressed taxonomy** of Ibis built‑ins and roughly which SQL function buckets they cover. I’ll focus on the “high‑value” families and point you to the exact reference pages.

### 3.1 Numeric & boolean expressions

**NumericValue / NumericColumn** methods map to the usual SQL math + statistics, plus some extra analytics. ([Ibis][4])

**Scalar math / numeric transforms**

Examples of methods on `NumericValue`:

* `abs`, `negate` → `ABS`, unary minus.
* `exp`, `ln`, `log`, `log2`, `log10` → `EXP`, `LN/LOG`, `LOG2`, `LOG10`.
* `ceil`, `floor`, `round` → `CEIL`, `FLOOR`, `ROUND`.
* `sqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`.
* `degrees`, `radians`, `sign`, `clip`.

These are backed by `numeric` operations like `Abs`, `Exp`, `Ln`, `Round`, `Sin`, `Cos`, etc., in the operations reference, which the backend compilers translate to dialect‑specific functions (`CEIL`, `TRUNC`, `POWER`, etc.). ([Ibis][1])

**Column‑level statistics / reductions**

On `NumericColumn` (and generally reductions):

* `sum`, `mean`, `std`, `var`, `min`, `max`, `median`, `quantile`, `approx_quantile`, `approx_median`, `approx_count_distinct`. ([Ibis][4])
* `corr`, `cov` → `CORR`, `COVAR_POP`/`COVAR_SAMP` equivalents.
* `histogram`, `bucket` → histogram/binning; mapped to vendor equivalents where available.

**Integer / bitwise / boolean extras**

* `IntegerValue.as_interval`, `as_timestamp`, `convert_base` (base conversion). ([Ibis][4])
* `IntegerColumn.bit_and`, `bit_or`, `bit_xor` → bitwise reductions.
* `BooleanValue.ifelse`, `BooleanColumn.all`, `any`, `cumall`, `cumany`, `notall`, `notany` → CASE/IF, `BOOL_AND`, `BOOL_OR`, windowed booleans. ([Ibis][4])

Coverage: this family hits the **usual math & stats functions** in SQL plus a decent set of approximate stats (approx count distinct/quantile) when the backend supports them.

---

### 3.2 String expressions

The `StringValue` methods cover the bulk of ANSI string / regex / URL functions. ([Ibis][12])

**Basic string transforms**

* `lower`, `upper`, `capitalize`, `reverse`.
* `length`.
* `substr(start, length)`, `left`, `right`.
* `lpad`, `rpad`, `strip`, `lstrip`, `rstrip`.

These translate to `LOWER/UPPER`, `LENGTH/CHAR_LENGTH`, `SUBSTR/SUBSTRING`, `LPAD/RPAD`, `TRIM` etc.

**Search & pattern matching**

* `contains(pattern)`, `startswith`, `endswith`.
* `like(pattern)`, `ilike(pattern)`.
* Regex: `re_search`, `re_extract`, `re_replace`, `re_split`.

Backends implement these via `LIKE`, `ILIKE`, and regex functions (`REGEXP_CONTAINS`, `REGEXP_EXTRACT`, `REGEXP_REPLACE`, `REGEXP_SPLIT`, etc.). ([Ibis][12])

**Tokenization & composition**

* `split(delim)`, `repeat(n)`, `replace(old, new)`, `concat`, `join`.

**Utility / URL‑ish helpers**

Ibis exposes a bunch of small helpers that map to more specialized SQL functions, e.g.:

* `as_date`, `as_time`, `as_timestamp` (string → temporal cast).
* `ascii_str`, `convert_base`.
* URL parsing: `protocol`, `host`, `path`, `query`, `file`, `fragment`, `userinfo`, `authority`. ([Ibis][12])

These cover a surprising amount of “string munging” that often shows up in SQL data cleaning.

---

### 3.3 Temporal & interval expressions

Temporal built‑ins cover most of what you’d expect for `DATE`, `TIME`, `TIMESTAMP`, and intervals. ([Ibis][13])

**Construction / casts**

* Top‑level constructors: `ibis.date`, `ibis.time`, `ibis.timestamp`.
* Casts: `str_col.as_timestamp()`, `integer.as_timestamp()` etc. ([Ibis][13])

**Extraction**

On `TimestampValue`, `DateValue`, `TimeValue`:

* `year`, `quarter`, `month`, `week_of_year`, `day`, `day_of_year`, `hour`, `minute`, `second`, `microsecond`, `millisecond`.
* `date()`, `time()`.

These map directly to SQL datetime extract functions (`EXTRACT`, `DATE_PART`, etc.). ([Ibis][13])

**Truncation & bucketing**

* `truncate("day" | "week" | "month" | ...)` → date truncation (e.g. `DATE_TRUNC`).
* `bucket` for time bucketing.

**Epoch & formatting**

* `epoch_seconds()` → integer epoch.
* `strftime(format)` → `TO_CHAR`‑like formatting.

**Current time/value**

* Top‑level `ibis.now()`, `ibis.today()` for “current” timestamp/date (via `TimestampNow` / `DateNow` in ops). ([Ibis][13])

**Intervals**

* `IntervalValue.to_unit` for unit conversion; numeric → interval via `.as_interval()`.

Together, these cover the **standard SQL date/time functions** and a good amount of time‑series bucketing.

---

### 3.4 Collections: arrays, maps, structs

Ibis has first‑class **Array**, **Map**, and **Struct** expressions with a rich method set. ([Ibis][14])

**ArrayValue**

* Reductions: `alls`, `anys`, `sums`, `means`, `mins`, `maxs`, `modes`.
* Structural ops: `length`, `index(i)`, `contains(x)`, `concat`, `intersect`, `union`, `unique`, `flatten`, `unnest`, `sort`, `zip`, `repeat`, `filter`, `map`.

These map to array functions like `CARDINALITY`, `ARRAY_POSITION`, `ARRAY_CONCAT`, `ARRAY_INTERSECT`, `UNNEST`, etc., where the backend has them (especially BigQuery, DuckDB, Postgres, Snowflake, Trino/Starburst).

**MapValue**

* `contains(key)`, `get(key)`, `keys()`, `values()`, `length()`.

**StructValue**

* `.field` accessors and `lift` (lifting fields into columns).

Plus constructors `array()`, `map()`, `struct()` for literal / expression creation. ([Ibis][14])

Coverage: this family cleanly covers most **ARRAY / MAP / STRUCT** capabilities in modern warehouses.

---

### 3.5 JSON expressions

JSON in Ibis is modeled as a dynamic collection type (`JSONValue`) with methods for nested access and type‑safe unwrapping. ([Ibis][15])

Key pieces:

* Column type `json`.
* Subscript/index: `t.js["a"]` → `JSONGetItem(js, 'a')`.
* Chained indexing: `t.js["a"][0]` for nested arrays.
* `unwrap_as(dtype)` to cast JSON to a typed value or collection (internally using operations like `UnwrapJSONString`, `UnwrapJSONInt64`, etc.).

These map to backend JSON functions like `JSON_EXTRACT`, `JSON_VALUE`, `JSON_QUERY`, `->`, `->>`, etc., depending on dialect.

---

### 3.6 Geospatial expressions

Ibis has a **large geospatial surface** similar to PostGIS/GEOS/ST_*, via `GeoSpatialValue` and `GeoSpatialColumn`. ([Ibis][16])

Examples:

* Geometry export: `as_text`, `as_ewkt`, `as_binary`, `as_ewkb`.
* Measurements: `area`, `length`, `perimeter`, `distance`, `max_distance`.
* Topology predicates: `contains`, `within`, `covers`, `covered_by`, `intersects`, `disjoint`, `overlaps`, `touches`, `crosses`, `is_valid`, `geo_equals`, `ordering_equals`.
* Transformations: `buffer`, `intersection`, `difference`, `union`, `unary_union`, `simplify`, `transform` (CRS changes), `set_srid`.
* Point helpers: `x`, `y`, `x_min`, `x_max`, `y_min`, `y_max`, `point_n`, `start_point`, `end_point`.

Behind the scenes, these compile to vendor geospatial functions on backends that support them (e.g. BigQuery GIS, PostGIS, DuckDB extensions, Snowflake GEO), and are not available on simpler backends without geospatial support.

---

### 3.7 Generic expressions & conditionals

Via `Generic` and `Logical` operations you get a lot of glue that maps to “generic” SQL functions. ([Ibis][1])

Highlights:

* `cast`, `try_cast`, `typeof`.
* Null handling: `isnull`, `notnull`, `coalesce`, `nullif`, `identical_to`.
* CASE: `case().when(...).else_(...).end()` or `ifelse`. (`SearchedCase`, `SimpleCase` operations).
* Hashing: `hash`, `hexdigest`.
* Randomness: `random()`, `random_uuid()`.
* Misc constants: `pi`, `e`, `row_number`, `rowid`.

These map onto `CAST`, `TRY_CAST`, `COALESCE`, `NULLIF`, `CASE`, `HASH`, `MD5/SHA*`, `RAND/RANDOM`, `UUID`/`GEN_RANDOM_UUID`, etc.

---

### 3.8 Aggregations & analytic/window functions

At the expression level, **analytic** and **reduction** operations give you:

* Reductions: `sum`, `mean`, `std`, `var`, `count`, `count_distinct`, `approx_count_distinct`, `quantile`, `approx_quantile`, `argmax`, `argmin`, `group_concat`, `array_collect`, `kurtosis`, etc. ([Ibis][1])
* Analytic/window: `lag`, `lead`, `nth_value`, `row_number`, `dense_rank`, `percent_rank`, `cume_dist`, `ntile`, cumulative variants (`cumsum`, `cummean`, `cummin`, `cummax`), plus window frame helpers like `ibis.window`, `cumulative_window`, `range_window`, `trailing_range_window`. ([Ibis][11])

These compile to standard SQL window functions (`OVER(PARTITION BY ... ORDER BY ...)`) with `ROWS`/`RANGE` clauses.

For many analytics workloads, **this set plus strings + temporals covers 90–95% of functions you’d normally write by hand in SQL.**

---

## 4. UDFs: extending Ibis to full SQL function coverage

Now to the interesting part: **what happens when the built‑in API doesn’t expose a function your database has?**

Ibis’s UDF system has three major roles:

1. **Call backend built‑in functions that Ibis doesn’t model yet.**
2. **Execute custom Python logic, vectorized where possible.**
3. **(Experimental) Wrap backend aggregate functions.**

### 4.1 Scalar UDF flavors

From the `scalar-udfs` reference: ([Ibis][6])

```python
ibis.udf.scalar.builtin   # wrap a *database* builtin function
ibis.udf.scalar.pandas    # vectorized Python using pandas Series
ibis.udf.scalar.pyarrow   # vectorized Python using PyArrow Arrays
ibis.udf.scalar.python    # non-vectorized row-by-row Python
```

#### 4.1.1 `scalar.builtin`: vendor built‑in SQL function escape hatch

This is the **primary tool for function coverage**: it lets you call *any* built‑in function that the backend exposes, even if Ibis doesn’t.

Signature (simplified): ([Ibis][6])

```python
@ibis.udf.scalar.builtin(
    name="backend_function_name",  # optional
    database="...", catalog="...", # optional
    signature=((arg_types...), return_type)  # optional; only return type required
)
def my_func(a: ArgType1, b: ArgType2) -> ReturnType:
    ...
```

Key semantics:

* **Function body is ignored** – Ibis never executes or inspects it; only type annotations + decorator metadata matter. ([Ibis][17])
* At compile time, Ibis emits **`FUNCTION_NAME(args...)`** in the target SQL dialect (with `database.catalog` qualifiers if set). ([Ibis][17])
* If you omit `name`, the SQL function name defaults to the Python function name.
* You usually only need to annotate the **return type**; argument types can be inferred for builtin UDFs. ([Ibis][17])

**Example: exposing DuckDB text similarity functions** (from the docs):

```python
import ibis

@ibis.udf.scalar.builtin
def mismatches(left: str, right: str) -> int:
    ...

@ibis.udf.scalar.builtin(name="jaro_winkler_similarity")
def jw_sim(a: str, b: str) -> float:
    ...
```

These compile to `MISMATCHES(left, right)` and `JARO_WINKLER_SIMILARITY(a, b)` in DuckDB, and can be **freely composed** with other Ibis expressions (filters, joins, windows, etc.). ([Ibis][17])

**Use this when:**

* The backend has *any* useful builtin you care about (e.g., `TEXT_SIM`, `VECTOR_DISTANCE`, `HLL_CARDINALITY`, vendor‑specific ML functions, etc.), and
* You want to stay inside Ibis’s expression DAG with minimal portability guarantees (portable only to backends that share the function name/semantics).

#### 4.1.2 `scalar.pandas`: vectorized Python on pandas‑like backends

`@ibis.udf.scalar.pandas` defines a vectorized UDF that takes **pandas Series** as input. Ibis registers/executes it for backends that operate via a pandas-like execution path (pandas backend, and in some cases PySpark with pandas UDFs). ([Ibis][6])

Example:

```python
import ibis

@ibis.udf.scalar.pandas
def str_cap(x: str) -> str:
    # x is a pandas.Series
    return x.str.capitalize()

tbl = ibis.memtable({"str_col": ["a", "b", "c"]})
expr = str_cap(tbl.str_col)
```

You can also accept **Struct** / **Map** types and operate on them using pandas operations. ([Ibis][6])

Use when:

* You’re on a **local DataFrame backend** (pandas, PySpark with pandas UDFs, etc.).
* The logic is easier in Python/pandas than SQL.
* You’re OK with backend‑specific semantics (this won’t automatically port to e.g. BigQuery).

#### 4.1.3 `scalar.pyarrow`: vectorized Python on Arrow‑based backends

`@ibis.udf.scalar.pyarrow` defines UDFs over **PyArrow Arrays** (e.g., for DataFusion or arrow‑native pipelines). ([Ibis][6])

Example (from docs, using `pyarrow.compute.weeks_between`):

```python
import ibis
import pyarrow.compute as pc
from datetime import date

@ibis.udf.scalar.pyarrow
def weeks_between(start: date, end: date) -> int:
    return pc.weeks_between(start, end)
```

Use when:

* Your backend is Arrow‑centric (DataFusion, Polars+Arrow, etc.).
* You want to reuse PyArrow’s rich compute kernel library.

#### 4.1.4 `scalar.python`: non‑vectorized row‑by‑row Python

`@ibis.udf.scalar.python` defines UDFs that operate on **scalar Python values** row‑by‑row. They are explicitly documented as **likely slow**: one Python call per row. ([Ibis][6])

Example:

```python
@ibis.udf.scalar.python
def add_one_py(x: int) -> int:
    return x + 1
```

Use when:

* Data volume is small or already materialized locally.
* You need some bespoke logic and don’t care about vectorization.
* You accept that this is **not** a scalable strategy for big data.

---

### 4.2 Aggregate UDFs (experimental)

Aggregate UDFs are currently **builtin only**, i.e. they wrap backend aggregate functions not yet modeled by Ibis. ([Ibis][7])

```python
@ibis.udf.agg.builtin
def favg(a: float) -> float:
    """E.g., Kahan compensated average"""
    ...
```

This compiles to an aggregate call `FAVG(a)` (or whatever `name=` you specify). It’s marked **experimental**, but it’s the right tool for:

* vendor‑specific aggregates (e.g., approximate percentile, sketch‑based aggregates),
* custom aggregate UDFs provided by the database.

---

### 4.3 How builtin UDFs interact with SQL function coverage

Broadly:

* Any **scalar SQL function** that takes columns and returns a column can be surfaced via `scalar.builtin`.
* Any **aggregate SQL function** (`F(expr)` in `SELECT` with `GROUP BY`) can be surfaced via `agg.builtin` (where supported).
* These UDF calls are **first‑class expressions**: they can be nested inside `mutate`, `filter`, `group_by`, windows, etc., and they will show up in the generated SQL via `ibis.to_sql(...)`. ([Ibis][17])

This is how you get to “near‑complete SQL function coverage” for a backend even when Ibis hasn’t introduced canonical methods for every vendor function.

---

## 5. Raw SQL escape hatches

If a function or pattern is **too irregular or composite** for UDFs (e.g., multi‑statement CTEs, vendor‑specific syntax), you can use the “SQL strings with Ibis” APIs: ([Ibis][8])

* `Table.sql("SELECT ... FROM this_table_expr")`

  * Allows `SELECT` queries using a table expression as a FROM subquery.
* `Backend.sql("SELECT ... FROM existing_table")`

  * Arbitrary SELECT against DB tables; returns an Ibis table expression you can further transform.
* `Backend.raw_sql("CREATE TABLE ...")`

  * For DDL / non‑SELECT statements.

These let you **surgically embed vendor SQL** and still bring the result back into the Ibis world.

---

## 6. Reasoning about function coverage for a specific backend

If you want a pragmatic coverage picture for, say, DuckDB / BigQuery / Snowflake:

1. **Start with the Operation Support Matrix**

   * Filter to your backend(s) and see which operations (numeric, string, temporal, geospatial, reductions, analytics, etc.) are marked supported. ([Ibis][9])
   * This tells you what’s *already implemented* in that backend’s compiler.

2. **Cross‑check with Expression API pages**

   * Numeric, string, temporal, collections, JSON, geospatial – all list methods and examples. ([Ibis][4])
   * If a method exists and your backend supports the operation, you’re done.

3. **For missing vendor functions**

   * Check the backend’s own docs: if it has a function but Ibis doesn’t, **wrap it with `udf.scalar.builtin` or `udf.agg.builtin`**. ([Ibis][17])

4. **For custom logic**

   * Use `scalar.pandas` / `scalar.pyarrow` / `scalar.python` UDFs on suitable backends. ([Ibis][6])

5. **Only then fall back to raw SQL**

   * Use `Table.sql` or `Backend.sql` for large blobs of hand‑written SQL; treat them as transitional. ([Ibis][8])

---

## 7. Minimal “good” recipes (copy/paste)

### 7.1 Expose a backend string similarity function (builtin scalar UDF)

Example: DuckDB `jaro_winkler_similarity` wrapped for use in Ibis filters. ([Ibis][17])

```python
import ibis

@ibis.udf.scalar.builtin(name="jaro_winkler_similarity")
def jw_sim(a: str, b: str) -> float:
    ...

con = ibis.connect("duckdb://")
pkgs = ibis.read_parquet("packages.parquet")

similar_to_pandas = pkgs.filter(
    jw_sim(pkgs.name, "pandas") >= 0.9
).order_by("name")
```

You can now use `jw_sim` anywhere you’d use a normal string scalar expression.

---

### 7.2 Wrap a vendor aggregate (builtin agg UDF)

Suppose your warehouse has a `FAVG(x)` aggregate implementing Kahan‑compensated average:

```python
import ibis
import ibis.expr.datatypes as dt

@ibis.udf.agg.builtin
def favg(x: dt.float64) -> dt.float64:
    ...

t = con.table("measurements")
agg_expr = t.group_by(t.sensor_id).aggregate(
    mean=favg(t.temperature)
)
```

Ibis will emit `FAVG(temperature)` in the aggregated `SELECT`.

---

### 7.3 Custom feature transform on pandas backend (scalar pandas UDF)

```python
import ibis
import ibis.expr.datatypes as dt

@ibis.udf.scalar.pandas
def zscore(x: dt.float64) -> dt.float64:
    # x is a pandas.Series
    return (x - x.mean()) / x.std()

t = ibis.memtable({"x": [1.0, 2.0, 3.0, 4.0]})
expr = t.mutate(z=zscore(t.x))
```

On a pandas backend, this runs purely in Python with vectorization.

---

## 8. Prioritized view: what matters most for coverage

If you’re triaging “SQL function coverage” in a concrete system:

1. **Tier 1 – Core SELECT + aggregates + windowing**

   * Table operations, reductions, and window functions are **extensively covered and thoroughly tested** across major SQL backends. ([Ibis][1])

2. **Tier 2 – Strings, temporals, numeric + approximations**

   * Ibis’s string and temporal APIs intentionally mirror the commonly used subset of vendor functions.
   * Approximations (approx count distinct, quantiles) exist as operations, but backend support varies; the matrix tells you which. ([Ibis][12])

3. **Tier 3 – Collections, JSON, geospatial, exotic types (UUID/INET/MACADDR)**

   * These have rich APIs but more backend variance; they are strong on Postgres‑like/Trino/BigQuery/DuckDB backends and weaker on others. ([Ibis][14])

4. **Tier 4 – Vendor‑specific bells & whistles**

   * Text similarity, custom ML scoring functions, domain‑specific aggregates, specialized statistical routines: use **builtin UDFs** to surface these as Ibis expressions. ([Ibis][17])

5. **Tier 5 – DDL / procedural / non‑SELECT**

   * Not core Ibis territory; use `raw_sql` or orchestrate outside Ibis. ([Ibis][8])

---

If you tell me which backend(s) you care about (e.g., DuckDB + BigQuery + Snowflake), I can sketch a **backend‑specific function coverage checklist** (built‑ins vs builtin UDFs vs raw SQL) tuned to that stack.

[1]: https://ibis-project.org/reference/operations "operations – Ibis"
[2]: https://ibis-project.org/?utm_source=chatgpt.com "Ibis"
[3]: https://ibis-project.org/tutorials/coming-from/sql?utm_source=chatgpt.com "SQL - Ibis"
[4]: https://ibis-project.org/reference/expression-numeric "expression-numeric – Ibis"
[5]: https://ibis-project.org/reference/ "index – Ibis"
[6]: https://ibis-project.org/reference/scalar-udfs "scalar-udfs – Ibis"
[7]: https://ibis-project.org/reference/aggregate-udfs "aggregate-udfs – Ibis"
[8]: https://ibis-project.org/how-to/extending/sql?utm_source=chatgpt.com "Using SQL strings with Ibis"
[9]: https://ibis-project.org/backends/support/matrix "Operation support matrix – Ibis"
[10]: https://ibis-project.org/reference/expression-tables?utm_source=chatgpt.com "Table expressions - Ibis"
[11]: https://ibis-project.org/reference/expression-generic?utm_source=chatgpt.com "expression-generic - Ibis"
[12]: https://ibis-project.org/reference/expression-strings "expression-strings – Ibis"
[13]: https://ibis-project.org/reference/expression-temporal "expression-temporal – Ibis"
[14]: https://ibis-project.org/reference/expression-collections "expression-collections – Ibis"
[15]: https://ibis-project.org/reference/expression-json "expression-json – Ibis"
[16]: https://ibis-project.org/reference/expression-geospatial "expression-geospatial – Ibis"
[17]: https://ibis-project.org/how-to/extending/builtin "builtin – Ibis"
