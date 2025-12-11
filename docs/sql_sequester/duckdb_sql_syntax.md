Here’s a “what you actually need to know” tour of DuckDB’s DDL and SQL syntax, oriented toward someone who already speaks Postgres/SQL and wants the DuckDB‑specific details and gotchas.

I’m loosely mirroring the prioritized, inventory‑style structure of your FAISS wheel overview.

---

## 1. Mental model: DuckDB’s SQL dialect in one page

* **Postgres‑ish analytic SQL dialect**: syntax and semantics are close to PostgreSQL, with some SQLite‑like touches and a lot of analytic sugar (“friendly SQL”).([DuckDB][1])
* **Columnar, OLAP‑oriented**: DDL is “normal SQL”, but performance assumptions are columnar, vectorized and batch‑oriented; no row‑store indexes, no manually managed clustering.
* **Objects you can create/drop** via DDL:

  * `SCHEMA`, `TABLE`, `VIEW`, `INDEX`, `SEQUENCE`, `TYPE`, `MACRO` / `FUNCTION`, `SECRET`, plus extensions, etc.([DuckDB][2])
* **Namespaces**:

  * Databases (catalogs) → Schemas → Objects.
  * Default database is often `memory` or a file; default schema is `main`.([DuckDB][3])
* **Identifiers**: case‑insensitive everywhere (even when quoted), but original case is preserved for display; literals and operators behave as in Postgres.([DuckDB][4])

---

## 2. Catalog & namespaces

### 2.1 Databases / ATTACH / USE

DuckDB can work with multiple database files in the same process:

```sql
ATTACH 'file1.duckdb' AS db1;
ATTACH 'file2.duckdb' AS db2 (READ_ONLY);
USE db1;           -- default db = db1, schema = db1.main
USE main;          -- switch schema only within current db
```

Key points:([DuckDB][5])

* `ATTACH 'file.db' [AS alias] [(options...)]` attaches another database file.
* Options include `READ_ONLY`, `BLOCK_SIZE`, `ROW_GROUP_SIZE`, `STORAGE_VERSION`, etc.
* `USE db.schema` or `USE schema` sets defaults for unqualified references and DDL.
* Attachments are **not persisted**; you must re‑`ATTACH` each session.

### 2.2 Schemas

```sql
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE OR REPLACE SCHEMA staging;
CREATE TABLE analytics.events (...);
```

* Default schema is `main`.
* Schemas are just name containers; no separate ownership model.([DuckDB][6])

`DROP SCHEMA` participates in dependency tracking:

```sql
DROP SCHEMA analytics RESTRICT;  -- error if anything under it
DROP SCHEMA analytics CASCADE;   -- also drops tables, views, etc.
```

([DuckDB][7])

---

## 3. Data types (DDL‑relevant subset)

You’ll mostly see these categories:([DuckDB][8])

* **Numeric**: `TINYINT`, `SMALLINT`, `INTEGER`, `BIGINT`, `HUGEINT`, `REAL`, `DOUBLE`, `DECIMAL(p,s)`.
* **Boolean**: `BOOLEAN`.
* **Text / binary**: `VARCHAR` (alias `TEXT`), `BLOB`, `BIT`, `BITSTRING`.
* **Temporal**: `DATE`, `TIME`, `TIMESTAMP`, `TIMESTAMP WITH TIME ZONE`, `INTERVAL`.
* **Nested & semi‑structured**:

  * `LIST`, `ARRAY`, `MAP`, `STRUCT`, `UNION`, `JSON` (via functions; data type is usually `VARCHAR` or `JSON` depending on extension).
  * `ARRAY` = **fixed‑length** array (`INTEGER[3]`), good for embeddings.([DuckDB][9])
  * `LIST` = variable‑length list (`INTEGER[]`).([DuckDB][10])
  * `STRUCT` = typed record/row; keys are case‑insensitive.([DuckDB][11])
* **ENUM**: dictionary encoded, great for low‑cardinality strings.([DuckDB][12])

For custom schemas, you can define **user types**:

```sql
CREATE TYPE mood AS ENUM ('sad','ok','happy');
CREATE TYPE embedding AS FLOAT[384];
CREATE TYPE customer_id AS INTEGER;
```

([DuckDB][13])

---

## 4. Tables: CREATE / ALTER / DROP in depth

### 4.1 CREATE TABLE – canonical form

Canonical, fully‑spelled form (ignoring all production grammar):

```sql
CREATE [OR REPLACE] [TEMPORARY | TEMP] TABLE [schema.]name (
    column_name data_type
        [DEFAULT default_expr]
        [NOT NULL]
        [UNIQUE]
        [PRIMARY KEY]
        [CHECK (check_expression)]
        [REFERENCES ref_table [(ref_col_list)]],
    [table_constraint [, ...]]
)
```

Table constraints:([DuckDB][14])

```sql
[CONSTRAINT constraint_name]
    { PRIMARY KEY (col_list)
    | UNIQUE (col_list)
    | CHECK (expression)
    | FOREIGN KEY (col_list) REFERENCES other_table [(col_list)]
    }
```

Examples:

```sql
CREATE TABLE orders (
    order_id     BIGINT       PRIMARY KEY,
    customer_id  BIGINT       NOT NULL,
    status       mood         NOT NULL DEFAULT 'pending',
    created_at   TIMESTAMP    NOT NULL DEFAULT now(),
    amount       DECIMAL(18,2) CHECK (amount >= 0),
    CONSTRAINT fk_orders_customer
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
```

**Temporary tables**:

```sql
CREATE TEMP TABLE tmp_orders AS
    FROM read_csv('orders.csv');
```

* Live in `temp.main`.
* Session‑scoped; priority in name resolution over persistent tables.([DuckDB][14])

### 4.2 CTAS (CREATE TABLE AS SELECT) & FROM‑first

DuckDB strongly encourages CTAS for ingestion:([DuckDB][14])

```sql
CREATE TABLE orders AS
    FROM read_parquet('s3://bucket/orders/*.parquet');

CREATE OR REPLACE TABLE orders_2024 AS
    SELECT * FROM orders WHERE order_date >= DATE '2024-01-01';
```

Notes:

* CTAS copies **column names and types**, but **not** indexes, constraints, or defaults.([DuckDB][14])
* FROM‑first syntax — `CREATE TABLE t AS FROM expr` — is equivalent to `SELECT *` from the source.

### 4.3 CREATE OR REPLACE / IF NOT EXISTS

DuckDB has “friendly” DDL modifiers:([DuckDB][15])

```sql
CREATE OR REPLACE TABLE t (...);   -- overwrite schema & data
CREATE TABLE IF NOT EXISTS t (...); -- no-op if t exists
```

These also exist for `SCHEMA`, `SEQUENCE`, `MACRO`, some others.

### 4.4 ALTER TABLE – schema evolution

Core operations: add/drop columns, change types, defaults, nullability, rename, add primary key.([DuckDB][2])

```sql
ALTER TABLE t RENAME TO t_old;

ALTER TABLE t
    ADD COLUMN extra JSON,
    ADD COLUMN created_at TIMESTAMP DEFAULT now();

ALTER TABLE t
    DROP COLUMN extra;

ALTER TABLE t
    ALTER COLUMN amount TYPE DOUBLE;

ALTER TABLE t
    ALTER COLUMN customer_id SET NOT NULL;

ALTER TABLE t
    ALTER COLUMN customer_id DROP NOT NULL;

ALTER TABLE t
    ALTER COLUMN amount SET DEFAULT 0,
    ALTER COLUMN amount DROP DEFAULT;

ALTER TABLE t
    RENAME COLUMN old_name TO new_name;

ALTER TABLE t
    ADD PRIMARY KEY (id);
```

Important quirks:([DuckDB][2])

* `ADD CONSTRAINT` / `DROP CONSTRAINT` clauses **are not supported** yet. Once constraints are there, you can’t generically rename/drop them via DDL; often you CTAS into a fresh table.
* Many `ALTER` operations are **blocked if there are dependent indexes** (including implicit PK/UNIQUE/foreign‑key indexes). You must drop those indexes (or the dependent constraint/table) first.
* Type changes require all historic values in the column to be castable; even deleted rows can block an `ALTER TYPE`. The suggested workaround is `CREATE OR REPLACE TABLE new AS FROM old;`.([DuckDB][2])

### 4.5 DROP TABLE

```sql
DROP TABLE [IF EXISTS] t;
```

* Frees memory immediately; disk blocks are marked free but file size may not shrink until you `VACUUM` or otherwise reclaim space.([DuckDB][7])

---

## 5. Constraints & how they’re implemented

DuckDB supports: `NOT NULL`, `PRIMARY KEY`, `UNIQUE`, `CHECK`, and `FOREIGN KEY`.([DuckDB][16])

### 5.1 Column vs table constraints

You can put constraints inline or at table level:

```sql
CREATE TABLE students (
    id    INTEGER PRIMARY KEY,
    name  VARCHAR NOT NULL,
    email VARCHAR UNIQUE,
    CONSTRAINT chk_name CHECK (length(name) > 0)
);
```

Composite constraints must be table‑level:

```sql
CREATE TABLE exams (
    student_id  INTEGER,
    subject_id  INTEGER,
    grade       INTEGER,
    PRIMARY KEY (student_id, subject_id),
    FOREIGN KEY (student_id) REFERENCES students(id)
);
```

### 5.2 Semantics & implementation

* `NOT NULL` – standard; enforced on inserts/updates.([DuckDB][16])
* `PRIMARY KEY` / `UNIQUE`:

  * Enforced via **Adaptive Radix Tree (ART)** secondary index.
  * ART is also used for user‑defined `CREATE INDEX` with optional `UNIQUE`.([DuckDB][17])
* `FOREIGN KEY`:

  * Requires referenced column(s) to be `PRIMARY KEY` or `UNIQUE`.
  * Foreign keys automatically get an ART index as well.([DuckDB][16])
* `CHECK` – arbitrary expression; evaluated row‑by‑row on write.([DuckDB][16])

Index‑backed constraints currently have some MVCC limitations:

* Index structures don’t fully track transaction timestamps yet, so some concurrent write patterns cause “violates primary key / unique constraint” even when logically OK.([DuckDB][18])
* Updates to indexed columns are implemented as delete+insert; combined with foreign keys, that can produce constraint errors earlier than you’d expect.([DuckDB][17])

Practical pattern: **define constraints at `CREATE TABLE` time**; don’t rely on ALTER‑time constraint manipulation.

---

## 6. Index DDL

### 6.1 Automatic zonemaps

* Every persistent table gets **min‑max (zonemap) indexes** per column automatically; you don’t manage these in DDL.
* Benefit scales with ordering: pre‑sort data on typical filter columns before writing for best pruning.([DuckDB][19])

### 6.2 ART indexes via CREATE INDEX

```sql
CREATE [UNIQUE] INDEX idx_name
ON table_name (column_list_or_expression)
[USING index_type] [WITH (option = value, ...)];
```

Examples:([DuckDB][20])

```sql
CREATE UNIQUE INDEX films_id_idx ON films (id);
CREATE INDEX s_idx ON films (revenue);
CREATE INDEX gy_idx ON films (genre, year);
CREATE INDEX expr_idx ON t ((lower(col)));
DROP INDEX IF EXISTS s_idx;
```

Notes:

* Index type is optional; built‑in choice is ART; extra index types (e.g. R‑tree) come from extensions.([DuckDB][17])
* `CREATE INDEX IF NOT EXISTS` is not fast‑exit: it still builds, then checks before commit.([DuckDB][20])
* Indexes are fully persisted and used for constraints as well as performance.

---

## 7. Views & schemas

### 7.1 CREATE VIEW / ALTER VIEW / DROP VIEW

```sql
CREATE [OR REPLACE] [TEMP] VIEW name [(col1, col2, ...)] AS
    select_query;

ALTER VIEW view1 RENAME TO view2;
DROP VIEW IF EXISTS view1;
```

Key semantics:([DuckDB][21])

* Views are **not materialized**; the stored SQL is re‑planned each time.
* `CREATE OR REPLACE VIEW` updates the definition atomically.
* Temporary views cannot specify a schema name; they live in a special temp schema.
* Dependency tracking for views is limited:

  * Dropping a base table does **not** automatically drop dependent views; they just become invalid and will error at runtime.
* `duckdb_views()` holds view definitions (`sql` column).

### 7.2 SCHEMA + DROP dependency rules

The `DROP` statement is one entry point for dependency semantics:

```sql
DROP TABLE t;
DROP SCHEMA s CASCADE;
DROP TYPE mood;
DROP SEQUENCE serial;
DROP INDEX idx;
DROP MACRO m;
```

* With `RESTRICT` (default), DuckDB blocks drops when something depends on that object (e.g. table depends on schema; index depends on table).([DuckDB][7])
* With `CASCADE`, dependents are dropped recursively.

---

## 8. Types, sequences, macros: “advanced DDL”

### 8.1 CREATE TYPE: enums, struct/union and aliases

```sql
CREATE TYPE mood AS ENUM ('happy', 'sad','curious');
CREATE TYPE point2d AS STRUCT(x DOUBLE, y DOUBLE);
CREATE TYPE value_or_error AS UNION(val DOUBLE, err VARCHAR);
CREATE TYPE object_id AS BIGINT;
```

These types can be used everywhere a built‑in type would be.([DuckDB][13])

Limitations:

* `CREATE TYPE` does **not** support `OR REPLACE`; you must `DROP TYPE` then re‑create.([DuckDB][13])

ENUM specifics:

* Values must be non‑`NULL` and unique.
* Can be defined per schema or anonymously via casts (`'clubs'::ENUM ('spades',...)`).([DuckDB][12])

### 8.2 CREATE SEQUENCE

```sql
CREATE SEQUENCE id_seq START WITH 1 INCREMENT BY 1;
CREATE OR REPLACE SEQUENCE id_seq;
CREATE SEQUENCE IF NOT EXISTS id_seq;
DROP SEQUENCE IF EXISTS id_seq CASCADE;
```

Use in DDL:

```sql
CREATE SEQUENCE id_seq START 1;

CREATE TABLE tbl (
    id BIGINT PRIMARY KEY DEFAULT nextval('id_seq'),
    s  VARCHAR
);
```

* Sequences are `BIGINT`‑based (8‑byte range).
* Support `START`, `INCREMENT BY`, `MINVALUE`, `MAXVALUE`, `[NO] CYCLE`, `TEMP` etc.([DuckDB][22])
* `nextval('seq')`, `currval('seq')` are the primary APIs.
* Some dependency edge cases require `DROP SEQUENCE ... CASCADE` (e.g. if a column formerly used the sequence).([DuckDB][22])

### 8.3 CREATE MACRO / FUNCTION

DuckDB’s macros are SQL‑level “functions” that expand to expressions or subqueries.([DuckDB][23])

Scalar macro:

```sql
CREATE MACRO add(a, b) AS a + b;
CREATE OR REPLACE MACRO ifelse(a, b, c) AS CASE WHEN a THEN b ELSE c END;
```

Table macro:

```sql
CREATE MACRO top_n(t, n) AS TABLE
    (SELECT * FROM t ORDER BY metric DESC LIMIT n);

SELECT * FROM top_n(my_table, 10);
```

Notes:

* Macros live in schemas and can also be declared with the alias keyword `FUNCTION`.
* Support `OR REPLACE`, `IF NOT EXISTS`, default parameters, typed parameters.([DuckDB][23])
* `DROP MACRO` / `DROP FUNCTION` follow normal `DROP` semantics.([DuckDB][7])

---

## 9. Overall SELECT / query syntax (DuckDB‑specific highlights)

This is the core *query* shape, mostly standard SQL with extensions:([DuckDB][24])

```sql
[WITH cte_name [(col1, ...)] AS (subquery) , ...]
SELECT [DISTINCT | DISTINCT ON (...)] select_list
FROM   from_source
       [JOIN ... ON ...] ...
WHERE  predicate
GROUP BY group_exprs
HAVING group_predicate
WINDOW window_name AS (window_spec), ...
QUALIFY qualify_predicate
ORDER BY order_exprs
LIMIT {n | n%} [OFFSET m]
```

### 9.1 Friendly SELECT features

From the `SELECT` and `Friendly SQL` docs:([DuckDB][25])

* **Star expression extensions**:

  * `SELECT * EXCLUDE (col1, col2) FROM t;`
  * `SELECT * REPLACE (lower(city) AS city) FROM addresses;`
  * `SELECT COLUMNS('^metric_\\d+$') FROM t;`
* **Lateral / reusable column aliases**:

  ```sql
  SELECT i + 1 AS j,
         j + 2 AS k
  FROM range(0,3) t(i);
  ```
* **Column aliases usable in `WHERE`, `GROUP BY`, `HAVING`** (not in `JOIN ON`).
* `GROUP BY ALL` and `ORDER BY ALL` to infer grouping/sorting from `SELECT` list.
* `UNION BY NAME` to align columns by name instead of position.
* `LIMIT 10%` for percentage limits.

### 9.2 FROM / JOIN / WITH

Standard join syntax with some niceties:([DuckDB][2])

* `FROM` sources can be:

  * Base tables or views.
  * Subqueries: `(SELECT ...) AS alias`.
  * Table functions (e.g. `read_csv`, `range`, `json_scan`, `parquet_scan`).
  * `VALUES` clauses.
* `FROM`‑first shorthand:

  ```sql
  FROM my_table WHERE ...  -- equivalent to SELECT * FROM ...
  ```
* `WITH` supports named CTEs with optional column lists.
* `JOIN` features:

  * All common join types (`INNER`, `LEFT`, `RIGHT`, `FULL`, `SEMI`, `ANTI`, `CROSS`).
  * `USING (col)` syntax, `NATURAL JOIN`.
  * `VALUES` in `JOIN` and in the `WITH` anchor part.

### 9.3 Aggregation, windowing, QUALIFY

* All usual aggregates; plus advanced `GROUPING SETS`, `ROLLUP`, `CUBE`.([DuckDB][15])
* Window functions via `OVER (...)`; separate `WINDOW` clause to name specs.
* `FILTER` on aggregates: `sum(x) FILTER (WHERE x > 0)`.
* `QUALIFY` filters **after** window functions (like `HAVING` but for windows).([DuckDB][26])

---

## 10. Identifiers, literals, and case rules

DuckDB’s identifier rules differ slightly from Postgres:([DuckDB][4])

* **Keywords & functions** are case‑insensitive (`SELECT`, `Select`, `select` all same).
* **Identifiers** (table/column names) are case‑insensitive even when quoted:

  * `SELECT col_A FROM (SELECT 'x' AS col_a);` works.
  * `CREATE TABLE "MyTaBLe"(x INT); SELECT * FROM mytable;` also works.
* Original case is **preserved** in the catalog; you can see it via meta tables like `duckdb_tables()`.
* `preserve_identifier_case = false` makes DuckDB **lowercase** identifiers like Postgres, while still treating lookups case‑insensitively.

Quoting rules:([DuckDB][4])

* Unquoted identifiers: must not be reserved keywords, can’t start with digits, no spaces.
* Quoted identifiers: double‑quotes; can contain keywords/spaces/emoji.
* Case insensitivity uses ASCII rules: `col_A` == `col_a`, but `col_á` is distinct.

---

## 11. Practical DDL patterns & “recipes”

### 11.1 Ingest → normalized table

```sql
-- Raw landing from Parquet
CREATE OR REPLACE TABLE raw_events AS
    FROM 's3://data/events/*.parquet';

-- Derived, constrained analytic table
CREATE TABLE events (
    event_id      BIGINT PRIMARY KEY,
    user_id       BIGINT NOT NULL,
    ts            TIMESTAMP NOT NULL,
    event_type    VARCHAR NOT NULL,
    payload       JSON,
    created_at    TIMESTAMP NOT NULL DEFAULT now(),
    CHECK (ts >= TIMESTAMP '2000-01-01')
);

INSERT INTO events
SELECT
    event_id,
    user_id,
    ts,
    event_type,
    payload,
    now()
FROM raw_events;
```

Use `CREATE OR REPLACE TABLE ... AS` on the *derived* table when you want a full rebuild.

### 11.2 Schema evolution without painful ALTER

Given `old_table`, instead of many `ALTER TABLE` steps:

```sql
CREATE OR REPLACE TABLE new_table AS
SELECT
    id,
    customer_id,
    amount::DOUBLE AS amount,
    added_col,
    ...
FROM old_table;
```

Then point consumers at `new_table` (or keep the name by `CREATE OR REPLACE TABLE old_table AS FROM new_table;`).

### 11.3 Sequences for surrogate keys

```sql
CREATE SEQUENCE id_seq START 1;
CREATE TABLE t (
    id BIGINT PRIMARY KEY DEFAULT nextval('id_seq'),
    ...
);
```

If you need to retrofit an ID onto an existing table, DuckDB’s own example is:([DuckDB][22])

```sql
CREATE SEQUENCE id_sequence START 1;
ALTER TABLE t ADD COLUMN id BIGINT DEFAULT nextval('id_sequence');
```

### 11.4 Using macros for “schema‑local UDFs”

```sql
CREATE MACRO standardized_amount(currency, amount) AS
    CASE currency
        WHEN 'USD' THEN amount
        WHEN 'EUR' THEN amount * 1.1
        ELSE amount
    END;

SELECT standardized_amount(currency, amount) AS amt_usd
FROM payments;
```

Keeps logic in the catalog, versionable with DDL.

---

If you’d like, next step I can:

* Turn this into a compact **“DDL cheat sheet”** (one page of syntax only), or
* Walk through **schema design patterns** in DuckDB (nested types, arrays for embeddings, etc.), or
* Compare **DuckDB vs Postgres DDL** explicitly line‑by‑line.

[1]: https://duckdb.org/docs/stable/sql/dialect/postgresql_compatibility.html?utm_source=chatgpt.com "PostgreSQL Compatibility – DuckDB"
[2]: https://duckdb.org/docs/stable/sql/statements/alter_table.html "ALTER TABLE Statement – DuckDB"
[3]: https://duckdb.org/docs/stable/sql/statements/use.html "USE Statement – DuckDB"
[4]: https://duckdb.org/docs/stable/sql/dialect/keywords_and_identifiers?utm_source=chatgpt.com "Keywords and Identifiers – DuckDB"
[5]: https://duckdb.org/docs/stable/sql/statements/attach.html "ATTACH and DETACH Statements – DuckDB"
[6]: https://duckdb.org/docs/stable/sql/statements/create_schema.html "CREATE SCHEMA Statement – DuckDB"
[7]: https://duckdb.org/docs/stable/sql/statements/drop.html "DROP Statement – DuckDB"
[8]: https://duckdb.org/docs/stable/sql/data_types/overview.html "Data Types – DuckDB"
[9]: https://duckdb.org/docs/stable/sql/data_types/array.html "Array Type – DuckDB"
[10]: https://duckdb.org/docs/stable/sql/data_types/list.html "List Type – DuckDB"
[11]: https://duckdb.org/docs/stable/sql/data_types/struct.html "Struct Data Type – DuckDB"
[12]: https://duckdb.org/docs/stable/sql/data_types/enum.html "Enum Data Type – DuckDB"
[13]: https://duckdb.org/docs/stable/sql/statements/create_type.html "CREATE TYPE Statement – DuckDB"
[14]: https://duckdb.org/docs/stable/sql/statements/create_table.html "CREATE TABLE Statement – DuckDB"
[15]: https://duckdb.org/docs/stable/sql/dialect/friendly_sql.html "Friendly SQL – DuckDB"
[16]: https://duckdb.org/docs/stable/sql/constraints.html?utm_source=chatgpt.com "Constraints – DuckDB"
[17]: https://duckdb.org/docs/1.3/sql/indexes.html?utm_source=chatgpt.com "Indexes – DuckDB"
[18]: https://duckdb.org/docs/stable/sql/indexes.html?utm_source=chatgpt.com "Indexes – DuckDB"
[19]: https://duckdb.org/docs/stable/guides/performance/indexing.html?utm_source=chatgpt.com "Indexing – DuckDB"
[20]: https://duckdb.org/docs/stable/sql/statements/create_index.html "CREATE INDEX Statement – DuckDB"
[21]: https://duckdb.org/docs/stable/sql/statements/create_view.html "CREATE VIEW Statement – DuckDB"
[22]: https://duckdb.org/docs/stable/sql/statements/create_sequence.html "CREATE SEQUENCE Statement – DuckDB"
[23]: https://duckdb.org/docs/stable/sql/statements/create_macro.html "CREATE MACRO Statement – DuckDB"
[24]: https://duckdb.org/docs/stable/sql/query_syntax/select?utm_source=chatgpt.com "SELECT Clause – DuckDB"
[25]: https://duckdb.org/docs/stable/sql/query_syntax/select.html "SELECT Clause – DuckDB"
[26]: https://duckdb.net.cn/docs/stable/sql/statements/select.html?utm_source=chatgpt.com "SELECT 语句 – DuckDB 数据库"
