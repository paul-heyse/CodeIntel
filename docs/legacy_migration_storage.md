You can treat this as two tightly scoped cleanups:

1. **Config docs & phantom legacy step configs (`config.models`)**
2. **Storage metadata schema & deprecation/compat shims (`storage.metadata_bootstrap`)** 

I’ll walk through each with concrete before/after code and the migration steps.

---

## 1. Config: kill phantom “legacy step configs” + `config.compat` reference

Right now:

* `config/config/models.py` docstring says:

  * “**Legacy Step Configs** (frozen dataclasses): These are DEPRECATED…”
  * “See `codeintel.config.compat` for converters between old and new configs.”
* There are **no legacy step config dataclasses** in this file.
* There is **no `config.compat` module** anywhere.

So this is pure doc drift and a pointer to a non‑existent module.

### 1.1. Fix the `models.py` docstring

**Before** (top of `config/config/models.py`):

```python
"""Configuration models used by the CodeIntel CLI and pipeline steps.

This module contains:
- **CLI Boundary Models** (Pydantic): `RepoConfig`, `CliPathsInput`, `ToolsConfig`,
  `CodeIntelConfig` - use these for CLI argument parsing and validation.
- **Legacy Step Configs** (frozen dataclasses): These are DEPRECATED and will be
  removed in a future version.

Migration Guide
---------------
For step configurations, prefer the new composition-based system:

Preferred:
    from codeintel.config import ConfigBuilder
    builder = ConfigBuilder.from_snapshot(repo="r", commit="c", repo_root=Path("."))
    cfg = builder.graph_metrics()
...

See `codeintel.config.builder` for the new ConfigBuilder API.
See `codeintel.config.compat` for converters between old and new configs.
"""
```

**After** (clean, no phantom legacy section):

```python
"""Configuration models used by the CodeIntel CLI and pipeline steps.

This module contains:
- **CLI Boundary Models** (Pydantic): `RepoConfig`, `CliPathsInput`, `ToolsConfig`,
  `CodeIntelConfig` – use these for CLI argument parsing and validation.

Migration Guide
---------------
For step configurations, prefer the composition-based system:

Preferred:
    from codeintel.config import ConfigBuilder
    builder = ConfigBuilder.from_snapshot(repo="r", commit="c", repo_root=Path("."))
    cfg = builder.graph_metrics()

See `codeintel.config.builder` for the ConfigBuilder API.
"""
```

Actions:

1. Delete the “Legacy Step Configs” bullet.
2. Delete the line referencing `codeintel.config.compat`.
3. Slightly reword the migration guide to be present‑tense and not imply a second, surviving step‑config API.

No code changes needed here—only text.

### 1.2. Sanity check: no actual “legacy step configs”

You already effectively confirmed this, but to be explicit:

* Grep the repo for `StepConfig` dataclasses outside `config/primitives.py`.
* If none exist, you’re done; the only StepConfig is the **canonical** one in `config.primitives`.

If you *do* find stray “old” step configs (unlikely at this point):

* Inline them into the builder or delete them, depending on usage.
* But based on the current snapshot, this doesn’t appear necessary.

### 1.3. Tests / external references

Because we only changed docstrings:

* You don’t need to update any code.
* If any tests assert on the content of `models.__doc__` (rare), update them accordingly.

---

## 2. Storage: aggressive cleanup of metadata compat / deprecation shims

Here we do **actual behavior changes**. We’re going to:

* Stop supporting upgrading *older* metadata schemas in‑place.
* Assume all deployed instances can either:

  * be fully re‑bootstrapped, or
  * run a one‑time migration script you control.

There are three small compat behaviours:

1. Extra `ALTER TABLE ... ADD COLUMN IF NOT EXISTS ...` statements to patch old metadata tables.
2. An `ALTER TABLE` that retrofits a `schema_hash` column onto `metadata.macro_registry`.
3. A `getattr(contract, "deprecated", False)` fallback even though `DatasetContract` **always** has `deprecated: bool = False` now.

We’ll remove all three.

### 2.1. Remove “upgrade in place” ALTERs for `metadata.datasets`

In `storage/metadata_bootstrap.py`, you currently have:

```python
METADATA_SCHEMA_DDL_REST: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS metadata.datasets (
        table_key        TEXT PRIMARY KEY,
        name             TEXT NOT NULL,
        is_view          BOOLEAN NOT NULL,
        jsonl_filename   TEXT,
        parquet_filename TEXT,
        family           TEXT,
        description      TEXT,
        schema_version   TEXT,
        deprecated       BOOLEAN DEFAULT FALSE
    );
    """,
    """
    ALTER TABLE metadata.datasets ADD COLUMN IF NOT EXISTS family TEXT;
    """,
    """
    ALTER TABLE metadata.datasets ADD COLUMN IF NOT EXISTS description TEXT;
    """,
    """
    ALTER TABLE metadata.datasets ADD COLUMN IF NOT EXISTS schema_version TEXT;
    """,
    """
    ALTER TABLE metadata.datasets ADD COLUMN IF NOT EXISTS deprecated BOOLEAN DEFAULT FALSE;
    """,
    """
    CREATE OR REPLACE MACRO metadata.dataset_rows(
        table_key TEXT,
        row_limit BIGINT := 100,
        ...
```

Those `ALTER TABLE` statements exist solely to **upgrade old metadata schemas** that didn’t have `family`, `description`, `schema_version`, or `deprecated`.

**After** (new canonical schema, no upgrade path):

```python
METADATA_SCHEMA_DDL_REST: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS metadata.datasets (
        table_key        TEXT PRIMARY KEY,
        name             TEXT NOT NULL,
        is_view          BOOLEAN NOT NULL,
        jsonl_filename   TEXT,
        parquet_filename TEXT,
        family           TEXT,
        description      TEXT,
        schema_version   TEXT,
        deprecated       BOOLEAN DEFAULT FALSE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS metadata.dataset_schema_registry (
        table_key TEXT PRIMARY KEY,
        schema_hash TEXT NOT NULL
    );
    """,
    """
    CREATE OR REPLACE MACRO metadata.dataset_rows(
        table_key TEXT,
        row_limit BIGINT := 100,
        row_offset BIGINT := 0
    ) AS TABLE
    ...
    """,
    # ... rest of macros exactly as they are now ...
)
```

Actions:

1. Delete the four `ALTER TABLE metadata.datasets ADD COLUMN IF NOT EXISTS ...` statements.
2. Leave the `CREATE TABLE` for `metadata.datasets` and `metadata.dataset_schema_registry` as‑is.

**Effect:**

* New deployments (fresh DB / no metadata tables) remain fine: `CREATE TABLE IF NOT EXISTS` creates tables with full schema.
* Existing deployments whose `metadata.datasets` lacks those columns **will no longer be silently upgraded** by `metadata_bootstrap`.

Because you said you control all consumers and want aggressive cleanup, you can require:

* Either drop the `metadata` schema before redeploy, or
* Run an explicit one‑time migration (see §2.4 below).

### 2.2. Remove `ALTER TABLE` compat for `metadata.macro_registry.schema_hash`

Further down, `apply_metadata_ddl` has:

```python
def apply_metadata_ddl(con: DuckDBPyConnection) -> None:
    for stmt in METADATA_SCHEMA_DDL_BASE:
        con.execute(stmt)
    for stmt in METADATA_SCHEMA_DDL_REST:
        con.execute(stmt)

    con.execute("ALTER TABLE metadata.macro_registry ADD COLUMN IF NOT EXISTS schema_hash TEXT")
```

But the `macro_registry` table **already** includes `schema_hash` in its `CREATE TABLE` definition:

```python
CREATE TABLE IF NOT EXISTS metadata.macro_registry (
    macro_name TEXT PRIMARY KEY,
    dataset_table_key TEXT,
    ddl_hash TEXT NOT NULL,
    schema_hash TEXT
);
```

The `ALTER TABLE` is purely for old DBs.

**After**:

```python
def apply_metadata_ddl(con: DuckDBPyConnection) -> None:
    for stmt in METADATA_SCHEMA_DDL_BASE:
        con.execute(stmt)
    for stmt in METADATA_SCHEMA_DDL_REST:
        con.execute(stmt)
    # No ALTER TABLE compat – assume macro_registry already has schema_hash.
```

Actions:

1. Delete the `con.execute("ALTER TABLE metadata.macro_registry ADD COLUMN IF NOT EXISTS schema_hash TEXT")` line.
2. Optionally add a short comment saying that `macro_registry` is assumed to be at the current schema version.

Again, this is only a problem if you have existing DBs where `schema_hash` is missing; see §2.4.

### 2.3. Remove `getattr(contract, "deprecated", False)` fallback

Near the bottom of `metadata_bootstrap`, you have:

```python
for name, contract in sorted(DATASET_CONTRACTS.items(), key=lambda item: item[1].table_key):
    if contract.is_view and not include_views:
        continue

    table_key = contract.table_key
    ...
    # Check for deprecated field (added in new contracts.py)
    deprecated = getattr(contract, "deprecated", False)

    _upsert_dataset_row(
        con,
        _DatasetUpsert(
            table_key=table_key,
            name=name,
            is_view=contract.is_view,
            ...
            schema_version=contract.schema_version,
            deprecated=deprecated,
        ),
    )
```

But `DatasetContract` **always** has `deprecated: bool = False` in `config/config/datasets/contracts.py`:

```python
@dataclass(frozen=True)
class DatasetContract:
    ...
    schema_version: str | None = None
    upstream_dependencies: tuple[str, ...] | None = None
    validation_profile: Literal["strict", "lenient"] = "strict"
    composition: CompositeSchema | None = None
    deprecated: bool = False
    deprecation_message: str | None = None
```

So the `getattr` defense is only for old `DatasetContract` versions that didn’t know about `deprecated`.

**After**:

```python
for name, contract in sorted(DATASET_CONTRACTS.items(), key=lambda item: item[1].table_key):
    if contract.is_view and not include_views:
        continue

    table_key = contract.table_key
    ...
    _upsert_dataset_row(
        con,
        _DatasetUpsert(
            table_key=table_key,
            name=name,
            is_view=contract.is_view,
            ...
            schema_version=contract.schema_version,
            deprecated=contract.deprecated,
        ),
    )
```

Actions:

1. Replace `deprecated = getattr(contract, "deprecated", False)` with `deprecated = contract.deprecated`.
2. Remove the comment “added in new contracts.py”; it’s no longer “new,” it’s canonical.

This makes the linkage between `DatasetContract` and the metadata table **explicit and strict**.

### 2.4. One‑time DB migration strategy (aggressive, but safe)

Because we’re *removing* backward compatibility code, we must ensure old schemas don’t linger.

You have two main options:

#### Option A – Drop and re‑bootstrap metadata

If your DuckDB database is fully owned by this service and metadata is purely internal:

1. Before deploying the code change:

   * Back up any data you care about (if needed).

2. Drop the metadata schema (or the entire DB, if it’s only used for this project):

   ```sql
   DROP SCHEMA IF EXISTS metadata CASCADE;
   ```

3. Redeploy new code and let `metadata_bootstrap.initialize_metadata(...)` recreate everything.

This is the simplest and most robust if you don’t need the old metadata.

#### Option B – In‑place migration script (if you want to keep data)

If you want to keep existing contents of `metadata.datasets` / `macro_registry`, you can run a one‑time migration script before (or alongside) the code update:

In Python (or via DuckDB CLI):

```python
import duckdb

con = duckdb.connect("your.db")

# 1. Ensure datasets table has all columns
con.execute("""
    ALTER TABLE metadata.datasets
        ADD COLUMN IF NOT EXISTS family TEXT;
""")
con.execute("""
    ALTER TABLE metadata.datasets
        ADD COLUMN IF NOT EXISTS description TEXT;
""")
con.execute("""
    ALTER TABLE metadata.datasets
        ADD COLUMN IF NOT EXISTS schema_version TEXT;
""")
con.execute("""
    ALTER TABLE metadata.datasets
        ADD COLUMN IF NOT EXISTS deprecated BOOLEAN DEFAULT FALSE;
""")

# 2. Ensure macro_registry has schema_hash
con.execute("""
    ALTER TABLE metadata.macro_registry
        ADD COLUMN IF NOT EXISTS schema_hash TEXT;
""")

con.close()
```

Then:

* Deploy the new code (with the `ALTER TABLE` calls removed).
* All DB instances are now guaranteed to be on the new schema.

After this one‑time script, you can **delete** the migration code itself; future DBs will always be created with the full schema by `CREATE TABLE IF NOT EXISTS ...`.

### 2.5. Tests and dataset contracts

After these changes:

* Tests that assert on generated SQL DDL might need to be updated:

  * They should no longer expect the `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` statements.
* If any tests explicitly exercise the `getattr(contract, "deprecated", False)` behaviour, update them to:

  * Expect that `deprecated` is a required field on `DatasetContract`, and
  * Verify that metadata rows reflect `contract.deprecated` correctly.

Because `DatasetContract` already has `deprecated: bool = False`, the only semantic change is:

* If someone constructed a non‑standard “contract” object missing that field, it will now raise an attribute error instead of silently behaving as `False`. For an aggressively internal codebase, that’s the right behaviour.

---

## 3. Final checklist for this cleanup

You’re done with this cluster when:

* [ ] `config/config/models.py` no longer mentions:

  * “Legacy Step Configs,” and
  * `codeintel.config.compat`.
* [ ] There is **no** `config/compat.py` or `codeintel.config.compat` import anywhere.
* [ ] `storage/metadata_bootstrap.py`:

  * Has `METADATA_SCHEMA_DDL_REST` without any `ALTER TABLE metadata.datasets ADD COLUMN IF NOT EXISTS ...` statements.
  * Does **not** execute `ALTER TABLE metadata.macro_registry ADD COLUMN IF NOT EXISTS schema_hash TEXT`.
  * Uses `contract.deprecated` directly, not `getattr`.
* [ ] All `DatasetContract` instances include the `deprecated: bool` field (already true in `config/config/datasets/contracts.py`).
* [ ] Any tests that relied on the old DDL strings or `getattr` fallback are updated.
* [ ] Any live DuckDB instances have been:

  * Either dropped and re‑bootstrapped, or
  * Migrated once so their schemas match the new expectations.

After that, there’s no lingering storage “compat” or phantom legacy config story: the codebase assumes **one** configuration model and **one** metadata schema, with no upgrade shims or references to missing legacy modules.
