# Implementation plan for docs views as first-class derived datasets #

You absolutely still want the `docs.*` layer, but its job changes a bit now that the dataset system is in place:

* **Base reality** lives in `core.*`, `graph.*`, `analytics.*` datasets.
* **Docs views** become **derived, doc-ready datasets** (complex joins, human/LLM-oriented schemas).
* We should **remove trivial alias views**, integrate the remaining `docs.*` views into the dataset infra, and add strong typing only at the serving edge where it matters.

Below is a **full implementation plan** across storage, dataset infra, repositories, serving, and tests. Think of it as “Epic 2: docs views as first-class derived datasets”.

---

## 0. Objectives

1. **Reduce duplication**: kill `docs.*` views that are just `SELECT * FROM analytics.*`.
2. **Keep + strengthen non-trivial docs views**:

   * `docs.v_function_summary`, `docs.v_call_graph_enriched`, `docs.v_file_summary`,
   * `docs.v_module_architecture`, `docs.v_subsystem_profile`, etc.
3. **Treat docs views as datasets**:

   * They appear in `metadata.datasets`, support `metadata.dataset_rows`, export, and validation like any other dataset.
4. **Add strong typing only at serving edge**:

   * Pydantic models per key doc surface (function summary/profile, file summary, subsystem views…),
   * generic dataset browsing still uses the dataset infra.

---

## 1. Rationalize the `docs.*` surface (alias vs derived)

### 1.1. Inventory & classify docs views

**Files:**

* `storage/storage/views/function_views.py`
* `storage/storage/views/module_views.py`
* `storage/storage/views/graph_views.py`
* `storage/storage/views/test_views.py`
* `storage/storage/views/subsystem_views.py`
* `storage/storage/views/ide_views.py`
* `storage/storage/views/data_model_views.py`
* `storage/storage/views/__init__.py`

**Action:**

Create a small local list (or a test) that classifies `DOCS_VIEWS` into:

* **Alias views** (pure passthrough):

  * `docs.v_function_profile` → `analytics.function_profile`
  * `docs.v_file_profile` → `analytics.file_profile`
  * `docs.v_module_profile` → `analytics.module_profile`
  * `docs.v_config_graph_metrics_keys` → `analytics.config_graph_metrics_keys`
  * `docs.v_config_graph_metrics_modules` → `analytics.config_graph_metrics_modules`
  * `docs.v_config_projection_key_edges` → `analytics.config_projection_key_edges`
  * `docs.v_config_projection_module_edges` → `analytics.config_projection_module_edges`

* **Derived views** (non-trivial SQL: joins, aggregations, JSON packing):

  * e.g. `docs.v_function_summary`, `docs.v_call_graph_enriched`,
    `docs.v_file_summary`, `docs.v_module_architecture`,
    `docs.v_subsystem_profile`, `docs.v_subsystem_coverage`, etc.

You can codify this list in `views/__init__.py`:

```python
ALIAS_DOCS_VIEWS: dict[str, str] = {
    "docs.v_function_profile": "analytics.function_profile",
    "docs.v_file_profile": "analytics.file_profile",
    "docs.v_module_profile": "analytics.module_profile",
    "docs.v_config_graph_metrics_keys": "analytics.config_graph_metrics_keys",
    "docs.v_config_graph_metrics_modules": "analytics.config_graph_metrics_modules",
    "docs.v_config_projection_key_edges": "analytics.config_projection_key_edges",
    "docs.v_config_projection_module_edges": "analytics.config_projection_module_edges",
}

DERIVED_DOCS_VIEWS: tuple[str, ...] = tuple(
    v for v in DOCS_VIEWS if v not in ALIAS_DOCS_VIEWS
)
```

> This gives you a clear “alias vs derived” split for later steps.

---

### 1.2. Stop *depending* on alias docs views in Python

**Files:**

* `storage/storage/gateway.py` (`DocsViews` methods)
* `storage/storage/repositories/functions.py`
* `storage/storage/repositories/modules.py`
* `storage/storage/repositories/graphs.py` (for config graph projection aliases)
* `serving/serving/mcp/query_service.py` (indirectly via repos/gateway)
* Any other place you see `docs.v_function_profile`, `docs.v_file_profile`, etc.

**Pattern:**

For each alias, change queries from `docs.v_*` to the underlying analytics table:

**Before** (gateway):

```python
class DocsViews:
    ...

    def function_profile(self) -> DuckDBRelation:
        """
        Return relation for docs.v_function_profile.
        """
        return self.con.table("docs.v_function_profile")
```

**After**:

```python
    def function_profile(self) -> DuckDBRelation:
        """
        Return relation for analytics.function_profile.

        The docs.v_function_profile alias is deprecated; use the base analytics
        dataset directly.
        """
        return self.con.table("analytics.function_profile")
```

**Before** (repository function):

```python
sql = """
    SELECT *
    FROM docs.v_function_profile
    WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
    LIMIT 1
"""
```

**After:**

```python
sql = """
    SELECT *
    FROM analytics.function_profile
    WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
    LIMIT 1
"""
```

Do the same for:

* `docs.v_file_profile` → `analytics.file_profile`
* `docs.v_module_profile` → `analytics.module_profile`
* config metrics/projection aliases in `graph_views.py` / `repositories/graphs.py`.

At this point, **no Python code relies on the alias views**; they’re just SQL sugar.

---

### 1.3. Remove alias view definitions and shrink `DOCS_VIEWS`

Once you’ve migrated call sites, you can safely remove the alias view definitions:

**Example (in `function_views.py`):**

```sql
CREATE OR REPLACE VIEW docs.v_function_profile AS
    SELECT *
    FROM analytics.function_profile;
```

→ delete entirely.

In `module_views.py` / `graph_views.py`, remove the equivalent alias `CREATE VIEW` blocks.

Then, **update `FUNCTION_VIEW_NAMES`, `MODULE_VIEW_NAMES`, `GRAPH_VIEW_NAMES`** so they no longer include the alias names.

```python
FUNCTION_VIEW_NAMES: tuple[str, ...] = (
    "docs.v_function_summary",
    "docs.v_function_architecture",
    "docs.v_function_history",
    "docs.v_function_history_timeseries",
    # "docs.v_function_profile",  # removed: now use analytics.function_profile
    "docs.v_cfg_block_architecture",
    "docs.v_dfg_block_architecture",
)
```

`DOCS_VIEWS` in `views/__init__.py` will then automatically only contain **derived** views.

---

## 2. Treat remaining docs views as “derived datasets”

Now that only non-trivial docs views remain, we want to make them first-class citizens in the dataset infra without inventing a parallel system.

### 2.1. Ensure every docs view is present in `metadata.datasets`

This is already true via `bootstrap_metadata_datasets`:

**File:** `storage/storage/metadata_bootstrap.py`

```python
if include_views:
    for view_key in DOCS_VIEWS:
        _, name = view_key.split(".", maxsplit=1)
        _upsert_dataset_row(
            con,
            table_key=view_key,
            name=name,
            is_view=True,
            filenames=(jsonl_mapping.get(view_key), parquet_mapping.get(view_key)),
        )
```

Once alias views are removed from `DOCS_VIEWS`, this loop will only register **derived** docs views.

No code changes needed here beyond the earlier DOCS_VIEWS change.

---

### 2.2. (Optional) Record simple metadata about docs datasets

If you want to make docs datasets more discoverable, you can extend `Dataset` and/or metadata to carry a “kind” and description.

**File:** `storage/storage/datasets.py`

Add optional fields to `Dataset`:

```python
@dataclass(frozen=True)
class Dataset:
    """Metadata describing a logical dataset backed by a DuckDB table or view."""

    table_key: str
    name: str
    schema: TableSchema | None
    row_binding: RowBinding | None = None
    jsonl_filename: str | None = None
    parquet_filename: str | None = None
    is_view: bool = False
    # new, optional:
    family: str | None = None  # e.g. "core", "analytics", "docs"
    description: str | None = None
```

In `load_dataset_registry`, you can initially keep these fields as `None`, and later populate them if you decide to extend `metadata.datasets` or add a companion metadata table.

**Optional DB extension:**

* Add columns `family TEXT` and `description TEXT` to `metadata.datasets`.
* Populate them in `_upsert_dataset_row`:

  * `family = 'docs'` for `docs.*`,
  * `family = 'analytics'` for analytics tables,
  * etc.

This is strictly optional; you can also store descriptive metadata elsewhere.

---

### 2.3. Use docs views through the dataset reader when convenient

Because they’re in `metadata.datasets`, you can access docs views generically like any dataset:

**File:** `storage/storage/repositories/datasets.py`

```python
@dataclass(frozen=True)
class DatasetReadRepository(BaseRepository):
    ...

    def read_dataset_rows(
        self,
        dataset_name: str,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[RowDict]:
        """
        Read rows from a dataset using metadata.dataset_rows.

        Works with both tables and views, including docs.*.
        """
        sql = "SELECT * FROM metadata.dataset_rows(?, ?, ?)"
        rows = self.con.execute(sql, [dataset_name, limit, offset]).fetchall()
        return [dict(row) for row in rows]
```

This means **agents** can explore `docs.v_function_summary` the same way they explore `analytics.function_profile`, without adding new special cases.

---

## 3. Typed contracts at the serving edge for key docs views

Now we add **strong typing only where clients / LLMs actually rely on it**.

### 3.1. Pick the “tier-1” docs views

A good initial set:

* `docs.v_function_summary` → used to explore functions + risk.
* `docs.v_call_graph_enriched` → used for call graph context.
* `docs.v_file_summary` → file-level doc view.
* `docs.v_module_architecture` → module-level architecture.
* `docs.v_subsystem_profile` / `docs.v_subsystem_coverage` → subsystem-level story.

Everything else can stay generic for now.

---

### 3.2. Add view-specific Pydantic models

**File:** `serving/serving/mcp/models.py`

Right now you have a generic:

```python
class ViewRow(BaseModel):
    """Generic row wrapper for DuckDB view/table results."""
    model_config = ConfigDict(extra="allow")

    def __getitem__(self, key: str) -> object:
        ...
```

And responses like:

```python
class FunctionProfileResponse(BaseModel):
    found: bool
    profile: ViewRow | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
```

For the **docs views**, replace the `ViewRow` usage for tier-1 responses with strongly typed models:

**Example: function summary**

```python
class FunctionSummaryRow(BaseModel):
    """Row shape for docs.v_function_summary used by MCP surfaces."""

    repo: str
    commit: str
    rel_path: str
    function_goid_h128: int
    urn: str
    qualname: str
    language: str | None = None
    kind: str | None = None
    loc: int | None = None
    logical_loc: int | None = None
    cyclomatic_complexity: int | None = None
    complexity_bucket: str | None = None
    risk_score: float | None = None
    risk_level: str | None = None
    tested: bool | None = None
    test_count: int | None = None
    failing_test_count: int | None = None
    last_test_status: str | None = None
    # and any other fields from docs.v_function_summary that you care about


class FunctionSummaryResponse(BaseModel):
    """Response wrapper for a function summary lookup."""

    found: bool
    summary: FunctionSummaryRow | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
```

Do similar for:

* `CallGraphEnrichedRow` / `CallGraphEnrichedResponse` (for `docs.v_call_graph_enriched`),
* `FileSummaryRow` / `FileSummaryResponse` (for `docs.v_file_summary`),
* `ModuleArchitectureRow` / `ModuleArchitectureResponse`, etc.

Where other endpoints should remain generic, keep `ViewRow`.

---

### 3.3. Wire query_service to construct these typed responses

**File:** `serving/serving/mcp/query_service.py`

Find the methods that currently return generic view rows based on `docs.*` views, e.g.:

```python
def get_function_summary(self, *, goid_h128: int) -> FunctionSummaryResponse:
    """
    Return a function summary from docs.v_function_summary.
    """
    row = self._repos.functions.get_function_summary(
        repo=self.snapshot.repo,
        commit=self.snapshot.commit,
        goid_h128=goid_h128,
    )
    meta = ResponseMeta()
    if row is None:
        return FunctionSummaryResponse(found=False, summary=None, meta=meta)
    return FunctionSummaryResponse(found=True, summary=FunctionSummaryRow(**row), meta=meta)
```

Key details:

* Repository methods already return `RowDict` (`dict[str, Any]`).
* Pydantic model will ignore unknown fields by default if configured (`extra="ignore"`), or you can let it fail if you want strictness.

If you want to keep generic dataset browsing for these views as well, that’s fine; you’re just wrapping a strongly-typed version on top.

---

### 3.4. Leave `DatasetRowsResponse` generic

**File:** `serving/serving/mcp/models.py`

Keep the dataset browsing endpoints generic for all datasets (including docs.*):

```python
class DatasetRowsResponse(BaseModel):
    dataset: DatasetDescriptor
    rows: list[ViewRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
```

This is your “raw dataset explorer”.
The typed responses are *optimized facades* over specific docs views.

---

## 4. Optional: extend dataset infra slightly for docs views

If you want docs views to be more discoverable by type, you can:

### 4.1. Add a helper to `DatasetRegistry` for docs-only datasets

**File:** `storage/storage/datasets.py`

```python
@dataclass(frozen=True)
class DatasetRegistry:
    ...
    def docs_dataset_names(self) -> tuple[str, ...]:
        """Return dataset names corresponding to docs.* views."""
        return tuple(
            name for name, ds in self.by_name.items() if ds.is_view and ds.table_key.startswith("docs.")
        )
```

This gives agents an easy way to say, “show me all docs datasets”.

### 4.2. Expose docs datasets via HTTP/MCP

**File:** `serving/serving/services/query_service.py`

When building a dataset catalog, you can choose to separate docs vs non-docs:

```python
def list_datasets(self) -> list[DatasetDescriptor]:
    registry = self.gateway.datasets
    result: list[DatasetDescriptor] = []
    for name in registry.all_datasets:
        ds = registry.by_name[name]
        result.append(
            DatasetDescriptor(
                name=name,
                table=ds.table_key,
                is_view=ds.is_view,
                # family and description if you added them:
                family=getattr(ds, "family", None),
                description=getattr(ds, "description", None),
            )
        )
    return result
```

---

## 5. Tests & validation

### 5.1. Storage-level tests: docs views behave as datasets

**New file:** `storage/storage/tests/test_docs_views_as_datasets.py`

Tests:

1. **Every docs view is present in `metadata.datasets`**:

   ```python
   def test_docs_views_registered_in_metadata(con: DuckDBPyConnection) -> None:
       bootstrap_metadata_datasets(con)
       rows = con.execute(
           "SELECT table_key, is_view FROM metadata.datasets WHERE table_key LIKE 'docs.%'"
       ).fetchall()
       table_keys = {row[0] for row in rows}
       assert set(DOCS_VIEWS) <= table_keys
       assert all(row[1] for row in rows)  # is_view = true
   ```

2. **Docs views are readable through `DatasetReadRepository`**:

   ```python
   def test_docs_view_readable_via_dataset_rows(gateway: StorageGateway) -> None:
       repo = DatasetReadRepository(gateway=gateway)
       rows = repo.read_dataset_rows("v_function_summary", limit=5)
       # At least returns something; shape checked at serving level.
       assert isinstance(rows, list)
   ```

### 5.2. Serving-level tests: Pydantic contracts stay in sync

**New file:** `serving/serving/tests/test_function_summary_contract.py`

```python
def test_function_summary_response_model_matches_view_shape(
    backend: LocalQueryService,
) -> None:
    # Assume there is at least one function in the test snapshot
    any_summary = backend.get_function_summary(goid_h128=some_known_goid)
    assert isinstance(any_summary, FunctionSummaryResponse)
    if any_summary.found:
        row = any_summary.summary
        assert row is not None
        # basic type sanity checks
        assert isinstance(row.repo, str)
        assert isinstance(row.function_goid_h128, int)
```

You can add more detailed tests if you want to enforce that columns haven’t disappeared.

---

## 6. Implementation order (for your agent)

1. **Classify docs views** and introduce `ALIAS_DOCS_VIEWS` / `DERIVED_DOCS_VIEWS` (optional helper).
2. **Refactor gateway & repositories** to stop using alias docs views; point them at analytics tables.
3. **Remove alias view definitions** from `function_views.py`, `module_views.py`, `graph_views.py`; update `*_VIEW_NAMES` and `DOCS_VIEWS`.
4. **(Optional) Extend `Dataset` / `metadata.datasets`** with `family` / `description` if you want extra metadata.
5. **Add docs dataset helpers** (`docs_dataset_names`, etc.) if useful.
6. **Define typed Pydantic models** for tier-1 docs views in `serving/serving/mcp/models.py`.
7. **Wire `query_service`** methods to return typed responses for those views.
8. **Add tests** at storage and serving layers.

If you’d like, next we can zoom in on one slice (e.g. “remove alias docs views and refactor call sites”) and I can give you a PR-style patch plan like we did for the dataset contract.


# zoom in on remove alias doc views and refactor call sites #

Nice, this is a fun slice to tidy up. Let’s treat this as a mini-PR: **“Remove alias docs views and refactor call sites to use analytics datasets”**.

I’ll give you a **file-by-file patch plan** in unified-diff style so your implementation agent can drop these into a PR or patch script.

---

## Scope of this slice

We will:

1. **Change all call sites** that refer to these aliases:

   * `docs.v_function_profile` → `analytics.function_profile`
   * `docs.v_file_profile` → `analytics.file_profile`
   * `docs.v_module_profile` → `analytics.module_profile`

2. **Remove the alias view definitions** and their names from:

   * `storage/storage/views/function_views.py`
   * `storage/storage/views/module_views.py`
   * `storage/storage/views/graph_views.py` (for the config alias views)

3. **Update docstrings only** in `serving/serving/mcp/query_service.py` so they no longer talk about the docs aliases.

No tests reference these aliases, so no test edits are required for this slice.

---

## 1) Refactor call sites to use analytics tables

### 1.1. Functions repository → `analytics.function_profile`

**File:** `storage/storage/repositories/functions.py`

**Change:** SQL for `get_function_profile` should hit the analytics table directly.

```diff
diff --git a/storage/storage/repositories/functions.py b/storage/storage/repositories/functions.py
index AAAABBB..CCC DDD 100644
--- a/storage/storage/repositories/functions.py
+++ b/storage/storage/repositories/functions.py
@@ -120,16 +120,16 @@ class FunctionRepository(BaseRepository):
     def get_function_profile(self, goid_h128: int) -> RowDict | None:
         """
         Fetch a function profile by GOID.
@@ -132,12 +132,12 @@ class FunctionRepository(BaseRepository):
         -------
         RowDict | None
             Function profile row when found.
         """
         sql = """
             SELECT *
-            FROM docs.v_function_profile
+            FROM analytics.function_profile
             WHERE repo = ?
               AND commit = ?
               AND function_goid_h128 = ?
             LIMIT 1
         """
         return fetch_one_dict(self.con, sql, [self.repo, self.commit, goid_h128])
```

---

### 1.2. Modules repository → `analytics.file_profile` / `analytics.module_profile`

**File:** `storage/storage/repositories/modules.py`

**Change 1:** `get_file_profile` uses `analytics.file_profile`.

```diff
diff --git a/storage/storage/repositories/modules.py b/storage/storage/repositories/modules.py
index EEEEFFF..GGGHHH 100644
--- a/storage/storage/repositories/modules.py
+++ b/storage/storage/repositories/modules.py
@@ -85,12 +85,12 @@ class ModuleRepository(BaseRepository):
     def get_file_profile(self, rel_path: str) -> RowDict | None:
         """
         Return file profile row.
@@ -92,12 +92,12 @@ class ModuleRepository(BaseRepository):
         -------
         RowDict | None
             File profile when present.
         """
         sql = """
             SELECT *
-            FROM docs.v_file_profile
+            FROM analytics.file_profile
             WHERE repo = ?
               AND commit = ?
               AND rel_path = ?
             LIMIT 1
         """
         return fetch_one_dict(self.con, sql, [self.repo, self.commit, rel_path])
```

**Change 2:** `get_module_profile` uses `analytics.module_profile`.

```diff
@@ -120,12 +120,12 @@ class ModuleRepository(BaseRepository):
     def get_module_profile(self, module: str) -> RowDict | None:
         """
         Return module profile row.
@@ -126,12 +126,12 @@ class ModuleRepository(BaseRepository):
         -------
         RowDict | None
             Module profile when found.
         """
         sql = """
             SELECT *
-            FROM docs.v_module_profile
+            FROM analytics.module_profile
             WHERE repo = ?
               AND commit = ?
               AND module = ?
             LIMIT 1
         """
         return fetch_one_dict(self.con, sql, [self.repo, self.commit, module])
```

(Other uses of `docs.v_ide_hints` etc. remain unchanged; those are non-alias docs views.)

---

### 1.3. Storage gateway → analytics table instead of docs alias

**File:** `storage/storage/gateway.py`

**Change:** `DocsViews.function_profile()` should target `analytics.function_profile`.

```diff
diff --git a/storage/storage/gateway.py b/storage/storage/gateway.py
index IIIJJJ..KKKLLL 100644
--- a/storage/storage/gateway.py
+++ b/storage/storage/gateway.py
@@ -10450,15 +10450,17 @@ class DocsViews:
     def call_graph_enriched(self) -> DuckDBRelation:
         """
         Return relation for docs.v_call_graph_enriched.
@@ -10464,16 +10466,20 @@ class DocsViews:
         """
         return self.con.table("docs.v_call_graph_enriched")

     def function_profile(self) -> DuckDBRelation:
         """
-        Return relation for docs.v_function_profile.
+        Return relation for analytics.function_profile.
@@ -10470,10 +10476,10 @@ class DocsViews:
         Returns
         -------
         DuckDBRelation
-            Relation selecting docs.v_function_profile.
+            Relation selecting analytics.function_profile.
         """
-        return self.con.table("docs.v_function_profile")
+        return self.con.table("analytics.function_profile")
```

---

## 2) Remove alias view definitions and names

### 2.1. Remove `docs.v_function_profile` alias

**File:** `storage/storage/views/function_views.py`

**Change 1:** drop `docs.v_function_profile` from `FUNCTION_VIEW_NAMES`.

```diff
diff --git a/storage/storage/views/function_views.py b/storage/storage/views/function_views.py
index MMMNNN..OOOPPP 100644
--- a/storage/storage/views/function_views.py
+++ b/storage/storage/views/function_views.py
@@ -4,13 +4,12 @@ from __future__ import annotations
 
 from duckdb import DuckDBPyConnection
 
 FUNCTION_VIEW_NAMES: tuple[str, ...] = (
     "docs.v_function_summary",
     "docs.v_function_architecture",
     "docs.v_function_history",
     "docs.v_function_history_timeseries",
-    "docs.v_function_profile",
     "docs.v_cfg_block_architecture",
     "docs.v_dfg_block_architecture",
 )
```

**Change 2:** remove the alias `CREATE VIEW` block.

Original block (approximate):

```sql
    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_function_profile AS
        SELECT *
        FROM analytics.function_profile;
        """
    )
```

Delete this block entirely:

```diff
@@ -9805,11 +9805,6 @@ def create_function_views(con: DuckDBPyConnection) -> None:
-    con.execute(
-        """
-        CREATE OR REPLACE VIEW docs.v_function_profile AS
-        SELECT *
-        FROM analytics.function_profile;
-        """
-    )
+    # docs.v_function_profile has been removed; use analytics.function_profile
+    # directly in Python code (gateway/repositories) instead of an alias view.
```

(You can either delete the block completely or replace it with a comment as above.)

---

### 2.2. Remove `docs.v_file_profile` / `docs.v_module_profile` aliases

**File:** `storage/storage/views/module_views.py`

**Change 1:** drop both aliases from `MODULE_VIEW_NAMES`.

```diff
diff --git a/storage/storage/views/module_views.py b/storage/storage/views/module_views.py
index QQQRRR..SSSTTT 100644
--- a/storage/storage/views/module_views.py
+++ b/storage/storage/views/module_views.py
@@ -4,15 +4,13 @@ from __future__ import annotations
 
 from duckdb import DuckDBPyConnection
 
 MODULE_VIEW_NAMES: tuple[str, ...] = (
     "docs.v_module_history_timeseries",
     "docs.v_module_architecture",
     "docs.v_file_summary",
     "docs.v_entrypoints",
     "docs.v_external_dependencies",
     "docs.v_external_dependency_calls",
-    "docs.v_file_profile",
-    "docs.v_module_profile",
 )
```

**Change 2:** remove the alias view definitions.

Original (approximate):

```sql
    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_file_profile AS
        SELECT *
        FROM analytics.file_profile;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_module_profile AS
        SELECT *
        FROM analytics.module_profile;
        """
    )
```

Delete these blocks:

```diff
@@ -8525,19 +8525,6 @@ def create_module_views(con: DuckDBPyConnection) -> None:
-    con.execute(
-        """
-        CREATE OR REPLACE VIEW docs.v_file_profile AS
-        SELECT *
-        FROM analytics.file_profile;
-        """
-    )
-
-    con.execute(
-        """
-        CREATE OR REPLACE VIEW docs.v_module_profile AS
-        SELECT *
-        FROM analytics.module_profile;
-        """
-    )
+    # docs.v_file_profile and docs.v_module_profile aliases have been removed;
+    # use analytics.file_profile and analytics.module_profile directly instead.
```

---

### 2.3. Remove config metrics/projection alias docs views

These aliases are only referenced in `graph_views.py`, so they can be safely removed entirely (no call sites).

**File:** `storage/storage/views/graph_views.py`

**Change 1:** shrink `GRAPH_VIEW_NAMES`.

```diff
diff --git a/storage/storage/views/graph_views.py b/storage/storage/views/graph_views.py
index UUU VVV..WWWXXX 100644
--- a/storage/storage/views/graph_views.py
+++ b/storage/storage/views/graph_views.py
@@ -4,14 +4,11 @@ from __future__ import annotations
 
 from duckdb import DuckDBPyConnection
 
 GRAPH_VIEW_NAMES: tuple[str, ...] = (
     "docs.v_call_graph_enriched",
     "docs.v_symbol_module_graph",
-    "docs.v_config_graph_metrics_keys",
-    "docs.v_config_graph_metrics_modules",
-    "docs.v_config_projection_key_edges",
-    "docs.v_config_projection_module_edges",
     "docs.v_validation_summary",
 )
```

**Change 2:** delete the alias view definitions.

Original (approximate):

```sql
    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_config_graph_metrics_keys AS
        SELECT * FROM analytics.config_graph_metrics_keys;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_config_graph_metrics_modules AS
        SELECT * FROM analytics.config_graph_metrics_modules;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_config_projection_key_edges AS
        SELECT * FROM analytics.config_projection_key_edges;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_config_projection_module_edges AS
        SELECT * FROM analytics.config_projection_module_edges;
        """
    )
```

Delete these blocks:

```diff
@@ -2910,37 +2910,6 @@ def create_graph_views(con: DuckDBPyConnection) -> None:
-    con.execute(
-        """
-        CREATE OR REPLACE VIEW docs.v_config_graph_metrics_keys AS
-        SELECT * FROM analytics.config_graph_metrics_keys;
-        """
-    )
-
-    con.execute(
-        """
-        CREATE OR REPLACE VIEW docs.v_config_graph_metrics_modules AS
-        SELECT * FROM analytics.config_graph_metrics_modules;
-        """
-    )
-
-    con.execute(
-        """
-        CREATE OR REPLACE VIEW docs.v_config_projection_key_edges AS
-        SELECT * FROM analytics.config_projection_key_edges;
-        """
-    )
-
-    con.execute(
-        """
-        CREATE OR REPLACE VIEW docs.v_config_projection_module_edges AS
-        SELECT * FROM analytics.config_projection_module_edges;
-        """
-    )
+    # Alias docs views for config_graph_metrics_* and config_projection_* have
+    # been removed; use the analytics.* datasets directly instead.
```

After this, `DOCS_VIEWS` (assembled in `views/__init__.py`) will naturally stop including these aliases.

---

## 3) Update docstrings in `query_service` (optional but nice)

The only references in `serving/serving/mcp/query_service.py` are in docstrings; the logic already goes through the repositories.

**File:** `serving/serving/mcp/query_service.py`

Search for:

* `"Return a function profile from docs.v_function_profile."`
* `"Return a file profile from docs.v_file_profile."`
* `"Return a module profile from docs.v_module_profile."`

Patch them to refer to the canonical analytics datasets instead:

```diff
diff --git a/serving/serving/mcp/query_service.py b/serving/serving/mcp/query_service.py
index YYY ZZZ..AAA BBB 100644
--- a/serving/serving/mcp/query_service.py
+++ b/serving/serving/mcp/query_service.py
@@ -24640,7 +24640,7 @@ class QueryService:
-        """
-        Return a function profile from docs.v_function_profile.
+        """
+        Return a function profile from analytics.function_profile.
@@ -25680,7 +25680,7 @@ class QueryService:
-        """
-        Return a file profile from docs.v_file_profile.
+        """
+        Return a file profile from analytics.file_profile.
@@ -26700,7 +26700,7 @@ class QueryService:
-        """
-        Return a module profile from docs.v_module_profile.
+        """
+        Return a module profile from analytics.module_profile.
```

This keeps the documentation aligned with what the Python code now does.

---

## 4) Sanity checklist after applying this slice

For your implementation agent:

1. **Apply all file-level patches** above.
2. **Run the DuckDB bootstrap**:

   * Ensure `storage/storage/metadata_bootstrap.bootstrap_metadata_datasets` still works with the reduced `DOCS_VIEWS`.
   * Confirm `SELECT table_key FROM metadata.datasets WHERE table_key LIKE 'docs.%'` no longer shows the alias views.
3. **Run existing tests**:

   * There were no tests referencing the alias views, so nothing should break; but a full test run will confirm.
4. **Smoke test key flows**:

   * Function profile endpoint (CLI or MCP): now powered by `analytics.function_profile` directly.
   * File profile and module profile endpoints behave as before.
   * Any consumers of graph config/projection tables still work via `analytics.*` datasets.

If you’d like, next we can do a similar PR-style pass for the **“typed Pydantic models for docs views”** slice (e.g., adding `FunctionSummaryRow` / `FunctionSummaryResponse` and wiring `get_function_summary` to use them).


# zoom in on typed pydantic models for docs views #

Nice, let’s make `get_function_summary` properly typed end-to-end.

This slice will:

* Introduce a **typed row model** `FunctionSummaryRow` for `docs.v_function_summary`.
* Update `FunctionSummaryResponse` to use that row type instead of generic `ViewRow`.
* Wire `QueryService.get_function_summary` to construct that typed model.

I’ll give this as PR-style diffs.

---

## 1) Add `FunctionSummaryRow` and update `FunctionSummaryResponse`

**File:** `serving/serving/mcp/models.py`

We’ll add a new `FunctionSummaryRow` right after `ResponseMeta`, then update `FunctionSummaryResponse.summary` to use it.

```diff
diff --git a/serving/serving/mcp/models.py b/serving/serving/mcp/models.py
index AAAABBB..CCCCDDD 100644
--- a/serving/serving/mcp/models.py
+++ b/serving/serving/mcp/models.py
@@ -60,6 +60,32 @@ class Message(BaseModel):
     context: dict[str, object] | None = None
 
 
 class ResponseMeta(BaseModel):
     """Response metadata including clamping and messaging."""
@@ -73,11 +99,37 @@ class ResponseMeta(BaseModel):
     applied_offset: int | None = None
     truncated: bool = False
     messages: list[Message] = Field(default_factory=list)
 
 
+class FunctionSummaryRow(BaseModel):
+    """
+    Typed row model for ``docs.v_function_summary``.
+
+    This is a curated subset of columns that are most relevant for MCP
+    consumers and LLM agents. The underlying view may expose additional
+    columns, which remain accessible via generic dataset endpoints.
+
+    Fields are intentionally aligned with analytics.function_profile /
+    goid_risk_factors semantics.
+    """
+
+    repo: str
+    commit: str
+    rel_path: str
+    function_goid_h128: int
+    urn: str | None = None
+    language: str | None = None
+    kind: str | None = None
+    qualname: str | None = None
+    loc: int | None = None
+    logical_loc: int | None = None
+    cyclomatic_complexity: int | None = None
+    complexity_bucket: str | None = None
+    risk_score: float | None = None
+    risk_level: str | None = None
+
+
 class FunctionSummaryResponse(BaseModel):
     """Response wrapper for function summary lookups."""
 
-    found: bool
-    summary: ViewRow | None = None
-    meta: ResponseMeta = Field(default_factory=ResponseMeta)
+    found: bool
+    summary: FunctionSummaryRow | None = None
+    meta: ResponseMeta = Field(default_factory=ResponseMeta)
```

Notes:

* We only model a **subset** of the view’s columns. Extra columns in the row dict are ignored by Pydantic, so you stay flexible as the view evolves.
* If later you want tags/owners/etc., you can extend `FunctionSummaryRow` without touching the storage side.

---

## 2) Wire `QueryService.get_function_summary` to return the typed row

**File:** `serving/serving/mcp/query_service.py`

We need two changes:

1. Import `FunctionSummaryRow` from `models`.
2. Use it when constructing the response instead of `ViewRow`.

### 2.1. Update imports

Find the import block at the top:

```python
from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    DatasetRowsResponse,
    DatasetSchemaColumn,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
    FileHintsResponse,
    FileProfileResponse,
    FileSummaryResponse,
    FunctionArchitectureResponse,
    FunctionProfileResponse,
    FunctionSummaryResponse,
    GraphNeighborhoodResponse,
    HighRiskFunctionsResponse,
    ImportBoundaryResponse,
    Message,
    ModuleArchitectureResponse,
    ModuleProfileResponse,
    ModuleSubsystemResponse,
    ResponseMeta,
    SubsystemModulesResponse,
    SubsystemSearchResponse,
    TestsForFileResponse,
    TestsForFunctionResponse,
    ViewRow,
)
```

Patch it to include `FunctionSummaryRow`:

```diff
diff --git a/serving/serving/mcp/query_service.py b/serving/serving/mcp/query_service.py
index EEEEFFF..GGGHHH 100644
--- a/serving/serving/mcp/query_service.py
+++ b/serving/serving/mcp/query_service.py
@@ -18,6 +18,7 @@ from codeintel.serving.mcp.models import (
     DatasetSpecDescriptor,
     FileHintsResponse,
     FileProfileResponse,
     FileSummaryResponse,
     FunctionArchitectureResponse,
+    FunctionSummaryRow,
     FunctionProfileResponse,
     FunctionSummaryResponse,
@@ -33,7 +34,7 @@ from codeintel.serving.mcp.models import (
     ModuleProfileResponse,
     ModuleSubsystemResponse,
     ResponseMeta,
     SubsystemModulesResponse,
     SubsystemSearchResponse,
@@ -41,7 +42,7 @@ from codeintel.serving.mcp.models import (
     TestsForFunctionResponse,
     ViewRow,
 )
```

(Exact positions may differ slightly; just make sure `FunctionSummaryRow` is imported alongside the other models.)

### 2.2. Update `get_function_summary` implementation

Right now, the method ends like this:

```python
        row = self.functions.get_function_summary_by_goid(resolved)
        if row is None:
            meta.messages.append(
                Message(
                    code="not_found",
                    severity="info",
                    detail="Function not found",
                    context={
                        "urn": urn,
                        "goid_h128": goid_h128 or resolved,
                        "rel_path": rel_path,
                        "qualname": qualname,
                    },
                )
            )
            return FunctionSummaryResponse(found=False, summary=None, meta=meta)

        return FunctionSummaryResponse(
            found=True,
            summary=ViewRow.model_validate(row),
            meta=meta,
        )
```

We want it to build a `FunctionSummaryRow` instead.

```diff
diff --git a/serving/serving/mcp/query_service.py b/serving/serving/mcp/query_service.py
index GGGHHH..IIIJJJ 100644
--- a/serving/serving/mcp/query_service.py
+++ b/serving/serving/mcp/query_service.py
@@ -371,7 +371,7 @@ class QueryService:
         *,
         urn: str | None = None,
         goid_h128: int | None = None,
         rel_path: str | None = None,
         qualname: str | None = None,
@@ -380,7 +380,7 @@ class QueryService:
         """
         Return a function summary row from docs.v_function_summary.
@@ -396,6 +396,7 @@ class QueryService:
             If no identifier is provided.
         """
         meta = ResponseMeta()
@@ -418,6 +419,19 @@ class QueryService:
             )
             return FunctionSummaryResponse(found=False, summary=None, meta=meta)
 
-        return FunctionSummaryResponse(
-            found=True,
-            summary=ViewRow.model_validate(row),
-            meta=meta,
-        )
+        # At this point we have a single row from docs.v_function_summary.
+        # Map it into the typed FunctionSummaryRow model; extra columns from
+        # the view are ignored by the model.
+        summary = FunctionSummaryRow.model_validate(row)
+
+        return FunctionSummaryResponse(
+            found=True,
+            summary=summary,
+            meta=meta,
+        )
```

The rest of the method (argument validation, `_resolve_function_goid`, not-found handling) stays exactly as-is.

---

## 3) Optional: tiny test to lock the contract (if you want it)

If you’re ready to let tests enforce this, you can add a minimal test file.

**File (new):** `serving/serving/tests/test_function_summary_typed.py`

```python
from __future__ import annotations

from codeintel.serving.mcp.models import FunctionSummaryResponse
from codeintel.serving.services.query_service import LocalQueryService


def test_get_function_summary_returns_typed_row(example_query_service: LocalQueryService) -> None:
    """
    Basic smoke test: ensure get_function_summary returns a typed
    FunctionSummaryResponse with a FunctionSummaryRow payload when found.
    """

    # Arrange: pick a known GOID or use a helper that discovers one.
    # For now we assume the fixture provides a valid goid.
    goid = example_query_service.functions.list_high_risk_functions(min_risk=0.0).functions[0]["function_goid_h128"]

    # Act
    resp = example_query_service.get_function_summary(goid_h128=goid)

    # Assert
    assert isinstance(resp, FunctionSummaryResponse)
    assert resp.found is True
    assert resp.summary is not None
    # Spot-check a couple of typed fields
    assert isinstance(resp.summary.repo, str)
    assert isinstance(resp.summary.function_goid_h128, int)
```

You’d need to adapt this to your actual test fixtures (e.g. how you construct `LocalQueryService` in tests), but structurally this locks in:

* `FunctionSummaryResponse` is the return type,
* `.summary` is non-`None` for a valid GOID,
* `.summary` has the typed attributes we defined.

---

## Summary of this slice

* **Models:**
  Added `FunctionSummaryRow` and swapped `FunctionSummaryResponse.summary` from `ViewRow | None` to `FunctionSummaryRow | None`.

* **Query service:**
  `get_function_summary` now wraps the repository row in `FunctionSummaryRow` instead of `ViewRow`.

* **Behavior:**
  The underlying SQL and repository are unchanged; you’ve just given a strongly-typed shape to one of your key doc surfaces. Generic dataset browsing and other responses continue to use `ViewRow`.

If you’d like, we can do a follow-up slice for one more view (e.g. `docs.v_file_summary` → `FileSummaryRow`) in the same style, so you have a clear pattern to replicate.


# zoom in on `docs.v_file_summary` → `FileSummaryRow` #

Love it, let’s make `docs.v_file_summary` first-class and typed the same way we did for `docs.v_function_summary`.

This slice will:

* Add a **`FileSummaryRow`** Pydantic model that matches the file-summary view + nested function summaries.
* Change `FileSummaryResponse.file` to use `FileSummaryRow` instead of generic `ViewRow`.
* Wire `QueryService.get_file_summary` to build that model.

I’ll keep this in the same PR-style diff format as before.

---

## 1) Add `FileSummaryRow` and update `FileSummaryResponse`

### 1.1. Define `FileSummaryRow`

**File:** `serving/serving/mcp/models.py`

We’ll introduce `FileSummaryRow` just above `FileSummaryResponse`, and we’ll reuse `FunctionSummaryRow` for the nested `functions` field.

The view `docs.v_file_summary` looks like:

```sql
SELECT
    m.repo,
    m.commit,
    m.path          AS rel_path,
    m.module,
    m.language,
    m.tags,
    m.owners,
    am.node_count,
    am.function_count,
    am.class_count,
    am.avg_depth,
    am.max_depth,
    am.complexity   AS ast_complexity,
    h.score         AS hotspot_score,
    ty.type_error_count,
    ty.annotation_ratio,
    ty.untyped_defs,
    ty.overlay_needed,
    sd.total_errors,
    sd.has_errors,
    r.function_count,
    r.high_risk_functions,
    r.medium_risk_functions,
    r.low_risk_functions,
    r.max_risk_score
FROM core.modules m
...
```

We’ll model a **curated subset** plus a `functions` list.

```diff
diff --git a/serving/serving/mcp/models.py b/serving/serving/mcp/models.py
index AAAABBB..CCCCDDD 100644
--- a/serving/serving/mcp/models.py
+++ b/serving/serving/mcp/models.py
@@ -210,11 +210,44 @@ class ImportBoundaryResponse(BaseModel):
     nodes: list[ViewRow]
     edges: list[ViewRow]
     meta: ResponseMeta = Field(default_factory=ResponseMeta)
 
 
+class FileSummaryRow(BaseModel):
+    """
+    Typed row model for ``docs.v_file_summary`` used by MCP surfaces.
+
+    This captures the most important file-level fields plus nested function
+    summaries. The underlying view may expose additional columns (e.g. tags,
+    owners, AST counts) which remain accessible via generic dataset browsing.
+    """
+
+    repo: str
+    commit: str
+    rel_path: str
+    module: str | None = None
+    language: str | None = None
+
+    # AST / hotspot metrics
+    ast_complexity: float | None = None
+    hotspot_score: float | None = None
+
+    # Static diagnostics
+    total_errors: int | None = None
+    has_errors: bool | None = None
+
+    # Risk summary
+    function_count: int | None = None
+    high_risk_functions: int | None = None
+    medium_risk_functions: int | None = None
+    low_risk_functions: int | None = None
+    max_risk_score: float | None = None
+
+    # Nested per-function summaries for this file
+    functions: list[FunctionSummaryRow] = Field(default_factory=list)
+
+
 class FileSummaryResponse(BaseModel):
     """Summary of a file plus nested function rows."""
 
-    found: bool
-    file: ViewRow | None = None
-    meta: ResponseMeta = Field(default_factory=ResponseMeta)
+    found: bool
+    file: FileSummaryRow | None = None
+    meta: ResponseMeta = Field(default_factory=ResponseMeta)
```

> Notes:
>
> * We’re intentionally *not* modeling every column (tags, owners, AST counts, etc.). They still exist in the view and are usable via generic dataset endpoints; the typed model is a curated contract for MCP clients / LLM use.
> * If you later want tags/owners, you can extend `FileSummaryRow` without touching storage.

---

## 2) Wire `get_file_summary` to return `FileSummaryRow`

### 2.1. Import `FileSummaryRow` in `query_service`

**File:** `serving/serving/mcp/query_service.py`

At the top, you already import `FileSummaryResponse` & `FunctionSummaryRow` (from the previous slice). We’ll add `FileSummaryRow`.

Find the block:

```python
from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    DatasetRowsResponse,
    DatasetSchemaColumn,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
    FileHintsResponse,
    FileProfileResponse,
    FileSummaryResponse,
    FunctionArchitectureResponse,
    FunctionSummaryRow,
    FunctionProfileResponse,
    FunctionSummaryResponse,
    GraphNeighborhoodResponse,
    HighRiskFunctionsResponse,
    ...
    ViewRow,
)
```

Patch it:

```diff
diff --git a/serving/serving/mcp/query_service.py b/serving/serving/mcp/query_service.py
index EEEEFFF..GGGHHH 100644
--- a/serving/serving/mcp/query_service.py
+++ b/serving/serving/mcp/query_service.py
@@ -20,8 +20,10 @@ from codeintel.serving.mcp.models import (
     DatasetSpecDescriptor,
     FileHintsResponse,
     FileProfileResponse,
-    FileSummaryResponse,
+    FileSummaryResponse,
+    FileSummaryRow,
     FunctionArchitectureResponse,
     FunctionSummaryRow,
     FunctionProfileResponse,
     FunctionSummaryResponse,
```

(Exact ordering doesn’t matter; just ensure `FileSummaryRow` is imported.)

---

### 2.2. Change `get_file_summary` to use typed rows

Still in `query_service.py`, locate `get_file_summary` (you saw this earlier):

Current shape (simplified):

```python
    def get_file_summary(
        self,
        *,
        rel_path: str,
    ) -> FileSummaryResponse:
        ...
        file_row = self.modules.get_file_summary(rel_path)
        if not file_row:
            meta = ResponseMeta(
                messages=[Message(...)]
            )
            return FileSummaryResponse(found=False, file=None, meta=meta)

        funcs = self.functions.list_function_summaries_for_file(rel_path)
        file_payload = dict(file_row)
        file_payload["functions"] = [ViewRow.model_validate(r) for r in funcs]
        return FileSummaryResponse(
            found=True,
            file=ViewRow.model_validate(file_payload),
            meta=ResponseMeta(),
        )
```

We’ll now:

* Map each function to `FunctionSummaryRow`.
* Build a `FileSummaryRow` from `file_row` plus `functions`.

```diff
diff --git a/serving/serving/mcp/query_service.py b/serving/serving/mcp/query_service.py
index GGGHHH..IIIJJJ 100644
--- a/serving/serving/mcp/query_service.py
+++ b/serving/serving/mcp/query_service.py
@@ -520,24 +520,34 @@ class QueryService:
         Returns
         -------
         FileSummaryResponse
             Summary payload indicating whether the file was found.
         """
-        file_row = self.modules.get_file_summary(rel_path)
-        if not file_row:
-            meta = ResponseMeta(
-                messages=[
-                    Message(
-                        code="not_found",
-                        severity="info",
-                        detail="File not found",
-                        context={"rel_path": rel_path},
-                    )
-                ]
-            )
-            return FileSummaryResponse(found=False, file=None, meta=meta)
-
-        funcs = self.functions.list_function_summaries_for_file(rel_path)
-        file_payload = dict(file_row)
-        file_payload["functions"] = [ViewRow.model_validate(r) for r in funcs]
-        return FileSummaryResponse(
-            found=True,
-            file=ViewRow.model_validate(file_payload),
-            meta=ResponseMeta(),
-        )
+        file_row = self.modules.get_file_summary(rel_path)
+        if not file_row:
+            meta = ResponseMeta(
+                messages=[
+                    Message(
+                        code="not_found",
+                        severity="info",
+                        detail="File not found",
+                        context={"rel_path": rel_path},
+                    )
+                ]
+            )
+            return FileSummaryResponse(found=False, file=None, meta=meta)
+
+        # Collect per-function summaries for this file and map to typed rows.
+        func_rows = self.functions.list_function_summaries_for_file(rel_path)
+        functions = [FunctionSummaryRow.model_validate(r) for r in func_rows]
+
+        # Build a typed FileSummaryRow; extra columns from docs.v_file_summary
+        # (tags, owners, raw AST counts, etc.) are tolerated by the model.
+        file_payload = dict(file_row)
+        file_payload["functions"] = functions
+        file_model = FileSummaryRow.model_validate(file_payload)
+
+        return FileSummaryResponse(
+            found=True,
+            file=file_model,
+            meta=ResponseMeta(),
+        )
```

Behaviorally:

* If file not found: same as before.
* If found:

  * `functions` is now a list of `FunctionSummaryRow` models.
  * `file` is now a `FileSummaryRow` model that includes those functions.

---

## 3) Optional test snippet

If you’d like tests to lock this in, you can add a small smoke test similar to the function summary one.

**File (new):** `serving/serving/tests/test_file_summary_typed.py`

```python
from __future__ import annotations

from codeintel.serving.mcp.models import FileSummaryResponse
from codeintel.serving.services.query_service import LocalQueryService


def test_get_file_summary_returns_typed_row(example_query_service: LocalQueryService) -> None:
    """
    Ensure get_file_summary returns a typed FileSummaryResponse with
    a FileSummaryRow payload when the file exists.
    """
    # Discover a file we know exists in the test snapshot; adapt to your fixtures.
    any_file = example_query_service.functions.list_function_summaries(limit=1)[0].rel_path  # pseudo-code

    resp = example_query_service.get_file_summary(rel_path=any_file)

    assert isinstance(resp, FileSummaryResponse)
    if resp.found:
        assert resp.file is not None
        # Spot-check a few fields
        assert isinstance(resp.file.repo, str)
        assert isinstance(resp.file.rel_path, str)
        # Nested function summaries should be typed as well
        for fn in resp.file.functions:
            assert hasattr(fn, "function_goid_h128")
```

You’d adjust the “discover any_file” line to match how your test fixtures expose sample files.

---

## Summary of this slice

* **Models**:

  * Added `FileSummaryRow` for `docs.v_file_summary`.
  * Changed `FileSummaryResponse.file` from `ViewRow | None` to `FileSummaryRow | None`.

* **Query service**:

  * `get_file_summary` now maps:

    * `modules.get_file_summary(rel_path)` → `FileSummaryRow`,
    * `functions.list_function_summaries_for_file(rel_path)` → list of `FunctionSummaryRow`,
    * and returns a `FileSummaryResponse` populated with the typed model.

* **Generic dataset browsing** remains unchanged:

  * Your generic dataset endpoints still use `ViewRow` and the dataset infra.
  * These typed models are just “nice facades” on top of the docs views.

You now have a clean pattern:

* **Function summary**: `FunctionSummaryRow` / `FunctionSummaryResponse`.
* **File summary**: `FileSummaryRow` / `FileSummaryResponse`.

You can replicate the same approach for `docs.v_module_architecture`, `docs.v_subsystem_profile`, etc., following exactly this pattern.
