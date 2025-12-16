
Got it. I pulled your **PR‑82 + cleanup** repo snapshot apart and, as expected, a bunch of “next consolidation layers” are now *much* clearer (and now far less risky to implement cleanly).

Below is a **PR‑by‑PR tracking board** for the major “opportunities” (A–L from the prior list), updated to match what you’ve already built:

* Hamilton is the single graph source (but there are still a few lingering `targetgraph` defaults to excise)
* dynamic schema generation is “real” now (provider + compile + manifest)
* serving has a proper read‑only snapshot manager + pool + semantic kernel/query builder
* semantic registry compilation exists (and is already pretty close to being fully automatable)

I’ve also included **representative code snippets** for the core/complex piece of each big capability, plus **legacy deletions** to do as you go.

---

## Quick status check (post PR‑82 + cleanup)

### Already basically done (no big new PR needed)

* **Opportunity K (parameterization / no raw SQL)**: your serving query path is already Ibis‑first and doesn’t appear to accept arbitrary SQL. That’s already the right direction.
* **Opportunity C (batch schema inference)**: your schema inference provider is already doing the “compile to ephemeral DuckDB + DESCRIBE” style.

### Still clearly worth doing next (these are the boards below)

* **Opportunity A**: BuildSpec as the single inventory artifact needs a *bit* more (semantic + tags)
* **Opportunity D**: semantic registry should be compiled from Hamilton tags (not bespoke attrs / registries)
* **Opportunity I**: runtime allowlist should be enforced against schema manifest (stronger “safe by construction”)
* **Opportunity J**: add a DuckDB FTS‑based search primitive
* **Opportunity L**: move serving result extraction to DuckDB→Arrow/Polars instead of pandas
* **Opportunity E**: migrate remaining plugin targets to native Hamilton nodes (pandas-native or ibis-native), then delete plugin subsystem
* **Opportunity H**: once plugin subsystem is gone and nodes are tagged cleanly, enable Hamilton parallel execution safely

---


---

## PR‑84 — Make semantic metadata *Hamilton-native tags* (foundation for auto-registry)

This is the “don’t finalize full taxonomy yet, but make tags first-class and queryable via Hamilton”.

### Goal

Your semantic layer metadata should live in **Hamilton tags** so it can be discovered the same way you discover targets/datasets.

Hamilton tags are intended exactly for this sort of node metadata (tag filtering, structured discovery, etc.).

### Tasks

* **Add semantic tag constants**

  * `src/codeintel/build/hamilton/tags.py`

    * Add keys (names are examples):

      * `TAG_OUTPUT_KIND = "output_kind"`
      * `OUTPUT_KIND_SEMANTIC_VIEW = "semantic_view"`
      * `TAG_SEMANTIC_ID = "semantic_id"`
      * `TAG_ENTITY = "entity"`
      * `TAG_GRAIN = "grain"`
      * `TAG_TABLE_KEY = "table_key"`
      * `TAG_MCP_VISIBLE = "mcp_visible"`
* **Update the semantic decorator to also apply Hamilton tags**

  * `src/codeintel/build/serving/semantic_tags.py`

    * Keep your current `SemanticViewTags` attribute if you want, but *also* apply Hamilton’s `@tag(...)`.
    * Hamilton supports tag values as strings or list[str].

### Representative code snippet (core idea)

```python
# src/codeintel/build/serving/semantic_tags.py
from __future__ import annotations

from dataclasses import asdict
from hamilton.function_modifiers import tag as h_tag  # Hamilton tag decorator 

from codeintel.build.hamilton import tags as ht

SEMANTIC_VIEW_TAG_ATTR = "_codeintel_semantic_view_tags"


def semantic_view(*, semantic_id: str, table_key: str, entity: str, grain: str,
                  description: str | None = None,
                  columns: list[str] | None = None,
                  mcp_visible: bool = True):
    def decorator(fn):
        # Preserve your existing metadata mechanism (optional)
        setattr(fn, SEMANTIC_VIEW_TAG_ATTR, {
            "semantic_id": semantic_id,
            "table_key": table_key,
            "entity": entity,
            "grain": grain,
            "description": description,
            "columns": columns,
            "mcp_visible": mcp_visible,
        })

        # Make it Hamilton-native: tags are discoverable/filterable
        hamilton_tags = {
            ht.TAG_OUTPUT_KIND: ht.OUTPUT_KIND_SEMANTIC_VIEW,
            ht.TAG_SEMANTIC_ID: semantic_id,
            ht.TAG_TABLE_KEY: table_key,
            ht.TAG_ENTITY: entity,
            ht.TAG_GRAIN: grain,
            ht.TAG_MCP_VISIBLE: "true" if mcp_visible else "false",
        }
        if description:
            hamilton_tags["description"] = description
        if columns:
            # list[str] is supported 
            hamilton_tags["columns"] = columns

        return h_tag(**hamilton_tags)(fn)

    return decorator
```

### Tests

* `tests/build/serving/test_pr84_semantic_view_hamilton_tags.py`

  * Import `storage.views.ibis_views`
  * Assert at least one view function has Hamilton tags applied.

### Snapshots

* None yet.

### Legacy deletions

* None yet (this is “additive foundation”).

---

## PR‑85 — Compile semantic_registry.json from Hamilton tag discovery (delete bespoke registry plumbing)

This is Opportunity D.

### Goal

Stop relying on “manual registries” (including `register_view(...)`) to decide what semantic views exist.
Instead:

* discover semantic views by scanning a module with Hamilton,
* filter nodes by tag (`output_kind=semantic_view`),
* write `semantic_registry.json`.

Hamilton supports listing variables with tag filters.

### Tasks

* **Add a semantic compiler that uses Hamilton tag filtering**

  * New module: `src/codeintel/build/serving/semantic_compile_hamilton.py`

    * Build a Driver with the semantic view module(s)
    * `list_available_variables(tag_filter=...)`
    * For each node:

      * read `semantic_id`, `table_key`, `entity`, `grain`, etc from node tags
      * determine columns:

        * if `columns` tag exists → use it
        * else → pull from schema provider (manifest)
* **Switch existing compile path to use it**

  * Update `src/codeintel/build/serving/semantic_compile.py`

    * Replace `collect_semantic_view_tags(view_registry, ...)` with Hamilton discovery.
* **Keep compatibility temporarily**

  * Keep your existing JSON schema output identical so serving doesn’t change.

### Representative code snippet (core idea)

```python
# src/codeintel/build/serving/semantic_compile_hamilton.py
from __future__ import annotations

from hamilton import driver
from codeintel.build.hamilton import tags as ht
from codeintel.storage.views import ibis_views

def discover_semantic_nodes() -> list[tuple[str, dict[str, object]]]:
    dr = driver.Driver({}, ibis_views)

    # Filter by tags; Hamilton supports tag-based filtering on discovery 
    vars_ = dr.list_available_variables(
        tag_filter={ht.TAG_OUTPUT_KIND: ht.OUTPUT_KIND_SEMANTIC_VIEW}
    )

    out = []
    for v in vars_:
        # v.name is the node name (function name)
        # v.tags is the tag dictionary
        out.append((v.name, v.tags))
    return out
```

### Tests

* `tests/build/serving/test_pr85_semantic_registry_compiles_from_tags.py`

  * Compile registry
  * Assert it contains expected semantic IDs
  * Assert no registry source file needs `register_view(...)`

### Snapshots

* Add a CLI command if you don’t already have one:

  * `codeintel build semantic compile --format json`
* Snapshot:

  * `tests/build/hamilton/snapshots/pr85_semantic_compile_snapshot.json`

### Legacy deletions (in this PR or next)

* If `semantic_sources.py` becomes unused:

  * delete `src/codeintel/build/serving/semantic_sources.py`

---

## PR‑86 — Remove `IbisViewRegistry` + `register_view` (pure tag/module discovery)

This is the real “no more registry edits ever” cut.

### Tasks

* Delete:

  * `src/codeintel/storage/views/ibis_registry.py`
* Update:

  * `src/codeintel/storage/views/ibis_views.py`

    * remove `@register_view(...)` decorators
  * `src/codeintel/storage/views/creation.py`

    * discover views via Hamilton tag scan (same as PR‑85)
    * create views by calling the functions by name (or call directly on the function objects)

### Tests

* `tests/storage/views/test_pr86_no_ibis_view_registry.py`

  * Import `ibis_views`
  * Ensure views can be created without registry object

### Snapshots

* None.

### Legacy deletions

* `src/codeintel/storage/views/ibis_registry.py`
* Remove any remaining imports/usage of `register_view` across repo.

---

## PR‑87 — Runtime semantic query safety: enforce allowlists against schema manifest

This is Opportunity I “make it impossible to query a column that isn’t in the schema manifest”.

### Goal

Even if registry + schema drift, the runtime should *always* use the schema manifest as the source of truth for:

* what table_key exists
* what columns exist

### Tasks

* Update serving kernel to compute `allowed_columns` like:

  * `schema_cols = schema_inventory[view.table_key].columns`
  * if registry provides columns → enforce `set(registry_cols) ⊆ set(schema_cols)`

    * strict: raise error
    * warn: intersect and continue
  * if registry provides none → use `schema_cols` (canonical)
* Add Serving setting:

  * `SERVING_SCHEMA_ENFORCEMENT = "strict" | "warn" | "off"`

### Representative code snippet (core idea)

```python
# src/codeintel/serving/semantic/kernel.py
def _resolve_allowed_columns(self, view: SemanticViewSpec) -> list[str]:
    schema = self.schema_inventory.get(view.table_key)
    if not schema:
        raise ValueError(f"View table_key not present in schema manifest: {view.table_key}")

    schema_cols = list(schema.columns)

    if not view.columns:
        return schema_cols

    unknown = sorted(set(view.columns) - set(schema_cols))
    if unknown:
        mode = self.settings.schema_enforcement
        if mode == "strict":
            raise ValueError(f"Semantic view {view.semantic_id} exposes unknown columns: {unknown}")
        if mode == "warn":
            # log warning then intersect
            pass

    return [c for c in view.columns if c in schema_cols]
```

### Tests

* `tests/serving/semantic/test_pr87_allowed_columns_enforced.py`

  * Create a fake semantic registry spec that includes a bogus column
  * Assert strict mode errors
  * Assert warn mode intersects

### Snapshots

* None.

### Legacy deletions

* None (but this allows you to stop manually curating allowlists entirely).

---

## PR‑88 — Replace pandas result extraction with DuckDB→Arrow/Polars pipeline

This is Opportunity L (serving performance + better typing).

DuckDB’s Python client supports producing Arrow and Polars outputs directly (e.g., `.pl()` for Polars). ([DuckDB][1])

### Tasks

* Add an execution path:

  * compile Ibis expr → SQL
  * run SQL via DuckDB connection
  * fetch as Arrow/Polars
  * convert to JSON‑safe rows
* Keep pandas fallback behind a flag for safety while migrating.

### Representative code snippet (core idea)

```python
# src/codeintel/serving/semantic/execution.py
from __future__ import annotations
import datetime as dt
import polars as pl

def polars_rows(df: pl.DataFrame) -> list[dict]:
    rows = df.to_dicts()
    # normalize datetimes/decimals if needed for JSON
    for r in rows:
        for k, v in list(r.items()):
            if isinstance(v, (dt.datetime, dt.date)):
                r[k] = v.isoformat()
    return rows

def execute_ibis_to_polars(gateway, expr) -> tuple[list[dict], list[str]]:
    sql = gateway.ibis.compile(expr)  # SQL string
    # duckdb connection supports fetching Polars directly (pl()) :contentReference[oaicite:7]{index=7}
    df = gateway.con.execute(sql).pl()
    return polars_rows(df), list(df.columns)
```

### Tests

* `tests/serving/semantic/test_pr88_polars_execution_path.py`

  * Seed a tiny DuckDB with a minimal table
  * Run a semantic query and assert output rows identical to prior path

### Snapshots

* None.

### Legacy deletions

* Once stable:

  * remove pandas‑first execution path in `serving/semantic/kernel.py`
  * keep pandas only where needed for build steps

---

## PR‑89 — Add CLI “semantic” utilities for offline introspection + snapshot gating

This is optional but *high leverage* for CI gating and reproducibility.

### Goal

Make semantic layer testable without spinning up FastAPI:

* `codeintel serve semantic catalog`
* `codeintel serve semantic describe <id>`
* `codeintel serve semantic query --view ... --limit ...`

### Tasks

* Extend `src/codeintel/cli/commands/serve.py`

  * Add a `semantic` Typer group
  * Implement commands that:

    * load ServingDBManager from `--snapshot-dir` (or default)
    * call kernel methods and print deterministic JSON

### Tests

* Use your snapshot runner:

  * Add commands to `tests/build/hamilton/snapshots/manifest.yaml`

### Snapshots

* `tests/build/hamilton/snapshots/pr89_semantic_catalog.json`
* `tests/build/hamilton/snapshots/pr89_semantic_describe_function_summary.json`
* `tests/build/hamilton/snapshots/pr89_semantic_query_function_summary.json`

### Legacy deletions

* None.

---

## PR‑90 — Build a DuckDB FTS index for code metadata (search primitive, build side)

This is Opportunity J part 1.

DuckDB’s FTS extension is created via `INSTALL fts; LOAD fts;` and `PRAGMA create_fts_index(...)`. ([DuckDB][2])
It creates a schema like `fts_main_<table>` and macros like `match_bm25`. ([DuckDB][2])

### Goal

Create a search index table **inside the serving snapshot DB** that supports:

* fuzzy text search across symbols/functions/modules/docstrings
* stable ranking

### Tasks

* Add build/serving step:

  * `src/codeintel/build/serving/search_index.py` (new)

    * `build_search_documents_table(gateway)`:

      * create `docs.search_documents` with canonical columns, e.g.:

        * `kind` (`function|module|symbol|docstring`)
        * `name`
        * `module`
        * `text`
        * `ref_goid` (optional)
    * `ensure_fts_index(gateway)`:

      * INSTALL/LOAD
      * `PRAGMA create_fts_index('docs.search_documents', 'text', 'name', 'module')`
* Wire into publisher:

  * `src/codeintel/build/serving/publisher.py`

    * After copying DB → open snapshot DB and build index

### Representative code snippet (core idea)

```python
# src/codeintel/build/serving/search_index.py
def ensure_fts_index(con) -> None:
    con.execute("INSTALL fts;")  # :contentReference[oaicite:10]{index=10}
    con.execute("LOAD fts;")     # :contentReference[oaicite:11]{index=11}
    con.execute("""
        PRAGMA create_fts_index(
            'docs.search_documents',
            'text',
            'name',
            'module'
        );
    """)  # :contentReference[oaicite:12]{index=12}
```

### Tests

* `tests/build/serving/test_pr90_search_index_builds.py`

  * Create a tiny DuckDB file
  * Create a minimal `docs.search_documents`
  * Run `ensure_fts_index`
  * Assert that `fts_main_docs_search_documents` schema exists (via `information_schema.schemata`)

### Snapshots

* If you add CLI command:

  * `codeintel build serving build-search-index --snapshot <dir>`
* Snapshot:

  * `tests/build/hamilton/snapshots/pr90_build_search_index.txt`

### Legacy deletions

* None.

---



## PR‑96 — enable safe Hamilton parallel execution

 Add Hamilton parallel execution (safe mode)

Hamilton supports executing DAGs with threadpool-based adapters for I/O bound work. ([Hamilton][3])
GraphAdapters are the mechanism for customizing execution. ([Hamilton][4])

#### Practical safe approach for CodeIntel

* Run **compute nodes in parallel**
* Run **materialize nodes under a global “write lock”**
* Use **separate DuckDB connections per thread** for reads if any node touches DB

#### Tasks

* Add build option:

  * `codeintel build run --workers N`
* In `driver_factory.py`:

  * create a threadpool adapter / graph adapter
  * gate write nodes by tag (`node_type=materialize`)

#### Representative snippet (conceptual)

```python
# Pseudocode: a “write-locked” adapter wrapping threadpool execution
# (exact adapter classes depend on your chosen Hamilton adapter API) :contentReference[oaicite:16]{index=16}

write_lock = threading.Lock()

def execute_node(node, kwargs):
    if node.tags.get("node_type") == "materialize":
        with write_lock:
            return node.callable(**kwargs)
    return node.callable(**kwargs)
```

#### Tests

* `tests/build/hamilton/test_pr96_parallel_execution_smoke.py`

  * run a small closure with `--workers 4`
  * assert deterministic manifest + no deadlocks

#### Snapshots

* Add a snapshot case:

  * `codeintel build run --target risk_factors --workers 4 --dry-run` (or similar)
  * `tests/build/hamilton/snapshots/pr96_parallel_run_help.txt`

---

# Additional “best-in-class” opportunities beyond A–L (optional follow-ons)

These aren’t in the original A–L list, but they’re *very aligned* with your “MCP-first consumer” goal and Hamilton centricity:

1. **Expose “semantic layer prompt pack” in `/meta` + MCP**

   * Include:

     * semantic IDs + descriptions
     * common join paths (“if you need callers, use X then join Y on …”)
     * recommended defaults for limit/order_by
   * This dramatically improves LLM agent performance without any custom query language.

2. **Add `EXPLAIN` / query plan tool**

   * `semantic_explain(view_id, request)` returns DuckDB plan + estimated rows
   * Useful for debugging and “agent self-correction”.

3. **Semantic view test harness**

   * For each view:

     * verify it compiles (Ibis→SQL)
     * verify its output schema matches schema manifest
   * This becomes your “semantic CI gate”.



---

# What I’d delete as “legacy” as these land (high confidence list)

You asked to highlight deletions per change — above I did per PR. Here’s the big-picture deletion set you should end up with once PR‑96 lands:

* `src/codeintel/build/plugins/**` (entire subsystem)
* any “target execution wrappers” that:

  * directly write to DuckDB inside analytics/ingestion/graphs code
  * exist solely to support plugin execution
* `src/codeintel/storage/views/ibis_registry.py` + `register_view` usage (once PR‑86 lands)
* any remaining semantic registry compilation paths that do not flow through Hamilton tags

---


M

[1]: https://duckdb.org/docs/stable/guides/python/polars.html "Integration with Polars – DuckDB"
[2]: https://duckdb.org/docs/stable/core_extensions/full_text_search.html "Full-Text Search Extension – DuckDB"
[3]: https://hamilton.apache.org/concepts/parallel-task/?utm_source=chatgpt.com "Dynamic DAGs/Parallel Execution - Hamilton"
[4]: https://hamilton.staged.apache.org/reference/graph-adapters/SimplePythonGraphAdapter/?utm_source=chatgpt.com "SimplePythonGraphAdapter - Hamilton"
