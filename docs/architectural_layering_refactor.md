
# Improved architectural layering refactor #

Here’s how I’d turn that layering idea into concrete refactors on your **current** repo (i.e., the updated zips you just sent).

I’ll structure this as three main workstreams:

1. Move NetworkX backend config out of `cli` into `graphs` (fixing the CLI → graphs layering leak).
2. Move `load_module_map` into a neutral storage-layer module (so analytics/graphs don’t depend on ingestion).
3. Untangle the config ↔ storage coupling around SQL helpers so “core” really is core.

At the end I’ll suggest a tiny static layering check you can wire into CI.

---

## 1. Move `maybe_enable_nx_gpu` into `graphs` (domain), not `cli` (application)

### 1.1. Add `graphs.nx_backend` as the canonical backend helper

**New file**: `graphs/nx_backend.py`

Copy the existing implementation from `cli/nx_backend.py`, but under the graphs package:

```python
# graphs/nx_backend.py
"""Helpers for configuring NetworkX backends (CPU vs GPU)."""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable, MutableMapping

from codeintel.config.primitives import GraphBackendConfig

LOG = logging.getLogger(__name__)
_GPU_AUTOCONFIG_ENV = "NX_CUGRAPH_AUTOCONFIG"


def _enable_nx_cugraph_backend() -> None:
    """
    Import and enable the nx-cugraph backend if available.

    Raises
    ------
    RuntimeError
        If the backend cannot be imported or enabled.
    """
    nx_cugraph = importlib.import_module("networkx.algorithms.centrality.nxcugraph")
    # Just importing is typically enough to register the backend; call a hook
    # if the package exposes one (kept generic to avoid tight coupling).
    if hasattr(nx_cugraph, "ensure_backend"):
        nx_cugraph.ensure_backend()


def maybe_enable_nx_gpu(
    cfg: GraphBackendConfig,
    *,
    env: MutableMapping[str, str] | None = None,
    enabler: Callable[[], None] | None = None,
) -> None:
    """
    Configure NetworkX backend based on GraphBackendConfig.

    Parameters
    ----------
    cfg : GraphBackendConfig
        Backend selection options.
    env : MutableMapping[str, str] | None, optional
        Environment mapping to mutate; defaults to os.environ.
    enabler : Callable[[], None] | None, optional
        Callback that enables the GPU backend; defaults to nx-cugraph enabler.
    """
    env_vars = os.environ if env is None else env
    enable_backend = enabler or _enable_nx_cugraph_backend

    backend = cfg.backend
    if backend == "cpu":
        env_vars[_GPU_AUTOCONFIG_ENV] = "False"
        LOG.info("Graph backend pinned to CPU.")
        return

    if backend in {"auto", "nx-cugraph"}:
        env_vars.setdefault(_GPU_AUTOCONFIG_ENV, "True")
        try:
            enable_backend()
        except RuntimeError:
            if cfg.strict:
                LOG.exception("Failed to enable GPU backend (strict=True).")
                raise
            LOG.exception("Failed to enable GPU backend; continuing with CPU backend.")
        return

    LOG.warning("Unknown graph backend '%s'; using CPU backend.", backend)
```

This module:

* Depends only on `config.primitives.GraphBackendConfig` (core) and standard libs.
* Lives in `graphs` (domain), so it’s perfectly fine for both CLI and pipeline to call it.

### 1.2. Update `graphs.engine_factory` to depend on `graphs.nx_backend`

Current import (from `graphs/engine_factory.py`):

```python
from codeintel.cli.nx_backend import maybe_enable_nx_gpu
```

Change it to:

```python
from codeintel.graphs.nx_backend import maybe_enable_nx_gpu
```

Nothing else in `engine_factory` needs to change — the call site:

```python
if graph_backend is not None:
    if graph_backend.backend not in allowed_backends:
        ...
    maybe_enable_nx_gpu(graph_backend, env=env)
```

stays exactly the same, just with the new import.

This removes the **domain → application** dependency.

### 1.3. Make `cli.nx_backend` a thin shim around `graphs.nx_backend` (optional but nice)

To avoid breaking any internal tooling that might import `codeintel.cli.nx_backend`, you can keep the module but make it a re-export:

```python
# cli/nx_backend.py
"""CLI-facing shim around graphs.nx_backend."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Callable

from codeintel.config.primitives import GraphBackendConfig
from codeintel.graphs.nx_backend import maybe_enable_nx_gpu as _maybe_enable_nx_gpu

__all__ = ["maybe_enable_nx_gpu"]


def maybe_enable_nx_gpu(
    cfg: GraphBackendConfig,
    *,
    env: MutableMapping[str, str] | None = None,
    enabler: Callable[[], None] | None = None,
) -> None:
    _maybe_enable_nx_gpu(cfg, env=env, enabler=enabler)
```

You could also just delete `cli/nx_backend.py` and update imports everywhere, but the shim keeps git history and external callers happy.

### 1.4. Update application-layer imports (CLI + Prefect)

**CLI** (`cli/main.py`):

```python
- from codeintel.cli.nx_backend import maybe_enable_nx_gpu
+ from codeintel.graphs.nx_backend import maybe_enable_nx_gpu
```

Leave the two call sites as-is:

```python
maybe_enable_nx_gpu(cfg.graph_backend)
```

**Prefect flow** (`pipeline/orchestration/prefect_flow.py`):

```python
- from codeintel.cli.nx_backend import maybe_enable_nx_gpu
+ from codeintel.graphs.nx_backend import maybe_enable_nx_gpu
```

Call site:

```python
graph_backend = args.graph_backend or GraphBackendConfig()
_GRAPH_BACKEND_STATE["config"] = graph_backend
maybe_enable_nx_gpu(graph_backend)
```

unchanged.

At this point:

* **graphs** no longer imports `codeintel.cli.*`.
* **CLI** and **pipeline** both depend on `graphs.nx_backend` (application → domain), which is what we want.

---

## 2. Move `load_module_map` into storage and decouple analytics/graphs from ingestion

Right now:

* `load_module_map` lives in **`ingestion/common.py`**.
* It’s used by:

  * `ingestion/docstrings_ingest.py`
  * `analytics/context.py`
  * `graphs/function_catalog.py`

This makes analytics and graphs depend on ingestion, even though `load_module_map` is purely a **read-side** helper over `core.modules`.

### 2.1. Introduce `storage.module_index` (or `storage.queries`)

Create a new module, e.g. **`storage/module_index.py`**:

```python
# storage/module_index.py
"""Read-side helpers for module metadata from core.modules."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Final

from codeintel.ingestion.paths import normalize_rel_path  # or move normalize into storage
from codeintel.storage.gateway import StorageGateway

LOG: Final = logging.getLogger(__name__)


def load_module_map(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    *,
    language: str | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, str]:
    """
    Load path -> module mapping from core.modules.

    Parameters
    ----------
    gateway :
        Storage gateway bound to the target DuckDB database.
    repo : str
        Repository slug.
    commit : str
        Commit SHA for the snapshot.
    language : str | None, optional
        Filter by language if provided.
    logger : logging.Logger | None, optional
        Logger for warnings; defaults to module logger.

    Returns
    -------
    dict[str, str]
        Normalized mapping of relative path -> module name.
    """
    con = gateway.con
    params: list[object] = [repo, commit]
    query = """
        SELECT path, module
        FROM core.modules
        WHERE repo = ? AND commit = ?
        """
    if language is not None:
        query += " AND language = ?"
        params.append(language)

    rows = con.execute(query, params).fetchall()
    module_map = {
        normalize_rel_path(str(path)): str(module)
        for path, module in rows
    }
    if not module_map:
        (logger or LOG).warning("No modules found in core.modules for %s@%s", repo, commit)
    return module_map
```

Notes:

* This is basically your current `ingestion.common.load_module_map`, just relocated.
* I’ve left `normalize_rel_path` in `ingestion.paths` for now; if you want, you can later move it into a neutral path module (e.g. `storage.paths` or `core.paths`).

### 2.2. Stop defining `load_module_map` inside `ingestion.common`

In **`ingestion/common.py`**:

* Delete the existing `load_module_map` definition.
* Remove its import of `StorageGateway` if only used there.
* Add an import from storage:

```python
- from codeintel.storage.gateway import StorageGateway
+ from codeintel.storage.gateway import StorageGateway
+ from codeintel.storage.module_index import load_module_map
```

(You likely still need `StorageGateway` in other helpers in this file, so keep that import if used.)

Within `ingestion.common.iter_modules`, change the use of the now-removed local `load_module_map` to import from storage (if it still calls it).

### 2.3. Update all callers to use `storage.module_index`

**1) `ingestion/docstrings_ingest.py`**

Currently:

```python
from codeintel.ingestion.common import (
    iter_modules,
    load_module_map,
    run_batch,
    time_operation,
)
```

Change to:

```python
from codeintel.ingestion.common import iter_modules, run_batch, time_operation
from codeintel.storage.module_index import load_module_map
```

Call site stays identical:

```python
module_map = load_module_map(gateway, cfg.repo, cfg.commit, language="python", logger=log)
```

**2) `analytics/context.py`**

Currently:

```python
from codeintel.ingestion.common import load_module_map
from codeintel.storage.gateway import StorageGateway
```

Change to:

```python
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.module_index import load_module_map
```

Call site:

```python
module_map = load_module_map(gateway, cfg.repo, cfg.commit)
```

unchanged.

**3) `graphs/function_catalog.py`**

Currently:

```python
from codeintel.graphs.function_index import FunctionSpan, FunctionSpanIndex
from codeintel.ingestion.common import load_module_map
from codeintel.ingestion.paths import normalize_rel_path
from codeintel.storage.gateway import StorageGateway
```

Change to:

```python
from codeintel.graphs.function_index import FunctionSpan, FunctionSpanIndex
from codeintel.ingestion.paths import normalize_rel_path
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.module_index import load_module_map
```

The bottom of the file becomes:

```python
    module_by_path = load_module_map(gateway, repo, commit)
    return FunctionCatalog(functions=functions, module_by_path=module_by_path)
```

unchanged.

At this point:

* **ingestion** still uses `load_module_map` as part of iterating modules (totally fine).
* **analytics** and **graphs** now depend on `storage.module_index` instead of `ingestion.common`, which matches the “domain → storage/core” direction we want.

---

## 3. Clarify config ↔ storage boundaries around SQL helpers

Today we have:

* `config.schemas.tables` — logical schema definitions (clean, core).
* `storage.schemas` — DuckDB DDL creation, imports `TABLE_SCHEMAS` from `config.schemas.tables` (core → core, fine).
* `config.schemas.sql_builder` — column lists **plus** connection-aware helpers:

  * `PreparedStatements`
  * `prepared_statements_dynamic(con: DuckDBConnection, table_key: str)`
  * `ensure_schema(con: DuckDBConnection, table_key: str)`
  * And it imports `DuckDBConnection` from `storage.gateway` (config → storage).
* `ingestion.common` imports `prepared_statements_dynamic` from `config.schemas.sql_builder`.
* `graphs.validation` imports `ensure_schema` from `config.schemas.sql_builder`.

We want:

* “Config” side (`config.schemas.*`) to be **pure metadata + SQL string generation**, not connection-aware.
* “Storage” side (`storage.*`) to be where anything that takes a DuckDB connection lives.

### 3.1. Move connection-aware helpers into `storage.sql_helpers`

Create **`storage/sql_helpers.py`**:

```python
# storage/sql_helpers.py
"""Connection-aware schema helpers for ingestion and validations."""

from __future__ import annotations

from dataclasses import dataclass

from duckdb import DuckDBPyConnection

from codeintel.config.schemas.ingestion_sql import verify_ingestion_columns
from codeintel.config.schemas.registry_adapter import load_registry_columns


DuckDBConnection = DuckDBPyConnection


@dataclass
class PreparedStatements:
    """Prepared insert/delete SQL for a table."""

    insert_sql: str
    delete_sql: str | None = None


def prepared_statements_dynamic(
    con: DuckDBConnection,
    table_key: str,
) -> PreparedStatements:
    """
    Return prepared SQL using registry-derived column order for macro-backed tables.
    """
    registry_cols = load_registry_columns(con).get(table_key)
    if registry_cols is None:
        message = f"Table {table_key} missing from registry"
        raise RuntimeError(message)
    column_list = ", ".join(registry_cols)
    placeholders = ", ".join("?" for _ in registry_cols)
    insert_sql = f"INSERT INTO {table_key} ({column_list}) VALUES ({placeholders})"
    delete_sql = f"DELETE FROM {table_key} WHERE repo = ? AND commit = ?"  # or None if not needed
    return PreparedStatements(insert_sql=insert_sql, delete_sql=delete_sql)


def ensure_schema(con: DuckDBConnection, table_key: str) -> None:
    """
    Validate that the live DuckDB table matches the registry definition.

    Checks column presence/order and NOT NULL flags.
    """
    verify_ingestion_columns(con)
    registry_cols = load_registry_columns(con).get(table_key)
    if registry_cols is None:
        message = f"Table {table_key} missing from registry"
        raise RuntimeError(message)

    schema_name, table_name = table_key.split(".", maxsplit=1)
    info = con.execute(f"PRAGMA table_info({schema_name}.{table_name})").fetchall()
    if not info:
        message = f"Table {table_key} is missing"
        raise RuntimeError(message)

    names = [row[1] for row in info]
    if names != registry_cols:
        message = f"Column mismatch for {table_key}: {names} != {registry_cols}"
        raise RuntimeError(message)

    # You can keep your existing NOT NULL check here as well
```

This is essentially your current `PreparedStatements`, `prepared_statements_dynamic` and `ensure_schema` from `config.schemas.sql_builder`, just relocated and rewritten to use `duckdb.DuckDBPyConnection` directly.

> If you want to preserve the exact logic, literally copy-paste the bodies from `sql_builder.py` into this module.

### 3.2. Slim down `config.schemas.sql_builder` to metadata-only

In **`config/schemas/sql_builder.py`**:

* Remove the `DuckDBConnection` import from `codeintel.storage.gateway`.
* Remove the `PreparedStatements` class and the `prepared_statements_dynamic` / `ensure_schema` functions.
* Keep:

  * The column lists (`AST_NODES_COLUMNS`, `FUNCTION_METRICS_COLUMNS`, etc.).
  * Any utilities that produce **SQL text only** and don’t touch a connection.

So at the top:

```python
- from codeintel.storage.gateway import DuckDBConnection
```

is deleted.

And the bottom section with:

```python
class PreparedStatements: ...
def prepared_statements_dynamic(...): ...
def ensure_schema(...): ...
```

is removed.

This returns `config.schemas.sql_builder` to the “pure schema metadata and SQL literal” domain.

### 3.3. Update all call sites to use `storage.sql_helpers`

**1) `ingestion/common.py`**

Currently:

```python
from codeintel.config.schemas.sql_builder import prepared_statements_dynamic
from codeintel.ingestion.ingest_service import ensure_schema, ingest_via_macro
```

Change to:

```python
from codeintel.ingestion.ingest_service import ensure_schema, ingest_via_macro
from codeintel.storage.sql_helpers import PreparedStatements, prepared_statements_dynamic
```

Then adjust any type annotations that refer to `PreparedStatements` (if present) to import it from `storage.sql_helpers`.

**2) `graphs/validation.py`**

Currently:

```python
from codeintel.config.schemas.sql_builder import ensure_schema
```

Change to:

```python
from codeintel.storage.sql_helpers import ensure_schema
```

Call site:

```python
ensure_schema(gateway.con, "analytics.call_graph_edges_ext")
```

(or similar) remains unchanged except for the import.

After that:

* No runtime `config.schemas.*` module imports any `codeintel.storage.*`.
* All DuckDB-connection-aware helpers live under `storage.*`.
* `config.schemas.tables` remains the “logical schema” core.
* `storage.schemas` still imports `TABLE_SCHEMAS` (storage → config), but **config no longer imports storage** at runtime.

### 3.4. Keep type-checking-only imports safe

You already use `TYPE_CHECKING` in a couple of config modules:

* `config/builder.py` importing `codeintel.storage.rows.*` under `TYPE_CHECKING`.
* `config/schemas/ingestion_sql.py` and `registry_adapter.py` refer to `DuckDBConnection` only under `TYPE_CHECKING`.

That’s fine — these don’t affect runtime layering and keep static typing strong.

Just ensure these follow the pattern you already have:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.storage.gateway import DuckDBConnection
```

so nothing from `storage` gets imported at runtime by config modules.

---

## 4. Document the layering and add a lightweight static check

Once you’ve done the refactors above, your packages will effectively have this layering:

* **Core**

  * `config.primitives`, `config.schemas.*` (tables, ingestion_sql, registry_adapter)
  * `storage.schemas`, `storage.rows`, `storage.datasets`, `storage.registry_contracts`, `storage.sql_helpers`, `storage.module_index`
* **Domain**

  * `ingestion.*`, `graphs.*`, `analytics.*`
* **Application**

  * `pipeline.*`, `serving.*`, `cli.*`, `config.serving_models`, `config.models`, `config.builder`

### 4.1. Add a short architecture note

Create e.g. `docs/ARCHITECTURE_LAYERING.md`:

* Describe the three layers.

* Spell out **allowed imports**:

  * Core: may only import other core modules.
  * Domain: may import core + storage, but not `cli` or `serving`.
  * Application: may import anything.

* Give a couple of examples, including the ones you just fixed:

  * ✅ `graphs.engine_factory` importing `graphs.nx_backend`
  * ✅ `analytics.context` importing `storage.module_index.load_module_map`
  * ❌ `graphs.engine_factory` importing `cli.nx_backend` (now banned)

This doubles as a guide for agents.

### 4.2. Add a simple import-layer checker (optional but powerful)

Add a small script under `tools/check_layering.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import ast
import pathlib
from typing import Final

ROOT = pathlib.Path(__file__).resolve().parents[1]

LAYER_FOR_PREFIX: Final = {
    "codeintel.config.primitives": "core",
    "codeintel.config.schemas": "core",
    "codeintel.storage": "core",
    "codeintel.ingestion": "domain",
    "codeintel.graphs": "domain",
    "codeintel.analytics": "domain",
    "codeintel.pipeline": "app",
    "codeintel.serving": "app",
    "codeintel.cli": "app",
}

ALLOWED: Final = {
    "core": {"core"},
    "domain": {"core", "domain"},
    "app": {"core", "domain", "app"},
}


def classify_module(module: str) -> str | None:
    for prefix, layer in LAYER_FOR_PREFIX.items():
        if module == prefix or module.startswith(prefix + "."):
            return layer
    return None


def find_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("codeintel."):
                    modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("codeintel."):
                modules.add(node.module)
    return modules


def main() -> int:
    errors: list[str] = []
    for py in ROOT.rglob("*.py"):
        rel = py.relative_to(ROOT)
        module = "codeintel." + ".".join(rel.with_suffix("").parts)
        src_layer = classify_module(module)
        if src_layer is None:
            continue
        for imported in find_imports(py):
            dst_layer = classify_module(imported)
            if dst_layer is None:
                continue
            if dst_layer not in ALLOWED[src_layer]:
                errors.append(f"{module} ({src_layer}) -> {imported} ({dst_layer}) not allowed")
    if errors:
        for e in sorted(errors):
            print(e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Wire this into CI / pre-commit, maybe with a `make check-layering` target.

Initially you can run it in **“report-only”** mode (just print, don’t fail) until you’re confident everything’s cleaned up.

---

If you’d like, next step we can zoom in on one concrete change (e.g. the `storage.module_index` move, or the `storage.sql_helpers` split) and write an explicit before/after diff for the relevant files so you can paste it straight into your repo.

# storage.sql_helpers zoom in #

Got you — this is exactly the right place to be fussy, because this is where “who’s really the source of truth?” can quietly drift.

I walked through your *current* code in the new zips, especially:

* `config/schemas/sql_builder.py`
* `config/schemas/ingestion_sql.py`
* `config/schemas/registry_adapter.py`
* `storage/gateway.py`
* `storage/registry_contracts.py`
* `ingestion/common.py`
* `graphs/validation.py`

You already have a nice “DuckDB catalog → registry contracts → Python helpers” chain; we’ll keep that and just move the **connection-aware bits** from `config.schemas.sql_builder` into `storage.sql_helpers`, while still driving everything from the registry/catalog.

Below is a concrete, file-level plan.

---

## 1. Confirm the current registry-driven design (we keep this)

Right now the flow is:

* `storage.registry_contracts.build_registry_contracts(con)`
  → reads DuckDB’s **catalog** (tables, columns, nullability, types, etc.) and returns `Mapping[str, list[ColumnDef]]`.

* `config.schemas.registry_adapter.load_registry_columns(con)`
  → turns those contracts into `Mapping[table_key, list[column_name]]`.

* `config.schemas.ingestion_sql.LITERAL_COLUMNS`
  → a big dict of literal column lists per table, built from your `*_COLUMNS` constants.

* `config.schemas.ingestion_sql.verify_ingestion_columns(con)`
  → compares `LITERAL_COLUMNS` to `load_registry_columns(con)` and raises if they differ:

  ```python
  registry = load_registry_columns(con)
  for table_key, literal_cols in LITERAL_COLUMNS.items():
      registry_cols = registry.get(table_key)
      ...
      if registry_cols != literal_cols:
          raise RuntimeError(...)
  ```

* `config.schemas.sql_builder` currently contains:

  ```python
  from codeintel.config.schemas.ingestion_sql import verify_ingestion_columns
  from codeintel.config.schemas.registry_adapter import load_registry_columns
  from codeintel.storage.gateway import DuckDBConnection

  _INGESTION_COLUMNS_VERIFIED: list[bool] = [False]

  @dataclass(frozen=True)
  class PreparedStatements:
      insert_sql: str
      delete_sql: str | None = None

  def prepared_statements_dynamic(con: DuckDBConnection, table_key: str) -> PreparedStatements: ...
  def ensure_schema(con: DuckDBConnection, table_key: str) -> None: ...
  ```

  * `prepared_statements_dynamic` builds `INSERT` SQL **purely from `load_registry_columns(con)`**.
  * `ensure_schema` calls `verify_ingestion_columns(con)` once and then checks `PRAGMA table_info(schema.table)` matches the registry column list.

Call sites:

* `ingestion/common.py` imports `prepared_statements_dynamic` from `config.schemas.sql_builder`.
* `graphs/validation.py` imports `ensure_schema` from `config.schemas.sql_builder`.

So **DuckDB’s catalog is already the source of truth**; we just need to move the “things that take a connection” down into `storage.*` while preserving this registry chain.

---

## 2. Add `storage/sql_helpers.py` with the connection-aware helpers

Create a new module:

**`storage/sql_helpers.py`**

```python
# storage/sql_helpers.py
"""Connection-aware schema helpers driven by the DuckDB registry metadata."""

from __future__ import annotations

from dataclasses import dataclass

from duckdb import DuckDBPyConnection

from codeintel.config.schemas.ingestion_sql import verify_ingestion_columns
from codeintel.config.schemas.registry_adapter import load_registry_columns


# We only need to run verify_ingestion_columns once per process.
_INGESTION_COLUMNS_VERIFIED: list[bool] = [False]


@dataclass(frozen=True)
class PreparedStatements:
    """Prepared insert/delete SQL for a table (registry-driven)."""

    insert_sql: str
    delete_sql: str | None = None


def prepared_statements_dynamic(
    con: DuckDBPyConnection,
    table_key: str,
) -> PreparedStatements:
    """
    Return prepared SQL using registry-derived column order for a table.

    Parameters
    ----------
    con :
        Active DuckDB connection.
    table_key :
        Registry key (e.g., "core.ast_nodes", "analytics.function_metrics").

    Returns
    -------
    PreparedStatements
        Insert (and optional delete) SQL with column order sourced from the
        DuckDB registry via `build_registry_contracts`.

    Raises
    ------
    RuntimeError
        If the table is missing from the registry.
    """
    registry_cols = load_registry_columns(con).get(table_key)
    if registry_cols is None:
        message = f"Table {table_key} missing from registry"
        raise RuntimeError(message)

    cols_sql = ", ".join(registry_cols)
    placeholders = ", ".join("?" for _ in registry_cols)
    schema_name, table_name = table_key.split(".", maxsplit=1)
    table_sql = f'"{schema_name}"."{table_name}"'

    # Column order comes entirely from the registry → DuckDB catalog.
    insert_sql = (
        f"INSERT INTO {table_sql} ({cols_sql}) VALUES ({placeholders})"
    )

    # Today you don't use table-specific delete SQL here; if you later want it,
    # you can still derive it from registry metadata.
    return PreparedStatements(
        insert_sql=insert_sql,
        delete_sql=None,
    )


def ensure_schema(con: DuckDBPyConnection, table_key: str) -> None:
    """
    Validate that the live DuckDB table matches the registry definition.

    This:
    - Ensures that the literal column lists in `ingestion_sql` haven't drifted
      from the registry (once per process).
    - Ensures that the DuckDB table's columns & order match the registry.

    Parameters
    ----------
    con :
        Active DuckDB connection.
    table_key :
        Fully qualified table name (schema.table).

    Raises
    ------
    RuntimeError
        If the table is missing or deviates from the registry.
    """
    # First ensure registry ↔ literals alignment (once per process).
    if not _INGESTION_COLUMNS_VERIFIED[0]:
        verify_ingestion_columns(con)
        _INGESTION_COLUMNS_VERIFIED[0] = True

    registry_cols = load_registry_columns(con).get(table_key)
    if registry_cols is None:
        message = f"Table {table_key} missing from registry"
        raise RuntimeError(message)

    schema_name, table_name = table_key.split(".", maxsplit=1)
    # Let DuckDB tell us the live schema.
    info = con.execute(
        f"PRAGMA table_info({schema_name}.{table_name})"
    ).fetchall()
    if not info:
        message = f"Table {table_key} is missing"
        raise RuntimeError(message)

    names = [row[1] for row in info]
    expected_cols = registry_cols
    if names != expected_cols:
        message = (
            f"Column order mismatch for {table_key}: "
            f"db={names}, registry={expected_cols}"
        )
        raise RuntimeError(message)
```

Key points:

* We still call **`verify_ingestion_columns` and `load_registry_columns`**, so the direction of truth remains:

  **DuckDB catalog → registry_contracts → registry_adapter → ingestion_sql literals / SQL helpers.**

* We use DuckDB’s own `PRAGMA table_info` to confirm the table structure.

* No import from `storage.gateway` (so no extra cycles); we take a raw `DuckDBPyConnection`, and the call sites will pass `gateway.con`.

If you like, you can also add:

```python
__all__ = ["PreparedStatements", "prepared_statements_dynamic", "ensure_schema"]
```

at the bottom.

---

## 3. Slim `config.schemas.sql_builder` down to metadata-only

Now we remove connection-aware logic from `config/schemas/sql_builder.py` so that it only holds:

* Column lists (`*_COLUMNS`).
* Literal SQL strings (`*_INSERT`, `*_DELETE`, etc.).
* A metadata-oriented `__all__`.

### 3.1. Remove connection-dependent imports and globals

At the top of `config/schemas/sql_builder.py` you currently have:

```python
from dataclasses import dataclass

from codeintel.config.schemas.ingestion_sql import verify_ingestion_columns
from codeintel.config.schemas.registry_adapter import load_registry_columns
from codeintel.storage.gateway import DuckDBConnection

_INGESTION_COLUMNS_VERIFIED: list[bool] = [False]
```

Change to:

```python
# No dataclass or connection-aware imports needed anymore.
# Keep only what you need for column lists / SQL literals.
```

So:

* Drop `dataclass` import.
* Drop imports of `verify_ingestion_columns`, `load_registry_columns`, `DuckDBConnection`.
* Drop `_INGESTION_COLUMNS_VERIFIED`.

### 3.2. Remove `PreparedStatements`, `prepared_statements_dynamic`, `ensure_schema`

Delete this block entirely:

```python
@dataclass(frozen=True)
class PreparedStatements:
    """Prepared insert/delete SQL for a table."""

    insert_sql: str
    delete_sql: str | None = None


def prepared_statements_dynamic(con: DuckDBConnection, table_key: str) -> PreparedStatements:
    ...
    return PreparedStatements(
        insert_sql=insert_sql,
        delete_sql=None,
    )


def ensure_schema(con: DuckDBConnection, table_key: str) -> None:
    ...
```

Those are now in `storage.sql_helpers`.

### 3.3. Update `__all__` in `sql_builder.py`

At the bottom of `sql_builder.py`, `__all__` currently contains:

```python
    "TYPEDNESS_COLUMNS",
    "TYPEDNESS_DELETE",
    "TYPEDNESS_INSERT",
    "PreparedStatements",
    "ensure_schema",
    "prepared_statements_dynamic",
]
```

Remove the last three:

```python
    "TYPEDNESS_COLUMNS",
    "TYPEDNESS_DELETE",
    "TYPEDNESS_INSERT",
]
```

After this, `config.schemas.sql_builder` is purely about **static metadata and static SQL literals**, and all connection-aware behavior is in `storage.sql_helpers`.

---

## 4. Update call sites to use `storage.sql_helpers`

Two places currently use the moved helpers.

### 4.1. `ingestion/common.py` — prepared statements

At the top:

```python
from codeintel.config.schemas.sql_builder import prepared_statements_dynamic
from codeintel.ingestion.ingest_service import ensure_schema, ingest_via_macro
```

Change to:

```python
from codeintel.ingestion.ingest_service import ensure_schema, ingest_via_macro
from codeintel.storage.sql_helpers import PreparedStatements, prepared_statements_dynamic
```

(You only need `PreparedStatements` import if you reference its type explicitly; you can omit it if you only treat the result as a duck-typed object.)

The call site doesn’t change:

```python
stmts = prepared_statements_dynamic(con, table_key)
con.execute(stmts.insert_sql, row)
```

Semantics remain:

* `prepared_statements_dynamic` now lives in `storage.sql_helpers`, but still:

  * Uses `load_registry_columns(con)`, which is built from DuckDB’s catalog e.g. via `build_registry_contracts`.
  * So row insertion order is fully dictated by the **live database schema**.

### 4.2. `graphs/validation.py` — ensure schema for `analytics.graph_validation`

At the top:

```python
from codeintel.config.schemas.sql_builder import ensure_schema
```

Change to:

```python
from codeintel.storage.sql_helpers import ensure_schema
```

Call site:

```python
con = gateway.con
ensure_schema(con, "analytics.graph_validation")
con.execute(
    "DELETE FROM analytics.graph_validation WHERE repo = ? AND commit = ?",
    [repo, commit],
)
```

behaves exactly as before, but now goes through the storage layer:

* It still calls `verify_ingestion_columns(con)` once.
* It still uses `load_registry_columns(con)` to discover expected columns.
* It still uses `PRAGMA table_info` to compare DB columns to registry columns.

---

## 5. How this keeps the “schema flow” DB-driven

After this refactor, the picture looks like:

1. **DuckDB catalog**
   ↳ `storage.registry_contracts.build_registry_contracts(con)`
   ↳ `config.schemas.registry_adapter.load_registry_columns(con)`.

2. **Ingestion literals sanity**
   ↳ `config.schemas.ingestion_sql.verify_ingestion_columns(con)`
   checks `LITERAL_COLUMNS` vs `load_registry_columns(con)`.

3. **Connection-aware helpers** (now in `storage.sql_helpers`)

   * `prepared_statements_dynamic(con, table_key)` builds `INSERT` SQL directly from `load_registry_columns(con)` (registry → DB).
   * `ensure_schema(con, table_key)`:

     * First ensures literals haven’t drifted: `verify_ingestion_columns(con)`.
     * Then ensures DB matches registry: compares `PRAGMA table_info` to `load_registry_columns(con)[table_key]`.

4. **Usage**

   * Ingestion uses `prepared_statements_dynamic` for macro-driven inserts.
   * Graph validation uses `ensure_schema` before writing to `analytics.graph_validation`.

So:

* **DuckDB** is still the ultimate source of truth.
* `config.schemas.*` remains the place where column *names* and literal SQL are declared, but they’re kept in lock-step with the DB via the registry adapter + verification.
* All “I have a connection, please validate / build SQL” logic is centralized in the **storage layer**, which is exactly where you’d expect it in the layered architecture.

---

## 6. Optional: a quick regression test idea

Once you make these changes, a small integration test will help lock it in:

```python
def test_registry_and_helpers_agree(gateway: StorageGateway) -> None:
    con = gateway.con

    # 1. All ingestion literals must match registry.
    from codeintel.config.schemas.ingestion_sql import verify_ingestion_columns
    verify_ingestion_columns(con)

    # 2. prepared_statements_dynamic should work for a known table.
    from codeintel.storage.sql_helpers import prepared_statements_dynamic, ensure_schema

    table_key = "core.ast_nodes"
    ensure_schema(con, table_key)
    stmts = prepared_statements_dynamic(con, table_key)
    assert "INSERT INTO" in stmts.insert_sql
    assert "core"."ast_nodes" in stmts.insert_sql or "core.ast_nodes" in stmts.insert_sql
```

You can add more tables (especially any you touch in ingestion) to this test over time.

---

