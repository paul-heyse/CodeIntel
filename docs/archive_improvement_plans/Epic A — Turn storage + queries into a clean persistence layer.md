#Epic A — Turn storage + queries into a clean persistence layer#

Cool, let’s turn Epic A into something your implementation agent can actually ship.

I’ll structure this as:

1. **Target end-state** (what the storage layer looks like when we’re done)
2. **New module layout**
3. **Step-by-step refactor plan**
4. **Mapping of existing queries → repository methods**
5. **Test & migration strategy**

I’ll reference concrete modules from your repo (e.g. `codeintel.storage.gateway`, `serving.mcp.query_service`) so it’s easy to wire.

---

## 1. Target end-state (what we’re aiming for)

Conceptually, the “end state” of Epic A is:

* `StorageGateway` is a **small composition root**:

  * opens DuckDB
  * exposes table groups: `.core`, `.graph`, `.docs`, `.analytics`
  * exposes a **dataset registry** that knows dataset names → table keys
* `views` are **split by domain** with a small **view registry**:

  * each `docs.v_*` view lives in a small module, registered in one place
  * `DOCS_VIEWS` and `create_all_views()` still exist and are imported from `codeintel.storage.views`
* All **high-level SQL** lives in **repositories**, not in serving/analytics:

  * `serving.mcp.query_service.DuckDBQueryService` calls repositories
  * `storage.data_models` becomes (or is wrapped as) a `DataModelRepository`
  * analytics modules that still embed SELECTs gradually move over to repos

So your layers become:

* **storage schema / views** → `config.schemas.*`, `storage.schemas`, `storage.views.*`
* **persistence & datasets** → `storage.gateway`, `storage.config`, `storage.datasets`, `storage.metadata_bootstrap`
* **domain data access** → `storage.repositories.*`
* **application logic** → `serving.*`, `analytics.*`, `pipeline.*` calling the repositories

---

## 2. New module layout

### Storage package (before vs after)

**Today (simplified):**

```text
codeintel/storage/storage/
    __init__.py
    gateway.py
    views.py
    metadata_bootstrap.py
    datasets.py
    rows.py
    data_models.py
    registry_contracts.py
    module_index.py
    sql_helpers.py
    schemas.py
    docs.v_function_summary
    docs.v_call_graph_enriched
```

**Target layout:**

```text
codeintel/storage/storage/
    __init__.py

    # 2A. Config & gateway
    config.py            # StorageConfig and ctor helpers
    gateway.py           # StorageGateway Protocol + _DuckDBGateway + table groups
    ingest_helpers.py    # _macro_insert_rows and related helpers
    registry_helpers.py  # dataset registry glue & “rows-only” helpers

    # 2B. Views (package instead of one 40k-char file)
    views/
        __init__.py          # DOCS_VIEWS, create_all_views(), registry
        function_views.py    # v_function_summary, v_function_architecture, v_function_history*
        module_views.py      # v_module_architecture, v_module_history_timeseries, v_module_profile
        test_views.py        # v_test_architecture, v_behavioral_classification_input, v_behavioral_coverage
        subsystem_views.py   # v_subsystem_summary, v_subsystem_risk, v_subsystem_agreement*
        graph_views.py       # v_call_graph_enriched, graph_stats-ish docs views
        ide_views.py         # v_ide_hints and any IDE-facing docs views
        data_model_views.py  # v_data_models_normalized, config/data-flow docs views

    # 2C. Existing metadata pieces stay, but depend on new helpers
    datasets.py             # Dataset + DatasetRegistry (current)
    metadata_bootstrap.py   # uses DOCS_VIEWS and registry_helpers
    rows.py                 # TypedDict insert models (unchanged)
    data_models.py          # raw data model accessors (to be wrapped by repo)
    module_index.py
    sql_helpers.py
    schemas.py
    registry_contracts.py

    # 2D. New repository layer
    repositories/
        __init__.py
        base.py             # BaseRepository / RowDict helpers
        functions.py        # FunctionRepository
        modules.py          # ModuleRepository (file+module views)
        subsystems.py       # SubsystemRepository
        tests.py            # TestRepository
        datasets.py         # DatasetReadRepository (wraps metadata.dataset_rows)
        data_models.py      # DataModelRepository (wraps storage.data_models)
        graphs.py           # GraphRepository (call_graph_enriched + IDE hints)
```

External imports continue to work:

* `from codeintel.storage.views import DOCS_VIEWS, create_all_views`
* `from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway, build_snapshot_gateway_resolver`

Because `gateway.py` and `views/__init__.py` re-export the right symbols.

---

## 3. Step-by-step implementation plan

### Step 0 — Safety and scaffolding

1. Create a **feature branch** for the epic (`feature/storage-repos`).
2. Add a **smoke test** if you don’t already have one that:

   * builds a DuckDB snapshot from a small test repo,
   * runs a few representative `DuckDBQueryService` calls:

     * `get_function_summary`
     * `list_high_risk_functions`
     * `get_module_architecture`
     * `list_subsystems`
     * `read_dataset_rows("analytics.function_profile")`
3. That becomes your “behaviour hasn’t changed” guard as you refactor.

---

### Step 1 — Tighten `StorageGateway` + extract config & helpers

#### 1.1 Create `storage/config.py`

Move the `StorageConfig` class from `storage.gateway` into a new module:

```python
# codeintel/storage/storage/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class StorageConfig:
    db_path: Path
    read_only: bool = False
    apply_schema: bool = False
    ensure_views: bool = False
    validate_schema: bool = False
    attach_history: bool = False
    history_db_path: Path | None = None
    repo: str | None = None
    commit: str | None = None

    @classmethod
    def for_ingest(cls, db_path: Path, *, repo: str, commit: str) -> StorageConfig: ...
    @classmethod
    def for_analytics(cls, db_path: Path, *, repo: str, commit: str) -> StorageConfig: ...
    @classmethod
    def for_readonly(cls, db_path: Path) -> StorageConfig: ...
```

* **Copy** the implementations of `for_ingest`, `for_analytics` (or `for_export`), `for_readonly` exactly from `gateway.py`.

* In `gateway.py`:

  ```python
  # gateway.py
  from codeintel.storage.config import StorageConfig  # and re-export

  __all__ = ["StorageConfig", "StorageGateway", "open_gateway", "build_snapshot_gateway_resolver"]
  ```

* Update **internal** imports that use `StorageConfig` inside `gateway.py` to import from `.config` (even though you re-export).

Call-sites (`cli.main`, `pipeline.orchestration.prefect_flow`, `serving.http.fastapi`) can keep importing `StorageConfig` from `codeintel.storage.gateway` for now.

#### 1.2 Move macro insert logic to `ingest_helpers.py`

In `gateway.py` you currently have `_macro_insert_rows(con, table_key, rows)` with the “pad to schema width and INSERT via temp table” logic.

Create:

```python
# codeintel/storage/storage/ingest_helpers.py
from __future__ import annotations

from collections.abc import Iterable, Sequence

from duckdb import DuckDBPyConnection

def macro_insert_rows(
    con: DuckDBPyConnection,
    table_key: str,
    rows: Iterable[Sequence[object]],
) -> None:
    """Exactly the body of the current _macro_insert_rows."""
    ...
```

Then in `gateway.py`:

* Remove `_macro_insert_rows`.

* Import it:

  ```python
  from codeintel.storage.ingest_helpers import macro_insert_rows
  ```

* Replace calls:

  ```python
  # before
  _macro_insert_rows(self.con, "analytics.graph_stats", rows)

  # after
  macro_insert_rows(self.con, "analytics.graph_stats", rows)
  ```

This keeps `CoreTables` / `AnalyticsTables` methods small and moves ingest-specific logic out of the gateway.

#### 1.3 Extract dataset registry glue to `registry_helpers.py`

You currently have **two** dataset registries:

* `storage.datasets.DatasetRegistry` – rich metadata (schema, row bindings, filenames).
* `storage.gateway.DatasetRegistry` – thin mapping `name → table_key` + views/tables lists and filename maps.

In `gateway.py` there is `build_dataset_registry(con, include_views=True)` that:

* calls `load_dataset_registry(con)` (from `storage.datasets`)
* flattens that into the simpler mapping used at the gateway.

Move that function and the “thin” `DatasetRegistry` dataclass into a new module:

```python
# codeintel/storage/storage/registry_helpers.py
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from duckdb import DuckDBPyConnection

from codeintel.storage.datasets import Dataset, load_dataset_registry as _load_dataset_registry

@dataclass(frozen=True)
class DatasetRegistry:
    mapping: Mapping[str, str]
    tables: tuple[str, ...]
    views: tuple[str, ...]
    meta: Mapping[str, Dataset] | None = None
    jsonl_mapping: Mapping[str, str] | None = None
    parquet_mapping: Mapping[str, str] | None = None

    @property
    def all_datasets(self) -> tuple[str, ...]: ...
    def table_for_name(self, name: str) -> str: ...
    # (copy exactly from current gateway.DatasetRegistry)

def build_dataset_registry(con: DuckDBPyConnection, *, include_views: bool = True) -> DatasetRegistry:
    """Copy body from current gateway.build_dataset_registry."""
```

Then in `gateway.py`:

* Delete the `DatasetRegistry` definition and `build_dataset_registry` implementation.

* Import them:

  ```python
  from codeintel.storage.registry_helpers import DatasetRegistry, build_dataset_registry
  ```

* Keep `StorageGateway` Protocol unchanged but now referencing `registry_helpers.DatasetRegistry`.

This makes `gateway.py` mostly about:

* connecting to DuckDB (`_connect`)
* calling `bootstrap_metadata_datasets`
* building `DatasetRegistry`
* instantiating `_DuckDBGateway`.

#### 1.4 Keep `StorageGateway` and `_DuckDBGateway` as the composition root

You can now trim `gateway.py` to:

* `StorageGateway` **Protocol**: has `config`, `datasets`, `con`, `core`, `graph`, `docs`, `analytics`.

* `_DuckDBGateway` dataclass implementing `StorageGateway` using:

  * `StorageConfig` (from `config.py`)
  * `DatasetRegistry` (from `registry_helpers.py`)
  * `CoreTables`, `GraphTables`, `DocsViews`, `AnalyticsTables` (unchanged table groups).

* `open_gateway(config: StorageConfig) -> StorageGateway`:

  * `_connect(config)` (existing logic, unchanged)
  * `bootstrap_metadata_datasets` (if not `read_only`)
  * `build_dataset_registry(con)`
  * return `_DuckDBGateway(config=config, datasets=datasets, con=con)`

* `build_snapshot_gateway_resolver(...)` remains, but the body is simpler to read.

At this point, nothing *external* should have changed – just better separation inside storage.

---

### Step 2 — Split `storage.views` into a package with a view registry

#### 2.1 Turn `views.py` into a package

* Rename `codeintel/storage/storage/views.py` → `codeintel/storage/storage/views/__init__.py`.
* Keep the existing `DOCS_VIEWS` and `create_all_views(con)` temporarily as-is so nothing breaks.

Then create new modules:

```text
storage/views/function_views.py
storage/views/module_views.py
storage/views/test_views.py
storage/views/subsystem_views.py
storage/views/graph_views.py
storage/views/ide_views.py
storage/views/data_model_views.py
```

#### 2.2 Extract SQL into per-domain creator functions

In each new module, move the relevant chunks from the giant `create_all_views`:

Example for `function_views.py`:

```python
# storage/views/function_views.py
from __future__ import annotations

from duckdb import DuckDBPyConnection

FUNCTION_VIEW_NAMES: tuple[str, ...] = (
    "docs.v_function_summary",
    "docs.v_function_architecture",
    "docs.v_function_history",
    "docs.v_function_history_timeseries",
)

def create_function_views(con: DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_function_summary AS
        SELECT
            ...
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_function_architecture AS
        SELECT
            ...
        """
    )
    ...
```

Do the same for:

* `module_views.create_module_views()` — module architecture + module history
* `test_views.create_test_views()` — test architecture + behavioral coverage input
* `subsystem_views.create_subsystem_views()` — subsystem summary/risk
* `graph_views.create_graph_views()` — v_call_graph_enriched, graph stats-like views
* `ide_views.create_ide_views()` — v_ide_hints
* `data_model_views.create_data_model_views()` — v_data_models_normalized and related config/data-flow docs views

#### 2.3 Implement a simple view registry in `views/__init__.py`

Replace the old monolithic `create_all_views` with a registry-driven version:

```python
# storage/views/__init__.py
from __future__ import annotations

from collections.abc import Callable, Iterable
from duckdb import DuckDBPyConnection

from .function_views import FUNCTION_VIEW_NAMES, create_function_views
from .module_views import MODULE_VIEW_NAMES, create_module_views
from .test_views import TEST_VIEW_NAMES, create_test_views
from .subsystem_views import SUBSYSTEM_VIEW_NAMES, create_subsystem_views
from .graph_views import GRAPH_VIEW_NAMES, create_graph_views
from .ide_views import IDE_VIEW_NAMES, create_ide_views
from .data_model_views import DATA_MODEL_VIEW_NAMES, create_data_model_views

DOCS_VIEWS: tuple[str, ...] = (
    *FUNCTION_VIEW_NAMES,
    *MODULE_VIEW_NAMES,
    *TEST_VIEW_NAMES,
    *SUBSYSTEM_VIEW_NAMES,
    *GRAPH_VIEW_NAMES,
    *IDE_VIEW_NAMES,
    *DATA_MODEL_VIEW_NAMES,
)

_VIEW_CREATORS: tuple[Callable[[DuckDBPyConnection], None], ...] = (
    create_function_views,
    create_module_views,
    create_test_views,
    create_subsystem_views,
    create_graph_views,
    create_ide_views,
    create_data_model_views,
)

def create_all_views(con: DuckDBPyConnection) -> None:
    """Create or replace all docs.* views."""
    for create in _VIEW_CREATORS:
        create(con)
```

All current importers (`metadata_bootstrap`, `serving.http.datasets`, `serving.services.wiring`, `gateway`) still do:

```python
from codeintel.storage.views import DOCS_VIEWS, create_all_views
```

…and see identical behaviour.

---

### Step 3 — Introduce the repository layer

#### 3.1 Base repository + fetch helpers

Create `storage/repositories/base.py`:

```python
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from codeintel.storage.gateway import DuckDBConnection, StorageGateway

RowDict = dict[str, Any]

def fetch_one_dict(con: DuckDBConnection, sql: str, params: Sequence[object]) -> RowDict | None:
    result = con.execute(sql, list(params))
    row = result.fetchone()
    if row is None:
        return None
    cols = [desc[0] for desc in result.description]
    return {col: row[idx] for idx, col in enumerate(cols)}

def fetch_all_dicts(con: DuckDBConnection, sql: str, params: Sequence[object]) -> list[RowDict]:
    result = con.execute(sql, list(params))
    rows = result.fetchall()
    cols = [desc[0] for desc in result.description]
    return [{col: row[idx] for idx, col in enumerate(cols)} for row in rows]

@dataclass(frozen=True)
class BaseRepository:
    gateway: StorageGateway
    repo: str
    commit: str

    @property
    def con(self) -> DuckDBConnection:  # convenience
        return self.gateway.con
```

**Note:** this is basically `_fetch_one_dict` / `_fetch_all_dicts` from `serving.mcp.query_service` but in a reusable storage layer.

Later we’ll replace the query_service copies to import these.

#### 3.2 `FunctionRepository`

Create `storage/repositories/functions.py`:

```python
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .base import BaseRepository, RowDict, fetch_all_dicts, fetch_one_dict

@dataclass(frozen=True)
class FunctionRepository(BaseRepository):
    """Read functions, risk, tests, and callgraph data from docs/analytics views."""

    def resolve_function_goid(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> int | None:
        # mostly a move of DuckDBQueryService._resolve_function_goid SQL
        ...

    def get_function_summary_by_goid(self, goid_h128: int) -> RowDict | None:
        sql = """
            SELECT *
            FROM docs.v_function_summary
            WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
            LIMIT 1
        """
        return fetch_one_dict(self.con, sql, [self.repo, self.commit, goid_h128])

    def list_function_summaries_for_file(self, rel_path: str) -> list[RowDict]:
        sql = """
            SELECT *
            FROM docs.v_function_summary
            WHERE rel_path = ?
              AND repo = ?
              AND commit = ?
            ORDER BY qualname
        """
        return fetch_all_dicts(self.con, sql, [rel_path, self.repo, self.commit])

    def list_high_risk_functions(
        self,
        *,
        min_risk: float,
        limit: int,
        tested_only: bool,
    ) -> list[RowDict]:
        base_sql = """
            SELECT
                function_goid_h128,
                urn,
                rel_path,
                qualname,
                risk_score,
                risk_level,
                coverage_ratio,
                tested,
                complexity_bucket,
                typedness_bucket,
                hotspot_score
            FROM analytics.goid_risk_factors
            WHERE repo = ? AND commit = ? AND risk_score >= ?
        """
        if tested_only:
            base_sql += " AND tested = TRUE"
        base_sql += " ORDER BY risk_score DESC LIMIT ?"
        return fetch_all_dicts(self.con, base_sql, [self.repo, self.commit, min_risk, limit])

    def get_function_profile(self, goid_h128: int) -> RowDict | None:
        # SELECT * FROM docs.v_function_profile WHERE repo/commit/goid_h128…
        ...

    def get_function_architecture(self, goid_h128: int) -> RowDict | None:
        # SELECT * FROM docs.v_function_architecture WHERE repo/commit/goid_h128…
        ...

    def get_tests_for_function(self, goid_h128: int) -> list[RowDict]:
        # The existing SQL joining test_* tables/views
        ...
```

This captures **all SQL currently used** in:

* `_resolve_function_goid`
* `get_function_summary`
* `list_high_risk_functions`
* `get_function_profile`
* `get_function_architecture`
* `get_tests_for_function`

…moved out of `serving.mcp.query_service`.

#### 3.3 `ModuleRepository` and `FileRepository` (or just `ModulesRepository`)

Create `storage/repositories/modules.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from .base import BaseRepository, RowDict, fetch_one_dict, fetch_all_dicts

@dataclass(frozen=True)
class ModuleRepository(BaseRepository):
    def get_file_summary(self, rel_path: str) -> RowDict | None:
        sql = """
            SELECT *
            FROM docs.v_file_summary
            WHERE repo = ? AND commit = ? AND rel_path = ?
            LIMIT 1
        """
        return fetch_one_dict(self.con, sql, [self.repo, self.commit, rel_path])

    def get_file_profile(self, rel_path: str) -> RowDict | None:
        # docs.v_file_profile
        ...

    def get_module_profile(self, module: str) -> RowDict | None:
        # docs.v_module_profile
        ...

    def get_file_hints(self, rel_path: str) -> list[RowDict]:
        sql = """
            SELECT *
            FROM docs.v_ide_hints
            WHERE repo = ?
              AND commit = ?
              AND rel_path = ?
        """
        return fetch_all_dicts(self.con, sql, [self.repo, self.commit, rel_path])

    def get_module_architecture(self, module: str) -> RowDict | None:
        # docs.v_module_architecture
        ...
```

This captures SQL used by:

* `get_file_summary`
* `get_file_profile`
* `get_module_profile`
* `get_file_hints`
* `get_module_architecture`

#### 3.4 `SubsystemRepository`

Create `storage/repositories/subsystems.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from .base import BaseRepository, RowDict, fetch_all_dicts, fetch_one_dict

@dataclass(frozen=True)
class SubsystemRepository(BaseRepository):
    def list_subsystems(self) -> list[RowDict]:
        sql = """
            SELECT *
            FROM docs.v_subsystem_summary
            WHERE repo = ? AND commit = ?
            ORDER BY subsystem_id
        """
        return fetch_all_dicts(self.con, sql, [self.repo, self.commit])

    def get_subsystem_summary(self, subsystem_id: int) -> RowDict | None:
        sql = """
            SELECT *
            FROM docs.v_subsystem_summary
            WHERE repo = ?
              AND commit = ?
              AND subsystem_id = ?
            LIMIT 1
        """
        return fetch_one_dict(self.con, sql, [self.repo, self.commit, subsystem_id])

    def search_subsystems(self, query: str, limit: int) -> list[RowDict]:
        # same WHERE / ORDER logic as DuckDBQueryService.search_subsystems
        ...

    def list_subsystem_modules(self, subsystem_id: int) -> list[RowDict]:
        # SELECT * FROM docs.v_subsystem_modules WHERE subsystem_id…
        ...
```

This captures:

* `list_subsystems`
* `get_subsystem_modules`
* `search_subsystems`
* `summarize_subsystem` (summary + module list).

#### 3.5 `TestRepository`

Create `storage/repositories/tests.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from .base import BaseRepository, RowDict, fetch_all_dicts

@dataclass(frozen=True)
class TestRepository(BaseRepository):
    def get_tests_for_function(self, goid_h128: int) -> list[RowDict]:
        # same SQL currently in DuckDBQueryService.get_tests_for_function
        ...

    # later: helpers for behavioral coverage views, test profiles, etc.
```

For now this may be a very thin wrapper; that’s fine – it lets you expand later (e.g. to support additional query surfaces).

#### 3.6 `GraphRepository`

Create `storage/repositories/graphs.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from .base import BaseRepository, RowDict, fetch_all_dicts

@dataclass(frozen=True)
class GraphRepository(BaseRepository):
    def get_callgraph_neighbors(
        self,
        caller_goid_h128: int,
        limit: int,
    ) -> list[RowDict]:
        sql = """
            SELECT *
            FROM docs.v_call_graph_enriched
            WHERE caller_goid_h128 = ?
              AND caller_repo = ?
              AND caller_commit = ?
            ORDER BY callee_qualname
            LIMIT ?
        """
        return fetch_all_dicts(self.con, sql, [caller_goid_h128, self.repo, self.commit, limit])
```

For this epic, keep the Nx graph usage (`networkx`, `NxGraphEngine`) in `DuckDBQueryService` – GraphRepository is just a relational access layer. When you do Epic B (graph runtime consolidation) you can give it richer methods.

#### 3.7 Dataset + data model repositories

* `storage/repositories/datasets.py`:

  ```python
  from __future__ import annotations

  from dataclasses import dataclass

  from .base import BaseRepository, RowDict, fetch_all_dicts

  @dataclass(frozen=True)
  class DatasetReadRepository(BaseRepository):
      def read_dataset_rows(self, table_key: str, limit: int, offset: int) -> list[RowDict]:
          sql = "SELECT * FROM metadata.dataset_rows(?, ?, ?)"
          return fetch_all_dicts(self.con, sql, [table_key, limit, offset])
  ```

  This matches what `DuckDBQueryService.read_dataset_rows` does today.

* `storage/repositories/data_models.py`:

  Wrap the functions in `storage.data_models` rather than rewriting them:

  ```python
  from __future__ import annotations

  from dataclasses import dataclass
  from collections.abc import Sequence

  from codeintel.storage.data_models import (
      DataModelRow,
      DataModelFieldRow,
      DataModelRelationshipRow,
      NormalizedDataModel,
      fetch_models,
      fetch_models_normalized,
      fetch_fields,
      fetch_relationships,
  )
  from codeintel.storage.gateway import StorageGateway

  @dataclass(frozen=True)
  class DataModelRepository:
      gateway: StorageGateway

      def models(self, repo: str, commit: str) -> list[DataModelRow]:
          return fetch_models(self.gateway, repo, commit)

      def models_normalized(self, repo: str, commit: str) -> list[NormalizedDataModel]:
          return fetch_models_normalized(self.gateway, repo, commit)

      def fields(self, repo: str, commit: str, model_ids: Sequence[str] | None = None) -> list[DataModelFieldRow]:
          return fetch_fields(self.gateway, repo, commit, model_ids=model_ids)

      def relationships(
          self, repo: str, commit: str, model_ids: Sequence[str] | None = None
      ) -> list[DataModelRelationshipRow]:
          return fetch_relationships(self.gateway, repo, commit, model_ids=model_ids)
  ```

This keeps your existing normalization logic intact but gives a consistent “repository” story.

---

### Step 4 — Refactor `DuckDBQueryService` to use repositories

Now wire the new repositories into `serving.mcp.query_service.DuckDBQueryService`.

#### 4.1 Add repository fields

At the top of `DuckDBQueryService`:

```python
from codeintel.storage.repositories.functions import FunctionRepository
from codeintel.storage.repositories.modules import ModuleRepository
from codeintel.storage.repositories.subsystems import SubsystemRepository
from codeintel.storage.repositories.tests import TestRepository
from codeintel.storage.repositories.datasets import DatasetReadRepository
from codeintel.storage.repositories.graphs import GraphRepository

@dataclass
class DuckDBQueryService:
    gateway: StorageGateway
    repo: str
    commit: str
    limits: BackendLimits
    engine: NxGraphEngine | None = None
    _engine: NxGraphEngine | None = field(default=None, init=False, repr=False)

    # new, lazily initialized repositories
    _functions: FunctionRepository | None = field(default=None, init=False, repr=False)
    _modules: ModuleRepository | None = field(default=None, init=False, repr=False)
    _subsystems: SubsystemRepository | None = field(default=None, init=False, repr=False)
    _tests: TestRepository | None = field(default=None, init=False, repr=False)
    _datasets: DatasetReadRepository | None = field(default=None, init=False, repr=False)
    _graphs: GraphRepository | None = field(default=None, init=False, repr=False)

    @property
    def functions(self) -> FunctionRepository:
        if self._functions is None:
            self._functions = FunctionRepository(self.gateway, self.repo, self.commit)
        return self._functions

    # same for modules/subsystems/tests/datasets/graphs
```

#### 4.2 Replace `_fetch_*` helpers with imports

* Delete the local `_fetch_one_dict` / `_fetch_all_dicts`.
* Replace uses with calls to repository layer (they already wrap these helpers).

If you still need `_fetch_*` in this module for some graph-adjacent queries you haven’t moved yet, import them from `storage.repositories.base`.

#### 4.3 Re-implement methods in terms of repos

Examples:

* `_resolve_function_goid`:

  ```python
  def _resolve_function_goid(...):
      return self.functions.resolve_function_goid(
          urn=urn, goid_h128=goid_h128, rel_path=rel_path, qualname=qualname
      )
  ```

* `get_function_summary`:

  ```python
  def get_function_summary(...):
      goid = self._resolve_function_goid(...)
      if goid is None:
          # build meta, return FunctionSummaryResponse(found=False, ...)
      row = self.functions.get_function_summary_by_goid(goid)
      if row is None:
          # not_found meta
      summary = ViewRow.model_validate(row)
      return FunctionSummaryResponse(found=True, summary=summary, meta=meta)
  ```

* `list_high_risk_functions`:

  ```python
  rows = self.functions.list_high_risk_functions(
      min_risk=min_risk,
      limit=limit_clamp.applied,
      tested_only=tested_only,
  )
  return HighRiskFunctionsResponse(
      rows=[ViewRow.model_validate(r) for r in rows],
      meta=meta,
  )
  ```

* `get_callgraph_neighbors`:

  ```python
  rows = self.graphs.get_callgraph_neighbors(
      caller_goid_h128=goid,
      limit=limit_clamp.applied,
  )
  return CallGraphNeighborsResponse(
      neighbors=[ViewRow.model_validate(r) for r in rows],
      meta=meta,
  )
  ```

* `get_file_summary`, `get_file_profile`, `get_module_profile`, `get_file_hints`, `get_module_architecture` delegate to `self.modules`.

* `list_subsystems`, `get_module_subsystems`, `get_subsystem_modules`, `search_subsystems`, `summarize_subsystem` delegate to `self.subsystems`.

* `get_tests_for_function` delegates to `self.tests`.

* `read_dataset_rows`:

  ```python
  table = self.gateway.datasets.mapping.get(dataset_name)
  if table is None:
      raise errors.invalid_argument(...)

  rows = self.datasets.read_dataset_rows(
      table_key=table,
      limit=limit_clamp.applied,
      offset=offset_clamp.applied,
  )
  return DatasetRowsResponse(
      dataset=dataset_name,
      limit=limit_clamp.applied,
      offset=offset_clamp.applied,
      rows=[ViewRow.model_validate(r) for r in rows],
      meta=meta,
  )
  ```

The **only SQL left** in `DuckDBQueryService` after this step should be the graph-adjacent things you deliberately leave for Epic B (if any).

---

### Step 5 — Start migrating analytics modules (optional within this epic)

This can be incremental and opportunistic:

* Search for `execute("""SELECT` and `FROM docs.` / `FROM analytics.` in `analytics/*`.
* For each cluster of queries:

  * Decide which repository they logically belong to:

    * function/subsystem/test/module/graph/data_model.
  * Move the SQL into a new repo method.
  * Call that repo from the analytics module.

Example: in `analytics/tests/profiles.py`, queries against:

* `analytics.test_catalog`
* `analytics.test_coverage_edges`
* `docs.v_function_summary`

…should become methods on `TestRepository` and `FunctionRepository`, with `profiles.py` working purely in terms of Python objects.

Since this is a bigger effort, you can treat it as “Phase 2” of the epic or defer to a separate epic, but the patterns are now in place.

---

## 4. Query mapping cheat-sheet (today → repository)

This is the “where did that SQL go” table for your future self / LLM agent:

| Current location / method                                  | New repository method                                 |
| ---------------------------------------------------------- | ----------------------------------------------------- |
| `DuckDBQueryService._resolve_function_goid`                | `FunctionRepository.resolve_function_goid`            |
| `get_function_summary` (docs.v_function_summary)           | `FunctionRepository.get_function_summary_by_goid`     |
| `list_high_risk_functions` (analytics.goid_risk_factors)   | `FunctionRepository.list_high_risk_functions`         |
| `get_callgraph_neighbors` (docs.v_call_graph_enriched)     | `GraphRepository.get_callgraph_neighbors`             |
| `get_tests_for_function` (test tables/views)               | `TestRepository.get_tests_for_function`               |
| `get_callgraph_neighborhood` (Nx graph + docs views)       | **mixed**: GraphRepository (relational) + GraphEngine |
| `get_import_boundary` (import graph + subsystem_modules)   | SubsystemRepository (membership) + GraphEngine        |
| `get_file_summary` (docs.v_file_summary)                   | `ModuleRepository.get_file_summary`                   |
| `get_file_profile` (docs.v_file_profile)                   | `ModuleRepository.get_file_profile`                   |
| `get_module_profile` (docs.v_module_profile)               | `ModuleRepository.get_module_profile`                 |
| `get_function_profile` (docs.v_function_profile)           | `FunctionRepository.get_function_profile`             |
| `get_function_architecture` (docs.v_function_architecture) | `FunctionRepository.get_function_architecture`        |
| `get_module_architecture` (docs.v_module_architecture)     | `ModuleRepository.get_module_architecture`            |
| `list_subsystems` (docs.v_subsystem_summary)               | `SubsystemRepository.list_subsystems`                 |
| `get_module_subsystems` (docs.v_subsystem_modules)         | `SubsystemRepository.list_subsystem_modules`          |
| `search_subsystems`                                        | `SubsystemRepository.search_subsystems`               |
| `summarize_subsystem`                                      | `SubsystemRepository.get_subsystem_summary` + modules |
| `get_file_hints` (docs.v_ide_hints)                        | `ModuleRepository.get_file_hints`                     |
| `read_dataset_rows` (metadata.dataset_rows)                | `DatasetReadRepository.read_dataset_rows`             |
| `storage.data_models.fetch_*`                              | `DataModelRepository.*`                               |

---

## 5. Testing & rollout

1. **Unit tests for repositories**

   * For each repo, add a small test module under `tests/storage/repositories/test_*.py`:

     * Use a fixture that builds a tiny DuckDB DB with just enough tables/views and sample rows.
     * Assert that repository methods return the expected dicts / row models.

2. **Golden tests for `DuckDBQueryService`**

   * Before refactor, capture a small snapshot of responses for a known fixture repo:

     ```python
     def test_query_service_golden(snapshot_db: StorageGateway):
         svc = DuckDBQueryService(...)
         res = svc.get_function_summary(...).model_dump()
         assert res == EXPECTED
     ```

   * After refactor, those tests should still pass without modification.

3. **Runtime smoke tests**

   * Run:

     * pipeline ingest + analytics
     * `cli export-docs`
     * FastAPI / MCP backend in dev mode
   * Hit a couple of endpoints/tools to verify nothing broken:

     * function summary
     * file summary
     * subsystem search
     * dataset browse.

---

