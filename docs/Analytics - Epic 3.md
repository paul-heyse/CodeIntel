Here’s how I’d turn **Epic 3** into a concrete, implementable refactor, wired to your *current* repo (with Epic 1 & 2 already in place).

I’ll show full detail for:

* **Function metrics/types** (currently raw tuples → fully typed + contracts).
* **Graph metrics** (functions/modules) as a pattern.
* **Test profiles** (already fairly typed via `storage.rows`, we formalize contracts).
* A proper **`analytics/datasets.py`** that sits on top of your existing `storage.datasets` and `config.schemas.tables`.

---

## 0. Big-picture design

You already have:

* **Central table schemas:** `config/schemas/tables.py` (`TABLE_SCHEMAS["analytics.function_metrics"]`, etc.).
* **Central dataset registry:** `storage/datasets.py` (`Dataset` + `ROW_BINDINGS_BY_TABLE_KEY`).
* **Row models for many analytics tables:** `storage/rows.py`
  (e.g. `FunctionProfileRowModel`, `ProfileRowModel`, `BehavioralCoverageRowModel`, etc.).
* **Ad-hoc tuples** for others:

  * `analytics/functions/metrics.py` → `FunctionAnalyticsResult.metrics_rows: list[tuple]`
  * `analytics/graph_rows/*.py` → `FunctionMetricRow = tuple[...]`, `ModuleMetricRow = tuple[...]`, etc.

Epic 3’s goal is to make **analytics tables feel as strongly typed and contract-bound** as core tables, but without fighting your existing storage layer.

So:

* `storage/rows.py` remains the **source of truth** for row models (for all schemas).
* `storage/datasets.py` remains the **source of truth** for dataset metadata (table name, JSONL filenames, etc.).
* **New** `analytics/rows` becomes the **analytics-facing facade** for row types & `to_row` helpers.
* **New** `analytics/datasets.py` becomes the **analytics-facing facade** for dataset contracts, computed from `TABLE_SCHEMAS` + `DatasetRegistry` + row models.

And analytics code (plugins, profiles, etc.) should almost always:

1. Build **row dicts** using `analytics.rows.*`.
2. Use **dataset contracts** from `analytics.datasets` to insert rows (via `run_batch` / `macro_insert_rows`).

---

## 1. Create `analytics/rows/` as the canonical analytics row layer

### 1.1 Package skeleton

**New directory:** `analytics/rows/`

Files:

* `analytics/rows/__init__.py`
* `analytics/rows/function_metrics.py`
* `analytics/rows/function_types.py`
* `analytics/rows/graph_metrics.py` (functions/modules)
* `analytics/rows/graph_metrics_ext.py` (functions_ext/modules_ext)
* `analytics/rows/test_profiles.py` (thin re-exports from `storage.rows`)
* (Plus anything similar you want to cover: config_values, typedness, hotspots, etc.)

#### `analytics/rows/__init__.py`

Keep this lightweight and re-export per-submodule types. You can start with:

```python
# analytics/rows/__init__.py

from __future__ import annotations

from .function_metrics import FunctionMetricsRow
from .function_types import FunctionTypesRow
from .graph_metrics import FunctionGraphMetricsRow, ModuleGraphMetricsRow
from .graph_metrics_ext import FunctionGraphMetricsExtRow, ModuleGraphMetricsExtRow
from .test_profiles import TestProfileRow, BehavioralCoverageRow

__all__ = [
    "FunctionMetricsRow",
    "FunctionTypesRow",
    "FunctionGraphMetricsRow",
    "ModuleGraphMetricsRow",
    "FunctionGraphMetricsExtRow",
    "ModuleGraphMetricsExtRow",
    "TestProfileRow",
    "BehavioralCoverageRow",
]
```

You’ll add more symbols as you flesh out additional datasets.

---

### 1.2 Function metrics row type

We know the exact schema from `config/schemas/tables.py`:

```python
"analytics.function_metrics": TableSchema(
    schema="analytics",
    name="function_metrics",
    columns=[
        Column("function_goid_h128", "DECIMAL(38,0)"),
        Column("urn", "VARCHAR"),
        "repo", "commit", "rel_path", "language", "kind", "qualname",
        "start_line", "end_line", "loc", "logical_loc",
        "param_count", "positional_only_params", "positional_params",
        "keyword_only_params", "vararg_params", "kwarg_params",
        "has_vararg", "has_kwarg", "positional_default_count",
        "keyword_default_count", "has_returns", "has_yield",
        "has_raise", "is_async", "is_generator",
        "return_count", "yield_count", "raise_count",
        "cyclomatic_complexity", "max_nesting_depth",
        "stmt_count", "decorator_count", "has_docstring",
        "complexity_bucket", "created_at",
    ],
    ...
)
```

Let’s define a `TypedDict` for this:

```python
# analytics/rows/function_metrics.py

from __future__ import annotations

from datetime import datetime
from typing import TypedDict

from codeintel.storage.rows import _serialize_row  # internal helper
from codeintel.storage.rows import FUNCTION_METRICS_COLUMNS  # you'll add this


class FunctionMetricsRow(TypedDict):
    """Row shape for analytics.function_metrics inserts."""

    function_goid_h128: int
    urn: str | None
    repo: str
    commit: str
    rel_path: str
    language: str | None
    kind: str | None
    qualname: str | None
    start_line: int | None
    end_line: int | None
    loc: int | None
    logical_loc: int | None
    param_count: int | None
    positional_only_params: int | None
    positional_params: int | None
    keyword_only_params: int | None
    vararg_params: int | None
    kwarg_params: int | None
    has_vararg: bool
    has_kwarg: bool
    positional_default_count: int | None
    keyword_default_count: int | None
    has_returns: bool
    has_yield: bool
    has_raise: bool
    is_async: bool
    is_generator: bool
    return_count: int | None
    yield_count: int | None
    raise_count: int | None
    cyclomatic_complexity: int | None
    max_nesting_depth: int | None
    stmt_count: int | None
    decorator_count: int | None
    has_docstring: bool
    complexity_bucket: str | None
    created_at: datetime
```

Now define a serializer:

```python
def function_metrics_row_to_tuple(row: FunctionMetricsRow) -> tuple[object, ...]:
    """
    Serialize a FunctionMetricsRow into INSERT column order.

    Relies on FUNCTION_METRICS_COLUMNS (defined in storage.rows).
    """
    return _serialize_row(row, FUNCTION_METRICS_COLUMNS)
```

> **Storage side:** you’ll add `FUNCTION_METRICS_COLUMNS` and a small wrapper in `storage/rows.py` (see 2.1 below).

---

### 1.3 Function types row type

Same pattern; schema is fully defined in `TABLE_SCHEMAS["analytics.function_types"]`.

```python
# analytics/rows/function_types.py

from __future__ import annotations

from datetime import datetime
from typing import TypedDict

from codeintel.storage.rows import _serialize_row, FUNCTION_TYPES_COLUMNS


class FunctionTypesRow(TypedDict):
    """Row shape for analytics.function_types inserts."""

    function_goid_h128: int
    urn: str | None
    repo: str
    commit: str
    rel_path: str
    language: str | None
    kind: str | None
    qualname: str | None
    start_line: int | None
    end_line: int | None
    total_params: int | None
    annotated_params: int | None
    annotated_param_ratio: float | None
    has_return_annotation: bool
    file_param_ratio: float | None
    file_annotated_ratio: float | None
    file_fully_typed: bool
    file_any_typed: bool
    fully_typed: bool
    partial_typed: bool
    untyped: bool
    typedness_bucket: str | None
    typedness_source: str | None
    created_at: datetime
```

Serializer:

```python
def function_types_row_to_tuple(row: FunctionTypesRow) -> tuple[object, ...]:
    return _serialize_row(row, FUNCTION_TYPES_COLUMNS)
```

Again, you’ll define `FUNCTION_TYPES_COLUMNS` in `storage/rows.py`.

---

### 1.4 Graph metrics row types

You already have compact tuple shapes in `analytics/graph_rows/graph_metrics.py`:

```python
FunctionMetricRow = tuple[str, str, int, int, int, int, int, float | None, ...]
ModuleMetricRow = tuple[str, str, str, int, int, int, int, float | None, ...]
```

We can make them **row dicts** instead, aligned with `TABLE_SCHEMAS["analytics.graph_metrics_functions"]` / `modules`.

Example for functions:

```python
# analytics/rows/graph_metrics.py

from __future__ import annotations

from datetime import datetime
from typing import TypedDict

from codeintel.storage.rows import _serialize_row, GRAPH_METRICS_FUNCTIONS_COLUMNS


class FunctionGraphMetricsRow(TypedDict):
    """Row shape for analytics.graph_metrics_functions inserts."""

    repo: str
    commit: str
    function_goid_h128: int
    call_fan_in: int
    call_fan_out: int
    call_in_degree: int
    call_out_degree: int
    call_pagerank: float | None
    call_betweenness: float | None
    call_closeness: float | None
    call_cycle_member: bool
    call_cycle_id: int | None
    call_layer: int | None
    import_fan_in: int
    import_fan_out: int
    import_in_degree: int
    import_out_degree: int
    import_pagerank: float | None
    import_betweenness: float | None
    import_closeness: float | None
    import_cycle_member: bool
    import_cycle_id: int | None
    import_layer: int | None
    symbol_fan_in: int
    symbol_fan_out: int
    created_at: datetime
```

Serializer:

```python
def function_graph_metrics_row_to_tuple(row: FunctionGraphMetricsRow) -> tuple[object, ...]:
    return _serialize_row(row, GRAPH_METRICS_FUNCTIONS_COLUMNS)
```

Same idea for `ModuleGraphMetricsRow` (based on `TABLE_SCHEMAS["analytics.graph_metrics_modules"]`).

Then update `analytics/graph_rows/graph_metrics.py` to build **row dicts** instead of tuples, and to re-use this type:

```python
from codeintel.analytics.rows.graph_metrics import FunctionGraphMetricsRow

def build_function_graph_metric_rows(
    inputs: FunctionGraphMetricInputs,
) -> list[FunctionGraphMetricsRow]:
    created_at = inputs.created_at
    return [
        FunctionGraphMetricsRow(
            repo=inputs.cfg.repo,
            commit=inputs.cfg.commit,
            function_goid_h128=node,
            call_fan_in=inputs.stats.call_fan_in.get(node, 0),
            call_fan_out=inputs.stats.call_fan_out.get(node, 0),
            call_in_degree=inputs.stats.call_in_degree.get(node, 0),
            call_out_degree=inputs.stats.call_out_degree.get(node, 0),
            call_pagerank=inputs.centrality["pagerank"].get(node),
            call_betweenness=inputs.centrality["betweenness"].get(node),
            call_closeness=inputs.centrality["closeness"].get(node),
            call_cycle_member=inputs.components.in_cycle.get(node, False),
            call_cycle_id=inputs.components.scc_id.get(node),
            call_layer=inputs.components.layer.get(node),
            import_fan_in=inputs.import_stats.in_degree.get(node, 0),
            import_fan_out=inputs.import_stats.out_degree.get(node, 0),
            import_in_degree=inputs.import_stats.in_degree.get(node, 0),
            import_out_degree=inputs.import_stats.out_degree.get(node, 0),
            import_pagerank=inputs.centrality["import_pagerank"].get(node),
            import_betweenness=inputs.centrality["import_betweenness"].get(node),
            import_closeness=inputs.centrality["import_closeness"].get(node),
            import_cycle_member=inputs.components.import_in_cycle.get(node, False),
            import_cycle_id=inputs.components.import_scc_id.get(node),
            import_layer=inputs.components.import_layer.get(node),
            symbol_fan_in=inputs.symbol_inbound.get(node, 0),
            symbol_fan_out=inputs.symbol_outbound.get(node, 0),
            created_at=created_at,
        )
        for node in inputs.graph_nodes
    ]
```

Same pattern for `ModuleGraphMetricsRow` + `_ext` rows.

---

### 1.5 Test profile / behavioral coverage rows (re-export)

These are already defined in `storage/rows.py`:

* `ProfileRowModel` (`analytics.test_profile`).
* `BehavioralCoverageRowModel` (`analytics.behavioral_coverage`).

We can just re-export them for analytics code to import via `analytics.rows.test_profiles`:

```python
# analytics/rows/test_profiles.py

from __future__ import annotations

from codeintel.storage.rows import (
    ProfileRowModel as TestProfileRowModel,
    BehavioralCoverageRowModel,
    serialize_test_profile_row,
    behavioral_coverage_row_to_tuple,
)

TestProfileRow = TestProfileRowModel
BehavioralCoverageRow = BehavioralCoverageRowModel

__all__ = [
    "TestProfileRow",
    "BehavioralCoverageRow",
    "serialize_test_profile_row",
    "behavioral_coverage_row_to_tuple",
]
```

Analytics code (e.g., `analytics/tests_profiles/rows.py`) already uses these models + serializers; now they can be imported through `analytics.rows` for consistency.

---

## 2. Introduce `AnalyticsDatasetContract` in `analytics/datasets.py`

Now we build a **thin layer** over:

* `TABLE_SCHEMAS` (for columns, primary_key, indexes).
* `storage.datasets.Dataset` (for metadata).
* Row types from `analytics.rows` / `storage.rows`.

### 2.1 Define the contract type

**New file:** `analytics/datasets.py`

```python
# analytics/datasets.py

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from duckdb import DuckDBPyConnection

from codeintel.config.schemas.tables import TABLE_SCHEMAS, TableSchema
from codeintel.storage.datasets import Dataset, load_dataset_registry
from codeintel.storage.registry_helpers import build_dataset_registry
from codeintel.storage.gateway import StorageGateway
from codeintel.storage import rows as row_models

from codeintel.analytics.rows.function_metrics import (
    FunctionMetricsRow,
    function_metrics_row_to_tuple,
)
from codeintel.analytics.rows.function_types import (
    FunctionTypesRow,
    function_types_row_to_tuple,
)
from codeintel.analytics.rows.test_profiles import (
    TestProfileRow,
    BehavioralCoverageRow,
    serialize_test_profile_row,
    behavioral_coverage_row_to_tuple,
)
# plus graph metrics row types when you add them

RowType: TypeAlias = Mapping[str, object]
ToTuple = Callable[[RowType], tuple[object, ...]]
```

```python
@dataclass(frozen=True)
class AnalyticsDatasetContract:
    """
    Analytics-facing dataset contract for a DuckDB table/view.

    Attributes
    ----------
    name
        Logical dataset name, e.g. "analytics.function_metrics".
    table_key
        Fully-qualified DuckDB identifier (usually same as name).
    schema
        TableSchema entry, including column names and types.
    row_type
        Typed row model (usually a TypedDict).
    to_tuple
        Serializer from row dict -> tuple in INSERT column order.
    primary_key
        Primary key columns (if known).
    indexes
        Index definitions (column tuples).
    """

    name: str
    table_key: str
    schema: TableSchema | None
    row_type: type[RowType]
    to_tuple: ToTuple
    primary_key: tuple[str, ...]
    indexes: tuple[tuple[str, ...], ...]
    dataset_meta: Dataset | None = None
```

```python
def _schema_for(table_key: str) -> TableSchema | None:
    return TABLE_SCHEMAS.get(table_key)


def _dataset_meta(con: DuckDBPyConnection, name: str) -> Dataset | None:
    registry = build_dataset_registry(con)
    return registry.by_name.get(name)
```

### 2.2 Build a static map for key analytics tables

We’ll define contracts for a first set of analytics tables:

* `analytics.function_metrics`
* `analytics.function_types`
* `analytics.function_profile`
* `analytics.test_profile`
* `analytics.behavioral_coverage`
* `analytics.graph_metrics_functions` (once you add row types)
* `analytics.graph_metrics_modules` etc.

```python
def build_analytics_dataset_contracts(
    gateway: StorageGateway,
) -> dict[str, AnalyticsDatasetContract]:
    """
    Build dataset contracts for analytics tables, reusing dataset registry metadata.

    This is cheap and can be called at startup, or cached in the gateway.
    """
    con = gateway.con
    registry = load_dataset_registry(con)

    def _contract(
        name: str,
        *,
        row_type: type[RowType],
        to_tuple: ToTuple,
    ) -> AnalyticsDatasetContract:
        dataset = registry.by_name.get(name)
        table_key = dataset.table_key if dataset is not None else name
        schema = TABLE_SCHEMAS.get(table_key)
        pk = schema.primary_key if schema is not None else ()
        idx = tuple((idx.columns,) for idx in (schema.indexes or ())) if schema is not None else ()
        return AnalyticsDatasetContract(
            name=name,
            table_key=table_key,
            schema=schema,
            row_type=row_type,
            to_tuple=to_tuple,
            primary_key=pk,
            indexes=idx,
            dataset_meta=dataset,
        )

    return {
        "analytics.function_metrics": _contract(
            "analytics.function_metrics",
            row_type=FunctionMetricsRow,  # type: ignore[arg-type]
            to_tuple=function_metrics_row_to_tuple,
        ),
        "analytics.function_types": _contract(
            "analytics.function_types",
            row_type=FunctionTypesRow,  # type: ignore[arg-type]
            to_tuple=function_types_row_to_tuple,
        ),
        "analytics.function_profile": _contract(
            "analytics.function_profile",
            row_type=row_models.FunctionProfileRowModel,  # reuse storage.rows
            to_tuple=row_models.function_profile_row_to_tuple,
        ),
        "analytics.test_profile": _contract(
            "analytics.test_profile",
            row_type=TestProfileRow,  # re-export from analytics.rows
            to_tuple=serialize_test_profile_row,
        ),
        "analytics.behavioral_coverage": _contract(
            "analytics.behavioral_coverage",
            row_type=BehavioralCoverageRow,
            to_tuple=behavioral_coverage_row_to_tuple,
        ),
        # Add graph metrics tables once graph row types are defined
        # "analytics.graph_metrics_functions": _contract(...),
        # "analytics.graph_metrics_modules": _contract(...),
    }
```

Optionally expose a helper:

```python
def get_analytics_dataset_contract(
    gateway: StorageGateway,
    name: str,
) -> AnalyticsDatasetContract:
    contracts = build_analytics_dataset_contracts(gateway)
    try:
        return contracts[name]
    except KeyError as exc:
        raise KeyError(f"Unknown analytics dataset: {name}") from exc
```

---

### 2.3 Helper to insert rows for a contract

We’ll use existing `run_batch` but now with row dicts:

```python
from codeintel.ingestion.common import run_batch

def insert_analytics_rows(
    gateway: StorageGateway,
    contract: AnalyticsDatasetContract,
    rows: list[RowType],
    *,
    delete_params: list[object] | None = None,
    scope: str | None = None,
) -> None:
    """
    Insert rows for a dataset contract using run_batch.

    - Converts row dicts to tuples via contract.to_tuple.
    - Uses contract.table_key for table resolution.
    """
    if not rows:
        return
    tuple_rows = [contract.to_tuple(row) for row in rows]
    run_batch(
        gateway,
        contract.table_key,
        tuple_rows,
        delete_params=delete_params,
        scope=scope,
    )
```

This is the **one place** that knows about `run_batch` and tuple conversion for analytics; plugins and analytics modules just deal in row dicts + contracts.

---

## 3. Use contracts inside analytics code

### 3.1 Function metrics/types: `analytics/functions/metrics.py`

Right now:

* `FunctionAnalyticsResult.metrics_rows: list[tuple]`
* `FunctionAnalyticsResult.types_rows: list[tuple]`
* `persist_function_analytics(...)` calls `run_batch` with those tuples.

We’ll:

1. Change `FunctionAnalyticsResult` to carry typed rows.
2. Use `insert_analytics_rows` with contracts.

#### 3.1.1 Change the result type

At the top of `analytics/functions/metrics.py`:

```python
from codeintel.analytics.rows.function_metrics import FunctionMetricsRow
from codeintel.analytics.rows.function_types import FunctionTypesRow
```

Change the dataclass:

```python
@dataclass(frozen=True)
class FunctionAnalyticsResult:
    """Pure analysis output for function metrics/types plus validation."""

    metrics_rows: list[FunctionMetricsRow]
    types_rows: list[FunctionTypesRow]
    reporter: FunctionValidationReporter
```

Update the internal helpers to build **row dicts** instead of tuples.

For example, where you currently have:

```python
metrics_rows: list[tuple] = []
types_rows: list[tuple] = []
...
metrics_rows.append(metrics_row)
types_rows.append(types_row)
...
return FunctionAnalyticsResult(
    metrics_rows=metrics_rows,
    types_rows=types_rows,
    reporter=reporter,
)
```

Change to:

```python
metrics_rows: list[FunctionMetricsRow] = []
types_rows: list[FunctionTypesRow] = []
...
metrics_rows.append(metrics_row)  # now FunctionMetricsRow dicts
types_rows.append(types_row)      # now FunctionTypesRow dicts
...
return FunctionAnalyticsResult(
    metrics_rows=metrics_rows,
    types_rows=types_rows,
    reporter=reporter,
)
```

HOW you build `metrics_row` & `types_row`:

* either inline `FunctionMetricsRow(...)` / `FunctionTypesRow(...)` in the function where you derive metrics,
* or add little pure helpers like `build_function_metrics_row(meta, stats) -> FunctionMetricsRow`.

---

#### 3.1.2 `persist_function_analytics` uses contracts

Replace the raw `run_batch` calls:

```python
from codeintel.analytics.datasets import (
    get_analytics_dataset_contract,
    insert_analytics_rows,
)

def persist_function_analytics(
    gateway: StorageGateway,
    cfg: FunctionAnalyticsStepConfig,
    result: FunctionAnalyticsResult,
) -> dict[str, int]:
    """
    Persist analytics rows and validation to DuckDB.
    """
    scope = f"{cfg.repo}@{cfg.commit}"

    metrics_contract = get_analytics_dataset_contract(
        gateway, "analytics.function_metrics"
    )
    types_contract = get_analytics_dataset_contract(
        gateway, "analytics.function_types"
    )

    delete_params = [cfg.repo, cfg.commit]

    insert_analytics_rows(
        gateway,
        metrics_contract,
        result.metrics_rows,
        delete_params=delete_params,
        scope=scope,
    )
    insert_analytics_rows(
        gateway,
        types_contract,
        result.types_rows,
        delete_params=delete_params,
        scope=scope,
    )

    result.reporter.flush(gateway)
    ...
```

You can remove or demote the old direct `run_batch` import if nothing else uses it in this module.

---

### 3.2 Graph metrics: `analytics/graph_rows/graph_metrics.py`

Right now it returns bare tuples; now you have `FunctionGraphMetricsRow` and `ModuleGraphMetricsRow`.

Update builders (as shown above) to return lists of row dicts, and adapt the **plugins** to call `insert_analytics_rows` with the new contracts:

In the graph metrics plugin module (e.g. `analytics/graphs/graph_metrics.py` or your graph plugin handlers), replace:

```python
rows = build_function_graph_metric_rows(inputs)
gateway.insert_graph_metrics_functions(rows)
```

with:

```python
from codeintel.analytics.datasets import get_analytics_dataset_contract, insert_analytics_rows

rows = build_function_graph_metric_rows(inputs)
contract = get_analytics_dataset_contract(
    gateway, "analytics.graph_metrics_functions"
)
insert_analytics_rows(
    gateway,
    contract,
    rows,
    delete_params=[cfg.repo, cfg.commit],
    scope=f"{cfg.repo}@{cfg.commit}",
)
```

You’ll need to define row models & `GRAPH_METRICS_FUNCTIONS_COLUMNS` / row_bindings in `storage.rows` / `storage.datasets` for these tables similarly to function_metrics/types.

---

### 3.3 Test profiles: `analytics/tests_profiles/rows.py`

This module is already strongly typed, using:

* `TestProfileRowModel` (`ProfileRowModel` in `storage.rows`).
* `BehavioralCoverageRowModel`.

And it uses `write_rows_with_registry_guard` with a writer that internally uses `serialize_test_profile_row` / `behavioral_coverage_row_to_tuple`.

You don’t need to change much; but you can optionally “normalize” on dataset contracts by switching to `insert_analytics_rows` and `AnalyticsDatasetContract`. For example, inside `write_test_profile_rows` you can:

* Fetch the `analytics.test_profile` contract.
* Use `insert_analytics_rows` instead of storage-specific guard, if you want fully consistent behavior.

Given this is already very strongly typed and uses registry-based writer guards, I’d treat Epic 3’s ask here as **“re-export and document, not rewrite”**. The main wins are for function & graph metrics.

---

## 4. Contract-level invariants & tests

Finally, we add tests that assert:

1. **Row models match table schemas** (column name parity).
2. **Row insertion is idempotent by (repo, commit)**.
3. **Foreign keys across analytics tables are consistent**.

### 4.1 Row model vs schema parity

**New test:** `tests/analytics/test_analytics_rows_contracts.py`

```python
from __future__ import annotations

from typing import get_type_hints

from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.analytics.rows.function_metrics import FunctionMetricsRow
from codeintel.analytics.rows.function_types import FunctionTypesRow
from codeintel.analytics.rows.test_profiles import TestProfileRow, BehavioralCoverageRow


def _assert_row_matches_table(row_type: type[dict], table_key: str) -> None:
    schema = TABLE_SCHEMAS[table_key]
    expected_cols = [col.name for col in schema.columns]
    annotations = get_type_hints(row_type)
    actual_keys = list(annotations.keys())
    assert actual_keys == expected_cols, f"{table_key} mismatch: {actual_keys} != {expected_cols}"


def test_function_metrics_row_matches_schema() -> None:
    _assert_row_matches_table(FunctionMetricsRow, "analytics.function_metrics")


def test_function_types_row_matches_schema() -> None:
    _assert_row_matches_table(FunctionTypesRow, "analytics.function_types")


def test_test_profile_row_matches_schema() -> None:
    _assert_row_matches_table(TestProfileRow, "analytics.test_profile")


def test_behavioral_coverage_row_matches_schema() -> None:
    _assert_row_matches_table(BehavioralCoverageRow, "analytics.behavioral_coverage")
```

> If column ordering differs (e.g. you don’t want to enforce ordering on the TypedDict), you can compare sets but still ensure the serializer respects `TABLE_SCHEMAS` order.

---

### 4.2 Idempotency by (repo, commit)

For any analytics dataset whose schema includes `repo`/`commit` and for which your ingestion deletes by `repo, commit`, verify that:

* Delete + reinsert yields identical row counts and no duplicates.

Example test for function metrics:

```python
from __future__ import annotations

from codeintel.analytics.datasets import (
    get_analytics_dataset_contract,
    insert_analytics_rows,
)
from codeintel.analytics.rows.function_metrics import FunctionMetricsRow
from codeintel.config import ConfigBuilder
from tests._helpers.fixtures import provisioned_gateway


def test_function_metrics_idempotent_by_repo_commit(provisioned_gateway) -> None:
    gateway = provisioned_gateway.gateway
    snapshot = provisioned_gateway.snapshot

    builder = ConfigBuilder.from_snapshot(snapshot)
    cfg = builder.function_analytics(fail_on_missing_spans=False, parser=None)

    # Run full function analytics via plugin/harness to populate data
    # (or call compute_function_metrics_and_types + persist directly)

    con = gateway.con
    count_before = con.execute(
        """
        SELECT COUNT(*) FROM analytics.function_metrics
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    ).fetchone()[0]

    # Load back rows as FunctionMetricsRow dicts (simplest: using SELECT * + dict)
    rows = [
        FunctionMetricsRow(**dict(row))
        for row in con.execute(
            "SELECT * FROM analytics.function_metrics WHERE repo = ? AND commit = ?",
            [cfg.repo, cfg.commit],
        ).fetchall()
    ]

    contract = get_analytics_dataset_contract(gateway, "analytics.function_metrics")
    insert_analytics_rows(
        gateway,
        contract,
        rows,
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )

    count_after = con.execute(
        """
        SELECT COUNT(*) FROM analytics.function_metrics
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    ).fetchone()[0]

    assert count_before == count_after
```

You can generalize this into a helper and reuse for:

* `analytics.function_types`
* `analytics.graph_metrics_functions`
* `analytics.test_profile`
* `analytics.behavioral_coverage`
* etc.

---

### 4.3 Foreign key consistency checks

For example: `entrypoint_tests.entrypoint_id` → `entrypoints.entrypoint_id`.

You can write a generic helper that:

* Takes source dataset/table, FK columns, and target dataset/table and PK columns.
* SELECTS all distinct FK values and checks that they all exist in the target table.

This is mostly schema/signup work, so I won’t dump a ton of FK-specific SQL here, but the pattern is:

```python
def assert_fk(
    con,
    src_table: str,
    src_cols: tuple[str, ...],
    dst_table: str,
    dst_cols: tuple[str, ...],
) -> None:
    src_list = ", ".join(src_cols)
    dst_list = ", ".join(dst_cols)
    rows = con.execute(
        f"""
        SELECT s.{src_list}
        FROM {src_table} s
        LEFT JOIN {dst_table} t
          ON ({' AND '.join(f's.{c} = t.{c}' for c in src_cols)})
        WHERE {' OR '.join(f't.{c} IS NULL' for c in dst_cols)}
        LIMIT 1
        """
    ).fetchall()
    assert not rows, f"FK violation from {src_table}({src_list}) -> {dst_table}({dst_list})"
```

Then use:

```python
assert_fk(
    con,
    "analytics.entrypoint_tests",
    ("repo", "commit", "entrypoint_id"),
    "analytics.entrypoints",
    ("repo", "commit", "entrypoint_id"),
)
```

in a test.

---

## Summary

Epic 3 boils down to:

* **analytics/rows/**: where analytics code gets strongly typed row models for its own tables (function_metrics/types, graph metrics, test profile, behavioral coverage, etc.).

* **analytics/datasets.py**: an **analytics-facing view** of the dataset registry, providing `AnalyticsDatasetContract` objects that encode row models + schema + PK/index info.

* **Analytics code** (plugins, metrics, profiles) moves from “raw tuples into `run_batch`” to:

  ```python
  contract = get_analytics_dataset_contract(gateway, "analytics.function_metrics")
  rows: list[FunctionMetricsRow] = ...
  insert_analytics_rows(gateway, contract, rows, delete_params=[repo, commit])
  ```

* **Tests** enforce row model ↔ schema parity and idempotent insertion by `(repo, commit)`.

If you’d like, next I can:

* Pick **graph metrics functions** specifically and walk through the exact row model + contract + plugin changes end-to-end, or
* Help you design a small **introspection CLI** (“list analytics datasets”, “show contract for analytics.function_metrics”) on top of `analytics/datasets.py` for agents and manual debugging.

# graph metrics function zoom in #

Nice, this is the fun one 😄 — let’s make **graph metrics (functions)** a fully contract-driven path end-to-end.

I’ll walk through:

1. Current flow (what’s happening today).
2. **Storage layer**: add a proper row model + binding for `analytics.graph_metrics_functions`.
3. **Analytics row builder**: update `analytics/graph_rows/graph_metrics.py` to build typed rows, not anonymous tuples.
4. **Graph metrics engine**: update `analytics/graphs/graph_metrics.py` to insert via dataset contract instead of a bespoke gateway method.
5. Tests and invariants.

I’ll keep the changes **local to graph metrics functions**; you can mirror the pattern for modules/ext later.

---

## 0. Current flow (today)

Right now (from the code you just uploaded):

* **Row builder:** `analytics/graph_rows/graph_metrics.py` defines:

  ```python
  FunctionMetricRow = tuple[
      str,  # repo
      str,  # commit
      int,  # function_goid_h128
      int,  # call_fan_in
      int,  # call_fan_out
      int,  # call_in_degree
      int,  # call_out_degree
      float | None,  # call_pagerank
      float | None,  # call_betweenness
      float | None,  # call_closeness
      bool,          # call_cycle_member
      int | None,    # call_cycle_id
      int | None,    # call_layer
      str,           # created_at ISO string
  ]
  ```

  and `build_function_graph_metric_rows(...) -> list[FunctionMetricRow]`.

* **Compute & persist:** in `analytics/graphs/graph_metrics.py`:

  ```python
  con.execute(
      "DELETE FROM analytics.graph_metrics_functions WHERE repo = ? AND commit = ?",
      [cfg.repo, cfg.commit],
  )

  rows = build_function_graph_metric_rows(...)
  if rows:
      gateway.analytics.insert_graph_metrics_functions(rows)
  ```

* **Gateway:** `storage/gateway.py` has:

  ```python
  def insert_graph_metrics_functions(
      self,
      rows: Iterable[tuple[...]],
  ) -> None:
      macro_insert_rows(self.con, "analytics.graph_metrics_functions", rows)
  ```

* **Schema:** `config/schemas/tables.py`:

  ```python
  "analytics.graph_metrics_functions": TableSchema(
      schema="analytics",
      name="graph_metrics_functions",
      columns=[
          Column("repo", "VARCHAR", nullable=False),
          Column("commit", "VARCHAR", nullable=False),
          Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),
          Column("call_fan_in", "INTEGER", nullable=False),
          Column("call_fan_out", "INTEGER", nullable=False),
          Column("call_in_degree", "INTEGER", nullable=False),
          Column("call_out_degree", "INTEGER", nullable=False),
          Column("call_pagerank", "DOUBLE"),
          Column("call_betweenness", "DOUBLE"),
          Column("call_closeness", "DOUBLE"),
          Column("call_cycle_member", "BOOLEAN", nullable=False),
          Column("call_cycle_id", "INTEGER"),
          Column("call_layer", "INTEGER"),
          Column("created_at", "TIMESTAMP", nullable=False),
      ],
      primary_key=("repo", "commit", "function_goid_h128"),
      ...
  )
  ```

We’re going to:

* Give this table a **row model** in `storage/rows.py`.
* Bind it in `storage/datasets.py`.
* Have the graph metrics code build **row dicts** of that model and use a dataset contract / helper to insert.

---

## 1. Storage: row model + binding for `analytics.graph_metrics_functions`

### 1.1 Add row model in `storage/rows.py`

In `storage/rows.py`, near other analytics rows (e.g. `CoverageLineRow`, `HotspotRow`, etc.), define a `TypedDict` and serializer.

**Add:**

```python
# storage/rows.py

from typing import TypedDict

# ... existing imports & helpers (including _serialize_row) ...


class GraphMetricsFunctionsRow(TypedDict):
    """
    Row shape for analytics.graph_metrics_functions inserts.

    Mirrors TABLE_SCHEMAS["analytics.graph_metrics_functions"].
    """

    repo: str
    commit: str
    function_goid_h128: int
    call_fan_in: int
    call_fan_out: int
    call_in_degree: int
    call_out_degree: int
    call_pagerank: float | None
    call_betweenness: float | None
    call_closeness: float | None
    call_cycle_member: bool
    call_cycle_id: int | None
    call_layer: int | None
    created_at: datetime
```

We want a canonical **column order list** so we can use `_serialize_row`:

```python
_GRAPH_METRICS_FUNCTIONS_COLUMNS: list[str] = [
    "repo",
    "commit",
    "function_goid_h128",
    "call_fan_in",
    "call_fan_out",
    "call_in_degree",
    "call_out_degree",
    "call_pagerank",
    "call_betweenness",
    "call_closeness",
    "call_cycle_member",
    "call_cycle_id",
    "call_layer",
    "created_at",
]
```

Serializer:

```python
def graph_metrics_functions_row_to_tuple(
    row: GraphMetricsFunctionsRow,
) -> tuple[object, ...]:
    """
    Serialize a GraphMetricsFunctionsRow into INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values aligned with analytics.graph_metrics_functions columns.
    """
    return _serialize_row(row, _GRAPH_METRICS_FUNCTIONS_COLUMNS)
```

Add the symbols to `__all__` at the top of the file:

```python
__all__ = [
    # ...
    "GraphMetricsFunctionsRow",
    "graph_metrics_functions_row_to_tuple",
    # ...
]
```

> You can add row models for `analytics.graph_metrics_functions_ext` later in exactly the same style.

---

### 1.2 Bind the row model in `storage/datasets.py`

In `storage/datasets.py`, `ROW_BINDINGS_BY_TABLE_KEY` tells the dataset registry how to map tables to row models.

Find `ROW_BINDINGS_BY_TABLE_KEY` and add an entry for `analytics.graph_metrics_functions`:

```python
# storage/datasets.py

from codeintel.storage import rows as row_models

ROW_BINDINGS_BY_TABLE_KEY: dict[str, RowBinding] = {
    # ... existing bindings ...

    "analytics.graph_metrics_functions": _row_binding(
        row_type=row_models.GraphMetricsFunctionsRow,
        to_tuple=row_models.graph_metrics_functions_row_to_tuple,
    ),

    # "analytics.graph_metrics_functions_ext": ... (later)
}
```

You already have JSONL/parquet entries for this dataset at the bottom:

```python
JSONL_FILENAMES_BY_TABLE_KEY = {
    # ...
    "analytics.graph_metrics_functions": "graph_metrics_functions.jsonl",
    # ...
}
PARQUET_FILENAMES_BY_TABLE_KEY = {
    # ...
    "analytics.graph_metrics_functions": "graph_metrics_functions.parquet",
    # ...
}
```

So **no change** needed there; we’re just giving the dataset a row binding.

After this, the dataset registry will know:

* Table name
* Row model
* Serializer
* JSONL/parquet filenames

and any generic `Dataset.insert(...)` logic can use that.

---

## 2. Analytics row builder: use the new row model

Now we swap out the anonymous `FunctionMetricRow` tuple for our typed row dict.

### 2.1 Update `analytics/graph_rows/graph_metrics.py`

At the top of the file, import the row model:

```python
# analytics/graph_rows/graph_metrics.py

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from codeintel.analytics.graph_service import ComponentBundle, NeighborStats
from codeintel.config import GraphMetricsStepConfig
from codeintel.storage.gateway import DuckDBError, StorageGateway
from codeintel.storage.rows import GraphMetricsFunctionsRow
```

Remove (or ignore) the old alias:

```python
FunctionMetricRow = tuple[
    str,
    str,
    int,
    int,
    int,
    int,
    int,
    float | None,
    float | None,
    float | None,
    bool,
    int | None,
    int | None,
    str,
]
```

and instead use `GraphMetricsFunctionsRow`.

Your inputs type already looks like:

```python
@dataclass
class FunctionGraphMetricInputs:
    """Inputs required to build graph_metrics_functions rows."""

    cfg: GraphMetricsStepConfig
    stats: NeighborStats
    centrality: Mapping[str, Mapping[Any, float]]
    components: ComponentBundle
    graph_nodes: list[Any]
    created_at: datetime
```

We now change `build_function_graph_metric_rows` to return `list[GraphMetricsFunctionsRow]`:

```python
def build_function_graph_metric_rows(
    inputs: FunctionGraphMetricInputs,
) -> list[GraphMetricsFunctionsRow]:
    """
    Construct rows for analytics.graph_metrics_functions.

    Parameters
    ----------
    inputs :
        Aggregated inputs capturing configuration, metrics, and ordering.

    Returns
    -------
    list[GraphMetricsFunctionsRow]
        Row dicts ready for graph_metrics_functions insertion.
    """
    cfg = inputs.cfg
    stats = inputs.stats
    centrality = inputs.centrality
    components = inputs.components
    created_at = inputs.created_at

    rows: list[GraphMetricsFunctionsRow] = []
    for node in inputs.graph_nodes:
        rows.append(
            GraphMetricsFunctionsRow(
                repo=cfg.repo,
                commit=cfg.commit,
                function_goid_h128=int(node),
                call_fan_in=stats.call_fan_in.get(node, 0),
                call_fan_out=stats.call_fan_out.get(node, 0),
                call_in_degree=stats.call_in_degree.get(node, 0),
                call_out_degree=stats.call_out_degree.get(node, 0),
                call_pagerank=centrality["pagerank"].get(node),
                call_betweenness=centrality["betweenness"].get(node),
                call_closeness=centrality["closeness"].get(node),
                call_cycle_member=components.in_cycle.get(node, False),
                call_cycle_id=components.scc_id.get(node),
                call_layer=components.layer.get(node),
                created_at=created_at,
            )
        )
    return rows
```

Notes:

* We preserve the existing semantics:

  * `stats` / `centrality` / `components` lookups are unchanged.
  * `created_at` is still a `datetime` matching the `TIMESTAMP` column.

You can keep `ModuleMetricRow` + `build_module_graph_metric_rows` tuple-based for now, or convert it in the same style to `GraphMetricsModulesRow`.

---

## 3. Graph metrics engine: insert via dataset contract instead of bespoke gateway method

Now we change **compute_graph_metrics** to:

* Delete by `(repo, commit)` as before.
* Insert **row dicts** using the dataset binding we just created.

### 3.1 Add a helper in `analytics/datasets.py` (if you haven’t yet)

From the Epic 3 plan, we suggested a generic contract + helper; let’s define the minimal bits needed for graph metrics.

```python
# analytics/datasets.py

from __future__ import annotations

from collections.abc import Mapping, Callable
from dataclasses import dataclass
from typing import Any

from codeintel.config.schemas.tables import TABLE_SCHEMAS, TableSchema
from codeintel.storage.datasets import load_dataset_registry, Dataset
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.ingestion_common import macro_insert_rows  # or your helper
from codeintel.storage.rows import GraphMetricsFunctionsRow, graph_metrics_functions_row_to_tuple

RowDict = Mapping[str, object]
ToTuple = Callable[[RowDict], tuple[object, ...]]


@dataclass(frozen=True)
class AnalyticsDatasetContract:
    name: str
    table_key: str
    schema: TableSchema | None
    row_type: type[RowDict]
    to_tuple: ToTuple
    dataset_meta: Dataset | None = None
```

Contract builder for `analytics.graph_metrics_functions`:

```python
def get_graph_metrics_functions_contract(
    gateway: StorageGateway,
) -> AnalyticsDatasetContract:
    con = gateway.con
    registry = load_dataset_registry(con)

    name = "analytics.graph_metrics_functions"
    dataset = registry.by_name.get(name)
    table_key = dataset.table_key if dataset is not None else name
    schema = TABLE_SCHEMAS.get(table_key)

    return AnalyticsDatasetContract(
        name=name,
        table_key=table_key,
        schema=schema,
        row_type=GraphMetricsFunctionsRow,  # type: ignore[arg-type]
        to_tuple=graph_metrics_functions_row_to_tuple,
        dataset_meta=dataset,
    )
```

A tiny insert helper:

```python
def insert_analytics_rows(
    gateway: StorageGateway,
    contract: AnalyticsDatasetContract,
    rows: list[RowDict],
    *,
    delete_params: list[object] | None = None,
    scope: str | None = None,
) -> None:
    """
    Insert rows using a dataset contract.

    - Optionally DELETE by delete_params first (if provided).
    - Serialize row dicts -> tuples via contract.to_tuple.
    """
    con = gateway.con
    if delete_params is not None:
        con.execute(
            f"DELETE FROM {contract.table_key} WHERE repo = ? AND commit = ?",
            delete_params,
        )

    if not rows:
        return

    tuple_rows = [contract.to_tuple(row) for row in rows]
    macro_insert_rows(con, contract.table_key, tuple_rows)
```

> You can reuse a more generic `insert_analytics_rows` from the earlier Epic 3 plan instead of this local version; I’m just making the example self-contained.

---

### 3.2 Use the contract in `analytics/graphs/graph_metrics.py`

Find the function that populates function graph metrics; you already saw the core snippet:

```python
# analytics/graphs/graph_metrics.py

con.execute(
    "DELETE FROM analytics.graph_metrics_functions WHERE repo = ? AND commit = ?",
    [cfg.repo, cfg.commit],
)

rows = build_function_graph_metric_rows(
    FunctionGraphMetricInputs(
        cfg=cfg,
        stats=stats,
        centrality=centrality,
        components=components,
        graph_nodes=sorted(graph.nodes),
        created_at=created_at,
    )
)

if rows:
    gateway.analytics.insert_graph_metrics_functions(rows)
    log.info(
        "graph_metrics_functions populated: %d rows for %s@%s",
        len(rows),
        cfg.repo,
        cfg.commit,
    )
```

We replace this with **contract-based insertion** using the row dicts:

```python
from codeintel.analytics.datasets import (
    get_graph_metrics_functions_contract,
    insert_analytics_rows,
)

# ...

contract = get_graph_metrics_functions_contract(gateway)

rows = build_function_graph_metric_rows(
    FunctionGraphMetricInputs(
        cfg=cfg,
        stats=stats,
        centrality=centrality,
        components=components,
        graph_nodes=sorted(graph.nodes),
        created_at=created_at,
    )
)

insert_analytics_rows(
    gateway,
    contract,
    rows,
    delete_params=[cfg.repo, cfg.commit],
    scope=f"{cfg.repo}@{cfg.commit}",
)

log.info(
    "graph_metrics_functions populated: %d rows for %s@%s",
    len(rows),
    cfg.repo,
    cfg.commit,
)
```

You can now **delete** the bespoke gateway method `insert_graph_metrics_functions` from `storage/gateway.py` once everything else is migrated to the dataset helper (here and anywhere else that might call it).

If you prefer a softer transition, you can keep `insert_graph_metrics_functions` as a thin shim:

```python
# storage/gateway.py

def insert_graph_metrics_functions(
    self,
    rows: Iterable[GraphMetricsFunctionsRow],
) -> None:
    """
    Backwards-compatible wrapper around analytics dataset contract insertion.
    """
    from codeintel.analytics.datasets import get_graph_metrics_functions_contract, insert_analytics_rows

    contract = get_graph_metrics_functions_contract(self)
    insert_analytics_rows(self, contract, list(rows))
```

…but I’d lean towards updating callers to the new helper and then deleting this once your tests are green.

---

## 4. Tests & invariants (graph metrics functions)

### 4.1 Row model vs schema parity

**New test:** `tests/analytics/test_graph_metrics_functions_rows.py`

```python
from __future__ import annotations

from typing import get_type_hints

from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.storage.rows import GraphMetricsFunctionsRow


def test_graph_metrics_functions_row_matches_schema() -> None:
    schema = TABLE_SCHEMAS["analytics.graph_metrics_functions"]
    expected_cols = [col.name for col in schema.columns]
    hints = get_type_hints(GraphMetricsFunctionsRow)
    actual_keys = list(hints.keys())
    assert actual_keys == expected_cols, f"Cols mismatch: {actual_keys} != {expected_cols}"
```

This will immediately flag any drift between your `TypedDict` and table schema.

---

### 4.2 End-to-end compute + insert test

**New test:** `tests/analytics/test_graph_metrics_functions_pipeline.py`

Assuming you have helpers to provision a tiny repo snapshot and graphs:

```python
from __future__ import annotations

from datetime import UTC, datetime

from codeintel.analytics.graphs.graph_metrics import compute_graph_metrics, GraphMetricsDeps
from codeintel.analytics.graph_service import GraphEngine, NeighborStats
from codeintel.config import ConfigBuilder
from codeintel.config.steps_graphs import GraphMetricsStepConfig
from tests._helpers.fixtures import provisioned_gateway, simple_call_graph
from tests._helpers.analytics import make_graph_runtime


def test_compute_graph_metrics_functions_inserts_rows(provisioned_gateway) -> None:
    gateway = provisioned_gateway.gateway
    snapshot = provisioned_gateway.snapshot

    builder = ConfigBuilder.from_snapshot(snapshot)
    cfg: GraphMetricsStepConfig = builder.graph_metrics()

    runtime = make_graph_runtime(gateway, snapshot)
    deps = GraphMetricsDeps(
        catalog_provider=None,
        runtime=runtime,
        analytics_context=None,
        filters=None,
    )

    compute_graph_metrics(gateway, cfg, deps=deps)

    con = gateway.con
    rows = con.execute(
        """
        SELECT repo, commit, function_goid_h128, call_fan_in, call_fan_out
        FROM analytics.graph_metrics_functions
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    ).fetchall()

    assert rows, "Expected graph_metrics_functions rows to be populated"
```

You can then add a second test that:

* Runs once.
* Copies rows out as `GraphMetricsFunctionsRow` dicts.
* Deletes + re-inserts via `insert_analytics_rows` and asserts idempotent row counts (just like we did for function metrics).

---

## 5. Summary of concrete changes

For **graph metrics (functions)**, the concrete patch consists of:

1. **`storage/rows.py`**

   * Add `GraphMetricsFunctionsRow` `TypedDict`.
   * Add `_GRAPH_METRICS_FUNCTIONS_COLUMNS` list.
   * Add `graph_metrics_functions_row_to_tuple`.
   * Export these in `__all__`.

2. **`storage/datasets.py`**

   * In `ROW_BINDINGS_BY_TABLE_KEY`, add a binding for `analytics.graph_metrics_functions` using the new row model and serializer.

3. **`analytics/graph_rows/graph_metrics.py`**

   * Import `GraphMetricsFunctionsRow`.
   * Replace `FunctionMetricRow` tuple alias with `GraphMetricsFunctionsRow`.
   * Rewrite `build_function_graph_metric_rows` to build row dicts.

4. **`analytics/datasets.py`**

   * Define `AnalyticsDatasetContract` (if not already).
   * Add `get_graph_metrics_functions_contract(gateway)` and `insert_analytics_rows(...)`.

5. **`analytics/graphs/graph_metrics.py`**

   * Replace the direct `DELETE + gateway.analytics.insert_graph_metrics_functions(rows)` with:

     * `contract = get_graph_metrics_functions_contract(gateway)`
     * `insert_analytics_rows(gateway, contract, rows, delete_params=[repo, commit])`

6. **Tests**

   * Row model vs schema parity test.
   * End-to-end graph_metrics_functions pipeline test.
   * Optional idempotency test via dataset contract.

Once this is in place, you’ve got:

* Strong types for `analytics.graph_metrics_functions`.
* A dataset contract that agents & tools can introspect.
* A clear pattern you can mechanically apply to:

  * `analytics.graph_metrics_modules`
  * `analytics.graph_metrics_functions_ext`
  * `analytics.test_graph_metrics_*`
  * etc.

If you’d like, next we can do a **“batch conversion script” design** (or codemod sketch) to convert any remaining tuple-based `rows: list[tuple[...]]` into “row dict + contract + insert_analytics_rows” across the rest of `analytics/graph_rows` and other analytics modules.


# batch conversion script design #

Perfect, let’s build you a **semi-automated refactor toolchain** to mop up the remaining `list[tuple[...]]` usage across `analytics/graph_rows` and other analytics modules.

Think of this as:

* You define *what* each dataset’s row should look like (column names).
* The codemod does most of the mechanical work:

  * Finds tuple-based row creation.
  * Converts tuples → `TypedDict` constructors.
  * Swaps `run_batch` / bespoke gateway calls → `insert_analytics_rows` with dataset contracts.

I’ll show this in detail for **graph metrics functions** and make the design general so you can apply it to other datasets.

---

## 0. Assumptions

We’ll assume you have (or will have) for each analytics dataset:

1. A **row model** (TypedDict) in `storage/rows.py` or `analytics/rows/*`
   e.g. `GraphMetricsFunctionsRow`, `FunctionMetricsRow`, etc.
2. A **row serializer** in `storage/rows.py`
   e.g. `graph_metrics_functions_row_to_tuple`.
3. A **dataset contract** helper in `analytics/datasets.py`
   e.g. `get_graph_metrics_functions_contract` + `insert_analytics_rows`.

The codemod will then:

* Replace tuple-based row creation with row dict construction.
* Update `run_batch`/bespoke inserts to use `insert_analytics_rows`.

Because column names are table-specific, we’ll drive the codemod with a **config map**.

---

## 1. Config: describe each dataset you want to convert

**New file:** `tools/analytics_refactors/tuple_row_config.py`

This declares what the codemod needs to know per dataset.

```python
# tools/analytics_refactors/tuple_row_config.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TupleRowSpec:
    # Python module where tuples are built
    module: str  # e.g. "codeintel.analytics.graph_rows.graph_metrics"

    # Pattern for builder functions whose rows we want to convert
    builder_functions: Sequence[str]  # e.g. ["build_function_graph_metric_rows"]

    # Name of the variable that holds the list of rows in those builders
    rows_var: str  # e.g. "rows"

    # Fully-qualified row type name
    row_type_qualname: str  # e.g. "codeintel.storage.rows.GraphMetricsFunctionsRow"

    # Name to use locally in the module (import alias)
    row_type_local: str  # e.g. "GraphMetricsFunctionsRow"

    # Column names in tuple order for this dataset
    field_names: Sequence[str]

    # Optional: fully-qualified contract helper to use at insertion sites
    dataset_contract_getter: str | None = None  # e.g. "codeintel.analytics.datasets.get_graph_metrics_functions_contract"


# Example: analytics.graph_metrics_functions
GRAPH_METRICS_FUNCTIONS_SPEC = TupleRowSpec(
    module="codeintel.analytics.graph_rows.graph_metrics",
    builder_functions=["build_function_graph_metric_rows"],
    rows_var="rows",
    row_type_qualname="codeintel.storage.rows.GraphMetricsFunctionsRow",
    row_type_local="GraphMetricsFunctionsRow",
    field_names=[
        "repo",
        "commit",
        "function_goid_h128",
        "call_fan_in",
        "call_fan_out",
        "call_in_degree",
        "call_out_degree",
        "call_pagerank",
        "call_betweenness",
        "call_closeness",
        "call_cycle_member",
        "call_cycle_id",
        "call_layer",
        "created_at",
    ],
    dataset_contract_getter="codeintel.analytics.datasets.get_graph_metrics_functions_contract",
)


ALL_SPECS: list[TupleRowSpec] = [
    GRAPH_METRICS_FUNCTIONS_SPEC,
    # Add more as you go:
    # - FunctionMetrics
    # - FunctionTypes
    # - GraphMetricsModules
    # - etc.
]
```

For each new dataset you want to convert, you just add another `TupleRowSpec` entry with the right module / row type / fields.

---

## 2. Codemod: tuple rows → dict rows (`tools/analytics_refactors/tuple_rows_codemod.py`)

We’ll use **LibCST** to do structural edits.

Broadly, the codemod will:

1. For each configured module:

   * Insert an import for the row type (if missing).
   * Rewrite `rows.append((...))` patterns inside the builder functions:

     * Replace `Tuple(...)` with `RowType(field=..., ...)`.
2. Optionally, in insertion modules:

   * Replace `run_batch(..., "analytics.graph_metrics_functions", rows, ...)` with contract-based insertion.

### 2.1 Dependencies and base

**New file:** `tools/analytics_refactors/tuple_rows_codemod.py`

```python
# tools/analytics_refactors/tuple_rows_codemod.py

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import libcst as cst
import libcst.matchers as m
from libcst import CodemodContext
from libcst.codemod import CodemodCommand, parallel_exec_transform_with_pretty_print

from tools.analytics_refactors.tuple_row_config import ALL_SPECS, TupleRowSpec
```

### 2.2 Utility: parse qualname into module + attr

```python
@dataclass(frozen=True)
class ImportTarget:
    module: str
    name: str

    @classmethod
    def parse(cls, qualname: str) -> "ImportTarget":
        parts = qualname.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid qualname: {qualname}")
        return cls(module=".".join(parts[:-1]), name=parts[-1])
```

---

### 2.3 Transformer for a single module

We’ll create a `TupleRowToDictTransform` that:

* Only runs on modules that match `spec.module`.
* Only rewrites inside functions that match `spec.builder_functions`.
* Only touches calls like `rows.append(( ... ))` where `rows` matches `spec.rows_var`.

```python
class TupleRowToDictTransform(CodemodCommand):
    DESCRIPTION: str = "Convert tuple-based row appends to TypedDict constructors."

    def __init__(self, context: CodemodContext, spec: TupleRowSpec) -> None:
        super().__init__(context)
        self.spec = spec
        self._row_import_added = False

    def transform_module_impl(self, tree: cst.Module) -> cst.Module:
        # Add row type import if missing
        row_import = ImportTarget.parse(self.spec.row_type_qualname)
        row_alias = self.spec.row_type_local

        transformer = _TupleRowBodyTransformer(self.spec)

        new_body = tree.visit(transformer)

        # If we need to insert an import, do it at the top
        if transformer.row_constructor_used and not transformer.row_import_present:
            import_stmt = cst.SimpleStatementLine(
                body=[
                    cst.ImportFrom(
                        module=cst.Name.from_value(row_import.module),
                        names=[
                            cst.ImportAlias(
                                name=cst.Name(row_import.name),
                                asname=cst.AsName(cst.Name(row_alias)),
                            )
                        ],
                    )
                ]
            )
            new_body = new_body.with_changes(
                body=[import_stmt, *new_body.body],
            )

        return new_body
```

We delegate the real work to `_TupleRowBodyTransformer` which walks the syntax tree.

---

### 2.4 Inner transformer: map `rows.append((...))` → `RowType(...)`

```python
class _TupleRowBodyTransformer(cst.CSTTransformer):
    """
    Rewrites:

        rows.append(
            (
                expr0,
                expr1,
                ...
            )
        )

    into:

        rows.append(
            RowType(
                field0=expr0,
                field1=expr1,
                ...
            )
        )
    """

    def __init__(self, spec: TupleRowSpec) -> None:
        self.spec = spec
        self.in_target_function_stack: list[bool] = []
        self.row_constructor_used = False
        self.row_import_present = False

    # --- Track imports ---

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        # crude: check if row_type_local or underlying name is imported anywhere
        if m.matches(
            node,
            m.ImportFrom(
                module=m.Attribute()
                | m.Name(),
                names=m.OneOrMore(
                    m.ImportAlias(
                        name=m.Name(self.spec.row_type_local)
                        | m.Name(ImportTarget.parse(self.spec.row_type_qualname).name)
                    )
                ),
            ),
        ):
            self.row_import_present = True

    # --- Track whether we're in a target function ---

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        is_target = node.name.value in self.spec.builder_functions
        self.in_target_function_stack.append(is_target)

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        self.in_target_function_stack.pop()
        return updated_node

    # --- The main transformation ---

    def leave_Call(
        self,
        original_node: cst.Call,
        updated_node: cst.Call,
    ) -> cst.BaseExpression:
        # Only mutate inside target builder functions
        if not any(self.in_target_function_stack):
            return updated_node

        # Match: rows.append(<single arg>)
        if not m.matches(
            updated_node,
            m.Call(
                func=m.Attribute(
                    value=m.Name(self.spec.rows_var),
                    attr=m.Name("append"),
                ),
                args=m.OneOf(
                    m.Arg(value=m.Tuple(elements=m.ZeroOrMore())),
                    m.Arg(
                        value=m.Tuple(
                            elements=m.ZeroOrMore(m.Element()),
                        )
                    ),
                ),
            ),
        ):
            return updated_node

        args = list(updated_node.args)
        if len(args) != 1:
            return updated_node

        tuple_value = args[0].value
        if not isinstance(tuple_value, cst.Tuple):
            return updated_node

        elements = [e.value for e in tuple_value.elements]
        if len(elements) != len(self.spec.field_names):
            # Don't guess if arity doesn't match
            return updated_node

        # Build RowType(field=expr, ...) call
        row_call = cst.Call(
            func=cst.Name(self.spec.row_type_local),
            args=[
                cst.Arg(keyword=cst.Name(field), value=expr)
                for field, expr in zip(self.spec.field_names, elements)
            ],
        )

        self.row_constructor_used = True

        return updated_node.with_changes(
            args=[cst.Arg(value=row_call)],
        )
```

This is intentionally conservative:

* It only touches calls inside **named builder functions**.
* It only rewrites `rows.append((...))` patterns and only when arity matches `field_names`.
* If something doesn’t match, it leaves the code unchanged.

You can extend the matcher later to handle more patterns (e.g. `rows += [(...)]`).

---

### 2.5 Codemod runner

Finally, we wire up a driver that runs this codemod over your source tree.

```python
def _spec_for_module(module_name: str) -> TupleRowSpec | None:
    for spec in ALL_SPECS:
        if spec.module == module_name:
            return spec
    return None


class MultiSpecDriver(CodemodCommand):
    DESCRIPTION: str = "Apply tuple→dict row refactors for configured analytics datasets."

    def transform_module_impl(self, tree: cst.Module) -> cst.Module:
        module_name = self.context.filename.replace("/", ".").removesuffix(".py")
        spec = _spec_for_module(module_name)
        if spec is None:
            return tree

        inner = TupleRowToDictTransform(self.context, spec)
        return inner.transform_module_impl(tree)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to run the codemod on (e.g. src/codeintel/analytics/graph_rows)",
    )
    args = parser.parse_args(argv)

    parallel_exec_transform_with_pretty_print(
        MultiSpecDriver,
        args.paths,
        # CodemodContext can be enriched if needed
        context_override=None,
    )


if __name__ == "__main__":
    main()
```

Usage from repo root:

```bash
python -m tools.analytics_refactors.tuple_rows_codemod src/codeintel/analytics/graph_rows
```

For now, this only affects modules listed in `ALL_SPECS`.

---

## 3. Extending to other analytics modules

Once you’re happy with the behavior on `graph_metrics_functions`, you can:

1. Add more `TupleRowSpec` entries in `tuple_row_config.py`:

   * `analytics.function_metrics`
   * `analytics.function_types`
   * `analytics.graph_metrics_modules`
   * `analytics.graph_metrics_functions_ext`
   * `analytics.graph_metrics_modules_ext`
   * etc.

   For each, you specify:

   * `module`: where tuple rows are currently built (e.g. `codeintel.analytics.functions.metrics`).
   * `builder_functions`: usually the internal helper that builds the row list, or `persist_*` if it directly builds tuples.
   * `rows_var`: name of the row list (`rows`, `metrics_rows`, etc.).
   * `row_type_qualname`: the fully qualified row type you already created.
   * `field_names`: from `TABLE_SCHEMAS[table_key].columns`.

   This part is mechanical; you can even generate `field_names` by inspecting `TABLE_SCHEMAS` in a little helper script.

2. Run the codemod again on other directories:

   ```bash
   python -m tools.analytics_refactors.tuple_rows_codemod src/codeintel/analytics/functions
   python -m tools.analytics_refactors.tuple_rows_codemod src/codeintel/analytics/graph_rows
   ```

3. Manually review diffs:

   * Ensure **all** tuple → dict conversions look sane.
   * Confirm imports were added correctly.
   * For builders that don’t exactly follow `rows.append((...))`, either:

     * extend `_TupleRowBodyTransformer` with another pattern, or
     * fix those sites by hand.

---

## 4. Optional: codemod for insertion sites (`run_batch` → `insert_analytics_rows`)

The above focuses on **row construction**. You can similarly codemod insertion sites.

Patterns to look for:

* `run_batch(gateway, "analytics.graph_metrics_functions", rows, delete_params=[...])`
* `gateway.analytics.insert_graph_metrics_functions(rows)`

You can write a second codemod that:

* Matches calls to `run_batch` with a hard-coded table string, and rewrites to:

  ```python
  contract = get_graph_metrics_functions_contract(gateway)
  insert_analytics_rows(gateway, contract, rows, delete_params=[...], scope=...)
  ```

* Or turns `gateway.analytics.insert_graph_metrics_functions(rows)` into the same two-step pattern.

Because **insertion logic often includes additional parameters** (scope, delete_params), I’d keep this codemod *very specific* and treat it as “suggested patch” more than a fully generic tool. But the structure is identical: use LibCST, match `Call`, and replace with a small block.

---

## 5. Workflow recommendation

For each dataset family:

1. **Define the row model + serializer + binding** (Epic 3).
2. Add a `TupleRowSpec` entry with the right `field_names` order (from `TABLE_SCHEMAS`).
3. Run the codemod on that module.
4. Run tests and fix any corner cases by hand.
5. Update insertion sites to use dataset contracts and `insert_analytics_rows`.

Once you’ve done this for:

* `analytics.graph_metrics_functions`
* `analytics.function_metrics` / `function_types`
* `analytics.graph_metrics_modules` / `_ext`
* test profile / behavioral coverage (they’re already strongly typed, so you may skip)

you’ll have an analytics layer that:

* Never emits anonymous tuples.
* Always uses explicitly named row types.
* Has a clear, contract-driven insertion path everywhere.

