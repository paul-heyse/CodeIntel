
# explicit data model to data operations driven approach via pandera #

Totally with you on this – you’re sitting on a *lot* of type-ish row/column models that can be turned from “manual clutter” into “one, coherent schema layer that everything else derives from”.

Let’s design an **easy first step** that:

* treats **one thing** as the canonical schema (Pandera),
* makes existing row/column models **derivative** of that, not hand-maintained,
* keeps your existing dataset contracts intact (for now),
* and is simple enough that an LLM programmer can apply it across the repo.

I’ll frame it as a phased plan, customized to what’s inside `config` today.

---

## 0. Current situation (restating in our own words)

Right now you roughly have:

* In `config/`:

  * **dataset contracts** (`TableContract`, `ColumnDef`, `DATASET_CONTRACTS`),
  * **row models** as `TypedDict`s or dataclasses in `config/config/datasets/rows/*.py`,
  * sometimes **column models** or per-field models.
* In `storage/` and `analytics/` and `graphs/`:

  * these models get imported and used for typing & some serialization,
  * but there’s no single, authoritative schema object; it’s “contracts + row models + ad hoc knowledge”.

You want:

> Pandera outputs (DataFrameSchema) to be **tightly coupled** to row data and column specs, so:
>
> * less duplicate modeling,
> * more robustness to change,
> * clearer “where data comes from / goes to” for humans + LLMs.

So we’ll:

1. Introduce a **DatasetSchema** abstraction that wraps Pandera,
2. Build a bridge from your existing **dataset contracts** → Pandera,
3. Auto-derive (or at least auto-check) row models from Pandera,
4. Make it clear in code: “these rows/columns are derivative of Pandera, don’t re-specify them”.

---

## 1. Design the new central abstraction: `DatasetSchema`

We want one place to say:

> For dataset `"analytics.function_metrics"`, here’s:
>
> * The Pandera schema (shape and invariants),
> * Column metadata (DuckDB types, comments, etc.),
> * A row type for Python typing (TypedDict/dataclass/Pydantic),
> * Optional JSON Schema / documentation.

Start with a minimal `DatasetSchema` dataclass.

```python
# config/config/datasets/schema_registry.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Type

import pandera as pa
from pandera import DataFrameSchema

@dataclass(frozen=True)
class DatasetSchema:
    """Canonical schema metadata for a dataset.

    This is the single place describing:
    - name (dataset key)
    - Pandera schema (shape + invariants)
    - column metadata (from existing TableContract)
    - an optional Python row model type for static typing
    """

    name: str
    pandera: DataFrameSchema
    columns: Mapping[str, Any]  # e.g. map to ColumnDef or a simple descriptor
    row_model: Type[Any] | None = None
    # later: json_schema, docs, tags, etc.


DATASET_SCHEMAS: dict[str, DatasetSchema] = {}
```

This is the new “brain”; we’ll populate it from existing contracts, then let everything else use it.

---

## 2. Bridge from existing dataset contracts → Pandera

We *don’t* want to rewrite your big `config/dataset_contract.py` right away. Instead we:

1. Keep `TableContract` / `ColumnDef` as the existing SSOT for column types,
2. Write a **bridge** that builds Pandera schemas from those contracts,
3. Register `DatasetSchema` instances using that.

### 2.1 Mapping ColumnDef → Column + Pandera dtype

Example:

```python
# config/config/datasets/schema_bridge.py

from __future__ import annotations

from typing import Any

import pandera as pa
from pandera import Column, DataFrameSchema

from codeintel.config.datasets.contracts import DATASET_CONTRACTS, TableContract, ColumnDef
from codeintel.config.datasets.schema_registry import DatasetSchema, DATASET_SCHEMAS

PY_TYPE_MAP: dict[str, type[Any]] = {
    "TEXT": str,
    "VARCHAR": str,
    "INTEGER": int,
    "BIGINT": int,
    "UBIGINT": int,
    "DOUBLE": float,
    "BOOLEAN": bool,
    # add others as needed
}

def _columndef_to_pandera(col: ColumnDef) -> Column:
    py_type = PY_TYPE_MAP.get(col.duckdb_type.upper(), Any)
    return Column(
        py_type,
        nullable=col.nullable,
        # add optional checks here if you want (timestamps, ranges, etc.)
    )

def pandera_from_table_contract(contract: TableContract) -> DataFrameSchema:
    columns: dict[str, Column] = {}
    for col in contract.columns:
        columns[col.name] = _columndef_to_pandera(col)
    return DataFrameSchema(columns, strict=True, coerce=True)
```

### 2.2 Populate DatasetSchema from contracts

```python
# config/config/datasets/schema_bridge.py (continued)

def build_all_dataset_schemas() -> None:
    from codeintel.config.datasets.schema_registry import DatasetSchema, DATASET_SCHEMAS

    for key, contract in DATASET_CONTRACTS.items():
        pandera_schema = pandera_from_table_contract(contract)

        ds = DatasetSchema(
            name=key,
            pandera=pandera_schema,
            columns={c.name: c for c in contract.columns},
            row_model=None,  # we will fill this in later for key datasets
        )
        DATASET_SCHEMAS[key] = ds
```

Call `build_all_dataset_schemas()` at startup (e.g., in `config/config/__init__.py`).

This gives you:

* one `DatasetSchema` per dataset,
* with Pandera + column info derived from the existing declarative contract.

---

## 3. Couple row models to Pandera (instead of hand-maintaining)

Now we deal with the “tons of row models” in `config/config/datasets/rows/*.py`.

The **easy first step** is not to auto-generate them yet, but to:

* **explicitly tie** them to their Pandera schema,
* and add tests to ensure they are consistent.

Then, if you like, you can move to auto-generation.

### 3.1 Mark row models as “derived from Pandera”

In `config/config/datasets/rows/analytics.py`, you currently have something like:

```python
from typing import TypedDict

class FunctionMetricsRow(TypedDict):
    repo: str
    commit: str
    rel_path: str
    function_goid_h128: int
    qualname: str
    # ...
```

We can:

1. Import `DatasetSchema` for `"analytics.function_metrics"`,
2. Use it to **check** that `FunctionMetricsRow` matches the schema fields / types,
3. Or even generate the type automatically instead.

#### Option A (safe first step): add consistency tests

Create a small test helper:

```python
# tests/config/test_row_models_match_pandera.py

from typing import get_type_hints

from codeintel.config.datasets.schema_registry import DATASET_SCHEMAS
from codeintel.config.datasets.rows.analytics import FunctionMetricsRow

def test_function_metrics_row_matches_pandera() -> None:
    ds = DATASET_SCHEMAS["analytics.function_metrics"]
    pandera_cols = ds.pandera.columns

    hints = get_type_hints(FunctionMetricsRow)
    row_fields = set(hints.keys())
    schema_fields = set(pandera_cols.keys())

    assert row_fields == schema_fields, (
        "FunctionMetricsRow fields differ from Pandera schema fields: "
        f"row={row_fields}, schema={schema_fields}"
    )
```

Repeat for other key row models (or auto-generate tests with a mapping).

This doesn’t *yet* remove duplication, but it explicitly says:

> “Row model must match Pandera; Pandera is the canonical shape.”

#### Option B (slightly more ambitious): auto-generate TypedDict row models

If you want to go a step further and reduce code:

```python
# config/config/datasets/row_model_factory.py

from __future__ import annotations

from typing import Any, TypedDict, Type

from pandera import DataFrameSchema
from codeintel.config.datasets.schema_registry import DATASET_SCHEMAS

PY_TYPE_MAP = {
    int: int,
    float: float,
    str: str,
    bool: bool,
    "datetime64[ns]": "datetime",  # adapt this
}

def typed_dict_from_pandera(name: str, schema: DataFrameSchema) -> Type[TypedDict]:
    annotations: dict[str, Any] = {}
    for col_name, col in schema.columns.items():
        py_type = col.dtype
        mapped_type = PY_TYPE_MAP.get(py_type, Any)
        annotations[col_name] = mapped_type

    return TypedDict(name, annotations, total=True)
```

Then in `rows/analytics.py`:

```python
from codeintel.config.datasets.schema_registry import DATASET_SCHEMAS
from codeintel.config.datasets.row_model_factory import typed_dict_from_pandera

FunctionMetricsRow = typed_dict_from_pandera(
    "FunctionMetricsRow",
    DATASET_SCHEMAS["analytics.function_metrics"].pandera,
)
```

This **eliminates manual field lists** for row models. The SSOT is now:

* dataset contract → Pandera via `schema_bridge`,
* Pandera → row model via `typed_dict_from_pandera`.

You can start with **just a few key datasets** (e.g., `core.goids`, `graph.call_graph_edges`, `analytics.function_metrics`) and expand over time.

---

## 4. Tie data operations to `DatasetSchema` (easy win for LLM + humans)

Now that you have `DatasetSchema` objects, you can explicitly annotate data operations with *which dataset* they produce or consume.

### 4.1 Producers: plugins marking what they output

In plugin metadata (`CorePluginMetadata`), you already have `produces_tables` / `consumes_tables`. Use that to link code to `DatasetSchema`.

Example:

```python
# analytics/plugins/functions/metrics.py

from codeintel.config.datasets.schema_registry import DATASET_SCHEMAS

class FunctionMetricsPlugin(TargetPlugin):
    metadata = FUNCTION_METRICS_METADATA

    def execute(...):
        ...
        df = pd.DataFrame(rows)
        # Use dataset-specific schema:
        df = DATASET_SCHEMAS["analytics.function_metrics"].pandera.validate(df)
        # Write to DB
        ...
```

An LLM reading this can infer:

* This plugin produces `"analytics.function_metrics"` dataset,
* The schema is in `DATASET_SCHEMAS["analytics.function_metrics"]`,
* Row model type is `FunctionMetricsRow` derived from that.

### 4.2 Consumers: queries using `DatasetSchema` as key

In storage/serving:

```python
def read_function_metrics_df(gateway: StorageGateway, repo: str) -> pd.DataFrame:
    expr = gateway.ibis.con.table("analytics.function_metrics").filter(
        pl.col("repo") == repo
    )
    df = expr.execute()
    ds = DATASET_SCHEMAS["analytics.function_metrics"]
    return ds.pandera.validate(df)
```

That’s a nice, explicit “I consume this dataset” statement.

---

## 5. “Data operation & calculation driven config” – how this gets you closer

Right now, your config is very declarative: “table X has columns A, B, C”. What we’re doing in this first step is:

* Making those declarations **executable** via Pandera,
* Using them to drive:

  * row typing,
  * validation,
  * documentation,
* Linking producers & consumers to **dataset identifiers**, not just functions or row types.

The “operation-driven configuration” comes from the fact that:

* metadata (`CorePluginMetadata`) describes:

  * `requires` / `provides` capabilities,
  * `produces_tables` / `consumes_tables` datasets,
  * `options_model` (for behavior),
* `DatasetSchema` describes:

  * the shape + invariants of those tables,
  * the row model.

So you can answer questions like:

> “What does plugin P read and write?”
> “What does dataset D look like, how is it validated, and where is it used?”

purely by following `CorePluginMetadata` + `DatasetSchema` – which is exactly what both humans and LLMs want.

---

## 6. Concrete, LLM-friendly “easy first step” checklist

Here’s the condensed version an AI programmer can follow:

1. **Create `DatasetSchema` and central registry**

   * [ ] Add `config/config/datasets/schema_registry.py` with `DatasetSchema` and `DATASET_SCHEMAS`.
   * [ ] Add `config/config/datasets/schema_bridge.py` with:

     * [ ] `_columndef_to_pandera`
     * [ ] `pandera_from_table_contract`
     * [ ] `build_all_dataset_schemas()` that populates `DATASET_SCHEMAS` from `DATASET_CONTRACTS`.

2. **Wire basic Pandera schemas via the bridge**

   * [ ] Call `build_all_dataset_schemas()` at config init time.
   * [ ] For a small set of key datasets (`core.goids`, `graph.call_graph_edges`, `analytics.function_metrics`), refine the auto-generated Pandera schema to add checks/invariants.

3. **Tie row models to Pandera**

   * Option A (safe):

     * [ ] Add tests that ensure row model field names match `DatasetSchema.pandera.columns`.
   * Option B (ambitious):

     * [ ] Implement `typed_dict_from_pandera` and define row models as generated from Pandera.

4. **Use `DatasetSchema` in data operations**

   * [ ] In plugins, use `DATASET_SCHEMAS[dataset_key].pandera.validate(df)` before inserts.
   * [ ] In storage/serving, use `DatasetSchema` to validate outbound DataFrames.

5. **De-duplicate over time**

   * [ ] As confidence grows, delete redundant manual row/column models and rely on:

     * dataset contracts → Pandera → row models,
     * or eventually Pandera → dataset contracts + row models, if you decide Pandera is the ultimate SSOT.

This gives you that **quick win**:

* far less manual modeling,
* explicit, central mapping from datasets → schema → row models,
* a clear path for both humans and LLMs to understand data flow and shape,
* and a foundation you can later extend into auto-generated JSON Schema, docs, or DDL.

If you want, next we can pick a specific `rows/*.py` module (say `rows/analytics.py`) and write the exact PR-style diff to convert **one** row model (e.g., `FunctionMetricsRow`) from manual to `typed_dict_from_pandera` so you can see the transformation concretely before applying it across the board.

# detailed example #

Awesome, let’s make this nice and concrete.

I’ll show you a **PR-style diff** that:

1. Adds a small `typed_dict_from_pandera` factory.
2. Wires it into the dataset schema registry.
3. Converts **one** manual row model (`FunctionMetricsRow` in `rows/analytics.py`) to be **generated from Pandera**.

You can then replicate this pattern across other `rows/*.py` modules.

---

## 1. New factory: `row_model_factory.py`

**File:** `config/config/datasets/row_model_factory.py` (new)

This file creates `TypedDict` row models from a Pandera `DataFrameSchema`.

```diff
+++ config/config/datasets/row_model_factory.py
+from __future__ import annotations
+
+from typing import Any, TypedDict, Type
+
+import pandera as pa
+from pandera import DataFrameSchema
+
+
+def _pandera_dtype_to_py_type(dtype: pa.dtypes.DataType) -> type[Any]:
+    """Best-effort mapping from Pandera/NumPy dtype to Python type.
+
+    This does not need to be perfect for the first pass; we can refine
+    it over time as we encounter more types.
+    """
+    # Pandera dtype often wraps pandas/NumPy dtypes, so we normalize via str()
+    dtype_str = str(dtype)
+
+    if "int" in dtype_str:
+        return int
+    if "float" in dtype_str or "double" in dtype_str:
+        return float
+    if "bool" in dtype_str:
+        return bool
+    if "datetime" in dtype_str:
+        # You can tighten this to datetime.datetime if you want
+        import datetime as dt
+        return dt.datetime
+
+    # Fallback: treat as string
+    return str
+
+
+def typed_dict_from_pandera(name: str, schema: DataFrameSchema) -> Type[TypedDict]:
+    """Create a TypedDict row model from a Pandera DataFrameSchema.
+
+    Each column in the schema becomes a key in the TypedDict. The type
+    is derived from the column's dtype. All fields are marked as total
+    (required); nullable semantics are handled at the Pandera layer.
+    """
+    annotations: dict[str, Any] = {}
+
+    for col_name, col in schema.columns.items():
+        py_type = _pandera_dtype_to_py_type(col.dtype)
+        annotations[col_name] = py_type
+
+    # Using the functional TypedDict form: TypedDict(name, annotations)
+    RowModel = TypedDict(name, annotations, total=True)
+    return RowModel
```

---

## 2. Ensure `DatasetSchema` is present (if not already)

If you don’t already have it from the previous plan, here’s what it looks like.

**File:** `config/config/datasets/schema_registry.py`

```diff
+++ config/config/datasets/schema_registry.py
+from __future__ import annotations
+
+from dataclasses import dataclass
+from typing import Any, Mapping, Type
+
+from pandera import DataFrameSchema
+
+
+@dataclass(frozen=True)
+class DatasetSchema:
+    """Canonical schema metadata for a dataset.
+
+    - name: dataset key (e.g., "analytics.function_metrics")
+    - pandera: DataFrameSchema describing table shape & invariants
+    - columns: mapping of column name to ColumnDef (from existing contracts)
+    - row_model: Python row type (TypedDict/dataclass), derived from Pandera
+    """
+
+    name: str
+    pandera: DataFrameSchema
+    columns: Mapping[str, Any]
+    row_model: Type[Any] | None = None
+
+
+# Global registry: dataset_key -> DatasetSchema
+DATASET_SCHEMAS: dict[str, DatasetSchema] = {}
```

And a bridge that populates it from your existing `DATASET_CONTRACTS` (simplified):

**File:** `config/config/datasets/schema_bridge.py`

```diff
+++ config/config/datasets/schema_bridge.py
+from __future__ import annotations
+
+from typing import Any
+
+import pandera as pa
+from pandera import Column, DataFrameSchema
+
+from codeintel.config.datasets.contracts import DATASET_CONTRACTS, TableContract, ColumnDef
+from codeintel.config.datasets.schema_registry import DatasetSchema, DATASET_SCHEMAS
+
+
+PY_TYPE_MAP: dict[str, type[Any]] = {
+    "TEXT": str,
+    "VARCHAR": str,
+    "INTEGER": int,
+    "BIGINT": int,
+    "UBIGINT": int,
+    "DOUBLE": float,
+    "FLOAT": float,
+    "BOOLEAN": bool,
+}
+
+
+def _columndef_to_pandera(col: ColumnDef) -> Column:
+    py_type = PY_TYPE_MAP.get(col.duckdb_type.upper(), Any)
+    return Column(
+        py_type,
+        nullable=col.nullable,
+    )
+
+
+def pandera_from_table_contract(contract: TableContract) -> DataFrameSchema:
+    columns: dict[str, Column] = {}
+    for col in contract.columns:
+        columns[col.name] = _columndef_to_pandera(col)
+    return DataFrameSchema(columns, strict=True, coerce=True)
+
+
+def build_all_dataset_schemas() -> None:
+    """Populate DATASET_SCHEMAS from DATASET_CONTRACTS and Pandera schemas."""
+    for key, contract in DATASET_CONTRACTS.items():
+        pandera_schema = pandera_from_table_contract(contract)
+        ds = DatasetSchema(
+            name=key,
+            pandera=pandera_schema,
+            columns={c.name: c for c in contract.columns},
+            row_model=None,
+        )
+        DATASET_SCHEMAS[key] = ds
```

And call `build_all_dataset_schemas()` once at config init:

**File:** `config/config/__init__.py`

```diff
@@
-from .datasets.contracts import DATASET_CONTRACTS  # existing import
+from .datasets.contracts import DATASET_CONTRACTS
+from .datasets.schema_bridge import build_all_dataset_schemas
+
+# Initialize dataset schemas at import time
+build_all_dataset_schemas()
```

(If you already had something similar, tweak accordingly; this is just to show the pattern.)

---

## 3. Convert `FunctionMetricsRow` in `rows/analytics.py`

Now the fun part: change **one** row model from manual to derived.

### 3.1 Original (manual) version

**File:** `config/config/datasets/rows/analytics.py` (hypothetical original)

```diff
-from __future__ import annotations
-
-from typing import TypedDict
-
-
-class FunctionMetricsRow(TypedDict):
-    repo: str
-    commit: str
-    rel_path: str
-    function_goid_h128: int
-    qualname: str
-    language: str
-    loc: int
-    cyclomatic_complexity: int
-    fan_in: int
-    fan_out: int
-    has_type_hints: bool
-    coverage_ratio: float | None
-
-
-__all__ = [
-    "FunctionMetricsRow",
-]
```

### 3.2 New version: derived from Pandera via `DatasetSchema`

We import the schema registry and factory, then define `FunctionMetricsRow` as a `TypedDict` generated from the Pandera schema for `analytics.function_metrics`.

```diff
+from __future__ import annotations
+
+from codeintel.config.datasets.schema_registry import DATASET_SCHEMAS
+from codeintel.config.datasets.row_model_factory import typed_dict_from_pandera
+
+
+# The FunctionMetricsRow TypedDict is derived from the Pandera schema
+# registered for the "analytics.function_metrics" dataset. This ensures
+# that:
+# - the row model stays in sync with the dataset contract
+# - column additions/removals are centrally controlled
+FunctionMetricsRow = typed_dict_from_pandera(
+    "FunctionMetricsRow",
+    DATASET_SCHEMAS["analytics.function_metrics"].pandera,
+)
+
+
+__all__ = [
+    "FunctionMetricsRow",
+]
```

That’s it: the manual field list is gone; the shape is now **owned** by the Pandera schema (which itself is derived from the dataset contract, or refined there).

---

## 4. Optional: add a small test to prove the connection (for this dataset)

To give extra confidence and make the relationship explicit for future LLMs, you can add a tiny test that asserts that `FunctionMetricsRow`’s keys match the Pandera columns:

**File:** `tests/config/test_function_metrics_row_matches_schema.py`

```diff
+++ tests/config/test_function_metrics_row_matches_schema.py
+from __future__ import annotations
+
+from typing import get_type_hints
+
+from codeintel.config.datasets.schema_registry import DATASET_SCHEMAS
+from codeintel.config.datasets.rows.analytics import FunctionMetricsRow
+
+
+def test_function_metrics_row_matches_pandera_schema() -> None:
+    ds = DATASET_SCHEMAS["analytics.function_metrics"]
+    schema_fields = set(ds.pandera.columns.keys())
+
+    hints = get_type_hints(FunctionMetricsRow)
+    row_fields = set(hints.keys())
+
+    assert row_fields == schema_fields, (
+        "FunctionMetricsRow fields differ from Pandera schema fields: "
+        f"row={sorted(row_fields)}, schema={sorted(schema_fields)}"
+    )
```

Strictly speaking, with `typed_dict_from_pandera` this should always pass, but the test serves as **documentation** and a guardrail if someone later tweaks the factory or schema.

---

## 5. What this achieves (and how to replicate)

For this one example (`FunctionMetricsRow`):

* The **dataset contract** continues to live in `TableContract` / `DATASET_CONTRACTS`.
* `build_all_dataset_schemas` derives a **Pandera DataFrameSchema** for `analytics.function_metrics`.
* `FunctionMetricsRow` is now a **TypedDict generated from that Pandera schema**, not a hand-written model:

  * If you add/remove a column in the dataset contract and adjust the Pandera schema, the row model updates automatically.
* LLMs and humans now see a **single chain of authority**:

  * `DATASET_CONTRACTS` → `DATASET_SCHEMAS` (Pandera) → `FunctionMetricsRow`.

To apply this pattern across the repo, your AI programmer can:

1. Identify manual row models in `config/config/datasets/rows/*.py`.

2. Ensure the corresponding dataset key has an entry in `DATASET_SCHEMAS` with a Pandera schema.

3. Replace the manual `class FooRow(TypedDict)` with:

   ```python
   FooRow = typed_dict_from_pandera(
       "FooRow",
       DATASET_SCHEMAS["schema.table"].pandera,
   )
   ```

4. Optionally add small tests for each dataset to keep things explicit.

That’s your proof-of-concept PR — once it lands and feels good, you can roll this across the other key datasets (`core.goids`, `graph.call_graph_edges`, etc.) with essentially the same mechanical change.
