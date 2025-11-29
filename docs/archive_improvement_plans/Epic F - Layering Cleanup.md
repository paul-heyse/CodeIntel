You’re in a really good place to do this one now – the dataset + docs layers are solid, so this “core types” refactor can be quite clean.

I’ll structure this as:

1. What we’re aiming for (layering rules).
2. Exact new module to add: `codeintel.core.types`.
3. Moving the SCIP + pytest JSON types there.
4. Rewiring graphs/ingestion/serving to use it.
5. Fixing the remaining cross-layer leak (`storage.catalog` → `serving.http.datasets`).
6. Optional re-exports and layering checks.

All of this is based on the *current* zips you just uploaded.

---

## 1. Target layering (what “good” looks like)

We’ll enforce (by convention, and optionally by tests):

* **Inner/core**

  * `codeintel.config.primitives`
  * `codeintel.core.types` (new)
* **Middle**

  * `codeintel.ingestion`
  * `codeintel.storage`
  * `codeintel.graphs`
  * `codeintel.analytics`
* **Outer**

  * `codeintel.pipeline`
  * `codeintel.serving`
  * `codeintel.cli`

Rules we want:

* **Middle packages (ingestion/graphs/storage/analytics)** must **not import `codeintel.serving.*`**.
* Tests are allowed to, but they live under their own test modules.
* `serving` may depend on everything; `pipeline` may depend on everything; `cli` may depend on everything.
* `core.types` must be dependency-light: only `typing`, `pathlib` at most – no Pydantic, no FastAPI, no DuckDB.

Currently, the offenders are:

* `graphs/graphs/symbol_uses.py` → `codeintel.serving.protocols.ScipDocument`
* `ingestion/ingestion/tests_ingest.py` → `codeintel.serving.protocols.PytestTestEntry`
* `storage/storage/catalog.py` → `codeintel.serving.http.datasets.describe_dataset` (and `codeintel.pipeline.export.manifest`)

We’ll fix the first two via `core.types`, and the last one by removing dependency on serving.

---

## 2. Add `codeintel.core.types`

### 2.1. New package structure

Create a new top-level directory parallel to `analytics`, `config`, etc.:

* `core/`

  * `core/__init__.py`
  * `core/types.py`

So that import paths look like:

```python
from codeintel.core.types import ScipDocument, ScipOccurrence, PytestTestEntry, PytestCallEntry
```

### 2.2. `core/core/__init__.py`

Simple, just making it a package:

```python
# core/core/__init__.py
from __future__ import annotations

# Optionally re-export commonly used types here later.
```

### 2.3. `core/core/types.py`

**File:** `core/core/types.py`

These definitions come directly from the current `serving/serving/protocols.py`, but we keep only the pure domain types here.

```python
# core/core/types.py
from __future__ import annotations

from typing import TypedDict


# ---------------------------------------------------------------------------
# Pytest JSON report types
# ---------------------------------------------------------------------------


class PytestCallEntry(TypedDict, total=False):
    """Subset of pytest-json-report call data."""

    duration: float


class PytestTestEntry(TypedDict, total=False):
    """Shape of a pytest-json-report test object."""

    nodeid: str
    outcome: str
    keywords: list[str]
    # Optional fields used by ingestion
    duration: float | None
    call: PytestCallEntry | None


# ---------------------------------------------------------------------------
# SCIP JSON types
# ---------------------------------------------------------------------------


class ScipRange(TypedDict, total=False):
    """SCIP range structure (0-based line/column indices)."""

    start_line: int
    start_character: int
    end_line: int
    end_character: int


class ScipOccurrence(TypedDict, total=False):
    """Occurrence entry within a SCIP document."""

    range: ScipRange
    symbol: str
    symbol_roles: int | None


class ScipDocument(TypedDict, total=False):
    """SCIP JSON document emitted by scip-python."""

    relative_path: str
    occurrences: list[ScipOccurrence]
```

> We **do not** add `HasModelDump` here – that’s serving/pydantic-specific and stays in `serving.protocols`.

If you want, you can later add more domain-level TypedDicts (e.g. for your coverage JSON) here as well, but we’ll start with the ones that currently cause cross-layer imports.

---

## 3. Slim down `serving/serving/protocols.py`

Now that the domain types live in `core.types`, `serving/serving/protocols.py` should just re-use them and define only serving-specific protocols.

**File:** `serving/serving/protocols.py`

### 3.1. New imports

Replace:

```python
from typing import Protocol, TypedDict
```

with:

```python
from typing import Protocol

from codeintel.core.types import (
    PytestCallEntry,
    PytestTestEntry,
    ScipDocument,
    ScipOccurrence,
    ScipRange,
)
```

### 3.2. Remove TypedDict definitions from this file

Delete the classes:

* `PytestCallEntry`
* `PytestTestEntry`
* `ScipRange`
* `ScipOccurrence`
* `ScipDocument`

and leave only:

```python
"""Shared typed protocols to reduce use of ``Any`` in serving layer."""

from __future__ import annotations

from typing import Protocol

from codeintel.core.types import (
    PytestCallEntry,
    PytestTestEntry,
    ScipDocument,
    ScipOccurrence,
    ScipRange,
)


class HasModelDump(Protocol):
    """Protocol for Pydantic models used in MCP responses."""

    def model_dump(self) -> dict[str, object]:
        """Return a dictionary representation."""
        ...
```

Everything in serving that previously used those TypedDicts can continue to import them from `serving.protocols`, or you can progressively refactor serving code to import directly from `core.types` if you like.

---

## 4. Rewire graphs & ingestion to use `core.types` instead of `serving.protocols`

### 4.1. `graphs/graphs/symbol_uses.py`

Currently:

```python
from codeintel.ingestion.common import run_batch
from codeintel.serving.protocols import ScipDocument
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.rows import SymbolUseRow, symbol_use_to_tuple
```

Change to:

```python
from codeintel.ingestion.common import run_batch
from codeintel.core.types import ScipDocument
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.rows import SymbolUseRow, symbol_use_to_tuple
```

No other changes needed – the shape of `ScipDocument` has not changed, just its import path.

### 4.2. `ingestion/ingestion/tests_ingest.py`

Currently:

```python
from codeintel.ingestion.tool_service import ToolService
from codeintel.serving.protocols import PytestTestEntry
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.rows import TestCatalogRowModel, serialize_test_catalog_row
```

Change to:

```python
from codeintel.ingestion.tool_service import ToolService
from codeintel.core.types import PytestTestEntry
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.rows import TestCatalogRowModel, serialize_test_catalog_row
```

Everything else continues to work as-is; the ingestion code still treats those entries as `Mapping[str, Any]` / `TypedDict` values.

---

## 5. Clean up storage → serving leak in `storage/storage/catalog.py`

This isn’t strictly “core types”, but it’s part of the same layering clean-up.

Right now:

**File:** `storage/storage/catalog.py`

```python
from codeintel.pipeline.export.manifest import compute_file_hash
from codeintel.serving.http.datasets import describe_dataset
from codeintel.storage.datasets import Dataset, DatasetRegistry
...
```

This makes `storage` depend on both `pipeline` and `serving`, which breaks the desired layering.

### 5.1. Move the human-readable description logic into storage

We already have a generic `describe_dataset(ds: Dataset) -> dict` in `storage.storage.datasets`. For catalog generation we only need a **string** description, and that’s currently defined in `serving.http.datasets.describe_dataset`.

Copy that logic into `storage.storage.catalog` and stop importing from serving:

```python
# storage/storage/catalog.py
from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.storage.datasets import Dataset, DatasetRegistry
...

def describe_dataset_for_catalog(ds: Dataset) -> str:
    """
    Produce a human-friendly description for a dataset/table.

    Returns
    -------
    str
        Description string including a column preview when available.
    """
    schema = TABLE_SCHEMAS.get(ds.table_key)
    if schema is None:
        return f"DuckDB table/view {ds.table_key}"
    PREVIEW_COLUMN_COUNT = 5
    column_names = ", ".join(col.name for col in schema.columns[:PREVIEW_COLUMN_COUNT])
    extra = "" if len(schema.columns) <= PREVIEW_COLUMN_COUNT else "..."
    return f"{ds.name}: {ds.table_key} ({column_names}{extra})"
```

Then swap the import and usage:

**Before:**

```python
from codeintel.serving.http.datasets import describe_dataset
...
description=describe_dataset(ds.name, ds.table_key),
```

**After:**

```python
from codeintel.config.schemas.tables import TABLE_SCHEMAS
...
description=describe_dataset_for_catalog(ds),
```

Now `storage.catalog` depends only on config+storage+pipeline (for `compute_file_hash`), not serving. If you want to be strict and avoid even pipeline here, you could optionally:

* Move `compute_file_hash` into a small “shared utils” module, or
* Reimplement the little hash logic inline in catalog.

But that’s optional; the big leak is the `serving` import, which we just removed.

---

## 6. Optional: re-export core config primitives from `core.types`

If you’d like a single place for “core domain types”, you can add **aliases** (not moves!) for a few very central dataclasses:

* `SnapshotRef`
* `GraphBackendConfig`
* `GraphFeatureFlags`

**File:** `core/core/types.py`

At the bottom:

```python
from codeintel.config.primitives import (
    SnapshotRef,
    GraphBackendConfig,
    GraphFeatureFlags,
)

__all__ = [
    # TypedDicts
    "PytestCallEntry",
    "PytestTestEntry",
    "ScipRange",
    "ScipOccurrence",
    "ScipDocument",
    # Config primitives
    "SnapshotRef",
    "GraphBackendConfig",
    "GraphFeatureFlags",
]
```

Then middle-layer code that wants to talk about core domain concepts can either import from:

* `codeintel.config.primitives` (for full config semantics), or
* `codeintel.core.types` (for high-level types).

This is purely ergonomic – it doesn’t change layering.

---

## 7. Optional: add a simple layering test

To keep this from regressing, you can add a tiny test that asserts only allowed packages import serving:

**File:** `tests/test_layering_serving_imports.py` (or similar)

```python
from __future__ import annotations

import ast
import pathlib


ALLOWED_SERVING_IMPORTERS = {
    "cli",
    "pipeline",
    "serving",
}


def test_no_serving_imports_in_middle_packages() -> None:
    """
    Ensure analytics/graphs/ingestion/storage do not import codeintel.serving.*.
    """
    root = pathlib.Path(__file__).resolve().parents[2]  # adjust to repo root
    bad_imports: list[tuple[str, str]] = []

    for py in root.rglob("*.py"):
        rel = py.relative_to(root).as_posix()
        top_level = rel.split("/", 1)[0]
        if top_level in ALLOWED_SERVING_IMPORTERS:
            continue

        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("codeintel.serving"):
                bad_imports.append((rel, f"from {node.module} import ..."))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("codeintel.serving"):
                        bad_imports.append((rel, f"import {alias.name}"))

    assert not bad_imports, f"Disallowed imports of codeintel.serving: {bad_imports}"
```

This will catch any future cross-layer leaks.

---

## Implementation order (for you / the agent)

1. **Create the new `core/core/types.py` and `core/core/__init__.py`** with the SCIP + pytest TypedDicts.
2. **Slim `serving/serving/protocols.py`**:

   * Import types from `codeintel.core.types`.
   * Remove TypedDict definitions from that file, keep `HasModelDump`.
3. **Update imports** in:

   * `graphs/graphs/symbol_uses.py` → use `codeintel.core.types.ScipDocument`.
   * `ingestion/ingestion/tests_ingest.py` → use `codeintel.core.types.PytestTestEntry`.
4. **Fix `storage/storage/catalog.py`**:

   * Inline a `describe_dataset_for_catalog(ds)` using `TABLE_SCHEMAS`.
   * Drop `codeintel.serving.http.datasets.describe_dataset` import.
5. **(Optional) Re-export config primitives** in `core.types`.
6. **(Optional) Add layering test** to enforce “no serving imports in middle packages”.

Once that’s done, your layering will be much cleaner:

* Pure domain shapes like SCIP docs and pytest JSON are defined in `codeintel.core.types`.
* Middle layers (ingestion, graphs, storage) depend only on `core` and `config`, not on serving.
* Serving remains the outer edge, using those core types and exposing HTTP/MCP protocols on top.
