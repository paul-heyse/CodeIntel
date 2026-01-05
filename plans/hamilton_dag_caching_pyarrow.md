
According to a document from **January 5, 2026**, Hamilton’s “format caching” (e.g., `@cache(format="parquet")`) works by looking up a **(format name, Python type) → (DataSaver, DataLoader)** pair via `search_data_adapter_registry(...)`. Your error is exactly that lookup failing for `("parquet", pyarrow.Table)`.

### Why you’re getting `KeyError: "parquet doesn't have any DataSaver supporting type <class 'pyarrow.lib.Table'>"`

The official “Using Data Adapters” registry confirms that the built-in `parquet` **DataSaver** is registered for **pandas DataFrame** and **polars DataFrame/LazyFrame** — not `pyarrow.Table`. ([Hamilton][1])

So when you do something like:

```python
from hamilton.function_modifiers import cache

@cache(format="parquet")
def some_node(...) -> pa.Table:
    ...
```

Hamilton tries to find a `parquet` saver for `pa.Table` and can’t, so it raises the KeyError (exactly what you saw). This is consistent with the design described in your internal caching doc: format caching relies on DataLoader/DataSaver under the hood and the registry has to have a saver for that output type.

---

## What “proper caching” of PyArrow tables looks like (in Hamilton)

You basically have three viable patterns. Pick based on what you optimize for.

### Option A — Easiest: cache PyArrow tables with **pickle** (not parquet)

Hamilton’s caching docs explicitly say **pickle is the default** cache format “because it can accommodate almost all Python objects,” and you only switch to `parquet/json/...` when a compatible materializer exists. ([Hamilton][2])

So: for `pyarrow.Table`, either **don’t specify** `format=` (let it pickle), or explicitly set:

```python
@cache(format="pickle")
def some_node(...) -> pa.Table:
    ...
```

Pros:

* Zero extra plumbing.
* Works immediately for `pa.Table`.

Cons:

* Less portable/inspectable than parquet.
* Cache size can be big; pickle is more opaque.

### Option B — Recommended if you want parquet without custom adapters: return **Polars** (or Pandas) instead of PyArrow

Because `parquet` savers exist for Polars & Pandas, you can keep Arrow-centric internals but return a supported “dataframe-like” type at node boundaries:

```python
import polars as pl
import pyarrow as pa
from hamilton.function_modifiers import cache

@cache(format="parquet")
def facts_df(...) -> pl.DataFrame:
    tbl: pa.Table = build_arrow_table_somehow(...)
    return pl.from_arrow(tbl)  # often zero/low-copy depending on types

def facts_table(facts_df: pl.DataFrame) -> pa.Table:
    return facts_df.to_arrow()
```

Pros:

* Uses Hamilton’s built-in `parquet` caching path.
* Avoids custom registry code.
* Still Arrow-friendly.

Cons:

* You’re standardizing the DAG on a DataFrame type (which may be a design change).

(Again, the adapter registry shows `parquet` savers for pandas DataFrame and polars DataFrame/LazyFrame, not Arrow tables.) ([Hamilton][1])

### Option C — Best long-term if you want true PyArrow-first: implement and register a **PyArrow Parquet DataSaver/DataLoader**

Hamilton’s adapter reference spells out what you implement:

* `DataSaver`: `name()`, `applicable_types()`, `save_data(...) -> Dict[str, Any]`
* `DataLoader`: `name()`, `applicable_types()`, `load_data(type_) -> (value, metadata)` ([Hamilton][3])

And the “Using Data Adapters” doc notes you must register adapters (via `registry.register_adapters(...)` or importing a module that does it) before the key/type lookup can succeed. ([Hamilton][1])

A minimal implementation sketch (you’ll tailor params for filesystem/S3, partitioning, compression, etc.):

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Dict, Tuple, Type

import pyarrow as pa
import pyarrow.parquet as pq
from hamilton.io.data_adapters import DataLoader, DataSaver

@dataclass
class PyArrowParquetSaver(DataSaver):
    path: str

    @classmethod
    def name(cls) -> str:
        return "parquet"  # or "pyarrow_parquet" if you prefer a distinct key

    @classmethod
    def applicable_types(cls) -> Collection[Type]:
        return [pa.Table]

    def save_data(self, data: pa.Table) -> Dict[str, Any]:
        pq.write_table(data, self.path)
        return {"path": self.path, "format": "parquet", "rows": data.num_rows, "cols": data.num_columns}

@dataclass
class PyArrowParquetLoader(DataLoader):
    path: str

    @classmethod
    def name(cls) -> str:
        return "parquet"

    @classmethod
    def applicable_types(cls) -> Collection[Type]:
        return [pa.Table]

    def load_data(self, type_: Type[Type]) -> Tuple[pa.Table, Dict[str, Any]]:
        table = pq.read_table(self.path)
        return table, {"path": self.path, "format": "parquet", "rows": table.num_rows, "cols": table.num_columns}
```

Then register them (exact call shape depends on your Hamilton version, but the docs’ point is: **register adapters before using the key**). ([Hamilton][1])

Pros:

* You keep `pyarrow.Table` as the node output type.
* You get parquet caching (portable, inspectable).

Cons / gotchas:

* You now own adapter maintenance across Hamilton upgrades.
* **Eviction is imperfect for parquet caches**: Hamilton’s caching-logic docs note that today it writes the `.parquet` via a DataSaver and pickles a DataLoader for later retrieval, and the stores are “unaware” of the parquet files (making eviction harder). ([Hamilton][4])
  For “facts tables” that can be huge, you’ll want a cleanup policy (by run id / LRU / time-based).

---

## What I’d do in a CodeIntel-style DAG

Your internal cache policy guidance already pushes “large tabular facts” toward **DEFAULT + parquet** format. The missing piece is: Hamilton’s built-in parquet adapters aren’t for Arrow tables.

So:

* If you want **fastest path**: use **Option B** (Polars) for cached “facts tables”.
* If you want **PyArrow as the canonical DAG type**: implement **Option C** once, then standardize your `@cache(format="parquet")` on `pa.Table` outputs.

And if you’re trying to keep memory bounded, do what you already started doing with streaming — but make sure the **node output** that reaches caching is *not* a stream handle (record batch readers), and is something you can persist/load deterministically (parquet on disk, or a dataframe/table).

If you paste the snippet where you decorate the node (the `@cache(...)` line + return annotation) and how you configure `Builder.with_cache(...)`, I can tell you which option is the smallest code change while keeping your Arrow streaming design intact.

[1]: https://hamilton.apache.org/reference/io/available-data-adapters/ "Using Data Adapters - Hamilton"
[2]: https://hamilton.apache.org/concepts/caching/ "Caching - Hamilton"
[3]: https://hamilton.apache.org/reference/io/adapter-documentation/ "Data Adapters - Hamilton"
[4]: https://hamilton.apache.org/reference/caching/caching-logic/ "Caching logic - Hamilton"
