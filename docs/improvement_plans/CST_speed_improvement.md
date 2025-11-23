The short version: your CST pass is slow mostly because **LibCST’s metadata wrapper is expensive**, not because of DuckDB or your own logic. There *are* some pretty clean speedups you can get **without caching**, mainly:

1. Tell `MetadataWrapper` **not to deep-copy** the tree
2. Make per-node work cheaper (snippets, allocations)
3. Optionally parallelize across modules

I’ll walk through those in order of “bang for buck”.

---

## 1. Skip LibCST’s deep copy (`unsafe_skip_copy=True`)

Right now for each module you do: 

```python
wrapper = metadata.MetadataWrapper(cst.parse_module(source))
visitor = CstVisitor(
    rel_path=record.rel_path,
    module_name=record.module_name,
    source=source,
)
wrapper.visit(visitor)
```

By default, `MetadataWrapper` makes a **full copy** of the CST before attaching metadata to it. That’s great for safety, but you never mutate the tree; you just read metadata and walk it. So you can safely skip that copy:

```python
wrapper = metadata.MetadataWrapper(
    cst.parse_module(source),
    unsafe_skip_copy=True,  # <--- add this
)
visitor = CstVisitor(
    rel_path=record.rel_path,
    module_name=record.module_name,
    source=source,
)
wrapper.visit(visitor)
```

This change:

* Keeps your `CstVisitor.METADATA_DEPENDENCIES = (metadata.PositionProvider,)` behavior intact
* Usually knocks a **big chunk** off runtime for non-trivial modules (copying the whole tree + metadata is expensive)

This is the first thing I’d try. It’s a one-line change, zero change to schema, and safe for read-only analysis.

---

## 2. Make per-node work cheaper

From your `cst_nodes.jsonl` snapshot, you currently have:

* **61 Python files**
* **26,387 recorded CST nodes**
  – ~17,700 are `Name` nodes
  – ~2,600 `Attribute`, ~2,400 `Call`, etc.

So you’re doing your per-node work ~26k times per run, and names dominate the count.

### 2.1 Avoid extra work for every visit

Right now `on_visit` looks like (simplified): 

```python
def on_visit(self, node: cst.CSTNode) -> bool:
    self._parent_kinds.append(type(node).__name__)
    if isinstance(node, (cst.ClassDef, cst.FunctionDef, ASYNC_FUNC_DEF)):
        ...
    self._record_cst_row(node)
    return True
```

And `_record_cst_row` recomputes `kind = type(node).__name__` again.

You can shave a bit of overhead and attribute churn by computing `kind` once:

```python
def on_visit(self, node: cst.CSTNode) -> bool:
    kind = type(node).__name__
    self._parent_kinds.append(kind)

    if isinstance(node, (cst.ClassDef, cst.FunctionDef, ASYNC_FUNC_DEF)):
        name_node = getattr(node, "name", None)
        if name_node is not None and hasattr(name_node, "value"):
            self._scope_stack.append(name_node.value)

    self._record_cst_row(node, kind)
    return True

def _record_cst_row(self, node: cst.CSTNode, kind: str) -> None:
    if not self._should_capture(node):
        return
    ...
```

This is a micro-optimization, but combined with others it helps.

### 2.2 Faster `text_preview` extraction

Right now you do:

```python
self.source_lines = source.splitlines(keepends=True)
...
if start.line == end.line:
    line = self.source_lines[start.line - 1]
    snippet = line[start.column : end.column]
else:
    lines = list(self.source_lines[start.line - 1 : end.line])
    ...
    snippet = "".join(lines)
```

for **almost every recorded node** (26k times) and then slice `snippet[:200]`.

You can make this cheaper by:

1. Precomputing line offsets once
2. Slicing the **original string** directly with a single `source[start:end]`

In `CstVisitor.__init__`:

```python
class CstVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self, rel_path: str, module_name: str, source: str) -> None:
        self.rel_path = rel_path
        self.module_name = module_name
        self.source = source

        # Precompute line-start offsets: line number -> char index into source
        self._line_offsets: list[int] = []
        offset = 0
        for line in source.splitlines(keepends=True):
            self._line_offsets.append(offset)
            offset += len(line)

        self.cst_rows: list[CstRow] = []
        self._seen_ids: set[str] = set()
        self._scope_stack: list[str] = []
        self._parent_kinds: list[str] = []
```

And then in `_record_cst_row`:

```python
pos = self.get_metadata(metadata.PositionProvider, node)
start = pos.start
end = pos.end

span = {
    "start": [start.line, start.column],
    "end": [end.line, end.column],
}

try:
    start_idx = self._line_offsets[start.line - 1] + start.column
    end_idx = self._line_offsets[end.line - 1] + end.column
    snippet = self.source[start_idx:end_idx]
except (IndexError, ValueError):
    snippet = ""

text_preview = snippet[:200]
```

This avoids:

* allocating `lines = list(...)`
* multiple string slices + joins per node

It’s not as big as the LibCST copy change, but it removes a lot of Python-level overhead.

### 2.3 Reduce the amount of data you record (if acceptable)

In your snapshot, `Name` nodes alone are ~67% of all rows. For each of them you compute:

* A `span` dict
* A 20-ish char `text_preview`
* `parents` list
* `qnames` list

If you don’t actually need *all* of that for `Name` and `Attribute` nodes, there are easy wins:

* **Option A** – keep the rows but make them cheaper:

  * For `Name` nodes, skip full text slice and just use `node.value`:

    ```python
    if isinstance(node, cst.Name):
        snippet = node.value  # usually same as existing preview
    else:
        # use the more expensive slicing logic above
    ```

* **Option B** – stop recording `Name` nodes entirely:

  * Change `_should_capture` to drop `cst.Name` (and maybe `cst.Attribute` if you can live without it).

That would shrink your `cst_nodes` table from ~26k rows to ~8–9k, which cuts down:

* per-node visitor work
* memory usage
* DuckDB insert time

Obviously this depends on how you’re using `cst_nodes` later, but if most queries are around structural constructs (functions, classes, assignments, control flow, calls), dropping leaf-y nodes can be a very material speedup.

---

## 3. Parallelize across modules (no DB contention)

Your ingestion loop is completely module-local until the final `_flush_batch`:

```python
for record in iter_modules(...):
    source = read_module_source(...)
    ...
    wrapper = metadata.MetadataWrapper(cst.parse_module(source))
    visitor = CstVisitor(...)
    wrapper.visit(visitor)
    cst_values.extend([... for row in visitor.cst_rows])
...
_flush_batch(con, cst_values)
```

Every module can be parsed and walked **independently**; the only shared state is the final `cst_values` list and the DuckDB connection.

So a natural pattern is:

* Use a **process pool** to turn each `ModuleRecord` into a list of rows (pure Python data)
* Accumulate results in the parent process
* Call `run_batch` / `_flush_batch` once, as you do now

Conceptual sketch (not drop-in, but close):

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

def _process_module(record: ModuleRecord, repo_root: Path) -> list[list[object]]:
    source = read_module_source(record, logger=None)
    if source is None:
        return []
    try:
        wrapper = metadata.MetadataWrapper(
            cst.parse_module(source),
            unsafe_skip_copy=True,
        )
        visitor = CstVisitor(
            rel_path=record.rel_path,
            module_name=record.module_name,
            source=source,
        )
        wrapper.visit(visitor)
    except Exception:
        # log in parent later if you want
        return []

    return [
        [
            row.path,
            row.node_id,
            row.kind,
            row.span,
            row.text_preview,
            row.parents,
            row.qnames,
        ]
        for row in visitor.cst_rows
    ]
```

Then in `ingest_cst`:

```python
records = list(iter_modules(...))

cst_values: list[list[object]] = []
with ProcessPoolExecutor() as pool:
    futures = {pool.submit(_process_module, r, repo_root): r for r in records}
    for fut in as_completed(futures):
        rows = fut.result()
        if rows:
            cst_values.extend(rows)

_flush_batch(con, cst_values)
```

* Only the **parent** process ever touches DuckDB, so you don’t hit DB locks.
* On a 4–8 core machine, this can cut wall-clock time by a large factor, since LibCST parsing & metadata passes are CPU bound.

(If you want to stay simpler, you can also use `ThreadPoolExecutor` and see if LibCST releases the GIL enough to help; `ProcessPoolExecutor` is the safer bet for real speedup.)

---

## 4. Minor structural cleanups (small but easy)

These are smaller, but basically free:

* **Avoid the intermediate `CstRow` objects**:

  * Instead of storing dataclass instances inside `CstVisitor.cst_rows` and then mapping them to lists, you can have the visitor directly append the final row arrays. That cuts ~26k object allocations for this repo.
  * Or at least give `CstRow` a `__slots__` to reduce memory overhead.

* **Batch flushing in chunks**:

  * Right now you accumulate everything and flush once. For larger repos, you may want to flush every N rows (e.g. every 10k) to keep peak RAM lower.
  * This isn’t really your current bottleneck (26k rows is tiny for DuckDB), but it will matter as repo sizes grow.

---

## 5. What I’d actually do in your shoes

If you want to keep the current behavior and schema, but make it faster **without caching**, I’d implement in this order:

1. **Add `unsafe_skip_copy=True` to `MetadataWrapper`**
   – Almost no risk, large likely speedup.

2. **Switch `text_preview` extraction to single-slice via precomputed line offsets**
   – Keeps exactly the same semantics, less Python churn.

3. **Decide if you truly need `Name` (and maybe `Attribute`) rows**
   – If not, drop them from `_should_capture`; if yes, consider cheaper snippet logic for them.

4. **Add module-level parallelism** if you still want further speedups
   – Keep DuckDB writes in the parent process, workers produce pure Python data.

Do 1 + 2 first and time the change; my guess is you’ll see a very noticeable improvement even before going to parallel processing. If you’d like, I can sketch a more concrete patch that applies these changes directly to your current `cst_extract.py` file.
