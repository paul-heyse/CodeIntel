Short answer: yes, it’s absolutely reasonable to move to **per‑module, incremental AST/CST/SCIP** with parallelism, and your current code is actually very close to what you need. Below is a concrete implementation plan wired into the files you shared.

I’ll break it into:

1. Incremental updates for CST, AST, SCIP
2. Multiprocessing / parallel runners for each
3. A shared in‑memory “parsed module” layer so trees are only built once per refresh
4. Extra best practices / gotchas (including SCIP sharding)

---

## 1. Incremental CST / AST / SCIP

### 1.1. Store per‑file digests

Right now:

* `repo_scan.ingest_repo` enumerates modules and writes `core.modules` (path, module, repo, commit, tags, owners).
* Ingestion phases (AST, CST, typing, SCIP) all re‑scan `core.modules` and re‑read every file, with `run_batch(..., delete_params=[repo, commit])` wiping tables per repo+commit.

Add a new “file state” table:

```sql
CREATE TABLE core.file_state (
  repo        TEXT,
  commit      TEXT,
  rel_path    TEXT,
  language    TEXT,
  size_bytes  BIGINT,
  mtime_ns    BIGINT,
  content_hash TEXT,
  PRIMARY KEY (repo, commit, rel_path, language)
);
```

Then extend `repo_scan.ingest_repo` to compute and upsert a digest per module:

```python
def _file_digest(path: Path) -> tuple[int, int, str]:
    st = path.stat()
    size = st.st_size
    mtime_ns = st.st_mtime_ns
    # Optional: only hash when (size, mtime) changed vs previous row
    data = path.read_bytes()
    h = hashlib.blake2b(data, digest_size=16).hexdigest()
    return size, mtime_ns, h
```

Inside the repo scan loop:

```python
size, mtime_ns, h = _file_digest(path)
file_state_rows.append([
    cfg.repo, cfg.commit, rel_path, "python",
    size, mtime_ns, h,
])
```

After building `modules`, call `run_batch(gateway, "core.file_state", file_state_rows, delete_params=[cfg.repo, cfg.commit])`.

This gives every ingestion phase a cheap way to know “which files actually changed since last run”.

### 1.2. Generic “changed modules” helper

Add a small helper in `ingestion.common` that all phases can reuse:

```python
@dataclass(frozen=True)
class ChangeSet:
    added: list[ModuleRecord]
    modified: list[ModuleRecord]
    deleted: list[ModuleRecord]

def compute_changes(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    repo_root: Path,
    *,
    language: str = "python",
    scan_config: ScanConfig | None = None,
) -> ChangeSet:
    con = gateway.con

    # 1) Current module -> hash
    module_map = load_module_map(gateway, repo, commit, language=language)
    current = {}
    for rec in iter_modules(module_map, repo_root, scan_config=scan_config):
        size, mtime_ns, h = _file_digest(rec.file_path)
        current[rec.rel_path] = (rec, size, mtime_ns, h)

    # 2) Previous file_state snapshot
    rows = con.execute(
        """
        SELECT rel_path, size_bytes, mtime_ns, content_hash
        FROM core.file_state
        WHERE repo = ? AND commit = ? AND language = ?
        """,
        [repo, commit, language],
    ).fetchall()
    previous = {str(rel): (size, mtime, h) for rel, size, mtime, h in rows}

    added, modified = [], []
    for rel_path, (rec, size, mtime_ns, h) in current.items():
        old = previous.get(rel_path)
        if old is None:
            added.append(rec)
        elif old != (size, mtime_ns, h):
            modified.append(rec)

    deleted = []
    for rel_path in previous.keys() - current.keys():
        file_path = repo_root / rel_path
        deleted.append(
            ModuleRecord(
                rel_path=rel_path,
                module_name=module_map.get(rel_path, "<deleted>"),
                file_path=file_path,
                index=0,
                total=0,
            )
        )

    # Update core.file_state with the new snapshot so the next run sees it
    file_state_rows = [
        [repo, commit, rel, language, size, mtime_ns, h]
        for rel, (_, size, mtime_ns, h) in current.items()
    ]
    run_batch(
        gateway,
        "core.file_state",
        file_state_rows,
        delete_params=[repo, commit],  # single commit snapshot
    )

    return ChangeSet(added=added, modified=modified, deleted=deleted)
```

Now each ingestion phase just asks for `compute_changes(...)` and only touches those modules.

---

### 1.3. Incremental CST

Current CST ingest:

* Loads *all* modules from `core.modules`.
* Clears `core.cst_nodes` for the repo+commit via `run_batch(..., delete_params=[repo, commit])`.
* Uses a `ThreadPoolExecutor` over `_process_module(record)` and flushes rows in large batches.

Incremental variant:

1. Replace the initial full delete with **per‑file deletes** for changed/deleted modules:

```python
def _delete_cst_for_paths(gateway: StorageGateway, repo: str, commit: str, rel_paths: list[str]) -> None:
    if not rel_paths:
        return
    gateway.con.execute(
        """
        DELETE FROM core.cst_nodes
        WHERE rel_path IN (SELECT * FROM UNNEST(?))
          AND repo = ?
          AND commit = ?
        """,
        [rel_paths, repo, commit],
    )
```

2. Use `compute_changes` in `ingest_cst`:

```python
changes = compute_changes(
    gateway, repo, commit, repo_root, language="python", scan_config=scan_config
)

to_reparse = changes.added + changes.modified
_delete_cst_for_paths(gateway, repo, commit, [m.rel_path for m in to_reparse + changes.deleted])
```

3. Only submit `to_reparse` into the thread/process pool; keep batching exactly as before. Rows for unchanged modules remain in `core.cst_nodes`.

This gives you **per‑module CST updates** with minimal change to the current ingestion shape.

---

### 1.4. Incremental AST

Current AST ingest:

* Loads module map, iterates with `iter_modules`, parses each file with `ast.parse`, collects rows and metrics, then calls `run_batch` once per table with `delete_params=[repo, commit]` wiping everything.

Update it like CST:

1. Add per‑file delete helpers for `core.ast_nodes` and `core.ast_metrics`.

2. Use `compute_changes` to get `added/modified/deleted`.

3. For `deleted` modules, delete their rows:

```python
_delete_ast_for_paths(gateway, cfg.repo, cfg.commit, [m.rel_path for m in changes.deleted])
```

4. For `to_reparse = added + modified`, do exactly what `_collect_module_ast` does now, but only for those modules (see multiprocessing section for how).

5. Insert with `run_batch(..., delete_params=None)` — i.e., **no global truncate**, you’ve already cleared per‑file rows.

You said recomputing metrics is cheap relative to ingest; if that holds you can also “cheat” and still recompute metrics for all files from the updated `core.ast_nodes`. But wiring the per‑file delete + insert is not much harder, so I’d still do it here.

---

### 1.5. Incremental SCIP

Current SCIP ingest:

* Runs one `scip-python index` over the repo (or repo/src) to produce a single `index.scip`.
* Runs a single `scip print --json` to produce `index.scip.json`.
* Registers a DuckDB view over that JSON and backfills `core.goid_crosswalk.scip_symbol`.

You want: “SCIP per module” and concurrency.

**Feasibility**: `scip-python index . --target-only=src/my_module.py` is designed exactly for this; you can safely run multiple processes in parallel as long as:

* Each writes to a different output path.
* You later merge outputs into a single logical index.

Implementation plan:

1. **Per‑file SCIP shard format**

   Decide on a layout, e.g.:

   ```text
   build/scip_shards/
       src/package/foo.py.scip
       src/package/bar.py.scip
   build/scip_shards_json/
       src/package/foo.py.json
       src/package/bar.py.json
   ```

2. **Incremental SCIP runner**

   Add a new function in `scip_ingest.py` alongside `_run_scip_python` that processes a subset of paths:

   ```python
   def index_paths_incremental(
       binary: str,
       repo_root: Path,
       targets: list[Path],
       output_dir: Path,
       max_workers: int,
   ) -> list[Path]:
       output_dir.mkdir(parents=True, exist_ok=True)

       def _one(path: Path) -> Path | None:
           rel = path.relative_to(repo_root)
           out = output_dir / (str(rel).replace("/", "_") + ".scip")
           code, stdout, stderr = _run_command(
               [binary, "index", str(repo_root), "--target-only", str(rel),
                "--output", str(out)],
               cwd=repo_root,
           )
           if code != 0:
               log.warning("scip-python index failed for %s: %s", rel, stderr.strip() or stdout.strip())
               return None
           return out

       with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
           results = list(pool.map(_one, targets))

       return [r for r in results if r is not None]
   ```

3. **Merge SCIP shards to JSON**

   For each `.scip` shard:

   * Run `_run_scip_print` into a per‑file JSON in `scip_shards_json/`.
   * Each JSON will contain one or more `documents` entries for that file.

   Then build (or update) the global `index.scip.json` as:

   ```python
   def rebuild_index_json(shard_json_dir: Path, index_json: Path) -> None:
       docs: list[dict[str, object]] = []
       for path in shard_json_dir.glob("*.json"):
           payload = json.loads(path.read_text("utf8"))
           if isinstance(payload, dict) and isinstance(payload.get("documents"), list):
               docs.extend(payload["documents"])
           elif isinstance(payload, list):
               docs.extend(payload)
       index_json.write_text(json.dumps({"documents": docs}), encoding="utf8")
   ```

   For incremental updates, you don’t even need to re‑read all shards: you can:

   * Load the existing index JSON.
   * Filter out `documents` matching `relative_path` in the changed modules.
   * Append `documents` from updated shard JSONs.

   But first iteration: just rebuild index.json from all shards — it’s cheap compared to running scip‑python.

4. **Limit to changed modules**

   Use `compute_changes` again to get `added + modified` modules and pass just their `file_path`s into `index_paths_incremental`.

   For `deleted` modules:

   * Remove the corresponding `.scip` and `.json` shard.
   * Remove their `documents` from the global JSON as described above.

5. **Update crosswalk only for changed paths**

   `_update_scip_symbols` currently builds a `def_map` for *all* documents and joins it to *all* GOIDs.

   For incremental:

   * After rebuilding `index.scip.json`, load **only the documents for changed paths** into a small `docs` list.
   * Build `def_map` just for them.
   * Before inserting new `scip_symbol` values for those GOIDs, clear the old ones:

   ```python
   con.execute(
       """
       UPDATE core.goid_crosswalk
       SET scip_symbol = NULL
       WHERE repo = ? AND commit = ?
         AND file_path IN (SELECT * FROM UNNEST(?))
       """,
       [cfg.repo, cfg.commit, changed_rel_paths],
   )
   ```

   Then run `_build_symbol_updates` / `executemany` only for updated GOIDs.

6. **Concurrency**

   Because each `scip-python index` call is a separate process, you can safely run several at once. I’d cap `max_workers` to something like `min(4, os.cpu_count())` at first and tune empirically — scip‑python can be CPU‑heavy on its own.

---

## 2. Multiprocessing: AST, CST, and SCIP

### 2.1. AST: use `ProcessPoolExecutor`

`ingest_python_ast` currently loops serially:

```python
for record in iter_modules(...):
    module_data = _collect_module_ast(record)
    ...
```

You already have `_collect_module_ast(record)` as a pure function which returns `(rows, metrics) | None`. Perfect for a process pool.

Change to:

```python
from concurrent.futures import ProcessPoolExecutor

def ingest_python_ast(...):
    ...
    records = list(iter_modules(module_map, repo_root, ...))

    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        for module_data in pool.map(_collect_module_ast, records):
            if module_data is None:
                continue
            ast_rows, metrics = module_data
            ast_values.extend(ast_rows)
            metric_values.append([...])
```

When combined with the incremental `records = changes.added + changes.modified` described earlier, this gives you true **CPU‑parallel AST parsing**.

### 2.2. CST: switch to processes (optionally)

CST ingestion already uses `ThreadPoolExecutor`.

Because LibCST is pure Python, threads won’t scale beyond 1 core much due to the GIL. You can:

* Add a flag to choose backend:

  ```python
  from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

  def _executor(kind: str, max_workers: int):
      if kind == "process":
          return ProcessPoolExecutor(max_workers=max_workers)
      return ThreadPoolExecutor(max_workers=max_workers)
  ```

* Use `CODEINTEL_CST_EXECUTOR` env var:

  ```python
  exec_kind = os.getenv("CODEINTEL_CST_EXECUTOR", "thread")
  with _executor(exec_kind, worker_count) as pool:
      for result in pool.map(_process_module, records):
          ...
  ```

`ModuleRecord` and `ModuleResult` are dataclasses with simple fields; `CstVisitor` and `CstCaptureConfig` live at top level of the module; so pickling should Just Work™.

Combine this with incremental `records = changes.added + changes.modified` and you’ve got scalable CST ingestion.

### 2.3. SCIP: concurrent shards

Covered above in 1.5:

* Run multiple `scip-python index` processes via a process pool (or `asyncio.gather` with a concurrency semaphore).
* Each process writes its own `.scip` file; you then run `scip print` in parallel as well if needed.

Because each `index` invocation is separate, there’s no shared state inside scip‑python you need to worry about; only ensure they don’t share an output file.

---

## 3. Shared parsed‑module class structure

You also want: “a class structure for storing these parsings and how it should be used to ensure we just parse once per refresh, and then not at all for downstream calculations.”

Here’s a concrete design that fits into your existing `IngestionContext`.

### 3.1. Core dataclasses

```python
@dataclass
class ModuleSnapshot:
    rel_path: str
    module_name: str
    file_path: Path
    size_bytes: int
    mtime_ns: int
    content_hash: str

@dataclass
class ParsedModule:
    snapshot: ModuleSnapshot
    source: str
    ast_tree: ast.AST | None = None
    cst_module: cst.Module | None = None
    # Optional: precomputed rows to avoid recomputing in each phase
    ast_rows: list[list[object]] | None = None
    ast_metrics: AstMetrics | None = None
    cst_rows: list[Row] | None = None  # Row from cst_extract
```

### 3.2. Module index / parse cache

```python
class ModuleIndex:
    def __init__(self, ctx: IngestionContext) -> None:
        self.ctx = ctx
        self.snapshots: dict[str, ModuleSnapshot] = {}
        self.parsed: dict[str, ParsedModule] = {}

    def load_snapshots(self) -> None:
        module_map = load_module_map(
            self.ctx.gateway, self.ctx.repo, self.ctx.commit, language="python"
        )
        for rec in iter_modules(module_map, self.ctx.repo_root, scan_config=self.ctx.scan_config):
            size, mtime_ns, h = _file_digest(rec.file_path)
            snap = ModuleSnapshot(
                rel_path=rec.rel_path,
                module_name=rec.module_name,
                file_path=rec.file_path,
                size_bytes=size,
                mtime_ns=mtime_ns,
                content_hash=h,
            )
            self.snapshots[rec.rel_path] = snap

    def get_parsed(
        self,
        rel_path: str,
        *,
        want_ast: bool = False,
        want_cst: bool = False,
    ) -> ParsedModule:
        parsed = self.parsed.get(rel_path)
        if parsed is None:
            snap = self.snapshots[rel_path]
            source = snap.file_path.read_text(encoding="utf8")
            parsed = ParsedModule(snapshot=snap, source=source)
            self.parsed[rel_path] = parsed

        if want_ast and parsed.ast_tree is None:
            parsed.ast_tree = ast.parse(parsed.source, filename=str(parsed.snapshot.file_path))
        if want_cst and parsed.cst_module is None:
            parsed.cst_module = cst.parse_module(parsed.source)
        return parsed
```

This gives you:

* **Single file read** per run (`source` cached in `ParsedModule`).
* Lazily computed AST and CST trees only when needed by a phase.
* A place to stash precomputed AST/CST rows so downstream analytics can reuse them.

### 3.3. Wiring into ingestion steps

Extend `IngestionContext` with an optional `module_index`:

```python
@dataclass(frozen=True)
class IngestionContext:
    ...
    module_index: ModuleIndex | None = None
```

Initialize it in your orchestrator once per run:

```python
ctx = IngestionContext(...)
ctx.module_index = ModuleIndex(ctx)
ctx.module_index.load_snapshots()
```

Then in AST/CST phases, instead of `read_module_source` and `cst.parse_module` / `ast.parse`, use the cache:

**AST:**

```python
def _collect_module_ast(record: ModuleRecord, ctx: IngestionContext) -> tuple[list[list[object]], AstMetrics] | None:
    parsed = ctx.module_index.get_parsed(record.rel_path, want_ast=True)
    tree = parsed.ast_tree
    visitor = AstVisitor(rel_path=record.rel_path, module_name=record.module_name)
    visitor.visit(tree)
    parsed.ast_rows = visitor.ast_rows
    parsed.ast_metrics = visitor.metrics
    return visitor.ast_rows, visitor.metrics
```

**CST:**

```python
def _process_module(record: ModuleRecord, ctx: IngestionContext) -> ModuleResult:
    parsed = ctx.module_index.get_parsed(record.rel_path, want_cst=True)
    wrapper = metadata.MetadataWrapper(parsed.cst_module, unsafe_skip_copy=True)
    visitor = CstVisitor(record.rel_path, record.module_name, parsed.source)
    wrapper.visit(visitor)
    parsed.cst_rows = visitor.rows
    return ModuleResult(rel_path=record.rel_path, rows=visitor.rows)
```

To make this work with multiprocessing:

* You can keep the **digest computations + change detection** in the parent process.
* For the actual parsing, you have two options:

  1. **Process‑local caches:** pass only `ModuleSnapshot` into worker processes and read/parse inside them (no shared cache). This keeps IPC small and is simpler; the module index remains a parent‑side convenience.

  2. **Single process for AST/CST, multiprocessing for heavier downstream analytics:** if you benchmark and see that AST/CST parsing isn’t the bottleneck once you have incremental digests, you can keep them single‑process and move the heavy stuff (graph building) to processes instead.

Given your bottleneck right now is “parsing every file every time”, the combination of **incremental digests + module index + moderately parallel parsing** is usually enough.

---

## 4. Other best practices / pragmatic tips

### 4.1. Don’t over‑optimize downstream analytics on day one

You already noted: recomputing metrics is cheap relative to ingest. With incremental AST/CST/SCIP in place, I’d initially:

* Recompute AST‑driven analytics (function metrics, hotspots, typedness) from scratch per run, but:

  * Use the precomputed `core.ast_nodes` and `core.file_state` instead of re‑parsing source where possible.
* Only later, once parsing is cheap, consider per‑module incremental recomputation of:

  * `function_metrics`
  * `function_types`
  * call graph & CFG/DFG for changed modules
  * risk factors

The architecture you have makes that feasible, but it’s a lot of plumbing; you’ll get 80% of the win from incremental ingestion alone.

### 4.2. Unify command‑line / config controls

You already have env‑driven worker count for CST (`CODEINTEL_CST_WORKERS`).

Extend that idea:

* `CODEINTEL_AST_WORKERS`
* `CODEINTEL_SCIP_WORKERS`
* `CODEINTEL_SCIP_MODE=full|incremental`

so you can flip between “full rebuild” and “incremental shards” without changing code.

### 4.3. Avoid repeated `asyncio.run` per subprocess

Both `scip_ingest` and `typing_ingest` have small helpers that call `asyncio.run(_exec())` for each subprocess.

That’s fine now, but when you start calling these in parallel a lot, consider:

* Centralizing subprocess execution in `ToolRunner` (you already have it for pyright/pyrefly/ruff).
* Reusing a single event loop per process (or just using `subprocess.run` in a process pool).

Not critical, but a nice clean‑up once everything else works.

### 4.4. Make DB writes transactional per step

You’re already using `run_batch` with prepared statements and log progress for each table.

For incremental updates, wrap:

* The per‑file deletes
* The inserts for those files
* Any `file_state` updates

in a single DuckDB transaction per dataset (AST, CST, SCIP). That way a crash mid‑run doesn’t leave half‑updated rows for some files.

---

If you’d like, I can next:

* Sketch concrete patches for, say, **incremental CST** end‑to‑end (updated function signatures and queries), or
* Work out a minimal “v1” incremental SCIP that only uses per‑module JSON shards and doesn’t try to preserve a single `.scip` binary.


# incremental CST ingestion #

Here’s both of the concrete things I offered:

1. an incremental CST pipeline wired end‑to‑end, and
2. a minimal “v1” incremental SCIP that works off per‑module JSON shards and doesn’t try to keep a monolithic `.scip` up to date.

I’ll treat these as patch sketches against your existing files rather than hand‑wavey descriptions.

---

## 1. Incremental CST ingestion (end‑to‑end)

### 1.1. What you have today

`ingest_cst`:

* Loads *all* modules from `core.modules` via `load_module_map`.

* Immediately wipes all CST rows for `repo, commit` by calling:

  ```py
  run_batch(gateway, "core.cst_nodes", [], delete_params=[repo, commit])
  ```

* Builds a full `records = list(iter_modules(...))` and then parallelises `_process_module` across a `ThreadPoolExecutor`.

* `_process_module` reads the file from disk, parses LibCST, runs `CstVisitor` and returns rows.

So any change forces a full repo scan + delete + re‑insert of CST rows.

### 1.2. Target shape

* **New API**: `ingest_cst(..., changed_paths: Collection[str] | None = None)`.
* **Full mode** (`changed_paths is None`): exactly what you do today.
* **Incremental mode**:

  * Filter `module_map` down to `changed_paths`.
  * Delete `core.cst_nodes` rows only for those `rel_path`s.
  * Re‑run LibCST only for those modules and append rows.

You can compute `changed_paths` however you like (git diff, watcher, hash table); CST ingestion just needs a list of repo‑relative paths.

### 1.3. Patch sketch for `cst_extract.py`

Key ideas:

* Add helper to filter `module_map` down to changed paths.
* Add helper to delete old rows for those paths.
* Extend `ingest_cst` signature and branch on full vs incremental.

```py
# cst_extract.py
from __future__ import annotations

import logging
import os
import time
from collections.abc import Collection, Iterable
from concurrent.futures import ThreadPoolExecutor  # or ProcessPoolExecutor later
from dataclasses import dataclass
from pathlib import Path

import libcst as cst
from libcst import metadata

from codeintel.ingestion.common import (
    ModuleRecord,
    iter_modules,
    load_module_map,
    read_module_source,
    run_batch,
    should_skip_empty,
)
from codeintel.ingestion.cst_utils import CstCaptureConfig, CstCaptureVisitor
from codeintel.ingestion.source_scanner import ScanConfig
from codeintel.storage.gateway import StorageGateway
from codeintel.utils.paths import normalize_rel_path  # <— new import
```

```py
# helper: narrow module_map to changed paths
def _filter_module_map(
    module_map: dict[str, str],
    changed_paths: Collection[str] | None,
) -> dict[str, str]:
    if not changed_paths:
        return module_map

    normalized = {normalize_rel_path(p) for p in changed_paths}
    return {rel_path: module for rel_path, module in module_map.items() if rel_path in normalized}
```

```py
# helper: delete old CST rows for a set of rel_paths
def _delete_existing_cst_rows(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    rel_paths: Iterable[str],
) -> None:
    rel_paths = sorted(set(rel_paths))
    if not rel_paths:
        return
    params = [(repo, commit, rel_path) for rel_path in rel_paths]
    gateway.con.executemany(
        "DELETE FROM core.cst_nodes "
        "WHERE repo = ? AND commit = ? AND rel_path = ?",
        params,
    )
```

Now extend `ingest_cst`:

```py
def ingest_cst(
    gateway: StorageGateway,
    repo_root: Path,
    repo: str,
    commit: str,
    scan_config: ScanConfig | None = None,
    *,
    changed_paths: Collection[str] | None = None,
) -> None:
    """
    Parse modules listed in core.modules using LibCST and populate cst_nodes.

    When changed_paths is provided, only those modules are re‑indexed and
    their existing rows in core.cst_nodes are replaced.
    """
    repo_root = repo_root.resolve()
    module_map = load_module_map(gateway, repo, commit, language="python", logger=log)
    if should_skip_empty(module_map, logger=log):
        return

    # Narrow to changed paths when running incrementally
    module_map = _filter_module_map(module_map, changed_paths)
    total_modules = len(module_map)
    if total_modules == 0:
        log.info(
            "No CST‑eligible modules for %s@%s (changed_paths=%s)",
            repo,
            commit,
            changed_paths,
        )
        return

    log.info("Parsing CST for %d modules in %s@%s", total_modules, repo, commit)

    cst_values: list[Row] = []
    start_ts = time.perf_counter()

    # Full refresh wipes all rows; incremental deletes only affected paths.
    if changed_paths is None:
        run_batch(gateway, "core.cst_nodes", [], delete_params=[repo, commit])
    else:
        _delete_existing_cst_rows(gateway, repo, commit, module_map.keys())

    records = list(
        iter_modules(
            module_map,
            repo_root,
            logger=log,
            scan_config=scan_config,
        )
    )

    worker_count = _resolve_worker_count()
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        for result in pool.map(_process_module, records):
            if result.error is not None:
                log.warning("Failed to parse %s: %s", result.rel_path, result.error)
            if result.rows:
                cst_values.extend(result.rows)
            if len(cst_values) >= FLUSH_EVERY:
                cst_values = _flush_batch(gateway, cst_values)

    _flush_batch(gateway, cst_values)
    duration = time.perf_counter() - start_ts
    log.info(
        "CST extraction complete for %s@%s (%d modules, %.2fs)",
        repo,
        commit,
        total_modules,
        duration,
    )
```

This keeps your existing flush logic intact, but:

* full mode still clears `core.cst_nodes` in one go via `run_batch(... delete_params=[repo, commit])` as before, and
* incremental mode only deletes rows for the changed `rel_path`s.

### 1.4. Wire incremental CST into the ingestion “front door”

Update the centralized runner so orchestrated callers can use the new argument. Right now `run_cst_extract` just forwards `ctx` to `ingest_cst`.

You can extend it like:

```py
# ingestion/runner.py

from collections.abc import Collection

def run_cst_extract(
    ctx: IngestionContext,
    changed_paths: Collection[str] | None = None,
) -> None:
    """Extract LibCST nodes for the repository using the gateway connection."""
    start = _log_step_start("cst_extract", ctx)
    cst_extract.ingest_cst(
        ctx.gateway,
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
        scan_config=ctx.scan_config,
        changed_paths=changed_paths,
    )
    _log_step_done("cst_extract", start, ctx)
```

Your CLI / orchestrator (or eventual file‑watcher) can now call:

```py
run_cst_extract(ctx, changed_paths={"src/foo.py", "src/bar/baz.py"})
```

Full runs keep working by calling `run_cst_extract(ctx)` with no `changed_paths`.

---

## 2. Minimal “v1” incremental SCIP with per‑module JSON shards

Now the SCIP side.

### 2.1. What you have today

`ingest_scip` currently:

* Verifies `.git` and binaries.
* Writes a single `index.scip` and `index.scip.json` under `build_dir / "scip"`.
* Copies them into your document output dir.
* Registers a DuckDB view:

  ```py
  docs_table = con.execute(
      "SELECT unnest(documents, recursive:=true) AS document FROM read_json(?)",
      [str(index_json)],
  ).fetch_arrow_table()
  con.register("scip_index_view_temp", docs_table)
  con.execute("CREATE VIEW scip_index_view AS SELECT * FROM scip_index_view_temp")
  ```



* Calls `_update_scip_symbols` to backfill `core.goid_crosswalk.scip_symbol` by reading the entire JSON.

Any refresh re‑indexes the *entire* repo.

### 2.2. Target shape for incremental v1

We **don’t** try to maintain a single canonical `index.scip`:

* For **changed files only**, we run:

  ```bash
  scip-python index . --target-only=src/my_module.py --output <shard>.scip
  scip print --json <shard>.scip > build/scip/docs/<something>.json
  ```

* We treat each JSON as a **per‑module shard**; the `.scip` files are just temporary.

* For queries, we create `scip_index_view` from **all** shard JSONs via a glob.

* For GOID crosswalk updates, we:

  * parse only the **changed shards**,
  * build a definition map for the `(rel_path, start_line) → symbol` pairs inside those docs, and
  * update `core.goid_crosswalk.scip_symbol` only for GOIDs whose `rel_path` appears in that set.

Initial “full” SCIP runs can continue to use the existing monolithic mode; incremental runs only touch shards.

### 2.3. New helpers in `scip_ingest.py`

Add an encoder for shard filenames and a helper to build the DuckDB view from a directory of shard JSONs:

```py
# scip_ingest.py

from collections.abc import Collection, Iterable

def _shard_name(rel_path: str) -> str:
    """
    Encode a repo-relative path into a stable filename-safe shard id.
    """
    # simple and predictable; you can switch to a hash if you prefer
    return rel_path.replace("/", "__").replace("\\", "__")


def _register_scip_view_from_shards(con, shards_dir: Path) -> None:
    """
    Rebuild scip_index_view from all JSON shards under shards_dir.
    """
    pattern = str(shards_dir / "*.json")
    con.execute("DROP VIEW IF EXISTS scip_index_view")

    # If there are no shards yet, don't register a view.
    if not list(shards_dir.glob("*.json")):
        return

    docs_table = con.execute(
        "SELECT unnest(documents, recursive:=true) AS document "
        "FROM read_json(?)",
        [pattern],
    ).fetch_arrow_table()
    con.register("scip_index_view_temp", docs_table)
    con.execute("CREATE VIEW scip_index_view AS SELECT * FROM scip_index_view_temp")
```

Add a more flexible `_fetch_goids` so we can filter by `rel_path` when doing incremental updates:

```py
def _fetch_goids(
    gateway: StorageGateway,
    rel_paths: Collection[str] | None = None,
) -> list[tuple[str, str, int, str, str]]:
    con = gateway.con
    if rel_paths:
        rel_paths = sorted(set(rel_paths))
        placeholders = ",".join("?" for _ in rel_paths)
        query = f"""
            SELECT urn, rel_path, start_line, repo, commit
            FROM core.goids
            WHERE rel_path IN ({placeholders})
        """
        return list(con.execute(query, list(rel_paths)).fetchall())

    # existing behavior: all GOIDs
    return list(
        con.execute(
            """
            SELECT urn, rel_path, start_line, repo, commit
            FROM core.goids
            """
        ).fetchall()
    )
```

Adjust `_update_scip_symbols` to use the new signature when called in full mode:

```py
def _update_scip_symbols(gateway: StorageGateway, index_json: Path) -> None:
    """Populate core.goid_crosswalk.scip_symbol by matching SCIP definitions to GOIDs."""
    con = gateway.con
    ensure_schema(con, "core.goid_crosswalk")
    docs = _load_scip_documents(index_json)
    def_map = _build_definition_map(docs)
    if not def_map:
        return

    # full-table update uses all GOIDs
    updates = _build_symbol_updates(def_map, _fetch_goids(gateway, None))
    if not updates:
        return

    con.executemany(GOID_CROSSWALK_UPDATE_SCIP, updates)
    log.info("Updated SCIP symbols for %d GOIDs", len(updates))
```

New incremental variant that works off shard documents:

```py
def _update_scip_symbols_from_docs(
    gateway: StorageGateway,
    shard_paths: Iterable[Path],
) -> None:
    """
    Incrementally update SCIP symbols using a small set of per-module JSON shards.
    """
    all_docs: list[dict[str, object]] = []
    rel_paths: set[str] = set()

    for shard in shard_paths:
        docs = _load_scip_documents(shard)
        if not docs:
            continue
        all_docs.extend(docs)
        for doc in docs:
            rel = (
                doc.get("relative_path")
                if isinstance(doc, dict)
                else getattr(doc, "relative_path", None)
            )
            if isinstance(rel, str):
                rel_paths.add(rel)

    if not all_docs or not rel_paths:
        return

    def_map = _build_definition_map(all_docs)
    if not def_map:
        return

    goids = _fetch_goids(gateway, rel_paths)
    updates = _build_symbol_updates(def_map, goids)
    if not updates:
        return

    gateway.con.executemany(GOID_CROSSWALK_UPDATE_SCIP, updates)
    log.info("Incrementally updated SCIP symbols for %d GOIDs", len(updates))
```

### 2.4. Incremental variant of `ingest_scip`

Extend the public entrypoint to accept `changed_paths` and dispatch accordingly. In full mode, keep current behaviour. In incremental mode, only touch shards.

```py
def ingest_scip(
    gateway: StorageGateway,
    cfg: ScipIngestConfig,
    *,
    changed_paths: Collection[str] | None = None,
) -> ScipIngestResult:
    """
    Run scip-python + scip print, register view, and backfill SCIP symbols.

    When changed_paths is provided, index only those files and update the
    scip_index_view + GOID crosswalk incrementally using per-module JSON shards.
    """
    con = gateway.con
    repo_root = cfg.repo_root.resolve()
    if not (repo_root / ".git").is_dir():
        reason = "SCIP ingestion requires a git repository (.git missing)"
        log.warning(reason)
        return ScipIngestResult(
            status="unavailable", index_scip=None, index_json=None, reason=reason
        )

    if cfg.scip_runner is not None and changed_paths is None:
        # Custom runner handles full indexing only
        return cast("ScipIngestResult", cfg.scip_runner(gateway, cfg))

    probe_result = _probe_binaries(cfg)
    if probe_result is not None:
        return probe_result

    scip_dir = cfg.build_dir.resolve() / "scip"
    scip_dir.mkdir(parents=True, exist_ok=True)
    doc_dir = cfg.document_output_dir.resolve()
    doc_dir.mkdir(parents=True, exist_ok=True)

    # Incremental mode: per-module shards only
    if changed_paths:
        shards_dir = scip_dir / "docs"
        shards_dir.mkdir(parents=True, exist_ok=True)

        shard_paths: list[Path] = []
        for rel_path in changed_paths:
            shard_id = _shard_name(rel_path)
            shard_scip = scip_dir / f"{shard_id}.scip"
            shard_json = shards_dir / f"{shard_id}.json"

            target = repo_root / rel_path
            if not target.is_file():
                log.warning("SCIP incremental: %s does not exist; skipping", target)
                continue

            # index only this file
            code = _run_scip_python(
                cfg.scip_python_bin,
                repo_root,
                shard_scip,
                target_only=target,
            )
            if not code:
                continue
            if not _run_scip_print(cfg.scip_bin, shard_scip, shard_json):
                continue

            shard_paths.append(shard_json)

        # Always rebuild the view over *all* shards so queries see consistent state.
        _register_scip_view_from_shards(con, shards_dir)
        _update_scip_symbols_from_docs(gateway, shard_paths)

        # In incremental mode we don't have a single index.scip / index.scip.json
        return ScipIngestResult(
            status="success",
            index_scip=None,
            index_json=None,
            reason=None,
        )

    # Full mode: existing monolithic behaviour
    index_scip = scip_dir / "index.scip"
    index_json = scip_dir / "index.scip.json"

    if not _run_scip_python(cfg.scip_python_bin, repo_root, index_scip):
        message = f"SCIP indexing failed for {cfg.repo}@{cfg.commit}"
        log.warning(message)
        return ScipIngestResult(status="failed", index_scip=None, index_json=None, reason=message)
    if not _run_scip_print(cfg.scip_bin, index_scip, index_json):
        message = f"SCIP JSON export failed for {cfg.repo}@{cfg.commit}"
        log.warning(message)
        return ScipIngestResult(
            status="failed", index_scip=index_scip, index_json=None, reason=message
        )

    writer = cfg.artifact_writer or _copy_artifacts
    writer(index_scip, index_json, doc_dir)

    docs_table = con.execute(
        "SELECT unnest(documents, recursive:=true) AS document FROM read_json(?)",
        [str(index_json)],
    ).fetch_arrow_table()
    con.execute("DROP VIEW IF EXISTS scip_index_view")
    con.register("scip_index_view_temp", docs_table)
    con.execute("CREATE VIEW scip_index_view AS SELECT * FROM scip_index_view_temp")

    _update_scip_symbols(gateway, index_json)
    log.info("SCIP index ingested for %s@%s", cfg.repo, cfg.commit)
    return ScipIngestResult(status="success", index_scip=index_scip, index_json=index_json)
```

You also need to tweak `_run_scip_python` to support `target_only`:

```py
def _run_scip_python(
    binary: str,
    repo_root: Path,
    output_path: Path,
    target_only: Path | None = None,
) -> bool:
    target_dir = repo_root / "src"
    if not target_dir.is_dir():
        target_dir = repo_root

    args = [binary, "index", str(target_dir), "--output", str(output_path)]
    if target_only is not None:
        # repo-relative path to the file to index
        rel = target_only.relative_to(repo_root)
        args.extend(["--target-only", str(rel)])

    code, stdout, stderr = _run_command(args, cwd=repo_root)
    if code == 0:
        if stderr:
            log.debug("scip-python stderr: %s", stderr.strip())
        return True
    if code == MISSING_BINARY_EXIT_CODE:
        log.warning("scip-python binary %r not found; skipping SCIP indexing", binary)
        return False
    log.warning("scip-python index failed (code %s): %s", code, stderr.strip() or stdout.strip())
    return False
```

### 2.5. Wire incremental SCIP into the ingestion runner

Update the ingestion “front door” so you can call incremental SCIP the same way you do for CST. `run_scip_ingest` currently just builds `ScipIngestConfig` and calls `scip_ingest.ingest_scip(ctx.gateway, cfg=cfg)`.

Make it accept `changed_paths` and pass them through:

```py
# ingestion/runner.py

def run_scip_ingest(
    ctx: IngestionContext,
    changed_paths: Collection[str] | None = None,
) -> ScipIngestResult:
    """
    Execute scip-python indexing and register outputs.

    When changed_paths is provided, run incremental per-module indexing.
    """
    start = _log_step_start("scip_ingest", ctx)
    doc_dir = ctx.paths.document_output_dir or (ctx.repo_root / "Document Output")
    cfg = ScipIngestConfig(
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
        build_dir=ctx.build_paths.build_dir,
        document_output_dir=doc_dir,
        scip_python_bin=ctx.tools.scip_python_bin if ctx.tools else "scip-python",
        scip_bin=ctx.tools.scip_bin if ctx.tools else "scip",
        scip_runner=ctx.scip_runner,
        artifact_writer=ctx.artifact_writer,
    )
    result = scip_ingest.ingest_scip(ctx.gateway, cfg=cfg, changed_paths=changed_paths)
    _log_step_done("scip_ingest", start, ctx)
    return result
```

Now you can:

* Run a full SCIP index with `run_scip_ingest(ctx)`.
* Run an incremental update for a few modules:

  ```py
  run_scip_ingest(ctx, changed_paths={"src/foo.py", "src/bar/baz.py"})
  ```

Under the hood, that:

* generates or refreshes JSON shards under `build/scip/docs/*.json`,
* rebuilds the `scip_index_view` over all shards, and
* only re‑parses / updates GOID ↔ SCIP mappings for the modules you touched.

---

### 2.6. Notes / follow‑ups

* This v1 keeps concurrency simple: it still runs `scip-python` sequentially per changed file. You can parallelise shard creation later with `asyncio.gather` over multiple `_run_command` calls if needed.
* Because everything is keyed by `rel_path` and GOID spans, incremental SCIP remains consistent with the rest of your architecture (call graph, symbol use edges, etc.) as long as you always re‑index a file whenever its contents change.
* You can reuse the same `changed_paths` set for **CST, AST, SCIP, and typedness ingestion**, which gives you a clean story: detect changed modules once, fan it out into these incremental ingestion entrypoints, then recompute downstream analytics/views off the updated tables.
