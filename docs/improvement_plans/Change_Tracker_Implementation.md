Let’s turn “change tracker” from a vibe into a concrete little subsystem an AI agent can implement.

I’ll structure this like before:

1. Goals & invariants
2. High-level design (new types & responsibilities)
3. Implement `ChangeTracker`
4. Implement a generic incremental ingest harness
5. Wire it into `repo_scan`
6. Adapt AST ingestion
7. Adapt CST ingestion
8. Adapt SCIP ingestion
9. Testing & guardrails

I’ll use names that match what you already have (`ChangeRequest`, `ChangeSet`, `StorageGateway`, `ModuleRecord`, etc.) so this is drop-in.

---

## 1. Goals & invariants

After this refactor:

* Change detection is computed **once per pipeline run**, using your existing `compute_changes(...)`.
* AST, CST, SCIP (and later others) **do not** invent their own “what changed?” logic.
* Incremental vs full rebuild rules (thresholds, fallbacks) live in **one place**.
* Each ingest step just defines:

  * how to delete rows for a set of `rel_path`s, and
  * how to re-build rows for a given module.

Everything else (which modules to reparse, when to fall back, logging, batching, concurrency) is handled by a shared driver.

---

## 2. High-level design

### 2.1 New module: `ingestion.change_tracker`

Create `src/codeintel/ingestion/change_tracker.py` with:

* `ChangeTracker` – owns `ChangeSet` + modules + policy.
* `IncrementalIngestPolicy` – thresholds + batching knobs.
* `ChangeTrackerDatasetView` – “what should this dataset rebuild?”.
* `run_incremental_ingest(...)` – generic driver used by AST/CST/SCIP.

### 2.2 Inputs & outputs

Inputs:

* `gateway: StorageGateway`
* `snapshot` (repo, commit, etc. – from your `PipelineContext`)
* `modules: Sequence[ModuleRecord]` – from `repo_scan`
* `change_request: ChangeRequest | None` – already used with `compute_changes(...)`

Outputs:

* For each dataset key (e.g. `"core.ast_nodes"`, `"core.cst_nodes"`, `"analytics.scip_symbols"`):

  * `to_reparse: list[ModuleRecord]`
  * `deleted_paths: list[str]` (rel_paths to delete from that dataset)

---

## 3. Implement `ChangeTracker`

### 3.1 Leverage existing `ChangeSet`

You already have:

* `ChangeRequest`
* `ChangeSet`
* `compute_changes(gateway, modules, change_request)` (or similar signature) that reads/writes `core.file_state` and returns added/modified/deleted modules.

We’ll wrap that once and reuse everywhere.

```python
# src/codeintel/ingestion/change_tracker.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from codeintel.ingestion.common import ChangeRequest, ChangeSet, ModuleRecord, compute_changes
from codeintel.storage.gateway import StorageGateway
from codeintel.orchestration.context import SnapshotConfig  # or wherever this lives


@dataclass(frozen=True)
class IncrementalIngestPolicy:
    """
    Tuning knobs for incremental vs full rebuild.
    """
    max_changed_ratio: float = 0.7       # if >70% of modules changed, do full rebuild
    max_deleted_ratio: float = 0.7      # if >70% deleted, do full rebuild
    min_total_modules_for_ratio: int = 20
    # for logging / batching
    log_every: int = 100
    flush_every: int = 500
```

Now the tracker itself:

```python
@dataclass
class ChangeTracker:
    """
    Single source of truth for 'what changed?' in a snapshot.

    Computes ChangeSet once, then provides per-dataset views for incremental ingest.
    """
    gateway: StorageGateway
    snapshot: SnapshotConfig
    modules: Sequence[ModuleRecord]
    change_request: ChangeRequest | None
    change_set: ChangeSet
    policy: IncrementalIngestPolicy

    @classmethod
    def create(
        cls,
        gateway: StorageGateway,
        snapshot: SnapshotConfig,
        modules: Sequence[ModuleRecord],
        change_request: ChangeRequest | None,
        policy: IncrementalIngestPolicy | None = None,
    ) -> "ChangeTracker":
        if policy is None:
            policy = IncrementalIngestPolicy()

        # This is your existing function; it should:
        # - read old core.file_state
        # - compute added/modified/deleted
        # - write new core.file_state for this snapshot
        change_set = compute_changes(
            gateway=gateway,
            snapshot=snapshot,
            modules=modules,
            change_request=change_request,
        )
        return cls(
            gateway=gateway,
            snapshot=snapshot,
            modules=tuple(modules),
            change_request=change_request,
            change_set=change_set,
            policy=policy,
        )
```

We keep `compute_changes` as is; the change tracker just orchestrates it and holds the result.

---

## 4. Dataset-aware view on changes

Different datasets may care about different modules:

* AST/CST: only Python modules.
* SCIP: maybe only Python modules in `src/`, not tests (depending on your design).
* Later: config datasets, docstrings, etc.

We want one method to compute “what to reparse and delete” for a given dataset.

```python
from typing import Callable, NamedTuple


ModuleFilter = Callable[[ModuleRecord], bool]


class ChangeTrackerDatasetView(NamedTuple):
    to_reparse: list[ModuleRecord]
    deleted_paths: list[str]
    total_modules_considered: int
    changed_modules_count: int
    deleted_modules_count: int
    use_full_rebuild: bool
```

Add to `ChangeTracker`:

```python
@dataclass
class ChangeTracker:
    ...
    def view_for_dataset(
        self,
        *,
        dataset_name: str,
        module_filter: ModuleFilter | None = None,
    ) -> ChangeTrackerDatasetView:
        """
        Compute which modules to reparse and which paths to delete for a given dataset.
        Applies incremental policy; may fall back to full rebuild.
        """
        if module_filter is None:
            # default: all modules
            relevant_modules = list(self.modules)
        else:
            relevant_modules = [m for m in self.modules if module_filter(m)]

        # intersect added/modified/deleted with relevant modules
        rel_paths_set = {m.rel_path for m in relevant_modules}

        added = [m for m in self.change_set.added if m.rel_path in rel_paths_set]
        modified = [m for m in self.change_set.modified if m.rel_path in rel_paths_set]
        deleted = [m for m in self.change_set.deleted if m.rel_path in rel_paths_set]

        to_reparse = added + modified
        deleted_paths = [m.rel_path for m in deleted]

        total = len(relevant_modules)
        changed_count = len(to_reparse)
        deleted_count = len(deleted)

        use_full = False
        if total >= self.policy.min_total_modules_for_ratio and total > 0:
            changed_ratio = (changed_count + deleted_count) / total
            deleted_ratio = deleted_count / total
            if (
                changed_ratio >= self.policy.max_changed_ratio
                or deleted_ratio >= self.policy.max_deleted_ratio
                or self.change_request and self.change_request.full_rebuild
            ):
                use_full = True

        if use_full:
            # Full rebuild semantics: delete everything in this dataset for these modules,
            # and reparse all relevant modules.
            to_reparse = relevant_modules
            deleted_paths = [m.rel_path for m in relevant_modules]

        return ChangeTrackerDatasetView(
            to_reparse=to_reparse,
            deleted_paths=deleted_paths,
            total_modules_considered=total,
            changed_modules_count=changed_count,
            deleted_modules_count=deleted_count,
            use_full_rebuild=use_full,
        )
```

This centralizes:

* What counts as “too many changes” → fallback to full.
* The full-rebuild semantics.
* Dataset-specific module filtering.

---

## 5. Generic incremental ingest harness

We want each ingestion step to specify *only* how to delete and how to rebuild.

```python
from concurrent.futures import Executor
from typing import Iterable, Protocol, TypeVar

Row = dict  # or a TypedDict / dataclass, whatever your ingestion uses
TExecutorFactory = Callable[[], Executor]


class IncrementalIngestOps(Protocol):
    dataset_name: str

    def module_filter(self, module: ModuleRecord) -> bool:
        ...

    def delete_rows(self, gateway: StorageGateway, rel_paths: list[str]) -> None:
        ...

    def process_module(self, module: ModuleRecord) -> Iterable[Row]:
        ...

    def insert_rows(self, gateway: StorageGateway, rows: Iterable[Row]) -> None:
        ...
```

Generic driver (in `change_tracker.py`):

```python
import logging

logger = logging.getLogger(__name__)


def run_incremental_ingest(
    tracker: ChangeTracker,
    ops: IncrementalIngestOps,
    *,
    executor_factory: TExecutorFactory | None = None,
) -> None:
    """
    Shared driver for AST/CST/SCIP and similar per-module datasets.
    """
    view = tracker.view_for_dataset(
        dataset_name=ops.dataset_name,
        module_filter=ops.module_filter,
    )

    if not view.to_reparse and not view.deleted_paths:
        logger.info(
            "No changes for dataset %s (total=%d)",
            ops.dataset_name,
            view.total_modules_considered,
        )
        return

    if view.use_full_rebuild:
        logger.info(
            "Dataset %s: falling back to full rebuild (changed=%d, deleted=%d, total=%d)",
            ops.dataset_name,
            view.changed_modules_count,
            view.deleted_modules_count,
            view.total_modules_considered,
        )
    else:
        logger.info(
            "Dataset %s: incremental ingest (reparse=%d, delete=%d, total=%d)",
            ops.dataset_name,
            len(view.to_reparse),
            len(view.deleted_paths),
            view.total_modules_considered,
        )

    if view.deleted_paths:
        ops.delete_rows(tracker.gateway, view.deleted_paths)

    # Build rows (optionally in parallel)
    if executor_factory is None:
        all_rows: list[Row] = []
        for mod in view.to_reparse:
            all_rows.extend(ops.process_module(mod))
    else:
        all_rows = []
        with executor_factory() as ex:
            for rows in ex.map(ops.process_module, view.to_reparse):
                all_rows.extend(rows)

    if not all_rows:
        logger.info("Dataset %s: no rows to insert after processing", ops.dataset_name)
        return

    ops.insert_rows(tracker.gateway, all_rows)
```

You can extend this later with streaming/batching; for a first iteration this is enough and already much more uniform than 3 ad-hoc implementations.

---

## 6. Wire it into `repo_scan`

The change tracker needs:

* `modules`
* `snapshot` (repo, commit)
* `change_request` (full vs partial rebuild)

You already have this in `repo_scan.ingest_repo` today.

### 6.1 Modify `repo_scan.ingest_repo` to return a `ChangeTracker`

Before (conceptually):

```python
def ingest_repo(
    gateway: StorageGateway,
    ctx: IngestionContext,
    change_request: ChangeRequest | None,
) -> ChangeSet:
    modules = scan_and_write_modules(gateway, ctx)
    change_set = compute_changes(gateway, ctx.snapshot, modules, change_request)
    return change_set
```

After:

```python
from codeintel.ingestion.change_tracker import ChangeTracker, IncrementalIngestPolicy

def ingest_repo(
    gateway: StorageGateway,
    ctx: IngestionContext,
    change_request: ChangeRequest | None,
    policy: IncrementalIngestPolicy | None = None,
) -> ChangeTracker:
    modules = scan_and_write_modules(gateway, ctx)  # writes core.modules, core.repo_map, etc.

    tracker = ChangeTracker.create(
        gateway=gateway,
        snapshot=ctx.snapshot,
        modules=modules,
        change_request=change_request,
        policy=policy,
    )
    return tracker
```

Then in `orchestration.steps.RepoScanStep.run`, you store the tracker on the `PipelineContext`:

```python
@dataclass
class PipelineContext:
    gateway: StorageGateway
    snapshot: SnapshotConfig
    change_tracker: ChangeTracker | None = None
    # ... other fields

@dataclass
class RepoScanStep(PipelineStep):
    def run(self, ctx: PipelineContext) -> None:
        tracker = ingest_repo(
            gateway=ctx.gateway,
            ctx=ctx.ingestion_context,
            change_request=ctx.change_request,
        )
        ctx.change_tracker = tracker
```

Subsequent steps (AST/CST/SCIP) will read `ctx.change_tracker`.

---

## 7. Adapt AST ingestion to the new harness

### 7.1 Define AST-specific ops

In `ingestion/py_ast_extract.py` (or a sibling module):

```python
# src/codeintel/ingestion/ast_incremental.py

from dataclasses import dataclass
from typing import Iterable

from codeintel.ingestion.change_tracker import IncrementalIngestOps
from codeintel.storage.gateway import StorageGateway
from codeintel.ingestion.common import ModuleRecord


@dataclass
class AstIngestOps(IncrementalIngestOps):
    dataset_name: str = "core.ast_nodes"

    def module_filter(self, module: ModuleRecord) -> bool:
        # Or use whatever field you have to check "is python source"
        return module.is_python  # or module.lang == "python"

    def delete_rows(self, gateway: StorageGateway, rel_paths: list[str]) -> None:
        if not rel_paths:
            return
        gateway.con.execute(
            """
            DELETE FROM core.ast_nodes
            WHERE rel_path IN (SELECT * FROM UNNEST(?))
            """,
            [rel_paths],
        )
        gateway.con.execute(
            """
            DELETE FROM analytics.ast_metrics
            WHERE rel_path IN (SELECT * FROM UNNEST(?))
            """,
            [rel_paths],
        )

    def process_module(self, module: ModuleRecord) -> Iterable[dict]:
        # Delegate to your existing AST extraction function that returns rows
        from codeintel.ingestion.py_ast_extract import extract_ast_for_module

        return extract_ast_for_module(module)

    def insert_rows(self, gateway: StorageGateway, rows: Iterable[dict]) -> None:
        rows_list = list(rows)
        if not rows_list:
            return
        # reuse your existing bulk insert helper:
        from codeintel.ingestion.py_ast_extract import insert_ast_rows

        insert_ast_rows(gateway.con, rows_list)
```

### 7.2 Replace ad-hoc incremental logic

In `py_ast_extract.ingest_python_ast` today, you likely have:

* Logic to compute `to_reparse` vs `deleted_paths` vs full rebuild.
* Logic to delete `core.ast_nodes` / `analytics.ast_metrics` rows.
* Logic to run an executor and insert new rows.

We want to strip that down to:

```python
# src/codeintel/ingestion/py_ast_extract.py

from concurrent.futures import ThreadPoolExecutor

from codeintel.ingestion.change_tracker import ChangeTracker, run_incremental_ingest
from codeintel.ingestion.ast_incremental import AstIngestOps


def ingest_python_ast(
    tracker: ChangeTracker,
    *,
    max_workers: int | None = None,
) -> None:
    def _executor_factory() -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=max_workers)

    run_incremental_ingest(
        tracker=tracker,
        ops=AstIngestOps(),
        executor_factory=_executor_factory,
    )
```

Then the orchestration step becomes:

```python
@dataclass
class AstStep(PipelineStep):
    def run(self, ctx: PipelineContext) -> None:
        assert ctx.change_tracker is not None, "RepoScanStep must run before AstStep"
        ingest_python_ast(ctx.change_tracker, max_workers=ctx.parallelism.ast_workers)
```

No more re-implementing changed/deleted/full rebuild logic here.

---

## 8. Adapt CST ingestion

Exactly the same pattern as AST; the ops differ only in delete/insert and the dataset name.

```python
# src/codeintel/ingestion/cst_incremental.py

from dataclasses import dataclass
from typing import Iterable

from codeintel.ingestion.change_tracker import IncrementalIngestOps
from codeintel.ingestion.common import ModuleRecord
from codeintel.storage.gateway import StorageGateway


@dataclass
class CstIngestOps(IncrementalIngestOps):
    dataset_name: str = "core.cst_nodes"

    def module_filter(self, module: ModuleRecord) -> bool:
        return module.is_python

    def delete_rows(self, gateway: StorageGateway, rel_paths: list[str]) -> None:
        if not rel_paths:
            return
        gateway.con.execute(
            """
            DELETE FROM core.cst_nodes
            WHERE rel_path IN (SELECT * FROM UNNEST(?))
            """,
            [rel_paths],
        )

    def process_module(self, module: ModuleRecord) -> Iterable[dict]:
        from codeintel.ingestion.cst_extract import extract_cst_for_module

        return extract_cst_for_module(module)

    def insert_rows(self, gateway: StorageGateway, rows: Iterable[dict]) -> None:
        from codeintel.ingestion.cst_extract import insert_cst_rows

        rows_list = list(rows)
        if not rows_list:
            return
        insert_cst_rows(gateway.con, rows_list)
```

New `ingest_cst`:

```python
# src/codeintel/ingestion/cst_extract.py

from concurrent.futures import ThreadPoolExecutor

from codeintel.ingestion.change_tracker import ChangeTracker, run_incremental_ingest
from codeintel.ingestion.cst_incremental import CstIngestOps


def ingest_cst(
    tracker: ChangeTracker,
    *,
    max_workers: int | None = None,
) -> None:
    def _executor_factory() -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=max_workers)

    run_incremental_ingest(
        tracker=tracker,
        ops=CstIngestOps(),
        executor_factory=_executor_factory,
    )
```

And the orchestration step uses `ctx.change_tracker` exactly as AST does.

---

## 9. Adapt SCIP ingestion

SCIP already has incremental logic with `_gather_changed_paths` and `_should_fallback_to_full`. We want to absorb that into the shared policy and dataset view.

### 9.1 SCIP-specific ops

```python
# src/codeintel/ingestion/scip_incremental.py

from dataclasses import dataclass
from typing import Iterable

from codeintel.ingestion.change_tracker import IncrementalIngestOps
from codeintel.ingestion.common import ModuleRecord
from codeintel.storage.gateway import StorageGateway


@dataclass
class ScipIngestOps(IncrementalIngestOps):
    dataset_name: str = "analytics.scip_symbols"  # or whatever main table is

    def module_filter(self, module: ModuleRecord) -> bool:
        # You may want only src/ modules, or exclude tests; reflect your current logic.
        return module.is_python and module.rel_path.startswith("src/")

    def delete_rows(self, gateway: StorageGateway, rel_paths: list[str]) -> None:
        if not rel_paths:
            return
        gateway.con.execute(
            """
            DELETE FROM core.scip_occurrences
            WHERE rel_path IN (SELECT * FROM UNNEST(?))
            """,
            [rel_paths],
        )
        gateway.con.execute(
            """
            DELETE FROM core.scip_symbols
            WHERE rel_path IN (SELECT * FROM UNNEST(?))
            """,
            [rel_paths],
        )

    def process_module(self, module: ModuleRecord) -> Iterable[dict]:
        # delegate to existing per-module SCIP shard logic
        from codeintel.ingestion.scip_ingest import extract_scip_for_module

        return extract_scip_for_module(module)

    def insert_rows(self, gateway: StorageGateway, rows: Iterable[dict]) -> None:
        from codeintel.ingestion.scip_ingest import insert_scip_rows

        rows_list = list(rows)
        if not rows_list:
            return
        insert_scip_rows(gateway.con, rows_list)
```

### 9.2 New `ingest_scip` using the harness

```python
# src/codeintel/ingestion/scip_ingest.py

from concurrent.futures import ProcessPoolExecutor  # or ThreadPoolExecutor

from codeintel.ingestion.change_tracker import ChangeTracker, run_incremental_ingest
from codeintel.ingestion.scip_incremental import ScipIngestOps


def ingest_scip(
    tracker: ChangeTracker,
    *,
    max_workers: int | None = None,
) -> None:
    def _executor_factory() -> ProcessPoolExecutor:
        return ProcessPoolExecutor(max_workers=max_workers)

    run_incremental_ingest(
        tracker=tracker,
        ops=ScipIngestOps(),
        executor_factory=_executor_factory,
    )
```

Then in the orchestration step:

```python
@dataclass
class ScipStep(PipelineStep):
    def run(self, ctx: PipelineContext) -> None:
        assert ctx.change_tracker is not None
        ingest_scip(ctx.change_tracker, max_workers=ctx.parallelism.scip_workers)
```

All the existing `_gather_changed_paths`, `_should_fallback_to_full`, and “incremental vs full” branching logic in `scip_ingest.py` can now be deleted, because it’s expressed in:

* `IncrementalIngestPolicy`
* `ChangeTracker.view_for_dataset`
* `run_incremental_ingest`

---

## 10. Testing & guardrails

### 10.1 Unit tests for `ChangeTracker.view_for_dataset`

Add tests that:

* Given a small list of modules + synthetic `ChangeSet`, verify:

  * incremental case: a few modules changed → only those appear in `to_reparse`, only deleted appear in `deleted_paths`.
  * full rebuild case: >70% changed → `use_full_rebuild=True`, all modules go to `to_reparse`, all rel_paths appear in `deleted_paths`.

Also test `module_filter` semantics: e.g. AST/CST/SCIP filters drop non-Python modules.

### 10.2 “Golden” tests for old vs new incremental behavior

For AST/CST/SCIP:

* Capture a small repo fixture (few files).
* Run old logic one last time (before refactor) to dump:

  * modules selected for reparse
  * rel_paths selected for delete
* After refactor, ensure new logic chooses the same sets for common scenarios:

  * modify one file
  * delete one file
  * add one file
  * change many files to trigger full rebuild.

Even if you don’t keep the old code, you can encode the expected behavior manually in test cases.

### 10.3 Architecture test: no ad-hoc incremental logic

Optionally, add a simple “architecture” test that fails if ingestion modules re-introduce their own `ChangeSet` logic:

* grep for `_should_fallback_to_full`, `_gather_changed_paths`, or direct `ChangeSet` usage outside `change_tracker.py` and `repo_scan.py`.
* fail if found.

---

