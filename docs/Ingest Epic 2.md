Gotcha, thanks for pasting the thinking — that’s super helpful context.

Let’s treat this as **Epic 2, Part 1: fully incremental docstrings + typing**, wired into the new step registry and `ChangeTracker`. I’ll give you **patch-style instructions with concrete code** so you can drop them in with minimal friction.

I’ll focus on:

1. `docstrings_ingest.py` → `DocstringIngestOps` + incremental path.
2. `typing_ingest.py` → `TypingIngestOps` + incremental path.
3. `steps.py` → update `DocstringsIngestStep` + `TypingIngestStep` to use the tracker.
4. Quick notes on how to extend the same pattern to coverage/tests/config (you can do those next using the same pattern).

I’ll keep existing APIs working for tests by adding an **optional `tracker` keyword** instead of breaking signatures.

---

## 1. ChangeTracker harness (no structural change needed)

Your current harness already has everything we need:

```python
# ingestion/change_tracker.py (already present)

class ChangeTrackerDatasetView(NamedTuple):
    to_reparse: list[ModuleRecord]
    deleted_paths: list[str]
    total_modules_considered: int
    changed_modules_count: int
    deleted_modules_count: int
    use_full_rebuild: bool


@runtime_checkable
class IncrementalIngestOps(Protocol[RowT]):
    dataset_name: str

    @staticmethod
    def module_filter(module: ModuleRecord) -> bool: ...
    def delete_rows(self, gateway: StorageGateway, rel_paths: Sequence[str]) -> None: ...
    @staticmethod
    def process_module(module: ModuleRecord) -> Iterable[RowT]: ...
    def insert_rows(self, gateway: StorageGateway, rows: Sequence[RowT]) -> None: ...


@runtime_checkable
class SupportsFullRebuild(Protocol):
    def run_full_rebuild(self, tracker: ChangeTracker) -> bool: ...


def run_incremental_ingest[RowT](tracker: ChangeTracker, ops: IncrementalIngestOps[RowT], *, executor_factory: ExecutorFactory | None = None) -> None:
    ...
```

Scip already uses **instance** `process_module(self, module)` even though the protocol claims it’s a staticmethod, and everything works. So we’re safe to use **instance methods** in our new ops classes too (type checkers may complain, but runtime semantics are fine).

I’ll keep the harness as-is and just plug into it.

---

## 2. Docstrings: Incremental ingest

### 2.1. New Ops class in `ingestion/docstrings_ingest.py`

**File:** `src/codeintel/ingestion/docstrings_ingest.py`

At the top, extend imports:

```python
from collections.abc import Iterable, Sequence  # NEW
...
from codeintel.config.builder import DocstringStepConfig
from codeintel.ingestion.common import (
    iter_modules,
    read_module_source,
    run_batch,
    should_skip_empty,
    ModuleRecord,            # NEW
)
from codeintel.ingestion.change_tracker import (  # NEW
    ChangeTracker,
    IncrementalIngestOps,
    run_incremental_ingest,
)
from codeintel.ingestion.source_scanner import ScanProfile
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.module_index import load_module_map
from codeintel.storage.rows import DocstringRow, docstring_row_to_tuple
```

Now add a small helper + ops class **below** `DocstringContext` / `DocstringVisitor` (before `ingest_docstrings`):

```python
# NEW: per-dataset delete helper

def _delete_existing_docstrings(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    rel_paths: list[str],
) -> None:
    """
    Remove stale docstring rows for the provided paths.

    When rel_paths is empty, perform a full rebuild-style delete for the
    current repo@commit.
    """
    if rel_paths:
        gateway.con.execute(
            """
            DELETE FROM core.docstrings
            WHERE repo = ? AND commit = ? AND rel_path IN (SELECT * FROM UNNEST(?))
            """,
            [repo, commit, rel_paths],
        )
        return

    # Full rebuild path: wipe all rows for this repo@commit.
    run_batch(
        gateway,
        "core.docstrings",
        [],
        delete_params=[repo, commit],
    )


# NEW: incremental ops

@dataclass
class DocstringIngestOps(IncrementalIngestOps[DocstringRow]):
    """
    Incremental ingest operations for core.docstrings.

    This uses the same AST + DocstringVisitor logic as the previous
    full-scan implementation, but scoped to the modules surfaced by
    ChangeTracker.view_for_dataset.
    """

    dataset_name: str
    cfg: DocstringStepConfig
    created_at: datetime

    def __init__(self, cfg: DocstringStepConfig, *, created_at: datetime | None = None) -> None:
        self.dataset_name = "core.docstrings"
        self.cfg = cfg
        self.created_at = created_at or datetime.now(UTC)
        self._ctx = DocstringContext(cfg=self.cfg, created_at=self.created_at)

    @staticmethod
    def module_filter(module: ModuleRecord) -> bool:
        """
        Restrict docstring ingestion to Python modules.

        We rely on repo_scan having already filtered modules according
        to the configured code scan profile.
        """
        return module.rel_path.endswith(".py")

    def delete_rows(self, gateway: StorageGateway, rel_paths: Sequence[str]) -> None:
        """Delete rows for modules scheduled for removal."""
        _delete_existing_docstrings(
            gateway,
            repo=self.cfg.repo,
            commit=self.cfg.commit,
            rel_paths=list(rel_paths),
        )

    def process_module(self, module: ModuleRecord) -> Iterable[DocstringRow]:
        """
        Parse a single module and emit DocstringRow instances.

        Returns an empty iterable on parse failure or unreadable file.
        """
        source = read_module_source(module, logger=log)
        if source is None:
            return []

        try:
            tree = ast.parse(source, filename=str(module.file_path))
        except SyntaxError:
            log.warning("Failed to parse AST for docstrings: %s", module.file_path)
            return []

        visitor = DocstringVisitor(
            rel_path=module.rel_path,
            module_name=module.module_name,
            ctx=self._ctx,
        )
        visitor.visit(tree)
        return visitor.rows

    def insert_rows(self, gateway: StorageGateway, rows: Sequence[DocstringRow]) -> None:
        """
        Insert docstring rows into DuckDB without full-table delete.

        Row-level deletion for changed/deleted modules is handled separately
        by delete_rows().
        """
        if not rows:
            return
        run_batch(
            gateway,
            "core.docstrings",
            [docstring_row_to_tuple(row) for row in rows],
            delete_params=None,
            scope=f"{self.cfg.repo}@{self.cfg.commit}",
        )
```

### 2.2. Rewrite `ingest_docstrings` to support both full & incremental

**Before** (current shape, simplified):

```python
def ingest_docstrings(
    gateway: StorageGateway,
    cfg: DocstringStepConfig,
    code_profile: ScanProfile | None = None,
) -> None:
    repo_root = cfg.repo_root.resolve()
    module_map = load_module_map(gateway, cfg.repo, cfg.commit, language="python", logger=log)
    if should_skip_empty(module_map, logger=log):
        return

    rows: list[DocstringRow] = []
    ctx = DocstringContext(cfg=cfg, created_at=datetime.now(UTC))

    for record in iter_modules(
        module_map,
        repo_root,
        logger=log,
        scan_profile=code_profile,
    ):
        ...
        visitor = DocstringVisitor(...)
        visitor.visit(tree)
        rows.extend(visitor.rows)

    run_batch(
        gateway,
        "core.docstrings",
        [docstring_row_to_tuple(row) for row in rows],
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )
```

**After** — keep the existing behavior when `tracker=None`, add the incremental branch when a `ChangeTracker` is provided:

```python
def ingest_docstrings(
    gateway: StorageGateway,
    cfg: DocstringStepConfig,
    code_profile: ScanProfile | None = None,
    *,
    tracker: ChangeTracker | None = None,
) -> None:
    """
    Extract docstrings for Python modules and persist them.

    When `tracker` is provided, use the shared incremental ingest harness
    and only re-parse modules that changed in the current snapshot. When
    `tracker` is None, fall back to the legacy full-scan implementation.
    """
    # Incremental path (preferred in production ingestion)
    if tracker is not None:
        ops = DocstringIngestOps(cfg=cfg, created_at=datetime.now(UTC))
        run_incremental_ingest(tracker, ops)
        log.info(
            "Docstrings ingested incrementally for %s@%s",
            cfg.repo,
            cfg.commit,
        )
        return

    # Legacy full-scan path (used by tests and ad-hoc callers)
    repo_root = cfg.repo_root.resolve()
    module_map = load_module_map(gateway, cfg.repo, cfg.commit, language="python", logger=log)
    if should_skip_empty(module_map, logger=log):
        return

    rows: list[DocstringRow] = []
    ctx = DocstringContext(cfg=cfg, created_at=datetime.now(UTC))

    for record in iter_modules(
        module_map,
        repo_root,
        logger=log,
        scan_profile=code_profile,
    ):
        source = read_module_source(record, logger=log)
        if source is None:
            continue

        try:
            tree = ast.parse(source, filename=str(record.file_path))
        except SyntaxError:
            log.warning("Failed to parse AST for docstrings: %s", record.file_path)
            continue

        visitor = DocstringVisitor(
            rel_path=record.rel_path,
            module_name=record.module_name,
            ctx=ctx,
        )
        visitor.visit(tree)
        rows.extend(visitor.rows)

    run_batch(
        gateway,
        "core.docstrings",
        [docstring_row_to_tuple(row) for row in rows],
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )
    log.info("Docstrings ingested: %d rows for %s@%s", len(rows), cfg.repo, cfg.commit)
```

So:

* **Production pipeline / steps**: call this with `tracker=ctx.change_tracker` → incremental.
* **Tests / helpers**: keep calling the old `(gateway, cfg, code_profile=...)` signature → full scan.

---

## 3. Typing: Incremental typedness + static diagnostics

Here we’ll:

* Introduce a small result carrier for both row types.
* Define `TypingIngestOps` that uses the heavy tools once and then emits per-module rows.
* Extend `ingest_typing_signals` with an optional `tracker` arg and branch accordingly.

### 3.1. New result + ops in `ingestion/typing_ingest.py`

**File:** `src/codeintel/ingestion/typing_ingest.py`

Update imports at the top:

```python
from collections.abc import Iterable, Sequence   # add Sequence
from dataclasses import dataclass
from pathlib import Path

from codeintel.config import TypingIngestStepConfig
from codeintel.config.models import ToolsConfig
from codeintel.ingestion.change_tracker import (   # NEW
    ChangeTracker,
    IncrementalIngestOps,
    run_incremental_ingest,
)
from codeintel.ingestion.common import run_batch, ModuleRecord  # NEW ModuleRecord
from codeintel.ingestion.paths import repo_relpath
...
```

Then, **below** `AnnotationInfo` (and `_compute_annotation_info_for_file` / `_collect_error_maps`), add:

```python
@dataclass
class TypingIngestResult:
    """
    Combined typedness + diagnostics payload for a single module path.
    """

    typedness: TypednessRow | None
    diagnostics: StaticDiagnosticRow | None


def _delete_existing_typing_rows(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    rel_paths: list[str],
) -> None:
    """
    Delete existing typedness + diagnostics rows for specific paths.

    When rel_paths is empty, perform a full-table delete for the repo@commit.
    """
    if rel_paths:
        gateway.con.execute(
            """
            DELETE FROM analytics.typedness
            WHERE repo = ? AND commit = ? AND path IN (SELECT * FROM UNNEST(?))
            """,
            [repo, commit, rel_paths],
        )
        gateway.con.execute(
            """
            DELETE FROM analytics.static_diagnostics
            WHERE repo = ? AND commit = ? AND rel_path IN (SELECT * FROM UNNEST(?))
            """,
            [repo, commit, rel_paths],
        )
        return

    # Full rebuild path
    run_batch(
        gateway,
        "analytics.typedness",
        [],
        delete_params=[repo, commit],
    )
    run_batch(
        gateway,
        "analytics.static_diagnostics",
        [],
        delete_params=[repo, commit],
    )


@dataclass
class TypingIngestOps(IncrementalIngestOps[TypingIngestResult]):
    """
    Incremental ingest operations for analytics.typedness + static_diagnostics.

    This uses:
      * per-module AST parsing for annotation_ratios
      * repo-level tool runs (pyrefly/pyright/ruff) for error counts,
        but only emits new rows for modules flagged by ChangeTracker.
    """

    dataset_name: str
    cfg: TypingIngestStepConfig
    repo_root: Path
    error_maps: dict[str, dict[str, int]]

    def __init__(
        self,
        *,
        cfg: TypingIngestStepConfig,
        repo_root: Path,
        error_maps: dict[str, dict[str, int]],
    ) -> None:
        self.dataset_name = "analytics.typedness"
        self.cfg = cfg
        self.repo_root = repo_root
        self.error_maps = error_maps

    @staticmethod
    def module_filter(module: ModuleRecord) -> bool:
        """Only consider Python source files."""
        return module.rel_path.endswith(".py")

    def delete_rows(self, gateway: StorageGateway, rel_paths: Sequence[str]) -> None:
        """Delete rows for modules scheduled for removal."""
        _delete_existing_typing_rows(
            gateway,
            repo=self.cfg.repo,
            commit=self.cfg.commit,
            rel_paths=list(rel_paths),
        )

    def process_module(self, module: ModuleRecord) -> Iterable[TypingIngestResult]:
        """
        Compute typedness + diagnostic rows for a single module.

        We recompute annotation ratios from the module's AST, and look up
        error counts from the precomputed error_maps.
        """
        path = self.repo_root / module.rel_path
        info = _compute_annotation_info_for_file(path) or AnnotationInfo(
            params_ratio=0.0,
            returns_ratio=0.0,
            untyped_defs=0,
        )
        rel_path = module.rel_path

        pf_errors = self.error_maps["pyrefly"].get(rel_path, 0)
        py_errors = self.error_maps["pyright"].get(rel_path, 0)
        ruff_errors = self.error_maps["ruff"].get(rel_path, 0)
        total_errors = pf_errors + py_errors

        typedness = TypednessRow(
            repo=self.cfg.repo,
            commit=self.cfg.commit,
            path=rel_path,
            type_error_count=total_errors,
            annotation_ratio={
                "params": info.params_ratio,
                "returns": info.returns_ratio,
            },
            untyped_defs=info.untyped_defs,
            overlay_needed=bool(total_errors > 0 or info.untyped_defs > 0),
        )

        diagnostics = StaticDiagnosticRow(
            repo=self.cfg.repo,
            commit=self.cfg.commit,
            rel_path=rel_path,
            pyrefly_errors=pf_errors,
            pyright_errors=py_errors,
            ruff_errors=ruff_errors,
            total_errors=total_errors,
            has_errors=total_errors > 0,
        )

        return [TypingIngestResult(typedness=typedness, diagnostics=diagnostics)]

    def insert_rows(self, gateway: StorageGateway, rows: Sequence[TypingIngestResult]) -> None:
        """
        Insert typedness + static_diagnostics rows into DuckDB.

        Row-level deletes for changed/deleted modules are already handled
        by delete_rows().
        """
        typedness_rows: list[TypednessRow] = []
        diag_rows: list[StaticDiagnosticRow] = []

        for result in rows:
            if result.typedness is not None:
                typedness_rows.append(result.typedness)
            if result.diagnostics is not None:
                diag_rows.append(result.diagnostics)

        if typedness_rows:
            run_batch(
                gateway,
                "analytics.typedness",
                [typedness_row_to_tuple(row) for row in typedness_rows],
                delete_params=None,
                scope=f"{self.cfg.repo}@{self.cfg.commit}",
            )

        if diag_rows:
            run_batch(
                gateway,
                "analytics.static_diagnostics",
                [static_diagnostic_to_tuple(row) for row in diag_rows],
                delete_params=None,
                scope=f"{self.cfg.repo}@{self.cfg.commit}",
            )
```

### 3.2. Extend `ingest_typing_signals` to add incremental branch

**Before** (simplified current function):

```python
def ingest_typing_signals(
    gateway: StorageGateway,
    cfg: TypingIngestStepConfig,
    *,
    code_profile: ScanProfile | None = None,
    tools: ToolsConfig | None = None,
    tool_service: ToolService | None = None,
) -> None:
    repo_root = cfg.repo_root
    profile = code_profile or profile_from_env(default_code_profile(repo_root))
    active_tools = tools or ToolsConfig.model_validate({})
    active_service = tool_service
    if active_service is None:
        shared_runner = ToolRunner(...)
        active_service = ToolService(shared_runner, active_tools)

    annotation_info = {}
    for path in _iter_python_files(profile):
        rel_path = repo_relpath(repo_root, path)
        info = _compute_annotation_info_for_file(path)
        ...

    error_maps = asyncio.run(_collect_error_maps(repo_root, active_service))
    path_set = union(...)

    # build typedness_rows + diag_rows for all paths
    # run_batch(... delete_params=[cfg.repo, cfg.commit]) twice
```

**After** — add `tracker` kwarg and dual path:

```python
def ingest_typing_signals(
    gateway: StorageGateway,
    cfg: TypingIngestStepConfig,
    *,
    code_profile: ScanProfile | None = None,
    tools: ToolsConfig | None = None,
    tool_service: ToolService | None = None,
    tracker: ChangeTracker | None = None,  # NEW
) -> None:
    """
    Populate per-file typedness and static diagnostics.

      - analytics.typedness
      - analytics.static_diagnostics

    When `tracker` is provided, only modules flagged by the change tracker
    are reprocessed using the incremental ingest harness. When `tracker`
    is None, fall back to the legacy full-rebuild implementation.
    """
    repo_root = cfg.repo_root
    profile = code_profile or profile_from_env(default_code_profile(repo_root))
    active_tools = tools or ToolsConfig.model_validate({})
    active_service = tool_service
    if active_service is None:
        shared_runner = ToolRunner(
            tools_config=active_tools,
            cache_dir=repo_root / "build" / ".tool_cache",
        )
        active_service = ToolService(shared_runner, active_tools)

    # Shared: compute repo-level error maps once.
    error_maps = asyncio.run(_collect_error_maps(repo_root, active_service))

    # Incremental path using ChangeTracker
    if tracker is not None:
        ops = TypingIngestOps(
            cfg=cfg,
            repo_root=repo_root,
            error_maps=error_maps,
        )
        run_incremental_ingest(tracker, ops)
        log.info(
            "Typedness & static diagnostics ingested incrementally for %s@%s",
            cfg.repo,
            cfg.commit,
        )
        return

    # Legacy full-scan path (used by tests and older callers)
    annotation_info: dict[str, AnnotationInfo] = {}
    for path in _iter_python_files(profile):
        rel_path = repo_relpath(repo_root, path)
        info = _compute_annotation_info_for_file(path)
        if info is not None:
            annotation_info[rel_path] = info

    path_set = (
        set(annotation_info)
        | set(error_maps["pyrefly"])
        | set(error_maps["pyright"])
        | set(error_maps["ruff"])
    )

    typedness_rows: list[TypednessRow] = []
    diag_rows: list[StaticDiagnosticRow] = []
    default_info = AnnotationInfo(params_ratio=0.0, returns_ratio=0.0, untyped_defs=0)

    for rel_path in sorted(path_set):
        info = annotation_info.get(rel_path, default_info)
        pf_errors = error_maps["pyrefly"].get(rel_path, 0)
        py_errors = error_maps["pyright"].get(rel_path, 0)
        total_errors = pf_errors + py_errors

        typedness_rows.append(
            TypednessRow(
                repo=cfg.repo,
                commit=cfg.commit,
                path=rel_path,
                type_error_count=total_errors,
                annotation_ratio={
                    "params": info.params_ratio,
                    "returns": info.returns_ratio,
                },
                untyped_defs=info.untyped_defs,
                overlay_needed=bool(total_errors > 0 or info.untyped_defs > 0),
            )
        )

        diag_rows.append(
            StaticDiagnosticRow(
                repo=cfg.repo,
                commit=cfg.commit,
                rel_path=rel_path,
                pyrefly_errors=pf_errors,
                pyright_errors=py_errors,
                ruff_errors=error_maps["ruff"].get(rel_path, 0),
                total_errors=total_errors,
                has_errors=total_errors > 0,
            )
        )

    run_batch(
        gateway,
        "analytics.typedness",
        [typedness_row_to_tuple(row) for row in typedness_rows],
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )
    run_batch(
        gateway,
        "analytics.static_diagnostics",
        [static_diagnostic_to_tuple(row) for row in diag_rows],
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )

    log.info(
        "Typedness & static diagnostics ingested for %d files in %s@%s",
        len(path_set),
        cfg.repo,
        cfg.commit,
    )
```

So:

* **Production path** (via steps): incremental, uses `TypingIngestOps + run_incremental_ingest`.
* **Tests / helpers** (e.g. `tests/_helpers/fixtures.py`): still call `ingest_typing_signals(gateway, cfg, tool_service=..., tools=...)` and get the old full-rebuild semantics.

---

## 4. Wire both into the step registry (`ingestion/steps.py`)

Now we need to update the step implementations to pass the `ChangeTracker` into the new incremental code paths.

**File:** `src/codeintel/ingestion/steps.py`

You already have `_require_change_tracker` defined and you’re using it in `ScipIngestStep`. We’ll reuse that.

### 4.1. Typing step

Current snippet (you saw this):

```python
@dataclass(frozen=True)
class TypingIngestStep:
    ...

    def run(self, ctx: IngestionContextProtocol) -> None:
        log.debug("Running ingestion step %s", self.name)
        cfg = TypingIngestStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
            tool_runner=ctx.tool_runner,
        )
        runner = ctx.tool_runner or ToolRunner(
            cache_dir=ctx.paths.tool_cache,
            tools_config=ctx.active_tools,
        )
        service = ctx.tool_service or ToolService(runner, ctx.active_tools)
        typing_ingest.ingest_typing_signals(
            gateway=ctx.gateway,
            cfg=cfg,
            code_profile=ctx.code_profile,
            tools=ctx.active_tools,
            tool_service=service,
        )
```

Change to:

```python
    def run(self, ctx: IngestionContextProtocol) -> None:
        log.debug("Running ingestion step %s", self.name)
        tracker = _require_change_tracker(ctx, self.name)
        cfg = TypingIngestStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
            tool_runner=ctx.tool_runner,
        )
        runner = ctx.tool_runner or ToolRunner(
            cache_dir=ctx.paths.tool_cache,
            tools_config=ctx.active_tools,
        )
        service = ctx.tool_service or ToolService(runner, ctx.active_tools)
        typing_ingest.ingest_typing_signals(
            gateway=ctx.gateway,
            cfg=cfg,
            code_profile=ctx.code_profile,
            tools=ctx.active_tools,
            tool_service=service,
            tracker=tracker,  # NEW
        )
```

### 4.2. Docstrings step

Current snippet:

```python
@dataclass(frozen=True)
class DocstringsIngestStep:
    ...

    def run(self, ctx: IngestionContextProtocol) -> None:
        log.debug("Running ingestion step %s", self.name)
        cfg = DocstringStepConfig(snapshot=ctx.snapshot)
        docstrings_ingest.ingest_docstrings(
            ctx.gateway,
            cfg,
            code_profile=ctx.code_profile,
        )
```

Change to:

```python
    def run(self, ctx: IngestionContextProtocol) -> None:
        log.debug("Running ingestion step %s", self.name)
        tracker = _require_change_tracker(ctx, self.name)
        cfg = DocstringStepConfig(snapshot=ctx.snapshot)
        docstrings_ingest.ingest_docstrings(
            ctx.gateway,
            cfg,
            code_profile=ctx.code_profile,
            tracker=tracker,  # NEW
        )
```

That’s all that’s needed to push both datasets onto the incremental harness in the normal pipeline.

---

## 5. Tests / behaviour guarantees

Once you apply the above, you’ll have:

* `core.docstrings` and `analytics.typedness/static_diagnostics` participating in the same **incremental ingest story** as AST/CST/SCIP.
* All **existing tests** that call `ingest_typing_signals` or `ingest_docstrings` without a tracker continue to work with their **full rebuild semantics**.

To make the change observable and guarded, I’d add:

1. **New unit tests** in `tests/ingestion/test_change_tracker.py`:

   * Construct a fake `ChangeTracker` with 2 modules.
   * Use a fake `StorageGateway` (you already have helpers in `tests/_helpers/gateway.py`) and a small in-memory DuckDB.
   * Drive `DocstringIngestOps` and `TypingIngestOps` with:

     * a first run (baseline),
     * a second run with only one module changed,
     * confirm only rows for that path are updated/deleted.

2. **Integration-style test** in `tests/ingestion/test_runner_plumbing.py`:

   * Provision a repo via `ProvisionedGateway` helper.
   * Call your pipeline / ingestion steps so that:

     * `repo_scan` runs, producing a `ChangeTracker`.
     * `DocstringsIngestStep` and `TypingIngestStep` run.
   * Mutate a single module file, rerun those steps, and assert:

     * counts for unchanged modules are stable,
     * rows for changed module updated.

The existing helpers in `tests/_helpers/fixtures.py` already set up repo + gateway + tools; you just need to add small asserts on row counts if you want explicit coverage.

---

## 6. Coverage / tests / config (outline)

You asked to include these in the scope; here’s how I’d extend the same pattern:

* **`coverage_ingest.py`**

  * Add optional `tracker: ChangeTracker | None = None` to `ingest_coverage_lines`.
  * Factor the “collect coverage rows” logic into a helper `_collect_coverage_rows(...) -> tuple[list[CoverageLineRow], str]`.
  * Define `CoverageIngestOps(IncrementalIngestOps[CoverageLineRow])` with:

    * `dataset_name = "analytics.coverage_lines"`.
    * `module_filter` filtering `module.rel_path.endswith(".py")`.
    * `delete_rows` that deletes only rows for changed paths (or full wipe when paths empty).
    * `process_module` returning precomputed rows for that module’s `rel_path`.
    * `insert_rows` doing a `run_batch(..., delete_params=None)`.
  * In the incremental branch (`tracker is not None`), call `_collect_coverage_rows`, group rows by `rel_path` and feed them into `CoverageIngestOps + run_incremental_ingest`.

* **`tests_ingest.py`**

  * Add optional `tracker: ChangeTracker | None = None` to `ingest_tests`.
  * Factor out `_collect_test_rows(...) -> list[TestCatalogRowModel]`.
  * Define `TestsIngestOps(IncrementalIngestOps[TestCatalogRowModel])` with:

    * `dataset_name = "analytics.test_catalog"`.
    * `module_filter` restricting to test paths (e.g. `module.rel_path.startswith("tests/")`).
    * `delete_rows` deleting rows for changed test paths.
    * `process_module` returning precomputed rows grouped by `rel_path`.
    * `insert_rows` using `run_batch(..., delete_params=None)`.
  * In the incremental branch, use `run_incremental_ingest(tracker, ops)`; the legacy branch remains the full-rebuild you have now.

* **`config_ingest.py`**

  * Config is not obviously per-module (it’s per config file), but you can still:

    * Compute a `ChangeTrackerDatasetView` with a `module_filter` that selects only known config paths (e.g. `pyproject.toml`, `*.yaml` in config dirs).
    * Use `SupportsFullRebuild` on a `ConfigIngestOps` that always runs a full ingest when `view.use_full_rebuild` is set.

If you’d like, we can do the same **“before/after with concrete code”** drill for one of those (e.g. coverage) next, but with the docstrings + typing code above you’ve already got the hardest bits in hand.


# before/after with concrete code for coverage #

Short version: I’m going to give you a concrete, patch-level plan for **incremental coverage ingest**, in the same “before/after” style as SCIP/AST, but with one key caveat called out up front:

> **Important semantic caveat:** coverage is driven by *test runs*, not just *code changes*. If you gate coverage ingestion purely on `ChangeTracker` (which only looks at file content), you can end up with stale coverage if tests are re-run without code changing.
>
> So what I’ll do is:
>
> * Keep **“full rebuild on each coverage ingest”** as the default (semantically correct).
> * still introduce a **CoverageIngestOps** and wire it through the incremental harness via `SupportsFullRebuild`, so everything is consistent and future-extensible.
> * Not attempt “true per-module incremental coverage” based on file diffs, because that would be incorrect without adding a whole separate coverage-run change tracker.

That way, coverage is now formally part of the incremental framework, and we haven’t broken correctness.

---

## 1. Coverage ingest today (what you roughly have now)

You currently have something like this in `ingestion/coverage_ingest.py` (trimmed for clarity):

```python
def ingest_coverage_lines(
    gateway: StorageGateway,
    cfg: CoverageIngestStepConfig,
    *,
    tools: ToolsConfig | None = None,
    tool_service: ToolService | None = None,
    json_output_path: Path | None = None,
) -> None:
    """
    Read a `.coverage` database and populate `analytics.coverage_lines`.
    """
    repo_root = cfg.repo_root
    coverage_file = cfg.coverage_file

    if coverage_file is None or should_skip_missing_file(
        coverage_file, logger=log, label="coverage file"
    ):
        return

    now = datetime.now(UTC)
    active_tools = tools or ToolsConfig.model_validate({})
    service = tool_service
    if service is None:
        shared_runner = ToolRunner(
            tools_config=active_tools, cache_dir=cfg.repo_root / "build" / ".tool_cache"
        )
        service = ToolService(shared_runner, active_tools)
    json_path = json_output_path or (service.runner.cache_dir / "coverage.json")

    reports: list[CoverageFileReport] | None = None
    try:
        reports = asyncio.run(
            service.run_coverage_json(
                repo_root,
                coverage_file=coverage_file,
                output_path=json_path,
            )
        )
    except (ToolExecutionError, ToolNotFoundError) as exc:
        log.warning("coverage CLI failed; falling back to API parsing: %s", exc)

    source = "cli" if reports else "api"
    if reports:
        rows = _rows_from_reports(cfg, reports, now)
    else:
        rows = _collect_via_api(repo_root, coverage_file, cfg, now)

    if not rows:
        log.info("coverage_lines ingestion skipped (no rows) for %s@%s", cfg.repo, cfg.commit)
        return

    run_batch(
        gateway,
        "analytics.coverage_lines",
        [coverage_line_to_tuple(r) for r in rows],
        delete_params=[cfg.repo, cfg.commit],
    )
    log.info(
        "coverage_lines ingested for %s@%s rows=%d source=%s",
        cfg.repo,
        cfg.commit,
        len(rows),
        source,
    )
```

Plus helpers:

* `_rows_from_reports(cfg, reports, now) -> list[CoverageLineRow]`
* `_collect_via_api(repo_root, coverage_file, cfg, now) -> list[CoverageLineRow]`
* `_collect_file_coverage(...) -> list[CoverageLineRow]`
* `CoverageInsertContext`, `CoverageFileInfo`, etc.

This is a **pure full rebuild**: every time you call it, it wipes the rows for `repo@commit` and reinserts all coverage lines.

---

## 2. Step 1 – Factor the body into a reusable helper

First, we factor out the “collect all rows, decide source” logic into a helper that returns both rows and the source (“cli” vs “api”).

### 2.1. Add a helper `_collect_coverage_rows`

In `ingestion/coverage_ingest.py`, *above* `ingest_coverage_lines`, add:

```python
def _collect_coverage_rows(
    gateway: StorageGateway,
    cfg: CoverageIngestStepConfig,
    *,
    tools: ToolsConfig | None,
    tool_service: ToolService | None,
    json_output_path: Path | None,
    now: datetime,
) -> tuple[list[CoverageLineRow], str]:
    """
    Resolve coverage input (CLI JSON or coverage.py API) and return rows + source label.

    This helper is pure: it does not mutate DuckDB, it just returns the rows.
    """
    repo_root = cfg.repo_root
    coverage_file = cfg.coverage_file

    if coverage_file is None or should_skip_missing_file(
        coverage_file,
        logger=log,
        label="coverage file",
    ):
        return [], "missing"

    active_tools = tools or ToolsConfig.model_validate({})
    service = tool_service
    if service is None:
        shared_runner = ToolRunner(
            tools_config=active_tools,
            cache_dir=cfg.repo_root / "build" / ".tool_cache",
        )
        service = ToolService(shared_runner, active_tools)

    json_path = json_output_path or (service.runner.cache_dir / "coverage.json")

    reports: list[CoverageFileReport] | None = None
    try:
        reports = asyncio.run(
            service.run_coverage_json(
                repo_root,
                coverage_file=coverage_file,
                output_path=json_path,
            )
        )
    except (ToolExecutionError, ToolNotFoundError) as exc:
        log.warning("coverage CLI failed; falling back to API parsing: %s", exc)

    if reports:
        rows = _rows_from_reports(cfg, reports, now)
        source = "cli"
    else:
        rows = _collect_via_api(repo_root, coverage_file, cfg, now)
        source = "api"

    return rows, source
```

Nothing really changes yet; we just made the “collect rows” part explicit and reusable.

---

## 3. Step 2 – New `CoverageIngestOps` that uses full rebuild

Now we define an **ops class** that plugs into the incremental harness but **delegates to the full rebuild helper**.

Key idea:

* Implement `IncrementalIngestOps[CoverageLineRow]` + `SupportsFullRebuild`.
* In `run_full_rebuild`, call `_collect_coverage_rows` and then `run_batch` exactly as before.
* We do **not** attempt per-module incremental coverage; we always treat coverage as “full-rebuild dataset per run”.

Add near the bottom of `coverage_ingest.py` (e.g. after `_collect_file_coverage`), just above any `__all__` if you have one:

```python
from collections.abc import Iterable, Sequence  # near top of file if not present
from codeintel.ingestion.change_tracker import (  # near top of file
    ChangeTracker,
    IncrementalIngestOps,
    SupportsFullRebuild,
    run_incremental_ingest,
)
from codeintel.ingestion.common import ModuleRecord  # if not already imported


@dataclass
class CoverageIngestOps(IncrementalIngestOps[CoverageLineRow], SupportsFullRebuild):
    """
    Incremental ingest ops wrapper for analytics.coverage_lines.

    This implementation *always* performs a full rebuild via `run_full_rebuild`
    and does not attempt per-module incremental coverage, because coverage is
    a function of test runs rather than just code changes.
    """

    dataset_name: str
    cfg: CoverageIngestStepConfig
    tools: ToolsConfig | None
    tool_service: ToolService | None
    json_output_path: Path | None
    now: datetime

    def __init__(
        self,
        *,
        cfg: CoverageIngestStepConfig,
        tools: ToolsConfig | None,
        tool_service: ToolService | None,
        json_output_path: Path | None,
        now: datetime,
    ) -> None:
        self.dataset_name = "analytics.coverage_lines"
        self.cfg = cfg
        self.tools = tools
        self.tool_service = tool_service
        self.json_output_path = json_output_path
        self.now = now

    # These methods are never used because we always call run_full_rebuild(),
    # but we implement them to satisfy the protocol.

    @staticmethod
    def module_filter(module: ModuleRecord) -> bool:
        """Coverage is not tied to module diffs; we ignore per-module filtering."""
        return True

    def delete_rows(self, gateway: StorageGateway, rel_paths: Sequence[str]) -> None:
        """No-op: full rebuild handles deletion via run_batch(delete_params=[repo, commit])."""
        return

    @staticmethod
    def process_module(module: ModuleRecord) -> Iterable[CoverageLineRow]:
        """Not used; coverage operates as a full rebuild."""
        return []

    def insert_rows(self, gateway: StorageGateway, rows: Sequence[CoverageLineRow]) -> None:
        """Not used; run_full_rebuild handles insert via run_batch."""
        return

    # Full rebuild hook -----------------------------------------------------

    def run_full_rebuild(self, tracker: ChangeTracker) -> bool:
        """
        Perform a full rebuild of analytics.coverage_lines for cfg.repo@cfg.commit.

        Returns True so that run_incremental_ingest() knows no further work
        is required.
        """
        rows, source = _collect_coverage_rows(
            gateway=tracker.gateway,
            cfg=self.cfg,
            tools=self.tools,
            tool_service=self.tool_service,
            json_output_path=self.json_output_path,
            now=self.now,
        )
        if not rows:
            log.info(
                "coverage_lines ingestion skipped (no rows) for %s@%s",
                self.cfg.repo,
                self.cfg.commit,
            )
            return True

        run_batch(
            tracker.gateway,
            "analytics.coverage_lines",
            [coverage_line_to_tuple(r) for r in rows],
            delete_params=[self.cfg.repo, self.cfg.commit],
        )
        log.info(
            "coverage_lines ingested for %s@%s rows=%d source=%s",
            self.cfg.repo,
            self.cfg.commit,
            len(rows),
            source,
        )
        return True
```

This gives us a **formal IncrementalIngestOps** that’s fully encapsulated but intentionally full-rebuild.

---

## 4. Step 3 – Rewrite `ingest_coverage_lines` as a thin wrapper

Now we change `ingest_coverage_lines` to:

* Accept an **optional `tracker: ChangeTracker | None`**.
* If `tracker` is provided → run through the incremental harness, which will call `run_full_rebuild`.
* If `tracker` is `None` → call `_collect_coverage_rows` and `run_batch` directly (exactly what you do now).

### 4.1. Replace the old function with this new one

In `coverage_ingest.py`, replace the existing `ingest_coverage_lines` with:

```python
def ingest_coverage_lines(
    gateway: StorageGateway,
    cfg: CoverageIngestStepConfig,
    *,
    tools: ToolsConfig | None = None,
    tool_service: ToolService | None = None,
    json_output_path: Path | None = None,
    tracker: ChangeTracker | None = None,
) -> None:
    """
    Read a `.coverage` database and populate `analytics.coverage_lines`.

    Behaviour:
      * When `tracker` is None (legacy / tests): perform a full rebuild by
        computing coverage rows and replacing all rows for repo@commit.
      * When `tracker` is provided: still perform a full rebuild, but route
        the work through the incremental ingest harness using CoverageIngestOps.
        This keeps coverage ingestion consistent with other datasets without
        tying it incorrectly to code change diffs.
    """
    now = datetime.now(UTC)

    # Legacy full-rebuild path (no ChangeTracker; semantics match old behaviour)
    if tracker is None:
        rows, source = _collect_coverage_rows(
            gateway=gateway,
            cfg=cfg,
            tools=tools,
            tool_service=tool_service,
            json_output_path=json_output_path,
            now=now,
        )
        if not rows:
            log.info(
                "coverage_lines ingestion skipped (no rows) for %s@%s",
                cfg.repo,
                cfg.commit,
            )
            return

        run_batch(
            gateway,
            "analytics.coverage_lines",
            [coverage_line_to_tuple(r) for r in rows],
            delete_params=[cfg.repo, cfg.commit],
        )
        log.info(
            "coverage_lines ingested for %s@%s rows=%d source=%s",
            cfg.repo,
            cfg.commit,
            len(rows),
            source,
        )
        return

    # Harness-based full rebuild (ChangeTracker present)
    ops = CoverageIngestOps(
        cfg=cfg,
        tools=tools,
        tool_service=tool_service,
        json_output_path=json_output_path,
        now=now,
    )
    # We deliberately rely on run_full_rebuild and ignore per-module diffs.
    run_incremental_ingest(tracker, ops)
```

**Key properties:**

* **All existing call sites** that do not know about `tracker` keep working; their semantics are identical.
* Callers that *do* have a `ChangeTracker` can now call `ingest_coverage_lines(..., tracker=ctx.change_tracker)` and get the same full rebuild semantics, but with everything going through the common harness (so you can later add metadata recording, logging, etc., in one place).

---

## 5. Step 4 – Wire into the step registry (`CoverageIngestStep`)

In your `ingestion/steps.py` you already have something like:

```python
@dataclass(frozen=True)
class CoverageIngestStep:
    name: str = "coverage_ingest"
    description: str = "Load coverage.py data into analytics.coverage_lines."
    produces_tables: tuple[str, ...] = ("analytics.coverage_lines",)
    requires: tuple[str, ...] = ()

    def run(self, ctx: IngestionContextProtocol) -> None:
        log.debug("Running ingestion step %s", self.name)
        cfg = CoverageIngestStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
            coverage_file=ctx.active_tools.coverage_file,
            tool_runner=ctx.tool_runner,
        )
        runner = ctx.tool_runner or ToolRunner(
            cache_dir=ctx.paths.tool_cache,
            tools_config=ctx.active_tools,
        )
        service = ctx.tool_service or ToolService(runner, ctx.active_tools)
        coverage_ingest.ingest_coverage_lines(
            gateway=ctx.gateway,
            cfg=cfg,
            tools=ctx.active_tools,
            tool_service=service,
            json_output_path=ctx.paths.coverage_json,
        )
```

Update it to pass the tracker (which you already require for most other steps) and lean on the new `tracker` kwarg:

```python
    def run(self, ctx: IngestionContextProtocol) -> None:
        log.debug("Running ingestion step %s", self.name)
        cfg = CoverageIngestStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
            coverage_file=ctx.active_tools.coverage_file,
            tool_runner=ctx.tool_runner,
        )
        runner = ctx.tool_runner or ToolRunner(
            cache_dir=ctx.paths.tool_cache,
            tools_config=ctx.active_tools,
        )
        service = ctx.tool_service or ToolService(runner, ctx.active_tools)

        # Coverage is conceptually full-rebuild, but we still pass the tracker
        # so the dataset participates in the unified ingest harness.
        coverage_ingest.ingest_coverage_lines(
            gateway=ctx.gateway,
            cfg=cfg,
            tools=ctx.active_tools,
            tool_service=service,
            json_output_path=ctx.paths.coverage_json,
            tracker=ctx.change_tracker,
        )
```

If you want to be extra defensive, you can call `_require_change_tracker(ctx, self.name)` first and pass that value instead of `ctx.change_tracker`.

---

## 6. Tests to adjust / add

You don’t need to change existing tests that call `ingest_coverage_lines` directly; they’ll continue to go down the “no tracker” path.

You *can* add one new harness-level test to make sure the wiring is correct:

**File:** `tests/ingestion/test_coverage_ingest_incremental_wrapper.py` (or similar)

```python
from datetime import UTC, datetime
from pathlib import Path

from codeintel.config import CoverageIngestStepConfig, SnapshotRef
from codeintel.config.primitives import BuildPaths
from codeintel.ingestion.change_tracker import ChangeTracker, ChangeRequest, IncrementalIngestPolicy
from codeintel.ingestion.coverage_ingest import ingest_coverage_lines
from tests._helpers.gateway import open_memory_gateway_with_schema


def test_coverage_ingest_full_rebuild_via_tracker(tmp_path: Path) -> None:
    # Minimal in-memory gateway
    gw = open_memory_gateway_with_schema()

    # Fake snapshot + paths
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    coverage_file = repo_root / ".coverage"
    coverage_file.touch()  # empty but sufficient for should_skip_missing_file to pass
    snapshot = SnapshotRef(repo="demo", commit="abc123", repo_root=repo_root)
    paths = BuildPaths.for_snapshot(snapshot)

    cfg = CoverageIngestStepConfig(
        snapshot=snapshot,
        paths=paths,
        coverage_file=coverage_file,
        tool_runner=None,
    )

    # ChangeTracker is only used for wiring; full rebuild ignores per-module diffs.
    req = ChangeRequest.from_snapshot(snapshot, scan_profile=None, full_rebuild=True)  # type: ignore[arg-type]
    tracker = ChangeTracker.create(
        gw,
        change_request=req,
        modules=(),
        policy=IncrementalIngestPolicy(),
    )

    # Should not raise and should go through CoverageIngestOps.run_full_rebuild
    ingest_coverage_lines(
        gateway=gw,
        cfg=cfg,
        tools=None,
        tool_service=None,
        json_output_path=None,
        tracker=tracker,
    )
```

(That’s intentionally minimal; your actual test harness can use the real `.coverage` data & test helpers.)

---

## 7. Summary of what this buys you

* **Coverage ingestion is now expressed as a formal dataset step** (`CoverageIngestOps`, `CoverageIngestStep`) consistent with other ingestion datasets.
* You can **instrument all “incremental” datasets uniformly** (e.g., when you later add an `core.incremental_runs` table or logging hooks in `run_incremental_ingest`).
* **Semantics for coverage remain correct**: you still do a full rebuild whenever you ingest coverage, because coverage depends on test execution, not just on file-level diffs.

If you eventually want **true incremental coverage** (e.g. keyed on a hash of the `.coverage` file or a “coverage run id”), we can layer that on top by:

* Storing a coverage run digest + path set in a small metadata table.
* Comparing current digest vs previous.
* Only re-ingesting coverage_lines when that changes.

But that’s more of a dedicated “coverage run tracker” than a `ChangeTracker` concern, and I’d treat it as its own small epic.


