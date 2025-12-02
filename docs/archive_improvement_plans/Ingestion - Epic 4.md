Here’s a concrete implementation plan for **Epic 4 – canonical module inventory & scan profiles**, tied to your *current* code (the new zips you just attached).

I’ll focus on:

1. Making **`core.modules` + `ModuleRecord` + `iter_modules`** the canonical code inventory.
2. Making **`ScanProfile` from `IngestionContext`** the only way to control scanning.
3. Refactoring **typing ingestion** to use the module inventory instead of re-scanning.
4. Tightening **path normalization** (`rel_path` always POSIX).
5. Adding **tests** to lock this in and forbid regressions.

I’ll give patch-style snippets so you can drop them in with minimal editing.

---

## 0. Quick picture of the current state

From the new code:

* **Module inventory & change tracking**

  * `repo_scan._discover_modules`:

    * Uses `SourceScanner(code_profile)` to walk the tree.
    * For each `path`, calls `repo_relpath(repo_root, path)` → `rel_path`.
    * Builds `ModuleRecord(rel_path=rel_path, module_name=module, file_path=repo_root/rel_path, ...)`.
  * `common.iter_modules(module_map, repo_root, scan_profile=...)`:

    * `module_map` comes from `storage.module_index.load_module_map` which already normalizes paths with `normalize_rel_path`
    * Uses `scan_profile.include_globs` and `scan_profile.ignore_dirs` to filter.
    * Yields `ModuleRecord` with `file_path = repo_root / rel_path`.

* **Scanning**

  * `SourceScanner` in `source_scanner.py` wraps `os.walk` and globbing.
  * Used in:

    * `repo_scan._discover_modules` (for *code* inventory).
    * `config_ingest._iter_config_files` (for *config* files).
    * `typing_ingest._iter_python_files` (extra scan just for typing).

* **Scan profiles**

  * `ScanProfile` is in `source_scanner`.
  * `IngestionContext` holds `code_profile_cfg` and `config_profile_cfg` and exposes them as `.code_profile` / `.config_profile`.

* **Paths**

  * `normalize_rel_path(path)` returns POSIX-style `"foo/bar.py"` regardless of OS.
  * `repo_relpath(repo_root, path)` is used when converting absolute paths from tools to repo-relative strings.
  * `load_module_map` uses `normalize_rel_path` on `core.modules.path`

Goal now: **fully standardize on that combo** and stop doing ad-hoc scanning / path handling anywhere else.

---

## 1. Harden `core.modules` as *the* module inventory

You’re already close. The main “inventory oracle” is:

* `repo_scan.ingest_repo(...)` → writes `core.modules` and `core.repo_map`.
* `storage.module_index.load_module_map(...)` → reads `core.modules` and applies `normalize_rel_path`.
* `common.iter_modules(...)` → wraps module_map + `ScanProfile` into `ModuleRecord` iteration.

### 1.1. Ensure `_discover_modules` is the only code inventory scan

**File:** `ingestion/repo_scan.py`

You already have `_discover_modules`:

```python
def _discover_modules(
    repo_root: Path,
    code_profile: ScanProfile,
    cfg: RepoScanStepConfig,
    tags_entries: list[TagEntry],
) -> tuple[dict[str, ModuleRow], list[ModuleRecord]]:
    modules: dict[str, ModuleRow] = {}
    module_records: list[ModuleRecord] = []
    scanner = SourceScanner(code_profile)

    for idx, path in enumerate(scanner.iter_files(), start=1):
        rel_path = repo_relpath(repo_root, path)
        module = relpath_to_module(rel_path)
        tags = _tags_for_path(rel_path, tags_entries)
        ...
        module_records.append(
            ModuleRecord(
                rel_path=rel_path,
                module_name=module,
                file_path=repo_root / rel_path,
                index=idx,
                total=0,
            )
        )

    return modules, module_records
```

**Action:**

* Treat this as *the* place where we talk to the filesystem for code modules.
* In later steps we’ll remove `SourceScanner` usage from `typing_ingest`, so this remains the only code walk.

Optionally, you can also set `ModuleRecord.total` here for convenience:

```python
    total = idx  # after loop
    module_records = [
        ModuleRecord(
            rel_path=mr.rel_path,
            module_name=mr.module_name,
            file_path=mr.file_path,
            index=mr.index,
            total=total,
        )
        for mr in module_records
    ]
    return modules, module_records
```

That makes progress reporting in downstream jobs easier (and more consistent), but it’s not strictly required.

---

## 2. Canonical module iteration: always `load_module_map + iter_modules`

The canonical read path for code modules should be:

1. `module_map = load_module_map(gateway, repo, commit, language="python", logger=log)`
2. `for record in iter_modules(module_map, repo_root, scan_profile=code_profile): ...`

You already do this in `docstrings_ingest.ingest_docstrings`. We want to use the same pattern for any *code* dataset that needs a full scan (or that doesn’t participate in `ChangeTracker` yet).

We’ll leverage this heavily in **typing ingestion** (next section).

---

## 3. ScanProfile story: enforce use of `IngestionContext.code_profile/config_profile`

You already have:

* `IngestionContext.code_profile_cfg` / `config_profile_cfg`.
* Steps that accept a `ScanProfile`, like `docstrings_ingest.ingest_docstrings(..., code_profile: ScanProfile | None = None)`.

The rule going forward:

> **All code ingestion steps** should use `ctx.code_profile` (and only that) to determine which files are in-bounds.
> **Config ingestion** should use `ctx.config_profile`.

### 3.1. Steps: ensure we pass profiles consistently

**File:** `ingestion/steps.py`

Check and adjust these step `run()` methods:

#### Docstrings

You likely already have something like:

```python
docstrings_ingest.ingest_docstrings(
    ctx.gateway,
    cfg,
    code_profile=ctx.code_profile,
)
```

If not, change `code_profile` param to `ctx.code_profile`.

#### Typing

We’ll be refactoring typing ingestion below; but first, ensure the step passes `ctx.code_profile`:

```python
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
            code_profile=ctx.code_profile,   # <— enforce canonical profile
            tools=ctx.active_tools,
            tool_service=service,
        )
```

#### Config ingest

```python
config_ingest.ingest_config_values(
    ctx.gateway,
    cfg=cfg,
    config_profile=ctx.config_profile,     # <— use config_profile
)
```

Now the **only way** a profile enters an ingest function is via `ctx.code_profile` / `ctx.config_profile`.

---

## 4. Remove ad-hoc scanning from typing ingestion

Right now `typing_ingest` does its own scan:

```python
def _iter_python_files(profile: ScanProfile) -> Iterable[Path]:
    scanner = SourceScanner(profile)
    yield from scanner.iter_files()
```

…and `ingest_typing_signals` uses that:

```python
    repo_root = cfg.repo_root
    profile = code_profile or profile_from_env(default_code_profile(repo_root))
    ...
    annotation_info: dict[str, AnnotationInfo] = {}
    for path in _iter_python_files(profile):
        rel_path = repo_relpath(repo_root, path)
        info = _compute_annotation_info_for_file(path)
        ...
        annotation_info[rel_path] = info
```

We want typing ingestion to be driven by the **module inventory**, not a separate filesystem walk.

### 4.1. Use module_map + iter_modules

**File:** `ingestion/typing_ingest.py`

Add `load_module_map` + `iter_modules` imports:

```python
from codeintel.ingestion.common import run_batch, iter_modules, ModuleRecord
from codeintel.storage.module_index import load_module_map
```

(You can remove `SourceScanner` import and `_iter_python_files` once you’re done.)

Now refactor `ingest_typing_signals`:

#### BEFORE (simplified)

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
    active_service = tool_service or ToolService(...)

    annotation_info: dict[str, AnnotationInfo] = {}
    for path in _iter_python_files(profile):
        rel_path = repo_relpath(repo_root, path)
        info = _compute_annotation_info_for_file(path)
        ...

    error_maps = asyncio.run(_collect_error_maps(repo_root, active_service))
    path_set = union_of_keys(annotation_info, error_maps["pyrefly"], error_maps["pyright"], error_maps["ruff"])
    ...
```

#### AFTER — using `core.modules` + `iter_modules`

```python
def ingest_typing_signals(
    gateway: StorageGateway,
    cfg: TypingIngestStepConfig,
    *,
    code_profile: ScanProfile | None = None,
    tools: ToolsConfig | None = None,
    tool_service: ToolService | None = None,
) -> None:
    """
    Populate per-file typedness and static diagnostics.

      - analytics.typedness
      - analytics.static_diagnostics

    Notes
    -----
      * Pyrefly drives static error counts; annotation_ratio is computed from
        Python AST (params & returns).
      * Module inventory is derived from core.modules via load_module_map + iter_modules.
    """
    repo_root = cfg.repo_root
    profile = code_profile or profile_from_env(default_code_profile(repo_root))

    # Canonical module inventory: read from core.modules instead of re-scanning.
    module_map = load_module_map(
        gateway,
        cfg.repo,
        cfg.commit,
        language="python",
        logger=log,
    )
    annotation_info: dict[str, AnnotationInfo] = {}
    for record in iter_modules(
        module_map,
        repo_root,
        logger=log,
        scan_profile=profile,
    ):
        info = _compute_annotation_info_for_file(record.file_path)
        if info is not None:
            annotation_info[record.rel_path] = info

    active_tools = tools or ToolsConfig.model_validate({})
    active_service = tool_service
    if active_service is None:
        shared_runner = ToolRunner(
            tools_config=active_tools,
            cache_dir=repo_root / "build" / ".tool_cache",
        )
        active_service = ToolService(shared_runner, active_tools)

    error_maps = asyncio.run(_collect_error_maps(repo_root, active_service))
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

Then:

* **Delete** `_iter_python_files` and its `SourceScanner` import.
* Now **typing ingestion** is fully driven by `core.modules` + `ScanProfile`.

> Later, if you want to also wire it through `ChangeTracker` (Epic 2 extension), you can plug these `ModuleRecord`s into an `IncrementalIngestOps` like we sketched before. The important Epic 4 bit is: **no more independent filesystem scanning**.

---

## 5. Path normalization: make all `rel_path` POSIX and canonical

You already have:

* `normalize_rel_path(path: str | Path) -> str` returning `.as_posix()`
* `load_module_map` calling `normalize_rel_path` on `core.modules.path`

We want to ensure *all* ingestion paths obey that convention.

### 5.1. Tests ingest: normalize pytest node paths

**File:** `ingestion/tests_ingest.py`

You have:

```python
def _nodeid_to_path_and_qualname(nodeid: str) -> tuple[str, str | None]:
    parts = nodeid.split("::")
    rel_path = parts[0]
    qualname = "::".join(parts[1:]) if len(parts) > 1 else None
    return rel_path, qualname
```

On Windows, `rel_path` can contain backslashes. Normalize it:

```python
from codeintel.ingestion.paths import normalize_rel_path

def _nodeid_to_path_and_qualname(nodeid: str) -> tuple[str, str | None]:
    """
    Split a pytest nodeid into a normalized path and qualified test name.

    The path component is normalized to POSIX form (forward slashes) so it
    aligns with core.modules, coverage paths, and tool outputs.
    """
    parts = nodeid.split("::")
    raw_path = parts[0]
    rel_path = normalize_rel_path(raw_path)
    qualname = "::".join(parts[1:]) if len(parts) > 1 else None
    return rel_path, qualname
```

This ensures `analytics.test_catalog.rel_path` matches the rest of the ecosystem.

### 5.2. Coverage ingest: confirm normalization (already good)

**File:** `ingestion/coverage_ingest.py`

You already have:

```python
rel_path = normalize_rel_path(measured_path.relative_to(repo_root))
```

So coverage paths are POSIX, matching `core.modules` and tools.

### 5.3. ToolService parsers: always go through `repo_relpath`/`normalize_rel_path`

You’re already importing:

```python
from codeintel.ingestion.paths import normalize_rel_path, repo_relpath
```

The pattern to follow in any tool-output parser (pyright/pyrefly/ruff):

* Convert absolute or raw paths to repo-relative POSIX strings via:

```python
def _to_repo_relpath(repo_root: Path, path_str: str) -> str:
    return normalize_rel_path(repo_relpath(repo_root, Path(path_str)))
```

Then ensure `_parse_pyright_errors`, `_parse_pyrefly_errors`, `_parse_ruff_errors` all use this helper when mapping from tool JSON back to `rel_path`s.

If they already do something equivalent, just keep it; the important Epic 4 constraint is:

> **No code writes a path string into the DB without normalizing it via `normalize_rel_path` and/or `repo_relpath`.**

---

## 6. Static guard: no extra `SourceScanner` usage

We want **SourceScanner** only for:

* Code inventory in `repo_scan`.
* Config inventory in `config_ingest`.

You can add a small test that enforces this.

**File:** `tests/tests/ingestion/test_module_inventory.py` (new)

```python
from __future__ import annotations

from pathlib import Path

import inspect

import codeintel.ingestion.docstrings_ingest as docstrings_ingest
import codeintel.ingestion.typing_ingest as typing_ingest
import codeintel.ingestion.py_ast_extract as py_ast_extract
import codeintel.ingestion.cst_extract as cst_extract
import codeintel.ingestion.repo_scan as repo_scan
import codeintel.ingestion.config_ingest as config_ingest


def _source(module) -> str:
    return inspect.getsource(module)


def test_source_scanner_only_used_in_repo_scan_and_config_ingest() -> None:
    # Modules where SourceScanner is allowed.
    allowed = {
        "codeintel.ingestion.repo_scan",
        "codeintel.ingestion.config_ingest",
    }

    modules = {
        "codeintel.ingestion.repo_scan": repo_scan,
        "codeintel.ingestion.config_ingest": config_ingest,
        "codeintel.ingestion.docstrings_ingest": docstrings_ingest,
        "codeintel.ingestion.typing_ingest": typing_ingest,
        "codeintel.ingestion.py_ast_extract": py_ast_extract,
        "codeintel.ingestion.cst_extract": cst_extract,
    }

    offenders: list[str] = []
    for name, mod in modules.items():
        src = _source(mod)
        if "SourceScanner(" in src and name not in allowed:
            offenders.append(name)

    assert not offenders, f"SourceScanner used outside allowed modules: {offenders}"
```

You can add other ingestion modules to `modules` as needed; the principle is “explicit allowlist”.

---

## 7. Round-trip tests: repo scan → core.modules → iter_modules

Finally, add a test that validates the **round-trip** your epic calls out:

> repo scan → `core.modules` → `load_module_map` → `iter_modules` → dataset ingest.

Here’s a minimal example:

**File:** `tests/tests/ingestion/test_module_inventory.py` (same file, additional tests)

```python
from __future__ import annotations

from pathlib import Path

from codeintel.config import RepoScanStepConfig, SnapshotRef
from codeintel.config.primitives import BuildPaths
from codeintel.ingestion.common import iter_modules, ModuleRecord
from codeintel.ingestion.repo_scan import ingest_repo
from codeintel.ingestion.source_scanner import default_code_profile
from codeintel.storage.gateway import open_memory_gateway
from codeintel.storage.module_index import load_module_map


def _make_snapshot(tmp_path: Path) -> SnapshotRef:
    repo_root = tmp_path / "repo"
    src_dir = repo_root / "src" / "pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "a.py").write_text("print('a')\n", encoding="utf8")
    (src_dir / "b.py").write_text("print('b')\n", encoding="utf8")
    return SnapshotRef(repo="demo", commit="abc123", repo_root=repo_root)


def test_module_inventory_round_trip(tmp_path: Path) -> None:
    snapshot = _make_snapshot(tmp_path)
    paths = BuildPaths.for_snapshot(snapshot)
    gateway = open_memory_gateway(paths.db_path)

    cfg = RepoScanStepConfig(
        snapshot=snapshot,
        paths=paths,
        tool_runner=None,
    )
    profile = default_code_profile(snapshot.repo_root)

    tracker = ingest_repo(
        gateway,
        cfg=cfg,
        code_profile=profile,
        apply_schema=True,
    )

    module_map = load_module_map(
        gateway,
        cfg.repo,
        cfg.commit,
        language="python",
        logger=None,
    )
    records = list(iter_modules(module_map, snapshot.repo_root, logger=None, scan_profile=profile))

    # We expect two modules a.py and b.py in src/pkg
    rel_paths = sorted(r.rel_path for r in records)
    assert rel_paths == ["src/pkg/a.py", "src/pkg/b.py"]
    assert all("/" in rp for rp in rel_paths)  # POSIX-style

    # Ensure the change tracker uses the same module inventory.
    tracker_paths = sorted(m.rel_path for m in tracker.modules)
    assert tracker_paths == rel_paths
```

This test checks:

* `ingest_repo` wrote `core.modules` correctly.
* `load_module_map` + `iter_modules` reconstitute the same module inventory.
* All `rel_path` values use POSIX “/”.
* `ChangeTracker` uses the same module list.

---

## 8. What you get after Epic 4

Once you apply the above:

* **Single source of truth** for code modules:

  * `SourceScanner` → `core.modules` (via `repo_scan`).
  * Everything else → `core.modules` (via `load_module_map` + `iter_modules`).
* **Uniform scanning controls**:

  * Every code dataset uses `ctx.code_profile`.
  * Config ingest uses `ctx.config_profile`.
* **Path semantics guaranteed**:

  * All code `rel_path`s are POSIX and consistent across:

    * `core.modules`, `analytics.typedness`, `analytics.static_diagnostics`,
    * `core.docstrings`, `analytics.coverage_lines`, `analytics.test_catalog`,
    * tool outputs (pyright/pyrefly/ruff).
* **Static guardrails**:

  * Tests ensure no other module sneaks in a direct `SourceScanner` or `os.walk`.

If you want, next we can zoom in on **one concrete ingestion module** (e.g. `docstrings_ingest`) and show how to add a *tiny* integration test that asserts it’s using `iter_modules` + `ScanProfile` correctly (and not doing its own scanning).

# docstring ingest zoom in #

Perfect, let’s zoom in on `docstrings_ingest` and add a **small but very pointed integration test** (actually two) that:

1. Proves it honors **`ScanProfile`** when deciding which modules to process.
2. Proves it’s driven by the **`core.modules` inventory** (via `load_module_map + iter_modules`), not by its own filesystem scan.

I’ll assume your repo layout is the usual:

* `src/codeintel/...` → installed as `codeintel`
* tests live under `tests/`

so the test paths below would be `tests/tests/ingestion/...`.

---

## 1. Behaviour we want to lock in

From your current `docstrings_ingest.py`, the key bits are (simplified):

```python
from codeintel.config.builder import DocstringStepConfig
from codeintel.ingestion.common import (
    iter_modules,
    read_module_source,
    run_batch,
    should_skip_empty,
)
from codeintel.ingestion.source_scanner import ScanProfile
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.module_index import load_module_map
from codeintel.storage.rows import DocstringRow, docstring_row_to_tuple

...

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
        source = read_module_source(record, logger=log)
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

So:

* **Module inventory** comes from `core.modules` via `load_module_map`.
* `ScanProfile` is only used inside `iter_modules` to filter that inventory.
* No `SourceScanner` or `os.walk` inside `docstrings_ingest`.

We want our tests to **exercise exactly that contract**.

---

## 2. Helper: build a ScanProfile with extra ignores

We want to be able to say “ignore the `ignored/` directory” for scanning. That’s just a thin wrapper around `default_code_profile`.

Create a small helper inside the test file:

```python
# tests/tests/ingestion/test_docstrings_inventory.py

from __future__ import annotations

from pathlib import Path

from codeintel.ingestion.source_scanner import ScanProfile, default_code_profile


def _code_profile_ignoring_dir(snapshot_repo_root: Path, ignored_dir_name: str) -> ScanProfile:
    """
    Return a ScanProfile based on the default code profile, but with an
    additional ignored directory name.

    This purely adjusts the profile; it does *not* touch core.modules directly.
    """
    base = default_code_profile(snapshot_repo_root)
    ignore_dirs = base.ignore_dirs + (ignored_dir_name,)
    return ScanProfile(
        repo_root=base.repo_root,
        source_roots=base.source_roots,
        include_globs=base.include_globs,
        ignore_dirs=ignore_dirs,
        log_every=base.log_every,
        log_interval=base.log_interval,
    )
```

We’ll reuse this in both tests.

---

## 3. Test 1 – Docstrings respects `ScanProfile` + module inventory

This test makes sure **what you ask `ScanProfile` to ignore, docstrings actually ignores** (via the inventory).

```python
# tests/tests/ingestion/test_docstrings_inventory.py

from __future__ import annotations

from pathlib import Path

from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.config.builder import DocstringStepConfig, RepoScanStepConfig
from codeintel.ingestion.docstrings_ingest import ingest_docstrings
from codeintel.ingestion.repo_scan import ingest_repo
from tests._helpers.gateway import open_ingestion_gateway

from .test_docstrings_inventory import _code_profile_ignoring_dir  # or move helper above inside same file


def test_docstrings_respects_scan_profile_and_module_inventory(tmp_path: Path) -> None:
    # --- Arrange: synthetic repo with 3 modules, one under an ignored dir ---
    repo_root = tmp_path / "repo"
    src_pkg = repo_root / "src" / "pkg"
    src_ignored = repo_root / "src" / "ignored"
    src_pkg.mkdir(parents=True)
    src_ignored.mkdir(parents=True)

    (src_pkg / "a.py").write_text('"""doc A"""\n', encoding="utf8")
    (src_pkg / "b.py").write_text('"""doc B"""\n', encoding="utf8")
    (src_ignored / "c.py").write_text('"""ignored doc"""\n', encoding="utf8")

    snapshot = SnapshotRef(repo="demo/docstrings", commit="abc123", repo_root=repo_root)
    paths = BuildPaths.for_snapshot(snapshot)

    # Profile that *ignores* the "ignored" directory.
    code_profile = _code_profile_ignoring_dir(snapshot.repo_root, "ignored")

    gw = open_ingestion_gateway()

    # --- Step 1: repo_scan populates core.modules according to code_profile ---
    cfg_scan = RepoScanStepConfig(snapshot=snapshot, paths=paths)
    ingest_repo(
        gw,
        cfg=cfg_scan,
        code_profile=code_profile,
        apply_schema=True,
    )

    # --- Step 2: docstrings_ingest uses module inventory + same ScanProfile ---
    cfg_docs = DocstringStepConfig(snapshot=snapshot)
    ingest_docstrings(
        gw,
        cfg_docs,
        code_profile=code_profile,
    )

    # --- Assert: only src/pkg/a.py and src/pkg/b.py appear; src/ignored/c.py is omitted ---
    rows = gw.con.execute(
        "SELECT DISTINCT rel_path, module FROM core.docstrings ORDER BY rel_path"
    ).fetchall()
    rel_paths = [row[0] for row in rows]
    assert rel_paths == ["src/pkg/a.py", "src/pkg/b.py"]

    # Ensure paths are POSIX-style (sanity check for path normalization).
    assert all("/" in rp for rp in rel_paths)
```

What this confirms:

* The module inventory (`core.modules`) is filtered correctly by `ScanProfile` during the **repo scan**.
* `ingest_docstrings` respects that filtered inventory and doesn’t spontaneously “discover” `src/ignored/c.py`.

However, this could still pass if docstrings used *its own* `SourceScanner` with the same profile. So we add one more test to distinguish **inventory-driven** vs **filesystem-driven**.

---

## 4. Test 2 – Docstrings uses `core.modules`, not its own scan

To distinguish “uses module_map” vs “does its own walk”, we set up a scenario where:

* The filesystem contains `ghost.py`.
* `core.modules` does **not** contain `ghost.py` (we delete it manually after repo_scan).
* `ScanProfile` would *include* `ghost.py` if someone scanned the filesystem.

If docstrings ingestion uses `core.modules` exclusively, it will **never** see `ghost.py`, despite it being on disk and matching the profile.

```python
# tests/tests/ingestion/test_docstrings_inventory.py (continued)

def test_docstrings_uses_module_inventory_not_filesystem_scan(tmp_path: Path) -> None:
    # --- Arrange: repo with a visible module and a "ghost" module ---
    repo_root = tmp_path / "repo"
    src_pkg = repo_root / "src" / "pkg"
    src_pkg.mkdir(parents=True)

    (src_pkg / "visible.py").write_text('"""visible doc"""\n', encoding="utf8")
    (src_pkg / "ghost.py").write_text('"""ghost doc"""\n', encoding="utf8")

    snapshot = SnapshotRef(repo="demo/docstrings", commit="deadbeef", repo_root=repo_root)
    paths = BuildPaths.for_snapshot(snapshot)

    # Default profile: includes both visible.py and ghost.py.
    from codeintel.ingestion.source_scanner import default_code_profile

    code_profile = default_code_profile(snapshot.repo_root)

    gw = open_ingestion_gateway()

    # --- Step 1: run repo_scan to populate core.modules with BOTH modules ---
    cfg_scan = RepoScanStepConfig(snapshot=snapshot, paths=paths)
    ingest_repo(
        gw,
        cfg=cfg_scan,
        code_profile=code_profile,
        apply_schema=True,
    )

    # --- Step 2: delete ghost.py from core.modules but leave the file on disk ---
    gw.con.execute(
        """
        DELETE FROM core.modules
        WHERE repo = ? AND commit = ? AND path = ?
        """,
        [snapshot.repo, snapshot.commit, "src/pkg/ghost.py"],
    )

    # At this point:
    # - Filesystem: visible.py + ghost.py
    # - core.modules: only visible.py

    # --- Step 3: run docstrings ingestion with a profile that would include ghost.py ---
    cfg_docs = DocstringStepConfig(snapshot=snapshot)
    ingest_docstrings(
        gw,
        cfg_docs,
        code_profile=code_profile,  # this profile *would* include ghost.py if docstrings scanned itself
    )

    # --- Assert: docstrings only exist for visible.py ---
    rows = gw.con.execute(
        "SELECT DISTINCT rel_path FROM core.docstrings ORDER BY rel_path"
    ).fetchall()
    rel_paths = [row[0] for row in rows]
    assert rel_paths == ["src/pkg/visible.py"]

    # If docstrings_ingest were doing its own SourceScanner walk instead of
    # relying on core.modules + load_module_map + iter_modules, it would also
    # have emitted rows for src/pkg/ghost.py here.
```

This test specifically proves:

* Even when the **filesystem** and `ScanProfile` would include `ghost.py`, **docstrings ingestion only trusts `core.modules`**.
* That’s exactly the “single canonical module inventory” rule we want from Epic 4.

---

## 5. Where to put these tests

Create a new test file:

```text
tests/
  tests/
    ingestion/
      test_docstrings_inventory.py
```

With contents:

```python
from __future__ import annotations

from pathlib import Path

from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.config.builder import DocstringStepConfig, RepoScanStepConfig
from codeintel.ingestion.docstrings_ingest import ingest_docstrings
from codeintel.ingestion.repo_scan import ingest_repo
from codeintel.ingestion.source_scanner import ScanProfile, default_code_profile
from tests._helpers.gateway import open_ingestion_gateway


def _code_profile_ignoring_dir(snapshot_repo_root: Path, ignored_dir_name: str) -> ScanProfile:
    base = default_code_profile(snapshot_repo_root)
    ignore_dirs = base.ignore_dirs + (ignored_dir_name,)
    return ScanProfile(
        repo_root=base.repo_root,
        source_roots=base.source_roots,
        include_globs=base.include_globs,
        ignore_dirs=ignore_dirs,
        log_every=base.log_every,
        log_interval=base.log_interval,
    )


def test_docstrings_respects_scan_profile_and_module_inventory(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    src_pkg = repo_root / "src" / "pkg"
    src_ignored = repo_root / "src" / "ignored"
    src_pkg.mkdir(parents=True)
    src_ignored.mkdir(parents=True)

    (src_pkg / "a.py").write_text('"""doc A"""\n', encoding="utf8")
    (src_pkg / "b.py").write_text('"""doc B"""\n', encoding="utf8")
    (src_ignored / "c.py").write_text('"""ignored doc"""\n', encoding="utf8")

    snapshot = SnapshotRef(repo="demo/docstrings", commit="abc123", repo_root=repo_root)
    paths = BuildPaths.for_snapshot(snapshot)

    code_profile = _code_profile_ignoring_dir(snapshot.repo_root, "ignored")
    gw = open_ingestion_gateway()

    cfg_scan = RepoScanStepConfig(snapshot=snapshot, paths=paths)
    ingest_repo(
        gw,
        cfg=cfg_scan,
        code_profile=code_profile,
        apply_schema=True,
    )

    cfg_docs = DocstringStepConfig(snapshot=snapshot)
    ingest_docstrings(
        gw,
        cfg_docs,
        code_profile=code_profile,
    )

    rows = gw.con.execute(
        "SELECT DISTINCT rel_path, module FROM core.docstrings ORDER BY rel_path"
    ).fetchall()
    rel_paths = [row[0] for row in rows]
    assert rel_paths == ["src/pkg/a.py", "src/pkg/b.py"]
    assert all("/" in rp for rp in rel_paths)


def test_docstrings_uses_module_inventory_not_filesystem_scan(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    src_pkg = repo_root / "src" / "pkg"
    src_pkg.mkdir(parents=True)

    (src_pkg / "visible.py").write_text('"""visible doc"""\n', encoding="utf8")
    (src_pkg / "ghost.py").write_text('"""ghost doc"""\n', encoding="utf8")

    snapshot = SnapshotRef(repo="demo/docstrings", commit="deadbeef", repo_root=repo_root)
    paths = BuildPaths.for_snapshot(snapshot)

    code_profile = default_code_profile(snapshot.repo_root)
    gw = open_ingestion_gateway()

    cfg_scan = RepoScanStepConfig(snapshot=snapshot, paths=paths)
    ingest_repo(
        gw,
        cfg=cfg_scan,
        code_profile=code_profile,
        apply_schema=True,
    )

    # Remove ghost module from the canonical inventory but leave the file.
    gw.con.execute(
        """
        DELETE FROM core.modules
        WHERE repo = ? AND commit = ? AND path = ?
        """,
        [snapshot.repo, snapshot.commit, "src/pkg/ghost.py"],
    )

    cfg_docs = DocstringStepConfig(snapshot=snapshot)
    ingest_docstrings(
        gw,
        cfg_docs,
        code_profile=code_profile,
    )

    rows = gw.con.execute(
        "SELECT DISTINCT rel_path FROM core.docstrings ORDER BY rel_path"
    ).fetchall()
    rel_paths = [row[0] for row in rows]
    assert rel_paths == ["src/pkg/visible.py"]
```

With these in place, `docstrings_ingest` is now:

* **Integration-tested** against the canonical inventory + `ScanProfile`.
* **Protected** against regression to ad-hoc filesystem scanning (if someone reintroduced `SourceScanner` into this module in the future, the second test would fail).
