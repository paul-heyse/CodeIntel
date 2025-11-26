Love this axis—getting scan config in one place is going to make everything else feel much saner.

I’ll follow the same pattern as before:

1. Goals & invariants
2. New abstractions (`ScanProfile`, helpers)
3. Refactor `source_scanner`
4. Refactor contexts (`IngestionContext` / `PipelineContext`)
5. Refactor Prefect / CLI env overrides
6. Refactor consumers (`repo_scan`, `typing_ingest`, `config_ingest`, etc.)
7. Clean up old configs / constants
8. Tests & guardrails

I’ll keep it concrete so an AI agent can implement it without guessing.

---

## 1. Goals & invariants

After this refactor:

* There is **one canonical place** where “what files matter?” is defined.
* All ingestion steps that walk the repo (AST, CST, typing, config, docstrings, etc.) use a `ScanProfile` → `SourceScanner` pipeline.
* All environment overrides (`CODEINTEL_INCLUDE_PATTERNS`, `CODEINTEL_IGNORE_DIRS`, etc.) are applied **once**, not reinterpreted by each module.
* There is exactly **one** authoritative `IGNORE_DIRS` constant, colocated with `ScanProfile` / `SourceScanner`.

---

## 2. New abstractions

Create a new module or extend `ingestion/source_scanner.py`:

### 2.1 `ScanProfile`

```python
# src/codeintel/ingestion/source_scanner.py

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple


@dataclass(frozen=True)
class ScanProfile:
    """
    A reusable description of how to scan a repository for a certain kind of file.
    """
    repo_root: Path
    source_roots: Tuple[Path, ...]             # often (repo_root / "src",)
    include_globs: Tuple[str, ...]             # e.g. ("**/*.py",)
    ignore_dirs: Tuple[str, ...]               # directory names to skip anywhere
    log_every: int = 250                       # logging cadence for SourceScanner
```

You can tune the exact fields; the key is that all scanning rules live in this struct.

### 2.2 Default profiles

Add canonical “code” and “config” profiles:

```python
DEFAULT_IGNORE_DIRS: Tuple[str, ...] = (
    ".git",
    ".venv",
    "venv",
    ".mypy_cache",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "node_modules",
    ".DS_Store",
    # add others you currently ignore
)


def default_code_profile(repo_root: Path) -> ScanProfile:
    src_root = repo_root / "src"
    roots = (src_root,) if src_root.exists() else (repo_root,)
    return ScanProfile(
        repo_root=repo_root,
        source_roots=roots,
        include_globs=("**/*.py",),
        ignore_dirs=DEFAULT_IGNORE_DIRS,
        log_every=250,
    )


def default_config_profile(repo_root: Path) -> ScanProfile:
    # Config files are often spread around; use repo_root as root but same ignores.
    return ScanProfile(
        repo_root=repo_root,
        source_roots=(repo_root,),
        include_globs=("**/*.toml", "**/*.ini", "**/*.cfg", "**/*.yaml", "**/*.yml"),
        ignore_dirs=DEFAULT_IGNORE_DIRS,
        log_every=250,
    )
```

If you have more nuanced separation (tests, docs), you can add e.g. `default_test_profile`.

### 2.3 Env-based override helper

You already have some `_scan_from_env(...)` logic in Prefect; we want a single place to interpret those env vars into a modified `ScanProfile`:

```python
import os


def profile_from_env(base: ScanProfile) -> ScanProfile:
    """
    Apply CODEINTEL_INCLUDE_PATTERNS / CODEINTEL_IGNORE_DIRS overrides
    to a base profile.
    """
    include = base.include_globs
    ignore = list(base.ignore_dirs)

    raw_include = os.getenv("CODEINTEL_INCLUDE_PATTERNS")
    if raw_include:
        # comma or colon separated, strip whitespace
        include = tuple(p.strip() for p in raw_include.replace(";", ",").split(",") if p.strip())

    raw_ignore = os.getenv("CODEINTEL_IGNORE_DIRS")
    if raw_ignore:
        extra_ignores = [d.strip() for d in raw_ignore.replace(";", ",").split(",") if d.strip()]
        # ensure uniqueness but preserve order
        seen: set[str] = set(ignore)
        for d in extra_ignores:
            if d not in seen:
                ignore.append(d)
                seen.add(d)

    return ScanProfile(
        repo_root=base.repo_root,
        source_roots=base.source_roots,
        include_globs=include,
        ignore_dirs=tuple(ignore),
        log_every=base.log_every,
    )
```

Now **all** env override behavior is here; nothing else reads those env vars.

---

## 3. Refactor `SourceScanner` to take a `ScanProfile`

Today `SourceScanner` probably looks something like:

```python
class SourceScanner:
    def __init__(self, root: Path, include_patterns: Sequence[str], ignore_dirs: Sequence[str], log_every: int = 250) -> None:
        ...
```

Change it to:

```python
# src/codeintel/ingestion/source_scanner.py

from typing import Iterator


class SourceScanner:
    def __init__(self, profile: ScanProfile) -> None:
        self.profile = profile

    def iter_files(self) -> Iterator[Path]:
        """
        Yield paths under profile.source_roots matching include_globs and skipping ignore_dirs.
        """
        # Example implementation using pathlib.glob; if you currently use os.walk, adapt it.
        yielded = 0
        ignore_set = set(self.profile.ignore_dirs)

        for root in self.profile.source_roots:
            for pattern in self.profile.include_globs:
                for path in root.glob(pattern):
                    # Skip any path containing an ignored directory segment
                    if any(part in ignore_set for part in path.parts):
                        continue
                    yield path
                    yielded += 1
                    if yielded % self.profile.log_every == 0:
                        # log progress via your logging infra
                        ...
```

If your current implementation is `os.walk`-based with more nuanced behavior (e.g. symlinks), keep that logic but use `profile.source_roots`, `profile.include_globs`, `profile.ignore_dirs`.

---

## 4. Refactor contexts: `IngestionContext` / `PipelineContext`

Right now both contexts carry a `scan_config: ScanConfig | None`, and each consumer often re-derives globs or ignore lists.

We want them to instead carry **profiles**, not raw pieces, so that:

* `repo_scan`, `typing_ingest`, `config_ingest`, etc. just pick the right profile and feed it to `SourceScanner`.

### 4.1 Add profiles to contexts

In `orchestration/context.py` or `steps.py` (where `PipelineContext` lives):

```python
from codeintel.ingestion.source_scanner import ScanProfile

@dataclass
class IngestionContext:
    repo_root: Path
    code_profile: ScanProfile
    config_profile: ScanProfile
    # ... any others (test_profile, docs_profile, etc.)
```

`PipelineContext` then either embeds this or refers to them directly:

```python
@dataclass
class PipelineContext:
    gateway: StorageGateway
    snapshot: SnapshotConfig
    ingestion: IngestionContext
    # ...
```

### 4.2 Construct profiles once at pipeline creation

In `prefect_flow.py` (or wherever you build the pipeline args):

```python
from codeintel.ingestion.source_scanner import (
    default_code_profile,
    default_config_profile,
    profile_from_env,
)

def _build_ingestion_context(repo_root: Path) -> IngestionContext:
    base_code = default_code_profile(repo_root)
    base_config = default_config_profile(repo_root)

    # Apply env overrides *once* each
    code_profile = profile_from_env(base_code)
    config_profile = profile_from_env(base_config)

    return IngestionContext(
        repo_root=repo_root,
        code_profile=code_profile,
        config_profile=config_profile,
    )
```

Then `PipelineContext` gets this `IngestionContext`:

```python
ingestion_ctx = _build_ingestion_context(args.repo_root)

ctx = PipelineContext(
    gateway=gateway,
    snapshot=snapshot,
    ingestion=ingestion_ctx,
    # ...
)
```

Now no other code should ever parse `CODEINTEL_INCLUDE_PATTERNS` or `CODEINTEL_IGNORE_DIRS`.

---

## 5. Refactor consumers

### 5.1 `repo_scan` – code modules

`repo_scan.ingest_repo` currently does something like:

* Build a `SourceScanner` with its own `ScanConfig` or arguments
* Iterate files to populate `core.modules` / `core.repo_map`

Refactor to:

```python
# src/codeintel/ingestion/repo_scan.py

from codeintel.ingestion.source_scanner import SourceScanner

def ingest_repo(
    gateway: StorageGateway,
    ctx: IngestionContext,
    change_request: ChangeRequest | None,
) -> ChangeTracker:
    scanner = SourceScanner(ctx.code_profile)

    modules: list[ModuleRecord] = []
    for path in scanner.iter_files():
        rel_path = path.relative_to(ctx.repo_root).as_posix()
        modules.append(build_module_record_from_path(rel_path))
        # write to core.modules / core.repo_map as you currently do

    # then create ChangeTracker as per previous plan
    tracker = ChangeTracker.create(
        gateway=gateway,
        snapshot=ctx.snapshot,
        modules=modules,
        change_request=change_request,
    )
    return tracker
```

The only scanning logic here is *which profile* we pass into `SourceScanner`.

### 5.2 `typing_ingest` – code files

Right now `typing_ingest`:

* Re-declares `IGNORE_DIRS`
* Has `_iter_python_files(repo_root)` that either uses `SourceScanner` or own walk

Replace that with direct use of `IngestionContext.code_profile`:

```python
# src/codeintel/ingestion/typing_ingest.py

from codeintel.ingestion.source_scanner import SourceScanner

def _iter_python_files(ctx: IngestionContext) -> Iterable[Path]:
    # In fact, you can inline this; but keep as a small helper if you like.
    scanner = SourceScanner(ctx.code_profile)
    return scanner.iter_files()
```

Then in `ingest_typing_signals`:

```python
def ingest_typing_signals(
    gateway: StorageGateway,
    tool_service: ToolService,
    ctx: IngestionContext,
) -> None:
    python_files = list(_iter_python_files(ctx))
    # The rest is your existing AST / tool-based logic
```

**Delete** the redundant `IGNORE_DIRS` constant and old `_iter_python_files` implementation.

### 5.3 `config_ingest` – config files

Currently `config_ingest`:

* Imports `IGNORES` from `source_scanner`
* Has its own `_iter_config_files` with own logic

Replace with:

```python
# src/codeintel/ingestion/config_ingest.py

from codeintel.ingestion.source_scanner import SourceScanner

def _iter_config_files(ctx: IngestionContext) -> Iterable[Path]:
    scanner = SourceScanner(ctx.config_profile)
    return scanner.iter_files()
```

Then in `ingest_config`:

```python
def ingest_config(
    gateway: StorageGateway,
    ctx: IngestionContext,
) -> None:
    config_files = list(_iter_config_files(ctx))
    # parse + write config into DuckDB as before
```

Now **all** config scanning is governed by `default_config_profile` and its env overrides.

### 5.4 Other consumers: AST/CST/docstrings/tests

Anywhere else you currently:

* Manually call `SourceScanner(...)` with `root/include_patterns/ignore_dirs`, or
* Use `os.walk` / pathlib glob directly for “all Python files under src”

should be switched to the same pattern:

* For code-like artifacts (AST, CST, docstrings): use `ctx.code_profile`.
* For config-like artifacts: use `ctx.config_profile`.

Examples:

```python
# AST/CST ingestion
scanner = SourceScanner(ctx.code_profile)
for path in scanner.iter_files():
    ...

# Docstrings ingestion
scanner = SourceScanner(ctx.code_profile)
for path in scanner.iter_files():
    ...
```

If something has genuinely different semantics (e.g. scanning only tests), you can:

* Either pass a filtered list of modules, or
* Introduce a `default_tests_profile(repo_root)` and store `tests_profile` in `IngestionContext`.

But still: all scanning should go through `ScanProfile` + `SourceScanner`.

---

## 6. Remove / unify old `ScanConfig`

You currently have a `ScanConfig` struct used in multiple places. There are two reasonable options:

1. **Deprecate `ScanConfig` entirely** and migrate all code to `ScanProfile`.
2. **Re-implement `ScanConfig` as a thin wrapper / alias around `ScanProfile`** during a transition period.

I’d suggest option 1, but a gentle migration could look like:

```python
# old interface kept for compatibility
@dataclass(frozen=True)
class ScanConfig:
    root: Path
    include_patterns: tuple[str, ...]
    ignore_dirs: tuple[str, ...]
    log_every: int = 250

    def to_profile(self) -> ScanProfile:
        return ScanProfile(
            repo_root=self.root,
            source_roots=(self.root,),
            include_globs=self.include_patterns,
            ignore_dirs=self.ignore_dirs,
            log_every=self.log_every,
        )
```

Then gradually replace all `ScanConfig` fields in contexts with `ScanProfile` and delete `ScanConfig` once nothing uses it.

---

## 7. Tests & guardrails

### 7.1 Unit tests for `ScanProfile` helpers

Create tests like:

```python
def test_default_code_profile_uses_src_if_present(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    profile = default_code_profile(tmp_path)
    assert profile.source_roots == (tmp_path / "src",)

def test_default_code_profile_falls_back_to_repo_root(tmp_path: Path) -> None:
    profile = default_code_profile(tmp_path)
    assert profile.source_roots == (tmp_path,)

def test_profile_from_env_includes_and_ignores(monkeypatch) -> None:
    base = default_code_profile(Path("/repo"))
    monkeypatch.setenv("CODEINTEL_INCLUDE_PATTERNS", "**/*.py, **/*.pyi")
    monkeypatch.setenv("CODEINTEL_IGNORE_DIRS", "build,dist")
    profile = profile_from_env(base)
    assert ("**/*.pyi" in profile.include_globs)
    assert ("build" in profile.ignore_dirs and "dist" in profile.ignore_dirs)
```

### 7.2 Integration tests for `SourceScanner`

Use a small fake repo tree:

* Create nested dirs including ignored and non-ignored ones.
* Check that `SourceScanner(profile).iter_files()` yields exactly the expected paths.

Example:

```python
def test_source_scanner_respects_ignore_dirs(tmp_path: Path) -> None:
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "a.py").write_text("...")
    (tmp_path / "src" / "node_modules" / "b.py").mkdir(parents=True)
    (tmp_path / "src" / "node_modules" / "b.py").write_text("...")

    profile = default_code_profile(tmp_path)
    scanner = SourceScanner(profile)
    paths = {p.relative_to(tmp_path).as_posix() for p in scanner.iter_files()}

    assert "src/pkg/a.py" in paths
    assert all("node_modules" not in p for p in paths)
```

### 7.3 Architecture test: no stray scan config

Add a simple test that:

* Searches for multiple definitions of `IGNORE_DIRS` in `src/codeintel`.
* Fails if any module other than `source_scanner.py` defines its own.

Example:

```python
def test_ignore_dirs_defined_only_in_source_scanner() -> None:
    for path, text in _iter_python_files(Path("src/codeintel")):
        if "source_scanner.py" in str(path):
            continue
        if "IGNORE_DIRS" in text or "DEFAULT_IGNORE_DIRS" in text:
            pytest.fail(f"IGNORES should only be defined in source_scanner: found in {path}")
```

You can also:

* Fail if a module outside `ingestion/source_scanner.py` calls `os.walk` + `fnmatch` to filter `*.py`—to gently nudge everything toward `SourceScanner`.

---

If you’d like, we can next pick a specific consumer like `typing_ingest.py` and I can sketch a diff-style “before/after” for the scanning bits, so you can drop that straight into your AI coding agent as a concrete patch.
