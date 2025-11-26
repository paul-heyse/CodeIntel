Yes, let’s tame the Context Hydra 🐍🐍🐍.

We’ll turn the three overlapping context types into a small, composable core, in a way that fits cleanly with the other refactors you’re doing (StorageGateway, ScanProfile, ToolService, ChangeTracker).

I’ll structure this as:

1. Goals & invariants
2. New core building blocks (configs & paths)
3. Implement the new core module
4. Refactor `ExportArgs` → use core configs
5. Refactor `IngestionContext`
6. Refactor `PipelineContext`
7. Wiring in Prefect flow / CLI
8. Migration strategy & tests

---

## 1. Goals & invariants

After this refactor:

* There is a **small set of core config objects**:

  * `SnapshotConfig` – “what code snapshot are we indexing?”
  * `StorageConfig` – “how do we open DuckDB?” (already defined)
  * `ExecutionConfig` – “how do we *run* this pipeline?”
  * `PathsConfig` / `BuildPaths` – “where do artifacts live under the build dir?”

* `IngestionContext`, `PipelineContext`, and `ExportArgs` become **thin wrappers/containers** around these types instead of duplicating fields.

* Any code that cares about:

  * Repo root, repo name, commit → uses `SnapshotConfig`
  * DuckDB path / flags → uses `StorageConfig` via `StorageGateway`
  * Tools & scan behavior → uses `ExecutionConfig` (+ `ScanProfile`)
  * Build outputs / temp files → uses `PathsConfig` / `BuildPaths`

This reduces your “what are the inputs to a run?” surface to 3–4 structs that are easy for both you and agents to reason about.

---

## 2. New core building blocks

We’ll create a new module, e.g.:

* `src/codeintel/core/config.py` or
* `src/codeintel/orchestration/config.py`

I’ll call it `core/config.py` below; feel free to put it under `orchestration` if that feels more natural.

### 2.1 `SnapshotConfig`

Single source of truth for *what code snapshot* this run is about.

```python
# src/codeintel/core/config.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple


@dataclass(frozen=True)
class SnapshotConfig:
    """
    Immutable description of which code snapshot we are analyzing.
    """
    repo_root: Path         # e.g. Path.cwd()
    repo_slug: str          # "paul-heyse/CodeIntel" or similar logical name
    commit: str             # commit SHA or "HEAD"
    branch: str | None = None
    # Optional: remote, tag, etc. if you want them.
```

This is what you’re currently spreading across `IngestionContext`, `PipelineContext`, and `ExportArgs`.

### 2.2 `ExecutionConfig`

Everything about *how* we run the pipeline (excluding storage).

This should reference the other refactors you’re putting in place:

* `ScanProfile` (from the scan config refactor)
* `ToolsConfig` (from the ToolService refactor)
* graph config, history config, etc.

```python
from codeintel.ingestion.source_scanner import ScanProfile
from codeintel.ingestion.tools_config import ToolsConfig
from codeintel.graphs.config import GraphBackendConfig  # or wherever this lives


@dataclass(frozen=True)
class ExecutionConfig:
    """
    Immutable description of runtime behavior for a pipeline run.
    """
    build_dir: Path
    tools: ToolsConfig
    code_profile: ScanProfile
    config_profile: ScanProfile
    graph_backend: GraphBackendConfig

    # History / temporal analytics
    history_db_dir: Path | None = None
    history_commits: Tuple[str, ...] = ()
    # Function override knobs (e.g. toggling heavy analytics) if you have them:
    function_overrides: Tuple[str, ...] = ()
```

You can grow this over time but try to keep it “configuration-only, no behavior”.

### 2.3 `PathsConfig` / `BuildPaths`

You already have a `BuildPaths` helper in `ingestion.runner` and `Prefect` has `_resolve_output_dir`, `_resolve_log_db_path`, etc.

We want a *single* canonical “path derivation” helper that relies only on `SnapshotConfig` + `ExecutionConfig`.

```python
@dataclass(frozen=True)
class PathsConfig:
    """
    Derived paths under build_dir for this snapshot.
    """
    snapshot: SnapshotConfig
    execution: ExecutionConfig

    @property
    def build_dir(self) -> Path:
        return self.execution.build_dir

    @property
    def coverage_json(self) -> Path:
        return self.build_dir / "coverage" / "coverage.json"

    @property
    def pytest_report(self) -> Path:
        return self.build_dir / "pytest" / "report.json"

    @property
    def scip_temp_dir(self) -> Path:
        return self.build_dir / "scip"

    @property
    def log_db_path(self) -> Path:
        return self.build_dir / "logs.duckdb"

    # You can add more:
    # - tool caches
    # - history DB path defaults
    # - per-snapshot subdirs if you want multi-snapshot builds
```

Once this exists, anything that needs “where do I put X output?” should be passed a `PathsConfig` (directly or through a context) and never reinvent path rules.

---

## 3. Implement the core module

### 3.1 Create `core/config.py` with the three types

Put the `SnapshotConfig`, `ExecutionConfig`, and `PathsConfig` above into `src/codeintel/core/config.py` (or `orchestration/config.py` – just be consistent).

### 3.2 Add factory helpers for common cases

To keep Prefect/CLI code clean, add some small constructors:

```python
@dataclass(frozen=True)
class SnapshotConfig:
    ...
    @classmethod
    def from_args(cls, repo_root: Path, repo_slug: str, commit: str, branch: str | None = None) -> "SnapshotConfig":
        return cls(
            repo_root=repo_root,
            repo_slug=repo_slug,
            commit=commit,
            branch=branch,
        )


@dataclass(frozen=True)
class ExecutionConfig:
    ...
    @classmethod
    def for_default_pipeline(
        cls,
        build_dir: Path,
        tools: ToolsConfig,
        code_profile: ScanProfile,
        config_profile: ScanProfile,
        graph_backend: GraphBackendConfig,
        *,
        history_db_dir: Path | None = None,
        history_commits: Sequence[str] | None = None,
        function_overrides: Sequence[str] | None = None,
    ) -> "ExecutionConfig":
        return cls(
            build_dir=build_dir,
            tools=tools,
            code_profile=code_profile,
            config_profile=config_profile,
            graph_backend=graph_backend,
            history_db_dir=history_db_dir,
            history_commits=tuple(history_commits or ()),
            function_overrides=tuple(function_overrides or ()),
        )
```

This makes orchestration code nice and declarative.

---

## 4. Refactor `ExportArgs` (Prefect flow entrypoint)

Right now `ExportArgs` is probably a flat dataclass with things like:

* `repo_root: Path`
* `repo: str`
* `commit: str | None`
* `db_path: Path`
* `build_dir: Path`
* `graph_backend: str`
* `history_commits: list[str]`
* etc.

We want `ExportArgs` to basically become a serialization-friendly wrapper around:

* `SnapshotConfig`
* `ExecutionConfig`
* `StorageConfig` (for DB details)

### 4.1 New shape for `ExportArgs`

In `orchestration/prefect_flow.py` (or wherever it lives):

```python
# src/codeintel/orchestration/prefect_flow.py

from dataclasses import dataclass
from pathlib import Path
from codeintel.core.config import SnapshotConfig, ExecutionConfig
from codeintel.storage.gateway import StorageConfig


@dataclass
class ExportArgs:
    """
    CLI/Prefect-friendly form of the pipeline configs.

    This is intentionally thin: most logic lives in SnapshotConfig,
    ExecutionConfig, StorageConfig, and PathsConfig.
    """
    # Primitive / CLI-friendly fields
    repo_root: Path
    repo_slug: str
    commit: str
    branch: str | None

    db_path: Path
    db_read_only: bool = False

    build_dir: Path | None = None
    history_db_dir: Path | None = None
    history_commits: tuple[str, ...] = ()

    # Graph backend selection, tools options, etc.
    graph_backend_name: str = "duckdb"  # or whatever default
    # tool-related flags / overrides (if you expose them via CLI)
    # e.g. pyright_path: Path | None, etc.
```

### 4.2 Add methods to construct core configs

Still inside `ExportArgs`:

```python
    def snapshot_config(self) -> SnapshotConfig:
        return SnapshotConfig(
            repo_root=self.repo_root,
            repo_slug=self.repo_slug,
            commit=self.commit,
            branch=self.branch,
        )

    def storage_config(self) -> StorageConfig:
        # This uses your improved StorageConfig from refactor #0
        return StorageConfig.for_ingest(self.db_path) if not self.db_read_only else StorageConfig.for_readonly(self.db_path)

    def execution_config(
        self,
        tools: ToolsConfig,
        code_profile: ScanProfile,
        config_profile: ScanProfile,
        graph_backend: GraphBackendConfig,
    ) -> ExecutionConfig:
        build_dir = self.build_dir or (self.repo_root / ".codeintel" / "build")

        return ExecutionConfig.for_default_pipeline(
            build_dir=build_dir,
            tools=tools,
            code_profile=code_profile,
            config_profile=config_profile,
            graph_backend=graph_backend,
            history_db_dir=self.history_db_dir,
            history_commits=self.history_commits,
        )
```

Now the Prefect flow can be:

```python
def _build_configs(args: ExportArgs) -> tuple[SnapshotConfig, ExecutionConfig, StorageConfig, PathsConfig]:
    snapshot = args.snapshot_config()

    tools = build_tools_config_from_env(...)
    code_profile = profile_from_env(default_code_profile(snapshot.repo_root))
    config_profile = profile_from_env(default_config_profile(snapshot.repo_root))
    graph_backend = GraphBackendConfig.from_name(args.graph_backend_name)

    execution = args.execution_config(
        tools=tools,
        code_profile=code_profile,
        config_profile=config_profile,
        graph_backend=graph_backend,
    )

    storage = args.storage_config()
    paths = PathsConfig(snapshot=snapshot, execution=execution)
    return snapshot, execution, storage, paths
```

This is the “one true” place where CLI/Prefect arguments enter the core config world.

---

## 5. Refactor `IngestionContext`

Today `IngestionContext` probably has a bunch of things like:

* `repo_root`
* `repo`
* `commit`
* `scan_config`
* `build_paths` / `build_dir`
* tool info, maybe graph backend, etc.

We’ll make it a tiny container around the core configs:

```python
# src/codeintel/ingestion/runner.py (or ingestion/context.py)

from dataclasses import dataclass
from codeintel.core.config import SnapshotConfig, ExecutionConfig, PathsConfig
from codeintel.storage.gateway import StorageGateway


@dataclass
class IngestionContext:
    """
    Context object passed to ingestion entrypoints.

    Thin wrapper around core configs + gateway.
    """
    snapshot: SnapshotConfig
    execution: ExecutionConfig
    paths: PathsConfig
    gateway: StorageGateway
```

If you like, you can keep convenience properties:

```python
    @property
    def repo_root(self) -> Path:
        return self.snapshot.repo_root

    @property
    def build_dir(self) -> Path:
        return self.execution.build_dir
```

But avoid duplicating data; just forward to underlying configs.

> **Important**: don’t put “behavior” in `IngestionContext`; it should just be “bag of config + gateway” for ingestion entrypoints.

---

## 6. Refactor `PipelineContext`

Similarly, `PipelineContext` in `orchestration/steps.py` can be simplified to:

```python
# src/codeintel/orchestration/steps.py

from dataclasses import dataclass
from codeintel.core.config import SnapshotConfig, ExecutionConfig, PathsConfig
from codeintel.storage.gateway import StorageGateway
from codeintel.ingestion.change_tracker import ChangeTracker
from codeintel.ingestion.tool_service import ToolService


@dataclass
class PipelineContext:
    """
    Context shared across pipeline steps.

    Composes core configs and runtime services (gateway, tools, change tracker).
    """
    snapshot: SnapshotConfig
    execution: ExecutionConfig
    paths: PathsConfig
    gateway: StorageGateway

    tool_service: ToolService
    change_tracker: ChangeTracker | None = None
```

Pipeline steps now read very clearly:

```python
@dataclass
class RepoScanStep(PipelineStep):
    def run(self, ctx: PipelineContext) -> None:
        tracker = ingest_repo(
            gateway=ctx.gateway,
            snapshot=ctx.snapshot,
            execution=ctx.execution,
            paths=ctx.paths,
        )
        ctx.change_tracker = tracker


@dataclass
class AstStep(PipelineStep):
    def run(self, ctx: PipelineContext) -> None:
        assert ctx.change_tracker is not None
        ingest_python_ast(
            tracker=ctx.change_tracker,
            execution=ctx.execution,
            paths=ctx.paths,
        )
```

You can decide whether to pass `snapshot` / `execution` separately into ingestion functions, or just give them `IngestionContext` (which wraps those configs). Either is fine; the crucial part is that the *data* lives in `SnapshotConfig` / `ExecutionConfig` / `PathsConfig`, not duplicated.

---

## 7. Wiring in Prefect flow / CLI

Once `ExportArgs` is updated and you have `_build_configs(args)`, the Prefect flow becomes:

```python
# src/codeintel/orchestration/prefect_flow.py

@flow(...)
def export_flow(args: ExportArgs) -> None:
    snapshot, execution, storage, paths = _build_configs(args)

    gateway = _get_gateway(storage)

    tools = build_tools_config_from_env(...)
    tool_runner = ToolRunner(tools)
    tool_service = ToolService(runner=tool_runner, tools_config=tools)

    ctx = PipelineContext(
        snapshot=snapshot,
        execution=execution,
        paths=paths,
        gateway=gateway,
        tool_service=tool_service,
        change_tracker=None,
    )

    try:
        run_pipeline(ctx)  # uses PIPELINE_STEPS & PipelineStep.run(ctx)
    finally:
        _close_gateways()
```

CLI entrypoints (Typer) mostly become “parse CLI args into `ExportArgs` and call `export_flow`”, with no additional structural complexity.

---

## 8. Migration strategy & tests

### 8.1 Stepwise migration

To avoid a huge bang-bang change:

1. **Introduce core configs** (`SnapshotConfig`, `ExecutionConfig`, `PathsConfig`) in a new file.

2. **Augment** existing contexts (add `snapshot`, `execution`, `paths` fields) while keeping old fields:

   ```python
   @dataclass
   class PipelineContext:
       snapshot: SnapshotConfig
       execution: ExecutionConfig
       paths: PathsConfig
       # old fields retained temporarily:
       repo_root: Path
       commit: str
       build_dir: Path
       ...
   ```

3. Gradually migrate ingestion & steps to use `ctx.snapshot/ctx.execution/ctx.paths` instead of the old fields.

4. Once nothing reads the old duplicates, delete them from the context dataclasses.

5. Finally, simplify `ExportArgs` to the new shape and move `_build_configs` into Prefect flow.

### 8.2 Basic tests

**Config roundtrip tests**

In `tests/orchestration/test_configs.py`:

* Construct an `ExportArgs`, call `snapshot_config()`, `execution_config()`, `storage_config()`, `PathsConfig(snapshot, execution)`, and assert that:

  * `paths.build_dir` is as expected.
  * `snapshot.repo_root` matches CLI arg.
  * `execution.code_profile.repo_root` is same as snapshot’s.
  * Env overrides for scan profiles are correctly integrated.

**Context sanity tests**

Add small tests for `PipelineContext`:

```python
def test_pipeline_context_path_forwarding(tmp_path: Path) -> None:
    snapshot = SnapshotConfig(repo_root=tmp_path, repo_slug="x", commit="HEAD")
    exec_cfg = ExecutionConfig.for_default_pipeline(
        build_dir=tmp_path / "build",
        tools=ToolsConfig(),
        code_profile=default_code_profile(tmp_path),
        config_profile=default_config_profile(tmp_path),
        graph_backend=GraphBackendConfig.default(),
    )
    paths = PathsConfig(snapshot=snapshot, execution=exec_cfg)

    ctx = PipelineContext(
        snapshot=snapshot,
        execution=exec_cfg,
        paths=paths,
        gateway=FakeGateway(),
        tool_service=FakeToolService(),
        change_tracker=None,
    )

    assert ctx.paths.build_dir == exec_cfg.build_dir
    assert ctx.snapshot.repo_root == snapshot.repo_root
```

**Architecture tests (optional but nice)**

Add tests that assert:

* `ExportArgs` doesn’t have fields like `scan_config` or duplicate `repo_root` in multiple forms.
* At most one `repo_root` and one `commit` exist per context (i.e. not `repo_root` + `snapshot.repo_root`).

---

If you’d like, I can next take one of the real existing classes (e.g. your current `PipelineContext` from `steps.py`) and sketch a *diff-style* “before/after” showing exactly how to embed `SnapshotConfig`, `ExecutionConfig`, and `PathsConfig` while keeping the rest of the file intact, so your AI coding agent can apply it almost literally.
