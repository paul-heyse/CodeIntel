Below is a concrete Cyclopts-based redesign for your CLI, tailored to the structure of the code you shared:

* It **keeps your existing business logic** (handlers, project runtime, build system, datasets, docs, etc.).
* It **replaces Typer for parsing / dispatch** with Cyclopts.
* It **bundles arguments into small, explicitly typed dataclasses**, so command functions stay low-complexity.
* It’s structured so an LLM agent can systematically migrate & extend it.

I’ll show:

1. A quick map of your current Typer CLI (so the new layout is easy to line up).
2. New shared CLI types (`runtime`, `output`, `scopes`) via Cyclopts + dataclasses.
3. A new root `App` and sub-apps that mirror your existing `codeintel` CLI.
4. Full example implementation for **build** and **op/dataset/serve**.
5. Skeletons for docs/datasets/graphs/storage/history/ide/subsystem with clear patterns.

---

## 1. Current CLI structure (from your code)

From the files you provided:

* **Root app** (`cli/__init__.py`)

  * `app = typer.Typer(name="codeintel", ...)`
  * `app.add_typer(build_app, name="build")`
  * `app.add_typer(op_app, name="op")`
  * `app.add_typer(dataset_app, name="dataset")`
  * `app.add_typer(serve_app, name="serve")`
  * plus domain apps: `graphs_app`, `docs_app`, `storage_app`, `history_app`, `ide_app`, `subsystem_app`, `datasets_ext_app`.

* **Core shared glue** (`cli/commands/_common.py`)

  * Shared `typer.Option` aliases: `ProjectRootOpt`, `VerboseOpt`, `JsonOutputOpt`, `RepoRootOpt`, `DbPathOpt`, `BuildDirOpt`, `DocumentOutputDirOpt`, etc.
  * Dataclasses: `BackendFlags`, `RuntimeCliOptions`, `RepoSelection`, `PathSelection`, `RuntimeSelection`, etc.
  * Helpers like `build_runtime_or_exit`, `setup_logging`, etc.

* **Build CLI** (`cli/commands/build.py`)

  * `build_app = typer.Typer(name="build", ...)`
  * Dataclasses: `BuildRunOptions`, `BuildRunContext`.
  * `build_run_handler(options: BuildRunOptions, ctx_opts: BuildRunContext) -> None`.
  * `build_status_handler(...)`, `build_history` commands.
  * Typer wrappers built via `_option_shim.wrap_command` and `OptionSpec`.

* **Ops/dataset/serve** (`cli/main.py` + `cli/op_params.py`)

  * `op_app = typer.Typer(...)` and `op_list`, `op_show`, `op_run` etc.
  * `dataset_app = typer.Typer(...)` with `dataset_verify`.
  * `serve_app = typer.Typer(...)` with HTTP server + MCP server commands.
  * `op_params.py` provides dynamic op CLI generation (`CliParamSpec`, `DynamicCommandConfig`, `build_dynamic_command`, `register_dynamic_commands`, etc.), using a string-tunnel pattern.

* **Other domain CLIs**

  * `docs_app` (`cli/commands/docs.py`) – docs export.
  * `datasets_ext_app` (`cli/commands/datasets.py`) – extended dataset management.
  * `graphs_app` (`cli/commands/graphs.py`) – graph metrics and plugins.
  * `storage_app` (`cli/commands/storage.py`) – storage validation.
  * `history_app` (`cli/commands/history.py`) – timeseries & history.
  * `ide_app` (`cli/commands/ide.py`) – IDE hints.
  * `subsystem_app` (`cli/commands/subsystem.py`) – subsystem exploration.

The key constraint you mentioned (and that shows up in the code) is that you’ve had to build:

* `_option_shim` to keep function signatures small (handlers) while satisfying Typer.
* A string-tunnel in `op_params.py` because Typer can’t handle your union/kwargs use-cases.

Cyclopts lets us **drop `_option_shim` and the Typer-specific wiring** but keep the handlers and “string tunnel” coercion logic if desired.

---

## 2. Shared Cyclopts CLI types (`cli/cyclopts_common.py`)

Create a new module (e.g. `cli/cyclopts_common.py`, or replace parts of `_common.py` if you prefer) encapsulating **typed CLI config** and Cyclopts `Parameter` metadata.

```python
# cli/cyclopts_common.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter, validators
from cyclopts.group import Group  # or from cyclopts import Group, depending on version

from codeintel.cli.project import (
    ProjectRuntime,
    ProjectNotFoundError,
    build_project_runtime,
)
from codeintel.config import (
    CodeIntelConfig,
    SnapshotInit,
    SnapshotRef,
    BuildPaths,
    ToolsConfig,
    resolve_tools_config,
    resolve_graph_backend,
)

# -----------------------------------------------------------------------------
# Global app-level defaults
# -----------------------------------------------------------------------------

def make_root_app() -> App:
    """
    Construct the root Cyclopts App with global defaults.

    - No negative flags by default (we’ll opt-in on specific bools).
    - Show defaults in help.
    - Return values instead of sys.exit in tests if configured.
    """
    return App(
        name="codeintel",
        help="CodeIntel unified CLI for build, analytics, and serving operations.",
        default_parameter=Parameter(
            show_default=True,
        ),
        # For production, keep default result_action (print + exit).
        # For tests, construct a separate App with result_action="return_value".
    )

# -----------------------------------------------------------------------------
# Output & verbosity
# -----------------------------------------------------------------------------

class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"

Verbose = Annotated[
    int,
    Parameter(
        name=["--verbose", "-v"],
        help="Increase verbosity (can be repeated: -v=INFO, -vv=DEBUG).",
        # We'll interpret via a small helper; Cyclopts doesn't need count=.
    ),
]

OutputFmt = Annotated[
    OutputFormat,
    Parameter(
        name="--output-format",
        help="Output format.",
        show_choices=True,
        case_sensitive=False,
    ),
]

JsonFlag = Annotated[
    bool,
    Parameter(
        name="--json",
        help="Alias for --output-format json.",
        negative=(),  # do not auto-generate --no-json
    ),
]

# -----------------------------------------------------------------------------
# Repo / paths / runtime selection
# -----------------------------------------------------------------------------

RepoStr = Annotated[
    str | None,
    Parameter(
        name="--repo",
        help="Repository slug (e.g., org/repo). Uses codeintel.yaml if omitted.",
    ),
]

CommitStr = Annotated[
    str | None,
    Parameter(
        name="--commit",
        help="Commit SHA. Auto-detected from git or project config if omitted.",
    ),
]

RepoRootPath = Annotated[
    Path | None,
    Parameter(
        name="--repo-root",
        help="Path to repository root (default: current directory).",
    ),
]

DbPath = Annotated[
    Path | None,
    Parameter(
        name="--db-path",
        help="Path to DuckDB database. Uses project config if not specified.",
    ),
]

BuildDirPath = Annotated[
    Path | None,
    Parameter(
        name="--build-dir",
        help="Build directory (default: build/).",
    ),
]

DocOutputDirPath = Annotated[
    Path | None,
    Parameter(
        name="--document-output-dir",
        help="Override Document Output/ directory.",
    ),
]

ProjectRoot = Annotated[
    Path | None,
    Parameter(
        name=["--root", "-r"],
        help="Explicit project root directory.",
    ),
]

class NxBackend(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "nx-cugraph"

NxBackendName = Annotated[
    str | None,
    Parameter(
        name="--nx-backend",
        help="NetworkX backend selection: auto, cpu, or nx-cugraph.",
    ),
]

NxGpuFlag = Annotated[
    bool,
    Parameter(
        name="--nx-gpu",
        help="Prefer GPU backend for NetworkX (nx-cugraph) when available.",
        negative=(),
    ),
]

NxGpuStrictFlag = Annotated[
    bool,
    Parameter(
        name="--nx-gpu-strict",
        help="Fail if GPU backend unavailable.",
        negative=(),
    ),
]

@dataclass(frozen=True)
class BackendFlags:
    """Backend preferences provided via CLI."""
    nx_backend: str | None = None
    nx_gpu: bool = False
    nx_gpu_strict: bool = False

@dataclass(frozen=True)
class RepoSelection:
    repo: str | None
    commit: str | None

@dataclass(frozen=True)
class PathSelection:
    repo_root: Path | None
    db_path: Path | None
    build_dir: Path | None
    document_output_dir: Path | None = None

@dataclass(frozen=True)
class RuntimeCliOptions:
    """
    Inputs that control how we build a ProjectRuntime.

    This is intentionally small & reusable across command groups.
    """
    project_root: Path | None = None
    repo: str | None = None
    commit: str | None = None
    db_path: Path | None = None
    build_dir: Path | None = None
    repo_root: Path | None = None
    document_output_dir: Path | None = None
    backend: BackendFlags = field(default_factory=BackendFlags)

# -----------------------------------------------------------------------------
# Runtime construction utilities (Typer-free)
# -----------------------------------------------------------------------------

class RuntimeError(Exception):
    """High-level CLI runtime error, instead of Typer.Exit."""

def build_runtime_from_cli(
    opts: RuntimeCliOptions,
    *,
    allow_fallback: bool = True,
) -> ProjectRuntime:
    """
    Replaces build_runtime_or_exit without Typer.

    1. Try project discovery (codeintel.yaml).
    2. If not found and allow_fallback=True, look at explicit
       repo/commit/db-path/build-dir/etc.
    3. Raise RuntimeError with user-facing message on failure.
    """
    from codeintel.cli.project import (
        ProjectConfig,
        StorageProjectConfig,
        ServingConfig,
        build_project_runtime,
    )
    from codeintel.storage.config import StorageConfig
    from codeintel.storage.gateway import open_gateway
    from codeintel.config import CliPathsInput, CliRepoInput

    try:
        return build_project_runtime(opts.project_root)
    except ProjectNotFoundError:
        if not allow_fallback:
            raise RuntimeError(
                "No codeintel.yaml found and fallback is disabled."
            ) from None

    # Fallback: explicit repo/commit/paths
    if opts.repo is None or opts.commit is None:
        raise RuntimeError(
            "No codeintel.yaml found. Provide --repo and --commit explicitly, "
            "or create a project file."
        )

    resolved_repo_root = opts.repo_root or Path.cwd()
    resolved_db_path = opts.db_path or Path("build/db/codeintel.duckdb")
    resolved_build_dir = opts.build_dir or Path("build")

    paths_cfg = CliPathsInput(
        repo_root=resolved_repo_root,
        build_dir=resolved_build_dir,
        db_path=resolved_db_path,
        document_output_dir=opts.document_output_dir,
    )
    repo_cfg = CliRepoInput(
        repo=opts.repo,
        commit=opts.commit,
    )

    cfg = CodeIntelConfig.from_cli_args(
        repo_cfg=repo_cfg,
        paths_cfg=paths_cfg,
        options=None,
    )

    snapshot = SnapshotRef(
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        repo_root=cfg.paths.repo_root,
    )
    paths = cfg.build_paths

    storage_cfg = StorageConfig.for_ingest(db_path=paths.db_path)
    gateway = open_gateway(storage_cfg)
    tools = resolve_tools_config(cfg.paths)
    graphs_backend = resolve_graph_backend(tools, opts.backend.nx_backend)

    serving = ServingConfig(
        mode="local_db",
        repo_root=cfg.paths.repo_root,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        db_path=paths.db_path,
    )

    return ProjectRuntime(
        root=cfg.paths.repo_root,
        project=ProjectConfig(
            repo=cfg.repo.repo,
            storage=StorageProjectConfig(db_path=paths.db_path),
        ),
        cfg=cfg,
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tools=tools,
        serving=serving,
        graphs_backend=graphs_backend,
    )
```

This module gives us:

* A central `make_root_app()`.
* Small, reusable **runtime config** dataclasses that we’ll inject by Cyclopts.
* A Typer-free `build_runtime_from_cli` we can call from any command.

---

## 3. Root Cyclopts app (`cli/cyclopts_app.py`)

This is the new unified entry point. It wires up sub-apps that we will define in separate modules.

```python
# cli/cyclopts_app.py

from __future__ import annotations

from cyclopts import App

from .cyclopts_common import make_root_app

from .cyclopts_build import build_app
from .cyclopts_ops import op_app, dataset_app, serve_app
from .cyclopts_docs import docs_app
from .cyclopts_datasets import datasets_ext_app
from .cyclopts_graphs import graphs_app
from .cyclopts_storage import storage_app
from .cyclopts_history import history_app
from .cyclopts_ide import ide_app
from .cyclopts_subsystem import subsystem_app


app: App = make_root_app()

# Core
app.command(build_app, name="build")
app.command(op_app, name="op")
app.command(dataset_app, name="dataset")
app.command(serve_app, name="serve")

# Domain
app.command(graphs_app, name="graph")
app.command(docs_app, name="docs")
app.command(storage_app, name="storage")
app.command(history_app, name="history")
app.command(ide_app, name="ide")
app.command(subsystem_app, name="subsystem")
app.command(datasets_ext_app, name="datasets")


def main() -> None:
    """Entry point used by console_scripts."""
    app()
```

Then update **`cli/__init__.py`** to export this Cyclopts `app` (while still exporting your Typer-based sub-apps if you want backwards compat):

```python
# cli/__init__.py

from __future__ import annotations

from .cyclopts_app import app  # Cyclopts root

# Re-export old Typer apps if you want to keep them around for a while:
from .commands.build import build_app as _typer_build_app
from .main import op_app as _typer_op_app, dataset_app as _typer_dataset_app, serve_app as _typer_serve_app
from .commands.docs import docs_app as _typer_docs_app
from .commands.datasets import datasets_ext_app as _typer_datasets_ext_app
from .commands.graphs import graphs_app as _typer_graphs_app
from .commands.storage import storage_app as _typer_storage_app
from .commands.history import history_app as _typer_history_app
from .commands.ide import ide_app as _typer_ide_app
from .commands.subsystem import subsystem_app as _typer_subsystem_app

__all__ = [
    "app",
    # Optionally these, for migrations:
    "_typer_build_app",
    "_typer_op_app",
    "_typer_dataset_app",
    "_typer_serve_app",
    "_typer_docs_app",
    "_typer_datasets_ext_app",
    "_typer_graphs_app",
    "_typer_storage_app",
    "_typer_history_app",
    "_typer_ide_app",
    "_typer_subsystem_app",
]
```

Your packaging entrypoint (e.g. `codeintel = codeintel.cli:main`) then automatically uses the Cyclopts CLI.

---

## 4. Build commands with Cyclopts (`cli/cyclopts_build.py`)

We’ll:

* Reuse your existing **dataclasses** `BuildRunOptions` and `BuildRunContext`.
* Reuse `build_run_handler` and `build_status_handler` as business logic.
* Provide small, explicitly typed CLI dataclasses that flatten options.
* Map them into `BuildRunOptions` / `BuildRunContext`.

```python
# cli/cyclopts_build.py

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.build.registry import get_target_graph
from codeintel.build.state import StateValidator
from codeintel.build.executor import BuildResult
from codeintel.build.plan import PlanGenerator
from codeintel.build.state import DatabaseState

from .cyclopts_common import (
    RuntimeCliOptions,
    OutputFormat,
    OutputFmt,
    JsonFlag,
    ProjectRoot,
    Verbose,
    build_runtime_from_cli,
    RuntimeError,
)


# Import your existing types/handlers as business logic
from codeintel.cli.commands.build import (
    BuildRunOptions,
    BuildRunContext,
    build_run_handler,
    # If you have a build_status_handler etc. import them too.
)

build_app = App(
    name="build",
    help="Build system commands for minimal-work target computation.",
)


class TargetScope(str, Enum):
    REQUESTED = "requested"
    ALL = "all"


class RunMode(str, Enum):
    EXECUTE = "execute"
    DRY_RUN = "dry_run"


Targets = Annotated[
    list[str] | None,
    Parameter(
        name=None,  # positional
        help="Target names to build (e.g., function_metrics, call_graph).",
    ),
]

ModuleOpt = Annotated[
    str | None,
    Parameter(
        name=["--module", "-m"],
        help="Build all targets in a module (ingestion, graphs, analytics).",
    ),
]

AllTargetsFlag = Annotated[
    bool,
    Parameter(
        name="--all-targets",
        help="Build all targets in the graph (not just requested).",
        negative=(),
    ),
]

DryRunFlag = Annotated[
    bool,
    Parameter(
        name="--dry-run",
        help="Plan build without executing.",
        negative=(),
    ),
]

ForceOpt = Annotated[
    list[str] | None,
    Parameter(
        name="--force",
        help="Force recompute of specific targets (repeatable).",
    ),
]


@dataclass
class BuildRunCli:
    """CLI representation of `build run` arguments."""

    # Positional
    targets: Targets = None

    # Options
    module: ModuleOpt = None
    all_targets: AllTargetsFlag = False
    dry_run: DryRunFlag = False
    force: ForceOpt = None

    # Shared runtime / output options
    project_root: ProjectRoot = None
    verbose: Verbose = 0
    output_format: OutputFmt = OutputFormat.TEXT
    json: JsonFlag = False


@build_app.command
def run(cfg: Annotated[BuildRunCli, Parameter(name="*")]) -> None:
    """
    Build targets with automatic dependency resolution.

    This is a Cyclopts wrapper around `build_run_handler` that
    keeps the handler signature small and type-checked.
    """
    runtime_opts = RuntimeCliOptions(
        project_root=cfg.project_root,
    )

    try:
        runtime = build_runtime_from_cli(runtime_opts)
    except RuntimeError as exc:
        # You can use app.console here if you want Rich output
        raise SystemExit(str(exc))

    # Map CLI -> domain options
    target_scope = TargetScope.ALL if cfg.all_targets else TargetScope.REQUESTED
    run_mode = RunMode.DRY_RUN if cfg.dry_run else RunMode.EXECUTE

    output_format = cfg.output_format
    if cfg.json:
        output_format = OutputFormat.JSON

    options = BuildRunOptions(
        targets=cfg.targets,
        module=cfg.module,
        target_scope=target_scope,
        run_mode=run_mode,
        force=cfg.force,
    )
    ctx = BuildRunContext(
        runtime_options=runtime_opts,
        verbose=cfg.verbose,
        output_format=output_format,
    )

    build_run_handler(options, ctx)


# ---------------------------------------------------------------------------
# Example: status / history
# ---------------------------------------------------------------------------

@dataclass
class BuildStatusCli:
    project_root: ProjectRoot = None
    output_format: OutputFmt = OutputFormat.TEXT
    json: JsonFlag = False
    verbose: Verbose = 0


@build_app.command
def status(cfg: Annotated[BuildStatusCli, Parameter(name="*")]) -> None:
    """
    Show current target status.

    This can reuse your existing StateValidator, PlanGenerator, etc.
    """
    runtime_opts = RuntimeCliOptions(
        project_root=cfg.project_root,
    )
    try:
        runtime = build_runtime_from_cli(runtime_opts)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    state = StateValidator(runtime.gateway, runtime.snapshot, runtime.paths).validate()
    # You can call your existing `_format_status_text` / `_format_status_json`
    from codeintel.cli.commands.build import _format_status_text, _format_status_json

    output_format = cfg.output_format
    if cfg.json:
        output_format = OutputFormat.JSON

    if output_format is OutputFormat.JSON:
        import json as _json
        print(_json.dumps(_format_status_json(state), indent=2))
    else:
        print(_format_status_text(state))
```

This is enough for an agent to systematically port:

* `build history` in the same style as `status`.
* Any additional build commands.

Notice:

* **No Typer imports** here.
* Cyclopts receives a **single `BuildRunCli` object**, so function complexity stays low.
* We reuse the existing domain dataclasses (`BuildRunOptions`, `BuildRunContext`) and handler.

---

## 5. Ops / dataset / serve (`cli/cyclopts_ops.py`)

Here we mirror `cli/main.py`, but with Cyclopts Apps and Cyclopts-friendly dynamic command registration.

We’ll still reuse your `op_params` “string tunnel” and metadata builder; we just change the registration layer.

```python
# cli/cyclopts_ops.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from cyclopts import App, Parameter

from codeintel.cli.project import ProjectRuntime
from codeintel.cli.project import ProjectNotFoundError, build_project_runtime

from .cyclopts_common import RuntimeCliOptions, ProjectRoot, Verbose, build_runtime_from_cli, RuntimeError
from . import op_params  # your dynamic op metadata utilities
from codeintel.serving.operations import Operation  # whatever your Operation type is


op_app = App(
    name="op",
    help="Operation invocation commands.",
    no_args_is_help=True,
)

dataset_app = App(
    name="dataset",
    help="Dataset contract verification commands.",
    no_args_is_help=True,
)

serve_app = App(
    name="serve",
    help="HTTP and MCP server commands.",
    no_args_is_help=True,
)

CategoryOpt = Annotated[
    str | None,
    Parameter(
        name=["--category", "-c"],
        help="Filter by operation category.",
    ),
]

ProjectRootOpt = ProjectRoot  # reuse alias


# ---------------------------------------------------------------------------
# op list / show wrappers
# ---------------------------------------------------------------------------

@op_app.command
def list(category: CategoryOpt = None, project_root: ProjectRootOpt = None) -> None:
    """
    List operations available in the serving layer.
    """
    runtime_opts = RuntimeCliOptions(project_root=project_root)
    try:
        runtime = build_runtime_from_cli(runtime_opts)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    ops = op_params.get_operations_with_cli_support(runtime.gateway)
    if category:
        ops = [o for o in ops if o.category == category]

    # You already have formatting helpers in op_params / main.py;
    # reuse them here.
    for op in ops:
        print(f"{op.id:40} {op.category or '-':20} {op.summary or ''}")


@dataclass
class OpShowCli:
    op_id: Annotated[
        str,
        Parameter(
            name=None,
            help="Operation identifier.",
        ),
    ]
    project_root: ProjectRootOpt = None


@op_app.command
def show(cfg: Annotated[OpShowCli, Parameter(name="*")]) -> None:
    """
    Show details for a single operation.
    """
    runtime_opts = RuntimeCliOptions(project_root=cfg.project_root)
    try:
        runtime = build_runtime_from_cli(runtime_opts)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    op = op_params.build_operation_cli_metadata(runtime.gateway, cfg.op_id)
    # Reuse your existing rich panel formatting if desired.
    print(op.long_help)


# ---------------------------------------------------------------------------
# Dynamic op commands
# ---------------------------------------------------------------------------

def _dynamic_op_invoke(
    op_id: str,
    params: dict[str, Any],
    project_root: Path | None,
    *,
    skip_prereqs: bool,
    verbose: bool,
) -> None:
    """
    Callback that actually invokes operations (logic from main.py).
    """
    runtime_opts = RuntimeCliOptions(project_root=project_root)
    try:
        runtime = build_runtime_from_cli(runtime_opts)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    # follow your existing patterns in main.py:
    # - run_operation_prereqs(...)
    # - _invoke_operation(...)
    from codeintel.serving.runtime import run_operation_prereqs, invoke_operation

    if not skip_prereqs:
        run_operation_prereqs(
            op_id=op_id,
            gateway=runtime.gateway,
            snapshot=runtime.snapshot,
            paths=runtime.paths,
            tools=runtime.tools,
        )

    invoke_operation(op_id, params, runtime)


def register_dynamic_op_commands(app: App) -> int:
    """
    Cyclopts equivalent of `register_dynamic_commands`.

    It uses your existing `op_params.build_cli_param_specs_for_operation`
    and `op_params.coerce_params_from_strings`, but registers Cyclopts
    commands instead of Typer commands.
    """
    operations = op_params.get_operations_with_cli_support()
    count = 0

    for op in operations:
        cli_meta = op_params.build_operation_cli_metadata(op)

        # We keep the string-tunnel: each param is Optional[str].
        # CLI -> dict[str, str | None] -> coerce_params_from_strings.
        @dataclass
        class OpCliArgs:
            project_root: ProjectRootOpt = None
            skip_prereqs: Annotated[
                bool,
                Parameter(
                    name="--skip-prereqs",
                    help="Skip prerequisite operations.",
                    negative=(),
                ),
            ] = False
            verbose: Verbose = 0

            # dynamically add fields at runtime:
            #  for param in cli_meta.params:
            #    setattr(OpCliArgs, param.name, Annotated[ str | None, Parameter(...)])

        # In pure Python, you’d construct this class dynamically using
        # `dataclasses.make_dataclass` based on cli_meta.params, with
        # `Parameter` metadata in Annotated. For brevity, this is shown
        # as a conceptual pattern; an LLM agent can implement the actual
        # `make_dataclass` logic using cli_meta’s attributes.

        def make_command(op_id: str, meta=cli_meta):
            @app.command(name=meta.cli_name)
            def _cmd(args: Annotated[OpCliArgs, Parameter(name="*")]) -> None:
                # Extract raw param strings (string tunnel)
                raw: dict[str, str | None] = {
                    name: getattr(args, name)
                    for name in meta.param_names
                }
                coerced = op_params.coerce_params_from_strings(meta, raw)
                _dynamic_op_invoke(
                    op_id,
                    coerced,
                    args.project_root,
                    skip_prereqs=args.skip_prereqs,
                    verbose=bool(args.verbose),
                )

            _cmd.__doc__ = meta.help_text
            return _cmd

        make_command(op.id)
        count += 1

    return count


# Call this once when module is imported; or call from cyclopts_app.main
register_dynamic_op_commands(op_app)

# ---------------------------------------------------------------------------
# Dataset & serve wrappers (pattern)
# ---------------------------------------------------------------------------

@dataclass
class DatasetVerifyCli:
    table_key: Annotated[
        str | None,
        Parameter(
            name=None,
            help="Dataset table key to verify (verifies all if not specified).",
        ),
    ] = None
    project_root: ProjectRootOpt = None


@dataset_app.command
def verify(cfg: Annotated[DatasetVerifyCli, Parameter(name="*")]) -> None:
    """
    Verify dataset contracts against actual data.
    """
    from codeintel.analytics.datasets import collect_contract_issues

    runtime_opts = RuntimeCliOptions(project_root=cfg.project_root)
    try:
        runtime = build_runtime_from_cli(runtime_opts)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    issues = collect_contract_issues(runtime.gateway.con)
    if cfg.table_key:
        issues = [i for i in issues if cfg.table_key in i]

    if not issues:
        print("All dataset contracts verified successfully.")
    else:
        print(f"Found {len(issues)} contract issues:")
        for issue in issues:
            print(f"  - {issue}")


@dataclass
class ServeHttpCli:
    project_root: ProjectRootOpt = None
    # e.g. host/port options as needed


@serve_app.command
def http(cfg: Annotated[ServeHttpCli, Parameter(name="*")]) -> None:
    """
    Start the HTTP server.
    """
    from codeintel.serving.http import run_http_server

    runtime_opts = RuntimeCliOptions(project_root=cfg.project_root)
    runtime = build_runtime_from_cli(runtime_opts)
    run_http_server(runtime)


@dataclass
class ServeMcpCli:
    project_root: ProjectRootOpt = None
    auto_pipeline: Annotated[
        bool,
        Parameter(
            name="--auto-pipeline",
            help="Enable auto-pipeline execution on MCP requests.",
            negative=(),
        ),
    ] = False


@serve_app.command
def mcp(cfg: Annotated[ServeMcpCli, Parameter(name="*")]) -> None:
    """
    Start the MCP server.
    """
    from codeintel.serving.mcp import run_mcp_server

    runtime_opts = RuntimeCliOptions(project_root=cfg.project_root)
    runtime = build_runtime_from_cli(runtime_opts)
    # follow your logging pattern from main.py before running server
    run_mcp_server(runtime, auto_pipeline_enabled=cfg.auto_pipeline)
```

The dynamic-op part is intentionally “pattern-level”: an LLM agent can use `dataclasses.make_dataclass` + `Annotated` + `Parameter` based on the introspected `CliParamSpec`s in `op_params`, replacing the Typer-based builder. The rest (list/show/verify/http/mcp) is straightforward.

---

## 6. Other domains (docs, datasets, graphs, storage, history, ide, subsystem)

The pattern for the remaining command groups is exactly the same as **build** and **dataset**:

* Define a small `@dataclass` that’s the **CLI surface**.
* Keep function complexity low by passing a **single config object** to each handler.
* Reuse existing Typer-era handlers as business logic where possible.
* Use `RuntimeCliOptions` + `build_runtime_from_cli` to obtain `ProjectRuntime`.

For each:

### 6.1 Docs (`cli/cyclopts_docs.py`)

Mirror `cli/commands/docs.py`:

* Reuse your dataclasses for export options (`DocsExportOptions` etc.) if present.
* Map CLI dataclass → domain config dataclass → call `run_docs_export` / `run_docs_export_via_build_system`.

Skeleton:

```python
# cli/cyclopts_docs.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from cyclopts import App, Parameter

from .cyclopts_common import (
    RuntimeCliOptions,
    ProjectRoot,
    OutputFmt,
    JsonFlag,
    OutputFormat,
    build_runtime_from_cli,
    RuntimeError,
)

from codeintel.cli.commands.docs import (
    DocsExportOptions,
    run_docs_export,
)

docs_app = App(
    name="docs",
    help="Document Output export commands.",
    no_args_is_help=True,
)


@dataclass
class DocsExportCli:
    project_root: ProjectRoot = None
    output_format: OutputFmt = OutputFormat.TEXT
    json: JsonFlag = False
    # plus whatever docs-specific flags you have (macro validation, prereq mode, etc.)


@docs_app.command(name="export")
def export(cfg: Annotated[DocsExportCli, Parameter(name="*")]) -> None:
    runtime_opts = RuntimeCliOptions(project_root=cfg.project_root)
    try:
        runtime = build_runtime_from_cli(runtime_opts)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    out_fmt = cfg.output_format
    if cfg.json:
        out_fmt = OutputFormat.JSON

    options = DocsExportOptions(
        # map cfg fields into the options dataclass used by your existing handler
        output_format=out_fmt,
        # ...
    )
    run_docs_export(runtime, options)
```

### 6.2 Datasets extended (`cli/cyclopts_datasets.py`)

Mirror `cli/commands/datasets.py`:

* Keep your enums (`ExportValidationMode`, `MacroRequirement`, etc.).
* Define dataclasses per command: `DatasetsLintCli`, `DatasetsListCli`, `DatasetsDiffCli`, `DatasetsSnapshotCli`, `DatasetsGenerateSchemasCli`, `DatasetsCatalogCli`, `DatasetsScaffoldCli`.
* Each uses `RuntimeCliOptions` + `build_runtime_from_cli`.

Each command body just calls the existing functions from `cli/commands/datasets.py` that do the real work (you have a clear pattern of “handler-like” functions there).

### 6.3 Graphs (`cli/cyclopts_graphs.py`)

Mirror `cli/commands/graphs.py`:

* Reuse your scope flags from `_common.py` (module/path/time scopes).
* Express scopes as a separate dataclass:

```python
@dataclass
class GraphScopeCli:
    scope_module: Annotated[list[str] | None, Parameter(name="--scope-module")] = None
    scope_path: Annotated[list[str] | None, Parameter(name="--scope-path")] = None
    scope_time_start: Annotated[str | None, Parameter(name="--scope-time-start")] = None
    scope_time_end: Annotated[str | None, Parameter(name="--scope-time-end")] = None
```

* Flatten it into each graph command via `Parameter(name="*")`.
* Map to your existing scope-handling functions (`build_scope_filter` etc.).

### 6.4 Storage (`cli/cyclopts_storage.py`)

Mirror `cli/commands/storage.py`:

* Take your enums `MacroRequirement`, `AnalyzeMode` as-is.
* CLI dataclass:

```python
@dataclass
class ValidateMacrosCli:
    db_path: Annotated[Path, Parameter("--db-path", help="...")] = Path("build/db/codeintel.duckdb")
    macros: Annotated[MacroRequirement, Parameter("--macros", help="Ingest macro requirement policy.")] = MacroRequirement.REQUIRE
    verbose: Verbose = 0
```

* Command `@storage_app.command(name="validate-macros")` calls your existing validation function.

### 6.5 History, IDE, Subsystem

Same story:

* One `App` per domain (`history_app`, `ide_app`, `subsystem_app`).
* Small dataclasses for CLI surfaces, flatten with `Parameter(name="*")`.
* Call your existing domain functions.

---

## 7. How an LLM agent should systematically perform the migration

For each Typer command module:

1. **Identify business logic handlers**

   * e.g. `build_run_handler`, `run_docs_export`, dataset lint/diff/snapshot functions, graph analytics functions, storage validation routines.

2. **Create a Cyclopts module** (like `cyclopts_build.py`) that:

   * Defines an `App(name=..., help=...)`.
   * Defines one dataclass per CLI command with strong types and `Annotated[..., Parameter(...)]` fields.
   * Maps that dataclass into the handler inputs.

3. **Wire sub-apps into root `cyclopts_app.app`**, mirroring the existing Typer tree.

4. **Stop using `_option_shim`** and Typer’s proxy defaults:

   * You never use `typer.Option`, `typer.Argument`, or `wrap_command` in Cyclopts modules.
   * You only use domain dataclasses and `Parameter`.

5. **Keep Typer as a pure support dependency** (for now):

   * Old Typer-based CLI remains in `cli/commands/*` and `cli/main.py`.
   * New Cyclopts-based CLI never uses Typer for parsing; only the legacy handlers may still call `typer.echo` / `typer.Exit`.
   * Over time, you can replace those with `app.console` / `rich` and normal exceptions, and drop Typer entirely.

---

If you’d like, I can next take a specific command module (for example `cli/commands/datasets.py`) and write a complete `cli/cyclopts_datasets.py` with all commands (lint/list/diff/snapshot/conformance/generate-schemas/catalog/scaffold) wired up, so you have a fully worked template to replicate across the rest of your CLI.


# detailed example of cli/commands/datasets.py converted to cyclopts implementation #

Here’s a complete `cli/cyclopts_datasets.py` that wires all the datasets commands (`lint`, `list`, `snapshot`, `diff`, `conformance`, `generate-schemas`, `catalog`, `scaffold`) to your existing Typer-era handlers in `cli/commands/datasets.py`, but entirely through Cyclopts.

It keeps function signatures low-complexity by grouping options into dataclasses and converting them into your existing option dataclasses (`RuntimeOptions`, `LintOptions`, `DiffOptions`, etc.). Typer is only used internally for catching `typer.Exit` and normalizing it to `SystemExit`.

```python
from __future__ import annotations

"""
Cyclopts-based implementation of the extended datasets CLI.

This module mirrors the functionality of ``cli/commands/datasets.py`` (Typer-based)
but exposes it via Cyclopts with:

- Explicitly typed, grouped configuration objects.
- No reliance on Typer for argument parsing or inference.
- Very small command function signatures (runtime + options [+ verbose]).

The existing Typer command module remains the source of truth for the
business logic (handlers, dataclasses, enums). This module only handles:

- CLI surface definition (arguments, options, help text).
- Conversion from CLI dataclasses into the existing option dataclasses.
- Exception translation from ``typer.Exit`` into ``SystemExit`` so exit codes
  remain stable under Cyclopts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import typer
from cyclopts import App, Parameter

# Reuse the existing implementation and domain types.
from .commands import datasets as ds


# ---------------------------------------------------------------------------
# App and shared helpers
# ---------------------------------------------------------------------------

datasets_ext_app = App(
    name="datasets",
    help="Extended dataset management commands.",
    no_args_is_help=True,
)


VerboseLevel = Annotated[
    int,
    Parameter(
        name=["--verbose", "-v"],
        help="Increase logging verbosity (0 = warnings, higher = more detail).",
    ),
]


@dataclass
class DatasetRuntimeCli:
    """Runtime selection shared by all datasets commands."""

    project_root: Annotated[
        Path | None,
        Parameter(
            name=["--root", "-r"],
            help="Explicit project root directory (otherwise discovered).",
        ),
    ] = None
    repo: Annotated[
        str | None,
        Parameter(
            name="--repo",
            help="Repository slug (e.g., org/repo). Overrides project config.",
        ),
    ] = None
    commit: Annotated[
        str | None,
        Parameter(
            name="--commit",
            help="Commit SHA. Overrides project config when supplied.",
        ),
    ] = None
    repo_root: Annotated[
        Path | None,
        Parameter(
            name="--repo-root",
            help="Path to repository root; overrides auto-discovery.",
        ),
    ] = None
    db_path: Annotated[
        Path | None,
        Parameter(
            name="--db-path",
            help="Path to DuckDB database. Uses project config if omitted.",
        ),
    ] = None
    build_dir: Annotated[
        Path | None,
        Parameter(
            name="--build-dir",
            help="Build directory root. Uses project config if omitted.",
        ),
    ] = None


def _runtime_from_cli(cli: DatasetRuntimeCli) -> ds.RuntimeOptions:
    """Translate Cyclopts runtime selection into the datasets RuntimeOptions."""
    project = ds.ProjectSelection(
        project_root=cli.project_root,
        repo=cli.repo,
        commit=cli.commit,
        repo_root=cli.repo_root,
    )
    build = ds.BuildSelection(
        db_path=cli.db_path,
        build_dir=cli.build_dir,
    )
    return ds.RuntimeOptions(project=project, build=build)


def _run(handler, *args, **kwargs) -> None:
    """Invoke a Typer-era handler and normalize ``typer.Exit`` into ``SystemExit``.

    This keeps existing error codes and messages but makes the behavior
    predictable under Cyclopts.
    """
    try:
        handler(*args, **kwargs)
    except typer.Exit as exc:  # type: ignore[attr-defined]
        raise SystemExit(exc.exit_code) from exc


# ---------------------------------------------------------------------------
# lint
# ---------------------------------------------------------------------------


@dataclass
class LintCliOptions:
    """Options for ``codeintel datasets lint``."""

    schema_dir: Annotated[
        Path,
        Parameter(
            name="--schema-dir",
            help="Directory containing export JSON Schemas.",
        ),
    ] = Path("src/codeintel/config/schemas/export")
    sample_rows: Annotated[
        bool,
        Parameter(
            name="--sample-rows",
            help=(
                "Request row sampling. In the Typer CLI this toggled SamplingMode; "
                "here it is a simple flag and maps to SamplingMode.ENABLED. "
                "For actual sampling, prefer `codeintel datasets conformance`."
            ),
        ),
    ] = False


@datasets_ext_app.command(name="lint")
def lint(
    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")],
    options: Annotated[LintCliOptions, Parameter(name="*")],
    verbose: VerboseLevel = 0,
) -> None:
    """Validate dataset contract health."""
    runtime_opts = _runtime_from_cli(runtime)
    lint_opts = ds.LintOptions(
        schema_dir=options.schema_dir,
        sampling=ds.SamplingMode.ENABLED if options.sample_rows else ds.SamplingMode.DISABLED,
    )
    _run(ds.datasets_lint_handler, runtime_opts, lint_opts, verbose)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

DocsFilterMode = Literal["include", "only", "exclude"]
ReadOnlyFilterMode = Literal["include", "only", "exclude"]


@dataclass
class ListCliFilters:
    """Filters for ``codeintel datasets list``."""

    docs_view: Annotated[
        DocsFilterMode,
        Parameter(
            name="--docs-view",
            help='Docs view filter: "include", "exclude", or "only".',
        ),
    ] = "include"
    read_only: Annotated[
        ReadOnlyFilterMode,
        Parameter(
            name="--read-only",
            help='Read-only filter: "include", "exclude", or "only".',
        ),
    ] = "include"
    max_description: Annotated[
        int,
        Parameter(
            name="--max-description",
            help="Maximum description length before truncation.",
        ),
    ] = 80


@datasets_ext_app.command(name="list")
def list_datasets(
    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")],
    filters: Annotated[ListCliFilters, Parameter(name="*")],
    verbose: VerboseLevel = 0,
) -> None:
    """List datasets with capabilities and optional filters."""
    runtime_opts = _runtime_from_cli(runtime)
    filter_opts = ds.ListFilters(
        docs_view=filters.docs_view,
        read_only=filters.read_only,
        max_description=filters.max_description,
    )
    _run(ds.datasets_list_handler, runtime_opts, filter_opts, verbose)


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------


@dataclass
class SnapshotCliOptions:
    """Options for ``codeintel datasets snapshot``."""

    output: Annotated[
        Path,
        Parameter(
            name="--output",
            help="Output file path for JSON dataset specs.",
        ),
    ]


@datasets_ext_app.command(name="snapshot")
def snapshot(
    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")],
    options: Annotated[SnapshotCliOptions, Parameter(name="*")],
    verbose: VerboseLevel = 0,
) -> None:
    """Write current dataset specs to a JSON snapshot file."""
    runtime_opts = _runtime_from_cli(runtime)
    _run(ds.datasets_snapshot_handler, runtime_opts, options.output, verbose)


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


@dataclass
class DiffCliOptions:
    """Options for ``codeintel datasets diff``."""

    baseline: Annotated[
        Path | None,
        Parameter(
            name="--baseline",
            help="Path to JSON baseline from `codeintel datasets snapshot`.",
        ),
    ] = None
    output: Annotated[
        Path | None,
        Parameter(
            name="--output",
            help="Optional output file path for writing current specs.",
        ),
    ] = None
    against_ref: Annotated[
        str | None,
        Parameter(
            name="--against-ref",
            help="Git ref to diff against (e.g. HEAD~, main).",
        ),
    ] = None
    baseline_path: Annotated[
        Path,
        Parameter(
            name="--baseline-path",
            help="Path of the snapshot file inside the git ref.",
        ),
    ] = Path("build/dataset_specs.json")


@datasets_ext_app.command(name="diff")
def diff(
    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")],
    options: Annotated[DiffCliOptions, Parameter(name="*")],
    verbose: VerboseLevel = 0,
) -> None:
    """Diff current dataset specs against a baseline."""
    runtime_opts = _runtime_from_cli(runtime)
    diff_opts = ds.DiffOptions(
        baseline=options.baseline,
        output=options.output,
        against_ref=options.against_ref,
        baseline_path=options.baseline_path,
    )
    _run(ds.datasets_diff_handler, runtime_opts, diff_opts, verbose)


# ---------------------------------------------------------------------------
# conformance
# ---------------------------------------------------------------------------


@dataclass
class ConformanceCliOptions:
    """Options for ``codeintel datasets conformance``."""

    schema_dir: Annotated[
        Path,
        Parameter(
            name="--schema-dir",
            help="Directory containing export JSON Schemas.",
        ),
    ] = Path("src/codeintel/config/schemas/export")
    sample_rows: Annotated[
        bool,
        Parameter(
            name="--sample-rows",
            help="Enable row sampling against JSON Schemas.",
        ),
    ] = False
    sample_size: Annotated[
        int,
        Parameter(
            name="--sample-size",
            help="Number of rows to sample when sampling is enabled.",
        ),
    ] = 50


@datasets_ext_app.command(name="conformance")
def conformance(
    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")],
    options: Annotated[ConformanceCliOptions, Parameter(name="*")],
    verbose: VerboseLevel = 0,
) -> None:
    """Run full dataset conformance checks."""
    runtime_opts = _runtime_from_cli(runtime)
    conf_opts = ds.ConformanceOptions(
        schema_dir=options.schema_dir,
        sampling=ds.SamplingMode.ENABLED if options.sample_rows else ds.SamplingMode.DISABLED,
        sample_size=options.sample_size,
    )
    _run(ds.datasets_conformance_handler, runtime_opts, conf_opts, verbose)


# ---------------------------------------------------------------------------
# generate-schemas
# ---------------------------------------------------------------------------


@dataclass
class ExportCliOptions:
    """Shared export configuration for generate-schemas."""

    validation: Annotated[
        ds.ExportValidationMode,
        Parameter(
            name="--validation",
            help="Validation strategy for exports.",
        ),
    ] = ds.ExportValidationMode.REQUIRED
    macro_requirement: Annotated[
        ds.MacroRequirement,
        Parameter(
            name="--macro-requirement",
            help="Macro requirement policy for exports.",
        ),
    ] = ds.MacroRequirement.REQUIRE_NORMALIZED
    schemas: Annotated[
        list[str] | None,
        Parameter(
            name="--schema",
            help="Filter by export schema ID (repeatable).",
        ),
    ] = None
    datasets: Annotated[
        list[str] | None,
        Parameter(
            name="--dataset",
            help="Filter by dataset name (repeatable).",
        ),
    ] = None
    output_format: Annotated[
        ds.OutputFormat,
        Parameter(
            name="--output-format",
            help="Output format for command metadata (text or json).",
        ),
    ] = ds.OutputFormat.TEXT
    dry_run: Annotated[
        bool,
        Parameter(
            name="--dry-run",
            help="Plan schema generation without writing files.",
        ),
    ] = False


@dataclass
class GenerateSchemasCliOptions:
    """Options for ``codeintel datasets generate-schemas``."""

    output_dir: Annotated[
        Path,
        Parameter(
            name="--output-dir",
            help="Directory to write generated JSON Schemas.",
        ),
    ] = Path("src/codeintel/config/schemas/export")


def _export_from_cli(cfg: ExportCliOptions) -> ds.DatasetExportOptions:
    """Translate Cyclopts export options into DatasetExportOptions."""
    return ds.DatasetExportOptions(
        validation=cfg.validation,
        macro_requirement=cfg.macro_requirement,
        schemas=cfg.schemas,
        datasets=cfg.datasets,
        output_format=cfg.output_format,
        run_mode=ds.DryRunMode.DRY_RUN if cfg.dry_run else ds.DryRunMode.EXECUTE,
    )


@datasets_ext_app.command(name="generate-schemas")
def generate_schemas(
    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")],
    export: Annotated[ExportCliOptions, Parameter(name="*")],
    schema: Annotated[GenerateSchemasCliOptions, Parameter(name="*")],
    verbose: VerboseLevel = 0,
) -> None:
    """Generate export JSON Schemas from TypedDict row models."""
    runtime_opts = _runtime_from_cli(runtime)
    export_opts = _export_from_cli(export)
    schema_opts = ds.GenerateSchemasOptions(output_dir=schema.output_dir)
    _run(ds.datasets_generate_schemas_handler, runtime_opts, export_opts, schema_opts, verbose)


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------


@dataclass
class CatalogCliOptions:
    """Options for ``codeintel datasets catalog``."""

    output_dir: Annotated[
        Path,
        Parameter(
            name="--output-dir",
            help="Directory to write catalog artifacts (Markdown/HTML).",
        ),
    ] = Path("build/catalog")
    sample_rows_count: Annotated[
        int,
        Parameter(
            name="--sample-rows-count",
            help="Number of sample rows per dataset in the catalog.",
        ),
    ] = 3
    sample_rows_strict: Annotated[
        ds.SamplingStrictness,
        Parameter(
            name="--sample-rows-strict",
            help="Sampling strictness: lenient or strict.",
        ),
    ] = ds.SamplingStrictness.LENIENT


@datasets_ext_app.command(name="catalog")
def catalog(
    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")],
    options: Annotated[CatalogCliOptions, Parameter(name="*")],
    verbose: VerboseLevel = 0,
) -> None:
    """Generate Markdown/HTML dataset catalog."""
    runtime_opts = _runtime_from_cli(runtime)
    catalog_opts = ds.CatalogOptions(
        output_dir=options.output_dir,
        sample_rows_count=options.sample_rows_count,
        sample_rows_strict=options.sample_rows_strict,
    )
    _run(ds.datasets_catalog_handler, runtime_opts, catalog_opts, verbose)


# ---------------------------------------------------------------------------
# scaffold
# ---------------------------------------------------------------------------


@dataclass
class ScaffoldCliOptions:
    """Options for ``codeintel datasets scaffold``."""

    # Metadata
    kind: Annotated[
        str,
        Parameter(
            name="--kind",
            help='Kind of dataset: typically "table" or "view".',
        ),
    ] = "table"
    table_key: Annotated[
        str | None,
        Parameter(
            name="--table-key",
            help="Logical table key for the dataset.",
        ),
    ] = None
    owner: Annotated[
        str | None,
        Parameter(
            name="--owner",
            help="Owning team or contact identifier.",
        ),
    ] = None
    freshness_sla: Annotated[
        str | None,
        Parameter(
            name="--freshness-sla",
            help="Freshness SLA description (e.g. 1h, 1d).",
        ),
    ] = None
    retention_policy: Annotated[
        str | None,
        Parameter(
            name="--retention-policy",
            help="Retention policy summary for the dataset.",
        ),
    ] = None

    # Schema options
    schema_version: Annotated[
        str,
        Parameter(
            name="--schema-version",
            help="Schema version tag for the export schema.",
        ),
    ] = "1"
    validation_profile: Annotated[
        str,
        Parameter(
            name="--validation-profile",
            help="Validation profile name (e.g. strict, permissive).",
        ),
    ] = "strict"
    schema_id: Annotated[
        str | None,
        Parameter(
            name="--schema-id",
            help="Explicit JSON Schema $id for the dataset.",
        ),
    ] = None

    # File options
    jsonl_filename: Annotated[
        str | None,
        Parameter(
            name="--jsonl-filename",
            help="Filename for the JSONL export.",
        ),
    ] = None
    parquet_filename: Annotated[
        str | None,
        Parameter(
            name="--parquet-filename",
            help="Filename for the Parquet export.",
        ),
    ] = None
    stable_id: Annotated[
        str | None,
        Parameter(
            name="--stable-id",
            help="Stable identifier used for tracking",
        ),
    ] = None

    # IO and behavior
    output_dir: Annotated[
        Path,
        Parameter(
            name="--output-dir",
            help="Directory to write scaffold files.",
        ),
    ] = Path("build/dataset_scaffolds")
    overwrite_policy: Annotated[
        ds.OverwritePolicy,
        Parameter(
            name="--overwrite-policy",
            help="Overwrite policy when scaffold paths already exist.",
        ),
    ] = ds.OverwritePolicy.ERROR
    specs_snapshot: Annotated[
        Path,
        Parameter(
            name="--specs-snapshot",
            help="Path to dataset specs snapshot used for bootstrap hints.",
        ),
    ] = Path("build/catalog/dataset_specs.json")
    dry_run: Annotated[
        bool,
        Parameter(
            name="--dry-run",
            help="Show scaffold plan without writing files.",
        ),
    ] = False
    bootstrap: Annotated[
        ds.BootstrapSnippet,
        Parameter(
            name="--bootstrap",
            help="Control emission of bootstrap snippets in metadata.",
        ),
    ] = ds.BootstrapSnippet.SKIP
    registry_check: Annotated[
        bool,
        Parameter(
            name="--registry-check",
            help="Check existing dataset registry for conflicts.",
        ),
    ] = False


def _scaffold_options_from_cli(cfg: ScaffoldCliOptions) -> ds.ScaffoldCliOptions:
    """Translate Cyclopts scaffold config into ScaffoldCliOptions."""
    metadata = ds.ScaffoldMetadataOptions(
        kind=cfg.kind,
        table_key=cfg.table_key,
        owner=cfg.owner,
        freshness_sla=cfg.freshness_sla,
        retention_policy=cfg.retention_policy,
    )
    schema = ds.ScaffoldSchemaOptions(
        schema_version=cfg.schema_version,
        validation_profile=cfg.validation_profile,
        schema_id=cfg.schema_id,
    )
    files = ds.ScaffoldFileOptions(
        jsonl_filename=cfg.jsonl_filename,
        parquet_filename=cfg.parquet_filename,
        stable_id=cfg.stable_id,
    )
    dataset_opts = ds.DatasetScaffoldOptions(
        output_dir=cfg.output_dir,
        overwrite_policy=cfg.overwrite_policy,
    )
    io_opts = ds.ScaffoldIOOptions(
        specs_snapshot=cfg.specs_snapshot,
        scaffold=dataset_opts,
    )
    behavior = ds.ScaffoldBehaviorOptions(
        run_mode=ds.DryRunMode.DRY_RUN if cfg.dry_run else ds.DryRunMode.EXECUTE,
        bootstrap=cfg.bootstrap,
        registry_check=ds.RegistryCheck.ENABLED if cfg.registry_check else ds.RegistryCheck.DISABLED,
    )
    return ds.ScaffoldCliOptions(
        metadata=metadata,
        schema=schema,
        files=files,
        io=io_opts,
        behavior=behavior,
    )


@datasets_ext_app.command(name="scaffold")
def scaffold(
    name: Annotated[
        str,
        Parameter(
            help="Name of the dataset to scaffold (TypedDict / logical dataset name).",
        ),
    ],
    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")],
    options: Annotated[ScaffoldCliOptions, Parameter(name="*")],
    verbose: VerboseLevel = 0,
) -> None:
    """Create a new dataset scaffold."""
    runtime_opts = _runtime_from_cli(runtime)
    scaffold_opts = _scaffold_options_from_cli(options)
    _run(ds.datasets_scaffold_handler, name, runtime_opts, scaffold_opts, verbose)


__all__ = [
    "datasets_ext_app",
]
```

### How to hook this up

At the top level where you construct your Cyclopts root app, you can register this sub-app, for example:

```python
from cyclopts import App
from .cyclopts_datasets import datasets_ext_app

app = App(name="codeintel", help="Unified CodeIntel CLI")

app.mount("datasets", datasets_ext_app)
```

From there, you can mirror this pattern for other command groups (`ingest`, `graphs`, `docs`, etc.): define a `*RuntimeCli` dataclass, per-command option dataclasses, a small `_runtime_from_cli` adapter into your existing `_common`/handler types, and a tiny command function that just wires data together and calls `_run(handler, ...)`.
