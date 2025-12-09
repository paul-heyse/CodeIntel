"""Shared Cyclopts primitives and runtime helpers for the CodeIntel CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import (
    BackendFlags,
    OutputFormat,
    RuntimeCliOptions,
    build_config_from_options,
)
from codeintel.cli.project import (
    ProjectConfig,
    ProjectNotFoundError,
    ProjectRuntime,
    StorageProjectConfig,
    build_project_runtime,
)
from codeintel.config.models import CliPathsInput
from codeintel.config.primitives import SnapshotRef
from codeintel.config.serving_models import ServingConfig
from codeintel.storage.gateway import StorageConfig, open_gateway


def make_root_app() -> App:
    """Construct the root Cyclopts application with shared defaults.

    Returns
    -------
    App
        Root Cyclopts application configured with default parameters.
    """
    return App(
        name="codeintel",
        help="CodeIntel unified CLI for build, analytics, and serving operations.",
        default_parameter=Parameter(
            show_default=True,
        ),
    )


Verbose = Annotated[
    int,
    Parameter(
        name=["--verbose", "-v"],
        help="Increase verbosity (0=warnings, 1=info, 2=debug).",
    ),
]

ProjectRoot = Annotated[
    Path | None,
    Parameter(
        name=["--root", "-r"],
        help="Explicit project root directory.",
    ),
]

OutputFmt = Annotated[
    OutputFormat,
    Parameter(
        name="--output-format",
        help="Output format.",
        show_choices=True,
    ),
]

JsonFlag = Annotated[
    bool,
    Parameter(
        name="--json",
        help="Alias for --output-format json.",
        negative=(),
    ),
]


class RuntimeCliError(Exception):
    """Raised when CLI runtime resolution fails."""


def build_runtime_from_cli(
    options: RuntimeCliOptions,
    *,
    allow_fallback: bool = True,
) -> ProjectRuntime:
    """Build a :class:`ProjectRuntime` from CLI options without Typer exits.

    Returns
    -------
    ProjectRuntime
        Constructed runtime context.

    Raises
    ------
    RuntimeCliError
        If a project cannot be resolved from the provided options.
    """
    try:
        return build_project_runtime(options.project_root)
    except ProjectNotFoundError:
        if not allow_fallback:
            message = "No codeintel.yaml found and fallback disabled."
            raise RuntimeCliError(message) from None

    if options.repo is None or options.commit is None:
        msg = (
            "No codeintel.yaml found. Provide --repo and --commit explicitly, "
            "or create a project file."
        )
        raise RuntimeCliError(msg)

    resolved_repo_root = options.repo_root or Path.cwd()
    resolved_db_path = options.db_path or Path("build/db/codeintel.duckdb")
    resolved_build_dir = options.build_dir or Path("build")

    paths_cfg = CliPathsInput(
        repo_root=resolved_repo_root,
        build_dir=resolved_build_dir,
        db_path=resolved_db_path,
        document_output_dir=options.document_output_dir,
    )

    cfg = build_config_from_options(
        repo=options.repo,
        commit=options.commit,
        paths_cfg=paths_cfg,
        backend=options.backend,
    )

    snapshot = SnapshotRef(
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        repo_root=cfg.paths.repo_root,
    )
    paths = cfg.build_paths
    paths.db_path.parent.mkdir(parents=True, exist_ok=True)

    storage_cfg = StorageConfig.for_ingest(db_path=paths.db_path)
    gateway = open_gateway(storage_cfg)

    project = ProjectConfig(
        repo=cfg.repo.repo,
        storage=StorageProjectConfig(db_path=paths.db_path),
    )

    serving = ServingConfig(
        mode="local_db",
        repo_root=cfg.paths.repo_root,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        db_path=paths.db_path,
        read_only=True,
    )

    return ProjectRuntime(
        root=resolved_repo_root,
        project=project,
        cfg=cfg,
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tools=cfg.tools,
        serving=serving,
    )


@dataclass(frozen=True)
class RuntimeWithFormat:
    """Bundle runtime options with output formatting toggles."""

    runtime: RuntimeCliOptions
    output_format: OutputFormat


__all__ = [
    "BackendFlags",
    "JsonFlag",
    "OutputFmt",
    "OutputFormat",
    "ProjectRoot",
    "RuntimeCliError",
    "RuntimeCliOptions",
    "RuntimeWithFormat",
    "Verbose",
    "build_runtime_from_cli",
    "make_root_app",
]
