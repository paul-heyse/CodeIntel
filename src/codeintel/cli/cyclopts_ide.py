"""Cyclopts wiring for IDE helper commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import RuntimeCliOptions
from codeintel.cli.commands.ide import IdeHintsOptions, ide_hints_handler
from codeintel.cli.cyclopts_common import ProjectRoot, Verbose

ide_app = App(
    name="ide",
    help="IDE helper commands.",
)


@dataclass
class IdeRuntimeCli:
    """Runtime selection for IDE commands."""

    project_root: ProjectRoot = None
    repo: Annotated[
        str | None,
        Parameter(
            name="--repo",
            help="Repository slug (e.g., 'org/repo'). Uses project config if omitted.",
        ),
    ] = None
    commit: Annotated[
        str | None,
        Parameter(
            name="--commit",
            help="Commit SHA. Uses project config if omitted.",
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
            help="Build directory (default: build/).",
        ),
    ] = None
    repo_root: Annotated[
        Path | None,
        Parameter(
            name="--repo-root",
            help="Path to repository root (default: current directory).",
        ),
    ] = None
    verbose: Verbose = 0


@ide_app.command(name="hints")
def hints(
    rel_path: Annotated[
        str,
        Parameter(
            name=None,
            help="File path relative to repo root (e.g., pkg/module.py).",
        ),
    ],
    runtime: Annotated[IdeRuntimeCli, Parameter(name="*")] | None = None,
) -> None:
    """Emit IDE hints (module + subsystem context) for a relative file path."""
    cfg = runtime or IdeRuntimeCli()
    options = IdeHintsOptions(
        rel_path=rel_path,
        runtime_options=RuntimeCliOptions(
            project_root=cfg.project_root,
            repo=cfg.repo,
            commit=cfg.commit,
            db_path=cfg.db_path,
            build_dir=cfg.build_dir,
            repo_root=cfg.repo_root,
        ),
        verbose=cfg.verbose,
    )
    ide_hints_handler(options)


__all__ = ["ide_app"]
