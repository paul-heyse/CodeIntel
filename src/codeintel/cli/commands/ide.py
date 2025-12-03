"""IDE integration commands for the CodeIntel CLI.

This module provides Typer commands for IDE helper functionality,
including context hints for file paths.

Commands
--------
- **hints**: Emit IDE hints for a relative file path
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Annotated

import typer

from codeintel.cli.commands._common import (
    BuildDirOpt,
    CommitOpt,
    DbPathOpt,
    ProjectRootOpt,
    RepoOpt,
    RepoRootOpt,
    VerboseOpt,
    build_graph_runtime,
    build_runtime_or_exit,
    open_gateway_from_config,
    setup_logging,
)
from codeintel.serving.mcp.backend import DuckDBBackend

LOG = logging.getLogger(__name__)

ide_app = typer.Typer(
    name="ide",
    help="IDE helper commands.",
    no_args_is_help=True,
)


# -----------------------------------------------------------------------------
# Option Type Aliases
# -----------------------------------------------------------------------------

RelPathArg = Annotated[
    str,
    typer.Argument(help="File path relative to repo root (e.g., pkg/module.py)"),
]


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


@ide_app.command("hints")
def ide_hints(
    rel_path: RelPathArg,
    project_root: ProjectRootOpt = None,
    repo: RepoOpt = None,
    commit: CommitOpt = None,
    db_path: DbPathOpt = None,
    build_dir: BuildDirOpt = None,
    repo_root: RepoRootOpt = None,
    verbose: VerboseOpt = 0,
) -> None:
    """Emit IDE hints (module + subsystem context) for a relative file path.

    Returns JSON with module information, subsystem memberships, and other
    contextual hints useful for IDE integration.

    Examples
    --------
    .. code-block:: bash

        # Get hints for a file
        codeintel ide hints src/codeintel/cli/main.py

        # Using explicit repo configuration
        codeintel ide hints pkg/module.py --repo my-org/repo --commit abc123
    """
    setup_logging(verbose)

    runtime = build_runtime_or_exit(
        project_root=project_root,
        repo=repo,
        commit=commit,
        db_path=db_path,
        build_dir=build_dir,
        repo_root=repo_root,
    )

    gateway = open_gateway_from_config(runtime.cfg, read_only=True)
    graph_runtime = build_graph_runtime(runtime.cfg, gateway)
    engine = graph_runtime.engine

    backend = DuckDBBackend(
        gateway=gateway,
        repo=runtime.project.repo,
        commit=runtime.cfg.repo.commit,
        query_engine=engine,
    )

    response = backend.get_file_hints(rel_path=rel_path)
    if not response.found or not response.hints:
        LOG.error("No IDE hints found for %s", rel_path)
        typer.secho(f"No hints found for: {rel_path}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)

    payload = {
        "rel_path": rel_path,
        "hints": [hint.model_dump() for hint in response.hints],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")


__all__ = ["ide_app"]
