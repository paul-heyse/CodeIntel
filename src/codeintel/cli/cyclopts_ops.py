"""Cyclopts wiring for op, dataset, and serve command groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import typer
from cyclopts import App, Parameter

import codeintel.cli.main as legacy
from codeintel.cli.cyclopts_common import ProjectRoot

op_app = App(
    name="op",
    help="Operation invocation commands.",
)

dataset_app = App(
    name="dataset",
    help="Dataset inspection commands.",
)

serve_app = App(
    name="serve",
    help="HTTP and MCP server commands.",
)


# -----------------------------------------------------------------------------
# op commands
# -----------------------------------------------------------------------------


@dataclass
class OpListCli:
    """CLI surface for `codeintel op list`."""

    category: Annotated[
        str | None,
        Parameter(
            name=["--category", "-c"],
            help="Filter by operation category.",
        ),
    ] = None
    json_output: Annotated[
        bool,
        Parameter(
            name="--json",
            help="Output as JSON.",
            negative=(),
        ),
    ] = False


@op_app.command(name="list")
def op_list(
    cfg: Annotated[OpListCli, Parameter(name="*")] | None = None,
) -> None:
    """List available serving operations.

    Raises
    ------
    SystemExit
        When the underlying handler triggers a CLI exit.
    """
    cfg = cfg or OpListCli()
    try:
        legacy.op_list(category=cfg.category, json_output=cfg.json_output)
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from exc


@dataclass
class OpCallCli:
    """CLI surface for `codeintel op call`."""

    op_id: Annotated[
        str,
        Parameter(
            help="Operation ID to invoke.",
        ),
    ] = ""
    params: Annotated[
        list[str] | None,
        Parameter(
            name=None,
            help="Operation parameters as key=value pairs.",
        ),
    ] = None
    project_root: ProjectRoot = None
    skip_prereqs: Annotated[
        bool,
        Parameter(
            name="--skip-prereqs",
            help="Skip prerequisite pipeline execution.",
            negative=(),
        ),
    ] = False
    verbose: Annotated[
        bool,
        Parameter(
            name=["--verbose", "-v"],
            help="Enable verbose output.",
            negative=(),
        ),
    ] = False


@op_app.command(name="call")
def op_call(
    cfg: Annotated[OpCallCli, Parameter(name="*")] | None = None,
) -> None:
    """Invoke a serving operation end-to-end.

    Raises
    ------
    SystemExit
        When required arguments are missing or the handler exits.
    """
    cfg = cfg or OpCallCli()
    if not cfg.op_id:
        raise SystemExit(2)
    try:
        legacy.op_call(
            cfg.op_id,
            params=cfg.params,
            project_root=cfg.project_root,
            skip_prereqs=cfg.skip_prereqs,
            verbose=cfg.verbose,
        )
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from exc


# -----------------------------------------------------------------------------
# dataset commands
# -----------------------------------------------------------------------------


@dataclass
class DatasetListCli:
    """CLI surface for `codeintel dataset list`."""

    project_root: ProjectRoot = None
    json_output: Annotated[
        bool,
        Parameter(
            name="--json",
            help="Output as JSON.",
            negative=(),
        ),
    ] = False


@dataset_app.command(name="list")
def dataset_list(
    cfg: Annotated[DatasetListCli, Parameter(name="*")] | None = None,
) -> None:
    """List datasets from the registry.

    Raises
    ------
    SystemExit
        When the underlying handler triggers a CLI exit.
    """
    cfg = cfg or DatasetListCli()
    try:
        legacy.dataset_list(project_root=cfg.project_root, json_output=cfg.json_output)
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from exc


@dataclass
class DatasetDescribeCli:
    """CLI surface for `codeintel dataset describe`."""

    table_key: Annotated[
        str,
        Parameter(
            help="Dataset table key (e.g., 'core.goids').",
        ),
    ] = ""
    json_output: Annotated[
        bool,
        Parameter(
            name="--json",
            help="Output as JSON.",
            negative=(),
        ),
    ] = False


@dataset_app.command(name="describe")
def dataset_describe(
    cfg: Annotated[DatasetDescribeCli, Parameter(name="*")] | None = None,
) -> None:
    """Show contract details for a dataset.

    Raises
    ------
    SystemExit
        When required arguments are missing or the handler exits.
    """
    cfg = cfg or DatasetDescribeCli()
    if not cfg.table_key:
        raise SystemExit(2)
    try:
        legacy.dataset_describe(table_key=cfg.table_key, json_output=cfg.json_output)
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from exc


@dataclass
class DatasetVerifyCli:
    """CLI surface for `codeintel dataset verify`."""

    table_key: Annotated[
        str | None,
        Parameter(
            name=None,
            help="Dataset table key to verify (verifies all if not specified).",
        ),
    ] = None
    project_root: ProjectRoot = None


@dataset_app.command(name="verify")
def dataset_verify(
    cfg: Annotated[DatasetVerifyCli, Parameter(name="*")] | None = None,
) -> None:
    """Verify dataset contracts against actual data.

    Raises
    ------
    SystemExit
        When the underlying handler triggers a CLI exit.
    """
    cfg = cfg or DatasetVerifyCli()
    try:
        legacy.dataset_verify(table_key=cfg.table_key, project_root=cfg.project_root)
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from exc


# -----------------------------------------------------------------------------
# serve commands
# -----------------------------------------------------------------------------


@serve_app.command(name="http")
def serve_http(
    host: Annotated[
        str,
        Parameter(
            name=["--host", "-h"],
            help="Host to bind to.",
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        Parameter(
            name=["--port", "-p"],
            help="Port to bind to.",
        ),
    ] = 8000,
    *,
    auto_pipeline: Annotated[
        bool,
        Parameter(
            name="--auto-pipeline",
            help="Enable automatic prerequisite pipeline execution.",
            negative=(),
        ),
    ] = False,
    reload: Annotated[
        bool,
        Parameter(
            name="--reload",
            help="Enable auto-reload for development.",
            negative=(),
        ),
    ] = False,
    project_root: ProjectRoot = None,
) -> None:
    """Start the HTTP server.

    Raises
    ------
    SystemExit
        When the underlying handler triggers a CLI exit.
    """
    try:
        legacy.serve_http(
            host=host,
            port=port,
            auto_pipeline=auto_pipeline,
            reload=reload,
            project_root=project_root,
        )
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from exc


@serve_app.command(name="mcp")
def serve_mcp(
    *,
    auto_pipeline: Annotated[
        bool,
        Parameter(
            name="--auto-pipeline",
            help="Enable automatic prerequisite pipeline execution.",
            negative=(),
        ),
    ] = False,
    project_root: ProjectRoot = None,
) -> None:
    """Start the MCP server.

    Raises
    ------
    SystemExit
        When the underlying handler triggers a CLI exit.
    """
    try:
        legacy.serve_mcp(auto_pipeline=auto_pipeline, project_root=project_root)
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from exc


__all__ = ["dataset_app", "op_app", "serve_app"]
