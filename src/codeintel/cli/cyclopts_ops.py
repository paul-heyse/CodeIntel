"""Cyclopts wiring for op, dataset, and serve command groups."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import App, Parameter

import codeintel.cli.main as legacy
from codeintel.cli.cli_errors import invoke_with_typer_translation
from codeintel.cli.cyclopts_common import RuntimeCLI
from codeintel.serving.operations.catalog import Operation, iter_operations

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

# Track dynamically registered operation command names to avoid duplicates
_REGISTERED_OP_COMMANDS: set[str] = set()

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
    """List available serving operations."""
    cfg = cfg or OpListCli()
    invoke_with_typer_translation(
        legacy.op_list,
        category=cfg.category,
        json_output=cfg.json_output,
    )


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
    runtime: Annotated[RuntimeCLI, Parameter(name="*")] = field(default_factory=RuntimeCLI)
    skip_prereqs: Annotated[
        bool,
        Parameter(
            name="--skip-prereqs",
            help="Skip prerequisite pipeline execution.",
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
        If an operation ID is not provided.
    """
    cfg = cfg or OpCallCli()
    if not cfg.op_id:
        raise SystemExit(2)
    runtime = cfg.runtime
    invoke_with_typer_translation(
        legacy.op_call,
        cfg.op_id,
        params=cfg.params,
        project_root=runtime.project_root,
        skip_prereqs=cfg.skip_prereqs,
        verbose=bool(runtime.verbose),
    )


def _command_name_for_operation(op: Operation) -> str:
    """Normalize operation ID into a CLI-friendly command name.

    Returns
    -------
    str
        Operation identifier with dots replaced by hyphens.
    """
    return op.id.replace(".", "-")


def _register_dynamic_operation(op: Operation) -> None:
    """Register a dynamic subcommand for an operation."""
    command_name = _command_name_for_operation(op)
    if command_name in _REGISTERED_OP_COMMANDS:
        return

    @op_app.command(name=command_name, help=op.summary or op.id)
    def dynamic_op(  # type: ignore[unused-ignore]
        params: Annotated[
            list[str] | None,
            Parameter(
                name=None,
                help="Operation parameters as key=value pairs.",
            ),
        ] = None,
        *,
        runtime: Annotated[RuntimeCLI, Parameter(name="*")] | None = None,
        skip_prereqs: Annotated[
            bool,
            Parameter(
                name="--skip-prereqs",
                help="Skip prerequisite pipeline execution.",
                negative=(),
            ),
        ] = False,
    ) -> None:
        runtime_cfg = runtime or RuntimeCLI()
        invoke_with_typer_translation(
            legacy.op_call,
            op.id,
            params=params,
            project_root=runtime_cfg.project_root,
            skip_prereqs=skip_prereqs,
            verbose=bool(runtime_cfg.verbose),
        )

    _REGISTERED_OP_COMMANDS.add(command_name)


def register_dynamic_operations() -> None:
    """Register subcommands for all operations in the catalog."""
    for op in iter_operations():
        _register_dynamic_operation(op)


# -----------------------------------------------------------------------------
# dataset commands
# -----------------------------------------------------------------------------


@dataclass
class DatasetListCli:
    """CLI surface for `codeintel dataset list`."""

    runtime: Annotated[RuntimeCLI, Parameter(name="*")] = field(default_factory=RuntimeCLI)
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
    """List datasets from the registry."""
    cfg = cfg or DatasetListCli()  # type: ignore[call-arg]
    runtime = cfg.runtime
    invoke_with_typer_translation(
        legacy.dataset_list,
        project_root=runtime.project_root,
        json_output=cfg.json_output,
    )


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
        If the required dataset key is missing.
    """
    cfg = cfg or DatasetDescribeCli()
    if not cfg.table_key:
        raise SystemExit(2)
    invoke_with_typer_translation(
        legacy.dataset_describe,
        table_key=cfg.table_key,
        json_output=cfg.json_output,
    )


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
    runtime: Annotated[RuntimeCLI, Parameter(name="*")] = field(default_factory=RuntimeCLI)


@dataset_app.command(name="verify")
def dataset_verify(
    cfg: Annotated[DatasetVerifyCli, Parameter(name="*")] | None = None,
) -> None:
    """Verify dataset contracts against actual data."""
    cfg = cfg or DatasetVerifyCli()  # type: ignore[call-arg]
    runtime = cfg.runtime
    invoke_with_typer_translation(
        legacy.dataset_verify,
        table_key=cfg.table_key,
        project_root=runtime.project_root,
    )


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
    runtime: Annotated[RuntimeCLI, Parameter(name="*")] | None = None,
) -> None:
    """Start the HTTP server."""
    runtime_cfg = runtime or RuntimeCLI()
    invoke_with_typer_translation(
        legacy.serve_http,
        host=host,
        port=port,
        auto_pipeline=auto_pipeline,
        reload=reload,
        project_root=runtime_cfg.project_root,
    )


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
    runtime: Annotated[RuntimeCLI, Parameter(name="*")] | None = None,
) -> None:
    """Start the MCP server."""
    runtime_cfg = runtime or RuntimeCLI()
    invoke_with_typer_translation(
        legacy.serve_mcp,
        auto_pipeline=auto_pipeline,
        project_root=runtime_cfg.project_root,
    )


register_dynamic_operations()


__all__ = [
    "dataset_app",
    "op_app",
    "register_dynamic_operations",
    "serve_app",
]
