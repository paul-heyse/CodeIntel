"""Command group implementations for the CodeIntel CLI.

This module provides Typer sub-applications for:
- **pipeline**: Run full or operation-targeted pipelines, check status
- **op**: List and invoke serving operations (with dynamic per-operation commands)
- **dataset**: List, describe, and verify dataset contracts
- **serve**: Start HTTP or MCP servers

Each command group is a Typer app that can be composed into the main CLI.

Dynamic Operation Commands
--------------------------
Operations from the catalog are automatically registered as individual CLI
commands under the ``op`` group. For example:

    codeintel op function-summary --goid-h128 123456
    codeintel op datasets-list --limit 10

This uses the "string tunnel" pattern where all operation parameters are
accepted as strings via CLI and coerced to their proper Python types at
runtime. This works around Typer's limitations with Union types.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn

from codeintel.cli.op_params import register_dynamic_commands
from codeintel.cli.project import (
    ProjectNotFoundError,
    ProjectRuntime,
    build_project_runtime,
)
from codeintel.config.datasets import (
    get_dataset_contracts_by_table_key,
)
from codeintel.pipeline.executor import run_pipeline
from codeintel.pipeline.op_planner import (
    OperationPrereqOptions,
    build_prereq_summary,
    ensure_prerequisites_for_operation,
)
from codeintel.pipeline.planner import PipelinePlanOptions
from codeintel.pipeline.spec import FULL_PIPELINE
from codeintel.serving.bootstrap import build_service_stack
from codeintel.serving.http.fastapi import create_app as create_http_app
from codeintel.serving.mcp.server import main as run_mcp_server
from codeintel.serving.operations.catalog import (
    get_operation,
    iter_operations,
)
from codeintel.storage.validation import collect_contract_issues

LOG = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Type Aliases for CLI Options
# Using Optional[X] for Typer compatibility (Typer doesn't support X | None syntax)
# -----------------------------------------------------------------------------

ProjectRootOpt = Annotated[
    Path | None,
    typer.Option("--root", "-r", help="Explicit project root directory"),
]

VerboseOpt = Annotated[
    bool | None,
    typer.Option("--verbose", "-v", help="Enable verbose output", is_flag=True),
]
JsonOutputOpt = Annotated[
    bool | None,
    typer.Option("--json", help="Output as JSON", is_flag=True),
]
SkipAnalyticsOpt = Annotated[
    bool | None,
    typer.Option("--skip-analytics", help="Skip analytics stage", is_flag=True),
]
SkipPrereqsOpt = Annotated[
    bool | None,
    typer.Option("--skip-prereqs", help="Skip prerequisite pipeline execution", is_flag=True),
]
AutoPipelineOpt = Annotated[
    bool | None,
    typer.Option(
        "--auto-pipeline", help="Enable automatic prerequisite pipeline execution", is_flag=True
    ),
]
ReloadOpt = Annotated[
    bool | None,
    typer.Option("--reload", help="Enable auto-reload for development", is_flag=True),
]

# -----------------------------------------------------------------------------
# Pipeline Commands
# -----------------------------------------------------------------------------

pipeline_app = typer.Typer(
    name="pipeline",
    help="Pipeline orchestration commands.",
    no_args_is_help=True,
)


def _build_runtime_or_exit(project_root: Path | None) -> ProjectRuntime:
    """Build project runtime or exit with error message.

    Parameters
    ----------
    project_root
        Optional explicit project root path.

    Returns
    -------
    ProjectRuntime
        Constructed runtime context.

    Raises
    ------
    typer.Exit
        If project root cannot be found.
    """
    try:
        return build_project_runtime(project_root)
    except ProjectNotFoundError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


def _resolve_flag(value: object) -> bool:
    """Resolve an optional flag value to a boolean.

    Parameters
    ----------
    value
        Flag value from Typer (may be None, bool, or other).

    Returns
    -------
    bool
        True if value is truthy and not None, False otherwise.
    """
    if value is None:
        return False
    return bool(value)


@pipeline_app.command("run-full")
def pipeline_run_full(
    project_root: ProjectRootOpt = None,
    verbose: VerboseOpt = None,
) -> None:
    """Run the full pipeline (ingest → graphs → analytics)."""
    if _resolve_flag(verbose):
        logging.basicConfig(level=logging.DEBUG)

    runtime = _build_runtime_or_exit(project_root)

    typer.echo(f"Running full pipeline for {runtime.project.repo}...")

    options = PipelinePlanOptions(
        snapshot=runtime.snapshot,
        paths=runtime.paths,
        gateway=runtime.gateway,
        tools=runtime.tools,
        trigger="cli",
    )

    run_record = run_pipeline(spec=FULL_PIPELINE, options=options)

    typer.secho(
        f"Pipeline completed: run_id={run_record.run_id} status={run_record.status}",
        fg=typer.colors.GREEN if run_record.status == "completed" else typer.colors.RED,
    )


@pipeline_app.command("run-op")
def pipeline_run_op(
    op_id: Annotated[str, typer.Argument(help="Operation ID to run prerequisites for")],
    project_root: ProjectRootOpt = None,
    skip_analytics: SkipAnalyticsOpt = None,
    verbose: VerboseOpt = None,
) -> None:
    """Run minimal pipeline stages required for an operation.

    Raises
    ------
    typer.Exit
        If operation ID is unknown.
    """
    if _resolve_flag(verbose):
        logging.basicConfig(level=logging.DEBUG)

    runtime = _build_runtime_or_exit(project_root)

    op = get_operation(op_id)
    if op is None:
        typer.secho(f"Error: Unknown operation: {op_id}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Running prerequisites for operation '{op_id}'...")

    summary = build_prereq_summary(op_id, runtime.snapshot)
    typer.echo(f"  Required datasets: {len(summary.expanded_tables)}")
    typer.echo(f"  Required graphs: {list(summary.required_graphs)}")

    prereq_options = OperationPrereqOptions(
        snapshot=runtime.snapshot,
        paths=runtime.paths,
        gateway=runtime.gateway,
        tools=runtime.tools,
        include_analytics=not _resolve_flag(skip_analytics),
        trigger="cli",
    )

    run_record = ensure_prerequisites_for_operation(op_id=op_id, options=prereq_options)

    typer.secho(
        f"Prerequisites completed: run_id={run_record.run_id} status={run_record.status}",
        fg=typer.colors.GREEN if run_record.status == "completed" else typer.colors.RED,
    )


RunIdOpt = Annotated[
    str | None, typer.Option("--run-id", help="Specific run ID to show details for")
]
LimitOpt = Annotated[int, typer.Option("--limit", "-n", help="Number of recent runs to show")]


@pipeline_app.command("status")
def pipeline_status(
    run_id: RunIdOpt = None,
    limit: LimitOpt = 10,
    project_root: ProjectRootOpt = None,
) -> None:
    """Show pipeline run status and history.

    Raises
    ------
    typer.Exit
        If specified run ID is not found.
    """
    runtime = _build_runtime_or_exit(project_root)

    if run_id:
        record = runtime.gateway.runs.fetch_run(run_id)
        if record is None:
            typer.secho(f"Error: Run not found: {run_id}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        typer.echo(f"Run: {record.run_id}")
        typer.echo(f"  Kind: {record.kind}")
        typer.echo(f"  Pipeline: {record.pipeline_name or 'N/A'}")
        typer.echo(f"  Status: {record.status}")
        typer.echo(f"  Trigger: {record.trigger}")
        typer.echo(f"  Started: {record.started_at}")
        typer.echo(f"  Completed: {record.completed_at}")

        steps = runtime.gateway.runs.fetch_steps(run_id)
        if steps:
            typer.echo("  Steps:")
            for step in steps:
                typer.echo(f"    - {step.name}: {step.status}")
    else:
        runs = runtime.gateway.runs.fetch_recent_runs(limit=limit)
        if not runs:
            typer.echo("No pipeline runs found.")
            return

        typer.echo(f"Recent pipeline runs (showing {len(runs)}):")
        for record in runs:
            status_color = typer.colors.GREEN if record.status == "succeeded" else typer.colors.RED
            typer.secho(
                f"  {record.run_id[:8]}  {record.kind:<20} {record.status:<12} {record.started_at}",
                fg=status_color,
            )


# -----------------------------------------------------------------------------
# Operation Commands
# -----------------------------------------------------------------------------

op_app = typer.Typer(
    name="op",
    help="Operation invocation commands.",
    no_args_is_help=True,
)

CategoryOpt = Annotated[
    str | None, typer.Option("--category", "-c", help="Filter by operation category")
]


@op_app.command("list")
def op_list(
    category: CategoryOpt = None,
    json_output: JsonOutputOpt = None,
) -> None:
    """List available serving operations."""
    operations = list(iter_operations())

    if category:
        operations = [op for op in operations if op.category == category]

    if _resolve_flag(json_output):
        output = [
            {
                "id": op.id,
                "category": op.category,
                "summary": op.summary,
                "http_path": op.http_path,
                "tool_name": op.tool_name,
            }
            for op in operations
        ]
        typer.echo(json.dumps(output, indent=2))
    else:
        typer.echo(f"Available operations ({len(operations)}):")
        for op in sorted(operations, key=lambda o: o.id):
            typer.echo(f"  {op.id:<35} {op.summary}")


def _parse_param_value(value: str) -> str | int | float | bool:
    """Parse a parameter value from string to appropriate type.

    Parameters
    ----------
    value
        String value to parse.

    Returns
    -------
    str | int | float | bool
        Parsed value in appropriate type.
    """
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    with contextlib.suppress(ValueError):
        return int(value)
    with contextlib.suppress(ValueError):
        return float(value)
    return value


def _invoke_operation(
    op_id: str,
    kwargs: dict[str, Any],
    runtime: ProjectRuntime,
) -> None:
    """Invoke an operation and print the result.

    Parameters
    ----------
    op_id
        Operation identifier.
    kwargs
        Operation parameters.
    runtime
        Project runtime context.

    Raises
    ------
    typer.Exit
        If backend method is not found.
    """
    op = get_operation(op_id)
    if op is None:
        typer.secho(f"Error: Unknown operation: {op_id}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Invoking operation '{op_id}'...")

    stack = build_service_stack(runtime.serving, gateway=runtime.gateway)
    try:
        method = getattr(stack.service, op.backend_method, None)
        if method is None:
            typer.secho(
                f"Error: Backend method not found: {op.backend_method}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        result = method(**kwargs)

        # Serialize result to JSON
        if hasattr(result, "model_dump"):
            output = result.model_dump(mode="json")
        elif hasattr(result, "__dict__"):
            output = result.__dict__
        else:
            output = result

        typer.echo(json.dumps(output, indent=2, default=str))
    finally:
        stack.close()


ParamsArg = Annotated[
    list[str] | None, typer.Argument(help="Operation parameters as key=value pairs")
]


@op_app.command("call")
def op_call(
    op_id: Annotated[str, typer.Argument(help="Operation ID to invoke")],
    params: ParamsArg = None,
    project_root: ProjectRootOpt = None,
    skip_prereqs: SkipPrereqsOpt = None,
    verbose: VerboseOpt = None,
) -> None:
    """Invoke a serving operation end-to-end.

    Parameters are provided as key=value pairs after the operation ID.

    Example:
        codeintel-app op call function.summary goid_h128=123456

    Raises
    ------
    typer.Exit
        If operation ID is unknown or parameter format is invalid.
    """
    if _resolve_flag(verbose):
        logging.basicConfig(level=logging.DEBUG)

    runtime = _build_runtime_or_exit(project_root)

    op = get_operation(op_id)
    if op is None:
        typer.secho(f"Error: Unknown operation: {op_id}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Parse parameters
    kwargs: dict[str, str | int | float | bool] = {}
    for param in params or []:
        if "=" not in param:
            typer.secho(
                f"Error: Invalid parameter format: {param} (expected key=value)",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        key, value = param.split("=", 1)
        kwargs[key] = _parse_param_value(value)

    # Run prerequisites if not skipped
    if not _resolve_flag(skip_prereqs):
        typer.echo(f"Running prerequisites for '{op_id}'...")
        prereq_options = OperationPrereqOptions(
            snapshot=runtime.snapshot,
            paths=runtime.paths,
            gateway=runtime.gateway,
            tools=runtime.tools,
            include_analytics=True,
            trigger="cli",
        )
        ensure_prerequisites_for_operation(op_id=op_id, options=prereq_options)

    _invoke_operation(op_id, kwargs, runtime)


# -----------------------------------------------------------------------------
# Dynamic Operation Commands
# -----------------------------------------------------------------------------


def _dynamic_op_invoke_callback(
    op_id: str,
    params: dict[str, Any],
    project_root: Path | None,
    *,
    skip_prereqs: bool,
    verbose: bool,
) -> None:
    """Invoke a dynamically registered operation command.

    This function is called by dynamically registered operation commands
    to actually execute the operation.

    Parameters
    ----------
    op_id
        Operation identifier.
    params
        Operation parameters (already coerced to proper types).
    project_root
        Optional project root path.
    skip_prereqs
        Whether to skip prerequisite pipeline execution.
    verbose
        Whether to enable verbose output.
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    runtime = _build_runtime_or_exit(project_root)

    # Run prerequisites if not skipped
    if not skip_prereqs:
        typer.echo(f"Running prerequisites for '{op_id}'...")
        prereq_options = OperationPrereqOptions(
            snapshot=runtime.snapshot,
            paths=runtime.paths,
            gateway=runtime.gateway,
            tools=runtime.tools,
            include_analytics=True,
            trigger="cli",
        )
        ensure_prerequisites_for_operation(op_id=op_id, options=prereq_options)

    # Invoke the operation
    _invoke_operation(op_id, params, runtime)


# Register dynamic commands for all operations at module load time
# This enables per-operation commands like: codeintel op function-summary --goid-h128 123
_dynamic_command_count = register_dynamic_commands(op_app, _dynamic_op_invoke_callback)
LOG.debug("Registered %d dynamic operation commands", _dynamic_command_count)


# -----------------------------------------------------------------------------
# Dataset Commands
# -----------------------------------------------------------------------------

dataset_app = typer.Typer(
    name="dataset",
    help="Dataset inspection commands.",
    no_args_is_help=True,
)


@dataset_app.command("list")
def dataset_list(
    project_root: ProjectRootOpt = None,
    json_output: JsonOutputOpt = None,
) -> None:
    """List datasets from the registry."""
    runtime = _build_runtime_or_exit(project_root)

    # Get datasets from the registry's meta mapping
    registry = runtime.gateway.datasets
    meta = registry.meta or {}

    if _resolve_flag(json_output):
        output = [
            {
                "name": name,
                "table_key": contract.table_key,
                "is_view": contract.is_view,
                "owner_package": contract.owner_package,
            }
            for name, contract in sorted(meta.items())
        ]
        typer.echo(json.dumps(output, indent=2))
    else:
        typer.echo(f"Datasets ({len(meta)}):")
        for name in sorted(meta.keys()):
            contract = meta[name]
            view_marker = "(view)" if contract.is_view else ""
            typer.echo(f"  {contract.table_key:<40} {view_marker}")


@dataset_app.command("describe")
def dataset_describe(
    table_key: Annotated[str, typer.Argument(help="Dataset table key (e.g., 'core.goids')")],
    json_output: JsonOutputOpt = None,
) -> None:
    """Show contract details for a dataset.

    Raises
    ------
    typer.Exit
        If dataset is not found.
    """
    contracts = get_dataset_contracts_by_table_key()
    contract = contracts.get(table_key)

    if contract is None:
        typer.secho(f"Error: Dataset not found: {table_key}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    columns = contract.schema.columns if contract.schema else []

    if _resolve_flag(json_output):
        output = {
            "name": contract.name,
            "table_key": contract.table_key,
            "description": contract.description,
            "owner_package": contract.owner_package,
            "columns": [
                {"name": col.name, "type": col.type, "nullable": col.nullable} for col in columns
            ],
            "upstream_dependencies": list(contract.upstream_dependencies),
        }
        typer.echo(json.dumps(output, indent=2))
    else:
        typer.echo(f"Dataset: {contract.name}")
        typer.echo(f"  Table: {contract.table_key}")
        typer.echo(f"  Owner: {contract.owner_package}")
        typer.echo(f"  Description: {contract.description}")
        if columns:
            typer.echo("  Columns:")
            for col in columns:
                nullable = "nullable" if col.nullable else "required"
                typer.echo(f"    - {col.name}: {col.type} ({nullable})")
        if contract.upstream_dependencies:
            typer.echo(f"  Dependencies: {list(contract.upstream_dependencies)}")


TableKeyArg = Annotated[
    str | None,
    typer.Argument(help="Dataset table key to verify (verifies all if not specified)"),
]


@dataset_app.command("verify")
def dataset_verify(
    table_key: TableKeyArg = None,
    project_root: ProjectRootOpt = None,
) -> None:
    """Verify dataset contracts against actual data.

    Raises
    ------
    typer.Exit
        If contract issues are found.
    """
    runtime = _build_runtime_or_exit(project_root)

    issues = collect_contract_issues(runtime.gateway.con)

    # Filter issues by table_key if provided (basic string matching)
    if table_key:
        issues = [i for i in issues if table_key in i]

    if not issues:
        typer.secho("All dataset contracts verified successfully.", fg=typer.colors.GREEN)
    else:
        typer.secho(f"Found {len(issues)} contract issues:", fg=typer.colors.RED)
        for issue in issues:
            typer.echo(f"  {issue}")
        raise typer.Exit(code=1)


# -----------------------------------------------------------------------------
# Serve Commands
# -----------------------------------------------------------------------------

serve_app = typer.Typer(
    name="serve",
    help="Server startup commands.",
    no_args_is_help=True,
)

AUTO_PIPELINE_ENV = "CODEINTEL_AUTO_PIPELINE"


def _setup_serving_env(runtime: ProjectRuntime, *, auto_pipeline: bool) -> None:
    """Set up environment variables for the serving layer.

    Parameters
    ----------
    runtime
        Project runtime context.
    auto_pipeline
        Whether to enable auto-pipeline.
    """
    os.environ["CODEINTEL_REPO"] = runtime.project.repo
    os.environ["CODEINTEL_COMMIT"] = runtime.snapshot.commit
    os.environ["CODEINTEL_DB_PATH"] = str(runtime.paths.db_path)
    os.environ["CODEINTEL_REPO_ROOT"] = str(runtime.root)

    if auto_pipeline:
        os.environ[AUTO_PIPELINE_ENV] = "1"


HostOpt = Annotated[str, typer.Option("--host", "-h", help="Host to bind to")]
PortOpt = Annotated[int, typer.Option("--port", "-p", help="Port to bind to")]


@serve_app.command("http")
def serve_http(
    host: HostOpt = "127.0.0.1",
    port: PortOpt = 8000,
    auto_pipeline: AutoPipelineOpt = None,
    reload: ReloadOpt = None,
    project_root: ProjectRootOpt = None,
) -> None:
    """Start the HTTP server."""
    runtime = _build_runtime_or_exit(project_root)

    auto_pipeline_enabled = _resolve_flag(auto_pipeline)
    reload_enabled = _resolve_flag(reload)

    _setup_serving_env(runtime, auto_pipeline=auto_pipeline_enabled)

    typer.echo(f"Starting HTTP server at http://{host}:{port}")
    typer.echo(f"  Repo: {runtime.project.repo}")
    typer.echo(f"  Commit: {runtime.snapshot.commit}")
    typer.echo(f"  Database: {runtime.paths.db_path}")
    typer.echo(f"  Auto-pipeline: {'enabled' if auto_pipeline_enabled else 'disabled'}")

    if reload_enabled:
        # Use factory mode for reload support (uses env var for auto_pipeline)
        uvicorn.run(
            "codeintel.serving.http.fastapi:app",
            host=host,
            port=port,
            reload=True,
        )
    else:
        # Create app directly to pass auto_pipeline flag
        app = create_http_app(
            gateway=runtime.gateway,
            auto_pipeline=auto_pipeline_enabled,
        )
        uvicorn.run(app, host=host, port=port)


@serve_app.command("mcp")
def serve_mcp(
    auto_pipeline: AutoPipelineOpt = None,
    project_root: ProjectRootOpt = None,
) -> None:
    """Start the MCP server."""
    runtime = _build_runtime_or_exit(project_root)

    _setup_serving_env(runtime, auto_pipeline=_resolve_flag(auto_pipeline))

    auto_pipeline_enabled = _resolve_flag(auto_pipeline)

    typer.echo("Starting MCP server...", err=True)
    typer.echo(f"  Repo: {runtime.project.repo}", err=True)
    typer.echo(f"  Commit: {runtime.snapshot.commit}", err=True)
    typer.echo(f"  Database: {runtime.paths.db_path}", err=True)
    typer.echo(f"  Auto-pipeline: {'enabled' if auto_pipeline_enabled else 'disabled'}", err=True)

    sys.exit(run_mcp_server())


__all__ = [
    "dataset_app",
    "op_app",
    "pipeline_app",
    "serve_app",
]
