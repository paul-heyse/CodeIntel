"""Command group implementations for the CodeIntel CLI.

This module provides Typer sub-applications for:
- **pipeline**: Run full or operation-targeted pipelines, check status
- **op**: List and invoke serving operations
- **dataset**: List, describe, and verify dataset contracts
- **serve**: Start HTTP or MCP servers

Each command group is a Typer app that can be composed into the main CLI.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from codeintel.cli.project import (
    ProjectNotFoundError,
    ProjectRuntime,
    build_project_runtime,
)
from codeintel.config.datasets import (
    get_dataset_contracts_by_table_key,
)
from codeintel.pipeline.op_planner import (
    OperationPrereqOptions,
    build_prereq_summary,
    ensure_prerequisites_for_operation,
)
from codeintel.pipeline.spec import FULL_PIPELINE
from codeintel.serving.operations.catalog import (
    get_operation,
    iter_operations,
)

LOG = logging.getLogger(__name__)

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
    """
    try:
        return build_project_runtime(project_root)
    except ProjectNotFoundError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@pipeline_app.command("run-full")
def pipeline_run_full(
    project_root: Annotated[
        Optional[Path],
        typer.Option("--root", "-r", help="Explicit project root directory"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """Run the full pipeline (ingest → graphs → analytics)."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    runtime = _build_runtime_or_exit(project_root)

    typer.echo(f"Running full pipeline for {runtime.project.repo}...")

    from codeintel.pipeline.executor import run_pipeline
    from codeintel.pipeline.planner import PipelinePlanOptions

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
    project_root: Annotated[
        Optional[Path],
        typer.Option("--root", "-r", help="Explicit project root directory"),
    ] = None,
    skip_analytics: Annotated[
        bool,
        typer.Option("--skip-analytics", help="Skip analytics stage"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """Run minimal pipeline stages required for an operation."""
    if verbose:
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
        include_analytics=not skip_analytics,
        trigger="cli",
    )

    run_record = ensure_prerequisites_for_operation(op_id=op_id, options=prereq_options)

    typer.secho(
        f"Prerequisites completed: run_id={run_record.run_id} status={run_record.status}",
        fg=typer.colors.GREEN if run_record.status == "completed" else typer.colors.RED,
    )


@pipeline_app.command("status")
def pipeline_status(
    run_id: Annotated[
        Optional[str],
        typer.Option("--run-id", help="Specific run ID to show details for"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Number of recent runs to show"),
    ] = 10,
    project_root: Annotated[
        Optional[Path],
        typer.Option("--root", "-r", help="Explicit project root directory"),
    ] = None,
) -> None:
    """Show pipeline run status and history."""
    runtime = _build_runtime_or_exit(project_root)

    if run_id:
        record = runtime.gateway.runs.fetch_run(run_id)
        if record is None:
            typer.secho(f"Error: Run not found: {run_id}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        typer.echo(f"Run: {record.run_id}")
        typer.echo(f"  Spec: {record.spec_id}")
        typer.echo(f"  Kind: {record.run_kind}")
        typer.echo(f"  Status: {record.status}")
        typer.echo(f"  Trigger: {record.trigger}")
        typer.echo(f"  Started: {record.started_at}")
        typer.echo(f"  Ended: {record.ended_at}")

        steps = runtime.gateway.runs.fetch_steps(run_id)
        if steps:
            typer.echo("  Steps:")
            for step in steps:
                typer.echo(f"    - {step.step_name}: {step.status}")
    else:
        runs = runtime.gateway.runs.fetch_recent_runs(limit=limit)
        if not runs:
            typer.echo("No pipeline runs found.")
            return

        typer.echo(f"Recent pipeline runs (showing {len(runs)}):")
        for record in runs:
            status_color = typer.colors.GREEN if record.status == "completed" else typer.colors.RED
            typer.secho(
                f"  {record.run_id[:8]}  {record.spec_id:<20} {record.status:<12} {record.started_at}",
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


@op_app.command("list")
def op_list(
    category: Annotated[
        Optional[str],
        typer.Option("--category", "-c", help="Filter by operation category"),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
) -> None:
    """List available serving operations."""
    operations = list(iter_operations())

    if category:
        operations = [op for op in operations if op.category == category]

    if json_output:
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


@op_app.command("call")
def op_call(
    op_id: Annotated[str, typer.Argument(help="Operation ID to invoke")],
    params: Annotated[
        Optional[list[str]],
        typer.Argument(help="Operation parameters as key=value pairs"),
    ] = None,
    project_root: Annotated[
        Optional[Path],
        typer.Option("--root", "-r", help="Explicit project root directory"),
    ] = None,
    skip_prereqs: Annotated[
        bool,
        typer.Option("--skip-prereqs", help="Skip prerequisite pipeline execution"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """Invoke a serving operation end-to-end.

    Parameters are provided as key=value pairs after the operation ID.

    Example:
        codeintel-app op call function.summary goid_h128=123456
    """
    if verbose:
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
        # Try to parse as int, float, bool, or keep as string
        parsed_value: str | int | float | bool = value
        if value.lower() in ("true", "false"):
            parsed_value = value.lower() == "true"
        else:
            try:
                parsed_value = int(value)
            except ValueError:
                try:
                    parsed_value = float(value)
                except ValueError:
                    pass
        kwargs[key] = parsed_value

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

    # Build service stack
    from codeintel.serving.bootstrap import build_service_stack

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
    project_root: Annotated[
        Optional[Path],
        typer.Option("--root", "-r", help="Explicit project root directory"),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
) -> None:
    """List datasets from the registry."""
    runtime = _build_runtime_or_exit(project_root)

    datasets = runtime.gateway.datasets.list_datasets()

    if json_output:
        output = [
            {
                "table_key": ds.table_key,
                "schema_name": ds.schema_name,
                "table_name": ds.table_name,
                "row_count": ds.row_count,
            }
            for ds in datasets
        ]
        typer.echo(json.dumps(output, indent=2))
    else:
        typer.echo(f"Datasets ({len(datasets)}):")
        for ds in datasets:
            typer.echo(f"  {ds.table_key:<40} rows={ds.row_count}")


@dataset_app.command("describe")
def dataset_describe(
    table_key: Annotated[str, typer.Argument(help="Dataset table key (e.g., 'core.goids')")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
) -> None:
    """Show contract details for a dataset."""
    contracts = get_dataset_contracts_by_table_key()
    contract = contracts.get(table_key)

    if contract is None:
        typer.secho(f"Error: Dataset not found: {table_key}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if json_output:
        output = {
            "name": contract.name,
            "table_key": contract.table_key,
            "description": contract.description,
            "owner_package": contract.owner_package,
            "columns": [
                {"name": col.name, "dtype": col.dtype, "nullable": col.nullable}
                for col in (contract.columns or [])
            ],
            "upstream_dependencies": list(contract.upstream_dependencies or []),
        }
        typer.echo(json.dumps(output, indent=2))
    else:
        typer.echo(f"Dataset: {contract.name}")
        typer.echo(f"  Table: {contract.table_key}")
        typer.echo(f"  Owner: {contract.owner_package}")
        typer.echo(f"  Description: {contract.description}")
        if contract.columns:
            typer.echo("  Columns:")
            for col in contract.columns:
                nullable = "nullable" if col.nullable else "required"
                typer.echo(f"    - {col.name}: {col.dtype} ({nullable})")
        if contract.upstream_dependencies:
            typer.echo(f"  Dependencies: {list(contract.upstream_dependencies)}")


@dataset_app.command("verify")
def dataset_verify(
    table_key: Annotated[
        Optional[str],
        typer.Argument(help="Dataset table key to verify (verifies all if not specified)"),
    ] = None,
    project_root: Annotated[
        Optional[Path],
        typer.Option("--root", "-r", help="Explicit project root directory"),
    ] = None,
) -> None:
    """Verify dataset contracts against actual data."""
    runtime = _build_runtime_or_exit(project_root)

    from codeintel.config.datasets import collect_contract_issues

    issues = collect_contract_issues(runtime.gateway.con)

    if table_key:
        issues = [i for i in issues if i.table_key == table_key]

    if not issues:
        typer.secho("All dataset contracts verified successfully.", fg=typer.colors.GREEN)
    else:
        typer.secho(f"Found {len(issues)} contract issues:", fg=typer.colors.RED)
        for issue in issues:
            typer.echo(f"  {issue.table_key}: {issue.issue_type} - {issue.message}")
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


@serve_app.command("http")
def serve_http(
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Host to bind to"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port to bind to"),
    ] = 8000,
    auto_pipeline: Annotated[
        bool,
        typer.Option("--auto-pipeline", help="Enable automatic prerequisite pipeline execution"),
    ] = False,
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Enable auto-reload for development"),
    ] = False,
    project_root: Annotated[
        Optional[Path],
        typer.Option("--root", "-r", help="Explicit project root directory"),
    ] = None,
) -> None:
    """Start the HTTP server."""
    runtime = _build_runtime_or_exit(project_root)

    # Set environment variables for the serving layer
    os.environ["CODEINTEL_REPO"] = runtime.project.repo
    os.environ["CODEINTEL_COMMIT"] = runtime.snapshot.commit
    os.environ["CODEINTEL_DB_PATH"] = str(runtime.paths.db_path)
    os.environ["CODEINTEL_REPO_ROOT"] = str(runtime.root)

    if auto_pipeline:
        os.environ[AUTO_PIPELINE_ENV] = "1"

    typer.echo(f"Starting HTTP server at http://{host}:{port}")
    typer.echo(f"  Repo: {runtime.project.repo}")
    typer.echo(f"  Commit: {runtime.snapshot.commit}")
    typer.echo(f"  Database: {runtime.paths.db_path}")
    typer.echo(f"  Auto-pipeline: {'enabled' if auto_pipeline else 'disabled'}")

    import uvicorn

    uvicorn.run(
        "codeintel.serving.http.fastapi:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


@serve_app.command("mcp")
def serve_mcp(
    auto_pipeline: Annotated[
        bool,
        typer.Option("--auto-pipeline", help="Enable automatic prerequisite pipeline execution"),
    ] = False,
    project_root: Annotated[
        Optional[Path],
        typer.Option("--root", "-r", help="Explicit project root directory"),
    ] = None,
) -> None:
    """Start the MCP server."""
    runtime = _build_runtime_or_exit(project_root)

    # Set environment variables for the serving layer
    os.environ["CODEINTEL_REPO"] = runtime.project.repo
    os.environ["CODEINTEL_COMMIT"] = runtime.snapshot.commit
    os.environ["CODEINTEL_DB_PATH"] = str(runtime.paths.db_path)
    os.environ["CODEINTEL_REPO_ROOT"] = str(runtime.root)

    if auto_pipeline:
        os.environ[AUTO_PIPELINE_ENV] = "1"

    typer.echo("Starting MCP server...", err=True)
    typer.echo(f"  Repo: {runtime.project.repo}", err=True)
    typer.echo(f"  Commit: {runtime.snapshot.commit}", err=True)
    typer.echo(f"  Database: {runtime.paths.db_path}", err=True)
    typer.echo(f"  Auto-pipeline: {'enabled' if auto_pipeline else 'disabled'}", err=True)

    from codeintel.serving.mcp.server import main as run_mcp_server

    sys.exit(run_mcp_server())


__all__ = [
    "dataset_app",
    "op_app",
    "pipeline_app",
    "serve_app",
]

