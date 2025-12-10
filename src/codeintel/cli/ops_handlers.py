"""Typer-free handlers for legacy op, dataset, and serve commands.

These helpers keep the operational logic formerly hosted in ``cli.main`` while
allowing Cyclopts to invoke them without importing Typer. All user-facing
errors surface as :class:`~codeintel.cli.cli_errors.ValidationError` so the
CLI runner can normalize exit codes and stderr output consistently.

.. deprecated:: 2.0
    This module is deprecated. Use codeintel.cli.handlers.ops instead.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import uvicorn

from codeintel.cli.cli_errors import ValidationError
from codeintel.cli.project import (
    ProjectNotFoundError,
    ProjectRuntime,
    build_project_runtime,
)
from codeintel.cli.result_types import (
    DatasetDescribeResult,
    DatasetListResult,
    DatasetVerifyResult,
    OperationListResult,
)
from codeintel.cli.results import CliResult
from codeintel.config.datasets import get_dataset_contracts_by_table_key
from codeintel.serving.auto_pipeline import run_operation_prereqs
from codeintel.serving.bootstrap import build_service_stack
from codeintel.serving.http.fastapi import create_app as create_http_app
from codeintel.serving.mcp.server import main as run_mcp_server
from codeintel.serving.operations.catalog import (
    get_operation,
    iter_operations,
)
from codeintel.storage.validation import collect_contract_issues

if TYPE_CHECKING:
    from codeintel.cli.execution.context import ExecutionContext
    from codeintel.cli.resolution.types import ResolvedRuntime

warnings.warn(
    "codeintel.cli.ops_handlers is deprecated. Use codeintel.cli.handlers.ops instead.",
    DeprecationWarning,
    stacklevel=2,
)

LOG = logging.getLogger(__name__)


def _build_runtime_or_error(project_root: Path | None) -> ProjectRuntime:
    """Resolve a :class:`ProjectRuntime` or raise a validation error.

    Returns
    -------
    ProjectRuntime
        Resolved runtime for the provided project root.

    Raises
    ------
    ValidationError
        When no project configuration can be found.
    """
    try:
        return build_project_runtime(project_root)
    except ProjectNotFoundError as exc:
        raise ValidationError(str(exc)) from exc


def _parse_param_value(value: str) -> str | int | float | bool:
    """Parse CLI parameter text into a primitive Python type.

    Returns
    -------
    str | int | float | bool
        Parsed value coerced to bool/int/float when possible, otherwise the original string.
    """
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    with contextlib.suppress(ValueError):
        return int(value)
    with contextlib.suppress(ValueError):
        return float(value)
    return value


AUTO_PIPELINE_ENV = "CODEINTEL_AUTO_PIPELINE"


def _setup_serving_env(runtime: ProjectRuntime, *, auto_pipeline: bool) -> None:
    """Configure environment variables for serving operations.

    Parameters
    ----------
    runtime
        Project runtime with repo and path information.
    auto_pipeline
        Whether to enable automatic prerequisite pipeline execution.
    """
    os.environ["CODEINTEL_REPO"] = runtime.project.repo
    os.environ["CODEINTEL_COMMIT"] = runtime.snapshot.commit
    os.environ["CODEINTEL_DB_PATH"] = str(runtime.paths.db_path)
    os.environ["CODEINTEL_REPO_ROOT"] = str(runtime.root)

    if auto_pipeline:
        os.environ[AUTO_PIPELINE_ENV] = "1"


def invoke_operation(
    op_id: str,
    kwargs: dict[str, Any],
    runtime: ProjectRuntime,
) -> None:
    """Invoke a serving operation and render the JSON result.

    This is a legacy compatibility function used by dynamic command generation.
    New code should use the ``_ctx`` handlers instead.

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
    ValidationError
        When the operation is unknown or cannot be executed.
    """
    stdout = sys.stdout
    op = get_operation(op_id)
    if op is None:
        error = f"Unknown operation: {op_id}"
        raise ValidationError(error)

    stdout.write(f"Invoking operation '{op_id}'...\n")

    stack = build_service_stack(runtime.serving, gateway=runtime.gateway)
    try:
        method = getattr(stack.service, op.backend_method, None)
        if method is None:
            error = f"Backend method not found: {op.backend_method}"
            raise ValidationError(error)

        result = method(**kwargs)

        if hasattr(result, "model_dump"):
            output = result.model_dump(mode="json")
        elif hasattr(result, "__dict__"):
            output = result.__dict__
        else:
            output = result

        stdout.write(json.dumps(output, indent=2, default=str))
        stdout.write("\n")
    finally:
        stack.close()


# -----------------------------------------------------------------------------
# Structured Handlers (CliResult pattern)
# -----------------------------------------------------------------------------


def op_list_structured(
    *,
    category: str | None,
) -> CliResult[OperationListResult]:
    """List available serving operations (structured).

    Parameters
    ----------
    category
        Optional category filter.

    Returns
    -------
    CliResult[OperationListResult]
        List of operations matching the filter.
    """
    operations = list(iter_operations())
    if category:
        operations = [op for op in operations if op.category == category]

    operation_dicts = [
        {
            "id": op.id,
            "category": op.category,
            "summary": op.summary,
            "http_path": op.http_path,
            "tool_name": op.tool_name,
        }
        for op in sorted(operations, key=lambda o: o.id)
    ]

    return CliResult.ok(OperationListResult(operations=operation_dicts, count=len(operations)))


def op_list_ctx(
    ctx: ExecutionContext,
) -> CliResult[OperationListResult]:
    """List available serving operations (ExecutionContext pattern).

    Parameters
    ----------
    ctx
        Execution context with params:
        - category: Optional category filter

    Returns
    -------
    CliResult[OperationListResult]
        List of operations matching the filter.
    """
    category = ctx.get_str_param("category")
    return op_list_structured(category=category)


def dataset_list_structured(
    *,
    runtime: ProjectRuntime,
) -> CliResult[DatasetListResult]:
    """List datasets from the registry (structured).

    Parameters
    ----------
    runtime
        Project runtime context.

    Returns
    -------
    CliResult[DatasetListResult]
        List of datasets.
    """
    registry = runtime.gateway.datasets
    meta = registry.meta or {}

    dataset_dicts = [
        {
            "name": name,
            "table_key": contract.table_key,
            "is_view": str(contract.is_view),
            "owner_package": contract.owner_package,
        }
        for name, contract in sorted(meta.items())
    ]

    return CliResult.ok(DatasetListResult(datasets=dataset_dicts, count=len(meta)))


def dataset_describe_structured(
    *,
    table_key: str,
) -> CliResult[DatasetDescribeResult]:
    """Show contract details for a dataset (structured).

    Parameters
    ----------
    table_key
        Dataset table key.

    Returns
    -------
    CliResult[DatasetDescribeResult]
        Dataset details.

    Raises
    ------
    ValidationError
        When the dataset is not found.
    """
    contracts = get_dataset_contracts_by_table_key()
    contract = contracts.get(table_key)
    if contract is None:
        error = f"Dataset not found: {table_key}"
        raise ValidationError(error)

    columns = contract.schema.columns if contract.schema else []

    column_dicts: list[dict[str, str | bool]] = [
        {"name": col.name, "type": col.type, "nullable": col.nullable} for col in columns
    ]

    return CliResult.ok(
        DatasetDescribeResult(
            table_key=contract.table_key,
            name=contract.name,
            description=contract.description,
            owner_package=contract.owner_package,
            columns=column_dicts,
            row_count=None,
            upstream_dependencies=list(contract.upstream_dependencies),
        )
    )


def dataset_verify_structured(
    *,
    table_key: str | None,
    runtime: ProjectRuntime,
) -> CliResult[DatasetVerifyResult]:
    """Verify dataset contracts against actual data (structured).

    Parameters
    ----------
    table_key
        Optional dataset table key filter.
    runtime
        Project runtime context.

    Returns
    -------
    CliResult[DatasetVerifyResult]
        Verification result.
    """
    issues = collect_contract_issues(runtime.gateway.con)

    if table_key:
        issues = [issue for issue in issues if table_key in issue]

    return CliResult.ok(DatasetVerifyResult(verified=len(issues) == 0, issues=issues))


# -----------------------------------------------------------------------------
# ExecutionContext Handlers (_ctx pattern)
# -----------------------------------------------------------------------------


def op_call_ctx(ctx: ExecutionContext) -> CliResult[dict[str, Any]]:
    """Invoke an operation end-to-end with optional prerequisites.

    Parameters
    ----------
    ctx
        Execution context with params:
        - op_id: Operation ID to invoke
        - params: List of key=value parameter strings
        - skip_prereqs: Skip prerequisite execution

    Returns
    -------
    CliResult[dict[str, Any]]
        Operation result.

    Raises
    ------
    RuntimeError
        When operation ID is missing or invalid.
    """
    op_id = ctx.require_str_param("op_id")
    params_raw = ctx.params.get("params") or []
    skip_prereqs = ctx.get_bool_param("skip_prereqs", default=False)

    op = get_operation(op_id)
    if op is None:
        msg = f"Unknown operation: {op_id}"
        raise RuntimeError(msg)

    kwargs: dict[str, str | int | float | bool] = {}
    for param in params_raw:
        if "=" not in param:
            msg = f"Invalid parameter format: {param} (expected key=value)"
            raise RuntimeError(msg)
        key, value = param.split("=", 1)
        kwargs[key] = _parse_param_value(value)

    runtime = ctx.require_runtime()
    # Build ProjectRuntime from ResolvedRuntime for compatibility
    project_runtime = _resolved_to_project_runtime(runtime)

    if not skip_prereqs:
        run_operation_prereqs(
            op_id=op_id,
            gateway=project_runtime.gateway,
            snapshot=project_runtime.snapshot,
            paths=project_runtime.paths,
            tools=project_runtime.tools,
        )

    result = _invoke_operation_structured(op_id, kwargs, project_runtime)
    return CliResult.ok(result)


def _resolved_to_project_runtime(runtime: ResolvedRuntime) -> ProjectRuntime:
    """Convert ResolvedRuntime to ProjectRuntime for backward compatibility.

    Parameters
    ----------
    runtime
        ResolvedRuntime from ExecutionContext.

    Returns
    -------
    ProjectRuntime
        Compatible ProjectRuntime instance.
    """
    from codeintel.storage.gateway import StorageConfig, open_gateway  # noqa: PLC0415

    gateway = open_gateway(StorageConfig.for_readonly(runtime.db_path))
    return ProjectRuntime(
        root=runtime.root,
        project=runtime.project,
        cfg=runtime.config,
        snapshot=runtime.snapshot,
        paths=runtime.paths,
        gateway=gateway,
        tools=runtime.config.tools,
        serving=runtime.serving,
    )


def _invoke_operation_structured(
    op_id: str,
    kwargs: dict[str, Any],
    runtime: ProjectRuntime,
) -> dict[str, Any]:
    """Invoke operation and return structured result.

    Parameters
    ----------
    op_id
        Operation ID.
    kwargs
        Operation parameters.
    runtime
        Project runtime.

    Returns
    -------
    dict[str, Any]
        Operation result as dictionary.

    Raises
    ------
    RuntimeError
        If the operation or backend method is not found.
    """
    op = get_operation(op_id)
    if op is None:
        msg = f"Unknown operation: {op_id}"
        raise RuntimeError(msg)

    stack = build_service_stack(
        config=runtime.serving,
        gateway=runtime.gateway,
    )

    try:
        method = getattr(stack.service, op.backend_method, None)
        if method is None:
            msg = f"Backend method not found: {op.backend_method}"
            raise RuntimeError(msg)

        result = method(**kwargs)

        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        if hasattr(result, "__dict__"):
            return result.__dict__
        return {"result": result}
    finally:
        stack.close()


def dataset_list_ctx(ctx: ExecutionContext) -> CliResult[DatasetListResult]:
    """List datasets from the registry.

    Parameters
    ----------
    ctx
        Execution context.

    Returns
    -------
    CliResult[DatasetListResult]
        List of datasets.
    """
    runtime = ctx.require_runtime()
    project_runtime = _resolved_to_project_runtime(runtime)
    return dataset_list_structured(runtime=project_runtime)


def dataset_describe_ctx(ctx: ExecutionContext) -> CliResult[DatasetDescribeResult]:
    """Show contract details for a dataset.

    Parameters
    ----------
    ctx
        Execution context with params:
        - table_key: Dataset table key

    Returns
    -------
    CliResult[DatasetDescribeResult]
        Dataset details.
    """
    table_key = ctx.require_str_param("table_key")
    return dataset_describe_structured(table_key=table_key)


def dataset_verify_ctx(ctx: ExecutionContext) -> CliResult[DatasetVerifyResult]:
    """Verify dataset contracts against actual data.

    Parameters
    ----------
    ctx
        Execution context with params:
        - table_key: Optional dataset table key filter

    Returns
    -------
    CliResult[DatasetVerifyResult]
        Verification result.
    """
    table_key = ctx.get_str_param("table_key")
    runtime = ctx.require_runtime()
    project_runtime = _resolved_to_project_runtime(runtime)
    return dataset_verify_structured(table_key=table_key, runtime=project_runtime)


@dataclass(frozen=True)
class ServeStartResult:
    """Result of starting a server."""

    server_type: str
    host: str | None
    port: int | None
    auto_pipeline: bool
    repo: str
    commit: str
    db_path: str


def serve_http_ctx(ctx: ExecutionContext) -> CliResult[ServeStartResult]:
    """Start the HTTP server.

    Parameters
    ----------
    ctx
        Execution context with params:
        - host: Server host (default: 127.0.0.1)
        - port: Server port (default: 8000)
        - auto_pipeline: Enable auto-pipeline
        - reload: Enable hot reload

    Returns
    -------
    CliResult[ServeStartResult]
        Server start result.

    Notes
    -----
    This function blocks while the server is running.
    """
    host = ctx.get_str_param("host", "127.0.0.1") or "127.0.0.1"
    port = ctx.get_int_param("port", 8000)
    auto_pipeline = ctx.get_bool_param("auto_pipeline", default=False)
    reload = ctx.get_bool_param("reload", default=False)

    runtime = ctx.require_runtime()
    project_runtime = _resolved_to_project_runtime(runtime)

    _setup_serving_env(project_runtime, auto_pipeline=auto_pipeline)

    LOG.info(
        "Starting HTTP server at http://%s:%d (auto_pipeline=%s)",
        host,
        port,
        auto_pipeline,
    )

    if reload:
        uvicorn.run(
            "codeintel.serving.http.fastapi:app",
            host=host,
            port=port,
            reload=True,
        )
    else:
        app = create_http_app(
            gateway=project_runtime.gateway,
            auto_pipeline=auto_pipeline,
        )
        uvicorn.run(app, host=host, port=port)

    return CliResult.ok(
        ServeStartResult(
            server_type="http",
            host=host,
            port=port,
            auto_pipeline=auto_pipeline,
            repo=runtime.repo,
            commit=runtime.commit,
            db_path=str(runtime.db_path),
        )
    )


def serve_mcp_ctx(ctx: ExecutionContext) -> CliResult[ServeStartResult]:
    """Start the MCP server.

    Parameters
    ----------
    ctx
        Execution context with params:
        - auto_pipeline: Enable auto-pipeline

    Notes
    -----
    This function blocks while the server is running and exits the process
    via sys.exit(), so it never returns normally.
    """
    auto_pipeline = ctx.get_bool_param("auto_pipeline", default=False)

    runtime = ctx.require_runtime()
    project_runtime = _resolved_to_project_runtime(runtime)

    _setup_serving_env(project_runtime, auto_pipeline=auto_pipeline)

    LOG.info("Starting MCP server (auto_pipeline=%s)", auto_pipeline)

    # MCP server runs and exits
    sys.exit(run_mcp_server())


__all__ = [
    "AUTO_PIPELINE_ENV",
    "ServeStartResult",
    "dataset_describe_ctx",
    "dataset_describe_structured",
    "dataset_list_ctx",
    "dataset_list_structured",
    "dataset_verify_ctx",
    "dataset_verify_structured",
    "invoke_operation",
    "op_call_ctx",
    "op_list_ctx",
    "op_list_structured",
    "serve_http_ctx",
    "serve_mcp_ctx",
]
