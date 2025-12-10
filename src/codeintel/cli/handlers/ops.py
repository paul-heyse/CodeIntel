"""Operation handlers.

Handlers for operation listing, invocation, dataset management, and server control.
"""

from __future__ import annotations

import logging
import os
import sys

import uvicorn

from codeintel.cli.core import CliResult, parse_cli_value
from codeintel.cli.core.result_types import (
    DatasetDescribeResult,
    DatasetListResult,
    DatasetVerifyResult,
    OperationCallResult,
    OperationListResult,
    ServeStartResult,
)
from codeintel.cli.errors import ProblemDetail
from codeintel.cli.handlers._utilities import runtime_gateway
from codeintel.cli.handlers.context import HandlerContext
from codeintel.cli.resolution.types import ResolvedRuntime
from codeintel.config.datasets import get_dataset_contracts_by_table_key
from codeintel.serving.http.fastapi import create_app as create_http_app
from codeintel.serving.mcp.server import main as run_mcp_server
from codeintel.serving.operations.catalog import get_operation, iter_operations
from codeintel.storage.validation import collect_contract_issues

LOG = logging.getLogger(__name__)

AUTO_PIPELINE_ENV = "CODEINTEL_AUTO_PIPELINE"


def _setup_serving_env(runtime: ResolvedRuntime, *, auto_pipeline: bool) -> None:
    """Configure environment variables for serving operations.

    Parameters
    ----------
    runtime
        Resolved runtime with repo and path information.
    auto_pipeline
        Whether to enable automatic prerequisite pipeline execution.
    """
    os.environ["CODEINTEL_REPO"] = runtime.project.repo
    os.environ["CODEINTEL_COMMIT"] = runtime.snapshot.commit
    os.environ["CODEINTEL_DB_PATH"] = str(runtime.paths.db_path)
    os.environ["CODEINTEL_REPO_ROOT"] = str(runtime.root)

    if auto_pipeline:
        os.environ[AUTO_PIPELINE_ENV] = "1"


def op_list_structured(*, category: str | None) -> CliResult[OperationListResult]:
    """List available serving operations (structured, no context needed).

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

    operation_dicts: list[dict[str, str | None]] = [
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


def op_list_handler(ctx: HandlerContext) -> CliResult[OperationListResult]:
    """List available serving operations.

    Parameters
    ----------
    ctx
        Handler context with params:
        - category: Optional category filter

    Returns
    -------
    CliResult[OperationListResult]
        List of operations matching the filter.
    """
    category = ctx.param_str("category")
    return op_list_structured(category=category)


def op_call_handler(ctx: HandlerContext) -> CliResult[OperationCallResult]:
    """Invoke an operation end-to-end with optional prerequisites.

    Parameters
    ----------
    ctx
        Handler context with params:
        - op_id: Operation ID to invoke
        - params: List of key=value parameter strings
        - skip_prereqs: Skip prerequisite execution

    Returns
    -------
    CliResult[OperationCallResult]
        Operation result.
    """
    op_id = ctx.require_str("op_id")
    params_list = ctx.param_list("params")
    skip_prereqs = ctx.param_bool("skip_prereqs", default=False)

    # Validate operation exists first
    op = get_operation(op_id)
    if op is None:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:ops:unknown-operation",
                title="Unknown Operation",
                detail=f"Unknown operation: {op_id}",
                status=404,
            )
        )

    # Parse params list into kwargs
    kwargs: dict[str, object] = {}
    for param_str in params_list:
        if "=" not in param_str:
            return CliResult.fail(
                ProblemDetail(
                    type="urn:codeintel:ops:invalid-param",
                    title="Invalid Parameter Format",
                    detail=f"Invalid parameter format: {param_str} (expected key=value)",
                    status=400,
                )
            )
        key, value = param_str.split("=", 1)
        kwargs[key] = parse_cli_value(value)

    # Use unified serving operation invocation
    result = ctx.invoke_serving_operation(op_id, kwargs, skip_prereqs=skip_prereqs)
    return CliResult.ok(OperationCallResult(operation_id=op_id, result=result))


def dataset_describe_structured(*, table_key: str) -> CliResult[DatasetDescribeResult]:
    """Show contract details for a dataset (structured, no context needed).

    Parameters
    ----------
    table_key
        Dataset table key.

    Returns
    -------
    CliResult[DatasetDescribeResult]
        Dataset details.
    """
    contracts = get_dataset_contracts_by_table_key()
    contract = contracts.get(table_key)
    if contract is None:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:ops:dataset-not-found",
                title="Dataset Not Found",
                detail=f"Dataset not found: {table_key}",
                status=404,
            )
        )

    columns = contract.schema.columns if contract.schema else []

    column_dicts: list[dict[str, str | bool]] = [
        {"name": col.name, "type": col.type, "nullable": col.nullable} for col in columns
    ]

    return CliResult.ok(
        DatasetDescribeResult(
            table_key=contract.table_key,
            columns=column_dicts,
            row_count=None,
            name=contract.name,
            description=contract.description,
            owner_package=contract.owner_package,
            upstream_dependencies=list(contract.upstream_dependencies),
        )
    )


def dataset_list_handler(ctx: HandlerContext) -> CliResult[DatasetListResult]:
    """List datasets from the registry.

    Parameters
    ----------
    ctx
        Handler context.

    Returns
    -------
    CliResult[DatasetListResult]
        List of datasets.
    """
    gateway = ctx.gateway
    registry = gateway.datasets
    meta = registry.meta or {}

    dataset_dicts: list[dict[str, str | None]] = [
        {
            "name": name,
            "table_key": contract.table_key,
            "is_view": str(contract.is_view),
            "owner_package": contract.owner_package,
        }
        for name, contract in sorted(meta.items())
    ]

    return CliResult.ok(DatasetListResult(datasets=dataset_dicts, count=len(meta)))


def dataset_describe_handler(
    ctx: HandlerContext,
) -> CliResult[DatasetDescribeResult]:
    """Show contract details for a dataset.

    Parameters
    ----------
    ctx
        Handler context with params:
        - table_key: Dataset table key

    Returns
    -------
    CliResult[DatasetDescribeResult]
        Dataset details.
    """
    table_key = ctx.require_str("table_key")
    return dataset_describe_structured(table_key=table_key)


def dataset_verify_handler(
    ctx: HandlerContext,
) -> CliResult[DatasetVerifyResult]:
    """Verify dataset contracts against actual data.

    Parameters
    ----------
    ctx
        Handler context with params:
        - table_key: Optional dataset table key filter

    Returns
    -------
    CliResult[DatasetVerifyResult]
        Verification result.
    """
    table_key = ctx.param_str("table_key")
    gateway = ctx.gateway
    issues = collect_contract_issues(gateway.con)

    if table_key:
        issues = [issue for issue in issues if table_key in issue]

    return CliResult.ok(DatasetVerifyResult(verified=len(issues) == 0, issues=issues))


def serve_http_handler(ctx: HandlerContext) -> CliResult[ServeStartResult]:
    """Start the HTTP server.

    Parameters
    ----------
    ctx
        Handler context with params:
        - host: Server host (default: 127.0.0.1)
        - port: Server port (default: 8000)
        - auto_pipeline: Enable auto-pipeline
        - reload: Enable hot reload

    Returns
    -------
    CliResult[ServeStartResult]
        Server start result (after server stops).

    Notes
    -----
    This function blocks while the server is running.
    """
    host = ctx.param_str("host", "127.0.0.1") or "127.0.0.1"
    port = ctx.param_int("port", 8000)
    auto_pipeline = ctx.param_bool("auto_pipeline", default=False)
    reload = ctx.param_bool("reload", default=False)

    runtime = ctx.runtime

    _setup_serving_env(runtime, auto_pipeline=auto_pipeline)

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
        with runtime_gateway(runtime, read_only=False) as gateway:
            app = create_http_app(
                gateway=gateway,
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
            db_path=str(runtime.paths.db_path),
        )
    )


def serve_mcp_handler(ctx: HandlerContext) -> CliResult[ServeStartResult]:
    """Start the MCP server.

    Parameters
    ----------
    ctx
        Handler context with params:
        - auto_pipeline: Enable auto-pipeline

    Notes
    -----
    This function blocks while the server is running and exits the process
    via sys.exit(), so it never returns normally.
    """
    auto_pipeline = ctx.param_bool("auto_pipeline", default=False)

    _setup_serving_env(ctx.runtime, auto_pipeline=auto_pipeline)

    LOG.info("Starting MCP server (auto_pipeline=%s)", auto_pipeline)

    # MCP server runs and exits
    sys.exit(run_mcp_server())


__all__ = [
    "AUTO_PIPELINE_ENV",
    "DatasetDescribeResult",
    "DatasetListResult",
    "DatasetVerifyResult",
    "OperationCallResult",
    "OperationListResult",
    "ServeStartResult",
    "dataset_describe_handler",
    "dataset_describe_structured",
    "dataset_list_handler",
    "dataset_verify_handler",
    "op_call_handler",
    "op_list_handler",
    "op_list_structured",
    "serve_http_handler",
    "serve_mcp_handler",
]
