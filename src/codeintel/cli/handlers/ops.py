"""Operation handlers.

Handlers for operation listing, invocation, dataset management, and server control.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

import uvicorn

from codeintel.cli.core import CliResult, parse_cli_value
from codeintel.cli.core.result_types import (
    DatasetConstraintsResult,
    DatasetDescribeResult,
    DatasetFlowResult,
    DatasetInfoResult,
    DatasetListResult,
    DatasetVerifyResult,
    OperationCallResult,
    OperationListResult,
    ServeStartResult,
)
from codeintel.cli.errors.results import (
    fail_dataset_not_found,
    fail_invalid_param,
    fail_unknown_operation,
)
from codeintel.cli.handlers._utilities import runtime_gateway
from codeintel.config.datasets import get_dataset_contracts_by_table_key
from codeintel.config.datasets.constraints import extract_constraints_from_pandera
from codeintel.config.datasets.schema import DatasetMetadata
from codeintel.config.datasets.schema_registry import (
    SCHEMA_REGISTRY,
    DatasetSchemaRegistry,
)
from codeintel.serving.http.fastapi import create_app as create_http_app
from codeintel.serving.mcp.server import main as run_mcp_server
from codeintel.serving.operations.catalog import get_operation, iter_operations
from codeintel.storage.validation import collect_contract_issues

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext
    from codeintel.cli.resolution.types import ResolvedRuntime

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


def op_list_handler(ctx: CommandContext) -> CliResult[OperationListResult]:
    """List available serving operations.

    Parameters
    ----------
    ctx
        Command context with params:
        - category: Optional category filter

    Returns
    -------
    CliResult[OperationListResult]
        List of operations matching the filter.
    """
    category = ctx.params.get_str("category")
    return op_list_structured(category=category)


def op_call_handler(ctx: CommandContext) -> CliResult[OperationCallResult]:
    """Invoke an operation end-to-end with optional prerequisites.

    Parameters
    ----------
    ctx
        Command context with params:
        - op_id: Operation ID to invoke
        - params: List of key=value parameter strings
        - skip_prereqs: Skip prerequisite execution

    Returns
    -------
    CliResult[OperationCallResult]
        Operation result.
    """
    op_id = ctx.params.require_str("op_id")
    params_list = ctx.params.get_list("params")
    skip_prereqs = ctx.params.get_bool("skip_prereqs", default=False)

    # Validate operation exists first
    op = get_operation(op_id)
    if op is None:
        return fail_unknown_operation(op_id)

    # Parse params list into kwargs
    kwargs: dict[str, object] = {}
    for param_str in params_list:
        if "=" not in param_str:
            return fail_invalid_param(param_str)
        key, value = param_str.split("=", 1)
        kwargs[key] = parse_cli_value(value)

    # Use unified serving operation invocation via serving service
    result = ctx.serving.invoke(op_id, kwargs, skip_prereqs=skip_prereqs)
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
        return fail_dataset_not_found(table_key)

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


def dataset_list_handler(ctx: CommandContext) -> CliResult[DatasetListResult]:
    """List datasets from the registry.

    Parameters
    ----------
    ctx
        Command context.

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
    ctx: CommandContext,
) -> CliResult[DatasetDescribeResult]:
    """Show contract details for a dataset.

    Parameters
    ----------
    ctx
        Command context with params:
        - table_key: Dataset table key

    Returns
    -------
    CliResult[DatasetDescribeResult]
        Dataset details.
    """
    table_key = ctx.params.require_str("table_key")
    return dataset_describe_structured(table_key=table_key)


def dataset_verify_handler(
    ctx: CommandContext,
) -> CliResult[DatasetVerifyResult]:
    """Verify dataset contracts against actual data.

    Parameters
    ----------
    ctx
        Command context with params:
        - table_key: Optional dataset table key filter

    Returns
    -------
    CliResult[DatasetVerifyResult]
        Verification result.
    """
    table_key = ctx.params.get_str("table_key")
    gateway = ctx.gateway
    issues = collect_contract_issues(gateway.con)

    if table_key:
        issues = [issue for issue in issues if table_key in issue]

    return CliResult.ok(DatasetVerifyResult(verified=len(issues) == 0, issues=issues))


def _metadata_to_dict(metadata: object) -> dict[str, object]:
    """Convert DatasetMetadata to a dictionary.

    Parameters
    ----------
    metadata
        DatasetMetadata instance.

    Returns
    -------
    dict[str, object]
        Dictionary with non-empty metadata fields.
    """
    if not isinstance(metadata, DatasetMetadata):
        return {}

    field_map: dict[str, object] = {
        "description": metadata.description,
        "owner": metadata.owner,
        "family": metadata.family,
        "freshness_sla": metadata.freshness_sla,
        "retention_policy": metadata.retention_policy,
        "deprecated": metadata.deprecated if metadata.deprecated else None,
        "deprecation_message": metadata.deprecation_message,
    }
    result = {k: v for k, v in field_map.items() if v}

    # Handle tuple/frozenset fields that need list conversion
    if metadata.upstream_dependencies:
        result["upstream_dependencies"] = list(metadata.upstream_dependencies)
    if metadata.downstream_consumers:
        result["downstream_consumers"] = list(metadata.downstream_consumers)
    if metadata.tags:
        result["tags"] = list(metadata.tags)

    return result


def dataset_info_structured(*, table_key: str) -> CliResult[DatasetInfoResult]:
    """Show comprehensive schema information for a dataset (structured).

    Parameters
    ----------
    table_key
        Dataset table key.

    Returns
    -------
    CliResult[DatasetInfoResult]
        Schema information including columns, metadata, and JSON schema.
    """
    schema = SCHEMA_REGISTRY.get(table_key)
    if schema is None:
        return fail_dataset_not_found(table_key)

    return CliResult.ok(
        DatasetInfoResult(
            name=schema.name,
            columns=schema.column_names(),
            metadata=_metadata_to_dict(schema.metadata),
            json_schema=schema.json_schema(),
            has_pandera_schema=True,
        )
    )


def dataset_info_handler(ctx: CommandContext) -> CliResult[DatasetInfoResult]:
    """Show comprehensive schema information for a dataset.

    Parameters
    ----------
    ctx
        Command context with params:
        - table_key: Dataset table key

    Returns
    -------
    CliResult[DatasetInfoResult]
        Schema information.
    """
    table_key = ctx.params.require_str("table_key")
    return dataset_info_structured(table_key=table_key)


def dataset_flow_structured(*, table_key: str) -> CliResult[DatasetFlowResult]:
    """Show producer/consumer graph for a dataset (structured).

    Parameters
    ----------
    table_key
        Dataset table key.

    Returns
    -------
    CliResult[DatasetFlowResult]
        Flow result with producers and consumers.
    """
    # Verify the dataset exists
    schema = SCHEMA_REGISTRY.get(table_key)
    if schema is None:
        return fail_dataset_not_found(table_key)

    producers = DatasetSchemaRegistry.producers_of(table_key)
    consumers = DatasetSchemaRegistry.consumers_of(table_key)

    return CliResult.ok(
        DatasetFlowResult(
            table_key=table_key,
            producers=producers,
            consumers=consumers,
        )
    )


def dataset_flow_handler(ctx: CommandContext) -> CliResult[DatasetFlowResult]:
    """Show producer/consumer graph for a dataset.

    Parameters
    ----------
    ctx
        Command context with params:
        - table_key: Dataset table key

    Returns
    -------
    CliResult[DatasetFlowResult]
        Flow result with producers and consumers.
    """
    table_key = ctx.params.require_str("table_key")
    return dataset_flow_structured(table_key=table_key)


def dataset_constraints_structured(*, table_key: str) -> CliResult[DatasetConstraintsResult]:
    """Show constraint summary for a dataset (structured).

    Extracts constraints from the Pandera schema and returns them in
    a structured format for programmatic consumption.

    Parameters
    ----------
    table_key
        Dataset table key.

    Returns
    -------
    CliResult[DatasetConstraintsResult]
        Constraint information including kind, column, and expression.
    """
    schema = SCHEMA_REGISTRY.get(table_key)
    if schema is None:
        return fail_dataset_not_found(table_key)

    constraint_set = extract_constraints_from_pandera(table_key, schema.pandera_schema)

    constraints: list[dict[str, object]] = [
        {
            "kind": c.kind.value,
            "column": c.column,
            "expression": c.expression,
            "source": c.source,
        }
        for c in constraint_set.constraints
    ]

    return CliResult.ok(
        DatasetConstraintsResult(
            table_key=table_key,
            constraints=constraints,
            constraint_count=len(constraints),
            inferred_from=list(constraint_set.inferred_from),
        )
    )


def dataset_constraints_handler(ctx: CommandContext) -> CliResult[DatasetConstraintsResult]:
    """Show constraint summary for a dataset.

    Parameters
    ----------
    ctx
        Command context with params:
        - table_key: Dataset table key

    Returns
    -------
    CliResult[DatasetConstraintsResult]
        Constraint information.
    """
    table_key = ctx.params.require_str("table_key")
    return dataset_constraints_structured(table_key=table_key)


def serve_http_handler(ctx: CommandContext) -> CliResult[ServeStartResult]:
    """Start the HTTP server.

    Parameters
    ----------
    ctx
        Command context with params:
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
    host = ctx.params.get_str("host", "127.0.0.1") or "127.0.0.1"
    port = ctx.params.get_int("port", 8000)
    auto_pipeline = ctx.params.get_bool("auto_pipeline", default=False)
    reload = ctx.params.get_bool("reload", default=False)

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


def serve_mcp_handler(ctx: CommandContext) -> CliResult[ServeStartResult]:
    """Start the MCP server.

    Parameters
    ----------
    ctx
        Command context with params:
        - auto_pipeline: Enable auto-pipeline

    Notes
    -----
    This function blocks while the server is running and exits the process
    via sys.exit(), so it never returns normally.
    """
    auto_pipeline = ctx.params.get_bool("auto_pipeline", default=False)

    _setup_serving_env(ctx.runtime, auto_pipeline=auto_pipeline)

    LOG.info("Starting MCP server (auto_pipeline=%s)", auto_pipeline)

    # MCP server runs and exits
    sys.exit(run_mcp_server())


__all__ = [
    "AUTO_PIPELINE_ENV",
    "DatasetDescribeResult",
    "DatasetFlowResult",
    "DatasetInfoResult",
    "DatasetListResult",
    "DatasetVerifyResult",
    "OperationCallResult",
    "OperationListResult",
    "ServeStartResult",
    "dataset_describe_handler",
    "dataset_describe_structured",
    "dataset_flow_handler",
    "dataset_flow_structured",
    "dataset_info_handler",
    "dataset_info_structured",
    "dataset_list_handler",
    "dataset_verify_handler",
    "op_call_handler",
    "op_list_handler",
    "op_list_structured",
    "serve_http_handler",
    "serve_mcp_handler",
]
