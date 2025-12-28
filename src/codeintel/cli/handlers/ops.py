"""Operation handlers.

Handlers for operation listing, invocation, dataset management, and server control.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import uvicorn

from codeintel.build.schemas import (
    ContractResolutionMode,
    ContractResolutionSettings,
    get_contract_for_table_key,
    get_schema_service,
    iter_contracts,
    iter_contracts_by_table_key,
)
from codeintel.build.schemas.constraints import extract_constraints_from_table_schema
from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import (
    DatasetConstraintsResult,
    DatasetDescribeResult,
    DatasetFlowResult,
    DatasetInfoResult,
    DatasetListResult,
    DatasetVerifyResult,
    ServeStartResult,
)
from codeintel.cli.errors.results import fail_dataset_not_found
from codeintel.cli.handlers.runtime_helpers import compose_cli_runtime_bundle
from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.serving.http.app import create_serving_app
from codeintel.serving.mcp.server import create_mcp_server
from codeintel.serving.settings import get_serving_settings
from codeintel.storage.validation import collect_contract_issues

if TYPE_CHECKING:
    from codeintel.build.hamilton.dag_catalog import DagCatalog
    from codeintel.cli.context import CommandContext
    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.runtime.runtime_bundle import RuntimeBundle

LOG = logging.getLogger(__name__)


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
    contracts = dict(
        iter_contracts_by_table_key(
            settings=ContractResolutionSettings(mode=ContractResolutionMode.FULL)
        )
    )
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
    _ = ctx
    contracts = sorted(
        iter_contracts(settings=ContractResolutionSettings(mode=ContractResolutionMode.FULL)),
        key=lambda contract: contract.name,
    )

    dataset_dicts: list[dict[str, str | None]] = [
        {
            "name": contract.name,
            "table_key": contract.table_key,
            "is_view": str(contract.is_view),
            "owner_package": contract.owner_package,
        }
        for contract in contracts
    ]

    return CliResult.ok(DatasetListResult(datasets=dataset_dicts, count=len(dataset_dicts)))


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


def _metadata_to_dict(
    contract: object,
    *,
    downstream_consumers: tuple[str, ...],
) -> dict[str, object]:
    """Convert dataset contract metadata to a dictionary.

    Parameters
    ----------
    contract
        DatasetContract instance.
    downstream_consumers
        Dataset names that consume this dataset.

    Returns
    -------
    dict[str, object]
        Dictionary with non-empty metadata fields.
    """
    if not hasattr(contract, "description"):
        return {}

    field_map: dict[str, object] = {
        "description": getattr(contract, "description", None),
        "owner": getattr(contract, "owner", None),
        "family": getattr(contract, "family", None),
        "freshness_sla": getattr(contract, "freshness_sla", None),
        "retention_policy": getattr(contract, "retention_policy", None),
    }
    result = {k: v for k, v in field_map.items() if v}

    upstream_dependencies = getattr(contract, "upstream_dependencies", ())
    if upstream_dependencies:
        result["upstream_dependencies"] = list(upstream_dependencies)
    if downstream_consumers:
        result["downstream_consumers"] = list(downstream_consumers)
    tags = getattr(contract, "tags", None)
    if tags:
        result["tags"] = sorted(tags)

    return result


def _downstream_consumers_for_contract(
    *,
    contract_name: str,
    contracts: Sequence[DatasetContract],
) -> tuple[str, ...]:
    consumers: list[str] = []
    for candidate in contracts:
        upstream = getattr(candidate, "upstream_dependencies", ())
        if contract_name in upstream:
            name = getattr(candidate, "name", None)
            if isinstance(name, str) and name:
                consumers.append(name)
    return tuple(sorted(set(consumers)))


def _flow_targets_for_table_key(
    table_key: str,
    *,
    catalog: DagCatalog,
) -> tuple[list[str], list[str]]:
    surfaces = catalog.io_surfaces

    producers: set[str] = set()
    consumers: set[str] = set()
    for target_name, surface in surfaces.items():
        for write in surface.table_writes:
            if write.table_key == table_key:
                producers.add(target_name)
        for read in surface.reads:
            if read.table_key == table_key:
                consumers.add(target_name)
    return sorted(producers), sorted(consumers)


def dataset_info_structured(
    *,
    table_key: str,
    runtime: RuntimeBundle,
) -> CliResult[DatasetInfoResult]:
    """Show comprehensive schema information for a dataset (structured).

    Parameters
    ----------
    table_key
        Dataset table key.
    runtime
        Runtime bundle for catalog access.
    runtime
        Runtime bundle for catalog access.
    runtime
        Runtime bundle for catalog access.
    runtime
        Runtime bundle for catalog access.

    Returns
    -------
    CliResult[DatasetInfoResult]
        Schema information including columns, metadata, and JSON schema.
    """
    _ = runtime
    try:
        contract = get_contract_for_table_key(table_key)
    except KeyError:
        return fail_dataset_not_found(table_key)

    schema_service = get_schema_service()
    record = schema_service.get_record(table_key)
    table_schema = record.table_schema
    if table_schema is not None:
        columns = tuple(table_schema.column_names())
    else:
        columns = ()

    contracts = list(
        iter_contracts(settings=ContractResolutionSettings(mode=ContractResolutionMode.FULL))
    )
    downstream = _downstream_consumers_for_contract(
        contract_name=contract.name, contracts=contracts
    )

    return CliResult.ok(
        DatasetInfoResult(
            name=contract.table_key,
            columns=columns,
            metadata=_metadata_to_dict(contract, downstream_consumers=downstream),
            json_schema=record.json_schema or {},
            has_table_schema=table_schema is not None,
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
    runtime_bundle = compose_cli_runtime_bundle(runtime=ctx.runtime, gateway=ctx.gateway)
    return dataset_info_structured(table_key=table_key, runtime=runtime_bundle)


def dataset_flow_structured(
    *,
    table_key: str,
    runtime: RuntimeBundle,
) -> CliResult[DatasetFlowResult]:
    """Show producer/consumer graph for a dataset (structured).

    Parameters
    ----------
    table_key
        Dataset table key.
    runtime
        Runtime bundle for catalog access.

    Returns
    -------
    CliResult[DatasetFlowResult]
        Flow result with producers and consumers.
    """
    try:
        _ = get_contract_for_table_key(table_key)
    except KeyError:
        return fail_dataset_not_found(table_key)
    producers, consumers = _flow_targets_for_table_key(table_key, catalog=runtime.catalog)

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
    runtime_bundle = compose_cli_runtime_bundle(runtime=ctx.runtime, gateway=ctx.gateway)
    return dataset_flow_structured(table_key=table_key, runtime=runtime_bundle)


def dataset_constraints_structured(*, table_key: str) -> CliResult[DatasetConstraintsResult]:
    """Show constraint summary for a dataset (structured).

    Extracts constraints from the table schema and returns them in
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
    schema_service = get_schema_service()
    table_schema = schema_service.get_table_schema(table_key)
    if table_schema is None:
        return fail_dataset_not_found(table_key)

    constraint_set = extract_constraints_from_table_schema(table_schema)

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
    """Start the HTTP server with production-grade Uvicorn configuration.

    Parameters
    ----------
    ctx
        Command context with params:
        - host: Server host (default: CODEINTEL_HOST)
        - port: Server port (default: CODEINTEL_PORT)
        - reload: Enable hot reload
        - workers: Number of worker processes

    Returns
    -------
    CliResult[ServeStartResult]
        Server start result (after server stops).

    Notes
    -----
    This function blocks while the server is running.
    """
    settings = get_serving_settings()
    host = ctx.params.get_str("host") or settings.host
    port = ctx.params.get_int("port", settings.port)
    reload = ctx.params.get_bool("reload", default=False)
    workers = ctx.params.get_int("workers", settings.uvicorn_workers)

    pointer = ServingSnapshotPointer.load(settings.serve_dir / "current.json")
    LOG.info("Starting HTTP server at http://%s:%d (workers=%d)", host, port, workers)

    # Build Uvicorn configuration dict from settings
    uvicorn_config: dict[str, object] = {
        "host": host,
        "port": port,
        "loop": settings.uvicorn_loop,
        "http": settings.uvicorn_http,
        "timeout_keep_alive": settings.uvicorn_timeout_keep_alive,
        "backlog": settings.uvicorn_backlog,
        "access_log": settings.uvicorn_access_log,
        "log_level": "info",
    }

    # Optional concurrency limits
    if settings.uvicorn_limit_concurrency is not None:
        uvicorn_config["limit_concurrency"] = settings.uvicorn_limit_concurrency
    if settings.uvicorn_limit_max_requests is not None:
        uvicorn_config["limit_max_requests"] = settings.uvicorn_limit_max_requests

    # Security: hide server header
    if not settings.uvicorn_server_header:
        uvicorn_config["server_header"] = False

    # Proxy support
    if settings.uvicorn_proxy_headers:
        uvicorn_config["proxy_headers"] = True
        uvicorn_config["forwarded_allow_ips"] = settings.uvicorn_forwarded_allow_ips

    # Run with appropriate mode
    if workers > 1:
        uvicorn.run(
            "codeintel.cli.serving_factory:create_serving_app_from_env",
            factory=True,
            workers=workers,
            **uvicorn_config,  # type: ignore[arg-type]
        )
    elif reload:
        uvicorn.run(
            "codeintel.cli.serving_factory:create_serving_app_from_env",
            factory=True,
            reload=True,
            **uvicorn_config,  # type: ignore[arg-type]
        )
    else:
        app = create_serving_app(settings)
        uvicorn.run(app, **uvicorn_config)  # type: ignore[arg-type]

    return CliResult.ok(
        ServeStartResult(
            server_type="http",
            host=host,
            port=port,
            auto_pipeline=False,
            repo=pointer.repo,
            commit=pointer.commit,
            db_path=str(pointer.db_path),
        )
    )


def serve_mcp_handler(_ctx: CommandContext) -> CliResult[ServeStartResult]:
    """Start the MCP server.

    Parameters
    ----------
    _ctx
        Command context (unused for MCP start).

    Notes
    -----
    This function blocks while the server is running.

    Returns
    -------
    CliResult[ServeStartResult]
        Server start result (after server stops).
    """
    settings = get_serving_settings()
    pointer = ServingSnapshotPointer.load(settings.serve_dir / "current.json")

    LOG.info("Starting MCP server (transport=%s)", settings.mcp_transport)
    mcp = create_mcp_server(settings)
    if settings.mcp_transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(
            transport="streamable-http",
            host=settings.host,
            port=settings.port,
            json_response=True,
            stateless_http=False,
        )

    is_http = settings.mcp_transport != "stdio"
    host: str | None = settings.host if is_http else None
    port: int | None = settings.port if is_http else None

    return CliResult.ok(
        ServeStartResult(
            server_type="mcp",
            host=host,
            port=port,
            auto_pipeline=False,
            repo=pointer.repo,
            commit=pointer.commit,
            db_path=str(pointer.db_path),
        )
    )


__all__ = [
    "DatasetDescribeResult",
    "DatasetFlowResult",
    "DatasetInfoResult",
    "DatasetListResult",
    "DatasetVerifyResult",
    "ServeStartResult",
    "dataset_describe_handler",
    "dataset_describe_structured",
    "dataset_flow_handler",
    "dataset_flow_structured",
    "dataset_info_handler",
    "dataset_info_structured",
    "dataset_list_handler",
    "dataset_verify_handler",
    "serve_http_handler",
    "serve_mcp_handler",
]
