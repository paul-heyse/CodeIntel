"""Operation handlers.

Handlers for operation listing, invocation, dataset management, and server control.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import uvicorn

from codeintel.cli.cli_errors import ProblemDetail, ValidationError
from codeintel.cli.project import ProjectRuntime
from codeintel.cli.results import CliResult
from codeintel.config.datasets import get_dataset_contracts_by_table_key
from codeintel.serving.auto_pipeline import run_operation_prereqs
from codeintel.serving.bootstrap import build_service_stack
from codeintel.serving.http.fastapi import create_app as create_http_app
from codeintel.serving.mcp.server import main as run_mcp_server
from codeintel.serving.operations.catalog import get_operation, iter_operations
from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.storage.validation import collect_contract_issues

if TYPE_CHECKING:
    from codeintel.cli.handlers.protocol import EnhancedHandlerContext
    from codeintel.cli.resolution.types import ResolvedRuntime

LOG = logging.getLogger(__name__)

AUTO_PIPELINE_ENV = "CODEINTEL_AUTO_PIPELINE"


@dataclass(frozen=True)
class OpListResult:
    """Result from listing operations."""

    operations: list[dict[str, object]]
    count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "operations": self.operations,
            "count": self.count,
        }


@dataclass(frozen=True)
class DatasetListResult:
    """Result from listing datasets."""

    datasets: list[dict[str, object]]
    count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "datasets": self.datasets,
            "count": self.count,
        }


@dataclass(frozen=True)
class DatasetDescribeResult:
    """Result from describing a dataset."""

    table_key: str
    name: str
    description: str | None
    owner_package: str
    columns: list[dict[str, str | bool]]
    row_count: int | None
    upstream_dependencies: list[str]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "table_key": self.table_key,
            "name": self.name,
            "description": self.description,
            "owner_package": self.owner_package,
            "columns": self.columns,
            "row_count": self.row_count,
            "upstream_dependencies": self.upstream_dependencies,
        }


@dataclass(frozen=True)
class DatasetVerifyResult:
    """Result from verifying datasets."""

    verified: bool
    issues: list[str]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "verified": self.verified,
            "issues": self.issues,
        }


@dataclass(frozen=True)
class OpCallResult:
    """Result from invoking an operation."""

    op_id: str
    result: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "op_id": self.op_id,
            "result": self.result,
        }


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

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "server_type": self.server_type,
            "host": self.host,
            "port": self.port,
            "auto_pipeline": self.auto_pipeline,
            "repo": self.repo,
            "commit": self.commit,
            "db_path": self.db_path,
        }


def _get_str_param(
    ctx: EnhancedHandlerContext,
    name: str,
    default: str | None = None,
) -> str | None:
    """Extract string parameter from context.

    Parameters
    ----------
    ctx
        Handler context.
    name
        Parameter name.
    default
        Default value if not present.

    Returns
    -------
    str | None
        Parameter value or default.
    """
    value = ctx.params.get(name)
    if value is None:
        return default
    return str(value)


def _require_str_param(ctx: EnhancedHandlerContext, name: str) -> str:
    """Extract required string parameter from context.

    Parameters
    ----------
    ctx
        Handler context.
    name
        Parameter name.

    Returns
    -------
    str
        Parameter value.

    Raises
    ------
    ValueError
        If parameter is missing.
    """
    value = ctx.params.get(name)
    if value is None:
        msg = f"{name} parameter is required"
        raise ValueError(msg)
    return str(value)


def _get_int_param(
    ctx: EnhancedHandlerContext,
    name: str,
    default: int = 0,
) -> int:
    """Extract integer parameter from context.

    Parameters
    ----------
    ctx
        Handler context.
    name
        Parameter name.
    default
        Default value if not present.

    Returns
    -------
    int
        Parameter value.
    """
    value = ctx.params.get(name)
    if value is None:
        return default
    if isinstance(value, int):
        return value
    return int(str(value))


def _get_bool_param(
    ctx: EnhancedHandlerContext,
    name: str,
    *,
    default: bool = False,
) -> bool:
    """Extract boolean parameter from context.

    Parameters
    ----------
    ctx
        Handler context.
    name
        Parameter name.
    default
        Default value if not present.

    Returns
    -------
    bool
        Parameter value.
    """
    value = ctx.params.get(name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def _parse_param_value(value: str) -> str | int | float | bool:
    """Parse CLI parameter text into a primitive Python type.

    Parameters
    ----------
    value
        Raw string value.

    Returns
    -------
    str | int | float | bool
        Parsed value.
    """
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _resolved_to_project_runtime(runtime: ResolvedRuntime) -> ProjectRuntime:
    """Convert ResolvedRuntime to ProjectRuntime for backward compatibility.

    Parameters
    ----------
    runtime
        ResolvedRuntime from handler context.

    Returns
    -------
    ProjectRuntime
        Compatible ProjectRuntime instance.
    """
    gateway = open_gateway(StorageConfig.for_readonly(runtime.paths.db_path))
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


def op_list_handler(ctx: EnhancedHandlerContext) -> CliResult[OpListResult]:
    """List available serving operations.

    Parameters
    ----------
    ctx
        Handler context with params:
        - category: Optional category filter

    Returns
    -------
    CliResult[OpListResult]
        List of operations matching the filter.
    """
    category = _get_str_param(ctx, "category")

    operations = list(iter_operations())
    if category:
        operations = [op for op in operations if op.category == category]

    operation_dicts: list[dict[str, object]] = [
        {
            "id": op.id,
            "category": op.category,
            "summary": op.summary,
            "http_path": op.http_path,
            "tool_name": op.tool_name,
        }
        for op in sorted(operations, key=lambda o: o.id)
    ]

    return CliResult.ok(OpListResult(operations=operation_dicts, count=len(operations)))


def op_call_handler(ctx: EnhancedHandlerContext) -> CliResult[OpCallResult]:
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
    CliResult[OpCallResult]
        Operation result.
    """
    op_id = _require_str_param(ctx, "op_id")
    params_raw = ctx.params.get("params") or []
    skip_prereqs = _get_bool_param(ctx, "skip_prereqs", default=False)

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

    kwargs: dict[str, str | int | float | bool] = {}
    if params_raw is None:
        params_list: list[str] = []
    elif isinstance(params_raw, list):
        params_list = [str(p) for p in params_raw]
    else:
        params_list = [str(params_raw)]
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
        kwargs[key] = _parse_param_value(value)

    project_runtime = _resolved_to_project_runtime(ctx.runtime)

    if not skip_prereqs:
        run_operation_prereqs(
            op_id=op_id,
            gateway=project_runtime.gateway,
            snapshot=project_runtime.snapshot,
            paths=project_runtime.paths,
            tools=project_runtime.tools,
        )

    result = _invoke_operation_structured(op_id, kwargs, project_runtime)
    return CliResult.ok(OpCallResult(op_id=op_id, result=result))


def _invoke_operation_structured(
    op_id: str,
    kwargs: dict[str, Any],
    runtime: ProjectRuntime,
) -> dict[str, object]:
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
    dict[str, object]
        Operation result as dictionary.

    Raises
    ------
    ValidationError
        If the operation or backend method is not found.
    """
    op = get_operation(op_id)
    if op is None:
        msg = f"Unknown operation: {op_id}"
        raise ValidationError(msg)

    stack = build_service_stack(
        config=runtime.serving,
        gateway=runtime.gateway,
    )

    try:
        method = getattr(stack.service, op.backend_method, None)
        if method is None:
            msg = f"Backend method not found: {op.backend_method}"
            raise ValidationError(msg)

        result = method(**kwargs)

        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        if hasattr(result, "__dict__"):
            return result.__dict__
        return {"result": result}
    finally:
        stack.close()


def dataset_list_handler(ctx: EnhancedHandlerContext) -> CliResult[DatasetListResult]:
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
    project_runtime = _resolved_to_project_runtime(ctx.runtime)
    registry = project_runtime.gateway.datasets
    meta = registry.meta or {}

    dataset_dicts: list[dict[str, object]] = [
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
    ctx: EnhancedHandlerContext,
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
    table_key = _require_str_param(ctx, "table_key")

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
            name=contract.name,
            description=contract.description,
            owner_package=contract.owner_package or "",
            columns=column_dicts,
            row_count=None,
            upstream_dependencies=list(contract.upstream_dependencies),
        )
    )


def dataset_verify_handler(
    ctx: EnhancedHandlerContext,
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
    table_key = _get_str_param(ctx, "table_key")
    project_runtime = _resolved_to_project_runtime(ctx.runtime)

    issues = collect_contract_issues(project_runtime.gateway.con)

    if table_key:
        issues = [issue for issue in issues if table_key in issue]

    return CliResult.ok(DatasetVerifyResult(verified=len(issues) == 0, issues=issues))


def serve_http_handler(ctx: EnhancedHandlerContext) -> CliResult[ServeStartResult]:
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
    host = _get_str_param(ctx, "host", "127.0.0.1") or "127.0.0.1"
    port = _get_int_param(ctx, "port", 8000)
    auto_pipeline = _get_bool_param(ctx, "auto_pipeline", default=False)
    reload = _get_bool_param(ctx, "reload", default=False)

    project_runtime = _resolved_to_project_runtime(ctx.runtime)

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
            repo=ctx.runtime.repo,
            commit=ctx.runtime.commit,
            db_path=str(ctx.runtime.paths.db_path),
        )
    )


def serve_mcp_handler(ctx: EnhancedHandlerContext) -> CliResult[ServeStartResult]:
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
    auto_pipeline = _get_bool_param(ctx, "auto_pipeline", default=False)

    project_runtime = _resolved_to_project_runtime(ctx.runtime)

    _setup_serving_env(project_runtime, auto_pipeline=auto_pipeline)

    LOG.info("Starting MCP server (auto_pipeline=%s)", auto_pipeline)

    # MCP server runs and exits
    sys.exit(run_mcp_server())


__all__ = [
    "AUTO_PIPELINE_ENV",
    "DatasetDescribeResult",
    "DatasetListResult",
    "DatasetVerifyResult",
    "OpCallResult",
    "OpListResult",
    "ServeStartResult",
    "dataset_describe_handler",
    "dataset_list_handler",
    "dataset_verify_handler",
    "op_call_handler",
    "op_list_handler",
    "serve_http_handler",
    "serve_mcp_handler",
]
