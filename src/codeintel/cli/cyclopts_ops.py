"""Cyclopts wiring for op, dataset, and serve command groups."""

from __future__ import annotations

import inspect
import logging
import types
from dataclasses import MISSING, dataclass, field, make_dataclass
from typing import Annotated, Any, Protocol, Union, get_args, get_origin

from cyclopts import App, Parameter

from codeintel.cli.cli_errors import ValidationError
from codeintel.cli.commands._common import OutputFormat
from codeintel.cli.cyclopts_common import (
    OutputFormatCLI,
    RuntimeCLI,
    RuntimeCliError,
    build_runtime_from_cli,
    resolve_output_format,
)
from codeintel.cli.op_params import (
    CliParamSpec,
    OperationCliMetadata,
    build_operation_cli_metadata,
    get_operations_with_cli_support,
)
from codeintel.cli.ops_handlers import (
    dataset_describe_handler,
    dataset_list_handler,
    dataset_verify_handler,
    invoke_operation,
    op_call_handler,
    op_list_handler,
    serve_http_handler,
    serve_mcp_handler,
)
from codeintel.cli.project import ProjectRuntime
from codeintel.serving.auto_pipeline import run_operation_prereqs

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
FieldDef = tuple[str, object, Any] | tuple[str, object]


class OperationCliArgs(Protocol):
    """Attributes required for dynamic operation invocation."""

    runtime: RuntimeCLI
    skip_prereqs: bool


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
    output: Annotated[OutputFormatCLI, Parameter(name="*")] = field(default_factory=OutputFormatCLI)


@op_app.command(name="list")
def op_list(
    cfg: Annotated[OpListCli, Parameter(name="*")] | None = None,
) -> None:
    """List available serving operations."""
    cfg = cfg or OpListCli()
    output_format = resolve_output_format(
        json_flag=cfg.output.json,
        explicit=cfg.output.output_format,
        default=OutputFormat.TEXT,
    )
    op_list_handler(
        category=cfg.category,
        output_format=output_format,
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
    ValidationError
        If an operation ID is not provided.
    """
    cfg = cfg or OpCallCli()
    if not cfg.op_id:
        message = "Operation ID is required."
        raise ValidationError(message)
    runtime = cfg.runtime
    project_runtime = _runtime_from_cli(runtime)
    op_call_handler(
        op_id=cfg.op_id,
        params=cfg.params,
        runtime=project_runtime,
        skip_prereqs=cfg.skip_prereqs,
        verbose=bool(runtime.verbose),
    )


def _extract_base_type(type_hint: type[Any] | None) -> type[Any] | None:
    """Extract the underlying type from an optional or union hint.

    Returns
    -------
    type[Any] | None
        Base concrete type with ``None`` and union wrappers removed.
    """
    if type_hint is None:
        return None
    origin = get_origin(type_hint)
    if origin is None:
        return type_hint
    if origin in {types.UnionType, Union}:
        args = get_args(type_hint)
        non_none = [arg for arg in args if arg is not types.NoneType]
        if non_none:
            return non_none[0]
        return None
    return type_hint


def _cli_type_for_spec(spec: CliParamSpec) -> type[Any]:
    """Map a backend type hint to a Cyclopts-friendly CLI type.

    Returns
    -------
    type[Any]
        Primitive type accepted by Cyclopts for the parameter.
    """
    base_type = _extract_base_type(spec.python_type)
    if base_type is None:
        return str
    if base_type in {int, float, bool, str}:
        return base_type
    return str


def _make_param_annotation(spec: CliParamSpec) -> tuple[type[Any], Any]:
    """Return the annotation and default for a CLI parameter.

    Returns
    -------
    tuple[type[Any], Any]
        Annotation and default resolved for Cyclopts.
    """
    cli_type = _cli_type_for_spec(spec)
    default = spec.default
    if default is inspect.Parameter.empty:
        default = MISSING if spec.is_required else None

    annotation: type[Any] | Any = cli_type
    if default is None and spec.is_optional:
        annotation = cli_type | None
    return annotation, default


def _make_param_field(spec: CliParamSpec) -> FieldDef:
    """Construct a dataclass field tuple for dynamic CLI params.

    Returns
    -------
    tuple[str, object, Any]
        Field definition for ``make_dataclass``.
    """
    annotation, default = _make_param_annotation(spec)
    cli_type = _cli_type_for_spec(spec)
    parameter_kwargs = {
        "name": [f"--{spec.cli_name}"],
        "help": spec.help_text,
    }
    if cli_type is bool:
        parameter_kwargs["negative"] = []
    parameter = Parameter(**parameter_kwargs)
    annotated_type = Annotated[annotation, parameter]
    if default is MISSING:
        return (spec.name, annotated_type)
    return (spec.name, annotated_type, default)


def _make_operation_params_dataclass(metadata: OperationCliMetadata) -> type[Any]:
    """Build a keyword-only dataclass representing an operation's CLI surface.

    Returns
    -------
    type[Any]
        Dataclass type capturing operation parameters.
    """
    required_fields: list[FieldDef] = []
    optional_fields: list[FieldDef] = []
    required_field_len = 2

    for spec in metadata.params:
        field_def = _make_param_field(spec)
        if len(field_def) == required_field_len:
            required_fields.append(field_def)
        else:
            optional_fields.append(field_def)

    runtime_field = (
        "runtime",
        Annotated[RuntimeCLI, Parameter(name="*")],
        field(default_factory=RuntimeCLI),
    )
    skip_field = (
        "skip_prereqs",
        Annotated[
            bool,
            Parameter(
                name="--skip-prereqs",
                help="Skip prerequisite pipeline execution.",
                negative=(),
            ),
        ],
        False,
    )
    field_definitions = [*required_fields, *optional_fields, runtime_field, skip_field]

    cls_name = f"{metadata.cli_name.replace('-', '_').title().replace('_', '')}OpCli"
    params_cls = make_dataclass(
        cls_name,
        field_definitions,
        kw_only=True,
    )
    params_cls.__module__ = __name__
    return params_cls


def _runtime_from_cli(cli: RuntimeCLI) -> ProjectRuntime:
    """Build a runtime from CLI flags with Cyclopts-native error handling.

    Returns
    -------
    ProjectRuntime
        Resolved runtime for invoking operations.

    Raises
    ------
    ValidationError
        If runtime resolution fails.
    """
    try:
        return build_runtime_from_cli(cli)
    except RuntimeCliError as exc:
        raise ValidationError(str(exc)) from exc


def _invoke_operation_with_prereqs(
    op_id: str,
    params: dict[str, Any],
    runtime: ProjectRuntime,
    *,
    skip_prereqs: bool,
    verbose: bool,
) -> None:
    """Run optional prerequisites then invoke the operation."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not skip_prereqs:
        run_operation_prereqs(
            op_id=op_id,
            gateway=runtime.gateway,
            snapshot=runtime.snapshot,
            paths=runtime.paths,
            tools=runtime.tools,
        )

    invoke_operation(op_id, params, runtime)


def _build_params_dict(cfg: OperationCliArgs, specs: tuple[CliParamSpec, ...]) -> dict[str, Any]:
    """Extract non-null parameters from a CLI dataclass instance.

    Returns
    -------
    dict[str, Any]
        Mapping of parameter names to provided values.
    """
    params: dict[str, Any] = {}
    for spec in specs:
        value = getattr(cfg, spec.name)
        if value is not None:
            params[spec.name] = value
    return params


def _register_dynamic_operation(metadata: OperationCliMetadata) -> None:
    """Register a dynamic subcommand for an operation."""
    command_name = metadata.cli_name
    if command_name in _REGISTERED_OP_COMMANDS:
        return

    params_cls = _make_operation_params_dataclass(metadata)
    cfg_annotation = Annotated[params_cls, Parameter(name="*")]

    def dynamic_op(cfg: OperationCliArgs | None = None) -> None:  # type: ignore[unused-ignore]
        if cfg is None:
            message = "Operation parameters are required."
            raise ValidationError(message)
        typed_cfg = cfg
        runtime_cli = typed_cfg.runtime
        runtime = _runtime_from_cli(runtime_cli)
        params = _build_params_dict(typed_cfg, metadata.params)
        _invoke_operation_with_prereqs(
            metadata.operation.id,
            params,
            runtime,
            skip_prereqs=typed_cfg.skip_prereqs,
            verbose=bool(runtime_cli.verbose),
        )

    dynamic_op.__annotations__["cfg"] = cfg_annotation
    op_app.command(
        name=command_name,
        help=metadata.operation.summary or metadata.operation.id,
    )(dynamic_op)
    _REGISTERED_OP_COMMANDS.add(command_name)


def register_dynamic_operations() -> None:
    """Register subcommands for all operations with CLI support."""
    for op in get_operations_with_cli_support():
        metadata = build_operation_cli_metadata(op)
        _register_dynamic_operation(metadata)


# -----------------------------------------------------------------------------
# dataset commands
# -----------------------------------------------------------------------------


@dataclass
class DatasetListCli:
    """CLI surface for `codeintel dataset list`."""

    runtime: Annotated[RuntimeCLI, Parameter(name="*")] = field(default_factory=RuntimeCLI)
    output: Annotated[OutputFormatCLI, Parameter(name="*")] = field(default_factory=OutputFormatCLI)


@dataset_app.command(name="list")
def dataset_list(
    cfg: Annotated[DatasetListCli, Parameter(name="*")] | None = None,
) -> None:
    """List datasets from the registry."""
    cfg = cfg or DatasetListCli()  # type: ignore[call-arg]
    runtime = _runtime_from_cli(cfg.runtime)
    output_format = resolve_output_format(
        json_flag=cfg.output.json,
        explicit=cfg.output.output_format,
        default=OutputFormat.TEXT,
    )
    dataset_list_handler(
        runtime=runtime,
        output_format=output_format,
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
    output: Annotated[OutputFormatCLI, Parameter(name="*")] = field(default_factory=OutputFormatCLI)


@dataset_app.command(name="describe")
def dataset_describe(
    cfg: Annotated[DatasetDescribeCli, Parameter(name="*")] | None = None,
) -> None:
    """Show contract details for a dataset.

    Raises
    ------
    ValidationError
        If the dataset key is missing.
    """
    cfg = cfg or DatasetDescribeCli()
    if not cfg.table_key:
        message = "Dataset key is required."
        raise ValidationError(message)
    output_format = resolve_output_format(
        json_flag=cfg.output.json,
        explicit=cfg.output.output_format,
        default=OutputFormat.TEXT,
    )
    dataset_describe_handler(table_key=cfg.table_key, output_format=output_format)


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
    runtime = _runtime_from_cli(cfg.runtime)
    dataset_verify_handler(table_key=cfg.table_key, runtime=runtime)


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
    runtime_obj = _runtime_from_cli(runtime_cfg)
    serve_http_handler(
        host=host,
        port=port,
        auto_pipeline=auto_pipeline,
        reload=reload,
        runtime=runtime_obj,
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
    runtime_obj = _runtime_from_cli(runtime_cfg)
    serve_mcp_handler(
        auto_pipeline=auto_pipeline,
        runtime=runtime_obj,
    )


register_dynamic_operations()


__all__ = [
    "dataset_app",
    "op_app",
    "register_dynamic_operations",
    "serve_app",
]
