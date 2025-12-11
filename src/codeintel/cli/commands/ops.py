"""Op command group and dynamic operation registration.

Note: Operation commands require runtime/gateway and use dynamic registration.
"""

from __future__ import annotations

import inspect
import json
import logging
import sys
import types
from collections.abc import Callable, Iterable
from dataclasses import MISSING, dataclass, field, make_dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    Protocol,
    Union,
    Unpack,
    cast,
    get_args,
    get_origin,
)

from cyclopts import App, Group, Parameter

from codeintel.cli.commands._common import (
    SHARED_FLAGS_METADATA,
    OutputFormatCLI,
    RuntimeCLI,
    SharedFlags,
    get_output_format,
    get_verbose,
    runtime_field,
)
from codeintel.cli.commands._help import _AppCallKwargs
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.core import OutputEnvelope, iter_stdin_records
from codeintel.cli.core.output import merge_stdin_with_args
from codeintel.cli.errors import ValidationError
from codeintel.cli.handlers._utilities import runtime_gateway
from codeintel.cli.handlers.ops import (
    op_call_handler,
    op_list_handler,
)
from codeintel.cli.introspection import (
    CliParamSpec,
    OperationCliMetadata,
    build_operation_cli_metadata,
    get_operations_with_cli_support,
)
from codeintel.cli.project import (
    ProjectNotFoundError,
    plan_dry_run,
    render_dry_run,
)
from codeintel.cli.rendering.types import OutputFormat
from codeintel.cli.resolution import ResolutionError, resolve_from_params
from codeintel.cli.resolution.types import ResolvedRuntime
from codeintel.serving.auto_pipeline import run_operation_prereqs
from codeintel.serving.bootstrap import build_service_stack
from codeintel.serving.operations.catalog import (
    get_operation,
    register_test_operation,
    unregister_test_operation,
)

op_app = App(
    name="op",
    help="Operation invocation commands.",
)

# -----------------------------------------------------------------------------
# Operation Aliases for Progressive Disclosure
# -----------------------------------------------------------------------------

# Short aliases for frequently used operations.
# Maps short alias to full CLI command name.
OPERATION_ALIASES: dict[str, str] = {
    "hr": "functions-high-risk",
    "fs": "function-summary",
    "cg": "graph-call-neighbors",
    "cn": "graph-call-neighborhood",
    "ib": "graph-import-boundary",
    "file": "file-summary",
    "fp": "profiles-function",
    "mp": "profiles-module",
    "sl": "subsystems-list",
    "sd": "subsystems-detail",
    "ss": "subsystems-search",
}


def _get_aliases_for_operation(cli_name: str) -> list[str]:
    """Get all aliases for an operation CLI name.

    Parameters
    ----------
    cli_name
        The full CLI command name (e.g., 'functions-high_risk').

    Returns
    -------
    list[str]
        List of short aliases for this operation.
    """
    return [alias for alias, name in OPERATION_ALIASES.items() if name == cli_name]


SimpleNamespace = types.SimpleNamespace
_ROOT_APP_HOLDER: dict[str, App | None] = {"app": None}


def set_root_app(root: App) -> None:
    """Inject the root app for parse-only helpers without creating import cycles."""
    _ROOT_APP_HOLDER["app"] = root


def get_app() -> App:
    """Return the root Cyclopts app for embedding and testing.

    This function provides access to the root application instance after
    it has been initialized by ``cyclopts_app.py``. Use this for
    programmatic invocation or test scenarios.

    Returns
    -------
    App
        The root Cyclopts application instance.

    Raises
    ------
    RuntimeError
        If the root app has not been initialized via ``set_root_app()``.
    """
    root = _ROOT_APP_HOLDER["app"]
    if root is None:
        message = "Root app not initialized. Import cyclopts_app to initialize."
        raise RuntimeError(message)
    return root


def app_proxy(
    tokens: str | Iterable[str] | None = None, **call_kwargs: Unpack[_AppCallKwargs]
) -> types.SimpleNamespace:
    """Invoke the root Cyclopts app with typed kwargs for embedding and tests.

    This function provides a parse-only or execute-and-return interface
    to the CLI. Use it when you need to invoke CLI commands programmatically
    and retrieve parsed arguments or results.

    Parameters
    ----------
    tokens
        Command tokens to parse (e.g., ``["op", "list", "--category", "core"]``).
    **call_kwargs
        Typed keyword arguments for ``App.__call__``. Common options:
        - ``result_action``: Set to ``"return_value"`` for parse-only flows.
        - ``exit_on_error``: Set to ``False`` to raise exceptions instead of exiting.
        - ``print_error``: Set to ``False`` to suppress error output.

    Returns
    -------
    types.SimpleNamespace
        Parsed namespace when ``result_action`` includes ``"return_value"``.

    Examples
    --------
    Parse-only invocation returning kwargs:

    >>> ns = app_proxy(["op", "list"], result_action="return_value")
    >>> category = ns.kwargs.get("category")  # Access parsed values

    See Also
    --------
    get_app : Returns the root app; raises RuntimeError if not initialized.
    """
    root_app = get_app()
    result = root_app(tokens, **call_kwargs)
    return cast("types.SimpleNamespace", result)


# Track dynamically registered operation command names to avoid duplicates
_REGISTERED_OP_COMMANDS: set[str] = set()
_GROUPS_BY_ROLE: dict[str, Group] = {
    "selector": Group(
        "Target Selection",
        default_parameter=Parameter(negative=()),
    ),
    "filter": Group(
        "Filtering Options",
        default_parameter=Parameter(negative=()),
    ),
    "advanced": Group(
        "Advanced Options",
        default_parameter=Parameter(negative=()),
    ),
}
FieldDef = tuple[str, object, Any] | tuple[str, object]


class OperationCliArgs(Protocol):
    """Attributes required for dynamic operation invocation."""

    runtime: RuntimeCLI
    skip_prereqs: bool
    from_stdin: bool
    dry_run: bool


# -----------------------------------------------------------------------------
# op commands
# -----------------------------------------------------------------------------

# Config for op commands - no runtime needed for listing
_OP_NO_RUNTIME_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)
# Config for op call - requires runtime
_OP_RUNTIME_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)


@cli_command("op.list", handler=op_list_handler, config=_OP_NO_RUNTIME_CONFIG)
@op_app.command(name="list")
@dataclass
class OpListCommand:
    """List available serving operations."""

    category: Annotated[
        str | None,
        Parameter(
            name=["--category", "-c"],
            help="Filter by operation category.",
        ),
    ] = None
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("op.call", handler=op_call_handler, config=_OP_RUNTIME_CONFIG)
@op_app.command(name="call")
@dataclass
class OpCallCommand:
    """Invoke a serving operation end-to-end."""

    op_id: Annotated[
        str,
        Parameter(
            name=None,
            help="Operation ID to invoke.",
        ),
    ]
    params: Annotated[
        list[str] | None,
        Parameter(
            name=None,
            help="Operation parameters as key=value pairs.",
        ),
    ] = None
    skip_prereqs: Annotated[
        bool,
        Parameter(
            name="--skip-prereqs",
            help="Skip prerequisite pipeline execution.",
            negative=(),
        ),
    ] = False
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


# -----------------------------------------------------------------------------
# Dynamic operation helpers
# -----------------------------------------------------------------------------


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
    if isinstance(base_type, type) and issubclass(base_type, Path):
        return Path
    if isinstance(base_type, type) and issubclass(base_type, Enum):
        return base_type
    return str


def _is_choice_type(type_hint: type[Any] | None) -> bool:
    """Return True if the hint is an Enum or Literal[...] type.

    Returns
    -------
    bool
        True when the type maps to a finite choice set.
    """
    if type_hint is None:
        return False
    origin = get_origin(type_hint)
    if origin is Literal:
        return True
    base_type = _extract_base_type(type_hint)
    return isinstance(base_type, type) and issubclass(base_type, Enum)


def _is_path_type(type_hint: type[Any] | None) -> bool:
    """Return True if the hint resolves to a pathlib.Path.

    Returns
    -------
    bool
        True when the base type is Path.
    """
    base_type = _extract_base_type(type_hint)
    return isinstance(base_type, type) and issubclass(base_type, Path)


def _is_env_like(name: str) -> bool:
    """Heuristic for environment/venv path parameters.

    Returns
    -------
    bool
        True if the name implies a virtualenv/environment path.
    """
    lowered = name.lower()
    return "venv" in lowered or lowered.endswith(("_env", "env"))


def _is_output_like(name: str) -> bool:
    """Heuristic for output/destination path parameters.

    Returns
    -------
    bool
        True if the name implies an output/destination path.
    """
    lowered = name.lower()
    return any(token in lowered for token in ("output", "dest", "destination"))


def _path_validator(
    *, require_exists: bool, require_dir: bool | None
) -> Callable[[type[Any], Path], None]:
    """Build a simple path validator.

    Returns
    -------
    Callable[[type[Any], Path], None]
        Validator enforcing existence and shape constraints.
    """

    def _validate(_type: type[Any], value: Path) -> None:
        if require_exists and not value.exists():
            msg = f"Path does not exist: {value}"
            raise ValueError(msg)
        if require_dir is True and not value.is_dir():
            msg = f"Expected directory path: {value}"
            raise ValueError(msg)
        if require_dir is False and value.exists() and not value.is_file():
            msg = f"Expected file path: {value}"
            raise ValueError(msg)
        if not require_exists:
            parent = value.parent
            if parent and not parent.exists():
                msg = f"Parent directory does not exist: {parent}"
                raise ValueError(msg)

    return _validate


def _path_defaults_and_validator(
    spec: CliParamSpec,
) -> tuple[object, Callable[[type[Any], Path], None] | None]:
    """Infer a default and validator for path-like parameters.

    Returns
    -------
    tuple[object, Callable[[type[Any], Path], None] | None]
        (default, validator) tuple; validator may be None.
    """
    if not _is_path_type(spec.python_type):
        return spec.default, None

    validator: Callable[[type[Any], Path], None] | None = None
    default_override: object = spec.default

    if _is_env_like(spec.name) and spec.default is inspect.Parameter.empty:
        default_override = Path(".venv")
        validator = _path_validator(require_exists=True, require_dir=True)
    elif _is_output_like(spec.name):
        validator = _path_validator(require_exists=False, require_dir=None)
    else:
        validator = _path_validator(require_exists=True, require_dir=None)

    return default_override, validator


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
    default_override, path_validator = _path_defaults_and_validator(spec)
    patched_spec = CliParamSpec(
        name=spec.name,
        cli_name=spec.cli_name,
        python_type=spec.python_type,
        default=default_override,
        role=spec.role,
        help_text=spec.help_text,
        help_panel=spec.help_panel,
        is_optional=spec.is_optional,
    )

    annotation, default = _make_param_annotation(patched_spec)
    cli_type = _cli_type_for_spec(spec)
    show_choices = True if _is_choice_type(spec.python_type) else None
    converter: Callable[..., object] | str | None = None
    if isinstance(cli_type, type) and issubclass(cli_type, Path):
        converter = Path
    parameter = Parameter(
        name=[f"--{spec.cli_name}"],
        help=spec.help_text,
        show_choices=show_choices,
        converter=converter,
        validator=path_validator,
    )
    group = _GROUPS_BY_ROLE.get(spec.role)
    if group is not None:
        annotated_type = Annotated[annotation, group, parameter]
    else:
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

    runtime_field_def = ("runtime", RuntimeCLI, runtime_field())
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
    from_stdin_field = (
        "from_stdin",
        Annotated[
            bool,
            Parameter(
                name="--from-stdin",
                help="Read input records from stdin (JSON or JSONL).",
                negative=(),
            ),
        ],
        False,
    )
    dry_run_field = (
        "dry_run",
        Annotated[
            bool,
            Parameter(
                name="--dry-run",
                help="Show execution plan without running.",
                negative=(),
            ),
        ],
        False,
    )
    field_definitions = [
        *required_fields,
        *optional_fields,
        runtime_field_def,
        skip_field,
        from_stdin_field,
        dry_run_field,
    ]

    cls_name = f"{metadata.cli_name.replace('-', '_').title().replace('_', '')}OpCli"
    params_cls = make_dataclass(
        cls_name,
        field_definitions,
        kw_only=True,
    )
    params_cls.__module__ = __name__
    return params_cls


def _runtime_from_cli(cli: RuntimeCLI) -> ResolvedRuntime:
    """Build a runtime from CLI flags with Cyclopts-native error handling.

    Parameters
    ----------
    cli
        RuntimeCLI instance with project parameters.

    Returns
    -------
    ResolvedRuntime
        Resolved runtime for invoking operations.

    Raises
    ------
    ValidationError
        If runtime resolution fails.
    """
    try:
        # Pass ALL RuntimeCLI fields to enable fallback to explicit params
        # when no project file exists
        params: dict[str, object] = {
            "project_root": cli.project_root,
            "repo": cli.repo,
            "commit": cli.commit,
            "db_path": cli.db_path,
            "build_dir": cli.build_dir,
            "repo_root": cli.repo_root,
            "document_output_dir": cli.document_output_dir,
        }
        return resolve_from_params(params)
    except (ProjectNotFoundError, ResolutionError) as exc:
        msg = str(exc) or "No codeintel.yaml found. Provide --root or create a project file."
        raise ValidationError(msg) from exc


def _invoke_operation_with_prereqs(
    op_id: str,
    params: dict[str, Any],
    runtime: ResolvedRuntime,
    *,
    skip_prereqs: bool,
    verbose: bool,
) -> None:
    """Run optional prerequisites then invoke the operation and print result.

    Parameters
    ----------
    op_id
        Operation identifier.
    params
        Operation parameters.
    runtime
        Resolved runtime context.
    skip_prereqs
        Whether to skip prerequisite execution.
    verbose
        Whether to emit verbose output.
    """
    sys.stdout.write(f"Invoking operation '{op_id}'...\n")

    result = _invoke_operation_for_result(
        op_id,
        params,
        runtime,
        skip_prereqs=skip_prereqs,
        verbose=verbose,
    )

    sys.stdout.write(json.dumps(result, indent=2, default=str))
    sys.stdout.write("\n")


def _invoke_operation_for_result(
    op_id: str,
    params: dict[str, Any],
    runtime: ResolvedRuntime,
    *,
    skip_prereqs: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Invoke an operation and return the result as a dictionary.

    Parameters
    ----------
    op_id
        Operation identifier.
    params
        Operation parameters.
    runtime
        Resolved runtime context.
    skip_prereqs
        Whether to skip prerequisite execution.
    verbose
        Whether to emit verbose output.

    Returns
    -------
    dict[str, Any]
        The operation result.

    Raises
    ------
    ValidationError
        When the operation fails.
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    op = get_operation(op_id)
    if op is None:
        message = f"Unknown operation: {op_id}"
        raise ValidationError(message)

    with runtime_gateway(runtime) as gateway:
        if not skip_prereqs:
            run_operation_prereqs(
                op_id=op_id,
                gateway=gateway,
                snapshot=runtime.snapshot,
                paths=runtime.paths,
                tools=runtime.tools,
            )

        stack = build_service_stack(runtime.serving, gateway=gateway)
        try:
            method = getattr(stack.service, op.backend_method, None)
            if method is None:
                message = f"Backend method not found: {op.backend_method}"
                raise ValidationError(message)

            result = method(**params)

            # Convert result to dict
            if hasattr(result, "model_dump"):
                return result.model_dump(mode="json")
            if hasattr(result, "__dict__"):
                return dict(result.__dict__)
            return {"value": result}
        finally:
            stack.close()


def _execute_from_stdin(
    metadata: OperationCliMetadata,
    cfg: OperationCliArgs,
    runtime: ResolvedRuntime,
    *,
    verbose: bool,
) -> None:
    """Execute operation for each record from stdin.

    Parameters
    ----------
    metadata
        Operation CLI metadata.
    cfg
        CLI configuration with explicit arguments.
    runtime
        Resolved runtime context.
    verbose
        Whether to emit verbose output.
    """
    cli_args = _build_params_dict(cfg, metadata.params)

    # Read all records first to get count for progress (if needed later)
    records = list(iter_stdin_records())
    results: list[dict[str, object]] = []

    for stdin_record in records:
        # CLI args override stdin values
        merged_params = merge_stdin_with_args(stdin_record, cli_args)

        try:
            result = _invoke_operation_for_result(
                metadata.operation.id,
                merged_params,
                runtime,
                skip_prereqs=cfg.skip_prereqs,
                verbose=verbose,
            )
            results.append({"input": stdin_record, "result": result, "success": True})
        except (ValidationError, ResolutionError, RuntimeError, ValueError) as exc:
            results.append(
                {
                    "input": stdin_record,
                    "error": str(exc),
                    "success": False,
                }
            )

    # Output all results as JSON array
    output_format = get_output_format(
        getattr(cfg, "output", None) or OutputFormatCLI(),
        default=OutputFormat.JSON,
    )
    envelope = OutputEnvelope(
        data=results,
        metadata={"operation": metadata.operation.id, "count": len(results)},
    )
    envelope.write(output_format, sys.stdout)


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
    """Register a dynamic subcommand for an operation.

    If the operation has registered aliases, they are added as alternative
    command names for progressive disclosure.
    """
    command_name = metadata.cli_name
    if command_name in _REGISTERED_OP_COMMANDS:
        return

    params_cls = _make_operation_params_dataclass(metadata)
    cfg_annotation = Annotated[params_cls, Parameter(name="*")]

    def dynamic_op(cfg: OperationCliArgs | None = None) -> None:
        if cfg is None:
            message = "Operation parameters are required."
            raise ValidationError(message)
        typed_cfg = cfg
        runtime_cli = typed_cfg.runtime
        verbose = bool(get_verbose(runtime_cli))
        params = _build_params_dict(typed_cfg, metadata.params)

        # Handle dry-run mode
        if typed_cfg.dry_run:
            plan = plan_dry_run(
                metadata.operation.id,
                params,
                skip_prereqs=typed_cfg.skip_prereqs,
            )
            output_format = get_output_format(
                getattr(typed_cfg, "output", None) or OutputFormatCLI(),
                default=OutputFormat.TEXT,
            )
            render_dry_run(plan, output_format)
            return

        runtime = _runtime_from_cli(runtime_cli)

        # Handle stdin input for pipeable composition
        if typed_cfg.from_stdin:
            _execute_from_stdin(metadata, typed_cfg, runtime, verbose=verbose)
            return

        # Normal execution
        _invoke_operation_with_prereqs(
            metadata.operation.id,
            params,
            runtime,
            skip_prereqs=typed_cfg.skip_prereqs,
            verbose=verbose,
        )

    dynamic_op.__annotations__["cfg"] = cfg_annotation

    # Get aliases for this operation (if any)
    aliases = _get_aliases_for_operation(command_name)

    op_app.command(
        name=command_name,
        alias=aliases if aliases else None,
        help=metadata.operation.summary or metadata.operation.id,
    )(dynamic_op)
    _REGISTERED_OP_COMMANDS.add(command_name)


def register_dynamic_operations() -> None:
    """Register subcommands for all operations with CLI support."""
    for op in get_operations_with_cli_support():
        metadata = build_operation_cli_metadata(op)
        _register_dynamic_operation(metadata)


def build_param_field_for_spec(spec: CliParamSpec) -> FieldDef:
    """Public helper to construct a dataclass field tuple for a spec.

    Returns
    -------
    FieldDef
        Dataclass field definition including annotations/metadata.
    """
    return _make_param_field(spec)


def path_defaults_and_validator(
    spec: CliParamSpec,
) -> tuple[object, Callable[[type[Any], Path], None] | None]:
    """Public helper to infer path defaults/validators for a spec.

    Returns
    -------
    tuple[object, Callable[[type[Any], Path], None] | None]
        (default, validator) tuple; validator may be None.
    """
    return _path_defaults_and_validator(spec)


def path_validator(
    *,
    require_exists: bool = True,
    require_dir: bool | None = None,
) -> Callable[[type[Any], Path], None]:
    """Build a path validator for Cyclopts Parameter validation.

    Use this to create validators for path parameters in CLI commands.

    Parameters
    ----------
    require_exists
        When True, path must exist. When False, parent must exist but file can be missing.
    require_dir
        When True, path must be directory. When False, path must be file.
        When None, no shape constraint is applied.

    Returns
    -------
    Callable[[type[Any], Path], None]
        Validator function suitable for Cyclopts Parameter(validator=...).

    Examples
    --------
    Required existing path:

    >>> validator = path_validator(require_exists=True)

    Output path (parent must exist):

    >>> validator = path_validator(require_exists=False)

    Required directory:

    >>> validator = path_validator(require_exists=True, require_dir=True)
    """
    return _path_validator(require_exists=require_exists, require_dir=require_dir)


def register_dynamic_operation_for_tests(metadata: OperationCliMetadata) -> None:
    """Register a dynamic operation command for testing purposes.

    Register both the Cyclopts CLI command and the underlying operation
    in the operations catalog so that `get_operation()` will find it
    during invocation.

    Parameters
    ----------
    metadata
        Operation CLI metadata including the Operation instance.

    Notes
    -----
    Always call `unregister_dynamic_operation_for_tests()` to clean up
    after tests to avoid polluting the catalog.
    """
    # Register in the operations catalog so get_operation() finds it
    register_test_operation(metadata.operation)
    # Register the Cyclopts CLI command
    _register_dynamic_operation(metadata)


def unregister_dynamic_operation_for_tests(op_id: str) -> bool:
    """Remove a test operation from both the CLI and catalog.

    Parameters
    ----------
    op_id
        Operation identifier to remove (e.g., "test.choice.op").

    Returns
    -------
    bool
        True if the operation was found and removed from the catalog.

    Notes
    -----
    The Cyclopts command cannot be unregistered at runtime, but removing
    from the catalog prevents `invoke_operation()` from executing it.
    """
    return unregister_test_operation(op_id)


# Register dynamic operations on module import
register_dynamic_operations()


__all__ = [
    "OPERATION_ALIASES",
    "OperationCliArgs",
    "SimpleNamespace",
    "app_proxy",
    "build_param_field_for_spec",
    "get_app",
    "op_app",
    "path_defaults_and_validator",
    "path_validator",
    "register_dynamic_operation_for_tests",
    "register_dynamic_operations",
    "set_root_app",
    "unregister_dynamic_operation_for_tests",
]
