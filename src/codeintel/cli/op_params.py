"""Dynamic CLI parameter introspection and command factory for serving operations.

This module provides infrastructure for dynamically generating typed CLI commands
for each serving operation by introspecting the Protocol method signatures from
the query API layer.

The key components:
1. CliParamSpec - Typed specification for a CLI parameter with role classification
2. Signature introspection - Extract parameter info from Protocol methods
3. Role classification - Categorize parameters as selector/filter/advanced
4. Command factory - Generate Typer commands with proper type annotations
5. String tunnel pattern - Accept all params as Optional[str], coerce at runtime

The "string tunnel" pattern is used because Typer does not support:
- **kwargs in command functions
- Union types with multiple non-None types (e.g., str | int | float)

By accepting all parameters as Optional[str] and coercing to proper types at
runtime, we get full Typer integration (help, autocomplete) while maintaining
type safety.
"""

from __future__ import annotations

import inspect
import logging
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Protocol,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import typer

from codeintel.serving.backend import query_api
from codeintel.serving.operations.catalog import Operation, iter_operations

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

LOG = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Parameter Role Classification
# -----------------------------------------------------------------------------

ParamRole = Literal["selector", "filter", "advanced"]
"""Classification of CLI parameter purpose."""

# Selector parameters identify specific entities
_SELECTOR_NAMES: frozenset[str] = frozenset(
    {
        "goid_h128",
        "urn",
        "path",
        "rel_path",
        "module",
        "qualname",
        "subsystem_id",
        "dataset_name",
        "table_key",
    }
)

# Selector suffix patterns (e.g., function_goid_h128, caller_goid_h128)
_SELECTOR_SUFFIXES: tuple[str, ...] = ("_goid_h128", "_urn", "_id")

# Filter parameters restrict result sets
_FILTER_PREFIXES: tuple[str, ...] = ("min_", "max_")
_FILTER_SUFFIXES: tuple[str, ...] = ("_only",)
_FILTER_NAMES: frozenset[str] = frozenset(
    {
        "limit",
        "offset",
        "tested_only",
        "direction",
        "role",
        "q",
    }
)

# Advanced parameters control complex behavior
_ADVANCED_NAMES: frozenset[str] = frozenset(
    {
        "scope",
        "graph_scope",
        "radius",
        "max_nodes",
        "max_edges",
        "sample_limit",
        "module_limit",
    }
)


def _check_prefix_suffix_patterns(name: str) -> bool:
    """Check if name matches filter prefix or suffix patterns.

    Parameters
    ----------
    name
        Parameter name to check.

    Returns
    -------
    bool
        True if matches a filter pattern.
    """
    has_prefix = any(name.startswith(prefix) for prefix in _FILTER_PREFIXES)
    has_suffix = any(name.endswith(suffix) for suffix in _FILTER_SUFFIXES)
    return has_prefix or has_suffix


def _is_selector_by_suffix(name: str) -> bool:
    """Check if name matches selector suffix patterns.

    Parameters
    ----------
    name
        Parameter name to check.

    Returns
    -------
    bool
        True if matches a selector suffix pattern.
    """
    return any(name.endswith(suffix) for suffix in _SELECTOR_SUFFIXES)


def classify_param_role(
    name: str,
    *,
    operation: Operation | None = None,
) -> ParamRole:
    """Classify a parameter name into its functional role.

    Parameters
    ----------
    name
        Parameter name to classify.
    operation
        Optional operation for context-aware classification.

    Returns
    -------
    ParamRole
        Classification as selector, filter, or advanced.
    """
    # Direct name matches take priority (check all sets first)
    if name in _SELECTOR_NAMES or _is_selector_by_suffix(name):
        return "selector"
    if name in _ADVANCED_NAMES:
        return "advanced"
    if name in _FILTER_NAMES or _check_prefix_suffix_patterns(name):
        return "filter"

    # Graph-related operations default to advanced for unknown params
    if operation is not None and operation.required_graphs:
        return "advanced"

    # Default to filter for unknown parameters
    return "filter"


# -----------------------------------------------------------------------------
# Panel Name Mapping
# -----------------------------------------------------------------------------

_ROLE_TO_PANEL: Mapping[ParamRole, str] = {
    "selector": "Target Selection",
    "filter": "Filtering Options",
    "advanced": "Advanced Options",
}


def get_help_panel_for_role(role: ParamRole) -> str:
    """Return the rich_help_panel name for a parameter role.

    Parameters
    ----------
    role
        Parameter role classification.

    Returns
    -------
    str
        Panel name for Typer's rich_help_panel option.
    """
    return _ROLE_TO_PANEL[role]


# -----------------------------------------------------------------------------
# CLI Parameter Specification
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CliParamSpec:
    """Specification for a CLI parameter derived from a backend method signature.

    Attributes
    ----------
    name
        Parameter name as it appears in the method signature.
    cli_name
        CLI-friendly name (underscores replaced with hyphens).
    python_type
        Python type annotation for the parameter.
    default
        Default value if any, or inspect.Parameter.empty.
    role
        Functional role classification.
    help_text
        Generated help text for the parameter.
    help_panel
        Rich help panel name for grouping.
    is_optional
        Whether the parameter is optional (has default or Optional type).
    """

    name: str
    cli_name: str
    python_type: type[Any] | None
    default: Any
    role: ParamRole
    help_text: str
    help_panel: str
    is_optional: bool

    @property
    def is_required(self) -> bool:
        """Return True if the parameter is required.

        Returns
        -------
        bool
            True if required, False otherwise.
        """
        return not self.is_optional


def _make_cli_name(param_name: str) -> str:
    """Convert parameter name to CLI option format.

    Parameters
    ----------
    param_name
        Python parameter name with underscores.

    Returns
    -------
    str
        CLI-friendly name with hyphens.
    """
    return param_name.replace("_", "-")


def _is_optional_type(type_hint: type[Any] | None) -> bool:
    """Check if a type hint represents an Optional type.

    Parameters
    ----------
    type_hint
        Type annotation to check.

    Returns
    -------
    bool
        True if the type is Optional[X] or X | None.
    """
    if type_hint is None:
        return True

    origin = get_origin(type_hint)
    if origin is None:
        return False

    # Handle Union types (including X | None via types.UnionType or typing.Union)
    if origin is types.UnionType or origin is Union:
        args = get_args(type_hint)
        return type(None) in args

    return False


def _extract_base_type(type_hint: type[Any] | None) -> type[Any] | None:
    """Extract the base type from an Optional or Union type.

    Parameters
    ----------
    type_hint
        Type annotation that may be Optional[X] or X | None.

    Returns
    -------
    type | None
        Base type without None, or original type if not Optional.
    """
    if type_hint is None:
        return None

    origin = get_origin(type_hint)
    if origin is None:
        return type_hint

    # Handle Union types (types.UnionType for X | None, Union for Optional[X])
    union_origins = {types.UnionType, Union}

    if origin in union_origins:
        args = get_args(type_hint)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if non_none_args:
            return non_none_args[0]

    return type_hint


def _generate_help_text(name: str, role: ParamRole, python_type: type[Any] | None) -> str:
    """Generate help text for a CLI parameter.

    Parameters
    ----------
    name
        Parameter name.
    role
        Parameter role classification.
    python_type
        Python type annotation.

    Returns
    -------
    str
        Generated help text.
    """
    type_desc = ""
    if python_type is not None:
        base_type = _extract_base_type(python_type)
        if base_type is int:
            type_desc = " (integer)"
        elif base_type is float:
            type_desc = " (number)"
        elif base_type is bool:
            type_desc = " (flag)"
        elif base_type is str:
            type_desc = ""

    role_hints: dict[ParamRole, str] = {
        "selector": "Identify target",
        "filter": "Filter results",
        "advanced": "Advanced option",
    }

    # Convert name to human-readable form
    human_name = name.replace("_", " ")

    return f"{role_hints[role]}: {human_name}{type_desc}"


def build_cli_param_spec(
    param: inspect.Parameter,
    type_hints: dict[str, Any],
    operation: Operation | None = None,
) -> CliParamSpec:
    """Build a CliParamSpec from an inspect.Parameter.

    Parameters
    ----------
    param
        Inspect Parameter object from function signature.
    type_hints
        Type hints dictionary from get_type_hints.
    operation
        Optional operation for context-aware classification.

    Returns
    -------
    CliParamSpec
        Constructed CLI parameter specification.
    """
    name = param.name
    python_type = type_hints.get(name)
    default = param.default
    has_default = default is not inspect.Parameter.empty

    is_optional = has_default or _is_optional_type(python_type)
    role = classify_param_role(name, operation=operation)
    help_panel = get_help_panel_for_role(role)
    help_text = _generate_help_text(name, role, python_type)
    cli_name = _make_cli_name(name)

    return CliParamSpec(
        name=name,
        cli_name=cli_name,
        python_type=python_type,
        default=default,
        role=role,
        help_text=help_text,
        help_panel=help_panel,
        is_optional=is_optional,
    )


# -----------------------------------------------------------------------------
# Signature Introspection
# -----------------------------------------------------------------------------

# Map operation backend_method to Protocol class + method
# This avoids import cycles by using string-based lookup
_BACKEND_METHOD_TO_API: dict[str, tuple[str, str]] = {
    "get_function_summary": ("FunctionQueriesApi", "get_function_summary"),
    "list_high_risk_functions": ("FunctionQueriesApi", "list_high_risk_functions"),
    "get_callgraph_neighbors": ("FunctionQueriesApi", "get_callgraph_neighbors"),
    "get_tests_for_function": ("FunctionQueriesApi", "get_tests_for_function"),
    "get_callgraph_neighborhood": ("FunctionQueriesApi", "get_callgraph_neighborhood"),
    "get_import_boundary": ("FunctionQueriesApi", "get_import_boundary"),
    "get_function_profile": ("FunctionQueriesApi", "get_function_profile"),
    "get_function_architecture": ("FunctionQueriesApi", "get_function_architecture"),
    "get_file_summary": ("ProfileQueriesApi", "get_file_summary"),
    "get_file_profile": ("ProfileQueriesApi", "get_file_profile"),
    "get_module_profile": ("ProfileQueriesApi", "get_module_profile"),
    "get_module_architecture": ("ProfileQueriesApi", "get_module_architecture"),
    "get_file_hints": ("ProfileQueriesApi", "get_file_hints"),
    "list_subsystems": ("SubsystemQueriesApi", "list_subsystems"),
    "get_module_subsystems": ("SubsystemQueriesApi", "get_module_subsystems"),
    "get_subsystem_modules": ("SubsystemQueriesApi", "get_subsystem_modules"),
    "search_subsystems": ("SubsystemQueriesApi", "search_subsystems"),
    "summarize_subsystem": ("SubsystemQueriesApi", "summarize_subsystem"),
    "list_subsystem_profiles": ("SubsystemQueriesApi", "list_subsystem_profiles"),
    "list_subsystem_coverage": ("SubsystemQueriesApi", "list_subsystem_coverage"),
    "list_datasets": ("DatasetQueriesApi", "list_datasets"),
    "dataset_specs": ("DatasetQueriesApi", "dataset_specs"),
    "read_dataset_rows": ("DatasetQueriesApi", "read_dataset_rows"),
    "dataset_schema": ("DatasetQueriesApi", "dataset_schema"),
}


def _get_type_hints_safe(method: object) -> dict[str, Any]:
    """Get type hints safely, returning empty dict on failure.

    Parameters
    ----------
    method
        Method to introspect.

    Returns
    -------
    dict[str, Any]
        Type hints or empty dict if introspection fails.
    """
    try:
        return get_type_hints(method)
    except (NameError, AttributeError, TypeError):
        # NameError: forward reference not resolvable
        # AttributeError: method doesn't support type hints
        # TypeError: invalid type hint
        return {}


def get_backend_signature_for_operation(
    operation: Operation,
) -> tuple[inspect.Signature, dict[str, Any]] | None:
    """Get the signature and type hints for an operation's backend method.

    Parameters
    ----------
    operation
        Operation to introspect.

    Returns
    -------
    tuple[inspect.Signature, dict[str, Any]] | None
        Tuple of (signature, type_hints) or None if not found.
    """
    method_name = operation.backend_method
    api_info = _BACKEND_METHOD_TO_API.get(method_name)

    if api_info is None:
        LOG.debug("No API mapping for backend_method=%s", method_name)
        return None

    api_class_name, _method = api_info

    api_class = getattr(query_api, api_class_name, None)
    if api_class is None:
        LOG.debug("API class %s not found in query_api", api_class_name)
        return None

    method = getattr(api_class, method_name, None)
    if method is None:
        LOG.debug("Method %s not found on %s", method_name, api_class_name)
        return None

    sig = inspect.signature(method)
    hints = _get_type_hints_safe(method)

    return sig, hints


def build_cli_param_specs_for_operation(
    operation: Operation,
) -> Sequence[CliParamSpec]:
    """Build CLI parameter specifications for an operation.

    Parameters
    ----------
    operation
        Operation to generate parameter specs for.

    Returns
    -------
    Sequence[CliParamSpec]
        Ordered list of parameter specifications.
    """
    result = get_backend_signature_for_operation(operation)
    if result is None:
        return []

    sig, hints = result
    specs: list[CliParamSpec] = []

    for param in sig.parameters.values():
        # Skip self parameter
        if param.name == "self":
            continue
        spec = build_cli_param_spec(param, hints, operation)
        specs.append(spec)

    # Sort by role: selectors first, then filters, then advanced
    role_order: dict[ParamRole, int] = {"selector": 0, "filter": 1, "advanced": 2}
    specs.sort(key=lambda s: (role_order[s.role], s.name))

    return specs


# -----------------------------------------------------------------------------
# Operation Metadata
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class OperationCliMetadata:
    """CLI metadata for an operation including parameters and help text.

    Attributes
    ----------
    operation
        Source operation from the catalog.
    cli_name
        CLI command name (operation.id with dots replaced by hyphens).
    params
        CLI parameter specifications.
    help_text
        Generated help text for the command.
    """

    operation: Operation
    cli_name: str
    params: tuple[CliParamSpec, ...]
    help_text: str


def _make_operation_cli_name(op_id: str) -> str:
    """Convert operation ID to CLI command name.

    Parameters
    ----------
    op_id
        Operation identifier like "function.summary".

    Returns
    -------
    str
        CLI command name like "function-summary".
    """
    return op_id.replace(".", "-").replace("_", "-")


def build_operation_cli_metadata(operation: Operation) -> OperationCliMetadata:
    """Build complete CLI metadata for an operation.

    Parameters
    ----------
    operation
        Operation from the catalog.

    Returns
    -------
    OperationCliMetadata
        Complete CLI metadata including parameters.
    """
    cli_name = _make_operation_cli_name(operation.id)
    params = tuple(build_cli_param_specs_for_operation(operation))

    # Build help text
    help_lines = [operation.summary]
    if operation.description:
        help_lines.append("")
        help_lines.append(operation.description)

    if operation.required_datasets:
        help_lines.append("")
        help_lines.append(f"Required datasets: {', '.join(operation.required_datasets)}")
    if operation.required_graphs:
        help_lines.append("")
        help_lines.append(f"Required graphs: {', '.join(operation.required_graphs)}")

    help_text = "\n".join(help_lines)

    return OperationCliMetadata(
        operation=operation,
        cli_name=cli_name,
        params=params,
        help_text=help_text,
    )


# -----------------------------------------------------------------------------
# CLI Type Conversion
# -----------------------------------------------------------------------------


_BASIC_TYPES: frozenset[type[Any]] = frozenset({int, float, bool, str})
"""Basic Python types that map directly to Typer types."""


def _get_typer_type(spec: CliParamSpec) -> type[Any]:
    """Get the Typer-compatible type for a parameter spec.

    Parameters
    ----------
    spec
        CLI parameter specification.

    Returns
    -------
    type
        Type suitable for Typer Option/Argument.
    """
    base_type = _extract_base_type(spec.python_type)

    if base_type is None:
        return str
    if base_type in _BASIC_TYPES:
        return base_type

    # For complex types, use string and let the backend handle conversion
    return str


# -----------------------------------------------------------------------------
# Command Factory
# -----------------------------------------------------------------------------


@dataclass
class DynamicCommandConfig:
    """Configuration for dynamic command generation.

    Attributes
    ----------
    skip_prereqs
        If True, skip prerequisite pipeline execution.
    verbose
        If True, enable verbose output.
    project_root
        Optional explicit project root path.
    """

    skip_prereqs: bool = False
    verbose: bool = False
    project_root: str | None = None


def get_operations_with_cli_support() -> list[Operation]:
    """Get all operations that support CLI invocation.

    Returns operations that have a defined http_path or tool_name,
    indicating they are user-facing operations suitable for CLI.

    Returns
    -------
    list[Operation]
        Operations suitable for CLI command generation.
    """
    operations: list[Operation] = []
    for op in iter_operations():
        # Include operations with HTTP or MCP exposure
        if op.http_path is not None or op.tool_name is not None:
            # Skip health and meta operations
            if op.category in {"health"}:
                continue
            operations.append(op)
    return operations


# -----------------------------------------------------------------------------
# String Tunnel: Runtime Type Coercion
# -----------------------------------------------------------------------------


def coerce_string_param(
    value: str,
    target_type: type[object] | None,
) -> str | int | float | bool:
    """Coerce a string CLI parameter to its target Python type.

    Parameters
    ----------
    value
        String value from CLI.
    target_type
        Target Python type (may be Optional[X]).

    Returns
    -------
    str | int | float | bool
        Value coerced to the target type.
    """
    if target_type is None:
        return value

    base_type = _extract_base_type(target_type)

    if base_type is None or base_type is str:
        return value

    if base_type is int:
        return int(value)

    if base_type is float:
        return float(value)

    if base_type is bool:
        return value.lower() in {"true", "1", "yes", "on"}

    # For complex types (like enums, dataclasses), return as string
    # and let the backend handle conversion
    return value


def coerce_params_from_strings(
    raw_params: dict[str, str | None],
    specs: tuple[CliParamSpec, ...],
) -> dict[str, Any]:
    """Coerce string CLI parameters to their proper Python types.

    Parameters
    ----------
    raw_params
        Raw string parameters from CLI (may include None values).
    specs
        Parameter specifications with type information.

    Returns
    -------
    dict[str, Any]
        Parameters coerced to their proper types.
        None values are excluded from the result.
    """
    spec_map = {s.name: s for s in specs}
    result: dict[str, Any] = {}

    for name, value in raw_params.items():
        if value is None:
            continue

        spec = spec_map.get(name)
        if spec is None:
            # Unknown parameter - pass through as string
            result[name] = value
            continue

        try:
            result[name] = coerce_string_param(value, spec.python_type)
        except (ValueError, TypeError) as exc:
            LOG.warning("Failed to coerce %s=%r: %s", name, value, exc)
            result[name] = value

    return result


# -----------------------------------------------------------------------------
# Dynamic Command Factory
# -----------------------------------------------------------------------------

# Type aliases for common CLI options (Typer-compatible single-type unions)
_ProjectRootOpt = Annotated[
    Path | None,
    typer.Option("--root", "-r", help="Explicit project root directory"),
]
_SkipPrereqsOpt = Annotated[
    bool | None,
    typer.Option("--skip-prereqs", help="Skip prerequisite pipeline", is_flag=True),
]
_VerboseOpt = Annotated[
    bool | None,
    typer.Option("--verbose", "-v", help="Enable verbose output", is_flag=True),
]


class OperationInvokeCallback(Protocol):
    """Protocol for operation invocation callbacks.

    This protocol defines the signature for callbacks that execute operations
    from dynamically registered CLI commands. The boolean parameters are
    keyword-only to satisfy lint rules about boolean positional arguments.
    """

    def __call__(
        self,
        op_id: str,
        params: dict[str, Any],
        project_root: Path | None,
        *,
        skip_prereqs: bool,
        verbose: bool,
    ) -> None:
        """Invoke an operation.

        Parameters
        ----------
        op_id
            Operation identifier.
        params
            Operation parameters (coerced to proper types).
        project_root
            Optional project root path.
        skip_prereqs
            Whether to skip prerequisite pipeline execution.
        verbose
            Whether to enable verbose output.
        """
        ...


def _make_typer_option(spec: CliParamSpec) -> typer.models.OptionInfo:
    """Create a Typer Option for a parameter spec.

    Parameters
    ----------
    spec
        CLI parameter specification.

    Returns
    -------
    typer.models.OptionInfo
        Typer option configuration.
    """
    return typer.Option(
        f"--{spec.cli_name}",
        help=spec.help_text,
        rich_help_panel=spec.help_panel,
    )


def build_dynamic_command(
    metadata: OperationCliMetadata,
    invoke_callback: OperationInvokeCallback,
) -> Callable[..., None]:
    """Build a dynamic Typer command for an operation.

    Creates a command function with explicit parameters (not **kwargs) where
    all operation-specific parameters are typed as Optional[str]. Type coercion
    happens at runtime based on the parameter specifications.

    Parameters
    ----------
    metadata
        CLI metadata for the operation.
    invoke_callback
        Callback function to invoke the operation.
        Signature: (op_id, params, project_root, skip_prereqs, verbose) -> None

    Returns
    -------
    Callable[..., None]
        Command function suitable for Typer registration.
    """
    op = metadata.operation
    specs = metadata.params

    # Build the parameter list for the signature
    # Start with base parameters that every command has
    sig_params: list[inspect.Parameter] = [
        inspect.Parameter(
            "project_root",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=_ProjectRootOpt,
        ),
        inspect.Parameter(
            "skip_prereqs",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=_SkipPrereqsOpt,
        ),
        inspect.Parameter(
            "verbose",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=_VerboseOpt,
        ),
    ]

    # Add operation-specific parameters - ALL as Annotated[str | None, Option]
    for spec in specs:
        param_annotation = Annotated[
            str | None,
            typer.Option(
                f"--{spec.cli_name}",
                help=spec.help_text,
                rich_help_panel=spec.help_panel,
            ),
        ]
        sig_params.append(
            inspect.Parameter(
                spec.name,
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=param_annotation,
            )
        )

    # Create the actual command function
    # Use a factory to capture specs and op in closure
    def make_command() -> Callable[..., None]:
        # Capture these in closure
        captured_op_id = op.id
        captured_specs = specs
        captured_invoke = invoke_callback
        captured_help = metadata.help_text

        def command(
            project_root: _ProjectRootOpt = None,
            skip_prereqs: _SkipPrereqsOpt = None,
            verbose: _VerboseOpt = None,
            **op_params: str | None,
        ) -> None:
            # Extract operation-specific params from kwargs
            # Coerce string values to proper types
            coerced = coerce_params_from_strings(op_params, captured_specs)

            # Resolve flags
            skip_prereqs_flag = skip_prereqs if skip_prereqs is not None else False
            verbose_flag = verbose if verbose is not None else False

            # Invoke the operation
            captured_invoke(
                captured_op_id,
                coerced,
                project_root,
                skip_prereqs=skip_prereqs_flag,
                verbose=verbose_flag,
            )

        # Set the docstring
        command.__doc__ = captured_help
        return command

    # Create and configure the command
    cmd = make_command()
    cmd.__name__ = f"op_{op.id.replace('.', '_')}"

    # Attach the synthetic signature so Typer sees explicit parameters
    cmd.__signature__ = inspect.Signature(sig_params)  # type: ignore[attr-defined]

    return cmd


def register_dynamic_commands(
    app: typer.Typer,
    invoke_callback: OperationInvokeCallback,
) -> int:
    """Register dynamic commands for all CLI-supported operations.

    Parameters
    ----------
    app
        Typer application to register commands on.
    invoke_callback
        Callback to invoke operations.
        Signature: (op_id, params, project_root, skip_prereqs, verbose) -> None

    Returns
    -------
    int
        Number of commands registered.
    """
    operations = get_operations_with_cli_support()
    count = 0

    for op in operations:
        metadata = build_operation_cli_metadata(op)
        command = build_dynamic_command(metadata, invoke_callback)

        # Register with Typer
        app.command(
            name=metadata.cli_name,
            help=op.summary,
        )(command)
        count += 1

    LOG.debug("Registered %d dynamic operation commands", count)
    return count


__all__ = [
    "CliParamSpec",
    "DynamicCommandConfig",
    "OperationCliMetadata",
    "ParamRole",
    "build_cli_param_spec",
    "build_cli_param_specs_for_operation",
    "build_dynamic_command",
    "build_operation_cli_metadata",
    "classify_param_role",
    "coerce_params_from_strings",
    "coerce_string_param",
    "get_backend_signature_for_operation",
    "get_help_panel_for_role",
    "get_operations_with_cli_support",
    "register_dynamic_commands",
]
