"""CLI and operation introspection infrastructure.

This package provides:

- ``OperationRegistry``: Central registry for handler-based operations
- ``CliParamSpec``: Parameter introspection for operations
- ``Validator``: Input validation framework
- ``HelpRenderer``: Help system utilities
- Operation discovery and search
"""

from __future__ import annotations

# Registry from execution layer
from codeintel.cli.execution.registry import (
    OperationRegistry,
    OperationSpec,
    create_spec_from_serving_operation,
    get_registry,
    register_operation,
    reset_registry,
)
from codeintel.cli.introspection.discovery import (
    get_operation_info,
    list_all_operations,
    list_operations_by_group,
    search_operations,
)
from codeintel.cli.introspection.help import (
    HelpRenderer,
    get_help_renderer,
)
from codeintel.cli.introspection.params import (
    CliParamSpec,
    OperationCliMetadata,
    ParamRole,
    build_cli_param_spec,
    build_cli_param_specs_for_operation,
    build_operation_cli_metadata,
    classify_param_role,
    coerce_params_from_strings,
    coerce_string_param,
    get_backend_signature_for_operation,
    get_help_panel_for_role,
    get_operations_with_cli_support,
)
from codeintel.cli.introspection.validation import (
    IntValidator,
    StringValidator,
    ValidationError,
    ValidationResult,
    ValidationSchema,
    Validator,
)

__all__ = [
    "CliParamSpec",
    "HelpRenderer",
    "IntValidator",
    "OperationCliMetadata",
    "OperationRegistry",
    "OperationSpec",
    "ParamRole",
    "StringValidator",
    "ValidationError",
    "ValidationResult",
    "ValidationSchema",
    "Validator",
    "build_cli_param_spec",
    "build_cli_param_specs_for_operation",
    "build_operation_cli_metadata",
    "classify_param_role",
    "coerce_params_from_strings",
    "coerce_string_param",
    "create_spec_from_serving_operation",
    "get_backend_signature_for_operation",
    "get_help_panel_for_role",
    "get_help_renderer",
    "get_operation_info",
    "get_operations_with_cli_support",
    "get_registry",
    "list_all_operations",
    "list_operations_by_group",
    "register_operation",
    "reset_registry",
    "search_operations",
]
