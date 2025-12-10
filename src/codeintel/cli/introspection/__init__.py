"""CLI and operation introspection infrastructure.

This package provides:

- ``OperationRegistry``: Central registry for operations
- ``CliParamSpec``: Parameter introspection for operations
- ``Validator``: Input validation framework
- ``HelpRenderer``: Help system utilities
- Operation discovery and search

Note: This module exports from the LEGACY registry for backward compatibility.
New handler-based code should use ``codeintel.cli.execution.registry`` directly
for the new ``OperationSpec`` and ``OperationRegistry`` types.
"""

from __future__ import annotations

# Also expose new registry functions under different names for gradual migration
from codeintel.cli.execution.registry import (
    OperationRegistry as HandlerOperationRegistry,
)
from codeintel.cli.execution.registry import (
    OperationSpec as HandlerOperationSpec,
)
from codeintel.cli.execution.registry import (
    get_registry as get_handler_registry,
)
from codeintel.cli.execution.registry import (
    register_operation as register_handler_operation,
)
from codeintel.cli.execution.registry import (
    reset_registry as reset_handler_registry,
)
from codeintel.cli.introspection.discovery import (
    OperationInfo,
    get_operation_info,
    get_operation_schema,
    list_all_operations,
    list_operations_by_category,
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

# Re-export from legacy registry for backward compatibility
from codeintel.cli.introspection.registry import (
    OperationRegistry,
    get_operation_registry,
    register_operation,
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
    "HandlerOperationRegistry",
    "HandlerOperationSpec",
    "HelpRenderer",
    "IntValidator",
    "OperationCliMetadata",
    "OperationInfo",
    "OperationRegistry",
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
    "get_backend_signature_for_operation",
    "get_handler_registry",
    "get_help_panel_for_role",
    "get_help_renderer",
    "get_operation_info",
    "get_operation_registry",
    "get_operation_schema",
    "get_operations_with_cli_support",
    "list_all_operations",
    "list_operations_by_category",
    "register_handler_operation",
    "register_operation",
    "reset_handler_registry",
    "search_operations",
]
