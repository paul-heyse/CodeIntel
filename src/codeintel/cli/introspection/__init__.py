"""CLI and operation introspection infrastructure.

This package provides:

- ``OperationRegistry``: Central registry for handler-based operations
- ``CliParamSpec``: Parameter introspection for operations
- ``Validator``: Input validation framework
- ``HelpRenderer``: Help system utilities
- Operation discovery and search

Note: Phase 6 Migration Complete - LEGACY registry has been removed.
All registry functions now use the unified handler-based registry from
``codeintel.cli.execution.registry``.
"""

from __future__ import annotations

# Provide backward-compatible legacy registry for old code paths
# Note: The legacy operations/*.py files have been deleted, so this registry
# is empty. Code using the old executor.execute() flow should be migrated to
# use the @cli_command decorator pattern instead.
from codeintel.cli.execution.executor import get_executor as _get_executor

# Registry is now unified in execution layer
from codeintel.cli.execution.registry import (
    OperationRegistry,
    OperationSpec,
    get_registry,
    register_operation,
    reset_registry,
)
from codeintel.cli.introspection.discovery import (
    OperationInfo,
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


def get_operation_registry() -> OperationRegistry:
    """Get the operation registry (legacy compatibility).

    Returns
    -------
    OperationRegistry
        The unified operation registry.

    Notes
    -----
    This function is provided for backward compatibility. New code should
    use ``get_registry()`` from ``codeintel.cli.execution.registry``.
    """
    return get_registry()


__all__ = [
    "CliParamSpec",
    "HelpRenderer",
    "IntValidator",
    "OperationCliMetadata",
    "OperationInfo",
    "OperationRegistry",
    "OperationSpec",
    "ParamRole",
    "StringValidator",
    "ValidationError",
    "ValidationResult",
    "ValidationSchema",
    "Validator",
    "_get_executor",
    "build_cli_param_spec",
    "build_cli_param_specs_for_operation",
    "build_operation_cli_metadata",
    "classify_param_role",
    "coerce_params_from_strings",
    "coerce_string_param",
    "get_backend_signature_for_operation",
    "get_help_panel_for_role",
    "get_help_renderer",
    "get_operation_info",
    "get_operation_registry",
    "get_operations_with_cli_support",
    "get_registry",
    "list_all_operations",
    "list_operations_by_group",
    "register_operation",
    "reset_registry",
    "search_operations",
]
