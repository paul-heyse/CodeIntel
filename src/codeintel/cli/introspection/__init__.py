"""CLI and operation introspection infrastructure.

This package provides:

- ``OperationRegistry``: Central registry for handler-based operations
- ``Validator``: Input validation framework
- ``HelpRenderer``: Help system utilities
- Operation discovery and search
"""

from __future__ import annotations

from codeintel.cli.execution.registry import (
    OperationRegistry,
    OperationSpec,
    get_registry,
    register_operation,
    reset_registry,
)
from codeintel.cli.introspection.discovery import (
    get_operation_info,
    list_all_aliases,
    list_all_operations,
    list_operation_aliases,
    list_operations_by_group,
    search_operations,
)
from codeintel.cli.introspection.help import (
    HelpRenderer,
    get_help_renderer,
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
    "HelpRenderer",
    "IntValidator",
    "OperationRegistry",
    "OperationSpec",
    "StringValidator",
    "ValidationError",
    "ValidationResult",
    "ValidationSchema",
    "Validator",
    "get_help_renderer",
    "get_operation_info",
    "get_registry",
    "list_all_aliases",
    "list_all_operations",
    "list_operation_aliases",
    "list_operations_by_group",
    "register_operation",
    "reset_registry",
    "search_operations",
]
