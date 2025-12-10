"""Compatibility shim for op_params module.

.. deprecated::
    This module is deprecated. Import from ``codeintel.cli.introspection`` instead.
    This shim will be removed in a future version.

Example migration::

    # Old (deprecated):
    from codeintel.cli.op_params import CliParamSpec, build_cli_param_spec

    # New (preferred):
    from codeintel.cli.introspection import CliParamSpec, build_cli_param_spec
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'codeintel.cli.op_params' is deprecated. "
    "Use 'codeintel.cli.introspection' instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
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

__all__ = [
    "CliParamSpec",
    "OperationCliMetadata",
    "ParamRole",
    "build_cli_param_spec",
    "build_cli_param_specs_for_operation",
    "build_operation_cli_metadata",
    "classify_param_role",
    "coerce_params_from_strings",
    "coerce_string_param",
    "get_backend_signature_for_operation",
    "get_help_panel_for_role",
    "get_operations_with_cli_support",
]
