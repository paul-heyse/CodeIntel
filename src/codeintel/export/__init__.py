"""DEPRECATED: Use codeintel.build.exports instead.

This module is deprecated and will be removed in a future version.
All export functionality has been consolidated into the build system.

Migration guide:

Old import:
    from codeintel.export.export_jsonl import export_all_jsonl
    from codeintel.export.export_parquet import export_all_parquet
    from codeintel.export.runner import run_validated_exports

New import:
    from codeintel.build.exports import (
        export_all_jsonl,
        export_all_parquet,
        run_validated_exports,
    )
"""

from __future__ import annotations

import warnings

# Re-export from build.exports for backward compatibility
from codeintel.build.exports import (
    ExportCallOptions,
    ExportError,
    ExportOptions,
    default_validation_schemas,
    export_all_jsonl,
    export_all_parquet,
    run_validated_exports,
    validate_export_files,
)


def __getattr__(name: str) -> object:
    """Emit deprecation warning for any access to this module.

    Parameters
    ----------
    name
        Attribute name being accessed.

    Raises
    ------
    AttributeError
        Always raised since unknown attributes don't exist.
    """
    warnings.warn(
        f"codeintel.export is deprecated. Use codeintel.build.exports instead. "
        f"Accessed: codeintel.export.{name}",
        DeprecationWarning,
        stacklevel=2,
    )
    msg = f"module 'codeintel.export' has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "ExportCallOptions",
    "ExportError",
    "ExportOptions",
    "default_validation_schemas",
    "export_all_jsonl",
    "export_all_parquet",
    "run_validated_exports",
    "validate_export_files",
]
