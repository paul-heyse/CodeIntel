"""Compatibility shim for cli_types module.

.. deprecated::
    This module is deprecated. Import types from their canonical locations:
    - ``OutputFormat`` from ``codeintel.cli.rendering.types``
    - ``BackendFlags``, ``RuntimeParams`` from ``codeintel.cli.resolution.params``
    This shim will be removed in a future version.

Example migration::

    # Old (deprecated):
    from codeintel.cli.cli_types import OutputFormat, BackendFlags

    # New (preferred):
    from codeintel.cli.rendering.types import OutputFormat
    from codeintel.cli.resolution.params import BackendFlags
"""

from __future__ import annotations

import warnings
from typing import Literal

warnings.warn(
    "Importing from 'codeintel.cli.cli_types' is deprecated. "
    "Use 'codeintel.cli.rendering.types' for OutputFormat and "
    "'codeintel.cli.resolution.params' for BackendFlags/RuntimeParams. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical locations
from codeintel.cli.rendering.types import OutputFormat
from codeintel.cli.resolution.params import BackendFlags, RuntimeParams

# Type alias for help level (kept here as it's simple)
HelpLevel = Literal["brief", "full"]

# Backward compatibility alias
RuntimeOptions = RuntimeParams

__all__ = [
    "BackendFlags",
    "HelpLevel",
    "OutputFormat",
    "RuntimeOptions",
    "RuntimeParams",
]
