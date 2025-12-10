"""Compatibility shim for results module.

.. deprecated::
    This module is deprecated. Import from ``codeintel.cli.core`` instead.
    This shim will be removed in a future version.

Example migration::

    # Old (deprecated):
    from codeintel.cli.results import CliResult

    # New (preferred):
    from codeintel.cli.core import CliResult
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'codeintel.cli.results' is deprecated. "
    "Use 'codeintel.cli.core' instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from codeintel.cli.core.results import CliResult, TextRenderer

__all__ = [
    "CliResult",
    "TextRenderer",
]
