"""Compatibility shim for output module.

.. deprecated::
    This module is deprecated. Import from ``codeintel.cli.core`` instead.
    This shim will be removed in a future version.

Example migration::

    # Old (deprecated):
    from codeintel.cli.output import OutputEnvelope, iter_stdin_records

    # New (preferred):
    from codeintel.cli.core import OutputEnvelope, iter_stdin_records
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'codeintel.cli.output' is deprecated. "
    "Use 'codeintel.cli.core' instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from codeintel.cli.core.output import (
    OutputEnvelope,
    iter_stdin_records,
    merge_stdin_with_args,
    read_stdin_records,
)

__all__ = [
    "OutputEnvelope",
    "iter_stdin_records",
    "merge_stdin_with_args",
    "read_stdin_records",
]
