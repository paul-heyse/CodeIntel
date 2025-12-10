"""Compatibility shim for help_system module.

.. deprecated::
    This module is deprecated. Import from ``codeintel.cli.introspection`` instead.
    This shim will be removed in a future version.

Example migration::

    # Old (deprecated):
    from codeintel.cli.help_system import HelpRenderer, get_help_renderer

    # New (preferred):
    from codeintel.cli.introspection import HelpRenderer, get_help_renderer
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'codeintel.cli.help_system' is deprecated. "
    "Use 'codeintel.cli.introspection' instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from codeintel.cli.introspection.help import (
    HelpRenderer,
    get_help_renderer,
)

__all__ = [
    "HelpRenderer",
    "get_help_renderer",
]
