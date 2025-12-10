"""Compatibility shim for shell module.

.. deprecated::
    This module is deprecated. Import from ``codeintel.cli.shell`` package instead.
    This shim will be removed in a future version.

Example migration::

    # Old (deprecated):
    from codeintel.cli.shell import InteractiveShell, start_shell

    # New (preferred):
    from codeintel.cli.shell import InteractiveShell, start_shell
    # (import from the package, not this file)
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'codeintel.cli.shell' (module) is deprecated. "
    "The shell package now provides these exports directly. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from codeintel.cli.shell._shell import (
    InteractiveShell,
    ShellCompleter,
    ShellSession,
    start_shell,
)

__all__ = [
    "InteractiveShell",
    "ShellCompleter",
    "ShellSession",
    "start_shell",
]
