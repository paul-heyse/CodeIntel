"""Compatibility shim for dry_run module.

.. deprecated::
    This module is deprecated. Import from ``codeintel.cli.project`` instead.
    This shim will be removed in a future version.

Example migration::

    # Old (deprecated):
    from codeintel.cli.dry_run import plan_dry_run, render_dry_run

    # New (preferred):
    from codeintel.cli.project import plan_dry_run, render_dry_run
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'codeintel.cli.dry_run' is deprecated. "
    "Use 'codeintel.cli.project' instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from codeintel.cli.project.dry_run import (
    plan_dry_run,
    render_dry_run,
    render_dry_run_to,
)

__all__ = [
    "plan_dry_run",
    "render_dry_run",
    "render_dry_run_to",
]
