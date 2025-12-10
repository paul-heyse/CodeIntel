"""Compatibility shim for job_runner module.

.. deprecated::
    This module is deprecated. Import from ``codeintel.cli.jobs`` instead.
    This shim will be removed in a future version.

Example migration::

    # Old (deprecated):
    from codeintel.cli.job_runner import run_job

    # New (preferred):
    from codeintel.cli.jobs import run_job
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'codeintel.cli.job_runner' is deprecated. "
    "Use 'codeintel.cli.jobs' instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from codeintel.cli.jobs.runner import run_job

__all__ = [
    "run_job",
]
