"""Compatibility shim for jobs module.

.. deprecated::
    This module is deprecated. Import from ``codeintel.cli.jobs`` package instead.
    This shim will be removed in a future version.

Example migration::

    # Old (deprecated):
    from codeintel.cli.jobs import JobManager, get_job_manager

    # New (preferred):
    from codeintel.cli.jobs import JobManager, get_job_manager
    # (import from the package, not this file)
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'codeintel.cli.jobs' (module) is deprecated. "
    "The jobs package now provides these exports directly. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from codeintel.cli.jobs._jobs import (
    JobInfo,
    JobManager,
    JobStatus,
    JobStore,
    get_job_manager,
)

__all__ = [
    "JobInfo",
    "JobManager",
    "JobStatus",
    "JobStore",
    "get_job_manager",
]
