"""Background job management infrastructure.

This package provides:

- ``JobManager``: Job lifecycle management
- ``JobStore``: Persistent job storage
- ``JobInfo``: Job metadata and status
- Job runner entry point
"""

from __future__ import annotations

# Job management
from codeintel.cli.jobs._jobs import (
    JobInfo,
    JobManager,
    JobStatus,
    JobStore,
    get_job_manager,
    run_job,
)

__all__ = [
    "JobInfo",
    "JobManager",
    "JobStatus",
    "JobStore",
    "get_job_manager",
    "run_job",
]
