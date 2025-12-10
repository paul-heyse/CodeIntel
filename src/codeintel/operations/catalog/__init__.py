"""Operation catalog containing all operation implementations.

Operations are organized by domain (jobs, datasets, storage, etc.).
Each module registers its operations via the @operation decorator.
"""

from __future__ import annotations

# Import catalog modules to trigger operation registration
from codeintel.operations.catalog import jobs

# Re-export job operations for convenience
from codeintel.operations.catalog.jobs import (
    CancelJob,
    CleanupJobs,
    CleanupJobsParams,
    GetJobOutput,
    GetJobParams,
    GetJobStatus,
    JobCancelResult,
    JobInfo,
    JobOutputResult,
    JobsCleanupResult,
    JobsListResult,
    JobStatusResult,
    ListJobs,
    ListJobsParams,
)


def register_all_operations() -> None:
    """Import all catalog modules to register operations.

    Call this during application startup to ensure all operations
    are registered with the default registry.
    """
    # Import modules to trigger @operation decorator registration
    # These imports are already done at module level, but we include
    # this function for explicit initialization and future modules.
    _ = jobs  # Ensure import is used


__all__ = [
    "CancelJob",
    "CleanupJobs",
    "CleanupJobsParams",
    "GetJobOutput",
    "GetJobParams",
    "GetJobStatus",
    "JobCancelResult",
    "JobInfo",
    "JobOutputResult",
    "JobStatusResult",
    "JobsCleanupResult",
    "JobsListResult",
    "ListJobs",
    "ListJobsParams",
    "register_all_operations",
]
