"""Background job runner process.

Execute a job in a subprocess, updating status and storing results
in the job store. This module is invoked as a subprocess by the
JobManager.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from codeintel.cli.execution import get_executor
from codeintel.cli.introspection import get_operation_registry
from codeintel.cli.jobs import JobStatus, JobStore


def main() -> None:
    """Run a background job.

    Parse the job ID from command line, load the job metadata,
    execute the operation, and update the job status.
    """
    parser = argparse.ArgumentParser(description="Run a background job")
    parser.add_argument("--job-id", required=True, help="Job ID to execute")
    args = parser.parse_args()

    store = JobStore()
    job = store.load(args.job_id)

    if job is None:
        sys.exit(1)

    registry = get_operation_registry()
    spec = registry.get(job.operation_id)

    if spec is None:
        job.status = JobStatus.FAILED
        job.error = f"Unknown operation: {job.operation_id}"
        job.completed_at = datetime.now(UTC).isoformat()
        store.save(job)
        sys.exit(1)

    executor = get_executor()

    try:
        result = executor.execute(spec, job.params, render=False)

        if result.result.success:
            job.status = JobStatus.COMPLETED
            store.save_output(job.job_id, result.result.to_dict())
        else:
            job.status = JobStatus.FAILED
            error_detail = ""
            if result.result.error:
                error_detail = result.result.error.detail or "Unknown error"
            job.error = error_detail

        job.exit_code = 0 if result.result.success else 1

    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.exit_code = 1

    job.completed_at = datetime.now(UTC).isoformat()
    store.save(job)

    sys.exit(job.exit_code or 0)


if __name__ == "__main__":
    main()
