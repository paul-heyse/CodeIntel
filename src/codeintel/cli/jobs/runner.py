"""Background job runner process.

Execute a job in a subprocess, updating status and storing results
in the job store. This module is invoked as a subprocess by the
JobManager.
"""

from __future__ import annotations

import argparse
import sys

from codeintel.cli.jobs._jobs import run_job


def main() -> None:
    """Run a background job.

    Parse the job ID from command line, load the job metadata,
    execute the operation, and update the job status.
    """
    parser = argparse.ArgumentParser(description="Run a background job")
    parser.add_argument("--job-id", required=True, help="Job ID to execute")
    args = parser.parse_args()

    exit_code = run_job(args.job_id)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
