"""Background job management commands.

Provide commands to submit, monitor, and manage background jobs
for long-running operations.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Annotated, Literal

from cyclopts import App, Parameter
from rich.console import Console
from rich.table import Table

from codeintel.cli.jobs import JobStatus, get_job_manager

jobs_app = App(name="jobs", help="Manage background jobs")


@jobs_app.command(name="list")
@dataclass
class JobsListCommand:
    """List background jobs.

    Display a table of all background jobs with their status,
    operation, and timestamps.
    """

    status: Annotated[
        Literal["pending", "running", "completed", "failed", "cancelled"] | None,
        Parameter(help="Filter by status"),
    ] = None
    limit: Annotated[int, Parameter(help="Maximum jobs to show")] = 20
    output_format: Annotated[
        Literal["text", "json"],
        Parameter(name="--format", help="Output format"),
    ] = "text"

    def __call__(self) -> None:
        """Execute the jobs list command."""
        manager = get_job_manager()
        status_filter = JobStatus(self.status) if self.status else None
        jobs = manager.list_jobs(status=status_filter, limit=self.limit)

        if self.output_format == "json":
            sys.stdout.write(json.dumps([j.to_dict() for j in jobs], indent=2))
            sys.stdout.write("\n")
            return

        if not jobs:
            sys.stdout.write("No jobs found\n")
            return

        console = Console()
        table = Table(title="Background Jobs")
        table.add_column("Job ID", style="cyan")
        table.add_column("Operation")
        table.add_column("Status")
        table.add_column("Created")

        status_styles = {
            JobStatus.PENDING: "yellow",
            JobStatus.RUNNING: "blue",
            JobStatus.COMPLETED: "green",
            JobStatus.FAILED: "red",
            JobStatus.CANCELLED: "dim",
        }

        for job in jobs:
            style = status_styles.get(job.status, "")
            table.add_row(
                job.job_id,
                job.operation_id,
                f"[{style}]{job.status.value}[/{style}]",
                job.created_at[:19] if job.created_at else "",
            )

        console.print(table)


@jobs_app.command(name="status")
@dataclass
class JobsStatusCommand:
    """Get status of a background job.

    Display detailed status information for a specific job
    including timestamps and error messages.
    """

    job_id: Annotated[str, Parameter(help="Job ID")]
    output_format: Annotated[
        Literal["text", "json"],
        Parameter(name="--format", help="Output format"),
    ] = "text"

    def __call__(self) -> None:
        """Execute the jobs status command.

        Raises
        ------
        SystemExit
            If the job is not found.
        """
        manager = get_job_manager()
        job = manager.get_status(self.job_id)

        if job is None:
            sys.stdout.write(f"Job not found: {self.job_id}\n")
            raise SystemExit(1)

        if self.output_format == "json":
            sys.stdout.write(json.dumps(job.to_dict(), indent=2))
            sys.stdout.write("\n")
            return

        console = Console()
        console.print(f"[bold]Job ID:[/bold] {job.job_id}")
        console.print(f"[bold]Operation:[/bold] {job.operation_id}")
        console.print(f"[bold]Status:[/bold] {job.status.value}")
        console.print(f"[bold]Created:[/bold] {job.created_at}")

        if job.started_at:
            console.print(f"[bold]Started:[/bold] {job.started_at}")
        if job.completed_at:
            console.print(f"[bold]Completed:[/bold] {job.completed_at}")
        if job.error:
            console.print(f"[bold red]Error:[/bold red] {job.error}")


@jobs_app.command(name="output")
@dataclass
class JobsOutputCommand:
    """Get output of a completed job.

    Retrieve and display the result data from a completed
    background job.
    """

    job_id: Annotated[str, Parameter(help="Job ID")]

    def __call__(self) -> None:
        """Execute the jobs output command.

        Raises
        ------
        SystemExit
            If the job is not found or not completed.
        """
        manager = get_job_manager()
        job = manager.get_status(self.job_id)

        if job is None:
            sys.stdout.write(f"Job not found: {self.job_id}\n")
            raise SystemExit(1)

        if job.status != JobStatus.COMPLETED:
            sys.stdout.write(f"Job is not completed (status: {job.status.value})\n")
            raise SystemExit(1)

        result = manager.get_output(self.job_id)
        if result:
            sys.stdout.write(json.dumps(result, indent=2))
            sys.stdout.write("\n")
        else:
            sys.stdout.write("No output available\n")


@jobs_app.command(name="cancel")
@dataclass
class JobsCancelCommand:
    """Cancel a running job.

    Send a termination signal to a running job and mark
    it as cancelled.
    """

    job_id: Annotated[str, Parameter(help="Job ID")]

    def __call__(self) -> None:
        """Execute the jobs cancel command.

        Raises
        ------
        SystemExit
            If the job could not be cancelled.
        """
        manager = get_job_manager()

        if manager.cancel(self.job_id):
            sys.stdout.write(f"Job {self.job_id} cancelled\n")
        else:
            sys.stdout.write(f"Could not cancel job {self.job_id}\n")
            raise SystemExit(1)


@jobs_app.command(name="cleanup")
@dataclass
class JobsCleanupCommand:
    """Clean up old completed jobs.

    Remove job metadata and output files for jobs that
    completed more than the specified number of days ago.
    """

    max_age_days: Annotated[int, Parameter(help="Maximum age in days")] = 7

    def __call__(self) -> None:
        """Execute the jobs cleanup command."""
        manager = get_job_manager()
        cleaned = manager.cleanup(max_age_days=self.max_age_days)
        sys.stdout.write(f"Cleaned up {cleaned} jobs\n")


__all__ = [
    "jobs_app",
]
