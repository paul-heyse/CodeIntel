"""Background job management commands.

Provide commands to submit, monitor, and manage background jobs
for long-running operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from cyclopts import App, Parameter

from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.jobs import (
    jobs_cancel_handler,
    jobs_cleanup_handler,
    jobs_list_handler,
    jobs_output_handler,
    jobs_status_handler,
)
from codeintel.cli.rendering.types import OutputFormat

jobs_app = App(name="jobs", help="Manage background jobs")

# Config for jobs commands - no runtime or gateway needed
_JOBS_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)


@cli_command("jobs.list", handler=jobs_list_handler, config=_JOBS_CONFIG)
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
        OutputFormat,
        Parameter(name="--format", help="Output format"),
    ] = OutputFormat.TEXT
    verbose: Annotated[int, Parameter(name="-v", count=True, help="Verbosity level")] = 0


@cli_command("jobs.status", handler=jobs_status_handler, config=_JOBS_CONFIG)
@jobs_app.command(name="status")
@dataclass
class JobsStatusCommand:
    """Get status of a background job.

    Display detailed status information for a specific job
    including timestamps and error messages.
    """

    job_id: Annotated[str, Parameter(help="Job ID")]
    output_format: Annotated[
        OutputFormat,
        Parameter(name="--format", help="Output format"),
    ] = OutputFormat.TEXT
    verbose: Annotated[int, Parameter(name="-v", count=True, help="Verbosity level")] = 0


@cli_command("jobs.output", handler=jobs_output_handler, config=_JOBS_CONFIG)
@jobs_app.command(name="output")
@dataclass
class JobsOutputCommand:
    """Get output of a completed job.

    Retrieve and display the result data from a completed
    background job.
    """

    job_id: Annotated[str, Parameter(help="Job ID")]
    output_format: Annotated[
        OutputFormat,
        Parameter(name="--format", help="Output format"),
    ] = OutputFormat.TEXT
    verbose: Annotated[int, Parameter(name="-v", count=True, help="Verbosity level")] = 0


@cli_command("jobs.cancel", handler=jobs_cancel_handler, config=_JOBS_CONFIG)
@jobs_app.command(name="cancel")
@dataclass
class JobsCancelCommand:
    """Cancel a running job.

    Send a termination signal to a running job and mark
    it as cancelled.
    """

    job_id: Annotated[str, Parameter(help="Job ID")]
    output_format: Annotated[
        OutputFormat,
        Parameter(name="--format", help="Output format"),
    ] = OutputFormat.TEXT
    verbose: Annotated[int, Parameter(name="-v", count=True, help="Verbosity level")] = 0


@cli_command("jobs.cleanup", handler=jobs_cleanup_handler, config=_JOBS_CONFIG)
@jobs_app.command(name="cleanup")
@dataclass
class JobsCleanupCommand:
    """Clean up old completed jobs.

    Remove job metadata and output files for jobs that
    completed more than the specified number of days ago.
    """

    max_age_days: Annotated[int, Parameter(help="Maximum age in days")] = 7
    output_format: Annotated[
        OutputFormat,
        Parameter(name="--format", help="Output format"),
    ] = OutputFormat.TEXT
    verbose: Annotated[int, Parameter(name="-v", count=True, help="Verbosity level")] = 0


__all__ = [
    "jobs_app",
]
