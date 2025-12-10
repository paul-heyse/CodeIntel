"""Background job management commands.

Provide commands to submit, monitor, and manage background jobs
for long-running operations.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Annotated, Literal

from cyclopts import App, Parameter

from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.command_context import command_context
from codeintel.cli.cyclopts_common import OutputFormatCLI, RuntimeCLI
from codeintel.cli.handlers.jobs import (
    jobs_cancel_handler,
    jobs_cleanup_handler,
    jobs_list_handler,
    jobs_output_handler,
    jobs_status_handler,
)

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
        OutputFormat,
        Parameter(name="--format", help="Output format"),
    ] = OutputFormat.TEXT

    def __call__(self) -> None:
        """Execute the jobs list command."""
        runtime_cli = RuntimeCLI()
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {
            "status": self.status,
            "limit": self.limit,
        }

        with command_context(
            "jobs.list",
            runtime_cli,
            output_cli,
            params=params,
            require_runtime=False,
        ) as (ctx, renderer):
            result = jobs_list_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


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

    def __call__(self) -> None:
        """Execute the jobs status command."""
        runtime_cli = RuntimeCLI()
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {"job_id": self.job_id}

        with command_context(
            "jobs.status",
            runtime_cli,
            output_cli,
            params=params,
            require_runtime=False,
        ) as (ctx, renderer):
            result = jobs_status_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


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

    def __call__(self) -> None:
        """Execute the jobs output command."""
        runtime_cli = RuntimeCLI()
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {"job_id": self.job_id}

        with command_context(
            "jobs.output",
            runtime_cli,
            output_cli,
            params=params,
            require_runtime=False,
        ) as (ctx, renderer):
            result = jobs_output_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


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

    def __call__(self) -> None:
        """Execute the jobs cancel command."""
        runtime_cli = RuntimeCLI()
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {"job_id": self.job_id}

        with command_context(
            "jobs.cancel",
            runtime_cli,
            output_cli,
            params=params,
            require_runtime=False,
        ) as (ctx, renderer):
            result = jobs_cancel_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


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

    def __call__(self) -> None:
        """Execute the jobs cleanup command."""
        runtime_cli = RuntimeCLI()
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {"max_age_days": self.max_age_days}

        with command_context(
            "jobs.cleanup",
            runtime_cli,
            output_cli,
            params=params,
            require_runtime=False,
        ) as (ctx, renderer):
            result = jobs_cleanup_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


__all__ = [
    "jobs_app",
]
