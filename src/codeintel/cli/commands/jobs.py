"""Background job management commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from cyclopts import App

from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.jobs import (
    jobs_cancel_handler,
    jobs_cleanup_handler,
    jobs_list_handler,
    jobs_output_handler,
    jobs_status_handler,
)
from codeintel.cli.options.registry import (
    JOBS_JOB_ID,
    JOBS_LIMIT,
    JOBS_MAX_AGE_DAYS,
    JOBS_STATUS_FILTER,
)
from codeintel.cli.options.shared_flags import SharedFlagsProtocol, shared_flags_field
from codeintel.cli.options.types import CommandPath, option_param

jobs_app = App(name="jobs", help="Manage background jobs")

JOBS_LIST_PATH: CommandPath = ("jobs", "list")
JOBS_STATUS_PATH: CommandPath = ("jobs", "status")
JOBS_OUTPUT_PATH: CommandPath = ("jobs", "output")
JOBS_CANCEL_PATH: CommandPath = ("jobs", "cancel")
JOBS_CLEANUP_PATH: CommandPath = ("jobs", "cleanup")

_JOBS_LIST_FLAGS_FIELD = shared_flags_field(JOBS_LIST_PATH)
_JOBS_STATUS_FLAGS_FIELD = shared_flags_field(JOBS_STATUS_PATH)
_JOBS_OUTPUT_FLAGS_FIELD = shared_flags_field(JOBS_OUTPUT_PATH)
_JOBS_CANCEL_FLAGS_FIELD = shared_flags_field(JOBS_CANCEL_PATH)
_JOBS_CLEANUP_FLAGS_FIELD = shared_flags_field(JOBS_CLEANUP_PATH)


_JOBS_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)


@cli_command("jobs.list", handler=jobs_list_handler, config=_JOBS_CONFIG)
@jobs_app.command(name="list")
@dataclass
class ListJobsCommand:
    """List background jobs."""

    status: Annotated[
        Literal["pending", "running", "completed", "failed", "cancelled"] | None,
        option_param(JOBS_STATUS_FILTER, command_path=JOBS_LIST_PATH),
    ] = None
    limit: Annotated[int, option_param(JOBS_LIMIT, command_path=JOBS_LIST_PATH)] = 20
    flags: SharedFlagsProtocol = _JOBS_LIST_FLAGS_FIELD


@cli_command("jobs.status", handler=jobs_status_handler, config=_JOBS_CONFIG)
@jobs_app.command(name="status")
@dataclass
class GetJobStatusCommand:
    """Get status of a background job."""

    job_id: Annotated[str, option_param(JOBS_JOB_ID, command_path=JOBS_STATUS_PATH)]
    flags: SharedFlagsProtocol = _JOBS_STATUS_FLAGS_FIELD


@cli_command("jobs.output", handler=jobs_output_handler, config=_JOBS_CONFIG)
@jobs_app.command(name="output")
@dataclass
class GetJobOutputCommand:
    """Get output of a completed job."""

    job_id: Annotated[str, option_param(JOBS_JOB_ID, command_path=JOBS_OUTPUT_PATH)]
    flags: SharedFlagsProtocol = _JOBS_OUTPUT_FLAGS_FIELD


@cli_command("jobs.cancel", handler=jobs_cancel_handler, config=_JOBS_CONFIG)
@jobs_app.command(name="cancel")
@dataclass
class CancelJobCommand:
    """Cancel a running job."""

    job_id: Annotated[str, option_param(JOBS_JOB_ID, command_path=JOBS_CANCEL_PATH)]
    flags: SharedFlagsProtocol = _JOBS_CANCEL_FLAGS_FIELD


@cli_command("jobs.cleanup", handler=jobs_cleanup_handler, config=_JOBS_CONFIG)
@jobs_app.command(name="cleanup")
@dataclass
class CleanupJobsCommand:
    """Clean up old completed jobs."""

    max_age_days: Annotated[
        int,
        option_param(JOBS_MAX_AGE_DAYS, command_path=JOBS_CLEANUP_PATH),
    ] = 7
    flags: SharedFlagsProtocol = _JOBS_CLEANUP_FLAGS_FIELD


__all__ = [
    "CancelJobCommand",
    "CleanupJobsCommand",
    "GetJobOutputCommand",
    "GetJobStatusCommand",
    "ListJobsCommand",
    "jobs_app",
]
