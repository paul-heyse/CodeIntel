"""Compatibility adapters for gradual migration.

Provide adapters to convert between the old context types (HandlerContext, Deps)
and the new CommandContext during migration.

These adapters are temporary and will be removed in Phase 9 of the migration.

Note: This module intentionally accesses private members (_storage, _runtime, etc.)
to bridge between the old and new context implementations. These accesses are
necessary for the migration and will be removed with this module.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from codeintel.cli.context import CommandContext
from codeintel.cli.deps.protocols import JobManagerProtocol, ServingAccess, StorageAccess
from codeintel.cli.jobs import JobInfo, JobStatus
from codeintel.cli.rendering.types import OutputFormat
from codeintel.cli.services.jobs import JobService
from codeintel.cli.services.params import ParamService
from codeintel.cli.services.runtime import RuntimeService
from codeintel.cli.services.serving import ServingService
from codeintel.cli.services.storage import StorageService

if TYPE_CHECKING:
    from codeintel.cli.deps.container import Deps
    from codeintel.cli.handlers.context import HandlerContext
    from codeintel.storage.gateway import StorageGateway


class StorageAccessAdapter(StorageAccess):
    """Adapt CommandContext to StorageAccess protocol.

    Parameters
    ----------
    ctx
        CommandContext to adapt.
    """

    def __init__(self, ctx: CommandContext) -> None:
        """Initialize adapter."""
        self._ctx = ctx

    @property
    def gateway(self) -> StorageGateway:
        """Get storage gateway.

        Returns
        -------
        StorageGateway
            Open storage gateway.
        """
        return self._ctx.gateway

    @contextmanager
    def write_gateway(self) -> Iterator[StorageGateway]:
        """Context manager for write gateway.

        Yields
        ------
        StorageGateway
            Write-enabled gateway.
        """
        with self._ctx.write_gateway() as gw:
            yield gw


class ServingAccessAdapter(ServingAccess):
    """Adapt CommandContext to ServingAccess protocol.

    Parameters
    ----------
    ctx
        CommandContext to adapt.
    """

    def __init__(self, ctx: CommandContext) -> None:
        """Initialize adapter."""
        self._ctx = ctx

    def invoke(
        self,
        operation_id: str,
        params: dict[str, object],
        *,
        skip_prereqs: bool = False,
    ) -> dict[str, object]:
        """Invoke a serving operation.

        Parameters
        ----------
        operation_id
            Operation ID.
        params
            Operation parameters.
        skip_prereqs
            Whether to skip prerequisites.

        Returns
        -------
        dict[str, object]
            Operation result.
        """
        return self._ctx.serving.invoke(operation_id, params, skip_prereqs=skip_prereqs)


class JobManagerAdapter(JobManagerProtocol):
    """Adapt CommandContext to JobManagerProtocol.

    Parameters
    ----------
    ctx
        CommandContext to adapt.
    """

    def __init__(self, ctx: CommandContext) -> None:
        """Initialize adapter."""
        self._ctx = ctx

    def list_jobs(
        self,
        *,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[JobInfo]:
        """List jobs.

        Parameters
        ----------
        status
            Status filter.
        limit
            Maximum jobs.

        Returns
        -------
        list[JobInfo]
            Job list.
        """
        return self._ctx.jobs.list_jobs(status=status, limit=limit)

    def get_status(self, job_id: str) -> JobInfo | None:
        """Get job status.

        Parameters
        ----------
        job_id
            Job ID.

        Returns
        -------
        JobInfo | None
            Job info.
        """
        return self._ctx.jobs.get_status(job_id)

    def get_output(self, job_id: str) -> dict[str, object] | None:
        """Get job output.

        Parameters
        ----------
        job_id
            Job ID.

        Returns
        -------
        dict[str, object] | None
            Output data.
        """
        return self._ctx.jobs.get_output(job_id)

    def cancel(self, job_id: str) -> bool:
        """Cancel job.

        Parameters
        ----------
        job_id
            Job ID.

        Returns
        -------
        bool
            True if cancelled.
        """
        return self._ctx.jobs.cancel(job_id)

    def cleanup(self, *, max_age_days: int = 7) -> int:
        """Clean up old jobs.

        Parameters
        ----------
        max_age_days
            Maximum age.

        Returns
        -------
        int
            Number cleaned.
        """
        return self._ctx.jobs.cleanup(max_age_days=max_age_days)


def deps_from_command_context(ctx: CommandContext) -> Deps:
    """Create Deps from CommandContext.

    Provide backward compatibility for code expecting Deps.

    Parameters
    ----------
    ctx
        CommandContext to convert.

    Returns
    -------
    Deps
        Deps container wrapping the CommandContext.
    """
    # Import here to avoid circular dependency
    from codeintel.cli.deps.container import Deps  # noqa: PLC0415

    return Deps(
        config=ctx.config,
        logger=ctx.logger,
        jobs=JobManagerAdapter(ctx),
        _storage=StorageAccessAdapter(ctx) if ctx.has_storage else None,
        _serving=ServingAccessAdapter(ctx) if ctx.has_serving else None,
    )


def command_context_from_deps(
    deps: Deps,
    *,
    operation_id: str = "unknown",
    output_format: OutputFormat = OutputFormat.TEXT,
    verbosity: int = 0,
) -> CommandContext:
    """Create CommandContext from Deps.

    Allow gradual migration from Deps to CommandContext.

    Parameters
    ----------
    deps
        Deps container to convert.
    operation_id
        Operation identifier.
    output_format
        Output format.
    verbosity
        Verbosity level.

    Returns
    -------
    CommandContext
        CommandContext wrapping the Deps.
    """
    # Create param service (empty - Deps doesn't track raw params)
    params = ParamService({})

    # Create job service from the Deps job manager
    job_service = JobService(manager=deps.jobs)  # type: ignore[arg-type]

    # Build storage/serving services if available
    storage = None
    serving = None
    runtime = None

    if deps.has_storage:
        # Extract db_path from storage provider
        # Access private member intentionally for compat bridging
        storage_access = deps._storage  # noqa: SLF001
        if storage_access is not None:
            # Get the gateway's db_path
            gateway = storage_access.gateway
            db_path = gateway.config.db_path
            storage = StorageService.from_path(db_path)

            # Create runtime service from the db_path
            runtime = RuntimeService({"db_path": db_path})

    if deps.has_serving and storage is not None and runtime is not None:
        serving = ServingService(runtime, storage)

    return CommandContext(
        config=deps.config,
        logger=deps.logger,
        params=params,
        jobs=job_service,
        operation_id=operation_id,
        output_format=output_format,
        verbosity=verbosity,
        _runtime=runtime,
        _storage=storage,
        _serving=serving,
        _owns_storage=False,  # Deps owns the storage
    )


def handler_context_from_command_context(ctx: CommandContext) -> HandlerContext:
    """Create HandlerContext from CommandContext.

    Provide backward compatibility for legacy handlers.

    Parameters
    ----------
    ctx
        CommandContext to convert.

    Returns
    -------
    HandlerContext
        Legacy handler context.
    """
    # Import here to avoid circular dependency
    from codeintel.cli.handlers.context import HandlerContext  # noqa: PLC0415

    return HandlerContext(
        config=ctx.config,
        operation_id=ctx.operation_id,
        output_format=ctx.output_format,
        verbosity=ctx.verbosity,
        _params=dict(ctx.params.raw),
    )


def command_context_from_handler_context(
    handler_ctx: HandlerContext,
) -> CommandContext:
    """Create CommandContext from HandlerContext.

    Allow legacy handlers to use new CommandContext services.

    Parameters
    ----------
    handler_ctx
        Legacy handler context.

    Returns
    -------
    CommandContext
        New command context.
    """
    params = ParamService(handler_ctx.params)
    jobs = JobService()
    logger = logging.getLogger(f"codeintel.cli.{handler_ctx.operation_id}")

    # Create services based on what HandlerContext has
    runtime = None
    storage = None
    serving = None

    # If handler context has a runtime, use it
    # Access private members intentionally for compat bridging
    if handler_ctx._runtime is not None:  # noqa: SLF001
        runtime = RuntimeService(params)
        # Force the cached runtime to match
        runtime._resolved = handler_ctx._runtime  # noqa: SLF001

    # If handler context has a gateway, create storage service
    if handler_ctx._gateway is not None:  # noqa: SLF001
        db_path = handler_ctx._gateway.config.db_path  # noqa: SLF001
        storage = StorageService.from_path(db_path)

    # Can create serving if we have both
    if runtime is not None and storage is not None:
        serving = ServingService(runtime, storage)

    return CommandContext(
        config=handler_ctx.config,
        logger=logger,
        params=params,
        jobs=jobs,
        operation_id=handler_ctx.operation_id,
        output_format=handler_ctx.output_format,
        verbosity=handler_ctx.verbosity,
        _runtime=runtime,
        _storage=storage,
        _serving=serving,
        _owns_storage=False,  # HandlerContext owns resources
    )


__all__ = [
    "JobManagerAdapter",
    "ServingAccessAdapter",
    "StorageAccessAdapter",
    "command_context_from_deps",
    "command_context_from_handler_context",
    "deps_from_command_context",
    "handler_context_from_command_context",
]
