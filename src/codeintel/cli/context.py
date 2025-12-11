"""Unified command execution context.

This module provides the single, canonical context type that all CLI commands
receive. It provides:

The CommandContext provides:

- Lazy resource access (runtime, storage, serving, jobs)
- Typed parameter access via ParamService
- Automatic resource cleanup via context manager
- Declarative resource requirements
"""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

from codeintel.cli.config import load_config
from codeintel.cli.rendering.types import OutputFormat
from codeintel.cli.services.jobs import JobService
from codeintel.cli.services.params import ParamService
from codeintel.cli.services.runtime import RuntimeService
from codeintel.cli.services.serving import ServingService
from codeintel.cli.services.storage import StorageService

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.cli.config.model import CliConfig
    from codeintel.cli.resolution.types import ResolvedRuntime
    from codeintel.storage.gateway import StorageGateway

LOG = logging.getLogger(__name__)


@dataclass
class CommandContext:
    """Unified execution context for all CLI commands.

    Provide lazy access to resources (runtime, storage, serving) with
    automatic lifecycle management. Commands declare their resource
    requirements, and the context provides them on demand.

    Parameters
    ----------
    config
        CLI configuration.
    logger
        Logger for command output.
    params
        Parameter service for typed parameter access.
    jobs
        Job service for background job management.
    operation_id
        Unique identifier for this command execution.
    output_format
        Output format for rendering results.
    verbosity
        Logging verbosity level (0=WARNING, 1=INFO, 2+=DEBUG).

    Examples
    --------
    >>> with CommandContextBuilder().with_storage().build() as ctx:  # doctest: +SKIP
    ...     gateway = ctx.storage.gateway
    ...     result = gateway.query("SELECT * FROM modules")
    """

    config: CliConfig
    logger: logging.Logger
    params: ParamService
    jobs: JobService
    operation_id: str
    output_format: OutputFormat = OutputFormat.TEXT
    verbosity: int = 0

    # Lazy services (initialized on first access)
    _runtime: RuntimeService | None = field(default=None, repr=False)
    _storage: StorageService | None = field(default=None, repr=False)
    _serving: ServingService | None = field(default=None, repr=False)

    # Track if we own resources (for cleanup)
    _owns_storage: bool = field(default=False, repr=False)

    @property
    def runtime(self) -> ResolvedRuntime:
        """Get resolved runtime.

        Returns
        -------
        ResolvedRuntime
            Fully resolved runtime configuration.

        Raises
        ------
        RuntimeError
            If runtime was not configured.
        """
        if self._runtime is None:
            msg = "Runtime not available. Use CommandContextBuilder.with_runtime()"
            raise RuntimeError(msg)
        return self._runtime.runtime

    @property
    def storage(self) -> StorageService:
        """Get storage service.

        Returns
        -------
        StorageService
            Storage service for gateway access.

        Raises
        ------
        RuntimeError
            If storage was not configured.
        """
        if self._storage is None:
            msg = "Storage not available. Use CommandContextBuilder.with_storage()"
            raise RuntimeError(msg)
        return self._storage

    @property
    def gateway(self) -> StorageGateway:
        """Get storage gateway (convenience accessor).

        Returns
        -------
        StorageGateway
            Open storage gateway.
        """
        return self.storage.gateway

    @property
    def serving(self) -> ServingService:
        """Get serving service.

        Returns
        -------
        ServingService
            Serving service for operation invocation.

        Raises
        ------
        RuntimeError
            If serving was not configured.
        """
        if self._serving is None:
            msg = "Serving not available. Use CommandContextBuilder.with_serving()"
            raise RuntimeError(msg)
        return self._serving

    @property
    def has_runtime(self) -> bool:
        """Check if runtime is available.

        Returns
        -------
        bool
            True if runtime was configured.
        """
        return self._runtime is not None

    @property
    def has_storage(self) -> bool:
        """Check if storage is available.

        Returns
        -------
        bool
            True if storage was configured.
        """
        return self._storage is not None

    @property
    def has_serving(self) -> bool:
        """Check if serving is available.

        Returns
        -------
        bool
            True if serving was configured.
        """
        return self._serving is not None

    @contextmanager
    def write_gateway(self) -> Iterator[StorageGateway]:
        """Context manager for write-enabled gateway.

        Yields
        ------
        StorageGateway
            Write-enabled gateway closed on exit.
        """
        with self.storage.write_gateway() as gw:
            yield gw

    def close(self) -> None:
        """Close owned resources.

        Call this when done with the context. Automatically called when
        using CommandContextBuilder as a context manager.
        """
        if self._owns_storage and self._storage is not None:
            self._storage.close()


class CommandContextBuilder:
    """Builder for constructing CommandContext with appropriate services.

    Use the builder pattern to configure resource requirements before
    building the context.

    Examples
    --------
    >>> builder = CommandContextBuilder()  # doctest: +SKIP
    >>> builder = builder.with_storage().with_params({"name": "test"})
    >>> with builder.build() as ctx:
    ...     result = ctx.gateway.query("SELECT * FROM modules")
    """

    def __init__(self) -> None:
        """Initialize builder with default settings."""
        self._require_runtime = False
        self._require_storage = False
        self._require_serving = False
        self._project_root: Path | None = None
        self._db_path: Path | None = None
        self._params: dict[str, object] = {}
        self._output_format: OutputFormat = OutputFormat.TEXT
        self._verbosity: int = 0
        self._operation_id: str | None = None
        self._logger: logging.Logger | None = None
        self._injected_gateway: StorageGateway | None = None

    def with_runtime(self, *, project_root: Path | None = None) -> Self:
        """Enable runtime resolution.

        Parameters
        ----------
        project_root
            Optional explicit project root.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._require_runtime = True
        if project_root is not None:
            self._project_root = project_root
        return self

    def with_storage(self, *, db_path: Path | None = None) -> Self:
        """Enable storage access.

        Implicitly enables runtime resolution.

        Parameters
        ----------
        db_path
            Optional explicit database path.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._require_storage = True
        self._require_runtime = True
        if db_path is not None:
            self._db_path = db_path
        return self

    def with_serving(self) -> Self:
        """Enable serving access.

        Implicitly enables storage and runtime.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._require_serving = True
        self._require_storage = True
        self._require_runtime = True
        return self

    def with_params(self, params: dict[str, object]) -> Self:
        """Set command parameters.

        Parameters
        ----------
        params
            Raw parameters dictionary.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._params = params
        return self

    def with_output_format(self, fmt: OutputFormat) -> Self:
        """Set output format.

        Parameters
        ----------
        fmt
            Output format for rendering.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._output_format = fmt
        return self

    def with_verbosity(self, level: int) -> Self:
        """Set verbosity level.

        Parameters
        ----------
        level
            Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG).

        Returns
        -------
        Self
            Self for chaining.
        """
        self._verbosity = level
        return self

    def with_operation_id(self, operation_id: str) -> Self:
        """Set operation ID.

        Parameters
        ----------
        operation_id
            Unique operation identifier.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._operation_id = operation_id
        return self

    def with_logger(self, logger: logging.Logger) -> Self:
        """Set logger.

        Parameters
        ----------
        logger
            Logger for command output.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._logger = logger
        return self

    def with_injected_gateway(self, gateway: StorageGateway) -> Self:
        """Inject a pre-built gateway (for testing).

        Parameters
        ----------
        gateway
            Pre-built storage gateway.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._injected_gateway = gateway
        return self

    @contextmanager
    def build(self) -> Iterator[CommandContext]:
        """Build CommandContext and manage resource lifecycle.

        Yields
        ------
        CommandContext
            Configured command context.
        """
        # Load configuration
        config = load_config(validate=False)

        # Create logger
        logger = self._logger or logging.getLogger("codeintel.cli")

        # Create param service
        param_service = ParamService(self._params)

        # Create job service
        job_service = JobService()

        # Generate operation ID if not provided
        operation_id = self._operation_id or str(uuid.uuid4())[:8]

        # Create services based on requirements
        runtime_service: RuntimeService | None = None
        storage_service: StorageService | None = None
        serving_service: ServingService | None = None

        try:
            # Build runtime if required
            if self._require_runtime:
                runtime_service = RuntimeService(
                    param_service,
                    project_root=self._project_root,
                    db_path=self._db_path,
                )

            # Build storage if required
            if self._require_storage and runtime_service is not None:
                storage_service = StorageService.from_runtime(runtime_service)
            elif self._injected_gateway is not None:
                # Use injected gateway (for testing)
                storage_service = StorageService.from_gateway(self._injected_gateway)

            # Build serving if required
            if (
                self._require_serving
                and runtime_service is not None
                and storage_service is not None
            ):
                serving_service = ServingService(runtime_service, storage_service)

            ctx = CommandContext(
                config=config,
                logger=logger,
                params=param_service,
                jobs=job_service,
                operation_id=operation_id,
                output_format=self._output_format,
                verbosity=self._verbosity,
                _runtime=runtime_service,
                _storage=storage_service,
                _serving=serving_service,
                _owns_storage=True,
            )

            yield ctx

        finally:
            # Cleanup resources
            if storage_service is not None:
                storage_service.close()


__all__ = [
    "CommandContext",
    "CommandContextBuilder",
]
