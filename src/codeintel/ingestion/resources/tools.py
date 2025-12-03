"""Tools provider for lazy access to tool service.

This module provides `ToolsProvider`, a resource provider that
lazily initializes the tool service for running external tools
like pyright, scip, coverage, etc.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.ingestion.infrastructure_utilities.tool_runner import ToolRunner
from codeintel.ingestion.resources.protocol import LazyResource
from codeintel.ingestion.tool_service import ToolService

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.config.models import ToolsConfig

log = logging.getLogger(__name__)


class ToolsProvider(LazyResource["ToolService"]):
    """Lazy provider for tool service.

    Lazily initialize the tool service and tool runner for
    executing external tools.

    Parameters
    ----------
    tools_config
        Tools configuration with binary paths.
    cache_dir
        Cache directory for tool outputs.
    runner
        Optional pre-configured tool runner.
    service
        Optional pre-configured tool service.
    """

    def __init__(
        self,
        tools_config: ToolsConfig,
        cache_dir: Path,
        runner: ToolRunner | None = None,
        service: ToolService | None = None,
    ) -> None:
        """Initialize the tools provider.

        Parameters
        ----------
        tools_config
            Tools configuration with binary paths.
        cache_dir
            Cache directory for tool outputs.
        runner
            Optional pre-configured tool runner.
        service
            Optional pre-configured tool service.
        """
        super().__init__("ToolsProvider")
        self._tools_config = tools_config
        self._cache_dir = cache_dir
        self._runner = runner
        self._service = service

    def _load(self) -> ToolService:
        """Load or create the tool service.

        Returns
        -------
        ToolService
            The tool service.
        """
        # If service already provided, return it
        if self._service is not None:
            log.debug("Using pre-configured tool service")
            return self._service

        # Create runner if not provided
        runner = self._runner or ToolRunner(
            cache_dir=self._cache_dir,
            tools_config=self._tools_config,
        )

        log.debug("Creating tool service with cache_dir=%s", self._cache_dir)
        return ToolService(runner, self._tools_config)

    @property
    def runner(self) -> ToolRunner | None:
        """Return the underlying tool runner if available.

        Returns
        -------
        ToolRunner | None
            The tool runner or None if not loaded yet.
        """
        if not self.is_loaded:
            return self._runner
        # Access internal _resource directly - service stores runner
        if self._resource is not None:
            return self._resource.runner
        return self._runner


__all__ = ["ToolsProvider"]
