"""Base types for ingestion compute layer.

This module defines common types used by all ingestion compute modules,
analogous to base types in graphs/compute/.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord
    from codeintel.ingestion.ports.tools import IngestToolPort


class BaseExtractStep:
    """Base class for module extraction steps with port injection.

    Provides shared initialization and helper methods for steps that:

    - Iterate over Python modules and read source

    Parameters
    ----------
    discovery
        Discovery port for reading module source.
    """

    _discovery: ModuleDiscoveryPort

    def __init__(
        self,
        discovery: ModuleDiscoveryPort,
    ) -> None:
        """Initialize the step with discovery ports.

        Parameters
        ----------
        discovery
            Discovery port for reading module source.
        """
        self._discovery = discovery

    def _iter_python_sources(
        self, modules: Sequence[ModuleRecord]
    ) -> Iterator[tuple[ModuleRecord, str]]:
        """Yield (module, source) pairs for Python files with readable source.

        Parameters
        ----------
        modules
            Sequence of module records to iterate.

        Yields
        ------
        tuple[ModuleRecord, str]
            Module record and its source code for each readable Python file.
        """
        for module in modules:
            if not module.rel_path.endswith(".py"):
                continue
            source = self._discovery.read_module_source(module)
            if source is not None:
                yield module, source


class BaseToolIngestStep:
    """Base class for ingestion steps requiring tool execution.

    Provides shared initialization for steps that need tool ports.

    Parameters
    ----------
    tools
        Tool port for running external tools.
    """

    _tools: IngestToolPort

    def __init__(
        self,
        tools: IngestToolPort,
    ) -> None:
        """Initialize the step with tool ports.

        Parameters
        ----------
        tools
            Tool port for running external tools.
        """
        self._tools = tools


__all__ = ["BaseExtractStep", "BaseToolIngestStep"]
