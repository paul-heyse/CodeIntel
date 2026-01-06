"""Base types for ingestion compute layer.

This module defines common types used by all ingestion compute modules,
analogous to base types in graphs/compute/.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from codeintel.ingestion.infrastructure.py_frontend import PyFrontend
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
    frontend
        Optional shared frontend cache for source and AST reuse.
    """

    _discovery: ModuleDiscoveryPort
    _frontend: PyFrontend | None

    def __init__(
        self,
        discovery: ModuleDiscoveryPort,
        *,
        frontend: PyFrontend | None = None,
    ) -> None:
        """Initialize the step with discovery ports.

        Parameters
        ----------
        discovery
            Discovery port for reading module source.
        frontend
            Optional shared frontend cache for source and AST reuse.
        """
        self._discovery = discovery
        self._frontend = frontend

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
            if self._frontend is not None:
                source = self._frontend.get_source_text(module)
            else:
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
