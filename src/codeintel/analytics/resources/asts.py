"""AST resource provider for parsed AST access.

This module provides `AstProvider` for lazy loading of parsed AST
maps used in function analytics.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.parsing.ast_cache import (
    FunctionAstLoadRequest,
    load_function_asts,
)
from codeintel.analytics.resources.protocol import LazyResource, ResourceNotLoadedError

if TYPE_CHECKING:
    from codeintel.analytics.parsing.ast_cache import FunctionAst
    from codeintel.config.primitives import SnapshotRef
    from codeintel.graphs.catalog import FunctionCatalogProvider
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FunctionAstInfo:
    """AST information for a single function.

    Attributes
    ----------
    goid
        Global object identifier for the function.
    node
        The AST node for the function.
    lines
        Source lines for the function's file.
    rel_path
        Relative path to the source file.
    """

    goid: int
    node: ast.FunctionDef | ast.AsyncFunctionDef
    lines: list[str]
    rel_path: str


@dataclass
class AstMap:
    """Container for function AST data.

    Attributes
    ----------
    functions
        Mapping from GOID to AST info.
    missing_goids
        Set of GOIDs that could not be parsed.
    files_parsed
        Number of files successfully parsed.
    parse_errors
        Number of files with parse errors.
    """

    functions: dict[int, FunctionAstInfo]
    missing_goids: set[int]
    files_parsed: int = 0
    parse_errors: int = 0

    def get(self, goid: int) -> FunctionAstInfo | None:
        """Get AST info for a GOID.

        Parameters
        ----------
        goid
            The function GOID.

        Returns
        -------
        FunctionAstInfo | None
            The AST info, or None if not available.
        """
        return self.functions.get(goid)

    def __contains__(self, goid: int) -> bool:
        """Check if a GOID has AST info.

        Parameters
        ----------
        goid
            The function GOID.

        Returns
        -------
        bool
            True if AST info is available.
        """
        return goid in self.functions


@dataclass
class AstResourceData:
    """Container for AST resource data.

    Contains function AST map and missing GOIDs for analytics plugins.
    Uses `FunctionAst` from function_ast_cache for domain modules.

    Attributes
    ----------
    function_ast_map
        Mapping from GOID to FunctionAst.
    missing_function_goids
        Set of GOIDs that could not be parsed.
    """

    function_ast_map: dict[int, FunctionAst]
    missing_function_goids: set[int]


class AstProvider(LazyResource[AstResourceData]):
    """Provider for function ASTs with lazy loading.

    Parses source files and builds a map from function GOIDs to their
    AST nodes. Uses the FunctionAst format for domain modules.

    Parameters can be None when using factory methods like `from_asts()`
    that set a pre-loaded resource.

    Example
    -------
    >>> provider = AstProvider(gateway, snapshot)
    >>> data = provider.get()
    >>> func_ast = data.function_ast_map.get(function_goid)
    """

    RESOURCE_NAME: ClassVar[str] = "AstProvider"

    def __init__(
        self,
        gateway: StorageGateway | None = None,
        snapshot: SnapshotRef | None = None,
        *,
        catalog_provider: FunctionCatalogProvider | None = None,
        max_functions: int | None = None,
    ) -> None:
        """Initialize the AST provider.

        Parameters
        ----------
        gateway
            Storage gateway for GOID queries. Can be None if using
            `set_preloaded()` or `from_asts()` factory method.
        snapshot
            Repository snapshot reference. Can be None if using
            `set_preloaded()` or `from_asts()` factory method.
        catalog_provider
            Optional pre-loaded catalog provider.
        max_functions
            Maximum number of functions to parse (for resource limits).
        """
        super().__init__("AstData")
        self._gateway = gateway
        self._snapshot = snapshot
        self._catalog_provider = catalog_provider
        self._max_functions = max_functions

    @classmethod
    def from_asts(
        cls,
        function_ast_map: dict[int, FunctionAst],
        missing_goids: set[int],
    ) -> AstProvider:
        """Create a provider from existing AST data.

        Use this factory when AST data has already been loaded and you
        want to wrap it in a provider for the resource registry.

        Parameters
        ----------
        function_ast_map
            Pre-loaded function AST map.
        missing_goids
            Set of GOIDs that could not be parsed.

        Returns
        -------
        AstProvider
            Provider wrapping the existing data.

        Example
        -------
        >>> provider = AstProvider.from_asts(existing_asts, missing)
        >>> registry.register(AstProvider, provider)
        """
        # Create provider with None - valid since we set preloaded
        provider = cls(gateway=None, snapshot=None)
        provider.set_preloaded(
            AstResourceData(
                function_ast_map=function_ast_map,
                missing_function_goids=missing_goids,
            )
        )
        return provider

    def _load(self) -> AstResourceData:
        """Load and parse function ASTs.

        Returns
        -------
        AstResourceData
            Container with function AST map and missing GOIDs.

        Raises
        ------
        ResourceNotLoadedError
            If gateway or snapshot are None (provider created for pre-loading only).
        """
        if self._gateway is None or self._snapshot is None:
            raise ResourceNotLoadedError(
                self._name,
                "Cannot load - provider was created for pre-loaded resource only. "
                "Use from_asts() with pre-loaded data or provide gateway and snapshot.",
            )

        request = FunctionAstLoadRequest(
            repo=self._snapshot.repo,
            commit=self._snapshot.commit,
            repo_root=self._snapshot.repo_root,
            catalog_provider=self._catalog_provider,
            max_functions=self._max_functions,
        )

        function_ast_map, missing = load_function_asts(self._gateway, request)

        log.debug(
            "Loaded %d function ASTs (%d missing) for %s@%s",
            len(function_ast_map),
            len(missing),
            self._snapshot.repo,
            self._snapshot.commit,
        )

        return AstResourceData(
            function_ast_map=function_ast_map,
            missing_function_goids=missing,
        )


__all__ = [
    "AstMap",
    "AstProvider",
    "AstResourceData",
    "FunctionAstInfo",
]
