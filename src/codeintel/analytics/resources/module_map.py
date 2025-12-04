"""Module map resource provider for path-to-module mapping.

This module provides `ModuleMapProvider` for lazy loading of the
path-to-module mapping used in analytics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.resources.protocol import LazyResource, ResourceNotLoadedError
from codeintel.storage.helpers.module_index import load_module_map

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


class ModuleMapProvider(LazyResource[dict[str, str]]):
    """Provider for module map with lazy loading.

    The module map provides a mapping from file paths to their Python
    module names, used for module-level analytics and symbol resolution.

    Parameters can be None when using factory methods like `from_map()`
    that set a pre-loaded resource.

    Attributes
    ----------
    language
        Optional language filter for the module map.

    Example
    -------
    >>> provider = ModuleMapProvider(gateway, snapshot)
    >>> module_map = provider.get()
    >>> module_name = module_map.get("src/codeintel/analytics/core.py")
    """

    RESOURCE_NAME: ClassVar[str] = "ModuleMapProvider"

    def __init__(
        self,
        gateway: StorageGateway | None = None,
        snapshot: SnapshotRef | None = None,
        *,
        language: str | None = None,
    ) -> None:
        """Initialize the module map provider.

        Parameters
        ----------
        gateway
            Storage gateway for database access. Can be None if using
            `set_preloaded()` or `from_map()` factory method.
        snapshot
            Repository snapshot reference. Can be None if using
            `set_preloaded()` or `from_map()` factory method.
        language
            Optional language filter (e.g., "python").
        """
        super().__init__("ModuleMap")
        self._gateway = gateway
        self._snapshot = snapshot
        self._language = language

    @classmethod
    def from_map(cls, module_map: dict[str, str]) -> ModuleMapProvider:
        """Create a provider from an existing module map.

        Use this factory when a module map has already been loaded and you
        want to wrap it in a provider for the resource registry.

        Parameters
        ----------
        module_map
            Pre-loaded module map (path -> module name).

        Returns
        -------
        ModuleMapProvider
            Provider wrapping the existing map.

        Example
        -------
        >>> existing_map = {"src/foo.py": "foo", "src/bar.py": "bar"}
        >>> provider = ModuleMapProvider.from_map(existing_map)
        >>> registry.register(ModuleMapProvider, provider)
        """
        provider = cls(gateway=None, snapshot=None)
        provider.set_preloaded(module_map)
        return provider

    def _load(self) -> dict[str, str]:
        """Load the module map from the database.

        Returns
        -------
        dict[str, str]
            Mapping of file path to module name.

        Raises
        ------
        ResourceNotLoadedError
            If gateway or snapshot are None (provider created for pre-loading only).
        """
        if self._gateway is None or self._snapshot is None:
            raise ResourceNotLoadedError(
                self._name,
                "Cannot load - provider was created for pre-loaded resource only. "
                "Use from_map() with a pre-loaded map or provide gateway and snapshot.",
            )

        module_map = load_module_map(
            self._gateway,
            repo=self._snapshot.repo,
            commit=self._snapshot.commit,
            language=self._language,
        )

        log.debug(
            "Loaded module map with %d entries for %s@%s",
            len(module_map),
            self._snapshot.repo,
            self._snapshot.commit,
        )

        return module_map

    @property
    def module_map(self) -> dict[str, str]:
        """Return the module map.

        Convenience property for direct access without calling `get()`.

        Returns
        -------
        dict[str, str]
            Mapping of file path to module name.
        """
        return self.get()

    def get_module(self, path: str) -> str | None:
        """Get the module name for a file path.

        Parameters
        ----------
        path
            Relative file path.

        Returns
        -------
        str | None
            Module name, or None if path not in map.
        """
        return self.get().get(path)


__all__ = [
    "ModuleMapProvider",
]
