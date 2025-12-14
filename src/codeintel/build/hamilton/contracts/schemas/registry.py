"""Global registry for DatasetSchema instances.

This module provides the `DatasetSchemaRegistry` class and a module-level
singleton `SCHEMA_REGISTRY` that serves as the authoritative source for
dataset schemas.

Examples
--------
>>> from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY
>>> schema = SCHEMA_REGISTRY.get("analytics.function_metrics")
>>> schema is not None
True
>>> schema.column_names()
('function_goid_h128', 'urn', 'repo', ...)
"""

from __future__ import annotations

import logging
from functools import cache
from typing import TYPE_CHECKING

from codeintel.build.hamilton.contracts.schemas.builder import build_all_schemas
from codeintel.build.unified_registry import get_unified_registry

if TYPE_CHECKING:
    from collections.abc import ItemsView, Iterator, ValuesView

    from codeintel.build.hamilton.contracts.schemas.schema import DatasetSchema

__all__ = [
    "SCHEMA_REGISTRY",
    "DatasetSchemaRegistry",
    "get_schema",
]

log = logging.getLogger(__name__)


@cache
def _get_plugin_metadata() -> dict[str, object]:
    """Load plugin metadata from build registry.

    Returns
    -------
    dict[str, object]
        Mapping of target names to plugin instances with core_metadata.
    """
    registry = get_unified_registry().get_all_plugins()
    result: dict[str, object] = {}
    for name, cls in registry.items():
        try:
            plugin = cls()
        except (TypeError, ValueError, AttributeError):
            log.debug("Failed to instantiate plugin %s", name)
            continue

        if hasattr(plugin, "core_metadata"):
            result[name] = plugin

    return result


class DatasetSchemaRegistry:
    """Global registry for all DatasetSchema instances.

    This registry is the authoritative source for dataset schemas. It
    integrates with existing infrastructure (DATASET_CONTRACTS and
    Pandera schemas) to provide the unified schema architecture.

    Attributes
    ----------
    _schemas
        Internal mapping of table keys to DatasetSchema instances.
    _initialized
        Flag indicating whether lazy initialization has occurred.

    Examples
    --------
    >>> registry = DatasetSchemaRegistry()
    >>> registry.initialize()
    >>> schema = registry.get("analytics.function_metrics")
    >>> schema is not None
    True
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._schemas: dict[str, DatasetSchema] = {}
        self._initialized: bool = False

    def initialize(self) -> None:
        """Build schemas from existing contracts and Pandera definitions.

        This method bridges the current infrastructure with the new unified
        schema layer. It is called lazily on first access.

        Notes
        -----
        This method is idempotent; calling it multiple times has no effect
        after the first initialization.
        """
        if self._initialized:
            return

        self._schemas = build_all_schemas()
        self._initialized = True

    def get(self, table_key: str) -> DatasetSchema | None:
        """Retrieve a DatasetSchema by table key.

        Parameters
        ----------
        table_key
            Fully qualified table name (e.g., "analytics.function_metrics").

        Returns
        -------
        DatasetSchema | None
            The registered schema if found, otherwise None.

        Examples
        --------
        >>> schema = SCHEMA_REGISTRY.get("analytics.function_metrics")
        >>> schema is not None
        True
        """
        self.initialize()
        return self._schemas.get(table_key)

    def require(self, table_key: str) -> DatasetSchema:
        """Retrieve a DatasetSchema or raise if not found.

        Parameters
        ----------
        table_key
            Fully qualified table name.

        Returns
        -------
        DatasetSchema
            The registered schema.

        Raises
        ------
        KeyError
            If no schema is registered for the given table key.

        Examples
        --------
        >>> schema = SCHEMA_REGISTRY.require("analytics.function_metrics")
        >>> schema.name
        'analytics.function_metrics'
        """
        schema = self.get(table_key)
        if schema is None:
            msg = f"No DatasetSchema registered for '{table_key}'"
            raise KeyError(msg)
        return schema

    def all(self) -> dict[str, DatasetSchema]:
        """Return all registered schemas.

        Returns
        -------
        dict[str, DatasetSchema]
            Copy of the internal schema mapping.

        Examples
        --------
        >>> schemas = SCHEMA_REGISTRY.all()
        >>> len(schemas) > 0
        True
        """
        self.initialize()
        return dict(self._schemas)

    def keys(self) -> list[str]:
        """Return all registered table keys.

        Returns
        -------
        list[str]
            List of table keys with registered schemas.
        """
        self.initialize()
        return list(self._schemas.keys())

    def items(self) -> ItemsView[str, DatasetSchema]:
        """
        Return items view for registered schemas.

        Returns
        -------
        ItemsView[str, DatasetSchema]
            Items view over table key and schema pairs.
        """
        self.initialize()
        return self._schemas.items()

    def values(self) -> ValuesView[DatasetSchema]:
        """
        Return values view for registered schemas.

        Returns
        -------
        ValuesView[DatasetSchema]
            View over registered DatasetSchema instances.
        """
        self.initialize()
        return self._schemas.values()

    def __len__(self) -> int:
        """Return the number of registered schemas.

        Returns
        -------
        int
            Number of schemas in the registry.
        """
        self.initialize()
        return len(self._schemas)

    def __contains__(self, table_key: str) -> bool:
        """Check if a table key is registered.

        Parameters
        ----------
        table_key
            Fully qualified table name.

        Returns
        -------
        bool
            True if a schema is registered for the key.
        """
        self.initialize()
        return table_key in self._schemas

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over registered table keys.

        Returns
        -------
        Iterator[str]
            Iterator over schema-qualified table keys.
        """
        self.initialize()
        return iter(self._schemas.keys())

    @staticmethod
    def producers_of(table_key: str) -> list[str]:
        """Find plugins that produce the given dataset.

        Parameters
        ----------
        table_key
            Dataset to find producers for.

        Returns
        -------
        list[str]
            Plugin names that produce this dataset.

        Notes
        -----
        Uses the build registry as the single source of truth.
        """
        plugins = _get_plugin_metadata()
        result: list[str] = []
        for name, plugin in plugins.items():
            core_meta = getattr(plugin, "core_metadata", None)
            if core_meta is None:
                continue
            produces = getattr(core_meta, "produces_tables", None)
            if produces and table_key in produces:
                result.append(name)
        return result

    @staticmethod
    def consumers_of(table_key: str) -> list[str]:
        """Find plugins that consume the given dataset.

        Parameters
        ----------
        table_key
            Dataset to find consumers for.

        Returns
        -------
        list[str]
            Plugin names that consume this dataset.

        Notes
        -----
        Uses the build registry as the single source of truth.
        """
        plugins = _get_plugin_metadata()
        result: list[str] = []
        for name, plugin in plugins.items():
            core_meta = getattr(plugin, "core_metadata", None)
            if core_meta is None:
                continue
            consumes = getattr(core_meta, "consumes_tables", None)
            if consumes and table_key in consumes:
                result.append(name)
        return result


SCHEMA_REGISTRY = DatasetSchemaRegistry()


def get_schema(table_key: str) -> DatasetSchema | None:
    """Get a schema from the global registry.

    Parameters
    ----------
    table_key
        Fully qualified table name.

    Returns
    -------
    DatasetSchema | None
        The registered schema if found, otherwise None.

    Examples
    --------
    >>> schema = get_schema("analytics.function_metrics")
    >>> schema is not None
    True
    """
    return SCHEMA_REGISTRY.get(table_key)
