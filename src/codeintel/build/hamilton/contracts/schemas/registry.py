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
def _get_hamilton_target_metadata() -> dict[str, dict[str, list[str]]]:
    """Load target metadata from Hamilton native modules.

    Extracts schema information from Hamilton decorators (`@tag`, `@schema.output`)
    which define the authoritative source of truth for data schemas.

    Returns
    -------
    dict[str, dict[str, list[str]]]
        Mapping of target names to dict with 'produces', 'consumes', and 'schema' info.
    """
    from codeintel.build.hamilton.native.loader import get_loader

    result: dict[str, dict[str, list[str]]] = {}
    loader = get_loader()
    modules = loader.discover_modules()

    for module in modules:
        # Extract target info from module functions
        for name in dir(module):
            func = getattr(module, name, None)
            if func is None or not callable(func):
                continue

            # Hamilton stores decorators in decorate_nodes list
            decorators = getattr(func, "decorate_nodes", [])
            if not decorators:
                continue

            # Extract tags and schema from decorators
            target: str | None = None
            schema_str: str | None = None

            for dec in decorators:
                dec_tags = getattr(dec, "tags", {})

                # Extract target from @tag decorator
                if "target" in dec_tags:
                    target = dec_tags["target"]

                # Extract schema from @schema.output decorator
                schema_key = "hamilton.internal.schema_output"
                if schema_key in dec_tags:
                    schema_str = dec_tags[schema_key]

            if target is None:
                continue

            # Get consumes from function parameters (input dependencies)
            consumes: list[str] = []
            annotations = getattr(func, "__annotations__", {})
            for param_name in annotations:
                if param_name.startswith("q__"):
                    # Query parameter indicates table consumption
                    parts = param_name.split("__")
                    if len(parts) >= 3:
                        table_key = f"{parts[1]}.{parts[2]}"
                        consumes.append(table_key)

            if target not in result:
                result[target] = {"produces": [], "consumes": [], "schemas": []}

            result[target]["consumes"].extend(consumes)

            # Parse schema string if present
            if schema_str:
                result[target]["schemas"].append(schema_str)

    return result


def get_hamilton_schema_for_target(target: str) -> dict[str, str] | None:
    """Get the schema definition for a target from Hamilton metadata.

    Parameters
    ----------
    target
        Target name to get schema for.

    Returns
    -------
    dict[str, str] | None
        Column name to type mapping, or None if not found.
    """
    metadata = _get_hamilton_target_metadata()
    target_info = metadata.get(target)
    if not target_info or not target_info.get("schemas"):
        return None

    # Parse the schema string (format: "col1=type1,col2=type2,...")
    schema_str = target_info["schemas"][0]
    result: dict[str, str] = {}
    for pair in schema_str.split(","):
        if "=" in pair:
            col_name, col_type = pair.split("=", 1)
            result[col_name.strip()] = col_type.strip()

    return result


@cache
def get_hamilton_dataclass_schemas() -> dict[str, dict[str, str]]:
    """Extract schemas from dataclass return types in Hamilton modules.

    For modules that return dataclass results, the dataclass fields define
    the schema. This function extracts those field definitions.

    Looks for dataclass types in:
    1. @check_output(data_type=...) decorators (stored in transform)
    2. Function return type annotations

    Multiple functions may have the same target (compute + materialize).
    We only record a schema once a valid dataclass is found.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping of target name to field schema (field name -> type string).
    """
    import dataclasses
    from typing import get_type_hints

    from codeintel.build.hamilton.native.loader import get_loader

    result: dict[str, dict[str, str]] = {}
    loader = get_loader()
    modules = loader.discover_modules()

    for module in modules:
        for name in dir(module):
            func = getattr(module, name, None)
            if func is None or not callable(func):
                continue

            # Get target from decorators (stored in decorate_nodes)
            decorators = getattr(func, "decorate_nodes", [])
            target = None
            for dec in decorators:
                tags = getattr(dec, "tags", {})
                if "target" in tags:
                    target = tags["target"]
                    break

            if target is None:
                continue

            # First, check for @check_output(data_type=...) in transform attribute
            transforms = getattr(func, "transform", [])
            actual_type = None
            for transform in transforms:
                if type(transform).__name__ == "check_output":
                    # check_output stores the data_type in default_validator_kwargs
                    kwargs = getattr(transform, "default_validator_kwargs", {})
                    data_type = kwargs.get("data_type")
                    if data_type is not None and dataclasses.is_dataclass(data_type):
                        actual_type = data_type
                        break

            # If no @check_output data_type, try the return annotation
            if actual_type is None:
                try:
                    hints = get_type_hints(func, globalns=vars(module))
                except Exception:
                    continue

                return_type = hints.get("return")
                if return_type is None:
                    continue

                # Handle Optional types and extract the actual type
                actual_type = return_type
                origin = getattr(return_type, "__origin__", None)
                if origin is not None:
                    args = getattr(return_type, "__args__", ())
                    # For Union[X, None], extract X
                    if args and type(None) in args:
                        non_none = [a for a in args if a is not type(None)]
                        if non_none:
                            actual_type = non_none[0]

            # Check if it's a dataclass
            if actual_type is None or not dataclasses.is_dataclass(actual_type):
                continue

            # Extract field types
            try:
                # Try to get resolved type hints, but fall back to raw field types
                try:
                    dc_hints = get_type_hints(actual_type)
                except Exception:
                    dc_hints = {}

                schema: dict[str, str] = {}
                for field in dataclasses.fields(actual_type):
                    # Use resolved hint if available, else raw annotation
                    field_type = dc_hints.get(field.name, field.type)
                    schema[field.name] = _type_to_string(field_type)
                result[target] = schema
            except Exception:
                log.debug("Failed to extract schema from %s", actual_type)
                continue

    return result


def _type_to_string(t: object) -> str:
    """Convert a type annotation to a string representation.

    Parameters
    ----------
    t
        Type annotation to convert.

    Returns
    -------
    str
        Human-readable string representation of the type.
    """
    # Handle None type
    if t is type(None):
        return "None"
    # Handle string annotations (forward references)
    if isinstance(t, str):
        return t
    # Handle classes with __name__
    if hasattr(t, "__name__"):
        return t.__name__
    # Handle generic types
    origin = getattr(t, "__origin__", None)
    if origin is not None:
        args = getattr(t, "__args__", ())
        args_str = ", ".join(_type_to_string(a) for a in args)
        origin_name = getattr(origin, "__name__", str(origin))
        return f"{origin_name}[{args_str}]"
    return str(t)


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
        """Find targets that produce the given dataset.

        Parameters
        ----------
        table_key
            Dataset to find producers for.

        Returns
        -------
        list[str]
            Target names that produce this dataset.

        Notes
        -----
        Uses Hamilton native module metadata as the source of truth.
        """
        metadata = _get_hamilton_target_metadata()
        result: list[str] = []
        for target, info in metadata.items():
            if table_key in info.get("produces", []):
                result.append(target)
        return result

    @staticmethod
    def consumers_of(table_key: str) -> list[str]:
        """Find targets that consume the given dataset.

        Parameters
        ----------
        table_key
            Dataset to find consumers for.

        Returns
        -------
        list[str]
            Target names that consume this dataset.

        Notes
        -----
        Uses Hamilton native module metadata as the source of truth.
        """
        metadata = _get_hamilton_target_metadata()
        result: list[str] = []
        for target, info in metadata.items():
            if table_key in info.get("consumes", []):
                result.append(target)
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
