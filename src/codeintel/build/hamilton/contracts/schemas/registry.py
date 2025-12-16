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

import dataclasses
import logging
from functools import cache
from typing import TYPE_CHECKING, get_args, get_origin, get_type_hints

from codeintel.build.hamilton.contracts.schemas.builder import build_all_schemas
from codeintel.build.hamilton.native.loader import get_loader

if TYPE_CHECKING:
    from collections.abc import ItemsView, Iterator, ValuesView

    from codeintel.build.hamilton.contracts.schemas.schema import DatasetSchema

__all__ = [
    "SCHEMA_REGISTRY",
    "DatasetSchemaRegistry",
    "get_schema",
]

log = logging.getLogger(__name__)

_QUERY_PARAM_PREFIX = "q__"
_QUERY_PARAM_MIN_PARTS = 3
_SCHEMA_OUTPUT_TAG_KEY = "hamilton.internal.schema_output"
_TARGET_TAG_KEY = "target"


def _iter_module_callables(module: object) -> Iterator[object]:
    for attr_name in dir(module):
        value = getattr(module, attr_name, None)
        if value is None:
            continue
        if callable(value):
            yield value


def _get_decorators(func: object) -> list[object]:
    decorators = getattr(func, "decorate_nodes", None)
    if isinstance(decorators, list):
        return decorators
    if isinstance(decorators, tuple):
        return list(decorators)
    return []


def _extract_target_and_schema(decorators: list[object]) -> tuple[str | None, str | None]:
    target: str | None = None
    schema_str: str | None = None
    for decorator in decorators:
        tags = getattr(decorator, "tags", None)
        if not isinstance(tags, dict):
            continue
        target_raw = tags.get(_TARGET_TAG_KEY)
        if isinstance(target_raw, str) and target_raw:
            target = target_raw
        schema_raw = tags.get(_SCHEMA_OUTPUT_TAG_KEY)
        if isinstance(schema_raw, str) and schema_raw:
            schema_str = schema_raw
    return target, schema_str


def _extract_consumed_table_keys(func: object) -> list[str]:
    annotations = getattr(func, "__annotations__", None)
    if not isinstance(annotations, dict):
        return []

    consumes: list[str] = []
    for param_name in annotations:
        if not isinstance(param_name, str) or param_name == "return":
            continue
        if not param_name.startswith(_QUERY_PARAM_PREFIX):
            continue
        parts = param_name.split("__")
        if len(parts) < _QUERY_PARAM_MIN_PARTS:
            continue
        consumes.append(f"{parts[1]}.{parts[2]}")
    return consumes


def _find_dataclass_type_from_check_output(func: object) -> type[object] | None:
    transforms = getattr(func, "transform", None)
    if not isinstance(transforms, (list, tuple)):
        return None

    for transform in transforms:
        if type(transform).__name__ != "check_output":
            continue
        kwargs = getattr(transform, "default_validator_kwargs", None)
        if not isinstance(kwargs, dict):
            continue
        data_type = kwargs.get("data_type")
        if isinstance(data_type, type) and dataclasses.is_dataclass(data_type):
            return data_type
    return None


def _unwrap_optional_type(type_obj: object) -> object:
    origin = get_origin(type_obj)
    if origin is None:
        return type_obj

    args = [arg for arg in get_args(type_obj) if arg is not type(None)]
    if len(args) == 1:
        return args[0]
    return type_obj


def _find_dataclass_type_from_return_annotation(func: object, module: object) -> type[object] | None:
    try:
        hints = get_type_hints(func, globalns=vars(module))
    except (NameError, TypeError):
        log.debug("Failed to resolve type hints for %s", getattr(func, "__name__", "<unknown>"), exc_info=True)
        return None

    return_type = hints.get("return")
    if return_type is None:
        return None

    actual_type = _unwrap_optional_type(return_type)
    if isinstance(actual_type, type) and dataclasses.is_dataclass(actual_type):
        return actual_type
    return None


def _schema_from_dataclass(dataclass_type: type[object]) -> dict[str, str] | None:
    try:
        try:
            hints = get_type_hints(dataclass_type)
        except (NameError, TypeError):
            hints = {}

        fields_raw = getattr(dataclass_type, "__dataclass_fields__", None)
        if not isinstance(fields_raw, dict):
            return None

        schema: dict[str, str] = {}
        for field_name, field_obj in fields_raw.items():
            if not isinstance(field_name, str):
                continue
            default_type = getattr(field_obj, "type", object)
            schema[field_name] = _type_to_string(hints.get(field_name, default_type))
    except TypeError:
        log.debug("Failed to extract schema from %s", dataclass_type, exc_info=True)
        return None
    else:
        return schema


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
    result: dict[str, dict[str, list[str]]] = {}
    loader = get_loader()
    modules = loader.discover_modules()

    for module in modules:
        for func in _iter_module_callables(module):
            decorators = _get_decorators(func)
            if not decorators:
                continue

            target, schema_str = _extract_target_and_schema(decorators)
            if target is None:
                continue

            entry = result.setdefault(target, {"produces": [], "consumes": [], "schemas": []})
            entry["consumes"].extend(_extract_consumed_table_keys(func))
            if schema_str is not None:
                entry["schemas"].append(schema_str)

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
    result: dict[str, dict[str, str]] = {}
    loader = get_loader()
    modules = loader.discover_modules()

    for module in modules:
        for func in _iter_module_callables(module):
            decorators = _get_decorators(func)
            target, _ = _extract_target_and_schema(decorators)
            if target is None or target in result:
                continue

            dataclass_type = _find_dataclass_type_from_check_output(func)
            if dataclass_type is None:
                dataclass_type = _find_dataclass_type_from_return_annotation(func, module)
            if dataclass_type is None:
                continue

            schema = _schema_from_dataclass(dataclass_type)
            if schema is None:
                continue
            result[target] = schema

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
    name = getattr(t, "__name__", None)
    if isinstance(name, str):
        return name
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
