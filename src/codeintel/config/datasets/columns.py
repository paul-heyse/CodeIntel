"""Column and SQL helpers decoupled from legacy SQL builder APIs."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, TypeVar, cast

from codeintel.core.imports.lazy import lazy_getattr
from codeintel.core.schemas.service import get_schema_service

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from codeintel.core.schemas.service import SchemaService

_Column = TypeVar("_Column", bound=str)


def _resolve_schema_service() -> SchemaService:
    """Return a configured SchemaService, falling back to build defaults.

    Returns
    -------
    SchemaService
        Schema service configured for column enumeration.
    """
    try:
        return get_schema_service()
    except RuntimeError:
        service_factory = cast(
            "Callable[[], SchemaService]",
            lazy_getattr("codeintel.build.schemas.service", "get_schema_service"),
        )
        return service_factory()


@lru_cache(maxsize=1)
def load_columns_by_table() -> dict[str, list[str]]:
    """Return registry column lists keyed by table.

    Returns
    -------
    dict[str, list[str]]
        Mapping of table key to column names from the active schema service.
    """
    service = _resolve_schema_service()
    return {
        schema.table_key: list(schema.column_names()) for schema in service.iter_table_schemas()
    }


def serialize_row(row: Mapping[_Column, object], columns: Sequence[_Column]) -> tuple[object, ...]:
    """Serialize a mapping using a stable column sequence.

    Returns
    -------
    tuple[object, ...]
        Row values ordered according to the provided columns.
    """
    return tuple(row[column] for column in columns)


__all__ = [
    "load_columns_by_table",
    "serialize_row",
]
