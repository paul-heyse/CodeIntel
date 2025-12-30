"""Column and SQL helpers decoupled from legacy SQL builder APIs."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, TypeVar, cast

from codeintel.core.schemas.row_serialization import row_to_tuple, row_to_tuple_by_columns
from codeintel.core.schemas.service import get_schema_service

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_Column = TypeVar("_Column", bound=str)


@lru_cache(maxsize=1)
def load_columns_by_table() -> dict[str, list[str]]:
    """Return registry column lists keyed by table.

    Returns
    -------
    dict[str, list[str]]
        Mapping of table key to column names from the active schema service.
    """
    service = get_schema_service()
    return {
        schema.table_key: list(schema.column_names()) for schema in service.iter_table_schemas()
    }


def serialize_row(
    row: Mapping[_Column, object],
    columns: Sequence[_Column] | None,
    *,
    table_key: str | None = None,
) -> tuple[object, ...]:
    """Serialize a mapping using schema-backed or explicit columns.

    Parameters
    ----------
    row
        Row mapping keyed by column name.
    columns
        Explicit column order when table_key is not provided.
    table_key
        Optional table key for schema-backed serialization.

    Returns
    -------
    tuple[object, ...]
        Row values ordered according to the provided columns.

    Raises
    ------
    ValueError
        If columns are missing when table_key is not provided.
    """
    if table_key is not None:
        return row_to_tuple(table_key, cast("Mapping[str, object]", row))
    if columns is None:
        msg = "columns must be provided when table_key is not set"
        raise ValueError(msg)
    return row_to_tuple_by_columns(row, columns)


__all__ = [
    "load_columns_by_table",
    "serialize_row",
]
