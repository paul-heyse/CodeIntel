"""Column and SQL helpers decoupled from legacy SQL builder APIs."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, TypeVar

from codeintel.config.datasets.declared_schemas import TABLE_SCHEMAS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_Column = TypeVar("_Column", bound=str)


@lru_cache(maxsize=1)
def load_columns_by_table() -> dict[str, list[str]]:
    """Return registry column lists keyed by table.

    Returns
    -------
    dict[str, list[str]]
        Mapping of table key to column names.
    """
    return {
        table_key: [col.name for col in schema.columns] for table_key, schema in TABLE_SCHEMAS.items()
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
