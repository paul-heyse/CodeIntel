"""Generated TypedDict row models for table-shaped data."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Protocol, cast

from codeintel.core.schemas.generated_rows import analytics as _analytics_rows
from codeintel.core.schemas.generated_rows import core as _core_rows
from codeintel.core.schemas.generated_rows import graph as _graph_rows

_ROW_SUFFIX = "Row"
_CAMEL_RE_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_RE_2 = re.compile(r"([a-z0-9])([A-Z])")


class RowModelProtocol(Protocol):
    """Protocol for generated TypedDict-style row models."""

    __annotations__: Mapping[str, object]


def _camel_to_snake(value: str) -> str:
    step1 = _CAMEL_RE_1.sub(r"\1_\2", value)
    return _CAMEL_RE_2.sub(r"\1_\2", step1).lower()


def _table_key_from_row_model_name(*, prefix: str, name: str) -> str:
    if not name.startswith(prefix) or not name.endswith(_ROW_SUFFIX):
        msg = f"Invalid row model name for prefix {prefix}: {name}"
        raise ValueError(msg)
    suffix = name[len(prefix) : -len(_ROW_SUFFIX)]
    if not suffix:
        msg = f"Row model name missing suffix: {name}"
        raise ValueError(msg)
    return f"{prefix.lower()}.{_camel_to_snake(suffix)}"


def _row_models_from_module(*, module: object, prefix: str) -> dict[str, type[RowModelProtocol]]:
    names = getattr(module, "__all__", ())
    if not isinstance(names, tuple):
        msg = f"Invalid __all__ for generated rows module: {module}"
        raise TypeError(msg)
    result: dict[str, type[RowModelProtocol]] = {}
    for name in names:
        if not isinstance(name, str):
            continue
        model = getattr(module, name, None)
        if model is None or not isinstance(model, type):
            continue
        table_key = _table_key_from_row_model_name(prefix=prefix, name=name)
        result[table_key] = cast("type[RowModelProtocol]", model)
    return result


_ROW_MODEL_BY_TABLE_KEY: dict[str, type[RowModelProtocol]] = {}
_ROW_MODEL_BY_TABLE_KEY.update(_row_models_from_module(module=_analytics_rows, prefix="Analytics"))
_ROW_MODEL_BY_TABLE_KEY.update(_row_models_from_module(module=_core_rows, prefix="Core"))
_ROW_MODEL_BY_TABLE_KEY.update(_row_models_from_module(module=_graph_rows, prefix="Graph"))


def row_model_for_table_key(table_key: str) -> type[RowModelProtocol] | None:
    """Return the generated row model for a table key, when available."""
    return _ROW_MODEL_BY_TABLE_KEY.get(table_key)


def columns_for_table_key(table_key: str) -> tuple[str, ...] | None:
    """Return ordered column names for a table key based on row models."""
    model = row_model_for_table_key(table_key)
    if model is None:
        return None
    annotations = getattr(model, "__annotations__", None)
    if not isinstance(annotations, Mapping):
        return None
    return tuple(str(name) for name in annotations.keys())


__all__ = [
    "RowModelProtocol",
    "columns_for_table_key",
    "row_model_for_table_key",
]
