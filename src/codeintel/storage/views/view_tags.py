"""Decorators for tagging view builder functions.

View builders are plain Python functions that construct SQLGlot expressions. To
avoid manual registries and import-side effects, we tag view builder functions
using Hamilton's `@tag` modifier and discover them via Hamilton introspection.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.tagging_helpers import apply_raw_tags

_TFunc = TypeVar("_TFunc", bound=Callable[..., object])


def sql_view(table_key: str) -> Callable[[_TFunc], _TFunc]:
    """Tag a function as a SQLGlot view builder for a specific table/view.

    Parameters
    ----------
    table_key
        Fully qualified view name (schema.view).

    Returns
    -------
    Callable[[Callable[..., object]], Callable[..., object]]
        Decorator that applies Hamilton view tags.
    """

    def decorator(fn: _TFunc) -> _TFunc:
        return apply_raw_tags(
            fn,
            tags={
                ht.TAG_OUTPUT_KIND: ht.OUTPUT_KIND_VIEW,
                ht.TAG_TABLE_KEY: table_key,
            },
        )

    return decorator


view_builder = sql_view


__all__ = ["sql_view", "view_builder"]
