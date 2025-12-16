"""Decorators for tagging view builder functions.

View builders are plain Python functions that construct Ibis expressions. To
avoid manual registries and import-side effects, we tag view builder functions
using Hamilton's `@tag` modifier and discover them via Hamilton introspection.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

from hamilton.function_modifiers import tag as h_tag

from codeintel.hamilton import tags as ht

_TFunc = TypeVar("_TFunc", bound=Callable[..., object])


def ibis_view(table_key: str) -> Callable[[_TFunc], _TFunc]:
    """Tag a function as an Ibis view builder for a specific table/view.

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
        return cast(
            "_TFunc",
            h_tag(
                output_kind=ht.OUTPUT_KIND_VIEW,
                table_key=table_key,
            )(fn),
        )

    return decorator


__all__ = ["ibis_view"]
