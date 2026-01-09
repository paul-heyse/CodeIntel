"""Helpers for list ordering semantics in analytics outputs."""

from __future__ import annotations

from collections.abc import Iterable

from codeintel.build.graphs.rx.normalize import stable_key


def normalize_list_semantics[T](values: Iterable[T] | None) -> list[T]:
    """Return a list with stable ordering when list order is semantically relevant.

    Returns
    -------
    list[T]
        Sorted list preserving semantic ordering.
    """
    if values is None:
        return []
    return sorted(values, key=stable_key)


__all__ = ["normalize_list_semantics"]
