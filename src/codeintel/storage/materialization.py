"""Centralized view materialization helpers for DuckDB relations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import DuckDBRelation


def create_view_from_relation(
    *,
    relation: DuckDBRelation,
    view_name: str,
    replace: bool = True,
) -> None:
    """Create or replace a DuckDB view from a relation.

    Parameters
    ----------
    relation
        DuckDB relation to expose as a view.
    view_name
        View name scoped to the current schema/catalog.
    replace
        When True, replace any existing view with the same name.

    Raises
    ------
    ValueError
        If view_name is empty.
    """
    if not view_name:
        msg = "view_name is required to create a view"
        raise ValueError(msg)
    relation.create_view(view_name, replace=replace)


__all__ = ["create_view_from_relation"]
