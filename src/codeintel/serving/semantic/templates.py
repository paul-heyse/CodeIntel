"""Typed query templates for semantic execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class DbApiQuery:
    """Raw SQL + positional parameters for DB-API execution."""

    sql: str
    params: Sequence[object] | None = None


@dataclass(frozen=True, slots=True)
class DbApiTemplate:
    """Reusable DB-API query shape (SQL string + params bound at runtime)."""

    sql: str

    def bind(self, params: Sequence[object] | None = None) -> DbApiQuery:
        """Bind positional parameters for DB-API execution.

        Parameters
        ----------
        params
            Positional parameter values for ``?`` placeholders in ``sql``.

        Returns
        -------
        DbApiQuery
            Bound query ready for DB-API execution.
        """
        return DbApiQuery(sql=self.sql, params=params)


__all__ = [
    "DbApiQuery",
    "DbApiTemplate",
]
