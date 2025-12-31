"""Engine registry and selection helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.serving.semantic.engines.protocol import EngineContext, QueryEngine
from codeintel.serving.semantic.query_ast import ServingQuery
from codeintel.serving.semantic.routing import auto_preference

if TYPE_CHECKING:
    from collections.abc import Sequence


LOG = logging.getLogger("codeintel.serving.routing")


class EngineSelectionError(ValueError):
    """Raised when no semantic engine can satisfy a request."""


@dataclass(frozen=True, slots=True)
class QueryEngineRegistry:
    """Registry of query engines with selection helpers."""

    engines: tuple[QueryEngine, ...]

    def get(self, name: str) -> QueryEngine | None:
        """Return a registered engine by name.

        Returns
        -------
        QueryEngine | None
            Engine instance when registered, otherwise None.
        """
        normalized = name.lower()
        for engine in self.engines:
            if engine.name.lower() == normalized:
                return engine
        return None

    def select(
        self,
        *,
        preference: str,
        query: ServingQuery,
        ctx: EngineContext,
    ) -> QueryEngine:
        """Select an engine given a preference and query.

        Returns
        -------
        QueryEngine
            Selected engine instance.

        Raises
        ------
        EngineSelectionError
            If the preference is unknown or no engine can satisfy the query.
        """
        normalized = preference.lower().strip() or "auto"
        if normalized == "polars":
            normalized = "duckdb"
        if normalized == "auto":
            candidates = auto_preference(query, ctx=ctx)
            engine = self._select_first(candidates, query=query, ctx=ctx)
            _log_selection(
                preference=normalized,
                candidates=candidates,
                engine=engine.name,
                query=query,
            )
            return engine
        engine = self.get(normalized)
        if engine is None:
            msg = f"Unknown query engine preference: {preference}"
            raise EngineSelectionError(msg)
        if not engine.can_run(query, ctx=ctx):
            msg = f"Query engine {engine.name} cannot satisfy the request"
            raise EngineSelectionError(msg)
        _log_selection(
            preference=normalized,
            candidates=(engine.name,),
            engine=engine.name,
            query=query,
        )
        return engine

    def select_prefer(
        self,
        names: Sequence[str],
        *,
        query: ServingQuery,
        ctx: EngineContext,
    ) -> QueryEngine:
        """Select the first engine from names that can run the query.

        Returns
        -------
        QueryEngine
            Selected engine instance.
        """
        candidates = tuple(names)
        engine = self._select_first(candidates, query=query, ctx=ctx)
        _log_selection(
            preference="prefer",
            candidates=candidates,
            engine=engine.name,
            query=query,
        )
        return engine

    def _select_first(
        self,
        names: tuple[str, ...],
        *,
        query: ServingQuery,
        ctx: EngineContext,
    ) -> QueryEngine:
        """Return the first engine that can satisfy the query.

        Returns
        -------
        QueryEngine
            Selected engine instance.

        Raises
        ------
        EngineSelectionError
            If no registered engine can satisfy the query.
        """
        for name in names:
            engine = self.get(name)
            if engine is None:
                continue
            if engine.can_run(query, ctx=ctx):
                return engine
        msg = "No registered query engines can satisfy the request"
        raise EngineSelectionError(msg)


def _log_selection(
    *,
    preference: str,
    candidates: tuple[str, ...],
    engine: str,
    query: ServingQuery,
) -> None:
    if not LOG.isEnabledFor(logging.INFO):
        return
    LOG.info(
        "query_engine_selected",
        extra={
            "preference": preference,
            "candidates": candidates,
            "engine": engine,
            "view_id": query.spec.view_id,
            "table_key": query.spec.table_key,
        },
    )


def build_engine_registry(engines: Iterable[QueryEngine]) -> QueryEngineRegistry:
    """Build a registry from an iterable of engines.

    Returns
    -------
    QueryEngineRegistry
        Registry containing the provided engines.
    """
    return QueryEngineRegistry(tuple(engines))


__all__ = ["EngineSelectionError", "QueryEngineRegistry", "build_engine_registry"]
