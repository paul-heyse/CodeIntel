"""Search service for the advanced query engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tools.advanced_query_engine.context import SearchContext
from tools.advanced_query_engine.contracts import QueryBudget, QueryRequest, QueryResponse
from tools.advanced_query_engine.handlers.registry import HANDLERS
from tools.advanced_query_engine.packs.catalog import build_pack_catalog
from tools.advanced_query_engine.util.snippets import SnippetConfig


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for SearchService."""

    repo_root: Path
    query_pack_root: Path
    wiring_pack_root: Path
    default_budget: QueryBudget


class SearchService:
    """Dispatch advanced query types against a repository."""

    def __init__(self, config: SearchConfig) -> None:
        self._config = config
        self._query_catalog = build_pack_catalog(config.query_pack_root)
        self._wiring_catalog = build_pack_catalog(config.wiring_pack_root)

    @classmethod
    def from_repo(
        cls, repo_root: Path, *, default_budget: QueryBudget | None = None
    ) -> SearchService:
        """Construct a service using repository-relative default pack roots.

        Returns
        -------
        SearchService
            Configured search service instance.
        """
        resolved = repo_root.resolve()
        query_pack_root = resolved / "docs" / "advanced_query_engine" / "query_packs"
        wiring_pack_root = resolved / "docs" / "advanced_query_engine" / "wiring_packs" / "packs"
        budget = default_budget or QueryBudget()
        return cls(
            SearchConfig(
                repo_root=resolved,
                query_pack_root=query_pack_root,
                wiring_pack_root=wiring_pack_root,
                default_budget=budget,
            )
        )

    def run(self, request: QueryRequest) -> QueryResponse:
        """Execute a query request and return a response.

        Returns
        -------
        QueryResponse
            Query response payload.

        Raises
        ------
        ValueError
            If the repo root does not match or the query type is unsupported.
        """
        repo_root = Path(request.repo_root).resolve()
        if repo_root != self._config.repo_root:
            msg = "SearchService repo_root does not match the request repo_root"
            raise ValueError(msg)

        budget = request.budget or self._config.default_budget
        handler = HANDLERS.get(request.type)
        if handler is None:
            msg = f"Unsupported query type: {request.type}"
            raise ValueError(msg)

        snippet_config = SnippetConfig(
            before_lines=budget.context_lines,
            after_lines=budget.context_lines,
        )
        context = SearchContext(
            repo_root=repo_root,
            query_catalog=self._query_catalog,
            wiring_catalog=self._wiring_catalog,
            snippet_config=snippet_config,
            default_budget=budget,
        )
        return handler(request, context)


__all__ = ["SearchConfig", "SearchService"]
