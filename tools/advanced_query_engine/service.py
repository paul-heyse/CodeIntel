"""Search service for the advanced query engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tools.advanced_query_engine.analytics as aqe_analytics
from tools.advanced_query_engine.context import SearchContext
from tools.advanced_query_engine.contracts import (
    JSONValue,
    QueryBudget,
    QueryRequest,
    QueryResponse,
)
from tools.advanced_query_engine.handlers.registry import HANDLERS
from tools.advanced_query_engine.packs.catalog import build_pack_catalog
from tools.advanced_query_engine.storage.arrow_store import (
    PersistResult,
    match_record_schema,
    persist_query_response,
    schema_compatibility_issues,
    wiring_edge_schema,
)
from tools.advanced_query_engine.util.snippets import SnippetConfig


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for SearchService."""

    repo_root: Path
    query_pack_root: Path
    wiring_pack_root: Path
    default_budget: QueryBudget
    enable_persistence: bool = False
    persist_root: Path | None = None
    enable_analytics: bool = False
    enable_validation: bool = False


@dataclass(frozen=True)
class RepoServiceOptions:
    """Overrides for SearchService.from_repo defaults."""

    default_budget: QueryBudget | None = None
    enable_persistence: bool = False
    persist_root: Path | None = None
    enable_analytics: bool = False
    enable_validation: bool = False


class SearchService:
    """Dispatch advanced query types against a repository."""

    def __init__(self, config: SearchConfig) -> None:
        self._config = config
        self._query_catalog = build_pack_catalog(config.query_pack_root)
        self._wiring_catalog = build_pack_catalog(config.wiring_pack_root)

    @classmethod
    def from_repo(
        cls,
        repo_root: Path,
        options: RepoServiceOptions | None = None,
    ) -> SearchService:
        """Construct a service using repository-relative default pack roots.

        Returns
        -------
        SearchService
            Configured search service instance.
        """
        resolved = repo_root.resolve()
        settings = options or RepoServiceOptions()
        budget = settings.default_budget or QueryBudget()
        query_pack_root = resolved / "docs" / "advanced_query_engine" / "query_packs"
        wiring_pack_root = resolved / "docs" / "advanced_query_engine" / "wiring_packs" / "packs"
        return cls(
            SearchConfig(
                repo_root=resolved,
                query_pack_root=query_pack_root,
                wiring_pack_root=wiring_pack_root,
                default_budget=budget,
                enable_persistence=settings.enable_persistence,
                persist_root=settings.persist_root,
                enable_analytics=settings.enable_analytics,
                enable_validation=settings.enable_validation,
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
        response = handler(request, context)
        options = request.options or {}
        analytics_enabled = _option_bool(
            options.get("analytics"),
            default=self._config.enable_analytics,
        )
        validation_enabled = _option_bool(
            options.get("validate_persisted"),
            default=self._config.enable_validation,
        )
        persist_result = self._persist_if_requested(request, response)
        if persist_result is None:
            if analytics_enabled or validation_enabled:
                debug = dict(response.debug)
                debug["analytics"] = {
                    "error": "Persistence is required for analytics and validation.",
                }
                return QueryResponse(
                    summary=response.summary,
                    primary=response.primary,
                    related=response.related,
                    debug=debug,
                )
            return response
        debug = dict(response.debug)
        debug["persist"] = persist_result.to_dict()
        analytics = self._analyze_if_requested(
            request,
            persist_result,
            budget,
            analytics_enabled=analytics_enabled,
            validation_enabled=validation_enabled,
        )
        if analytics is not None:
            debug["analytics"] = analytics
        return QueryResponse(
            summary=response.summary,
            primary=response.primary,
            related=response.related,
            debug=debug,
        )

    def _persist_if_requested(
        self,
        request: QueryRequest,
        response: QueryResponse,
    ) -> PersistResult | None:
        if not self._config.enable_persistence:
            return None
        options = request.options or {}
        persist_flag = options.get("persist")
        persist_path = options.get("persist_path")
        if not persist_flag and persist_path is None:
            return None
        root = self._persist_root(persist_path)
        partition_by = options.get("persist_partition_by")
        partition_list = _partition_list(partition_by)
        return persist_query_response(
            request=request,
            response=response,
            output_root=root,
            partition_by=partition_list,
        )

    def _persist_root(self, persist_path: object) -> Path:
        if isinstance(persist_path, str) and persist_path:
            return Path(persist_path).resolve()
        if self._config.persist_root is not None:
            return self._config.persist_root
        return self._config.repo_root / "build" / "advanced_query_engine"

    @staticmethod
    def _analyze_if_requested(
        request: QueryRequest,
        persist_result: PersistResult,
        budget: QueryBudget,
        *,
        analytics_enabled: bool,
        validation_enabled: bool,
    ) -> dict[str, JSONValue] | None:
        if not analytics_enabled and not validation_enabled:
            return None
        if persist_result.schema_name == "match_records":
            expected_schema = match_record_schema()
            model = aqe_analytics.MatchRecordModel
            lf = aqe_analytics.scan_match_records(persist_result.path)
        elif persist_result.schema_name == "wiring_edges":
            expected_schema = wiring_edge_schema()
            model = aqe_analytics.WiringEdgeModel
            lf = aqe_analytics.scan_wiring_edges(persist_result.path)
        else:
            return {"error": f"Unsupported schema: {persist_result.schema_name}."}

        try:
            schema_issues = schema_compatibility_issues(persist_result.path, expected_schema)
        except FileNotFoundError as exc:
            return {"schema_compatible": False, "schema_issues": [str(exc)]}
        if schema_issues:
            return {"schema_compatible": False, "schema_issues": schema_issues}

        options = request.options or {}
        chunk_size = _int_option(options.get("analytics_chunk_size"), 1000)
        max_rows = _optional_int(options.get("analytics_max_rows"))
        if max_rows is None:
            stream_result = aqe_analytics.stream_with_budget(
                lf, budget=budget, chunk_size=chunk_size
            )
        else:
            stream_result = aqe_analytics.stream_batches(
                lf, chunk_size=chunk_size, max_rows=max_rows
            )

        summary: dict[str, JSONValue] = {
            "schema_compatible": True,
            "schema_name": persist_result.schema_name,
            "rows_seen": stream_result.rows_seen,
            "batch_count": len(stream_result.batches),
            "budget_exhausted": stream_result.budget_exhausted,
        }
        profile_enabled = _option_bool(options.get("analytics_profile"), default=False)
        if analytics_enabled and profile_enabled:
            _, profile_df = aqe_analytics.profile_query(lf)
            summary["profile"] = _sanitize_rows(profile_df.to_dicts())
        if validation_enabled:
            aqe_analytics.validate_batches(stream_result.batches, model=model)
            summary["validation"] = {
                "status": "ok",
                "rows_validated": stream_result.rows_seen,
            }
        return summary


def _partition_list(value: object) -> list[str] | None:
    if not value:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return None


def _option_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _int_option(value: object, default: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    return default


def _optional_int(value: object) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    return None


def _sanitize_rows(rows: list[dict[str, object]]) -> list[dict[str, JSONValue]]:
    return [_json_sanitize(row) for row in rows]


def _json_sanitize(value: object) -> JSONValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_sanitize(val) for key, val in value.items()}
    return str(value)


__all__ = ["RepoServiceOptions", "SearchConfig", "SearchService"]
