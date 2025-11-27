"""
Shared DuckDB query service used by MCP backends and FastAPI surface.

All SQL queries against docs.* and analytics.* views/tables live here.
Other modules must call this service (via LocalQueryService/QueryService)
instead of issuing custom SELECTs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from codeintel.graphs.engine import NxGraphEngine
from codeintel.graphs.engine_factory import build_graph_engine
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    DatasetRowsResponse,
    FileHintsResponse,
    FileProfileResponse,
    FileSummaryResponse,
    FunctionArchitectureResponse,
    FunctionProfileResponse,
    FunctionSummaryResponse,
    GraphNeighborhoodResponse,
    HighRiskFunctionsResponse,
    ImportBoundaryResponse,
    Message,
    ModuleArchitectureResponse,
    ModuleProfileResponse,
    ModuleSubsystemResponse,
    ResponseMeta,
    SubsystemModulesResponse,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
    TestsForFunctionResponse,
    ViewRow,
)
from codeintel.storage.gateway import DuckDBConnection, StorageGateway

RowDict = dict[str, object]


@dataclass(frozen=True)
class BackendLimits:
    """Safety limits applied uniformly across backends."""

    default_limit: int = 50
    max_rows_per_call: int = 500

    @classmethod
    def from_config(cls, cfg: object) -> BackendLimits:
        """
        Build limits from configuration objects exposing default_limit/max_rows_per_call.

        Parameters
        ----------
        cfg :
            Configuration object with optional `default_limit` and `max_rows_per_call` attributes.

        Returns
        -------
        BackendLimits
            Limits derived from the provided configuration.
        """
        default = getattr(cfg, "default_limit", cls.default_limit)
        maximum = getattr(cfg, "max_rows_per_call", cls.max_rows_per_call)
        return cls(default_limit=int(default), max_rows_per_call=int(maximum))


@dataclass(frozen=True)
class ClampResult:
    """Result of clamping limit/offset values with messaging."""

    applied: int
    messages: list[Message] = field(default_factory=list)
    has_error: bool = False


def clamp_limit_value(requested: int | None, *, default: int, max_limit: int) -> ClampResult:
    """
    Clamp a requested limit to safe bounds, returning warnings instead of errors.

    Parameters
    ----------
    requested:
        Requested limit value (may be None).
    default:
        Default limit to apply when requested is None.
    max_limit:
        Maximum allowable limit.

    Returns
    -------
    ClampResult
        Applied limit plus any informational messages.
    """
    messages: list[Message] = []
    limit = default if requested is None else requested

    if limit < 0:
        messages.append(
            Message(
                code="limit_invalid",
                severity="error",
                detail="limit must be non-negative",
                context={"requested": limit},
            )
        )
        return ClampResult(applied=0, messages=messages, has_error=True)

    if limit > max_limit:
        messages.append(
            Message(
                code="limit_clamped",
                severity="warning",
                detail=f"Requested {limit} rows; delivering {max_limit} (max allowed).",
                context={"requested": limit, "applied": max_limit, "max": max_limit},
            )
        )
        limit = max_limit

    return ClampResult(applied=limit, messages=messages, has_error=False)


def clamp_offset_value(offset: int) -> ClampResult:
    """
    Clamp an offset to a non-negative value, returning messaging instead of raising.

    Parameters
    ----------
    offset:
        Requested offset value.

    Returns
    -------
    ClampResult
        Applied offset and any validation messages.
    """
    if offset < 0:
        return ClampResult(
            applied=0,
            messages=[
                Message(
                    code="offset_invalid",
                    severity="error",
                    detail="offset must be non-negative",
                    context={"requested": offset},
                )
            ],
            has_error=True,
        )
    return ClampResult(applied=offset)


def _fetch_one_dict(cur: DuckDBConnection, sql: str, params: list[object]) -> RowDict | None:
    result = cur.execute(sql, params)
    row = result.fetchone()
    if row is None:
        return None
    cols = [desc[0] for desc in result.description]
    return {col: row[idx] for idx, col in enumerate(cols)}


def _fetch_all_dicts(cur: DuckDBConnection, sql: str, params: list[object]) -> list[RowDict]:
    result = cur.execute(sql, params)
    rows = result.fetchall()
    cols = [desc[0] for desc in result.description]
    return [{col: row[idx] for idx, col in enumerate(cols)} for row in rows]


def _normalize_entrypoints(rows: list[RowDict]) -> None:
    """Ensure entrypoints_json is always a list for downstream consumers."""
    for row in rows:
        if row.get("entrypoints_json") is None:
            row["entrypoints_json"] = []


def _normalize_entrypoints_dict(row: RowDict | None) -> None:
    """Ensure a single row has a non-null entrypoints_json list."""
    if row is None:
        return
    if row.get("entrypoints_json") is None:
        row["entrypoints_json"] = []


@dataclass  # noqa: PLR0904
class DuckDBQueryService:
    """Shared query runner for DuckDB-backed MCP and FastAPI surfaces."""

    gateway: StorageGateway
    repo: str
    commit: str
    limits: BackendLimits
    engine: NxGraphEngine | None = None
    _engine: NxGraphEngine | None = field(default=None, init=False, repr=False)

    @property
    def con(self) -> DuckDBConnection:
        """Underlying DuckDB connection."""
        return self.gateway.con

    def _graph_engine(self) -> NxGraphEngine:
        """
        Return a memoized graph engine for this snapshot.

        Returns
        -------
        NxGraphEngine
            Engine configured for the service repo/commit.
        """
        if self._engine is None:
            self._engine = self.engine or build_graph_engine(self.gateway, (self.repo, self.commit))
        return self._engine

    def graph_engine(self) -> NxGraphEngine:
        """
        Public accessor for the graph engine.

        Returns
        -------
        NxGraphEngine
            Cached or newly built graph engine for this service.
        """
        return self._graph_engine()

    def _resolve_function_goid(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> int | None:
        if goid_h128 is not None:
            return goid_h128

        if urn:
            row = _fetch_one_dict(
                self.con,
                """
                SELECT function_goid_h128
                FROM docs.v_function_summary
                WHERE repo = ? AND commit = ? AND urn = ?
                LIMIT 1
                """,
                [self.repo, self.commit, urn],
            )
        elif rel_path and qualname:
            row = _fetch_one_dict(
                self.con,
                """
                SELECT function_goid_h128
                FROM docs.v_function_summary
                WHERE repo = ? AND commit = ? AND rel_path = ? AND qualname = ?
                LIMIT 1
                """,
                [self.repo, self.commit, rel_path, qualname],
            )
        else:
            row = None

        if not row:
            return None
        value = row.get("function_goid_h128")
        if value is None:
            return None
        if isinstance(value, (int, float, str)):
            return int(value)
        message = f"Unexpected goid type: {type(value)!r}"
        raise errors.backend_failure(message)

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> FunctionSummaryResponse:
        """
        Return a function summary row from docs.v_function_summary.

        Parameters
        ----------
        urn, goid_h128, rel_path, qualname :
            Identifiers used to locate the function.

        Returns
        -------
        FunctionSummaryResponse
            Summary payload and metadata.

        Raises
        ------
        errors.invalid_argument
            If no identifier is provided.
        """
        meta = ResponseMeta()
        summary: ViewRow | None = None
        if urn:
            row = _fetch_one_dict(
                self.con,
                """
                SELECT *
                FROM docs.v_function_summary
                WHERE repo = ? AND commit = ? AND urn = ?
                LIMIT 1
                """,
                [self.repo, self.commit, urn],
            )
            summary = ViewRow.model_validate(row) if row else None
        elif goid_h128 is not None:
            row = _fetch_one_dict(
                self.con,
                """
                SELECT *
                FROM docs.v_function_summary
                WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
                LIMIT 1
                """,
                [self.repo, self.commit, goid_h128],
            )
            summary = ViewRow.model_validate(row) if row else None
        elif rel_path and qualname:
            row = _fetch_one_dict(
                self.con,
                """
                SELECT *
                FROM docs.v_function_summary
                WHERE repo = ? AND commit = ? AND rel_path = ? AND qualname = ?
                LIMIT 1
                """,
                [self.repo, self.commit, rel_path, qualname],
            )
            summary = ViewRow.model_validate(row) if row else None
        else:
            message = "Must provide urn or goid_h128 or (rel_path + qualname)."
            raise errors.invalid_argument(message)

        if summary is None:
            meta.messages.append(
                Message(
                    code="not_found",
                    severity="info",
                    detail="Function not found",
                    context={
                        "urn": urn,
                        "goid_h128": goid_h128,
                        "rel_path": rel_path,
                        "qualname": qualname,
                    },
                )
            )
            return FunctionSummaryResponse(found=False, summary=None, meta=meta)

        return FunctionSummaryResponse(found=True, summary=summary, meta=meta)

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
    ) -> HighRiskFunctionsResponse:
        """
        List high-risk functions using analytics.goid_risk_factors.

        Parameters
        ----------
        min_risk:
            Minimum risk score threshold.
        limit:
            Maximum number of rows to return.
        tested_only:
            When True, restrict to functions with test coverage.

        Returns
        -------
        HighRiskFunctionsResponse
            Functions, truncation flag, and metadata.
        """
        applied_limit = self.limits.default_limit if limit is None else limit
        extra_clause = " AND tested = TRUE" if tested_only else ""
        clamp = clamp_limit_value(
            applied_limit,
            default=applied_limit,
            max_limit=self.limits.max_rows_per_call,
        )
        meta = ResponseMeta(
            requested_limit=limit,
            applied_limit=clamp.applied,
            messages=list(clamp.messages),
        )
        if clamp.has_error:
            return HighRiskFunctionsResponse(functions=[], truncated=False, meta=meta)

        params: list[object] = [self.repo, self.commit, min_risk, clamp.applied]
        sql = (
            """
            SELECT
                function_goid_h128,
                urn,
                rel_path,
                qualname,
                risk_score,
                risk_level,
                coverage_ratio,
                tested,
                complexity_bucket,
                typedness_bucket,
                hotspot_score
            FROM analytics.goid_risk_factors
            WHERE repo = ? AND commit = ? AND risk_score >= ?"""
            + extra_clause
            + """
            ORDER BY risk_score DESC
            LIMIT ?
            """
        )
        rows = _fetch_all_dicts(self.con, sql, params)
        models = [ViewRow.model_validate(r) for r in rows]
        truncated = clamp.applied > 0 and len(rows) == clamp.applied
        meta.truncated = truncated
        return HighRiskFunctionsResponse(functions=models, truncated=truncated, meta=meta)

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
    ) -> CallGraphNeighborsResponse:
        """
        Return incoming and outgoing call graph neighbors.

        Parameters
        ----------
        goid_h128:
            Caller or callee GOID.
        direction:
            Whether to fetch incoming, outgoing, or both edge directions.
        limit:
            Maximum number of rows per direction.

        Returns
        -------
        CallGraphNeighborsResponse
            Neighboring edges with metadata.

        Raises
        ------
        errors.invalid_argument
            If the direction argument is invalid.
        """
        if direction not in {"in", "out", "both"}:
            message = "direction must be one of {'in','out','both'}"
            raise errors.invalid_argument(message)
        applied_limit = self.limits.default_limit if limit is None else limit
        clamp = clamp_limit_value(
            applied_limit,
            default=applied_limit,
            max_limit=self.limits.max_rows_per_call,
        )
        meta = ResponseMeta(
            requested_limit=limit,
            applied_limit=clamp.applied,
            messages=list(clamp.messages),
        )
        if clamp.has_error:
            return CallGraphNeighborsResponse(outgoing=[], incoming=[], meta=meta)
        outgoing: list[ViewRow] = []
        incoming: list[ViewRow] = []

        if direction in {"out", "both"}:
            out_sql = """
                SELECT *
                FROM docs.v_call_graph_enriched
                WHERE caller_goid_h128 = ?
                  AND caller_repo = ?
                  AND caller_commit = ?
                ORDER BY callee_qualname
                LIMIT ?
            """
            out_rows = _fetch_all_dicts(
                self.con,
                out_sql,
                [goid_h128, self.repo, self.commit, clamp.applied],
            )
            outgoing = [ViewRow.model_validate(r) for r in out_rows]

        if direction in {"in", "both"}:
            in_sql = """
                SELECT *
                FROM docs.v_call_graph_enriched
                WHERE callee_goid_h128 = ?
                  AND callee_repo = ?
                  AND callee_commit = ?
                ORDER BY caller_qualname
                LIMIT ?
            """
            in_rows = _fetch_all_dicts(
                self.con,
                in_sql,
                [goid_h128, self.repo, self.commit, clamp.applied],
            )
            incoming = [ViewRow.model_validate(r) for r in in_rows]

        meta.truncated = clamp.applied > 0 and (
            len(outgoing) == clamp.applied or len(incoming) == clamp.applied
        )
        return CallGraphNeighborsResponse(outgoing=outgoing, incoming=incoming, meta=meta)

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
    ) -> TestsForFunctionResponse:
        """
        List tests that exercised a given function.

        Parameters
        ----------
        goid_h128, urn:
            Function identifiers.
        limit:
            Maximum number of test rows to return.

        Returns
        -------
        TestsForFunctionResponse
            Tests and metadata, or empty results when not found.

        Raises
        ------
        errors.invalid_argument
            If no function identifier is provided.
        """
        if goid_h128 is None and urn is None:
            message = "Must provide goid_h128 or urn."
            raise errors.invalid_argument(message)

        applied_limit = self.limits.default_limit if limit is None else limit
        clamp = clamp_limit_value(
            applied_limit,
            default=applied_limit,
            max_limit=self.limits.max_rows_per_call,
        )
        meta = ResponseMeta(
            requested_limit=limit,
            applied_limit=clamp.applied,
            messages=list(clamp.messages),
        )
        if clamp.has_error:
            return TestsForFunctionResponse(tests=[], meta=meta)

        columns = {
            col[1]
            for col in self.con.execute("PRAGMA table_info('docs.v_test_to_function')").fetchall()
        }
        repo_field = (
            "test_repo" if "test_repo" in columns else ("repo" if "repo" in columns else None)
        )
        commit_field = (
            "test_commit"
            if "test_commit" in columns
            else ("commit" if "commit" in columns else None)
        )

        where_clauses = []
        params: list[object] = []

        if repo_field is not None:
            where_clauses.append(f"{repo_field} = ?")
            params.append(self.repo)
        if commit_field is not None:
            where_clauses.append(f"{commit_field} = ?")
            params.append(self.commit)

        if goid_h128 is not None:
            where_clauses.append("function_goid_h128 = ?")
            params.append(goid_h128)
        else:
            where_clauses.append("urn = ?")
            params.append(urn)

        where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"
        sql = "\n".join(
            [
                "SELECT *",
                "FROM docs.v_test_to_function",
                "WHERE " + where_sql,
                "ORDER BY test_id",
                "LIMIT ?",
            ]
        )
        rows = _fetch_all_dicts(self.con, sql, [*params, clamp.applied])
        meta.truncated = clamp.applied > 0 and len(rows) == clamp.applied
        if not rows:
            meta.messages.append(
                Message(
                    code="not_found",
                    severity="info",
                    detail="No tests found for function",
                    context={"urn": urn, "goid_h128": goid_h128},
                )
            )
        return TestsForFunctionResponse(tests=[ViewRow.model_validate(r) for r in rows], meta=meta)

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> GraphNeighborhoodResponse:
        """
        Return a bounded ego neighborhood in the call graph.

        Parameters
        ----------
        goid_h128 : int
            Node identifier to center the neighborhood on.
        radius : int, optional
            Hop distance to traverse when building the ego graph.
        max_nodes : int, optional
            Maximum nodes to return; defaults to backend limits when omitted.

        Returns
        -------
        GraphNeighborhoodResponse
            Nodes, edges, and metadata describing truncation and limits.

        Raises
        ------
        errors.invalid_argument
            If the requested max_nodes is negative.
        """
        clamp = clamp_limit_value(
            max_nodes,
            default=self.limits.default_limit,
            max_limit=self.limits.max_rows_per_call,
        )
        if clamp.has_error:
            message = "max_nodes must be non-negative"
            raise errors.invalid_argument(message)
        limit = clamp.applied
        meta_messages = list(clamp.messages)
        graph = self._graph_engine().call_graph()
        if goid_h128 not in graph:
            return GraphNeighborhoodResponse(
                nodes=[],
                edges=[],
                meta=ResponseMeta(
                    requested_limit=max_nodes,
                    applied_limit=limit,
                    messages=meta_messages,
                ),
            )
        ego = nx.ego_graph(graph, goid_h128, radius=radius)
        truncated = ego.number_of_nodes() > limit
        nodes = list(ego.nodes)[:limit]
        ego = ego.subgraph(nodes).copy()
        node_rows = [
            ViewRow.model_validate(
                {"id": n, "in_degree": ego.in_degree(n), "out_degree": ego.out_degree(n)}
            )
            for n in nodes
        ]
        edge_rows = [
            ViewRow.model_validate({"src": u, "dst": v, "weight": data.get("weight", 1)})
            for u, v, data in ego.edges(data=True)
        ]
        if truncated:
            meta_messages.append(
                Message(
                    code="truncated",
                    severity="warning",
                    detail="Neighborhood truncated to max_nodes",
                    context={"max_nodes": limit},
                )
            )
        meta = ResponseMeta(
            requested_limit=max_nodes,
            applied_limit=limit,
            truncated=truncated,
            messages=meta_messages,
        )
        return GraphNeighborhoodResponse(nodes=node_rows, edges=edge_rows, meta=meta)

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> ImportBoundaryResponse:
        """
        Return import edges that cross a subsystem boundary.

        Parameters
        ----------
        subsystem_id : str
            Subsystem identifier whose external edges are requested.
        max_edges : int, optional
            Maximum number of edges to return; defaults to backend limits.

        Returns
        -------
        ImportBoundaryResponse
            Subgraph capturing boundary-crossing nodes and edges.

        Raises
        ------
        errors.invalid_argument
            If the requested max_edges is negative.
        """
        clamp = clamp_limit_value(
            max_edges,
            default=self.limits.default_limit,
            max_limit=self.limits.max_rows_per_call,
        )
        if clamp.has_error:
            message = "max_edges must be non-negative"
            raise errors.invalid_argument(message)
        limit = clamp.applied
        meta_messages = list(clamp.messages)
        import_graph = self._graph_engine().import_graph()
        memberships = dict(
            self.con.execute(
                """
                SELECT module, subsystem_id
                FROM analytics.subsystem_modules
                WHERE repo = ? AND commit = ?
                """,
                [self.repo, self.commit],
            ).fetchall()
        )
        edges: list[tuple[str, str, dict[str, object]]] = []
        for src, dst, data in import_graph.edges(data=True):
            left = memberships.get(src)
            right = memberships.get(dst)
            if left is None or right is None:
                continue
            if (left == subsystem_id and right != subsystem_id) or (
                right == subsystem_id and left != subsystem_id
            ):
                edges.append((src, dst, data))
        truncated = len(edges) > limit
        edges = edges[:limit]
        nodes = {src for src, _, _ in edges} | {dst for _, dst, _ in edges}
        node_rows = [
            ViewRow.model_validate({"module": n, "subsystem_id": memberships.get(n)}) for n in nodes
        ]
        edge_rows = [
            ViewRow.model_validate(
                {
                    "src": src,
                    "dst": dst,
                    "weight": data.get("weight", 1),
                    "src_sub": memberships.get(src),
                    "dst_sub": memberships.get(dst),
                }
            )
            for src, dst, data in edges
        ]
        if truncated:
            meta_messages.append(
                Message(
                    code="truncated",
                    severity="warning",
                    detail="Boundary edges truncated to max_edges",
                    context={"max_edges": limit},
                )
            )
        meta = ResponseMeta(
            requested_limit=max_edges,
            applied_limit=limit,
            truncated=truncated,
            messages=meta_messages,
        )
        return ImportBoundaryResponse(nodes=node_rows, edges=edge_rows, meta=meta)

    def get_file_summary(
        self,
        *,
        rel_path: str,
    ) -> FileSummaryResponse:
        """
        Return file summary plus function summaries for a path.

        Parameters
        ----------
        rel_path:
            Repo-relative path for the file.

        Returns
        -------
        FileSummaryResponse
            Summary payload indicating whether the file was found.
        """
        file_row = _fetch_one_dict(
            self.con,
            """
            SELECT *
            FROM docs.v_file_summary
            WHERE rel_path = ?
              AND repo = ?
              AND commit = ?
            LIMIT 1
            """,
            [rel_path, self.repo, self.commit],
        )
        if not file_row:
            meta = ResponseMeta(
                messages=[
                    Message(
                        code="not_found",
                        severity="info",
                        detail="File not found",
                        context={"rel_path": rel_path},
                    )
                ]
            )
            return FileSummaryResponse(found=False, file=None, meta=meta)

        funcs = _fetch_all_dicts(
            self.con,
            """
            SELECT *
            FROM docs.v_function_summary
            WHERE rel_path = ?
              AND repo = ?
              AND commit = ?
            ORDER BY qualname
            """,
            [rel_path, self.repo, self.commit],
        )
        file_payload = dict(file_row)
        file_payload["functions"] = [ViewRow.model_validate(r) for r in funcs]
        return FileSummaryResponse(
            found=True,
            file=ViewRow.model_validate(file_payload),
            meta=ResponseMeta(),
        )

    def get_function_profile(self, *, goid_h128: int) -> FunctionProfileResponse:
        """
        Return a function profile from docs.v_function_profile.

        Parameters
        ----------
        goid_h128:
            Function GOID identifier.

        Returns
        -------
        FunctionProfileResponse
            Profile payload indicating whether it was found.
        """
        row = _fetch_one_dict(
            self.con,
            """
            SELECT *
            FROM docs.v_function_profile
            WHERE repo = ?
              AND commit = ?
              AND function_goid_h128 = ?
            LIMIT 1
            """,
            [self.repo, self.commit, goid_h128],
        )
        if row is None:
            meta = ResponseMeta(
                messages=[
                    Message(
                        code="not_found",
                        severity="info",
                        detail="Function profile not found",
                        context={"goid_h128": goid_h128},
                    )
                ]
            )
            return FunctionProfileResponse(found=False, profile=None, meta=meta)
        return FunctionProfileResponse(
            found=True,
            profile=ViewRow.model_validate(row),
            meta=ResponseMeta(),
        )

    def get_file_profile(self, *, rel_path: str) -> FileProfileResponse:
        """
        Return a file profile from docs.v_file_profile.

        Parameters
        ----------
        rel_path:
            Repo-relative file path.

        Returns
        -------
        FileProfileResponse
            Profile payload and metadata.
        """
        row = _fetch_one_dict(
            self.con,
            """
            SELECT *
            FROM docs.v_file_profile
            WHERE repo = ?
              AND commit = ?
              AND rel_path = ?
            LIMIT 1
            """,
            [self.repo, self.commit, rel_path],
        )
        if row is None:
            meta = ResponseMeta(
                messages=[
                    Message(
                        code="not_found",
                        severity="info",
                        detail="File profile not found",
                        context={"rel_path": rel_path},
                    )
                ]
            )
            return FileProfileResponse(found=False, profile=None, meta=meta)
        return FileProfileResponse(
            found=True,
            profile=ViewRow.model_validate(row),
            meta=ResponseMeta(),
        )

    def get_module_profile(self, *, module: str) -> ModuleProfileResponse:
        """
        Return a module profile from docs.v_module_profile.

        Parameters
        ----------
        module:
            Module name to query.

        Returns
        -------
        ModuleProfileResponse
            Profile payload and metadata.
        """
        row = _fetch_one_dict(
            self.con,
            """
            SELECT *
            FROM docs.v_module_profile
            WHERE repo = ?
              AND commit = ?
              AND module = ?
            LIMIT 1
            """,
            [self.repo, self.commit, module],
        )
        if row is None:
            meta = ResponseMeta(
                messages=[
                    Message(
                        code="not_found",
                        severity="info",
                        detail="Module profile not found",
                        context={"module": module},
                    )
                ]
            )
            return ModuleProfileResponse(found=False, profile=None, meta=meta)
        return ModuleProfileResponse(
            found=True,
            profile=ViewRow.model_validate(row),
            meta=ResponseMeta(),
        )

    def get_function_architecture(self, *, goid_h128: int) -> FunctionArchitectureResponse:
        """
        Return architecture metrics for a function.

        Returns
        -------
        FunctionArchitectureResponse
            Architecture payload and found flag.
        """
        row = _fetch_one_dict(
            self.con,
            """
            SELECT *
            FROM docs.v_function_architecture
            WHERE repo = ?
              AND commit = ?
              AND function_goid_h128 = ?
            LIMIT 1
            """,
            [self.repo, self.commit, goid_h128],
        )
        if row is None:
            return FunctionArchitectureResponse(
                found=False,
                architecture=None,
                meta=ResponseMeta(
                    messages=[
                        Message(
                            code="not_found",
                            severity="info",
                            detail="Function architecture not found",
                            context={"goid_h128": goid_h128},
                        )
                    ]
                ),
            )
        return FunctionArchitectureResponse(
            found=True,
            architecture=ViewRow.model_validate(row),
            meta=ResponseMeta(),
        )

    def get_module_architecture(self, *, module: str) -> ModuleArchitectureResponse:
        """
        Return architecture metrics for a module.

        Returns
        -------
        ModuleArchitectureResponse
            Architecture payload and found flag.
        """
        row = _fetch_one_dict(
            self.con,
            """
            SELECT *
            FROM docs.v_module_architecture
            WHERE repo = ?
              AND commit = ?
              AND module = ?
            LIMIT 1
            """,
            [self.repo, self.commit, module],
        )
        if row is None:
            return ModuleArchitectureResponse(
                found=False,
                architecture=None,
                meta=ResponseMeta(
                    messages=[
                        Message(
                            code="not_found",
                            severity="info",
                            detail="Module architecture not found",
                            context={"module": module},
                        )
                    ]
                ),
            )
        return ModuleArchitectureResponse(
            found=True,
            architecture=ViewRow.model_validate(row),
            meta=ResponseMeta(),
        )

    def list_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSummaryResponse:
        """
        Return subsystem summaries ordered by module_count desc.

        Returns
        -------
        SubsystemSummaryResponse
            Subsystem rows and metadata.
        """
        filters = ["s.repo = ?", "s.commit = ?"]
        params: list[object] = [self.repo, self.commit]
        if role:
            filters.append(
                """
                EXISTS (
                    SELECT 1
                    FROM analytics.subsystem_modules sm
                    WHERE sm.repo = s.repo
                      AND sm.commit = s.commit
                      AND sm.subsystem_id = s.subsystem_id
                      AND sm.role = ?
                )
                """
            )
            params.append(role)
        if q:
            filters.append("(s.name ILIKE ? OR s.description ILIKE ?)")
            pattern = f"%{q}%"
            params.extend([pattern, pattern])

        limit_value = limit if limit is not None else self.limits.default_limit
        where_clause = " AND ".join(filters)
        query_parts = [
            "SELECT *",
            "FROM docs.v_subsystem_summary s",
            "WHERE " + where_clause,
            "ORDER BY module_count DESC, subsystem_id",
            "LIMIT ?",
        ]
        query = "\n".join(query_parts)
        rows = _fetch_all_dicts(
            self.con,
            query,
            [*params, limit_value],
        )
        _normalize_entrypoints(rows)
        return SubsystemSummaryResponse(
            subsystems=[ViewRow.model_validate(r) for r in rows],
            meta=ResponseMeta(
                applied_limit=limit_value,
                requested_limit=limit,
            ),
        )

    def get_module_subsystems(self, *, module: str) -> ModuleSubsystemResponse:
        """
        Return subsystem memberships for a module.

        Returns
        -------
        ModuleSubsystemResponse
            Membership rows and metadata.
        """
        rows = _fetch_all_dicts(
            self.con,
            """
            SELECT *
            FROM docs.v_module_with_subsystem
            WHERE repo = ?
              AND commit = ?
              AND module = ?
            """,
            [self.repo, self.commit, module],
        )
        if not rows:
            return ModuleSubsystemResponse(
                found=False,
                memberships=[],
                meta=ResponseMeta(
                    messages=[
                        Message(
                            code="not_found",
                            severity="info",
                            detail="Module has no subsystem mappings",
                            context={"module": module},
                        )
                    ]
                ),
            )
        return ModuleSubsystemResponse(
            found=True,
            memberships=[ViewRow.model_validate(r) for r in rows],
            meta=ResponseMeta(),
        )

    def get_file_hints(self, *, rel_path: str) -> FileHintsResponse:
        """
        Return IDE-focused hints for a file path.

        Returns
        -------
        FileHintsResponse
            Hint rows scoped to the provided relative path.
        """
        rows = _fetch_all_dicts(
            self.con,
            """
            SELECT *
            FROM docs.v_ide_hints
            WHERE repo = ?
              AND commit = ?
              AND rel_path = ?
            """,
            [self.repo, self.commit, rel_path],
        )
        if not rows:
            return FileHintsResponse(
                found=False,
                hints=[],
                meta=ResponseMeta(
                    messages=[
                        Message(
                            code="not_found",
                            severity="info",
                            detail="No IDE hints found for path",
                            context={"rel_path": rel_path},
                        )
                    ]
                ),
            )
        return FileHintsResponse(
            found=True,
            hints=[ViewRow.model_validate(r) for r in rows],
            meta=ResponseMeta(),
        )

    def get_subsystem_modules(self, *, subsystem_id: str) -> SubsystemModulesResponse:
        """
        Return subsystem details and module memberships.

        Returns
        -------
        SubsystemModulesResponse
            Subsystem detail and module rows.
        """
        subsystem_row = _fetch_one_dict(
            self.con,
            """
            SELECT *
            FROM docs.v_subsystem_summary
            WHERE repo = ?
              AND commit = ?
              AND subsystem_id = ?
            LIMIT 1
            """,
            [self.repo, self.commit, subsystem_id],
        )
        _normalize_entrypoints_dict(subsystem_row)
        modules = _fetch_all_dicts(
            self.con,
            """
            SELECT *
            FROM docs.v_module_with_subsystem
            WHERE repo = ?
              AND commit = ?
              AND subsystem_id = ?
            ORDER BY module
            """,
            [self.repo, self.commit, subsystem_id],
        )
        if subsystem_row is None:
            return SubsystemModulesResponse(
                found=False,
                subsystem=None,
                modules=[],
                meta=ResponseMeta(
                    messages=[
                        Message(
                            code="not_found",
                            severity="info",
                            detail="Subsystem not found",
                            context={"subsystem_id": subsystem_id},
                        )
                    ]
                ),
            )
        return SubsystemModulesResponse(
            found=True,
            subsystem=ViewRow.model_validate(subsystem_row),
            modules=[ViewRow.model_validate(r) for r in modules],
            meta=ResponseMeta(),
        )

    def search_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSearchResponse:
        """
        Search wrapper returning subsystem summaries.

        Returns
        -------
        SubsystemSearchResponse
            Subsystem rows and metadata for search-oriented use.
        """
        result = self.list_subsystems(limit=limit, role=role, q=q)
        return SubsystemSearchResponse(subsystems=result.subsystems, meta=result.meta)

    def summarize_subsystem(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> SubsystemModulesResponse:
        """
        Return subsystem detail with optional module truncation.

        Returns
        -------
        SubsystemModulesResponse
            Subsystem detail payload, optionally with a limited module list.
        """
        detail = self.get_subsystem_modules(subsystem_id=subsystem_id)
        if not detail.found or detail.subsystem is None:
            return detail
        if module_limit is None:
            return detail
        limited_modules = detail.modules[:module_limit]
        return SubsystemModulesResponse(
            found=detail.found,
            subsystem=detail.subsystem,
            modules=limited_modules,
            meta=detail.meta,
        )

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int = 100,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        """
        Read dataset rows with clamping and messaging.

        Parameters
        ----------
        dataset_name:
            Registry name for the dataset.
        limit:
            Requested row limit.
        offset:
            Requested offset.

        Returns
        -------
        DatasetRowsResponse
            Dataset slice plus metadata.

        Raises
        ------
        errors.invalid_argument
            When the dataset name is unknown.
        """
        table = self.gateway.datasets.mapping.get(dataset_name)
        if table is None:
            message = f"Unknown dataset: {dataset_name}"
            raise errors.invalid_argument(message)

        limit_clamp = clamp_limit_value(
            limit,
            default=limit,
            max_limit=self.limits.max_rows_per_call,
        )
        offset_clamp = clamp_offset_value(offset)
        meta = ResponseMeta(
            requested_limit=limit,
            applied_limit=limit_clamp.applied,
            requested_offset=offset,
            applied_offset=offset_clamp.applied,
            messages=[*limit_clamp.messages, *offset_clamp.messages],
        )

        if limit_clamp.has_error or offset_clamp.has_error or limit_clamp.applied <= 0:
            return DatasetRowsResponse(
                dataset=dataset_name,
                limit=limit_clamp.applied,
                offset=offset_clamp.applied,
                rows=[],
                meta=meta,
            )

        result = self.con.execute(
            "SELECT * FROM metadata.dataset_rows(?, ?, ?)",
            [table, limit_clamp.applied, offset_clamp.applied],
        )
        rows = result.fetchall()
        cols = [desc[0] for desc in result.description]
        mapped = [{col: row[idx] for idx, col in enumerate(cols)} for row in rows]
        meta.truncated = limit_clamp.applied > 0 and len(mapped) == limit_clamp.applied
        if not mapped:
            meta.messages.append(
                Message(
                    code="dataset_empty",
                    severity="info",
                    detail="Dataset returned no rows for the requested slice.",
                    context={"dataset": dataset_name, "offset": offset_clamp.applied},
                )
            )
        return DatasetRowsResponse(
            dataset=dataset_name,
            limit=limit_clamp.applied,
            offset=offset_clamp.applied,
            rows=[ViewRow.model_validate(r) for r in mapped],
            meta=meta,
        )
