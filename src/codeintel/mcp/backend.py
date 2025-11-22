"""Backend implementations for MCP tools over DuckDB or HTTP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import duckdb

from codeintel.mcp import errors
from codeintel.mcp.models import (
    CallGraphNeighborsResponse,
    DatasetDescriptor,
    DatasetRowsResponse,
    FileSummaryResponse,
    FunctionSummaryResponse,
    HighRiskFunctionsResponse,
    Message,
    ResponseMeta,
    TestsForFunctionResponse,
    ViewRow,
)

MAX_ROWS_LIMIT = 500
VALID_DIRECTIONS = frozenset({"in", "out", "both"})
RowDict = dict[str, object]


@dataclass(frozen=True)
class ClampResult:
    """Result of clamping limit/offset values with messaging."""

    applied: int
    messages: list[Message] = field(default_factory=list)
    has_error: bool = False


def _clamp_limit_value(requested: int | None, *, default: int) -> ClampResult:
    """
    Clamp a requested limit to safe bounds, returning warnings instead of errors.

    Returns
    -------
    ClampResult
        Applied limit plus any informational/warning/error messages for callers.
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

    max_limit = MAX_ROWS_LIMIT
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


def _clamp_offset_value(offset: int) -> ClampResult:
    """
    Clamp an offset to a non-negative value, returning messaging instead of raising.

    Returns
    -------
    ClampResult
        Applied offset plus any informational/warning/error messages for callers.
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


def _column_exists(con: duckdb.DuckDBPyConnection, table: str, column: str) -> bool:
    """
    Return True when a table or view exposes the given column.

    Returns
    -------
    bool
        True if the column exists, otherwise False.
    """
    info_sql = f"PRAGMA table_info('{table}')"
    rows = con.execute(info_sql).fetchall()
    return any(col[1] == column for col in rows)


def _fetch_one_dict(
    cur: duckdb.DuckDBPyConnection, sql: str, params: list[object]
) -> RowDict | None:
    """
    Execute a parameterized query and return the first row as a dict.

    Parameters
    ----------
    cur:
        DuckDB connection or relation supporting `execute`.
    sql:
        SQL string with parameter placeholders.
    params:
        Parameters to bind to the query.

    Returns
    -------
    dict[str, object] | None
        Row dictionary if present, otherwise None.
    """
    result = cur.execute(sql, params)
    row = result.fetchone()
    if row is None:
        return None
    cols = [desc[0] for desc in result.description]
    return {col: row[idx] for idx, col in enumerate(cols)}


def _fetch_all_dicts(
    cur: duckdb.DuckDBPyConnection, sql: str, params: list[object]
) -> list[RowDict]:
    """
    Execute a parameterized query and return all rows as dictionaries.

    Parameters
    ----------
    cur:
        DuckDB connection or relation supporting `execute`.
    sql:
        SQL string with parameter placeholders.
    params:
        Parameters to bind to the query.

    Returns
    -------
    list[dict[str, object]]
        List of row dictionaries.
    """
    result = cur.execute(sql, params)
    rows = result.fetchall()
    cols = [desc[0] for desc in result.description]
    return [{col: row[idx] for idx, col in enumerate(cols)} for row in rows]


class QueryBackend(Protocol):
    """Abstract interface consumed by MCP tools."""

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> FunctionSummaryResponse:
        """Return a function summary from docs.v_function_summary."""
        ...

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int = 50,
        tested_only: bool = False,
    ) -> HighRiskFunctionsResponse:
        """List high-risk functions from analytics.goid_risk_factors."""
        ...

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int = 50,
    ) -> CallGraphNeighborsResponse:
        """Return incoming/outgoing call graph neighbors."""
        ...

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
    ) -> TestsForFunctionResponse:
        """List tests that exercised a function."""
        ...

    def get_file_summary(
        self,
        *,
        rel_path: str,
    ) -> FileSummaryResponse:
        """Return file summary plus function rows."""
        ...

    def list_datasets(self) -> list[DatasetDescriptor]:
        """List datasets available to browse."""
        ...

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int = 100,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        """Read rows from a dataset in small slices."""
        ...


@dataclass
class DuckDBBackend(QueryBackend):
    """
    DuckDB-backed implementation of QueryBackend.

    Assumes a single repo/commit per DuckDB file, but repo/commit filters
    are still applied for future multi-repo support.
    """

    con: duckdb.DuckDBPyConnection
    repo: str
    commit: str
    dataset_tables: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Populate default dataset table mapping if one is not provided."""
        if not self.dataset_tables:
            self.dataset_tables = {
                # core
                "docstrings": "core.docstrings",
                "goids": "core.goids",
                "goid_crosswalk": "core.goid_crosswalk",
                "modules": "core.modules",
                "ast_nodes": "core.ast_nodes",
                "cst_nodes": "core.cst_nodes",
                # graph
                "call_graph_nodes": "graph.call_graph_nodes",
                "call_graph_edges": "graph.call_graph_edges",
                "cfg_blocks": "graph.cfg_blocks",
                "cfg_edges": "graph.cfg_edges",
                "dfg_edges": "graph.dfg_edges",
                "import_graph_edges": "graph.import_graph_edges",
                "symbol_use_edges": "graph.symbol_use_edges",
                # analytics
                "function_metrics": "analytics.function_metrics",
                "function_types": "analytics.function_types",
                "coverage_functions": "analytics.coverage_functions",
                "goid_risk_factors": "analytics.goid_risk_factors",
                "hotspots": "analytics.hotspots",
                "typedness": "analytics.typedness",
                "static_diagnostics": "analytics.static_diagnostics",
                "test_catalog": "analytics.test_catalog",
                "test_coverage_edges": "analytics.test_coverage_edges",
                # docs views
                "v_function_summary": "docs.v_function_summary",
                "v_call_graph_enriched": "docs.v_call_graph_enriched",
                "v_test_to_function": "docs.v_test_to_function",
                "v_file_summary": "docs.v_file_summary",
            }

    def _resolve_function_goid(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> int | None:
        """
        Resolve a function to its GOID using docs.v_function_summary.

        Returns
        -------
        int | None
            GOID hash if found, otherwise None.

        Raises
        ------
        errors.backend_failure
            If the retrieved GOID has an unexpected type.
        """
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
        Fetch a single row from docs.v_function_summary.

        Returns
        -------
        RowDict | None
            Function summary if found, otherwise None.

        Raises
        ------
        errors.McpError
            If insufficient identifiers are provided (invalid_argument).

        Returns
        -------
        FunctionSummaryResponse
            Wrapped function summary payload.
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
            raise errors.McpError(
                detail=errors.ProblemDetail(
                    type="https://example.com/problems/invalid-argument",
                    title="Invalid argument",
                    detail=message,
                    status=400,
                )
            )

        if summary is None:
            meta.messages.append(
                Message(
                    code="not_found",
                    severity="info",
                    detail="Function not found",
                    context={"urn": urn, "goid_h128": goid_h128, "rel_path": rel_path, "qualname": qualname},
                )
            )
            return FunctionSummaryResponse(found=False, summary=None, meta=meta)

        return FunctionSummaryResponse(found=True, summary=summary, meta=meta)

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int = 50,
        tested_only: bool = False,
    ) -> HighRiskFunctionsResponse:
        """
        Query analytics.goid_risk_factors for high-risk functions.

        Returns
        -------
        HighRiskFunctionsResponse
            High-risk function rows sorted by risk_score plus metadata.
        """
        extra_clause = " AND tested = TRUE" if tested_only else ""
        clamp = _clamp_limit_value(limit, default=limit)
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
        limit: int = 50,
    ) -> CallGraphNeighborsResponse:
        """
        Return incoming/outgoing edges from docs.v_call_graph_enriched.

        Returns
        -------
        dict[str, list[RowDict]]
            Mapping with "outgoing" and "incoming" edge lists.

        Raises
        ------
        errors.invalid_argument
            If direction is not one of {'in','out','both'}.
        """
        if direction not in VALID_DIRECTIONS:
            message = "direction must be one of {'in','out','both'}"
            raise errors.invalid_argument(message)
        clamp = _clamp_limit_value(limit, default=limit)
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
                ORDER BY callee_qualname
                LIMIT ?
            """
            out_rows = _fetch_all_dicts(self.con, out_sql, [goid_h128, clamp.applied])
            outgoing = [ViewRow.model_validate(r) for r in out_rows]

        if direction in {"in", "both"}:
            in_sql = """
                SELECT *
                FROM docs.v_call_graph_enriched
                WHERE callee_goid_h128 = ?
                ORDER BY caller_qualname
                LIMIT ?
            """
            in_rows = _fetch_all_dicts(self.con, in_sql, [goid_h128, clamp.applied])
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
        Query docs.v_test_to_function for tests that hit a given function.

        Parameters
        ----------
        goid_h128:
            GOID hash of the function under test.
        urn:
            URN of the function under test when the GOID is unknown.
        limit:
            Maximum number of rows to return (clamped to MAX_ROWS_LIMIT when unset).

        Returns
        -------
        TestsForFunctionResponse
            Wrapped test rows.

        Raises
        ------
        errors.invalid_argument
            Raised when identifiers are missing.
        """
        if goid_h128 is None and urn is None:
            message = "Must provide goid_h128 or urn."
            raise errors.invalid_argument(message)

        clamp = _clamp_limit_value(limit, default=MAX_ROWS_LIMIT)
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
            "test_repo"
            if "test_repo" in columns
            else ("repo" if "repo" in columns else None)
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

    def get_file_summary(
        self,
        *,
        rel_path: str,
    ) -> FileSummaryResponse:
        """
        Return docs.v_file_summary row plus function summaries for a file.

        Returns
        -------
        RowDict | None
            File summary with nested functions, or None if missing.
        """
        file_row = _fetch_one_dict(
            self.con,
            """
            SELECT *
            FROM docs.v_file_summary
            WHERE rel_path = ?
            LIMIT 1
            """,
            [rel_path],
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

    def list_datasets(self) -> list[DatasetDescriptor]:
        """
        List datasets that can be browsed via MCP tools.

        Returns
        -------
        list[dict[str, object]]
            Dataset metadata entries.
        """
        return [
            DatasetDescriptor(
                name=name,
                table=table,
                description=f"DuckDB table/view {table}",
            )
            for name, table in sorted(self.dataset_tables.items())
        ]

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int = 100,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        """
        Read a slice of a dataset using DuckDB's relational API.

        Raises
        ------
        errors.invalid_argument
            Raised when dataset_name is unknown or when `limit` exceeds the
            allowed maximum.

        Returns
        -------
        DatasetRowsResponse
            Dataset slice payload with applied limits and offset.
        """
        table = self.dataset_tables.get(dataset_name)
        if not table:
            message = f"Unknown dataset: {dataset_name}"
            raise errors.invalid_argument(message)

        limit_clamp = _clamp_limit_value(limit, default=limit)
        offset_clamp = _clamp_offset_value(offset)
        meta = ResponseMeta(
            requested_limit=limit,
            applied_limit=limit_clamp.applied,
            requested_offset=offset,
            applied_offset=offset_clamp.applied,
            messages=[*limit_clamp.messages, *offset_clamp.messages],
        )

        if limit_clamp.has_error or offset_clamp.has_error:
            return DatasetRowsResponse(
                dataset=dataset_name,
                limit=limit_clamp.applied,
                offset=offset_clamp.applied,
                rows=[],
                meta=meta,
            )

        relation = self.con.table(table).limit(limit_clamp.applied, offset_clamp.applied)
        rows = relation.fetchall()
        cols = [desc[0] for desc in relation.description]
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
