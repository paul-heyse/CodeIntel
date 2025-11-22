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
    TestsForFunctionResponse,
    ViewRow,
)

MAX_ROWS_LIMIT = 500
VALID_DIRECTIONS = frozenset({"in", "out", "both"})
RowDict = dict[str, object]


def _column_exists(con: duckdb.DuckDBPyConnection, table: str, column: str) -> bool:
    """Return True when a table or view exposes the given column."""
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

        return FunctionSummaryResponse(found=bool(summary), summary=summary)

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
        list[RowDict]
            High-risk function rows sorted by risk_score.

        Raises
        ------
        errors.invalid_argument
            Raised when `limit` exceeds the allowed maximum.
        """
        extra_clause = " AND tested = TRUE" if tested_only else ""
        if limit > MAX_ROWS_LIMIT:
            message = f"limit cannot exceed {MAX_ROWS_LIMIT}"
            raise errors.invalid_argument(message)
        safe_limit = max(0, limit)
        params: list[object] = [self.repo, self.commit, min_risk, safe_limit]
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
        truncated = safe_limit > 0 and len(rows) == safe_limit
        return HighRiskFunctionsResponse(functions=models, truncated=truncated)

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
            Raised when `direction` is not one of {'in','out','both'} or when
            `limit` exceeds bounds.
        """
        if direction not in VALID_DIRECTIONS:
            message = "direction must be one of {'in','out','both'}"
            raise errors.invalid_argument(message)
        if limit > MAX_ROWS_LIMIT:
            message = f"limit cannot exceed {MAX_ROWS_LIMIT}"
            raise errors.invalid_argument(message)
        safe_limit = max(0, limit)
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
            out_rows = _fetch_all_dicts(self.con, out_sql, [goid_h128, safe_limit])
            outgoing = [ViewRow.model_validate(r) for r in out_rows]

        if direction in {"in", "both"}:
            in_sql = """
                SELECT *
                FROM docs.v_call_graph_enriched
                WHERE callee_goid_h128 = ?
                ORDER BY caller_qualname
                LIMIT ?
            """
            in_rows = _fetch_all_dicts(self.con, in_sql, [goid_h128, safe_limit])
            incoming = [ViewRow.model_validate(r) for r in in_rows]

        return CallGraphNeighborsResponse(outgoing=outgoing, incoming=incoming)

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
            Raised when identifiers are missing or the requested limit is invalid.
        """
        if goid_h128 is None and urn is None:
            message = "Must provide goid_h128 or urn."
            raise errors.invalid_argument(message)

        repo_col = "test_repo" if _column_exists(self.con, "docs.v_test_to_function", "test_repo") else "repo"
        commit_col = (
            "test_commit" if _column_exists(self.con, "docs.v_test_to_function", "test_commit") else "commit"
        )

        if limit is None:
            safe_limit = MAX_ROWS_LIMIT
        elif limit > MAX_ROWS_LIMIT:
            message = f"limit cannot exceed {MAX_ROWS_LIMIT}"
            raise errors.invalid_argument(message)
        else:
            safe_limit = max(0, limit)
        if goid_h128 is not None:
            rows = _fetch_all_dicts(
                self.con,
                f"""
                SELECT *
                FROM docs.v_test_to_function
                WHERE {repo_col} = ?
                  AND {commit_col} = ?
                  AND function_goid_h128 = ?
                ORDER BY test_id
                LIMIT ?
                """,
                [self.repo, self.commit, goid_h128, safe_limit],
            )
        else:
            rows = _fetch_all_dicts(
                self.con,
                f"""
                SELECT *
                FROM docs.v_test_to_function
                WHERE {repo_col} = ?
                  AND {commit_col} = ?
                  AND urn = ?
                ORDER BY test_id
                LIMIT ?
                """,
                [self.repo, self.commit, urn, safe_limit],
            )
        return TestsForFunctionResponse(tests=[ViewRow.model_validate(r) for r in rows])

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
            return FileSummaryResponse(found=False, file=None)

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
        return FileSummaryResponse(found=True, file=ViewRow.model_validate(file_payload))

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

        if limit > MAX_ROWS_LIMIT:
            message = f"limit cannot exceed {MAX_ROWS_LIMIT}"
            raise errors.invalid_argument(message)
        safe_limit = max(0, limit)
        safe_offset = max(0, offset)
        relation = self.con.table(table).limit(safe_limit, safe_offset)
        rows = relation.fetchall()
        cols = [desc[0] for desc in relation.description]
        mapped = [{col: row[idx] for idx, col in enumerate(cols)} for row in rows]
        return DatasetRowsResponse(
            dataset=dataset_name,
            limit=safe_limit,
            offset=safe_offset,
            rows=[ViewRow.model_validate(r) for r in mapped],
        )
