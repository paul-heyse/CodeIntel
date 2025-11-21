"""Backend implementations for MCP tools over DuckDB or HTTP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import duckdb


def _fetch_one_dict(cur: duckdb.DuckDBPyConnection, sql: str, params: list[Any]) -> dict[str, Any] | None:
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
    dict[str, Any] | None
        Row dictionary if present, otherwise None.
    """
    result = cur.execute(sql, params)
    row = result.fetchone()
    if row is None:
        return None
    cols = [desc[0] for desc in result.description]
    return {col: row[idx] for idx, col in enumerate(cols)}


def _fetch_all_dicts(cur: duckdb.DuckDBPyConnection, sql: str, params: list[Any]) -> list[dict[str, Any]]:
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
    list[dict[str, Any]]
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
    ) -> dict[str, Any] | None:
        """Return a function summary from docs.v_function_summary."""
        ...

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int = 50,
        tested_only: bool = False,
    ) -> list[dict[str, Any]]:
        """List high-risk functions from analytics.goid_risk_factors."""
        ...

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int = 50,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return incoming/outgoing call graph neighbors."""
        ...

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
    ) -> list[dict[str, Any]]:
        """List tests that exercised a function."""
        ...

    def get_file_summary(
        self,
        *,
        rel_path: str,
    ) -> dict[str, Any] | None:
        """Return file summary plus function rows."""
        ...

    def list_datasets(self) -> list[dict[str, Any]]:
        """List datasets available to browse."""
        ...

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
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
        return int(row["function_goid_h128"])

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Fetch a single row from docs.v_function_summary.

        Returns
        -------
        dict[str, Any] | None
            Function summary if found, otherwise None.

        Raises
        ------
        ValueError
            If insufficient identifiers are provided.
        """
        if urn:
            return _fetch_one_dict(
                self.con,
                """
                SELECT *
                FROM docs.v_function_summary
                WHERE repo = ? AND commit = ? AND urn = ?
                LIMIT 1
                """,
                [self.repo, self.commit, urn],
            )
        if goid_h128 is not None:
            return _fetch_one_dict(
                self.con,
                """
                SELECT *
                FROM docs.v_function_summary
                WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
                LIMIT 1
                """,
                [self.repo, self.commit, goid_h128],
            )
        if rel_path and qualname:
            return _fetch_one_dict(
                self.con,
                """
                SELECT *
                FROM docs.v_function_summary
                WHERE repo = ? AND commit = ? AND rel_path = ? AND qualname = ?
                LIMIT 1
                """,
                [self.repo, self.commit, rel_path, qualname],
            )
        message = "Must provide urn or goid_h128 or (rel_path + qualname)."
        raise ValueError(message)

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int = 50,
        tested_only: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Query analytics.goid_risk_factors for high-risk functions.

        Returns
        -------
        list[dict[str, Any]]
            High-risk function rows sorted by risk_score.
        """
        extra_clause = " AND tested = TRUE" if tested_only else ""
        params: list[Any] = [self.repo, self.commit, min_risk, limit]
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
        return _fetch_all_dicts(self.con, sql, params)

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int = 50,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Return incoming/outgoing edges from docs.v_call_graph_enriched.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Mapping with "outgoing" and "incoming" edge lists.
        """
        outgoing: list[dict[str, Any]] = []
        incoming: list[dict[str, Any]] = []

        if direction in {"out", "both"}:
            out_sql = """
                SELECT *
                FROM docs.v_call_graph_enriched
                WHERE caller_goid_h128 = ?
                ORDER BY callee_qualname
                LIMIT ?
            """
            outgoing = _fetch_all_dicts(self.con, out_sql, [goid_h128, limit])

        if direction in {"in", "both"}:
            in_sql = """
                SELECT *
                FROM docs.v_call_graph_enriched
                WHERE callee_goid_h128 = ?
                ORDER BY caller_qualname
                LIMIT ?
            """
            incoming = _fetch_all_dicts(self.con, in_sql, [goid_h128, limit])

        return {"outgoing": outgoing, "incoming": incoming}

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query docs.v_test_to_function for tests that hit a given function.

        Returns
        -------
        list[dict[str, Any]]
            Rows from docs.v_test_to_function.

        Raises
        ------
        ValueError
            If neither goid_h128 nor urn is provided.
        """
        if goid_h128 is None and urn is None:
            message = "Must provide goid_h128 or urn."
            raise ValueError(message)

        if goid_h128 is not None:
            return _fetch_all_dicts(
                self.con,
                """
                SELECT *
                FROM docs.v_test_to_function
                WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
                ORDER BY test_id
                """,
                [self.repo, self.commit, goid_h128],
            )
        return _fetch_all_dicts(
            self.con,
            """
            SELECT *
            FROM docs.v_test_to_function
            WHERE repo = ? AND commit = ? AND urn = ?
            ORDER BY test_id
            """,
            [self.repo, self.commit, urn],
        )

    def get_file_summary(
        self,
        *,
        rel_path: str,
    ) -> dict[str, Any] | None:
        """
        Return docs.v_file_summary row plus function summaries for a file.

        Returns
        -------
        dict[str, Any] | None
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
            return None

        funcs = _fetch_all_dicts(
            self.con,
            """
            SELECT *
            FROM docs.v_function_summary
            WHERE rel_path = ?
              AND repo = ?
              AND commit = ?
            ORDER BY start_line
            """,
            [rel_path, self.repo, self.commit],
        )
        file_row["functions"] = funcs
        return file_row

    def list_datasets(self) -> list[dict[str, Any]]:
        """
        List datasets that can be browsed via MCP tools.

        Returns
        -------
        list[dict[str, Any]]
            Dataset metadata entries.
        """
        return [
            {"name": name, "table": table, "description": f"DuckDB table/view {table}"}
            for name, table in sorted(self.dataset_tables.items())
        ]

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Read a slice of a dataset using DuckDB's relational API.

        Raises
        ------
        ValueError
            If the dataset_name is unknown.

        Returns
        -------
        list[dict[str, Any]]
            Rows fetched from the dataset.
        """
        table = self.dataset_tables.get(dataset_name)
        if not table:
            message = f"Unknown dataset: {dataset_name}"
            raise ValueError(message)

        relation = self.con.table(table).limit(limit, offset)
        rows = relation.fetchall()
        cols = [desc[0] for desc in relation.description]
        return [{col: row[idx] for idx, col in enumerate(cols)} for row in rows]
