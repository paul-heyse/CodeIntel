# src/codeintel/mcp/backend.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import duckdb


def _fetch_one_dict(con: duckdb.DuckDBPyConnection, sql: str, params: List[Any]) -> Optional[Dict[str, Any]]:
    cur = con.execute(sql, params)
    row = cur.fetchone()
    if row is None:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def _fetch_all_dicts(con: duckdb.DuckDBPyConnection, sql: str, params: List[Any]) -> List[Dict[str, Any]]:
    cur = con.execute(sql, params)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in rows]


class QueryBackend(Protocol):
    """
    Abstract interface for CodeIntel queries, consumed by the MCP server.

    This lets us swap in a DuckDB-backed implementation (local) or an
    HTTP-backed implementation (remote FastAPI) later without changing
    the MCP tools.
    """

    # ---- Core “function-centric” APIs ----------------------

    def get_function_summary(
        self,
        *,
        urn: Optional[str] = None,
        goid_h128: Optional[int] = None,
        rel_path: Optional[str] = None,
        qualname: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        ...

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int = 50,
        tested_only: bool = False,
    ) -> List[Dict[str, Any]]:
        ...

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",  # "in" | "out" | "both"
        limit: int = 50,
    ) -> Dict[str, List[Dict[str, Any]]]:
        ...

    def get_tests_for_function(
        self,
        *,
        goid_h128: Optional[int] = None,
        urn: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        ...

    def get_file_summary(
        self,
        *,
        rel_path: str,
    ) -> Optional[Dict[str, Any]]:
        ...

    # ---- Dataset / introspection APIs ----------------------

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        Return a small catalog of datasets available to the LLM, e.g.:

          [
            {"name": "goids", "table": "core.goids", "description": "..."},
            {"name": "call_graph_edges", "table": "graph.call_graph_edges", ...},
          ]
        """
        ...

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Read a small slice of a dataset for inspection (not bulk export).
        """
        ...


@dataclass
class DuckDBBackend(QueryBackend):
    """
    DuckDB-backed implementation of QueryBackend.

    This assumes:
      - A single repo/commit per DuckDB database (as in your current design).
      - docs.v_function_summary, docs.v_call_graph_enriched, docs.v_test_to_function,
        docs.v_file_summary views exist, plus analytics.goid_risk_factors. :contentReference[oaicite:1]{index=1}
    """

    con: duckdb.DuckDBPyConnection
    repo: str
    commit: str

    # ---------- Helpers ----------

    def _resolve_function_goid(
        self,
        *,
        urn: Optional[str] = None,
        goid_h128: Optional[int] = None,
        rel_path: Optional[str] = None,
        qualname: Optional[str] = None,
    ) -> Optional[int]:
        """
        Resolve a function to its GOID using docs.v_function_summary.
        """
        if goid_h128 is not None:
            return goid_h128

        params: List[Any] = []
        clauses: List[str] = ["repo = ?", "commit = ?"]
        params.extend([self.repo, self.commit])

        if urn:
            clauses.append("urn = ?")
            params.append(urn)
        elif rel_path and qualname:
            clauses.append("rel_path = ?")
            clauses.append("qualname = ?")
            params.extend([rel_path, qualname])
        else:
            # Not enough info
            return None

        sql = f"""
            SELECT function_goid_h128
            FROM docs.v_function_summary
            WHERE {' AND '.join(clauses)}
            LIMIT 1
        """
        row = _fetch_one_dict(self.con, sql, params)
        if not row:
            return None
        return int(row["function_goid_h128"])

    # ---------- Core implementations ----------

    def get_function_summary(
        self,
        *,
        urn: Optional[str] = None,
        goid_h128: Optional[int] = None,
        rel_path: Optional[str] = None,
        qualname: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single row from docs.v_function_summary.
        """
        params: List[Any] = [self.repo, self.commit]
        where = ["repo = ?", "commit = ?"]

        if urn:
            where.append("urn = ?")
            params.append(urn)
        elif goid_h128 is not None:
            where.append("function_goid_h128 = ?")
            params.append(goid_h128)
        elif rel_path and qualname:
            where.append("rel_path = ?")
            where.append("qualname = ?")
            params.extend([rel_path, qualname])
        else:
            raise ValueError("Must provide urn or goid_h128 or (rel_path + qualname).")

        sql = f"""
            SELECT *
            FROM docs.v_function_summary
            WHERE {' AND '.join(where)}
            LIMIT 1
        """
        return _fetch_one_dict(self.con, sql, params)

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int = 50,
        tested_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Query analytics.goid_risk_factors for high risk functions. :contentReference[oaicite:2]{index=2}
        """
        where = ["repo = ?", "commit = ?", "risk_score >= ?"]
        params: List[Any] = [self.repo, self.commit, min_risk]

        if tested_only:
            where.append("tested = TRUE")

        sql = f"""
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
            WHERE {' AND '.join(where)}
            ORDER BY risk_score DESC
            LIMIT ?
        """
        params.append(limit)
        return _fetch_all_dicts(self.con, sql, params)

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int = 50,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Use docs.v_call_graph_enriched to pull incoming/outgoing edges.
        We assume that view joins edges to function summaries. :contentReference[oaicite:3]{index=3}
        """
        outgoing: List[Dict[str, Any]] = []
        incoming: List[Dict[str, Any]] = []

        if direction in ("out", "both"):
            out_sql = """
                SELECT *
                FROM docs.v_call_graph_enriched
                WHERE caller_goid_h128 = ?
                ORDER BY callee_qualname
                LIMIT ?
            """
            outgoing = _fetch_all_dicts(self.con, out_sql, [goid_h128, limit])

        if direction in ("in", "both"):
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
        goid_h128: Optional[int] = None,
        urn: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query docs.v_test_to_function for tests that hit a given function. :contentReference[oaicite:4]{index=4}
        """
        if goid_h128 is None and urn is None:
            raise ValueError("Must provide goid_h128 or urn.")

        where = ["repo = ?", "commit = ?"]
        params: List[Any] = [self.repo, self.commit]

        if goid_h128 is not None:
            where.append("function_goid_h128 = ?")
            params.append(goid_h128)
        else:
            where.append("urn = ?")
            params.append(urn)

        sql = f"""
            SELECT *
            FROM docs.v_test_to_function
            WHERE {' AND '.join(where)}
            ORDER BY test_id
        """
        return _fetch_all_dicts(self.con, sql, params)

    def get_file_summary(
        self,
        *,
        rel_path: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Query docs.v_file_summary for a given file, plus a list of functions
        in the file attached as `functions`. :contentReference[oaicite:5]{index=5}
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

    # ---------- Dataset introspection ----------

    _DATASET_TABLES: Dict[str, str] = {
        "goids": "core.goids",
        "goid_crosswalk": "core.goid_crosswalk",
        "call_graph_nodes": "graph.call_graph_nodes",
        "call_graph_edges": "graph.call_graph_edges",
        "cfg_blocks": "graph.cfg_blocks",
        "cfg_edges": "graph.cfg_edges",
        "dfg_edges": "graph.dfg_edges",
        "import_graph_edges": "graph.import_graph_edges",
        "symbol_use_edges": "graph.symbol_use_edges",
        "function_metrics": "analytics.function_metrics",
        "function_types": "analytics.function_types",
        "coverage_functions": "analytics.coverage_functions",
        "goid_risk_factors": "analytics.goid_risk_factors",
        # add more as needed
    }

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        Simple static catalog of key datasets LLMs may want to browse.
        """
        out: List[Dict[str, Any]] = []
        for name, table in sorted(self._DATASET_TABLES.items()):
            out.append(
                {
                    "name": name,
                    "table": table,
                    "description": f"DuckDB table {table} for dataset {name}",
                }
            )
        return out

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Read a slice of a dataset as JSON-serializable rows.

        This is intended for small samples, not full exports (use
        Document Output Parquet/JSONL for that). :contentReference[oaicite:6]{index=6}
        """
        table = self._DATASET_TABLES.get(dataset_name)
        if not table:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        sql = f"SELECT * FROM {table} LIMIT ? OFFSET ?"
        return _fetch_all_dicts(self.con, sql, [limit, offset])
