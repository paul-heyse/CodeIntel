"""Build minimal control-flow and data-flow graph placeholders for functions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import duckdb

log = logging.getLogger(__name__)


@dataclass
class CFGBuilderConfig:
    """
    Configuration for deriving control-flow/data-flow graph placeholders.

    Parameters
    ----------
    repo : str
        Repository slug being analyzed.
    commit : str
        Commit SHA corresponding to the snapshot.
    """

    repo: str
    commit: str


def build_cfg_and_dfg(con: duckdb.DuckDBPyConnection, cfg: CFGBuilderConfig) -> None:
    """
    Emit minimal CFG/DFG scaffolding for each function GOID.

    Extended Summary
    ----------------
    For every function or method GOID, the builder writes a single `body`
    block covering the function span and leaves the edge tables empty. This
    provides a schema-compatible placeholder that downstream tools can enrich
    with precise control/data flow without blocking the rest of the pipeline.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection to a DuckDB database with GOID metadata available.
    cfg : CFGBuilderConfig
        Repository context identifying which snapshot to process.

    Returns
    -------
    None
        Results are written to `graph.cfg_blocks` with empty edge tables.
    """
    df_funcs = con.execute(
        """
        SELECT function_goid_h128, urn, repo, commit, rel_path, language, kind,
               qualname, start_line, end_line
        FROM analytics.function_metrics
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    ).fetch_df()

    if df_funcs.empty:
        # Fallback: use GOIDs directly if function_metrics isn't computed yet.
        df_funcs = con.execute(
            """
            SELECT goid_h128 AS function_goid_h128,
                   urn, repo, commit, rel_path, language, kind,
                   qualname, start_line, end_line
            FROM core.goids
            WHERE repo = ? AND commit = ?
              AND kind IN ('function', 'method')
            """,
            [cfg.repo, cfg.commit],
        ).fetch_df()

    con.execute("DELETE FROM graph.cfg_blocks")
    con.execute("DELETE FROM graph.cfg_edges")
    con.execute("DELETE FROM graph.dfg_edges")

    block_rows: list[tuple] = []

    if not df_funcs.empty:
        for _, row in df_funcs.iterrows():
            goid = int(row["function_goid_h128"])
            rel_path = row["rel_path"]
            start_line = int(row["start_line"])
            end_line = int(row["end_line"]) if row["end_line"] is not None else start_line

            block_idx = 0
            block_id = f"{goid}:block{block_idx}"
            label = f"body:{block_idx}"
            kind = "body"
            stmts_json = json.dumps([])

            block_rows.append(
                (
                    goid,
                    block_idx,
                    block_id,
                    label,
                    rel_path,
                    start_line,
                    end_line,
                    kind,
                    stmts_json,
                    0,  # in_degree
                    0,  # out_degree
                )
            )

    if block_rows:
        con.executemany(
            """
            INSERT INTO graph.cfg_blocks
              (function_goid_h128, block_idx, block_id, label,
               file_path, start_line, end_line, kind,
               stmts_json, in_degree, out_degree)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            block_rows,
        )

    log.info(
        "CFG/DFG build (minimal) complete for repo=%s commit=%s: %d blocks, 0 edges",
        cfg.repo,
        cfg.commit,
        len(block_rows),
    )
