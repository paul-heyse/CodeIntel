"""Build minimal control-flow and data-flow graph placeholders for functions."""

from __future__ import annotations

import json
import logging

import duckdb

from codeintel.config.models import CFGBuilderConfig
from codeintel.ingestion.common import run_batch
from codeintel.models.rows import CFGBlockRow, cfg_block_to_tuple

log = logging.getLogger(__name__)


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

    block_rows: list[CFGBlockRow] = []

    if not df_funcs.empty:
        for _, row in df_funcs.iterrows():
            goid = int(row["function_goid_h128"])
            rel_path = str(row["rel_path"])
            start_line = int(row["start_line"])
            end_line = int(row["end_line"]) if row["end_line"] is not None else start_line

            block_idx = 0
            block_id = f"{goid}:block{block_idx}"
            label = f"body:{block_idx}"
            kind = "body"
            stmts_json = json.dumps([])

            block_rows.append(
                CFGBlockRow(
                    function_goid_h128=goid,
                    block_idx=block_idx,
                    block_id=block_id,
                    label=label,
                    file_path=rel_path,
                    start_line=start_line,
                    end_line=end_line,
                    kind=kind,
                    stmts_json=stmts_json,
                    in_degree=0,
                    out_degree=0,
                )
            )

    run_batch(
        con,
        "graph.cfg_blocks",
        [cfg_block_to_tuple(row) for row in block_rows],
        delete_params=[],
        scope="cfg_blocks",
    )
    # Edges remain empty placeholders; run_batch keeps tables consistent.

    log.info(
        "CFG/DFG build (minimal) complete for repo=%s commit=%s: %d blocks, 0 edges",
        cfg.repo,
        cfg.commit,
        len(block_rows),
    )
