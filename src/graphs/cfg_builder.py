# src/codeintel/graphs/cfg_builder.py

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import duckdb

log = logging.getLogger(__name__)


@dataclass
class CFGBuilderConfig:
    repo: str
    commit: str


def build_cfg_and_dfg(con: duckdb.DuckDBPyConnection, cfg: CFGBuilderConfig) -> None:
    """
    Minimal CFG/DFG builder.

    For each function/method GOID, we emit a single `body` block that spans
    the function's lines, and no CFG/DFG edges. This gives you a usable
    `cfg_blocks` table that can be enriched later with a full CFG/DFG
    implementation. 
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

    block_rows: List[Tuple] = []

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
