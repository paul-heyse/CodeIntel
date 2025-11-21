# src/codeintel/graphs/goid_builder.py

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import duckdb

from ..ingestion.repo_scan import relpath_to_module

log = logging.getLogger(__name__)


@dataclass
class GoidBuilderConfig:
    repo: str
    commit: str
    language: str = "python"


def _compute_goid(
    repo: str,
    commit: str,
    language: str,
    rel_path: str,
    kind: str,
    qualname: str,
    start_line: int,
    end_line: int | None,
) -> int:
    """
    Compute a stable 128-bit GOID integer from the entity descriptor. 
    """
    payload = f"{repo}:{commit}:{language}:{rel_path}:{kind}:{qualname}:{start_line}:{end_line}"
    h = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(h, "big")


def _build_urn(
    repo: str,
    rel_path: str,
    language: str,
    kind: str,
    qualname: str,
    start_line: int,
    end_line: int | None,
) -> str:
    """
    Build a GOID URN following the README format:

      goid:<repo>/<path>#<language>:<kind>:<qualname>?s=<start>&e=<end> 
    """
    base = f"goid:{repo}/{rel_path}#{language}:{kind}:{qualname}"
    if end_line is None:
        return f"{base}?s={start_line}"
    return f"{base}?s={start_line}&e={end_line}"


def build_goids(con: duckdb.DuckDBPyConnection, cfg: GoidBuilderConfig) -> None:
    """
    Populate:
      - core.goids
      - core.goid_crosswalk

    using core.ast_nodes as the source of entities. 
    """
    df = con.execute(
        """
        SELECT
            path,
            node_type,
            name,
            qualname,
            lineno,
            end_lineno,
            parent_qualname
        FROM core.ast_nodes
        WHERE node_type IN ('Module', 'ClassDef', 'FunctionDef', 'AsyncFunctionDef')
        """
    ).fetch_df()

    if df.empty:
        log.warning("No AST nodes found in core.ast_nodes; cannot build GOIDs.")
        return

    # Clear existing GOIDs for this repo/commit and related crosswalk entries.
    con.execute("DELETE FROM core.goids WHERE repo = ? AND commit = ?", [cfg.repo, cfg.commit])
    con.execute("DELETE FROM core.goid_crosswalk")

    goid_rows: List[Tuple] = []
    xwalk_rows: List[Tuple] = []

    now = datetime.utcnow()

    for _, row in df.iterrows():
        rel_path = str(row["path"]).replace("\\", "/")
        node_type = row["node_type"]
        qualname = row["qualname"]
        parent_qualname = row["parent_qualname"]

        # Derive module path and kind
        module_path = relpath_to_module(Path(rel_path))
        if node_type == "Module":
            kind = "module"
        elif node_type == "ClassDef":
            kind = "class"
        else:
            # FunctionDef / AsyncFunctionDef → function vs method heuristic
            if parent_qualname and parent_qualname != module_path:
                kind = "method"
            else:
                kind = "function"

        start_line = int(row["lineno"]) if row["lineno"] is not None else 1
        end_line = int(row["end_lineno"]) if row["end_lineno"] is not None else None

        goid_h128 = _compute_goid(
            cfg.repo,
            cfg.commit,
            cfg.language,
            rel_path,
            kind,
            qualname,
            start_line,
            end_line,
        )
        urn = _build_urn(
            cfg.repo,
            rel_path,
            cfg.language,
            kind,
            qualname,
            start_line,
            end_line,
        )

        goid_rows.append(
            (
                goid_h128,
                urn,
                cfg.repo,
                cfg.commit,
                rel_path,
                cfg.language,
                kind,
                qualname,
                start_line,
                end_line,
                now,
            )
        )

        xwalk_rows.append(
            (
                urn,  # goid (URN string)
                cfg.language,
                module_path,
                rel_path,
                start_line,
                end_line,
                None,  # scip_symbol
                qualname,  # ast_qualname
                None,  # cst_node_id
                None,  # chunk_id
                None,  # symbol_id
                now,
            )
        )

    if goid_rows:
        con.executemany(
            """
            INSERT INTO core.goids
              (goid_h128, urn, repo, commit, rel_path, language, kind,
               qualname, start_line, end_line, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            goid_rows,
        )

    if xwalk_rows:
        con.executemany(
            """
            INSERT INTO core.goid_crosswalk
              (goid, lang, module_path, file_path, start_line, end_line,
               scip_symbol, ast_qualname, cst_node_id, chunk_id, symbol_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            xwalk_rows,
        )

    log.info(
        "GOID build complete for repo=%s commit=%s: %d entities",
        cfg.repo,
        cfg.commit,
        len(goid_rows),
    )
