"""Build symbol definition-to-use edges from SCIP JSON exports."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)


@dataclass
class SymbolUsesConfig:
    """
    Configuration for deriving symbol use edges.

    Parameters
    ----------
    repo_root : Path
        Root of the repository containing source files.
    scip_json_path : Path
        Path to the `index.scip.json` document produced by scip-python.
    """

    repo_root: Path
    scip_json_path: Path


def build_symbol_use_edges(
    con: duckdb.DuckDBPyConnection,
    cfg: SymbolUsesConfig,
) -> None:
    """
    Populate graph.symbol_use_edges from `index.scip.json`. 

    The SCIP JSON is expected to be an array of documents:
      { "relative_path": str, "occurrences": [...], "symbols": [...] }

    We treat occurrences with symbol_roles bit 1 as definitions,
    and bit 2 as references, producing edges def_path -> use_path.
    """
    repo_root = cfg.repo_root.resolve()
    scip_path = cfg.scip_json_path

    if not scip_path.exists():
        log.warning("SCIP JSON not found at %s; skipping symbol_use_edges", scip_path)
        return

    with scip_path.open("r", encoding="utf-8") as f:
        docs = json.load(f)

    if not isinstance(docs, list):
        log.warning("SCIP JSON root is not a list; aborting symbol_use_edges build.")
        return

    # Map path -> module for same_module flag
    df_modules = con.execute("SELECT path, module FROM core.modules").fetch_df()
    module_by_path: dict[str, str] = {}
    if not df_modules.empty:
        for _, row in df_modules.iterrows():
            module_by_path[str(row["path"]).replace("\\", "/")] = row["module"]

    # 1) First pass: map symbol -> def_path
    def_path_by_symbol: dict[str, str] = {}
    for doc in docs:
        rel_path = doc.get("relative_path")
        if not rel_path:
            continue
        rel_path = str(rel_path).replace("\\", "/")

        for occ in doc.get("occurrences", []):
            symbol = occ.get("symbol")
            if not symbol:
                continue
            roles = int(occ.get("symbol_roles", 0))
            is_def = bool(roles & 1)  # definition bit
            if is_def and symbol not in def_path_by_symbol:
                def_path_by_symbol[symbol] = rel_path

    # 2) Second pass: references -> edges
    rows: list[tuple] = []

    for doc in docs:
        use_path = doc.get("relative_path")
        if not use_path:
            continue
        use_path = str(use_path).replace("\\", "/")

        for occ in doc.get("occurrences", []):
            symbol = occ.get("symbol")
            if not symbol:
                continue
            roles = int(occ.get("symbol_roles", 0))
            is_ref = bool(roles & 2)  # reference bit
            if not is_ref:
                continue

            def_path = def_path_by_symbol.get(symbol)
            if not def_path:
                continue

            same_file = def_path == use_path
            m_def = module_by_path.get(def_path)
            m_use = module_by_path.get(use_path)
            same_module = m_def is not None and m_def == m_use

            rows.append((symbol, def_path, use_path, same_file, same_module))

    con.execute("DELETE FROM graph.symbol_use_edges")

    if rows:
        con.executemany(
            """
            INSERT INTO graph.symbol_use_edges
              (symbol, def_path, use_path, same_file, same_module)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )

    log.info(
        "symbol_use_edges build complete: %d edges from %s",
        len(rows),
        scip_path,
    )
