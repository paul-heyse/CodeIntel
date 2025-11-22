"""Build symbol definition-to-use edges from SCIP JSON exports."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

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
    repo: str
    commit: str


class ScipDoc(TypedDict, total=False):
    """Typed representation of a SCIP JSON document."""

    relative_path: str
    occurrences: list[dict[str, Any]]
    symbols: list[dict[str, Any]]


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
    scip_path = cfg.scip_json_path

    docs = _load_scip_docs(scip_path)
    if docs is None:
        return

    module_by_path = _module_map(con, cfg)
    def_path_by_symbol = _build_def_map(docs)
    rows = _build_symbol_edges(docs, def_path_by_symbol, module_by_path)

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


def _load_scip_docs(scip_path: Path) -> list[ScipDoc] | None:
    if not scip_path.exists():
        log.warning("SCIP JSON not found at %s; skipping symbol_use_edges", scip_path)
        return None

    try:
        with scip_path.open("r", encoding="utf-8") as f:
            docs_raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Failed to read %s: %s", scip_path, exc)
        return None

    if not isinstance(docs_raw, list):
        log.warning("SCIP JSON root is not a list; aborting symbol_use_edges build.")
        return None
    return [cast("ScipDoc", doc) for doc in docs_raw if isinstance(doc, dict)]


def _module_map(con: duckdb.DuckDBPyConnection, cfg: SymbolUsesConfig) -> dict[str, str]:
    df_modules = con.execute(
        """
        SELECT path, module
        FROM core.modules
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    ).fetch_df()
    module_by_path: dict[str, str] = {}
    if not df_modules.empty:
        for _, row in df_modules.iterrows():
            module_by_path[str(row["path"]).replace("\\", "/")] = str(row["module"])
    return module_by_path


def _build_def_map(docs: list[ScipDoc]) -> dict[str, str]:
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
    return def_path_by_symbol


def _build_symbol_edges(
    docs: list[ScipDoc],
    def_path_by_symbol: dict[str, str],
    module_by_path: dict[str, str],
) -> list[tuple]:
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
    return rows
