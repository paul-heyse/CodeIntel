"""Build symbol definition-to-use edges from SCIP JSON exports."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

import duckdb

from codeintel.config.models import SymbolUsesConfig
from codeintel.ingestion.common import load_module_map, run_batch
from codeintel.models.rows import SymbolUseRow, symbol_use_to_tuple
from codeintel.types import ScipDocument

log = logging.getLogger(__name__)


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

    module_by_path = load_module_map(con, cfg.repo, cfg.commit)
    def_path_by_symbol = _build_def_map(docs)
    rows = _build_symbol_edges(docs, def_path_by_symbol, module_by_path)

    run_batch(
        con,
        "graph.symbol_use_edges",
        [symbol_use_to_tuple(row) for row in rows],
        delete_params=[],
        scope=f"{cfg.repo}@{cfg.commit}",
    )

    log.info(
        "symbol_use_edges build complete: %d edges from %s",
        len(rows),
        scip_path,
    )


def _load_scip_docs(scip_path: Path) -> list[ScipDocument] | None:
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
    return [cast("ScipDocument", doc) for doc in docs_raw if isinstance(doc, dict)]


def _build_def_map(docs: list[ScipDocument]) -> dict[str, str]:
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
    docs: list[ScipDocument],
    def_path_by_symbol: dict[str, str],
    module_by_path: dict[str, str],
) -> list[SymbolUseRow]:
    rows: list[SymbolUseRow] = []

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

            rows.append(
                SymbolUseRow(
                    symbol=symbol,
                    def_path=def_path,
                    use_path=use_path,
                    same_file=same_file,
                    same_module=same_module,
                )
            )
    return rows
