"""Build symbol definition-to-use edges from SCIP JSON exports."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

from codeintel.config import SymbolUsesStepConfig
from codeintel.graphs.function_catalog import FunctionCatalog
from codeintel.graphs.function_catalog_service import (
    FunctionCatalogProvider,
    FunctionCatalogService,
)
from codeintel.ingestion.common import run_batch
from codeintel.models.rows import SymbolUseRow, symbol_use_to_tuple
from codeintel.storage.gateway import StorageGateway
from codeintel.types import ScipDocument

log = logging.getLogger(__name__)


def build_symbol_use_edges(
    gateway: StorageGateway,
    cfg: SymbolUsesStepConfig,
    catalog_provider: FunctionCatalogProvider | None = None,
) -> None:
    """
    Populate graph.symbol_use_edges from `index.scip.json`.

    The SCIP JSON is expected to be an array of documents:
      { "relative_path": str, "occurrences": [...], "symbols": [...] }

    We treat occurrences with symbol_roles bit 1 as definitions,
    and bit 2 as references, producing edges def_path -> use_path.
    """
    scip_path = cfg.scip_json_path

    docs = load_scip_documents(scip_path)
    if docs is None:
        return

    provider = catalog_provider or FunctionCatalogService.from_db(
        gateway, repo=cfg.repo, commit=cfg.commit
    )
    module_by_path = _merge_module_map(gateway, docs, cfg.repo, cfg.commit, provider.catalog())
    def_path_by_symbol = build_def_map(docs)
    rows = _build_symbol_edges(docs, def_path_by_symbol, module_by_path)

    run_batch(
        gateway,
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


def default_scip_json_path(repo_root: Path, build_dir: Path | None) -> Path | None:
    """
    Return default `index.scip.json` location if present.

    Returns
    -------
    Path | None
        Path when present, otherwise None.
    """
    base = build_dir if build_dir is not None else repo_root / "build"
    scip_path = (base / "scip" / "index.scip.json").resolve()
    return scip_path if scip_path.exists() else None


def load_scip_documents(scip_path: Path) -> list[ScipDocument] | None:
    """
    Load SCIP documents from a JSON file path.

    Returns
    -------
    list[ScipDocument] | None
        Parsed documents, or None when unreadable.
    """
    if not scip_path.exists():
        log.warning("SCIP JSON not found at %s; skipping symbol_use_edges", scip_path)
        return None

    try:
        with scip_path.open("r", encoding="utf-8") as f:
            docs_raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Failed to read %s: %s", scip_path, exc)
        return None

    if isinstance(docs_raw, dict):
        docs_raw = docs_raw.get("documents", [])

    if not isinstance(docs_raw, list):
        log.warning(
            "SCIP JSON root (or 'documents' key) is not a list; aborting symbol_use_edges build."
        )
        return None
    return [cast("ScipDocument", doc) for doc in docs_raw if isinstance(doc, dict)]


def build_def_map(docs: list[ScipDocument]) -> dict[str, str]:
    """
    Map symbol -> defining path from SCIP documents.

    Returns
    -------
    dict[str, str]
        Symbol identifier to definition path mapping.
    """
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
    # Track unique edges to prevent PK violations: (symbol, def_path, use_path)
    seen_edges: set[tuple[str, str, str]] = set()
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
            # Definition=1, Import=2, WriteAccess=4, ReadAccess=8
            # We consider Import, Write, and Read as references/uses.
            is_ref = bool(roles & (2 | 4 | 8))
            if not is_ref:
                continue

            def_path = def_path_by_symbol.get(symbol)
            if not def_path:
                continue

            if (symbol, def_path, use_path) in seen_edges:
                continue
            seen_edges.add((symbol, def_path, use_path))

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


def build_use_def_mapping(
    docs: list[ScipDocument], def_path_by_symbol: dict[str, str]
) -> dict[str, set[str]]:
    """
    Derive mapping of use_path -> definition path(s) from SCIP documents.

    Returns
    -------
    dict[str, set[str]]
        Mapping keyed by use_path to definition paths.
    """
    mapping: dict[str, set[str]] = {}
    for doc in docs:
        use_path_raw = doc.get("relative_path")
        if not use_path_raw:
            continue
        use_path = str(use_path_raw).replace("\\", "/")
        for occ in doc.get("occurrences", []):
            symbol = occ.get("symbol")
            if not symbol:
                continue
            roles = int(occ.get("symbol_roles", 0))
            is_ref = bool(roles & (2 | 4 | 8))
            if not is_ref:
                continue
            def_path = def_path_by_symbol.get(symbol)
            if not def_path:
                continue
            mapping.setdefault(use_path, set()).add(def_path)
    return mapping


def _collect_missing_paths(docs: list[ScipDocument], module_by_path: dict[str, str]) -> set[str]:
    missing: set[str] = set()
    for doc in docs:
        use_path = doc.get("relative_path")
        if use_path:
            use_path = str(use_path).replace("\\", "/")
            if use_path not in module_by_path:
                missing.add(use_path)
        for occ in doc.get("occurrences", []):
            symbol = occ.get("symbol")
            if not symbol:
                continue
            def_path = doc.get("relative_path") if int(occ.get("symbol_roles", 0)) & 1 else None
            if def_path:
                def_path = str(def_path).replace("\\", "/")
                if def_path not in module_by_path:
                    missing.add(def_path)
    return missing


def _load_modules_map(
    gateway: StorageGateway, repo: str, commit: str, paths: set[str] | None = None
) -> dict[str, str]:
    rows = gateway.con.execute(
        """
        SELECT path, module
        FROM core.modules
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    mapping = {str(path).replace("\\", "/"): str(module) for path, module in rows}
    if paths is None:
        return mapping
    normalized_paths = {path.replace("\\", "/") for path in paths}
    return {path: module for path, module in mapping.items() if path in normalized_paths}


def _merge_module_map(
    gateway: StorageGateway,
    docs: list[ScipDocument],
    repo: str,
    commit: str,
    catalog: FunctionCatalog,
) -> dict[str, str]:
    """
    Combine catalog module map with DB modules for missing paths.

    Returns
    -------
    dict[str, str]
        Normalized path -> module mapping combining catalog and DB.
    """
    base_map = {path.replace("\\", "/"): module for path, module in catalog.module_by_path.items()}
    if not base_map:
        return _load_modules_map(gateway, repo, commit)

    missing_paths = _collect_missing_paths(docs, base_map)
    if missing_paths:
        db_map = _load_modules_map(gateway, repo, commit, paths=missing_paths)
        for path, module in db_map.items():
            base_map.setdefault(path, module)
    return base_map
