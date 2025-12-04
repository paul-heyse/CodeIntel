"""Symbol uses builder plugin using factory pattern.

This module provides the symbol uses builder as a graph plugin. All
orchestration logic for building definition-to-use edges from SCIP JSON
exports is here.

Uses resource injection pattern via ctx.require() to access storage.

Architecture notes:
- Pure computation functions are in graphs.compute.symbols
- This plugin orchestrates file I/O and database persistence
- The compute layer is stateless and testable in isolation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from codeintel.config import SymbolUsesStepConfig
from codeintel.config.datasets import SymbolUseRow as DatasetSymbolUseRow
from codeintel.config.datasets import symbol_use_to_tuple
from codeintel.config.primitives import BuildPaths
from codeintel.core.types import (
    ScipDocument,
    ScipOccurrence,
    normalize_scip_document,
    validate_scip_document,
)
from codeintel.graphs.catalog import (
    FunctionCatalog,
    FunctionCatalogProvider,
    FunctionCatalogService,
)
from codeintel.graphs.compute.symbols import (
    parse_symbol_roles,
)
from codeintel.graphs.core import (
    ComputationResult,
    GraphPluginExecutionContext,
    GraphPluginProtocol,
    make_builder_plugin,
)
from codeintel.graphs.resources import StorageResource
from codeintel.ingestion.services.storage import IngestStorageService
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


def build_symbol_use_edges(
    gateway: StorageGateway,
    cfg: SymbolUsesStepConfig,
    catalog_provider: FunctionCatalogProvider | None = None,
) -> None:
    """Populate graph.symbol_use_edges from index.scip.json.

    The SCIP JSON is expected to be an array of documents:
      { "relative_path": str, "occurrences": [...], "symbols": [...] }

    We treat occurrences with symbol_roles bit 1 as definitions,
    and bit 2 as references, producing edges def_path -> use_path.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    cfg
        Symbol uses step configuration.
    catalog_provider
        Optional catalog provider for function metadata.
    """
    scip_path = cfg.resolved_scip_json_path

    docs = load_scip_documents(scip_path)
    if docs is None:
        return
    log.info("Loaded %d SCIP documents from %s", len(docs), scip_path)

    provider = catalog_provider or FunctionCatalogService.from_db(
        gateway, repo=cfg.repo, commit=cfg.commit
    )
    module_by_path = _merge_module_map(gateway, docs, cfg.repo, cfg.commit, provider.catalog())
    def_path_by_symbol = build_def_map(docs)
    rows = _build_symbol_edges(docs, def_path_by_symbol, module_by_path)

    storage_service = IngestStorageService.from_gateway(gateway)
    storage_service.run_batch(
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
    """Return default index.scip.json location if present.

    Parameters
    ----------
    repo_root
        Repository root directory.
    build_dir
        Optional build directory override.

    Returns
    -------
    Path | None
        Path when present, otherwise None.
    """
    base = build_dir if build_dir is not None else repo_root / "build"
    scip_path = (base / "scip" / "index.scip.json").resolve()
    return scip_path if scip_path.exists() else None


def load_scip_documents(scip_path: Path | None) -> list[ScipDocument] | None:
    """Load SCIP documents from a JSON file path.

    Parameters
    ----------
    scip_path
        Path to the SCIP JSON file.

    Returns
    -------
    list[ScipDocument] | None
        Parsed documents, or None when unreadable.
    """
    if scip_path is None or not scip_path.exists():
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
    normalized_docs: list[ScipDocument] = []
    skipped = 0
    for raw in docs_raw:
        if not isinstance(raw, dict):
            skipped += 1
            continue
        normalized = normalize_scip_document(raw)
        if normalized is None:
            skipped += 1
            continue
        try:
            validate_scip_document(normalized)
        except ValueError as exc:
            log.debug("Skipping invalid SCIP document: %s", exc)
            skipped += 1
            continue
        normalized_docs.append(normalized)
    if skipped:
        log.warning("Skipped %d invalid SCIP documents while loading %s", skipped, scip_path)
    return normalized_docs


def _symbol_roles(occurrence: ScipOccurrence) -> int:
    """Return normalized symbol_roles bits from an occurrence.

    Delegates to the compute layer for parsing logic.

    Parameters
    ----------
    occurrence
        SCIP occurrence entry providing symbol role flags.

    Returns
    -------
    int
        Symbol role bitmask, defaulting to 0 when missing.
    """
    return parse_symbol_roles(occurrence.get("symbol_roles"))


def build_def_map(docs: list[ScipDocument]) -> dict[str, str]:
    """Map symbol -> defining path from SCIP documents.

    Parameters
    ----------
    docs
        List of SCIP documents to process.

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
            roles = _symbol_roles(occ)
            is_def = bool(roles & 1)  # definition bit
            if is_def and symbol not in def_path_by_symbol:
                def_path_by_symbol[symbol] = rel_path
    return def_path_by_symbol


def _build_symbol_edges(
    docs: list[ScipDocument],
    def_path_by_symbol: dict[str, str],
    module_by_path: dict[str, str],
) -> list[DatasetSymbolUseRow]:
    """Build symbol use edges from SCIP documents.

    Parameters
    ----------
    docs
        List of SCIP documents to process.
    def_path_by_symbol
        Mapping from symbol to definition path.
    module_by_path
        Mapping from path to module name.

    Returns
    -------
    list[DatasetSymbolUseRow]
        Symbol use edge rows for persistence.
    """
    # Track unique edges to prevent PK violations: (symbol, def_path, use_path)
    seen_edges: set[tuple[str, str, str]] = set()
    rows: list[DatasetSymbolUseRow] = []

    for doc in docs:
        use_path = doc.get("relative_path")
        if not use_path:
            continue
        use_path = str(use_path).replace("\\", "/")

        for occ in doc.get("occurrences", []):
            symbol = occ.get("symbol")
            if not symbol:
                continue
            roles = _symbol_roles(occ)
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
                DatasetSymbolUseRow(
                    symbol=symbol,
                    def_path=def_path,
                    use_path=use_path,
                    same_file=same_file,
                    same_module=same_module,
                    def_goid_h128=None,
                    use_goid_h128=None,
                )
            )
    return rows


def build_use_def_mapping(
    docs: list[ScipDocument], def_path_by_symbol: dict[str, str]
) -> dict[str, set[str]]:
    """Derive mapping of use_path -> definition path(s) from SCIP documents.

    Parameters
    ----------
    docs
        List of SCIP documents to process.
    def_path_by_symbol
        Mapping from symbol to definition path.

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
            roles = _symbol_roles(occ)
            is_ref = bool(roles & (2 | 4 | 8))
            if not is_ref:
                continue
            def_path = def_path_by_symbol.get(symbol)
            if not def_path:
                continue
            mapping.setdefault(use_path, set()).add(def_path)
    return mapping


def _collect_missing_paths(docs: list[ScipDocument], module_by_path: dict[str, str]) -> set[str]:
    """Collect paths referenced in SCIP documents but missing from module map.

    Parameters
    ----------
    docs
        List of SCIP documents to process.
    module_by_path
        Current mapping from path to module name.

    Returns
    -------
    set[str]
        Paths not found in the module map.
    """
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
            roles = _symbol_roles(occ)
            def_path = doc.get("relative_path") if roles & 1 else None
            if def_path:
                def_path = str(def_path).replace("\\", "/")
                if def_path not in module_by_path:
                    missing.add(def_path)
    return missing


def _load_modules_map(
    gateway: StorageGateway, repo: str, commit: str, paths: set[str] | None = None
) -> dict[str, str]:
    """Load module mappings from the database.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    repo
        Repository identifier.
    commit
        Commit hash.
    paths
        Optional set of paths to filter by.

    Returns
    -------
    dict[str, str]
        Path to module name mapping.
    """
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
    """Combine catalog module map with DB modules for missing paths.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    docs
        List of SCIP documents being processed.
    repo
        Repository identifier.
    commit
        Commit hash.
    catalog
        Function catalog with module mappings.

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


def _build_symbol_uses(ctx: GraphPluginExecutionContext) -> ComputationResult:
    """Build symbol definition-to-use edges from SCIP JSON exports.

    Uses resource injection to access storage.

    Returns
    -------
    ComputationResult
        Success result after building symbol use edges.
    """
    storage = ctx.require(StorageResource)
    gateway = storage.gateway

    paths = BuildPaths.from_layout(repo_root=ctx.snapshot.repo_root)
    cfg = SymbolUsesStepConfig(snapshot=ctx.snapshot, paths=paths)
    build_symbol_use_edges(gateway, cfg)
    return ComputationResult.ok()


symbol_uses_builder_plugin = make_builder_plugin(
    name="symbol_uses_builder",
    computation=_build_symbol_uses,
    stage="edges",
    produces_graph_kinds=(),
    depends_on=("goid_builder",),
    provides=("symbol_uses",),
    produces_tables=("graph.symbol_use_edges",),
)


def get_symbol_uses_builder_plugin() -> GraphPluginProtocol:
    """Return the symbol uses builder plugin instance.

    Returns
    -------
    GraphPluginProtocol
        The configured symbol uses builder plugin.
    """
    return symbol_uses_builder_plugin


__all__ = [
    "build_def_map",
    "build_symbol_use_edges",
    "build_use_def_mapping",
    "default_scip_json_path",
    "get_symbol_uses_builder_plugin",
    "load_scip_documents",
    "symbol_uses_builder_plugin",
]
