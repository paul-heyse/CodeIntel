"""Run SCIP indexing and register outputs for downstream analytics."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from codeintel.config import ScipIngestStepConfig, ToolsConfig
from codeintel.config.schemas.sql_builder import GOID_CROSSWALK_UPDATE_SCIP
from codeintel.ingestion.change_tracker import (
    ChangeTracker,
    IncrementalIngestOps,
    SupportsFullRebuild,
    run_incremental_ingest,
)
from codeintel.ingestion.common import ModuleRecord
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.ingestion.tool_service import ToolExecutionError, ToolNotFoundError, ToolService
from codeintel.storage.gateway import DuckDBConnection, StorageGateway
from codeintel.storage.sql_helpers import ensure_schema

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScipIngestResult:
    """Outcome of SCIP ingestion."""

    status: Literal["success", "unavailable", "failed"]
    index_scip: Path | None
    index_json: Path | None
    reason: str | None = None


@dataclass(frozen=True)
class ScipRuntime:
    """Runtime context for SCIP ingestion."""

    repo_root: Path
    scip_dir: Path
    doc_dir: Path
    con: DuckDBConnection


@dataclass(frozen=True)
class ScipFullBuildContext:
    """Inputs required for a full SCIP rebuild."""

    gateway: StorageGateway
    cfg: ScipIngestStepConfig
    runtime: ScipRuntime
    index_scip: Path
    index_json: Path
    service: ToolService


@dataclass
class ScipIngestOps(SupportsFullRebuild, IncrementalIngestOps[Path]):
    """Incremental ingest operations for SCIP shards and symbol updates."""

    cfg: ScipIngestStepConfig
    runtime: ScipRuntime
    service: ToolService
    shards_dir: Path = field(init=False)

    dataset_name: str = "core.scip_symbols"

    def __post_init__(self) -> None:
        """Initialize shard output directory."""
        self.shards_dir = self.runtime.scip_dir / "docs"
        self.shards_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def module_filter(module: ModuleRecord) -> bool:
        """
        Restrict SCIP shard processing to Python files under src/ (skip tests/docs).

        Returns
        -------
        bool
            True when the module should be ingested for SCIP.
        """
        return module.rel_path.endswith(".py") and module.rel_path.startswith("src/")

    def delete_rows(self, gateway: StorageGateway, rel_paths: Sequence[str]) -> None:
        """
        Remove shard artifacts and clear symbols for deleted files.

        Parameters
        ----------
        gateway
            Storage gateway providing DuckDB connection.
        rel_paths
            Relative paths scheduled for deletion.
        """
        if not rel_paths:
            return
        _remove_shards(self.runtime, rel_paths)
        gateway.con.execute(
            """
            UPDATE core.goid_crosswalk
            SET scip_symbol = NULL
            WHERE repo = ? AND commit = ? AND file_path IN (SELECT * FROM UNNEST(?))
            """,
            [self.cfg.repo, self.cfg.commit, list(rel_paths)],
        )
        _register_scip_view_from_shards(self.runtime.con, self.shards_dir)

    def process_module(self, module: ModuleRecord) -> Iterable[Path]:
        """
        Generate SCIP shard for a single module using ToolService.

        Returns
        -------
        list[Path]
            Paths to generated shard JSON files (empty on failure).
        """
        target = self.runtime.repo_root / module.rel_path
        if not target.is_file():
            log.warning("SCIP incremental: %s missing; skipping", target)
            return []
        shard_id = _shard_name(module.rel_path)
        shard_scip = self.runtime.scip_dir / f"{shard_id}.scip"
        shard_json = self.shards_dir / f"{shard_id}.json"
        try:
            asyncio.run(
                self.service.run_scip_shard(
                    self.runtime.repo_root,
                    rel_paths=[module.rel_path],
                    output_scip=shard_scip,
                    output_json=shard_json,
                )
            )
        except ToolExecutionError as exc:
            log.warning("SCIP shard %s failed: %s", module.rel_path, exc)
            return []
        return [shard_json]

    def insert_rows(self, gateway: StorageGateway, rows: Sequence[Path]) -> None:
        """
        Update views and symbols from shard results.

        Parameters
        ----------
        gateway
            Storage gateway providing DuckDB connection.
        rows
            Shard JSON paths returned from processing.
        """
        shard_paths = [path for path in rows if path is not None]
        _register_scip_view_from_shards(self.runtime.con, self.shards_dir)
        if shard_paths:
            _update_scip_symbols_from_docs(gateway, shard_paths)

    def run_full_rebuild(self, tracker: ChangeTracker) -> bool:
        """
        Execute a full SCIP rebuild when incremental fallback triggers.

        Returns
        -------
        bool
            True when the full rebuild path has been executed.
        """
        index_scip = self.runtime.scip_dir / "index.scip"
        index_json = self.runtime.scip_dir / "index.scip.json"
        context = ScipFullBuildContext(
            gateway=tracker.gateway,
            cfg=self.cfg,
            runtime=self.runtime,
            index_scip=index_scip,
            index_json=index_json,
            service=self.service,
        )
        _run_full_scip(context)
        return True


def ingest_scip(
    gateway: StorageGateway,
    cfg: ScipIngestStepConfig,
    *,
    tracker: ChangeTracker | None = None,
    tool_service: ToolService | None = None,
) -> ScipIngestResult:
    """
    Run scip-python + scip print, register view, and backfill SCIP symbols.

    Returns
    -------
    ScipIngestResult
        Status and artifact paths. Unavailable/failed states do not raise.
    """
    con = gateway.con
    repo_root = cfg.repo_root.resolve()
    if not (repo_root / ".git").is_dir():
        reason = "SCIP ingestion requires a git repository (.git missing)"
        log.warning(reason)
        return ScipIngestResult(
            status="unavailable", index_scip=None, index_json=None, reason=reason
        )

    if cfg.scip_runner is not None and tracker is None:
        return cast("ScipIngestResult", cfg.scip_runner(gateway, cfg))

    tools_config = ToolsConfig.with_overrides(
        scip_python_bin=cfg.scip_python_bin,
        scip_bin=cfg.scip_bin,
    )
    runner = ToolRunner(
        tools_config=tools_config,
        cache_dir=cfg.build_dir / ".tool_cache",
    )
    service = tool_service or ToolService(runner, tools_config)

    scip_dir = cfg.build_dir.resolve() / "scip"
    scip_dir.mkdir(parents=True, exist_ok=True)
    doc_dir = cfg.document_output_dir.resolve()
    doc_dir.mkdir(parents=True, exist_ok=True)
    runtime = ScipRuntime(repo_root=repo_root, scip_dir=scip_dir, doc_dir=doc_dir, con=con)

    result: ScipIngestResult | None = None
    if tracker is None:
        index_scip = scip_dir / "index.scip"
        index_json = scip_dir / "index.scip.json"
        try:
            context = ScipFullBuildContext(
                gateway=gateway,
                cfg=cfg,
                runtime=runtime,
                index_scip=index_scip,
                index_json=index_json,
                service=service,
            )
            result = _run_full_scip(context)
        except ToolNotFoundError as exc:
            reason = f"SCIP tools unavailable: {exc}"
            log.warning(reason)
            result = ScipIngestResult(
                status="unavailable", index_scip=None, index_json=None, reason=reason
            )
        except ToolExecutionError as exc:
            reason = str(exc)
            log.warning("SCIP ingest failed: %s", reason)
            result = ScipIngestResult(
                status="failed", index_scip=None, index_json=None, reason=reason
            )
    else:
        ops = ScipIngestOps(cfg=cfg, runtime=runtime, service=service)
        try:
            run_incremental_ingest(tracker, ops)
            result = ScipIngestResult(status="success", index_scip=None, index_json=None)
        except ToolNotFoundError as exc:
            reason = f"SCIP tools unavailable: {exc}"
            log.warning(reason)
            result = ScipIngestResult(
                status="unavailable", index_scip=None, index_json=None, reason=reason
            )
        except ToolExecutionError as exc:
            reason = str(exc)
            log.warning("Incremental SCIP ingest failed: %s", reason)
            result = ScipIngestResult(
                status="failed", index_scip=None, index_json=None, reason=reason
            )

    if result is None:
        reason = "SCIP ingest did not produce a result"
        log.warning(reason)
        return ScipIngestResult(status="failed", index_scip=None, index_json=None, reason=reason)
    return result


def _copy_artifacts(index_scip: Path, index_json: Path, doc_dir: Path) -> None:
    """Copy SCIP artifacts into the document output directory."""
    shutil.copy2(index_scip, doc_dir / "index.scip")
    shutil.copy2(index_json, doc_dir / "index.scip.json")


def _run_full_scip(context: ScipFullBuildContext) -> ScipIngestResult:
    asyncio.run(
        context.service.run_scip_full(
            context.runtime.repo_root,
            output_scip=context.index_scip,
            output_json=context.index_json,
        )
    )

    writer = context.cfg.artifact_writer or _copy_artifacts
    writer(context.index_scip, context.index_json, context.runtime.doc_dir)

    docs_table = context.runtime.con.execute(
        "SELECT unnest(documents, recursive:=true) AS document FROM read_json(?)",
        [str(context.index_json)],
    ).fetch_arrow_table()
    context.runtime.con.execute("DROP VIEW IF EXISTS scip_index_view")
    context.runtime.con.register("scip_index_view_temp", docs_table)
    context.runtime.con.execute("CREATE VIEW scip_index_view AS SELECT * FROM scip_index_view_temp")

    _update_scip_symbols(context.gateway, context.index_json)
    log.info("SCIP index ingested for %s@%s", context.cfg.repo, context.cfg.commit)
    return ScipIngestResult(
        status="success", index_scip=context.index_scip, index_json=context.index_json
    )


def _shard_name(rel_path: str) -> str:
    """
    Return a filesystem-safe shard name derived from a relative path.

    Returns
    -------
    str
        Shard filename component for the provided relative path.
    """
    return rel_path.replace("/", "__").replace("\\", "__")


def _register_scip_view_from_shards(con: DuckDBConnection, shards_dir: Path) -> None:
    """Rebuild the scip_index_view from JSON shards."""
    con.execute("DROP VIEW IF EXISTS scip_index_view")
    shard_files = list(shards_dir.glob("*.json"))
    if not shard_files:
        return
    docs_table = con.execute(
        "SELECT unnest(documents, recursive:=true) AS document FROM read_json(?)",
        [str(shards_dir / "*.json")],
    ).fetch_arrow_table()
    con.register("scip_index_view_temp", docs_table)
    con.execute("CREATE VIEW scip_index_view AS SELECT * FROM scip_index_view_temp")


def _remove_shards(runtime: ScipRuntime, rel_paths: Sequence[str]) -> None:
    """Delete shard artifacts for removed files."""
    for rel_path in rel_paths:
        shard_id = _shard_name(rel_path)
        (runtime.scip_dir / f"{shard_id}.scip").unlink(missing_ok=True)
        (runtime.scip_dir / "docs" / f"{shard_id}.json").unlink(missing_ok=True)


def _update_scip_symbols(gateway: StorageGateway, index_json: Path) -> None:
    """Populate core.goid_crosswalk.scip_symbol by matching SCIP definitions to GOIDs."""
    con = gateway.con
    ensure_schema(con, "core.goid_crosswalk")
    docs = _load_scip_documents(index_json)
    def_map = _build_definition_map(docs)
    if not def_map:
        return

    updates = _build_symbol_updates(def_map, _fetch_goids(gateway, rel_paths=None))
    if not updates:
        return

    con.executemany(GOID_CROSSWALK_UPDATE_SCIP, updates)
    log.info("Updated SCIP symbols for %d GOIDs", len(updates))


def _update_scip_symbols_from_docs(gateway: StorageGateway, shard_paths: Iterable[Path]) -> None:
    """Incrementally update SCIP symbols from shard JSON payloads."""
    con = gateway.con
    ensure_schema(con, "core.goid_crosswalk")
    docs: list[dict[str, object]] = []
    rel_paths: set[str] = set()
    for shard in shard_paths:
        shard_docs = _load_scip_documents(shard)
        docs.extend(shard_docs)
        for doc in shard_docs:
            rel_path_obj = (
                doc.get("relative_path")
                if isinstance(doc, dict)
                else getattr(doc, "relative_path", None)
            )
            if isinstance(rel_path_obj, str):
                rel_paths.add(rel_path_obj)
    if not docs:
        return

    def_map = _build_definition_map(docs)
    if not def_map:
        return
    if not rel_paths:
        return
    goids = _fetch_goids(gateway, rel_paths=sorted(rel_paths))
    updates = _build_symbol_updates(def_map, goids)
    if not updates:
        return
    con.executemany(GOID_CROSSWALK_UPDATE_SCIP, updates)
    log.info("Incrementally updated SCIP symbols for %d GOIDs", len(updates))


def _load_scip_documents(index_json: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(index_json.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Failed to parse %s for SCIP symbols: %s", index_json, exc)
        return []

    if isinstance(payload, dict):
        docs = payload.get("documents", [])
        return docs if isinstance(docs, list) else []
    if isinstance(payload, list):
        return payload
    return []


def _build_definition_map(docs: list[dict[str, object]]) -> dict[tuple[str, int], str]:
    def_map: dict[tuple[str, int], str] = {}
    for doc in docs:
        rel_path_obj = (
            doc.get("relative_path")
            if isinstance(doc, dict)
            else getattr(doc, "relative_path", None)
        )
        if not isinstance(rel_path_obj, str):
            continue

        occurrences = doc.get("occurrences", []) if isinstance(doc, dict) else []
        if not isinstance(occurrences, list):
            continue
        for occurrence in occurrences:
            if not isinstance(occurrence, dict):
                continue
            roles = occurrence.get("symbol_roles", 0)
            if roles & 1 == 0:
                continue
            rng = occurrence.get("range") or []
            if not isinstance(rng, list) or not rng:
                continue
            start_line = int(rng[0]) + 1
            symbol = occurrence.get("symbol")
            if isinstance(symbol, str):
                def_map[rel_path_obj, start_line] = symbol
    return def_map


def _fetch_goids(
    gateway: StorageGateway, rel_paths: Collection[str] | None
) -> list[tuple[str, str, int, str, str]]:
    params: list[object] = []
    query = """
        SELECT urn, rel_path, start_line, repo, commit
        FROM core.goids
    """
    if rel_paths:
        placeholders = ",".join("?" for _ in rel_paths)
        query += f" WHERE rel_path IN ({placeholders})"
        params.extend(rel_paths)
    return cast(
        "list[tuple[str, str, int, str, str]]",
        gateway.con.execute(query, params).fetchall(),
    )


def _build_symbol_updates(
    def_map: dict[tuple[str, int], str], goids: list[tuple[str, str, int, str, str]]
) -> list[tuple[str, str, str, str]]:
    updates: list[tuple[str, str, str, str]] = []
    for urn, rel_path, start_line, repo, commit in goids:
        symbol = def_map.get((rel_path, int(start_line)))
        if symbol:
            updates.append((symbol, urn, repo, commit))
    return updates
