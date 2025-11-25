"""Build analytics.entrypoints and analytics.entrypoint_tests tables."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from codeintel.analytics.entrypoint_detectors import (
    DetectorSettings,
    EntryPointCandidate,
    detect_entrypoints,
)
from codeintel.analytics.profiles import SLOW_TEST_THRESHOLD_MS
from codeintel.config.models import EntryPointsConfig
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.function_catalog_service import (
    FunctionCatalogProvider,
    FunctionCatalogService,
)
from codeintel.ingestion.common import iter_modules, read_module_source
from codeintel.storage.gateway import StorageGateway
from codeintel.utils.paths import normalize_rel_path

if TYPE_CHECKING:
    from codeintel.ingestion.source_scanner import ScanConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModuleContext:
    """Per-module context pulled from core.modules."""

    module: str
    tags: object | None
    owners: object | None


@dataclass(frozen=True)
class TestEdge:
    """Coverage edge from a test to a function GOID."""

    test_id: str
    coverage_ratio: float | None


@dataclass(frozen=True)
class TestMeta:
    """Metadata for a test from analytics.test_catalog."""

    test_goid_h128: int | None
    status: str | None
    duration_ms: float | None
    flaky: bool | None


@dataclass(frozen=True)
class EntryPointContext:
    """Shared context for entrypoint materialization."""

    repo: str
    commit: str
    module_ctx: dict[str, ModuleContext]
    module_map: dict[str, str]
    coverage_by_goid: dict[int, float | None]
    edges_by_goid: dict[int, dict[str, TestEdge]]
    test_meta: dict[str, TestMeta]
    subsystem_by_module: dict[str, str]
    subsystem_names: dict[str, str]
    catalog: FunctionCatalogProvider
    now: datetime


@dataclass(frozen=True)
class TestSummary:
    """Aggregated test stats for an entrypoint handler."""

    tests_touching: int
    failing_tests: int
    slow_tests: int
    flaky_tests: int
    last_test_status: str


def build_entrypoints(
    gateway: StorageGateway,
    cfg: EntryPointsConfig,
    *,
    catalog_provider: FunctionCatalogProvider | None = None,
) -> None:
    """
    Populate analytics.entrypoints and analytics.entrypoint_tests.

    Parameters
    ----------
    gateway
        Storage gateway with active DuckDB connection.
    cfg
        EntryPointsConfig specifying repo context and detection toggles.
    catalog_provider
        Optional function catalog to reuse across steps.
    """
    con = gateway.con
    ensure_schema(con, "analytics.entrypoints")
    ensure_schema(con, "analytics.entrypoint_tests")

    con.execute(
        "DELETE FROM analytics.entrypoints WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )
    con.execute(
        "DELETE FROM analytics.entrypoint_tests WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    catalog = catalog_provider or FunctionCatalogService.from_db(
        gateway, repo=cfg.repo, commit=cfg.commit
    )
    context = _build_entrypoint_context(con, cfg, catalog)
    if context is None:
        log.warning("No modules available to scan for entrypoints in %s@%s", cfg.repo, cfg.commit)
        return

    settings = DetectorSettings(
        detect_fastapi=cfg.detect_fastapi,
        detect_flask=cfg.detect_flask,
        detect_click=cfg.detect_click,
        detect_typer=cfg.detect_typer,
        detect_cron=cfg.detect_cron,
        detect_django=cfg.detect_django,
        detect_celery=cfg.detect_celery,
        detect_airflow=cfg.detect_airflow,
        detect_generic_routes=cfg.detect_generic_routes,
    )
    entrypoint_rows, test_rows = _collect_entrypoint_rows(
        context=context, repo_root=cfg.repo_root, settings=settings, scan_config=cfg.scan_config
    )

    if entrypoint_rows:
        con.executemany(
            """
            INSERT INTO analytics.entrypoints (
                repo, commit, entrypoint_id, kind, framework,
                handler_goid_h128, handler_urn, handler_rel_path, handler_module, handler_qualname,
                http_method, route_path, status_codes, auth_required,
                command_name, arguments_schema, schedule, trigger, extra,
                subsystem_id, subsystem_name, tags, owners,
                tests_touching, failing_tests, slow_tests, flaky_tests,
                entrypoint_coverage_ratio, last_test_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            entrypoint_rows,
        )
    if test_rows:
        con.executemany(
            """
            INSERT INTO analytics.entrypoint_tests (
                repo, commit, entrypoint_id, test_id, test_goid_h128,
                coverage_ratio, status, duration_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            test_rows,
        )

    log.info(
        "entrypoints populated: %d entrypoints, %d entrypoint_test edges for %s@%s",
        len(entrypoint_rows),
        len(test_rows),
        cfg.repo,
        cfg.commit,
    )


def _collect_entrypoint_rows(
    *,
    context: EntryPointContext,
    repo_root: Path,
    settings: DetectorSettings,
    scan_config: ScanConfig | None,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    entrypoint_rows: list[tuple[object, ...]] = []
    test_rows: list[tuple[object, ...]] = []

    for record in iter_modules(context.module_map, repo_root, logger=log, scan_config=scan_config):
        source = read_module_source(record, logger=log)
        if source is None:
            continue
        candidates = detect_entrypoints(
            source,
            rel_path=record.rel_path,
            module=record.module_name,
            settings=settings,
        )
        for cand in candidates:
            rows = _materialize_candidate(cand, context)
            if rows is None:
                continue
            entry_row, tests_for_entry = rows
            entrypoint_rows.append(entry_row)
            test_rows.extend(tests_for_entry)
    return entrypoint_rows, test_rows


def _build_entrypoint_context(
    con: duckdb.DuckDBPyConnection,
    cfg: EntryPointsConfig,
    catalog: FunctionCatalogProvider,
) -> EntryPointContext | None:
    module_ctx = _load_module_context(con, cfg.repo, cfg.commit)
    if not module_ctx:
        catalog_modules = catalog.catalog().module_by_path
        module_ctx = {
            normalize_rel_path(path): ModuleContext(module=module, tags=[], owners=[])
            for path, module in catalog_modules.items()
        }
    if not module_ctx:
        return None
    coverage_by_goid = _load_coverage_by_goid(con, cfg.repo, cfg.commit)
    edges_by_goid = _load_test_edges(con, cfg.repo, cfg.commit)
    test_meta = _load_test_meta(con, cfg.repo, cfg.commit)
    subsystem_by_module, subsystem_names = _load_subsystem_maps(con, cfg.repo, cfg.commit)
    module_map = {path: ctx.module for path, ctx in module_ctx.items()}
    return EntryPointContext(
        repo=cfg.repo,
        commit=cfg.commit,
        module_ctx=module_ctx,
        module_map=module_map,
        coverage_by_goid=coverage_by_goid,
        edges_by_goid=edges_by_goid,
        test_meta=test_meta,
        subsystem_by_module=subsystem_by_module,
        subsystem_names=subsystem_names,
        catalog=catalog,
        now=datetime.now(tz=UTC),
    )


def _materialize_candidate(
    cand: EntryPointCandidate, ctx: EntryPointContext
) -> tuple[tuple[object, ...], list[tuple[object, ...]]] | None:
    goid = ctx.catalog.lookup_goid(cand.rel_path, cand.lineno, cand.end_lineno, cand.qualname)
    if goid is None:
        log.debug("Unable to resolve GOID for entrypoint %s (%s)", cand.qualname, cand.rel_path)
        return None
    urn = ctx.catalog.urn_for_goid(goid) or ""
    rel_path = normalize_rel_path(cand.rel_path)
    module_info = ctx.module_ctx.get(rel_path)
    if module_info is None:
        log.debug("Module context missing for %s; skipping entrypoint", rel_path)
        return None

    entrypoint_id = _entrypoint_id(ctx.repo, ctx.commit, cand, urn)
    subsystem_id = ctx.subsystem_by_module.get(module_info.module)
    subsystem_name = ctx.subsystem_names.get(subsystem_id) if subsystem_id is not None else None
    coverage_ratio = ctx.coverage_by_goid.get(goid)
    summary, edge_rows = _summarize_tests(goid, entrypoint_id, ctx)

    entrypoint_row = (
        ctx.repo,
        ctx.commit,
        entrypoint_id,
        cand.kind,
        cand.framework,
        _decimal(goid),
        urn,
        rel_path,
        module_info.module,
        cand.qualname,
        cand.http_method,
        cand.route_path,
        cand.status_codes,
        cand.auth_required,
        cand.command_name,
        cand.arguments_schema,
        cand.schedule,
        cand.trigger,
        _normalize_json(cand.extra),
        subsystem_id,
        subsystem_name,
        _normalize_json(module_info.tags),
        _normalize_json(module_info.owners),
        summary.tests_touching,
        summary.failing_tests,
        summary.slow_tests,
        summary.flaky_tests,
        coverage_ratio,
        summary.last_test_status,
        ctx.now,
    )
    return entrypoint_row, edge_rows


def _entrypoint_id(repo: str, commit: str, cand: EntryPointCandidate, urn: str) -> str:
    raw = ":".join(
        [
            repo,
            commit,
            cand.kind,
            cand.framework or "",
            urn,
            cand.http_method or "",
            cand.route_path or "",
            cand.command_name or "",
            cand.schedule or "",
        ]
    )
    return hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def _normalize_json(value: object | None) -> object | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _decimal(value: int) -> Decimal:
    return Decimal(value)


def _load_module_context(
    con: duckdb.DuckDBPyConnection, repo: str, commit: str
) -> dict[str, ModuleContext]:
    rows = con.execute(
        """
        SELECT path, module, tags, owners
        FROM core.modules
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    context: dict[str, ModuleContext] = {}
    for rel_path, module, tags, owners in rows:
        normalized = normalize_rel_path(str(rel_path))
        context[normalized] = ModuleContext(
            module=str(module),
            tags=tags,
            owners=owners,
        )
    return context


def _load_coverage_by_goid(
    con: duckdb.DuckDBPyConnection, repo: str, commit: str
) -> dict[int, float | None]:
    coverage: dict[int, float | None] = {}
    rows = con.execute(
        """
        SELECT function_goid_h128, coverage_ratio
        FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    for goid, ratio in rows:
        if goid is None:
            continue
        coverage[int(goid)] = float(ratio) if ratio is not None else None
    return coverage


def _load_test_edges(
    con: duckdb.DuckDBPyConnection, repo: str, commit: str
) -> dict[int, dict[str, TestEdge]]:
    edges_by_goid: dict[int, dict[str, TestEdge]] = defaultdict(dict)
    rows = con.execute(
        """
        SELECT function_goid_h128, test_id, coverage_ratio
        FROM analytics.test_coverage_edges
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    for goid, test_id, coverage_ratio in rows:
        if goid is None or test_id is None:
            continue
        edges_by_goid[int(goid)][str(test_id)] = TestEdge(
            test_id=str(test_id),
            coverage_ratio=float(coverage_ratio) if coverage_ratio is not None else None,
        )
    return edges_by_goid


def _load_test_meta(con: duckdb.DuckDBPyConnection, repo: str, commit: str) -> dict[str, TestMeta]:
    meta: dict[str, TestMeta] = {}
    rows = con.execute(
        """
        SELECT test_id, test_goid_h128, status, duration_ms, flaky
        FROM analytics.test_catalog
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    for test_id, test_goid_h128, status, duration_ms, flaky in rows:
        if test_id is None:
            continue
        meta[str(test_id)] = TestMeta(
            test_goid_h128=int(test_goid_h128) if test_goid_h128 is not None else None,
            status=str(status) if status is not None else None,
            duration_ms=float(duration_ms) if duration_ms is not None else None,
            flaky=bool(flaky) if flaky is not None else None,
        )
    return meta


def _load_subsystem_maps(
    con: duckdb.DuckDBPyConnection, repo: str, commit: str
) -> tuple[dict[str, str], dict[str, str]]:
    subsystem_by_module: dict[str, str] = {}
    subsystem_names: dict[str, str] = {}
    module_rows = con.execute(
        """
        SELECT module, subsystem_id
        FROM analytics.subsystem_modules
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    for module, subsystem_id in module_rows:
        if module is None or subsystem_id is None:
            continue
        subsystem_by_module[str(module)] = str(subsystem_id)

    subsystem_rows = con.execute(
        """
        SELECT subsystem_id, name
        FROM analytics.subsystems
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    for subsystem_id, name in subsystem_rows:
        if subsystem_id is None or name is None:
            continue
        subsystem_names[str(subsystem_id)] = str(name)
    return subsystem_by_module, subsystem_names


def _summarize_tests(
    goid: int,
    entrypoint_id: str,
    ctx: EntryPointContext,
) -> tuple[TestSummary, list[tuple[object, ...]]]:
    edges = ctx.edges_by_goid.get(goid, {})
    if not edges:
        return TestSummary(
            tests_touching=0,
            failing_tests=0,
            slow_tests=0,
            flaky_tests=0,
            last_test_status="untested",
        ), []

    failing = 0
    slow = 0
    flaky = 0
    rows: list[tuple[object, ...]] = []
    statuses: set[str] = set()
    for edge in edges.values():
        meta = ctx.test_meta.get(edge.test_id)
        status = meta.status if meta is not None else None
        duration_ms = meta.duration_ms if meta is not None else None
        flaky_flag = meta.flaky if meta is not None else None
        if status in {"failed", "error"}:
            failing += 1
        if duration_ms is not None and duration_ms > SLOW_TEST_THRESHOLD_MS:
            slow += 1
        if flaky_flag:
            flaky += 1
        if status:
            statuses.add(status)
        rows.append(
            (
                ctx.repo,
                ctx.commit,
                entrypoint_id,
                edge.test_id,
                _decimal(meta.test_goid_h128) if meta and meta.test_goid_h128 is not None else None,
                edge.coverage_ratio,
                status,
                duration_ms,
                ctx.now,
            )
        )

    if failing > 0:
        last_status = "some_failing"
    elif statuses == {"passed"}:
        last_status = "all_passing"
    else:
        last_status = "unknown"

    summary = TestSummary(
        tests_touching=len(edges),
        failing_tests=failing,
        slow_tests=slow,
        flaky_tests=flaky,
        last_test_status=last_status,
    )
    return summary, rows
