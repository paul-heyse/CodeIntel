"""Build analytics.entrypoints and analytics.entrypoint_tests tables.

Column definitions and internal helper functions for entrypoint detection.

The pure compute functions are available in ``codeintel.analytics.entrypoints.compute``:
- ``compute_entrypoints_pure`` returns ``EntrypointsResult``

The Hamilton native module is at:
``codeintel.build.hamilton.native.analytics.entrypoints``
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from codeintel.analytics.compute.entrypoints.detection import (
    DetectorSettings,
    detect_entrypoints,
)
from codeintel.analytics.profiles import SLOW_TEST_THRESHOLD_MS
from codeintel.core.paths import normalize_path
from codeintel.ingestion.adapters.filesystem_discovery import FilesystemDiscoveryAdapter

ENTRYPOINTS_COLS = [
    "repo",
    "commit",
    "entrypoint_id",
    "kind",
    "framework",
    "handler_goid_h128",
    "handler_urn",
    "handler_rel_path",
    "handler_module",
    "handler_qualname",
    "http_method",
    "route_path",
    "status_codes",
    "auth_required",
    "command_name",
    "arguments_schema",
    "schedule",
    "trigger",
    "extra",
    "subsystem_id",
    "subsystem_name",
    "tags",
    "owners",
    "tests_touching",
    "failing_tests",
    "slow_tests",
    "flaky_tests",
    "entrypoint_coverage_ratio",
    "last_test_status",
    "created_at",
]
ENTRYPOINT_TESTS_COLS = [
    "repo",
    "commit",
    "entrypoint_id",
    "test_id",
    "test_goid_h128",
    "coverage_ratio",
    "status",
    "duration_ms",
    "created_at",
]

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from codeintel.analytics.ast_features.model import FunctionAstFeatures
    from codeintel.analytics.compute.entrypoints.detection import (
        EntryPointCandidate,
    )
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.catalog import FunctionCatalogProvider
    from codeintel.ingestion.infrastructure.scanning import ScanProfile
    from codeintel.storage.gateway import DuckDBConnection

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
    features: Mapping[int, FunctionAstFeatures]


@dataclass(frozen=True)
class TestSummary:
    """Aggregated test stats for an entrypoint handler."""

    tests_touching: int
    failing_tests: int
    slow_tests: int
    flaky_tests: int
    last_test_status: str


@dataclass(frozen=True)
class EntrypointBuildInputs:
    """Bundled inputs for entrypoint detection.

    This dataclass groups the data dependencies for entrypoint detection,
    reducing function parameter count.
    """

    catalog_provider: FunctionCatalogProvider
    module_map: dict[str, str]
    features_map: Mapping[int, FunctionAstFeatures]
    settings: DetectorSettings | None = None
    scan_profile: ScanProfile | None = None


def _collect_entrypoint_rows(
    *,
    context: EntryPointContext,
    repo_root: Path,
    settings: DetectorSettings,
    scan_profile: ScanProfile | None,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    entrypoint_rows: list[tuple[object, ...]] = []
    test_rows: list[tuple[object, ...]] = []

    for record in FilesystemDiscoveryAdapter.iter_modules(
        context.module_map,
        repo_root,
        logger=log,
        scan_profile=scan_profile,
    ):
        source = FilesystemDiscoveryAdapter.read_module_source(record)
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
    con: DuckDBConnection,
    snapshot: SnapshotRef,
    catalog: FunctionCatalogProvider,
    module_map_override: dict[str, str] | None = None,
    features: Mapping[int, FunctionAstFeatures] | None = None,
) -> EntryPointContext | None:
    module_ctx = _load_module_context(con, snapshot.repo, snapshot.commit)
    if not module_ctx:
        catalog_modules = module_map_override or catalog.catalog().module_by_path
        module_ctx = {
            normalize_path(path): ModuleContext(module=module, tags=[], owners=[])
            for path, module in catalog_modules.items()
        }
    if not module_ctx:
        return None
    coverage_by_goid = _load_coverage_by_goid(con, snapshot.repo, snapshot.commit)
    edges_by_goid = _load_test_edges(con, snapshot.repo, snapshot.commit)
    test_meta = _load_test_meta(con, snapshot.repo, snapshot.commit)
    subsystem_by_module, subsystem_names = _load_subsystem_maps(con, snapshot.repo, snapshot.commit)
    module_map = {path: ctx.module for path, ctx in module_ctx.items()}
    return EntryPointContext(
        repo=snapshot.repo,
        commit=snapshot.commit,
        module_ctx=module_ctx,
        module_map=module_map,
        coverage_by_goid=coverage_by_goid,
        edges_by_goid=edges_by_goid,
        test_meta=test_meta,
        subsystem_by_module=subsystem_by_module,
        subsystem_names=subsystem_names,
        catalog=catalog,
        now=datetime.now(tz=UTC),
        features=features or {},
    )


def _materialize_candidate(
    cand: EntryPointCandidate, ctx: EntryPointContext
) -> tuple[tuple[object, ...], list[tuple[object, ...]]] | None:
    goid = ctx.catalog.lookup_goid(cand.rel_path, cand.lineno, cand.end_lineno, cand.qualname)
    if goid is None:
        log.debug("Unable to resolve GOID for entrypoint %s (%s)", cand.qualname, cand.rel_path)
        return None
    urn = ctx.catalog.urn_for_goid(goid) or ""
    rel_path = normalize_path(cand.rel_path)
    module_info = ctx.module_ctx.get(rel_path)
    if module_info is None:
        log.debug("Module context missing for %s; skipping entrypoint", rel_path)
        return None
    feature_vector = ctx.features.get(goid)

    entrypoint_id = _entrypoint_id(ctx.repo, ctx.commit, cand, urn)
    subsystem_id = ctx.subsystem_by_module.get(module_info.module)
    subsystem_name = ctx.subsystem_names.get(subsystem_id) if subsystem_id is not None else None
    coverage_ratio = ctx.coverage_by_goid.get(goid)
    if coverage_ratio is None:
        edge_coverages = [
            edge.coverage_ratio
            for edge in ctx.edges_by_goid.get(goid, {}).values()
            if edge.coverage_ratio is not None
        ]
        if edge_coverages:
            coverage_ratio = sum(edge_coverages) / len(edge_coverages)
    summary, edge_rows = _summarize_tests(goid, entrypoint_id, ctx)
    extra_payload = cand.extra or {}
    if feature_vector is not None:
        feature_summary = {
            "http_server_libs": sorted(feature_vector.http_server_libs),
            "http_client_libs": sorted(feature_vector.http_client_libs),
            "db_libs": sorted(feature_vector.db_libs),
            "message_libs": sorted(feature_vector.message_libs),
            "uses_network": feature_vector.io_flags.uses_network,
            "uses_db": feature_vector.io_flags.uses_db,
            "uses_filesystem": feature_vector.io_flags.uses_filesystem,
            "uses_subprocess": feature_vector.io_flags.uses_subprocess,
            "uses_concurrency_lib": feature_vector.uses_concurrency_lib,
        }
        extra_payload = {**extra_payload, "ast_features": feature_summary}

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
        _normalize_json(extra_payload),
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


def _load_module_context(con: DuckDBConnection, repo: str, commit: str) -> dict[str, ModuleContext]:
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
        normalized = normalize_path(str(rel_path))
        context[normalized] = ModuleContext(
            module=str(module),
            tags=tags,
            owners=owners,
        )
    return context


def _load_coverage_by_goid(
    con: DuckDBConnection, repo: str, commit: str
) -> dict[int, float | None]:
    coverage: dict[int, float | None] = {}
    rows = con.execute(
        """
        SELECT function_goid_h128, coverage_ratio
        FROM (
            SELECT
                function_goid_h128,
                coverage_ratio,
                executable_lines,
                created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY function_goid_h128
                    ORDER BY executable_lines DESC NULLS LAST, created_at DESC NULLS LAST
                ) AS rn
            FROM analytics.coverage_functions
            WHERE repo = ? AND commit = ?
        ) ranked
        WHERE rn = 1
        """,
        [repo, commit],
    ).fetchall()
    for goid, ratio in rows:
        if goid is None:
            continue
        coverage[int(goid)] = float(ratio) if ratio is not None else None
    return coverage


def _load_test_edges(
    con: DuckDBConnection, repo: str, commit: str
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


def _load_test_meta(con: DuckDBConnection, repo: str, commit: str) -> dict[str, TestMeta]:
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
    con: DuckDBConnection, repo: str, commit: str
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
