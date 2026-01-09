"""Build analytics.entrypoints and analytics.entrypoint_tests tables.

Column definitions and internal helper functions for entrypoint detection.

The pure compute functions are available in ``codeintel.build.analytics.entrypoints.compute``:
- ``compute_entrypoints_pure`` returns ``EntrypointsResult``

The Hamilton native module is at:
``codeintel.build.hamilton.native.analytics.entrypoints``

Runtime filesystem scanning helpers live in:
``codeintel.build.analytics.entrypoints.runtime``
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.analytics.compute.entrypoints.detection import detect_entrypoints
from codeintel.build.analytics.compute.row_builders import buffer_for_table
from codeintel.build.analytics.utilities.snapshot import require_columns, snapshot_table
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.core.columnar.rows import ColumnarRowBuffer
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.hashing import sha1_short
from codeintel.core.paths import normalize_path
from codeintel.core.query_results import (
    coerce_optional_float,
    coerce_optional_str,
    coerce_str,
)
from codeintel.core.schemas.row_models import columns_for_table_key

ENTRYPOINTS_TABLE_KEY = "analytics.entrypoints"
ENTRYPOINT_TESTS_TABLE_KEY = "analytics.entrypoint_tests"


def _columns_for_table(table_key: str) -> list[str]:
    columns = columns_for_table_key(table_key)
    if not columns:
        msg = f"No schema columns registered for {table_key}"
        raise ValueError(msg)
    return list(columns)


ENTRYPOINTS_COLS = _columns_for_table(ENTRYPOINTS_TABLE_KEY)
ENTRYPOINT_TESTS_COLS = _columns_for_table(ENTRYPOINT_TESTS_TABLE_KEY)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.build.analytics.ast_features.model import FunctionAstFeatures
    from codeintel.build.analytics.compute.entrypoints.detection import (
        DetectorSettings,
        EntryPointCandidate,
    )
    from codeintel.config.primitives import SnapshotRef
    from codeintel.ingestion.infrastructure.scanning import ScanProfile
    from codeintel.storage.catalog import FunctionCatalogProvider

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModuleContext:
    """Per-module context pulled from core.modules."""

    module: str
    tags: object | None
    owners: object | None


@dataclass(frozen=True)
class TestMeta:
    """Metadata for a test from analytics.test_catalog."""

    test_goid_h128: int | None
    status: str | None
    duration_ms: float | None
    flaky: bool | None


@dataclass(frozen=True)
class EntrypointModuleSource:
    """In-memory module source for entrypoint detection."""

    rel_path: str
    module: str
    source: str


@dataclass(frozen=True)
class EntryPointContext:
    """Shared context for entrypoint materialization."""

    repo: str
    commit: str
    module_ctx: dict[str, ModuleContext]
    module_map: dict[str, str]
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


@dataclass(frozen=True)
class EntrypointContextInputs:
    """Optional frames and overrides for entrypoint context building."""

    module_map_override: dict[str, str] | None = None
    features: Mapping[int, FunctionAstFeatures] | None = None
    modules_frame: pa.Table | None = None
    test_catalog_frame: pa.Table | None = None
    subsystem_modules_frame: pa.Table | None = None
    subsystems_frame: pa.Table | None = None
    ctx: ExecutionContext | None = None


def collect_entrypoint_rows(
    *,
    context: EntryPointContext,
    module_sources: Iterable[EntrypointModuleSource],
    settings: DetectorSettings,
) -> tuple[ColumnarRowBuffer, ColumnarRowBuffer]:
    """Collect entrypoint rows from in-memory module sources.

    Returns
    -------
    tuple[ColumnarRowBuffer, ColumnarRowBuffer]
        Entrypoint and entrypoint-test buffers.
    """
    entrypoint_rows = buffer_for_table(ENTRYPOINTS_TABLE_KEY)
    test_rows = buffer_for_table(ENTRYPOINT_TESTS_TABLE_KEY)

    for module_source in module_sources:
        candidates = detect_entrypoints(
            module_source.source,
            rel_path=module_source.rel_path,
            module=module_source.module,
            settings=settings,
        )
        for cand in candidates:
            result = _materialize_candidate(cand, context)
            if result is None:
                continue
            entry_row, tests_for_entry = result
            entrypoint_rows.append(entry_row)
            for test_row in tests_for_entry:
                test_rows.append(test_row)
    return entrypoint_rows, test_rows


def _build_entrypoint_context(
    snapshot: SnapshotRef,
    catalog: FunctionCatalogProvider,
    inputs: EntrypointContextInputs,
) -> EntryPointContext | None:
    module_ctx = _module_context_from_frame(
        inputs.modules_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
        ctx=inputs.ctx,
    )
    if not module_ctx:
        catalog_modules = inputs.module_map_override or catalog.catalog().module_by_path
        module_ctx = {
            normalize_path(path): ModuleContext(module=module, tags=[], owners=[])
            for path, module in catalog_modules.items()
        }
    if not module_ctx:
        return None
    test_meta = _test_meta_from_frame(
        inputs.test_catalog_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
        ctx=inputs.ctx,
    )
    subsystem_by_module, subsystem_names = _subsystem_maps_from_frame(
        inputs.subsystem_modules_frame,
        inputs.subsystems_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
        ctx=inputs.ctx,
    )
    module_map = {path: ctx.module for path, ctx in module_ctx.items()}
    return EntryPointContext(
        repo=snapshot.repo,
        commit=snapshot.commit,
        module_ctx=module_ctx,
        module_map=module_map,
        test_meta=test_meta,
        subsystem_by_module=subsystem_by_module,
        subsystem_names=subsystem_names,
        catalog=catalog,
        now=datetime.now(tz=UTC),
        features=inputs.features or {},
    )


def _materialize_candidate(
    cand: EntryPointCandidate, ctx: EntryPointContext
) -> tuple[dict[str, object], list[dict[str, object]]] | None:
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

    entrypoint_row: dict[str, object] = {
        "repo": ctx.repo,
        "commit": ctx.commit,
        "entrypoint_id": entrypoint_id,
        "kind": cand.kind,
        "framework": cand.framework,
        "handler_goid_h128": _decimal(goid),
        "handler_urn": urn,
        "handler_rel_path": rel_path,
        "handler_module": module_info.module,
        "handler_qualname": cand.qualname,
        "http_method": cand.http_method,
        "route_path": cand.route_path,
        "auth_required": cand.auth_required,
        "command_name": cand.command_name,
        "schedule": cand.schedule,
        "trigger": cand.trigger,
        "subsystem_id": subsystem_id,
        "subsystem_name": subsystem_name,
        "extras": {
            "status_codes": cand.status_codes,
            "arguments_schema": cand.arguments_schema,
            "extra": _normalize_json(extra_payload),
            "tags": _normalize_json(module_info.tags),
            "owners": _normalize_json(module_info.owners),
        },
        "tests_touching": summary.tests_touching,
        "failing_tests": summary.failing_tests,
        "slow_tests": summary.slow_tests,
        "flaky_tests": summary.flaky_tests,
        "last_test_status": summary.last_test_status,
        "created_at": ctx.now,
    }
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
    return sha1_short(raw, length=16, used_for_security=False)


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


def _module_context_from_frame(
    frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | None,
) -> dict[str, ModuleContext]:
    if frame is None or frame.num_rows == 0:
        return {}
    filtered = _rows_for_snapshot(frame, repo=repo, commit=commit, ctx=ctx)
    context: dict[str, ModuleContext] = {}
    for row in filtered:
        rel_path = row.get("path")
        module = row.get("module")
        tags = _normalize_json(row.get("tags"))
        owners = _normalize_json(row.get("owners"))
        normalized = normalize_path(coerce_str(rel_path, ctx="core.modules.path"))
        context[normalized] = ModuleContext(
            module=coerce_str(module, ctx="core.modules.module"),
            tags=tags,
            owners=owners,
        )
    return context


def _test_meta_from_frame(
    frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | None,
) -> dict[str, TestMeta]:
    meta: dict[str, TestMeta] = {}
    if frame is None or frame.num_rows == 0:
        return meta
    filtered = _rows_for_snapshot(frame, repo=repo, commit=commit, ctx=ctx)
    for row in filtered:
        test_id = row.get("test_id")
        test_goid_h128 = row.get("test_goid_h128")
        status = row.get("status")
        duration_ms = row.get("duration_ms")
        flaky = row.get("flaky")
        if test_id is None:
            continue
        test_id_text = coerce_str(test_id, ctx="test_catalog.test_id")
        meta[test_id_text] = TestMeta(
            test_goid_h128=normalize_decimal_id(test_goid_h128),
            status=coerce_optional_str(status, ctx="test_catalog.status"),
            duration_ms=coerce_optional_float(duration_ms, ctx="test_catalog.duration_ms"),
            flaky=bool(flaky) if flaky is not None else None,
        )
    return meta


def _subsystem_maps_from_frame(
    subsystem_modules_frame: pa.Table | None,
    subsystems_frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | None,
) -> tuple[dict[str, str], dict[str, str]]:
    subsystem_by_module: dict[str, str] = {}
    subsystem_names: dict[str, str] = {}
    if subsystem_modules_frame is not None and subsystem_modules_frame.num_rows > 0:
        filtered = _rows_for_snapshot(
            subsystem_modules_frame,
            repo=repo,
            commit=commit,
            ctx=ctx,
        )
        for row in filtered:
            module = row.get("module")
            subsystem_id = row.get("subsystem_id")
            if module is None or subsystem_id is None:
                continue
            subsystem_by_module[coerce_str(module, ctx="subsystem_modules.module")] = coerce_str(
                subsystem_id, ctx="subsystem_modules.subsystem_id"
            )

    if subsystems_frame is not None and subsystems_frame.num_rows > 0:
        filtered = _rows_for_snapshot(subsystems_frame, repo=repo, commit=commit, ctx=ctx)
        for row in filtered:
            subsystem_id = row.get("subsystem_id")
            name = row.get("name")
            if subsystem_id is None or name is None:
                continue
            subsystem_names[coerce_str(subsystem_id, ctx="subsystems.subsystem_id")] = coerce_str(
                name, ctx="subsystems.name"
            )
    return subsystem_by_module, subsystem_names


def _rows_for_snapshot(
    frame: pa.Table,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | None,
) -> list[dict[str, object]]:
    require_columns(frame, ("repo", "commit"))
    filtered = snapshot_table(frame, repo=repo, commit=commit, ctx=ctx)
    return list(iter_rows(filtered))


def _summarize_tests(
    _goid: int,
    _entrypoint_id: str,
    _ctx: EntryPointContext,
) -> tuple[TestSummary, list[dict[str, object]]]:
    return TestSummary(
        tests_touching=0,
        failing_tests=0,
        slow_tests=0,
        flaky_tests=0,
        last_test_status="untested",
    ), []
