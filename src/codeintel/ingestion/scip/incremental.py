"""Incremental SCIP indexing orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.ingestion.engine.infrastructure import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunOptions,
)
from codeintel.ingestion.scip.cli import build_scip_python_args, ensure_pip_available
from codeintel.ingestion.scip.hash_resolver import FileDigestResolver
from codeintel.ingestion.scip.index_store import (
    MergeIndexContext,
    load_index_proto,
    merge_indexes,
    write_index_proto,
)
from codeintel.ingestion.scip.manifest import (
    ScipShardManifest,
    ScipShardRecord,
    load_manifest,
    manifest_path,
    shard_path,
    update_manifest,
    write_manifest,
)
from codeintel.ingestion.scip.paths import resolve_target_base, scip_relative_path
from codeintel.ingestion.scip.policy import (
    ScipIncrementalDecision,
    ScipIncrementalInputs,
    ScipIncrementalPolicy,
)
from codeintel.ingestion.scip.telemetry import ScipRunTelemetry

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.config.models import ToolsConfig
    from codeintel.ingestion.engine.infrastructure import ToolRunner
    from codeintel.ingestion.ports.change_detection import ChangeSet, FileDigest
    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.ingestion.scip.proto_types import IndexProto

log = logging.getLogger(__name__)

_SCIP_TRACE_ENV = "CODEINTEL_SCIP_TRACE"
_SCIP_PROGRESS_INTERVAL_S = 30.0
_SCIP_TRACE_PROGRESS_INTERVAL_S = 5.0


def _resolve_decode_error() -> type[Exception]:
    try:
        module = importlib.import_module("google.protobuf.message")
    except ModuleNotFoundError:
        return RuntimeError
    decode_error = getattr(module, "DecodeError", None)
    if isinstance(decode_error, type) and issubclass(decode_error, Exception):
        return decode_error
    return RuntimeError


_DECODE_ERROR = _resolve_decode_error()


@dataclass(frozen=True)
class ScipIncrementalResult:
    """Outcome of incremental SCIP update."""

    success: bool
    index_path: Path | None
    manifest_path: Path | None
    full_rebuild: bool = False
    updated: bool = False
    error: str | None = None


@dataclass(frozen=True)
class ScipIncrementalConfig:
    """Configuration for incremental SCIP indexing."""

    repo_root: Path
    output_scip: Path
    proto_module_path: Path
    change_set: ChangeSet
    modules: Sequence[ModuleRecord]
    options_hash: str | None
    tools_config: ToolsConfig
    tool_runner: ToolRunner
    scope_paths: Sequence[str] | None
    environment_json: Path | None
    max_file_size_kb: int
    timeout_seconds: int
    target_dir: Path | None
    force_full_rebuild: bool = False
    batch_size: int = 200
    batch_max_bytes: int = 50_000_000
    full_rebuild_threshold_count: int = 1000
    full_rebuild_threshold_ratio: float = 0.3
    full_rebuild_ratio_min_modules: int = 200
    full_rebuild_ratio_min_changed: int = 25
    file_state_by_path: Mapping[str, FileDigest] | None = None
    module_state_by_path: Mapping[str, FileDigest] | None = None
    telemetry: ScipRunTelemetry | None = None


@dataclass(frozen=True)
class _ScipRunConfig:
    repo_root: Path
    tools_config: ToolsConfig
    tool_runner: ToolRunner
    target_base: Path
    timeout_seconds: int
    environment_json: Path | None


@dataclass(frozen=True)
class _ShardPlanContext:
    repo_root: Path
    target_base: Path
    scope_paths: Sequence[str] | None
    max_file_size_kb: int
    scip_dir: Path


@dataclass(frozen=True)
class _ModuleShardPlan:
    rel_path: str
    scip_rel_path: str
    content_hash: str
    size_bytes: int
    shard_path: Path


@dataclass(frozen=True)
class _BatchPlan:
    batch_id: str
    plans: tuple[_ModuleShardPlan, ...]
    size_bytes: int
    rel_paths: tuple[str, ...]


@dataclass(frozen=True)
class _ShardPlanStats:
    total_candidates: int
    hash_computed: int
    hash_reused: int
    hash_source: str | None
    hash_source_breakdown: str | None
    computed_ms: float


@dataclass(frozen=True)
class _ShardIndexResult:
    shard_indexes: tuple[IndexProto, ...]
    shard_updates: dict[str, ScipShardRecord]
    tool_ms: float
    parse_ms: float
    batch_count: int


@dataclass(frozen=True)
class _IncrementalMergeInputs:
    config: ScipIncrementalConfig
    base_index: IndexProto
    shard_indexes: Sequence[IndexProto]
    deleted_paths: tuple[str, ...]
    manifest: ScipShardManifest
    shard_updates: Mapping[str, ScipShardRecord]
    manifest_file: Path


@dataclass(frozen=True)
class _IncrementalRunContext:
    config: ScipIncrementalConfig
    run_config: _ScipRunConfig
    scip_dir: Path
    manifest_file: Path
    target_base: Path
    total_modules: int
    changed_modules: tuple[ModuleRecord, ...]
    deleted_modules: tuple[ModuleRecord, ...]
    changed_count: int
    changed_ratio: float | None


@dataclass(frozen=True)
class _IncrementalPlan:
    context: _IncrementalRunContext
    base_index: IndexProto
    manifest: ScipShardManifest
    changed_plans: tuple[_ModuleShardPlan, ...]
    deleted_paths: tuple[str, ...]
    plan_stats: _ShardPlanStats
    plan_ms: float
    hash_source: str | None


@dataclass(frozen=True)
class _ShardIndexRequest:
    run_config: _ScipRunConfig
    proto_module_path: Path
    plans: Sequence[_ModuleShardPlan]
    options_hash: str | None
    batch_size: int
    batch_max_bytes: int
    telemetry: ScipRunTelemetry | None


def update_index_incremental(
    *,
    config: ScipIncrementalConfig,
) -> ScipIncrementalResult:
    """Update index.scip using incremental per-module indexing.

    Returns
    -------
    ScipIncrementalResult
        Result describing the update outcome and paths.
    """
    start_total = time.perf_counter()
    context = _build_run_context(config)
    telemetry = context.config.telemetry
    decision = _resolve_decision(context)
    _initialize_telemetry(context, decision)
    _log_plan_summary(context, decision)
    if decision.mode == "full":
        result = _full_rebuild(
            run_config=context.run_config,
            output_scip=context.config.output_scip,
            scope_paths=context.config.scope_paths,
            manifest_file=context.manifest_file,
            telemetry=telemetry,
            error=None,
        )
    else:
        base_index, fallback = _load_base_index_or_full_rebuild(context)
        if fallback is not None:
            result = fallback
        elif base_index is None:
            result = ScipIncrementalResult(
                success=False,
                index_path=None,
                manifest_path=None,
                full_rebuild=True,
                updated=False,
                error="SCIP base index missing after fallback",
            )
        else:
            plan = _build_incremental_plan(context, base_index)
            skip_result = _maybe_skip_incremental_plan(plan)
            if skip_result is not None:
                result = skip_result
            else:
                shard_result, fallback = _index_shards_or_full_rebuild(plan)
                if fallback is not None:
                    result = fallback
                elif shard_result is None:
                    result = ScipIncrementalResult(
                        success=False,
                        index_path=None,
                        manifest_path=None,
                        full_rebuild=False,
                        updated=False,
                        error="SCIP shard indexing missing after fallback",
                    )
                else:
                    _record_shard_metrics(telemetry, shard_result)
                    log.info(
                        "SCIP shard indexing complete (batches=%d, tool_ms=%.1f, parse_ms=%.1f)",
                        shard_result.batch_count,
                        shard_result.tool_ms,
                        shard_result.parse_ms,
                    )
                    result = _apply_incremental_merge(
                        _IncrementalMergeInputs(
                            config=context.config,
                            base_index=plan.base_index,
                            shard_indexes=shard_result.shard_indexes,
                            deleted_paths=plan.deleted_paths,
                            manifest=plan.manifest,
                            shard_updates=shard_result.shard_updates,
                            manifest_file=context.manifest_file,
                        ),
                        telemetry=telemetry,
                    )
    return _finalize_result(result, telemetry=telemetry, start_total=start_total)


def _build_run_context(config: ScipIncrementalConfig) -> _IncrementalRunContext:
    scip_dir = config.output_scip.parent
    scip_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_path(scip_dir)
    target_base = resolve_target_base(config.repo_root, config.target_dir)
    run_config = _ScipRunConfig(
        repo_root=config.repo_root,
        tools_config=config.tools_config,
        tool_runner=config.tool_runner,
        target_base=target_base,
        timeout_seconds=config.timeout_seconds,
        environment_json=config.environment_json,
    )
    total_modules = len(config.modules)
    changed_modules = tuple(config.change_set.added) + tuple(config.change_set.modified)
    deleted_modules = tuple(config.change_set.deleted)
    changed_count = len(changed_modules) + len(deleted_modules)
    changed_ratio = (changed_count / total_modules) if total_modules else None
    return _IncrementalRunContext(
        config=config,
        run_config=run_config,
        scip_dir=scip_dir,
        manifest_file=manifest_file,
        target_base=target_base,
        total_modules=total_modules,
        changed_modules=changed_modules,
        deleted_modules=deleted_modules,
        changed_count=changed_count,
        changed_ratio=changed_ratio,
    )


def _initialize_telemetry(
    context: _IncrementalRunContext,
    decision: ScipIncrementalDecision,
) -> None:
    telemetry = context.config.telemetry
    if telemetry is None:
        return
    telemetry.total_modules = context.total_modules
    telemetry.changed_modules = len(context.changed_modules)
    telemetry.deleted_modules = len(context.deleted_modules)
    telemetry.changed_ratio = context.changed_ratio
    telemetry.batch_size = context.config.batch_size
    telemetry.decision = decision.reason
    telemetry.ratio_gate_applied = decision.ratio_gate_applied
    telemetry.ratio_gate_min_modules = decision.ratio_gate_min_modules
    telemetry.ratio_gate_min_changed = decision.ratio_gate_min_changed
    telemetry.output_scip = str(context.config.output_scip)


def _finalize_result(
    result: ScipIncrementalResult,
    *,
    telemetry: ScipRunTelemetry | None,
    start_total: float,
) -> ScipIncrementalResult:
    if telemetry is not None:
        telemetry.total_ms = _elapsed_ms(start_total)
    return result


def _load_base_index_or_full_rebuild(
    context: _IncrementalRunContext,
) -> tuple[IndexProto | None, ScipIncrementalResult | None]:
    try:
        base_index = load_index_proto(
            context.config.output_scip,
            proto_module_path=context.config.proto_module_path,
        )
    except (_DECODE_ERROR, OSError, AttributeError, TypeError, ValueError) as exc:
        log.warning("SCIP index parse failed; falling back to full rebuild: %s", exc)
        result = _full_rebuild(
            run_config=context.run_config,
            output_scip=context.config.output_scip,
            scope_paths=context.config.scope_paths,
            manifest_file=context.manifest_file,
            telemetry=context.config.telemetry,
            error=str(exc),
        )
        if context.config.telemetry is not None:
            context.config.telemetry.decision = "parse_failed_full_rebuild"
        return None, result
    return base_index, None


def _build_incremental_plan(
    context: _IncrementalRunContext,
    base_index: IndexProto,
) -> _IncrementalPlan:
    manifest = load_manifest(context.manifest_file)
    plan_context = _ShardPlanContext(
        repo_root=context.config.repo_root,
        target_base=context.target_base,
        scope_paths=context.config.scope_paths,
        max_file_size_kb=context.config.max_file_size_kb,
        scip_dir=context.scip_dir,
    )
    resolver = FileDigestResolver(
        file_state_by_path=context.config.file_state_by_path,
        module_state_by_path=context.config.module_state_by_path,
    )
    plan_started = time.perf_counter()
    changed_plans, plan_stats = _build_shard_plans(
        plan_context,
        modules=context.changed_modules,
        resolver=resolver,
    )
    deleted_paths = _filter_deleted_paths(plan_context, modules=context.deleted_modules)
    plan_ms = _elapsed_ms(plan_started)
    hash_source = plan_stats.hash_source
    _record_plan_metrics(
        context.config.telemetry,
        plan_ms=plan_ms,
        plan_stats=plan_stats,
        hash_source=hash_source,
    )
    log.info(
        "SCIP plan complete (plan_ms=%.1f, hash_ms=%.1f, hash_source=%s, candidates=%d)",
        plan_ms,
        plan_stats.computed_ms,
        hash_source or "none",
        plan_stats.total_candidates,
    )
    return _IncrementalPlan(
        context=context,
        base_index=base_index,
        manifest=manifest,
        changed_plans=changed_plans,
        deleted_paths=deleted_paths,
        plan_stats=plan_stats,
        plan_ms=plan_ms,
        hash_source=hash_source,
    )


def _record_plan_metrics(
    telemetry: ScipRunTelemetry | None,
    *,
    plan_ms: float,
    plan_stats: _ShardPlanStats,
    hash_source: str | None,
) -> None:
    if telemetry is None:
        return
    telemetry.plan_ms = plan_ms
    telemetry.hash_ms = plan_stats.computed_ms
    telemetry.hash_reused = plan_stats.hash_reused
    telemetry.hash_computed = plan_stats.hash_computed
    telemetry.hash_source = hash_source
    telemetry.hash_source_breakdown = plan_stats.hash_source_breakdown


def _maybe_skip_incremental_plan(plan: _IncrementalPlan) -> ScipIncrementalResult | None:
    if plan.changed_plans or plan.deleted_paths:
        return None
    log.info("SCIP incremental skipped (no changes detected)")
    telemetry = plan.context.config.telemetry
    if telemetry is not None:
        telemetry.mode = "incremental"
        telemetry.status = "skipped"
    return ScipIncrementalResult(
        success=True,
        index_path=plan.context.config.output_scip,
        manifest_path=(
            plan.context.manifest_file if plan.context.manifest_file.is_file() else None
        ),
        full_rebuild=False,
        updated=False,
    )


def _index_shards_or_full_rebuild(
    plan: _IncrementalPlan,
) -> tuple[_ShardIndexResult | None, ScipIncrementalResult | None]:
    request = _ShardIndexRequest(
        run_config=plan.context.run_config,
        proto_module_path=plan.context.config.proto_module_path,
        plans=plan.changed_plans,
        options_hash=plan.context.config.options_hash,
        batch_size=plan.context.config.batch_size,
        batch_max_bytes=plan.context.config.batch_max_bytes,
        telemetry=plan.context.config.telemetry,
    )
    try:
        shard_result = _index_changed_modules(request)
    except (
        _DECODE_ERROR,
        OSError,
        AttributeError,
        TypeError,
        ValueError,
        ToolExecutionError,
        ToolNotFoundError,
        RuntimeError,
    ) as exc:
        log.exception("Incremental SCIP indexing failed for change set")
        result = _full_rebuild(
            run_config=plan.context.run_config,
            output_scip=plan.context.config.output_scip,
            scope_paths=plan.context.config.scope_paths,
            manifest_file=plan.context.manifest_file,
            telemetry=plan.context.config.telemetry,
            error=str(exc),
        )
        if plan.context.config.telemetry is not None:
            plan.context.config.telemetry.decision = "incremental_failed_full_rebuild"
        return None, result
    return shard_result, None


def _record_shard_metrics(
    telemetry: ScipRunTelemetry | None,
    shard_result: _ShardIndexResult,
) -> None:
    if telemetry is None:
        return
    telemetry.mode = "incremental"
    telemetry.tool_ms = shard_result.tool_ms
    telemetry.parse_ms = shard_result.parse_ms
    telemetry.batch_count = shard_result.batch_count


def _full_rebuild(
    *,
    run_config: _ScipRunConfig,
    output_scip: Path,
    scope_paths: Sequence[str] | None,
    manifest_file: Path,
    telemetry: ScipRunTelemetry | None,
    error: str | None = None,
) -> ScipIncrementalResult:
    if telemetry is not None:
        telemetry.mode = "full"
        telemetry.tool_version = _resolve_scip_python_version(run_config)
    try:
        tool_start = time.perf_counter()
        _run_scip_python(
            run_config=run_config,
            output_scip=output_scip,
            rel_paths=None,
            scope_paths=scope_paths,
            log_prefix="scip-python full",
        )
        tool_ms = _elapsed_ms(tool_start)
        if telemetry is not None:
            telemetry.tool_ms = tool_ms
    except (
        ToolExecutionError,
        ToolNotFoundError,
        RuntimeError,
        OSError,
        ValueError,
    ) as exc:
        message = error or str(exc)
        if telemetry is not None:
            telemetry.status = "failed"
            telemetry.error_summary = message
        return ScipIncrementalResult(
            success=False,
            index_path=None,
            manifest_path=None,
            full_rebuild=True,
            updated=False,
            error=message,
        )

    write_start = time.perf_counter()
    write_manifest(manifest_file, ScipShardManifest.empty())
    write_ms = _elapsed_ms(write_start)
    if telemetry is not None:
        telemetry.write_ms = write_ms
        telemetry.status = "succeeded"
    log.info("SCIP full rebuild complete (tool_ms=%.1f, write_ms=%.1f)", tool_ms, write_ms)
    return ScipIncrementalResult(
        success=True,
        index_path=output_scip,
        manifest_path=manifest_file,
        full_rebuild=True,
        updated=True,
    )


def _apply_incremental_merge(
    inputs: _IncrementalMergeInputs,
    *,
    telemetry: ScipRunTelemetry | None,
) -> ScipIncrementalResult:
    base_updated_at = {path: record.updated_at for path, record in inputs.manifest.records.items()}
    shard_updated_at = {path: record.updated_at for path, record in inputs.shard_updates.items()}
    merge_context = MergeIndexContext(
        shard_updated_at=shard_updated_at,
        base_updated_at=base_updated_at,
    )
    merge_start = time.perf_counter()
    merged = merge_indexes(
        base_index=inputs.base_index,
        shard_indexes=inputs.shard_indexes,
        deleted_paths=inputs.deleted_paths,
        proto_module_path=inputs.config.proto_module_path,
        context=merge_context,
    )
    write_index_proto(merged, inputs.config.output_scip)
    merge_ms = _elapsed_ms(merge_start)
    if telemetry is not None:
        telemetry.merge_ms = merge_ms

    write_start = time.perf_counter()
    updated_manifest = update_manifest(
        inputs.manifest,
        updates=inputs.shard_updates,
        deleted=dict.fromkeys(inputs.deleted_paths, True),
    )
    write_manifest(inputs.manifest_file, updated_manifest)
    write_ms = _elapsed_ms(write_start)
    if telemetry is not None:
        telemetry.write_ms = write_ms
        telemetry.status = "succeeded"
    log.info("SCIP merge complete (merge_ms=%.1f, write_ms=%.1f)", merge_ms, write_ms)

    return ScipIncrementalResult(
        success=True,
        index_path=inputs.config.output_scip,
        manifest_path=inputs.manifest_file,
        full_rebuild=False,
        updated=True,
    )


def _options_mismatch(manifest: ScipShardManifest, options_hash: str | None) -> bool:
    if options_hash is None:
        return False
    if not manifest.records:
        return False
    return any(record.options_hash != options_hash for record in manifest.records.values())


def _build_policy(config: ScipIncrementalConfig) -> ScipIncrementalPolicy:
    return ScipIncrementalPolicy(
        full_rebuild_threshold_count=config.full_rebuild_threshold_count,
        full_rebuild_threshold_ratio=config.full_rebuild_threshold_ratio,
        ratio_gate_min_modules=config.full_rebuild_ratio_min_modules,
        ratio_gate_min_changed=config.full_rebuild_ratio_min_changed,
    )


def _resolve_decision(context: _IncrementalRunContext) -> ScipIncrementalDecision:
    policy = _build_policy(context.config)
    options_mismatch = _options_mismatch_for_decision(
        manifest_file=context.manifest_file,
        options_hash=context.config.options_hash,
    )
    inputs = ScipIncrementalInputs(
        total_modules=context.total_modules,
        changed_count=context.changed_count,
        changed_ratio=context.changed_ratio,
        output_exists=context.config.output_scip.is_file(),
        options_mismatch=options_mismatch,
        force_full_rebuild=context.config.force_full_rebuild,
    )
    decision = policy.decide(inputs)
    if decision.mode == "full":
        log.info("SCIP decision %s; forcing full rebuild", decision.reason)
    return decision


def _options_mismatch_for_decision(
    *,
    manifest_file: Path,
    options_hash: str | None,
) -> bool:
    if options_hash is None or not manifest_file.is_file():
        return False
    manifest = load_manifest(manifest_file)
    if _options_mismatch(manifest, options_hash):
        log.info("SCIP options changed; forcing full rebuild")
        return True
    return False


def _elapsed_ms(start_ts: float) -> float:
    return (time.perf_counter() - start_ts) * 1000


def _log_plan_summary(
    context: _IncrementalRunContext,
    decision: ScipIncrementalDecision,
) -> None:
    ratio_label = f"{context.changed_ratio:.2f}" if context.changed_ratio is not None else "n/a"
    gate_label = "on" if decision.ratio_gate_applied else "off"
    log.info(
        "SCIP incremental plan: total=%d changed=%d deleted=%d ratio=%s "
        "decision=%s batch_size=%d batch_max_bytes=%d "
        "threshold_count=%d threshold_ratio=%.2f ratio_gate=%s",
        context.total_modules,
        len(context.changed_modules),
        len(context.deleted_modules),
        ratio_label,
        decision.reason,
        context.config.batch_size,
        context.config.batch_max_bytes,
        context.config.full_rebuild_threshold_count,
        context.config.full_rebuild_threshold_ratio,
        gate_label,
    )


def _scip_trace_enabled() -> bool:
    value = os.environ.get(_SCIP_TRACE_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_shard_plans(
    context: _ShardPlanContext,
    *,
    modules: Sequence[ModuleRecord],
    resolver: FileDigestResolver,
) -> tuple[tuple[_ModuleShardPlan, ...], _ShardPlanStats]:
    plans: list[_ModuleShardPlan] = []
    computed_start = time.perf_counter()
    for module in modules:
        if not _in_scope(module.rel_path, context.scope_paths):
            continue
        scip_rel = scip_relative_path(
            repo_root=context.repo_root,
            target_base=context.target_base,
            rel_path=module.rel_path,
        )
        if scip_rel is None:
            continue
        digest = resolver.resolve(module)
        if digest is None:
            continue
        if context.max_file_size_kb > 0 and digest.size_bytes > context.max_file_size_kb * 1024:
            continue
        shard_file = shard_path(
            context.scip_dir,
            rel_path=scip_rel,
            content_hash=digest.content_hash,
        )
        shard_file.parent.mkdir(parents=True, exist_ok=True)
        plans.append(
            _ModuleShardPlan(
                rel_path=module.rel_path,
                scip_rel_path=scip_rel,
                content_hash=digest.content_hash,
                size_bytes=digest.size_bytes,
                shard_path=shard_file,
            )
        )
    computed_ms = (time.perf_counter() - computed_start) * 1000
    summary = resolver.summary()
    return (
        tuple(plans),
        _ShardPlanStats(
            total_candidates=len(modules),
            hash_computed=summary.hash_computed,
            hash_reused=summary.hash_reused,
            hash_source=summary.hash_source,
            hash_source_breakdown=summary.breakdown,
            computed_ms=computed_ms,
        ),
    )


def _filter_deleted_paths(
    context: _ShardPlanContext,
    *,
    modules: Sequence[ModuleRecord],
) -> tuple[str, ...]:
    deleted: list[str] = []
    for module in modules:
        if not _in_scope(module.rel_path, context.scope_paths):
            continue
        scip_rel = scip_relative_path(
            repo_root=context.repo_root,
            target_base=context.target_base,
            rel_path=module.rel_path,
        )
        if scip_rel is None:
            continue
        deleted.append(scip_rel)
    return tuple(sorted(set(deleted)))


def _build_batches(
    plans: Sequence[_ModuleShardPlan],
    *,
    batch_size: int,
    max_batch_bytes: int,
) -> tuple[_BatchPlan, ...]:
    sorted_plans = sorted(plans, key=lambda plan: plan.scip_rel_path)
    batches: list[_BatchPlan] = []
    current: list[_ModuleShardPlan] = []
    current_bytes = 0

    def _flush_batch() -> None:
        if not current:
            return
        rel_paths = tuple(plan.scip_rel_path for plan in current)
        batch_id = _hash_batch(current)
        batches.append(
            _BatchPlan(
                batch_id=batch_id,
                plans=tuple(current),
                size_bytes=current_bytes,
                rel_paths=rel_paths,
            )
        )

    for plan in sorted_plans:
        plan_bytes = plan.size_bytes
        exceeds_size = max_batch_bytes > 0 and current_bytes + plan_bytes > max_batch_bytes
        exceeds_count = batch_size > 0 and len(current) >= batch_size
        if current and (exceeds_size or exceeds_count):
            _flush_batch()
            current = []
            current_bytes = 0
        current.append(plan)
        current_bytes += plan_bytes

    _flush_batch()

    return tuple(batches)


def _batch_shard_path(batch: _BatchPlan) -> Path:
    if not batch.plans:
        msg = "batch_shard_path requires at least one plan"
        raise ValueError(msg)
    shard_dir = batch.plans[0].shard_path.parent
    return shard_dir / f"batch_{batch.batch_id}.scip"


def _hash_batch(batch: Sequence[_ModuleShardPlan]) -> str:
    hasher = hashlib.sha256()
    for plan in batch:
        hasher.update(plan.scip_rel_path.encode("utf-8"))
        hasher.update(b"|")
        hasher.update(plan.content_hash.encode("utf-8"))
        hasher.update(b"|")
    return hasher.hexdigest()[:16]


def _index_changed_modules(request: _ShardIndexRequest) -> _ShardIndexResult:
    shard_indexes: list[IndexProto] = []
    shard_updates: dict[str, ScipShardRecord] = {}
    tool_version = _resolve_scip_python_version(request.run_config)
    if request.telemetry is not None:
        request.telemetry.tool_version = tool_version
    tool_ms = 0.0
    parse_ms = 0.0
    batches = _build_batches(
        request.plans,
        batch_size=max(1, request.batch_size),
        max_batch_bytes=max(0, request.batch_max_bytes),
    )
    batch_bytes = sum(batch.size_bytes for batch in batches)
    batch_ids = ",".join(batch.batch_id for batch in batches)
    log.info(
        "SCIP batch plan (batches=%d, batch_size=%d, max_bytes=%d, total_bytes=%d, ids=%s)",
        len(batches),
        request.batch_size,
        request.batch_max_bytes,
        batch_bytes,
        batch_ids or "none",
    )
    for idx, batch in enumerate(batches, start=1):
        batch_path = _batch_shard_path(batch)
        rel_paths = list(batch.rel_paths)
        tool_start = time.perf_counter()
        _run_scip_python(
            run_config=request.run_config,
            output_scip=batch_path,
            rel_paths=rel_paths,
            scope_paths=None,
            log_prefix=f"scip-python batch {idx}/{len(batches)}",
        )
        tool_ms += _elapsed_ms(tool_start)
        parse_start = time.perf_counter()
        shard_index = load_index_proto(
            batch_path,
            proto_module_path=request.proto_module_path,
        )
        parse_ms += _elapsed_ms(parse_start)
        shard_indexes.append(shard_index)
        updated_at = datetime.now(tz=UTC)
        for plan in batch.plans:
            shard_updates[plan.scip_rel_path] = ScipShardRecord(
                rel_path=plan.scip_rel_path,
                content_hash=plan.content_hash,
                options_hash=request.options_hash,
                tool_version=tool_version,
                shard_path=str(batch_path),
                updated_at=updated_at,
            )

    return _ShardIndexResult(
        shard_indexes=tuple(shard_indexes),
        shard_updates=shard_updates,
        tool_ms=tool_ms,
        parse_ms=parse_ms,
        batch_count=len(batches),
    )


def _in_scope(rel_path: str, scope_paths: Sequence[str] | None) -> bool:
    if not scope_paths:
        return True
    normalized = rel_path.replace("\\", "/")
    return any(normalized.startswith(scope.rstrip("/")) for scope in scope_paths)


def _run_scip_python(
    *,
    run_config: _ScipRunConfig,
    output_scip: Path,
    rel_paths: Sequence[str] | None,
    scope_paths: Sequence[str] | None,
    log_prefix: str | None = None,
) -> None:
    trace_enabled = _scip_trace_enabled()
    progress_interval = (
        _SCIP_TRACE_PROGRESS_INTERVAL_S if trace_enabled else _SCIP_PROGRESS_INTERVAL_S
    )
    if run_config.environment_json is None:
        ensure_pip_available()
    args = build_scip_python_args(
        target_base=run_config.target_base,
        output_scip=output_scip,
        project_name=run_config.tools_config.scip_project_name,
        rel_paths=rel_paths,
        scope_paths=scope_paths,
        environment_json=run_config.environment_json,
    )
    result = asyncio.run(
        run_config.tool_runner.run_async(
            ToolName.SCIP_PYTHON,
            args,
            options=ToolRunOptions(
                cwd=run_config.repo_root,
                output_path=output_scip,
                timeout_s=float(run_config.timeout_seconds),
                progress_interval_s=progress_interval,
                log_prefix=log_prefix,
                stream_output=trace_enabled,
            ),
        )
    )
    if not result.ok:
        raise ToolExecutionError(result)


def _resolve_scip_python_version(run_config: _ScipRunConfig) -> str | None:
    try:
        result = asyncio.run(
            run_config.tool_runner.run_async(
                ToolName.SCIP_PYTHON,
                ["--version"],
                options=ToolRunOptions(
                    timeout_s=run_config.tools_config.default_timeout_s,
                ),
            )
        )
    except (ToolExecutionError, ToolNotFoundError, RuntimeError, OSError, ValueError):
        return None
    if not result.ok:
        return None
    stdout = result.stdout.strip()
    return stdout.splitlines()[0] if stdout else None


__all__ = [
    "ScipIncrementalConfig",
    "ScipIncrementalResult",
    "update_index_incremental",
]
