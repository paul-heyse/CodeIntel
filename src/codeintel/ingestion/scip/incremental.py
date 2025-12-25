"""Incremental SCIP indexing orchestration."""

from __future__ import annotations

import asyncio
import importlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.ingestion.adapters.hash_change_detection import HashChangeDetectionAdapter
from codeintel.ingestion.engine.infrastructure import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunOptions,
)
from codeintel.ingestion.scip.cli import build_scip_python_args
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

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.config.models import ToolsConfig
    from codeintel.ingestion.engine.infrastructure import ToolRunner
    from codeintel.ingestion.ports.change_detection import ChangeSet
    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.ingestion.scip.proto_types import IndexProto

log = logging.getLogger(__name__)


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
    max_file_size_kb: int
    timeout_seconds: int
    target_dir: Path | None
    force_full_rebuild: bool = False


@dataclass(frozen=True)
class _ScipRunConfig:
    repo_root: Path
    tools_config: ToolsConfig
    tool_runner: ToolRunner
    target_base: Path
    timeout_seconds: int


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
    shard_path: Path


@dataclass(frozen=True)
class _IncrementalMergeInputs:
    config: ScipIncrementalConfig
    base_index: IndexProto
    shard_indexes: Sequence[IndexProto]
    deleted_paths: tuple[str, ...]
    manifest: ScipShardManifest
    shard_updates: Mapping[str, ScipShardRecord]
    manifest_file: Path


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
    )

    if config.force_full_rebuild or not config.output_scip.is_file():
        return _full_rebuild(
            run_config=run_config,
            output_scip=config.output_scip,
            manifest_file=manifest_file,
        )

    try:
        base_index = load_index_proto(
            config.output_scip,
            proto_module_path=config.proto_module_path,
        )
    except (_DECODE_ERROR, OSError, AttributeError, TypeError, ValueError) as exc:
        log.warning("SCIP index parse failed; falling back to full rebuild: %s", exc)
        return _full_rebuild(
            run_config=run_config,
            output_scip=config.output_scip,
            manifest_file=manifest_file,
        )

    manifest = load_manifest(manifest_file)
    changed_modules = tuple(config.change_set.added) + tuple(config.change_set.modified)
    deleted_modules = tuple(config.change_set.deleted)
    if _options_mismatch(manifest, config.options_hash):
        log.info("SCIP options changed; reindexing all modules")
        changed_modules = tuple(config.modules)
    plan_context = _ShardPlanContext(
        repo_root=config.repo_root,
        target_base=target_base,
        scope_paths=config.scope_paths,
        max_file_size_kb=config.max_file_size_kb,
        scip_dir=scip_dir,
    )
    changed_plans = _build_shard_plans(plan_context, modules=changed_modules)
    deleted_paths = _filter_deleted_paths(plan_context, modules=deleted_modules)

    if not changed_plans and not deleted_paths:
        return ScipIncrementalResult(
            success=True,
            index_path=config.output_scip,
            manifest_path=manifest_file if manifest_file.is_file() else None,
            full_rebuild=False,
            updated=False,
        )

    try:
        shard_indexes, shard_updates = _index_changed_modules(
            run_config=run_config,
            proto_module_path=config.proto_module_path,
            plans=changed_plans,
            options_hash=config.options_hash,
        )
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
        return _full_rebuild(
            run_config=run_config,
            output_scip=config.output_scip,
            manifest_file=manifest_file,
            error=str(exc),
        )

    return _apply_incremental_merge(
        _IncrementalMergeInputs(
            config=config,
            base_index=base_index,
            shard_indexes=shard_indexes,
            deleted_paths=deleted_paths,
            manifest=manifest,
            shard_updates=shard_updates,
            manifest_file=manifest_file,
        )
    )


def _full_rebuild(
    *,
    run_config: _ScipRunConfig,
    output_scip: Path,
    manifest_file: Path,
    error: str | None = None,
) -> ScipIncrementalResult:
    try:
        _run_scip_python(
            run_config=run_config,
            output_scip=output_scip,
            rel_paths=None,
        )
    except (
        ToolExecutionError,
        ToolNotFoundError,
        RuntimeError,
        OSError,
        ValueError,
    ) as exc:
        message = error or str(exc)
        return ScipIncrementalResult(
            success=False,
            index_path=None,
            manifest_path=None,
            full_rebuild=True,
            updated=False,
            error=message,
        )

    write_manifest(manifest_file, ScipShardManifest.empty())
    return ScipIncrementalResult(
        success=True,
        index_path=output_scip,
        manifest_path=manifest_file,
        full_rebuild=True,
        updated=True,
    )


def _apply_incremental_merge(inputs: _IncrementalMergeInputs) -> ScipIncrementalResult:
    base_updated_at = {
        path: record.updated_at for path, record in inputs.manifest.records.items()
    }
    shard_updated_at = {
        path: record.updated_at for path, record in inputs.shard_updates.items()
    }
    merge_context = MergeIndexContext(
        shard_updated_at=shard_updated_at,
        base_updated_at=base_updated_at,
    )
    merged = merge_indexes(
        base_index=inputs.base_index,
        shard_indexes=inputs.shard_indexes,
        deleted_paths=inputs.deleted_paths,
        proto_module_path=inputs.config.proto_module_path,
        context=merge_context,
    )
    write_index_proto(merged, inputs.config.output_scip)

    updated_manifest = update_manifest(
        inputs.manifest,
        updates=inputs.shard_updates,
        deleted=dict.fromkeys(inputs.deleted_paths, True),
    )
    write_manifest(inputs.manifest_file, updated_manifest)

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


def _build_shard_plans(
    context: _ShardPlanContext,
    *,
    modules: Sequence[ModuleRecord],
) -> tuple[_ModuleShardPlan, ...]:
    plans: list[_ModuleShardPlan] = []
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
        digest = HashChangeDetectionAdapter.compute_file_digest(module.file_path)
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
                shard_path=shard_file,
            )
        )
    return tuple(plans)


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


def _index_changed_modules(
    *,
    run_config: _ScipRunConfig,
    proto_module_path: Path,
    plans: Sequence[_ModuleShardPlan],
    options_hash: str | None,
) -> tuple[tuple[IndexProto, ...], dict[str, ScipShardRecord]]:
    shard_indexes: list[IndexProto] = []
    shard_updates: dict[str, ScipShardRecord] = {}
    tool_version = _resolve_scip_python_version(run_config)

    for plan in plans:
        _run_scip_python(
            run_config=run_config,
            output_scip=plan.shard_path,
            rel_paths=(plan.scip_rel_path,),
        )
        shard_index = load_index_proto(plan.shard_path, proto_module_path=proto_module_path)
        shard_indexes.append(shard_index)
        shard_updates[plan.scip_rel_path] = ScipShardRecord(
            rel_path=plan.scip_rel_path,
            content_hash=plan.content_hash,
            options_hash=options_hash,
            tool_version=tool_version,
            shard_path=str(plan.shard_path),
            updated_at=datetime.now(tz=UTC),
        )

    return tuple(shard_indexes), shard_updates


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
) -> None:
    args = build_scip_python_args(
        target_base=run_config.target_base,
        output_scip=output_scip,
        project_name=run_config.tools_config.scip_project_name,
        rel_paths=rel_paths,
    )
    result = asyncio.run(
        run_config.tool_runner.run_async(
            ToolName.SCIP_PYTHON,
            args,
            options=ToolRunOptions(
                cwd=run_config.repo_root,
                output_path=output_scip,
                timeout_s=float(run_config.timeout_seconds),
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
