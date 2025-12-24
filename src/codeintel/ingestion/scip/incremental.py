"""Incremental SCIP indexing orchestration."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.ingestion.adapters.hash_change_detection import HashChangeDetectionAdapter
from codeintel.ingestion.engine.infrastructure import ToolName, ToolRunOptions
from codeintel.ingestion.scip.cli import build_scip_python_args
from codeintel.ingestion.scip.index_store import (
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
    from collections.abc import Sequence

    from codeintel.config.models import ToolsConfig
    from codeintel.ingestion.engine.infrastructure import ToolRunner
    from codeintel.ingestion.ports.change_detection import ChangeSet
    from codeintel.ingestion.ports.discovery import ModuleRecord

log = logging.getLogger(__name__)


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
class _ModuleShardPlan:
    rel_path: str
    scip_rel_path: str
    content_hash: str
    shard_path: Path


def update_index_incremental(
    *,
    repo_root: Path,
    output_scip: Path,
    proto_module_path: Path,
    change_set: ChangeSet,
    options_hash: str | None,
    tools_config: ToolsConfig,
    tool_runner: ToolRunner,
    scope_paths: Sequence[str] | None,
    max_file_size_kb: int,
    timeout_seconds: int,
    target_dir: Path | None,
    force_full_rebuild: bool = False,
) -> ScipIncrementalResult:
    """Update index.scip using incremental per-module indexing.

    Returns
    -------
    ScipIncrementalResult
        Result describing the update outcome and paths.
    """
    scip_dir = output_scip.parent
    scip_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_path(scip_dir)
    target_base = resolve_target_base(repo_root, target_dir)

    if force_full_rebuild:
        return _full_rebuild(
            repo_root=repo_root,
            output_scip=output_scip,
            tools_config=tools_config,
            tool_runner=tool_runner,
            target_base=target_base,
            timeout_seconds=timeout_seconds,
            manifest_file=manifest_file,
        )

    if not output_scip.is_file():
        return _full_rebuild(
            repo_root=repo_root,
            output_scip=output_scip,
            tools_config=tools_config,
            tool_runner=tool_runner,
            target_base=target_base,
            timeout_seconds=timeout_seconds,
            manifest_file=manifest_file,
        )

    try:
        base_index = load_index_proto(output_scip, proto_module_path=proto_module_path)
    except Exception as exc:
        log.warning("SCIP index parse failed; falling back to full rebuild: %s", exc)
        return _full_rebuild(
            repo_root=repo_root,
            output_scip=output_scip,
            tools_config=tools_config,
            tool_runner=tool_runner,
            target_base=target_base,
            timeout_seconds=timeout_seconds,
            manifest_file=manifest_file,
        )

    changed_modules = tuple(change_set.added) + tuple(change_set.modified)
    deleted_modules = tuple(change_set.deleted)

    changed_plans = _build_shard_plans(
        repo_root=repo_root,
        target_base=target_base,
        modules=changed_modules,
        scope_paths=scope_paths,
        max_file_size_kb=max_file_size_kb,
        scip_dir=scip_dir,
    )
    deleted_paths = _filter_deleted_paths(
        repo_root=repo_root,
        target_base=target_base,
        modules=deleted_modules,
        scope_paths=scope_paths,
    )

    if not changed_plans and not deleted_paths:
        return ScipIncrementalResult(
            success=True,
            index_path=output_scip,
            manifest_path=manifest_file if manifest_file.is_file() else None,
            full_rebuild=False,
            updated=False,
        )

    shard_indexes: list[object] = []
    shard_updates: dict[str, ScipShardRecord] = {}
    tool_version = _resolve_scip_python_version(tool_runner, tools_config)
    for plan in changed_plans:
        try:
            _run_scip_python(
                repo_root=repo_root,
                tools_config=tools_config,
                tool_runner=tool_runner,
                target_base=target_base,
                output_scip=plan.shard_path,
                rel_paths=(plan.scip_rel_path,),
                timeout_seconds=timeout_seconds,
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
        except Exception as exc:
            log.exception("Incremental SCIP indexing failed for %s", plan.rel_path)
            return _full_rebuild(
                repo_root=repo_root,
                output_scip=output_scip,
                tools_config=tools_config,
                tool_runner=tool_runner,
                target_base=target_base,
                timeout_seconds=timeout_seconds,
                manifest_file=manifest_file,
                error=str(exc),
            )

    merged = merge_indexes(
        base_index=base_index,
        shard_indexes=tuple(shard_indexes),
        deleted_paths=deleted_paths,
        proto_module_path=proto_module_path,
    )
    write_index_proto(merged, output_scip)

    manifest = load_manifest(manifest_file)
    updated_manifest = update_manifest(
        manifest,
        updates=shard_updates,
        deleted=dict.fromkeys(deleted_paths, True),
    )
    write_manifest(manifest_file, updated_manifest)

    return ScipIncrementalResult(
        success=True,
        index_path=output_scip,
        manifest_path=manifest_file,
        full_rebuild=False,
        updated=True,
    )


def _full_rebuild(
    *,
    repo_root: Path,
    output_scip: Path,
    tools_config: ToolsConfig,
    tool_runner: ToolRunner,
    target_base: Path,
    timeout_seconds: int,
    manifest_file: Path,
    error: str | None = None,
) -> ScipIncrementalResult:
    try:
        _run_scip_python(
            repo_root=repo_root,
            tools_config=tools_config,
            tool_runner=tool_runner,
            target_base=target_base,
            output_scip=output_scip,
            rel_paths=None,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
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


def _build_shard_plans(
    *,
    repo_root: Path,
    target_base: Path,
    modules: Sequence[ModuleRecord],
    scope_paths: Sequence[str] | None,
    max_file_size_kb: int,
    scip_dir: Path,
) -> tuple[_ModuleShardPlan, ...]:
    plans: list[_ModuleShardPlan] = []
    for module in modules:
        if not _in_scope(module.rel_path, scope_paths):
            continue
        scip_rel = scip_relative_path(
            repo_root=repo_root,
            target_base=target_base,
            rel_path=module.rel_path,
        )
        if scip_rel is None:
            continue
        digest = HashChangeDetectionAdapter.compute_file_digest(module.file_path)
        if digest is None:
            continue
        if max_file_size_kb > 0 and digest.size_bytes > max_file_size_kb * 1024:
            continue
        shard_file = shard_path(scip_dir, rel_path=scip_rel, content_hash=digest.content_hash)
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
    *,
    repo_root: Path,
    target_base: Path,
    modules: Sequence[ModuleRecord],
    scope_paths: Sequence[str] | None,
) -> tuple[str, ...]:
    deleted: list[str] = []
    for module in modules:
        if not _in_scope(module.rel_path, scope_paths):
            continue
        scip_rel = scip_relative_path(
            repo_root=repo_root,
            target_base=target_base,
            rel_path=module.rel_path,
        )
        if scip_rel is None:
            continue
        deleted.append(scip_rel)
    return tuple(sorted(set(deleted)))


def _in_scope(rel_path: str, scope_paths: Sequence[str] | None) -> bool:
    if not scope_paths:
        return True
    normalized = rel_path.replace("\\", "/")
    return any(normalized.startswith(scope.rstrip("/")) for scope in scope_paths)


def _run_scip_python(
    *,
    repo_root: Path,
    tools_config: ToolsConfig,
    tool_runner: ToolRunner,
    target_base: Path,
    output_scip: Path,
    rel_paths: Sequence[str] | None,
    timeout_seconds: int,
) -> None:
    args = build_scip_python_args(
        target_base=target_base,
        output_scip=output_scip,
        project_name=tools_config.scip_project_name,
        rel_paths=rel_paths,
    )
    result = asyncio.run(
        tool_runner.run_async(
            ToolName.SCIP_PYTHON,
            args,
            options=ToolRunOptions(
                cwd=repo_root,
                output_path=output_scip,
                timeout_s=float(timeout_seconds),
            ),
        )
    )
    if not result.ok:
        raise RuntimeError(result.stderr.strip() or "SCIP indexing failed")


def _resolve_scip_python_version(tool_runner: ToolRunner, tools_config: ToolsConfig) -> str | None:
    try:
        result = asyncio.run(
            tool_runner.run_async(
                ToolName.SCIP_PYTHON,
                ["--version"],
                options=ToolRunOptions(timeout_s=tools_config.default_timeout_s),
            )
        )
    except Exception:
        return None
    if not result.ok:
        return None
    stdout = result.stdout.strip()
    return stdout.splitlines()[0] if stdout else None


__all__ = ["ScipIncrementalResult", "update_index_incremental"]
