"""Hamilton build diagnostics emission helpers."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from hamilton.caching.adapter import HamiltonCacheAdapter

from codeintel.build.hamilton.decision_trace import (
    build_cache_manifest_entries,
    default_decision_trace_path,
    write_decision_trace,
)
from codeintel.build.hamilton.observability import (
    export_dag_dot,
    export_dag_json,
    export_dag_mermaid,
)
from codeintel.core.hamilton import tags as ht

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.core.hamilton.records import NodeExecutionRecord
    from codeintel.runtime.runtime_bundle import RuntimeBundle

log = logging.getLogger(__name__)


def diagnostics_dir(build_dir: Path) -> Path:
    """Return the diagnostics directory under the build root."""
    return build_dir / "diagnostics"


def emit_diagnostics(
    *,
    env: BuildEnv,
    runtime: RuntimeBundle,
    run_id: str,
    cache_dir: Path,
    cache_adapter: HamiltonCacheAdapter | None,
    telemetry_records: Sequence[NodeExecutionRecord] | None,
    requested_targets: Sequence[str],
    computed_targets: Sequence[str],
    skipped_targets: Sequence[str],
    failed_targets: Sequence[str],
    duration_ms: float,
) -> None:
    """Emit Hamilton-native diagnostics artifacts under build/diagnostics."""
    diag_dir = _ensure_dir(diagnostics_dir(env.paths.build_dir))
    _write_run_summary(
        diag_dir / "run_summary.json",
        run_id=run_id,
        repo=env.repo,
        commit=env.commit,
        profile=env.profile,
        domain=env.execution_context.run.requested_operation
        if env.execution_context is not None
        else None,
        requested=requested_targets,
        computed=computed_targets,
        skipped=skipped_targets,
        failed=failed_targets,
        duration_ms=duration_ms,
    )
    _write_dag_exports(
        diag_dir=diag_dir,
        runtime=runtime,
        targets=list(requested_targets),
    )
    if cache_adapter is None:
        return
    _write_cache_events(
        output_path=diag_dir / "cache_events.jsonl",
        cache_dir=cache_dir,
        cache_adapter=cache_adapter,
        run_id=run_id,
    )
    target_by_node = _target_map(runtime)
    _write_cache_keys_snapshot(
        output_path=diag_dir / "cache_keys.jsonl",
        cache_adapter=cache_adapter,
        run_id=run_id,
        target_by_node=target_by_node,
    )
    durations_ms = _node_duration_map(telemetry_records or ())
    decision_entries = build_cache_manifest_entries(
        cache_adapter=cache_adapter,
        run_id=run_id,
        target_by_node=target_by_node,
        durations_ms=durations_ms,
    )
    decision_path = default_decision_trace_path(env.paths.build_dir)
    try:
        write_decision_trace(decision_path, decision_entries)
    except (OSError, TypeError, ValueError) as exc:
        log.warning("build.diagnostics.decision_trace_failed run_id=%s error=%s", run_id, exc)
    _write_cache_visualization(
        output_path=diag_dir / "cache_run_visualization.svg",
        cache_adapter=cache_adapter,
        run_id=run_id,
    )


def _write_run_summary(
    path: Path,
    *,
    run_id: str,
    repo: str,
    commit: str,
    profile: str | None,
    domain: str | None,
    requested: Sequence[str],
    computed: Sequence[str],
    skipped: Sequence[str],
    failed: Sequence[str],
    duration_ms: float,
) -> None:
    payload: dict[str, object] = {
        "run_id": run_id,
        "repo": repo,
        "commit": commit,
        "profile": profile,
        "domain": domain,
        "requested_targets": list(requested),
        "computed_targets": list(computed),
        "skipped_targets": list(skipped),
        "failed_targets": list(failed),
        "duration_ms": duration_ms,
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }
    _write_json(path, payload)


def _write_dag_exports(
    *,
    diag_dir: Path,
    runtime: RuntimeBundle,
    targets: list[str],
) -> None:
    try:
        _write_text(diag_dir / "dag.dot", export_dag_dot(runtime, targets))
        _write_text(diag_dir / "dag.json", export_dag_json(runtime, targets))
        _write_text(diag_dir / "dag.mermaid", export_dag_mermaid(runtime, targets))
    except (OSError, TypeError, ValueError) as exc:
        log.warning("build.diagnostics.dag_export_failed error=%s", exc)


def _write_cache_events(
    *,
    output_path: Path,
    cache_dir: Path,
    cache_adapter: HamiltonCacheAdapter,
    run_id: str,
) -> None:
    entries = _read_cache_log_entries(cache_dir / "cache_logs.jsonl", run_id)
    if not entries:
        entries = _cache_events_from_adapter(cache_adapter, run_id)
    if not entries:
        return
    _write_jsonl(output_path, entries)


def _read_cache_log_entries(
    path: Path,
    run_id: str,
) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    entries: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("run_id") != run_id:
            continue
        entries.append(payload)
    return entries


def _cache_events_from_adapter(
    cache_adapter: HamiltonCacheAdapter,
    run_id: str,
) -> list[dict[str, object]]:
    logs_by_node = cache_adapter.logs(run_id=run_id, level="info")
    if not isinstance(logs_by_node, dict):
        return []
    rows: list[dict[str, object]] = []
    for key, events in logs_by_node.items():
        node_name, task_id = _cache_log_key_parts(key)
        if not isinstance(events, list):
            continue
        for event in events:
            rows.append(
                {
                    "run_id": run_id,
                    "node_name": node_name,
                    "task_id": task_id,
                    "actor": _normalize_json_value(getattr(event, "actor", None)),
                    "event_type": _normalize_json_value(getattr(event, "event_type", None)),
                    "msg": _normalize_json_value(getattr(event, "msg", None)),
                    "value": _normalize_json_value(getattr(event, "value", None)),
                    "timestamp": getattr(event, "timestamp", None),
                }
            )
    return rows


def _write_cache_keys_snapshot(
    *,
    output_path: Path,
    cache_adapter: HamiltonCacheAdapter,
    run_id: str,
    target_by_node: Mapping[str, str],
) -> None:
    logs_by_node = cache_adapter.logs(run_id=run_id, level="info")
    if not isinstance(logs_by_node, dict):
        return
    rows: list[dict[str, object]] = []
    for key in sorted(logs_by_node, key=_cache_log_key_sort_key):
        node_name, task_id = _cache_log_key_parts(key)
        cache_key = cache_adapter.get_cache_key(
            run_id=run_id,
            node_name=node_name,
            task_id=task_id,
        )
        cache_key_str = cache_key if isinstance(cache_key, str) else None
        data_version = cache_adapter.get_data_version(
            run_id=run_id,
            node_name=node_name,
            cache_key=cache_key_str,
            task_id=task_id,
        )
        rows.append(
            {
                "node": node_name,
                "task_id": task_id,
                "cache_key": cache_key_str,
                "data_version": data_version if isinstance(data_version, str) else None,
                "target": target_by_node.get(node_name),
            }
        )
    if not rows:
        return
    _write_jsonl(output_path, rows)


def _write_cache_visualization(
    *,
    output_path: Path,
    cache_adapter: HamiltonCacheAdapter,
    run_id: str,
) -> None:
    try:
        cache_adapter.view_run(run_id=run_id, output_file_path=str(output_path))
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        log.warning("build.diagnostics.cache_view_failed run_id=%s error=%s", run_id, exc)


def _target_map(runtime: RuntimeBundle) -> dict[str, str]:
    nodes = getattr(runtime.dr, "graph", None)
    if nodes is None:
        return {}
    node_map = getattr(nodes, "nodes", None)
    if not isinstance(node_map, dict):
        return {}
    mapping: dict[str, str] = {}
    for name, node in node_map.items():
        tags = node.tags if isinstance(node.tags, dict) else None
        if not tags:
            continue
        target = tags.get(ht.TAG_TARGET)
        if isinstance(target, str):
            mapping[name] = target
    return mapping


def _node_duration_map(
    records: Iterable[NodeExecutionRecord],
) -> dict[str, float | None]:
    durations: dict[str, float | None] = {}
    for record in records:
        durations[record.node_name] = record.duration_ms
    return durations


def _cache_log_key_parts(key: object) -> tuple[str, str | None]:
    if isinstance(key, str):
        return key, None
    if isinstance(key, tuple) and len(key) == 2 and all(isinstance(item, str) for item in key):
        return key[0], key[1]
    return str(key), None


def _cache_log_key_sort_key(key: object) -> tuple[str, str]:
    node_name, task_id = _cache_log_key_parts(key)
    return node_name, task_id or ""


def _normalize_json_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    _write_text(path, text + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    lines = [
        json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":")) for row in rows
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "diagnostics_dir",
    "emit_diagnostics",
]
