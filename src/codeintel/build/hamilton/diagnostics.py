"""Hamilton build diagnostics emission helpers."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
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
from codeintel.build.schemas import get_schema_provider
from codeintel.core.columnar.streaming import DatasetScanOptions, build_scanner
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.datasets.arrow_store import scan_dataset
from codeintel.core.datasets.paths import dataset_snapshot_dir
from codeintel.core.hamilton import tags as ht
from codeintel.core.schemas.primitives import TableSchema

if TYPE_CHECKING:
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.core.hamilton.records import NodeExecutionRecord
    from codeintel.runtime.runtime_bundle import RuntimeBundle

log = logging.getLogger(__name__)
CACHE_LOG_KEY_TUPLE_LEN = 2


@dataclass(frozen=True)
class DiagnosticsTargets:
    requested: Sequence[str]
    computed: Sequence[str]
    skipped: Sequence[str]
    failed: Sequence[str]


@dataclass(frozen=True)
class DiagnosticsInputs:
    env: BuildEnv
    runtime: RuntimeBundle
    run_id: str
    cache_dir: Path
    cache_adapter: HamiltonCacheAdapter | None
    telemetry_records: Sequence[NodeExecutionRecord] | None
    targets: DiagnosticsTargets
    duration_ms: float
    domain: str | None


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    repo: str
    commit: str
    profile: str | None
    domain: str | None
    targets: DiagnosticsTargets
    duration_ms: float


def diagnostics_dir(build_dir: Path) -> Path:
    """Return the diagnostics directory under the build root.

    Returns
    -------
    Path
        Diagnostics directory path.
    """
    return build_dir / "diagnostics"


def emit_diagnostics(inputs: DiagnosticsInputs) -> None:
    """Emit Hamilton-native diagnostics artifacts under build/diagnostics."""
    diag_dir = _ensure_dir(diagnostics_dir(inputs.env.paths.build_dir))
    summary = RunSummary(
        run_id=inputs.run_id,
        repo=inputs.env.repo,
        commit=inputs.env.commit,
        profile=inputs.env.profile,
        domain=inputs.domain,
        targets=inputs.targets,
        duration_ms=inputs.duration_ms,
    )
    _write_run_summary(diag_dir / "run_summary.json", summary)
    _write_dag_exports(
        diag_dir=diag_dir,
        runtime=inputs.runtime,
        targets=list(inputs.targets.requested),
    )
    _write_null_inventory(
        diag_dir=diag_dir,
        env=inputs.env,
        runtime=inputs.runtime,
        targets=inputs.targets.requested,
        run_id=inputs.run_id,
    )
    if inputs.cache_adapter is None:
        return
    _write_cache_events(
        output_path=diag_dir / "cache_events.jsonl",
        cache_dir=inputs.cache_dir,
        cache_adapter=inputs.cache_adapter,
        run_id=inputs.run_id,
    )
    target_by_node = _target_map(inputs.runtime)
    _write_cache_keys_snapshot(
        output_path=diag_dir / "cache_keys.jsonl",
        cache_adapter=inputs.cache_adapter,
        run_id=inputs.run_id,
        target_by_node=target_by_node,
    )
    durations_ms = _node_duration_map(inputs.telemetry_records or ())
    decision_entries = build_cache_manifest_entries(
        cache_adapter=inputs.cache_adapter,
        run_id=inputs.run_id,
        target_by_node=target_by_node,
        durations_ms=durations_ms,
    )
    decision_path = default_decision_trace_path(inputs.env.paths.build_dir)
    try:
        write_decision_trace(decision_path, decision_entries)
    except (OSError, TypeError, ValueError) as exc:
        log.warning(
            "build.diagnostics.decision_trace_failed run_id=%s error=%s", inputs.run_id, exc
        )
    _write_cache_visualization(
        output_path=diag_dir / "cache_run_visualization.svg",
        cache_adapter=inputs.cache_adapter,
        run_id=inputs.run_id,
    )


def _write_run_summary(path: Path, summary: RunSummary) -> None:
    payload: dict[str, object] = {
        "run_id": summary.run_id,
        "repo": summary.repo,
        "commit": summary.commit,
        "profile": summary.profile,
        "domain": summary.domain,
        "requested_targets": list(summary.targets.requested),
        "computed_targets": list(summary.targets.computed),
        "skipped_targets": list(summary.targets.skipped),
        "failed_targets": list(summary.targets.failed),
        "duration_ms": summary.duration_ms,
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }
    _write_json(path, payload)


def _write_dag_exports(
    *,
    diag_dir: Path,
    runtime: RuntimeBundle,
    targets: list[str],
) -> None:
    if not targets:
        return
    try:
        _write_text(diag_dir / "dag.dot", export_dag_dot(runtime, targets))
        _write_text(diag_dir / "dag.json", export_dag_json(runtime, targets))
        _write_text(diag_dir / "dag.mermaid", export_dag_mermaid(runtime, targets))
    except (OSError, TypeError, ValueError) as exc:
        log.warning("build.diagnostics.dag_export_failed error=%s", exc)


def _write_null_inventory(
    *,
    diag_dir: Path,
    env: BuildEnv,
    runtime: RuntimeBundle,
    targets: Sequence[str],
    run_id: str,
) -> None:
    table_keys = _table_keys_for_targets(runtime, targets)
    if not table_keys:
        return
    payload = _null_inventory_payload(
        env=env,
        runtime=runtime,
        run_id=run_id,
        table_keys=table_keys,
    )
    if payload is None:
        return
    _write_json(diag_dir / "null_inventory.json", payload)


def _table_keys_for_targets(runtime: RuntimeBundle, targets: Sequence[str]) -> tuple[str, ...]:
    table_keys: set[str] = set()
    for target in targets:
        outputs = runtime.catalog.table_outputs_by_target.get(target, ())
        for output in outputs:
            table_keys.add(output.key)
    return tuple(sorted(table_keys))


def _null_inventory_payload(
    *,
    env: BuildEnv,
    runtime: RuntimeBundle,
    run_id: str,
    table_keys: Sequence[str],
) -> dict[str, object] | None:
    rows: list[dict[str, object]] = []
    tables_with_nulls = 0
    missing_tables = 0
    total_columns_with_nulls = 0
    total_non_nullable_nulls = 0

    for table_key in table_keys:
        entry = _null_inventory_for_table(
            env=env,
            runtime=runtime,
            table_key=table_key,
        )
        rows.append(entry)
        if entry.get("status") == "missing":
            missing_tables += 1
            continue
        columns_with_nulls = entry.get("columns_with_nulls")
        if isinstance(columns_with_nulls, list) and columns_with_nulls:
            tables_with_nulls += 1
            total_columns_with_nulls += len(columns_with_nulls)
            for column_entry in columns_with_nulls:
                if not isinstance(column_entry, Mapping):
                    continue
                if column_entry.get("nullable") is False:
                    total_non_nullable_nulls += 1

    return {
        "run_id": run_id,
        "repo": env.repo,
        "commit": env.commit,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "table_count": len(table_keys),
        "tables_scanned": len(table_keys) - missing_tables,
        "tables_missing": missing_tables,
        "tables_with_nulls": tables_with_nulls,
        "columns_with_nulls": total_columns_with_nulls,
        "non_nullable_columns_with_nulls": total_non_nullable_nulls,
        "tables": rows,
    }


def _null_inventory_for_table(
    *,
    env: BuildEnv,
    runtime: RuntimeBundle,
    table_key: str,
) -> dict[str, object]:
    snapshot_dir = dataset_snapshot_dir(
        env.paths.dataset_root_dir,
        table_key=table_key,
        snapshot_id=env.commit,
    )
    try:
        dataset = scan_dataset(
            dataset_root=env.paths.dataset_root_dir,
            table_key=table_key,
            snapshot_id=env.commit,
        )
    except FileNotFoundError:
        return {
            "table_key": table_key,
            "status": "missing",
            "snapshot_dir": str(snapshot_dir) if snapshot_dir is not None else None,
        }
    except (OSError, ValueError, pa.ArrowInvalid) as exc:
        return {
            "table_key": table_key,
            "status": "error",
            "snapshot_dir": str(snapshot_dir) if snapshot_dir is not None else None,
            "error": str(exc),
        }

    table_schema = _table_schema_for_table(runtime, table_key)
    nullable_by_name = _nullable_columns(table_schema)
    missing_columns: list[str] = []
    available = list(dataset.schema.names)
    if table_schema is not None:
        missing_columns = [
            name for name in table_schema.column_names() if name not in available
        ]
        columns = [name for name in table_schema.column_names() if name in available]
    else:
        columns = available
    if not columns:
        columns = available

    row_count, null_counts = _null_counts_for_columns(dataset, columns)
    columns_with_nulls: list[dict[str, object]] = []
    for name, null_count in null_counts.items():
        if null_count <= 0:
            continue
        nullable = nullable_by_name.get(name)
        ratio = None if row_count == 0 else null_count / row_count
        columns_with_nulls.append(
            {
                "name": name,
                "null_count": null_count,
                "null_ratio": ratio,
                "nullable": nullable,
            }
        )
    status = "ok"
    if missing_columns:
        status = "missing_columns"
    if columns_with_nulls:
        status = "nulls_detected"
    return {
        "table_key": table_key,
        "status": status,
        "row_count": row_count,
        "missing_columns": missing_columns,
        "columns_with_nulls": columns_with_nulls,
        "snapshot_dir": str(snapshot_dir) if snapshot_dir is not None else None,
    }


def _table_schema_for_table(runtime: RuntimeBundle, table_key: str) -> TableSchema | None:
    schema_index = runtime.schema_index
    if schema_index is not None:
        schema = schema_index.get_table_schema(
            table_key,
            allow_inference=False,
            perform_inference=False,
        )
        if schema is not None:
            return schema
    try:
        provider = get_schema_provider()
    except RuntimeError:
        return None
    return provider.get_table_schema(table_key)


def _nullable_columns(table_schema: TableSchema | None) -> dict[str, bool]:
    if table_schema is None:
        return {}
    return {column.name: column.nullable for column in table_schema.columns}


def _null_counts_for_columns(
    dataset: object,
    columns: Sequence[str],
) -> tuple[int, dict[str, int]]:
    scanner = build_scanner(
        dataset,
        options=DatasetScanOptions(
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
            columns=columns,
            unify_schemas=True,
        ),
    )
    null_counts = {name: 0 for name in columns}
    row_count = 0
    for batch in scanner.to_batches():
        row_count += batch.num_rows
        for index, name in enumerate(batch.schema.names):
            null_counts[name] += batch.column(index).null_count
    return row_count, null_counts


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
    logs_by_node = _safe_cache_logs(cache_adapter, run_id)
    if not logs_by_node:
        return []
    rows: list[dict[str, object]] = []
    for key, events in logs_by_node.items():
        node_name, task_id = _cache_log_key_parts(key)
        if not isinstance(events, list):
            continue
        rows.extend(
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
            for event in events
        )
    return rows


def _write_cache_keys_snapshot(
    *,
    output_path: Path,
    cache_adapter: HamiltonCacheAdapter,
    run_id: str,
    target_by_node: Mapping[str, str],
) -> None:
    logs_by_node = _safe_cache_logs(cache_adapter, run_id)
    if not logs_by_node:
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
                "run_id": run_id,
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


def _safe_cache_logs(
    cache_adapter: HamiltonCacheAdapter,
    run_id: str,
) -> dict[object, object]:
    try:
        logs_by_node = cache_adapter.logs(run_id=run_id, level="info")
    except KeyError:
        return {}
    if not isinstance(logs_by_node, dict):
        return {}
    return logs_by_node


def _write_cache_visualization(
    *,
    output_path: Path,
    cache_adapter: HamiltonCacheAdapter,
    run_id: str,
) -> None:
    try:
        cache_adapter.view_run(run_id=run_id, output_file_path=str(output_path))
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
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
    if (
        isinstance(key, tuple)
        and len(key) == CACHE_LOG_KEY_TUPLE_LEN
        and all(isinstance(item, str) for item in key)
    ):
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
