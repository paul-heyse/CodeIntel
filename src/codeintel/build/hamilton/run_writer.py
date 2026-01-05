"""Build run persistence utilities.

This module centralizes persistence for the Hamilton build executor:

- start/complete build run records
- persist per-target run records
- persist node-level telemetry
- emit Phase 4 asset catalog records

All persistence operations are best-effort: failures are logged and execution continues.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import orjson as _orjson
except ImportError:  # pragma: no cover - optional dependency
    _orjson = None

from codeintel.build.hamilton.build_log import build_log_path
from codeintel.build.hamilton.tagging import tag_schema_spec, tag_schema_summary
from codeintel.core.datasets.manifests import dataset_manifest_path, read_dataset_manifest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from codeintel.build.hamilton.build_log import BuildLogContext
    from codeintel.build.hamilton.dag_catalog import DagCatalog
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.meta.bundle import BuildMetadataBundleWriter
    from codeintel.core.hamilton.records import NodeExecutionRecord, TargetRunRecord

log = logging.getLogger(__name__)

_RUN_REPORT_FILENAME = "run_report_{run_id}.jsonl"
_TAG_SCHEMA_FILENAME = "tag_schema.json"


def _sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_line(payload: Mapping[str, object]) -> str:
    if _orjson is not None:
        return _orjson.dumps(payload).decode("utf-8")
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _snapshot_id(env: BuildEnv, *, run_id: str) -> str:
    value = env.commit.strip()
    return value if value else run_id


def _snapshot_root(env: BuildEnv, *, run_id: str) -> Path | None:
    dataset_root = env.paths.dataset_root_dir
    if dataset_root is None:
        return None
    return dataset_root / _snapshot_id(env, run_id=run_id)


def _spec_hash(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return _sha256_text(serialized)


def _normalize_tag_value(value: object) -> object:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalize_tags(tags: Mapping[str, object] | None) -> dict[str, object]:
    if not tags:
        return {}
    return {key: _normalize_tag_value(val) for key, val in tags.items()}


def _dataset_manifest_path_for(
    *,
    dataset_root: Path | None,
    snapshot_id: str,
    table_key: str,
    metadata: Mapping[str, object],
) -> Path | None:
    raw = metadata.get("dataset_manifest_path")
    if isinstance(raw, str) and raw:
        return Path(raw)
    if dataset_root is None:
        return None
    return dataset_manifest_path(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )


def _load_manifest(path: Path | None) -> tuple[str | None, int | None]:
    if path is None or not path.is_file():
        return None, None
    try:
        manifest = read_dataset_manifest(path)
    except (OSError, TypeError, ValueError, KeyError):
        return None, None
    return manifest.schema_hash, manifest.row_count


@dataclass(frozen=True, slots=True)
class BuildRunWriter:
    """Persist build run lifecycle data to the metadata bundle."""

    metadata_bundle: BuildMetadataBundleWriter | None = None

    @staticmethod
    def start_run(
        *,
        env: BuildEnv,
        run_id: str,
        requested_targets: Sequence[str],
        started_at: datetime,
    ) -> None:
        """Record the start of a build run.

        Parameters
        ----------
        env
            Build environment containing repo/commit identifiers.
        run_id
            Run identifier.
        requested_targets
            Requested targets for the run.
        started_at
            Run start timestamp.
        """
        if not run_id:
            return
        if not requested_targets:
            return
        if not env.repo:
            return
        if started_at.tzinfo is None:
            return

    @staticmethod
    def complete_run(
        *,
        run_id: str,
        success: bool,
        computed_targets: Sequence[str],
        skipped_targets: Sequence[str],
        error_summary: str | None,
    ) -> None:
        """Complete the build run record.

        Parameters
        ----------
        run_id
            Run identifier.
        success
            Whether the run succeeded.
        computed_targets
            Targets that were computed.
        skipped_targets
            Targets that were skipped.
        error_summary
            Optional error summary if failed.
        """
        if not run_id:
            return
        if success and error_summary:
            return
        if not computed_targets and not skipped_targets:
            return

    @staticmethod
    def save_run_targets(
        *,
        env: BuildEnv,
        run_id: str,
        records: Sequence[TargetRunRecord],
    ) -> None:
        """Persist per-target execution records for a run.

        Parameters
        ----------
        env
            Build environment containing repo/commit identifiers.
        run_id
            Run identifier.
        records
            Target run records to persist.
        """
        if not run_id or not records:
            return
        if not env.repo:
            return

    @staticmethod
    def save_run_nodes(
        run_id: str,
        records: Sequence[NodeExecutionRecord],
    ) -> int:
        """Persist node-level execution telemetry for a run.

        Parameters
        ----------
        run_id
            Run identifier.
        records
            Node execution records to persist.

        Returns
        -------
        int
            Number of records persisted.
        """
        if not run_id or not records:
            return 0
        return 0

    @staticmethod
    def write_build_log(
        *,
        context: BuildLogContext,
        events: Sequence[Mapping[str, object]],
    ) -> Path | None:
        """Write the consolidated JSONL build log for a run.

        Returns
        -------
        Path | None
            Path to the JSONL artifact, or None when no events were written.
        """
        if not events:
            return None
        try:
            path = build_log_path(context=context)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for event in events:
                    handle.write(_json_line(event))
                    handle.write("\n")
        except (OSError, ValueError) as exc:
            log.warning(
                "build.hamilton.writer.build_log_failed run_id=%s error=%s",
                context.run_id,
                exc,
            )
            return None
        else:
            return path

    @staticmethod
    def persist_asset_catalog(
        *,
        env: BuildEnv,
        run_id: str,
        catalog: DagCatalog,
        records: Sequence[TargetRunRecord],
    ) -> None:
        """Emit Phase 4 asset catalog records for a run.

        Parameters
        ----------
        env
            Build environment containing gateway access and snapshot metadata.
        run_id
            Run identifier.
        catalog
            DAG catalog for resolving contracts/dependencies.
        records
            Target run records to emit as assets.
        """
        if not run_id or not records:
            return
        if not env.commit:
            return
        if not catalog.targets:
            return

    @staticmethod
    def write_run_report(*, inputs: RunReportInputs) -> Path | None:
        """Write the consolidated run report JSONL and tag schema artifact.

        Returns
        -------
        Path | None
            Path to the run report JSONL, or None when not written.
        """
        tag_spec = tag_schema_spec()
        spec_hash = _spec_hash(tag_spec)
        snapshot_root = _snapshot_root(inputs.env, run_id=inputs.run_id)
        tag_spec_path = None
        tag_spec_written = False
        if snapshot_root is not None:
            tag_spec_path = snapshot_root / _TAG_SCHEMA_FILENAME
            tag_spec_written = _write_json_payload(tag_spec_path, tag_spec)
        tag_summary = tag_schema_summary()
        tag_summary.update(
            {
                "spec_hash": spec_hash,
                "spec_path": str(tag_spec_path) if tag_spec_written and tag_spec_path else None,
            }
        )

        records_payload = _run_report_records(
            _RunReportPayloadInputs(
                env=inputs.env,
                run_id=inputs.run_id,
                catalog=inputs.catalog,
                records=inputs.records,
                computed_targets=inputs.computed_targets,
                skipped_targets=inputs.skipped_targets,
                failed_targets=inputs.failed_targets,
                started_at=inputs.started_at,
                duration_ms=inputs.duration_ms,
                success=inputs.success,
                error_summary=inputs.error_summary,
                tag_summary=tag_summary,
                snapshot_id=_snapshot_id(inputs.env, run_id=inputs.run_id),
            )
        )
        run_report_rel = f"runs/{_RUN_REPORT_FILENAME.format(run_id=inputs.run_id)}"
        bundle = inputs.env.metadata_bundle
        if bundle is None:
            log.warning(
                "build.hamilton.writer.run_report_skipped run_id=%s reason=missing_bundle",
                inputs.run_id,
            )
            return None
        for record in records_payload:
            bundle.append_jsonl(run_report_rel, record, schema_version="v1")
        bundle.append_jsonl(
            "runs/run_index.jsonl",
            {
                "run_id": inputs.run_id,
                "repo": inputs.env.repo,
                "commit": inputs.env.commit,
                "started_at": inputs.started_at.isoformat(),
                "duration_ms": inputs.duration_ms,
                "success": inputs.success,
                "report_path": run_report_rel,
                "computed_targets_count": len(inputs.computed_targets),
                "skipped_targets_count": len(inputs.skipped_targets),
                "failed_targets_count": len(inputs.failed_targets),
            },
            schema_version="v1",
        )
        return bundle.bundle_root / run_report_rel


@dataclass(frozen=True, slots=True)
class RunReportInputs:
    """Inputs for assembling a consolidated run report."""

    env: BuildEnv
    run_id: str
    catalog: DagCatalog
    records: Sequence[TargetRunRecord]
    computed_targets: Sequence[str]
    skipped_targets: Sequence[str]
    failed_targets: Sequence[str]
    started_at: datetime
    duration_ms: float
    success: bool
    error_summary: str | None


@dataclass(frozen=True, slots=True)
class _RunReportPayloadInputs:
    env: BuildEnv
    run_id: str
    catalog: DagCatalog
    records: Sequence[TargetRunRecord]
    computed_targets: Sequence[str]
    skipped_targets: Sequence[str]
    failed_targets: Sequence[str]
    started_at: datetime
    duration_ms: float
    success: bool
    error_summary: str | None
    tag_summary: Mapping[str, object]
    snapshot_id: str


def _run_report_records(inputs: _RunReportPayloadInputs) -> list[dict[str, object]]:
    run_record: dict[str, object] = {
        "record_type": "run_metadata",
        "run_id": inputs.run_id,
        "repo": inputs.env.repo,
        "commit": inputs.env.commit,
        "snapshot_id": inputs.snapshot_id,
        "started_at": inputs.started_at.isoformat(),
        "duration_ms": inputs.duration_ms,
        "success": inputs.success,
        "computed_targets": list(inputs.computed_targets),
        "skipped_targets": list(inputs.skipped_targets),
        "failed_targets": list(inputs.failed_targets),
        "error_summary": inputs.error_summary,
    }
    summary_record: dict[str, object] = {
        "record_type": "tag_schema_summary",
        "run_id": inputs.run_id,
        "repo": inputs.env.repo,
        "commit": inputs.env.commit,
        "snapshot_id": inputs.snapshot_id,
        "summary": dict(inputs.tag_summary),
    }
    output_records = _output_catalog_records(
        env=inputs.env,
        run_id=inputs.run_id,
        catalog=inputs.catalog,
        records=inputs.records,
        snapshot_id=inputs.snapshot_id,
    )
    return [run_record, summary_record, *output_records]


def _output_catalog_records(
    *,
    env: BuildEnv,
    run_id: str,
    catalog: DagCatalog,
    records: Sequence[TargetRunRecord],
    snapshot_id: str,
) -> list[dict[str, object]]:
    dataset_root = env.paths.dataset_root_dir
    entries: list[dict[str, object]] = []
    for record in sorted(records, key=lambda item: item.target):
        for dataset in sorted(record.datasets, key=lambda item: item.table_key):
            output = catalog.table_outputs.get(dataset.table_key)
            tags = _normalize_tags(output.tags) if output is not None else {}
            manifest_path = _dataset_manifest_path_for(
                dataset_root=dataset_root,
                snapshot_id=snapshot_id,
                table_key=dataset.table_key,
                metadata=dataset.metadata,
            )
            schema_hash, manifest_row_count = _load_manifest(manifest_path)
            entries.append(
                {
                    "record_type": "output_catalog",
                    "run_id": run_id,
                    "repo": env.repo,
                    "commit": env.commit,
                    "snapshot_id": snapshot_id,
                    "output_kind": "table",
                    "table_key": dataset.table_key,
                    "target": record.target,
                    "status": record.status,
                    "row_count": dataset.row_count,
                    "manifest_row_count": manifest_row_count,
                    "schema_hash": schema_hash,
                    "dataset_manifest_path": str(manifest_path) if manifest_path else None,
                    "output_role": output.role if output is not None else None,
                    "saver_node": output.saver_node if output is not None else None,
                    "sink": output.sink if output is not None else None,
                    "tags": tags,
                }
            )
        for artifact in sorted(record.artifacts, key=lambda item: item.name):
            output = catalog.artifact_outputs.get(artifact.name)
            tags = _normalize_tags(output.tags) if output is not None else {}
            entries.append(
                {
                    "record_type": "output_catalog",
                    "run_id": run_id,
                    "repo": env.repo,
                    "commit": env.commit,
                    "snapshot_id": snapshot_id,
                    "output_kind": "artifact",
                    "artifact_name": artifact.name,
                    "artifact_type": artifact.artifact_type,
                    "artifact_path": artifact.path,
                    "target": record.target,
                    "status": record.status,
                    "output_role": output.role if output is not None else None,
                    "saver_node": output.saver_node if output is not None else None,
                    "sink": output.sink if output is not None else None,
                    "tags": tags,
                }
            )
    return entries


def _write_json_payload(path: Path, payload: Mapping[str, object]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        path.write_text(serialized, encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        log.warning("build.hamilton.writer.json_write_failed path=%s error=%s", path, exc)
        return False
    return True


def _write_jsonl_payload(path: Path, payloads: Sequence[Mapping[str, object]]) -> bool:
    if not payloads:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(_json_line(payload))
                handle.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        log.warning("build.hamilton.writer.jsonl_write_failed path=%s error=%s", path, exc)
        return False
    return True
