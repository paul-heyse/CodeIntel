"""Finalize helpers for graph assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.graphs.engine.datasets import GraphRunMetadata, persist_finalize_artifacts
from codeintel.build.tabular.plan_ops import Plan
from codeintel.core.columnar.arrowdsl import ExecutionPlan, PipelineRunOptions, run_pipeline
from codeintel.core.columnar.finalize_ops import FinalizeResult, finalize_spec_for_table
from codeintel.core.columnar.run_manifest import RunManifestOptions
from codeintel.core.datasets.paths import SnapshotIdError, dataset_snapshot_dir

if TYPE_CHECKING:
    from codeintel.core.columnar.dedupe_ops import DedupeTier
    from codeintel.core.columnar.execution_context import ExecutionContext
    from codeintel.core.columnar.streaming import ScanTelemetry


@dataclass(frozen=True, slots=True)
class GraphFinalizeArtifacts:
    """Finalize artifact persistence configuration."""

    dataset_root: Path
    snapshot_id: str
    run_metadata: GraphRunMetadata | None = None
    scan_telemetry: ScanTelemetry | None = None
    manifest_dir: Path | None = None
    manifest_options: RunManifestOptions | None = None


def finalize_graph_plan(
    plan: Plan,
    *,
    table_key: str,
    determinism: DedupeTier,
    ctx: ExecutionContext | None,
    artifacts: GraphFinalizeArtifacts | None = None,
) -> FinalizeResult:
    """Run a graph plan through finalize and optionally persist artifacts.

    Returns
    -------
    FinalizeResult
        Finalize result containing good rows, errors, and artifacts.
    """
    manifest_dir = _manifest_dir_for_artifacts(artifacts, table_key) if artifacts else None
    manifest_options = (
        _manifest_options_for_artifacts(artifacts, table_key) if artifacts else None
    )
    scan_telemetry = artifacts.scan_telemetry if artifacts is not None else None
    result = run_pipeline(
        plan=ExecutionPlan.from_plan(plan, determinism=determinism),
        finalize=finalize_spec_for_table(
            table_key,
            mode="tolerant",
            determinism=determinism,
        ),
        options=PipelineRunOptions(
            ctx=ctx,
            manifest_dir=manifest_dir,
            manifest_options=manifest_options,
            scan_telemetry=scan_telemetry,
        ),
    )
    if artifacts is not None:
        persist_finalize_artifacts(
            dataset_root=artifacts.dataset_root,
            snapshot_id=artifacts.snapshot_id,
            base_table_key=table_key,
            result=result,
            run_metadata=artifacts.run_metadata,
    )
    return result


def _manifest_dir_for_artifacts(
    artifacts: GraphFinalizeArtifacts | None,
    table_key: str,
) -> Path | None:
    if artifacts is None:
        return None
    if artifacts.manifest_dir is not None:
        return artifacts.manifest_dir
    try:
        return dataset_snapshot_dir(
            artifacts.dataset_root,
            table_key=table_key,
            snapshot_id=artifacts.snapshot_id,
        )
    except SnapshotIdError:
        return None


def _manifest_options_for_artifacts(
    artifacts: GraphFinalizeArtifacts | None,
    table_key: str,
) -> RunManifestOptions | None:
    if artifacts is None:
        return None
    extras: dict[str, object] = {
        "table_key": table_key,
        "snapshot_id": artifacts.snapshot_id,
    }
    if artifacts.run_metadata is not None:
        extras["graph_run"] = artifacts.run_metadata.manifest_extras()
    base = artifacts.manifest_options
    if base is None:
        filename = f"run_manifest_{table_key.replace('.', '_')}.json"
        return RunManifestOptions(extras=extras, filename=filename)
    merged_extras = {**extras, **(base.extras or {})}
    return RunManifestOptions(
        determinism=base.determinism,
        ordering=base.ordering,
        scan_telemetry=base.scan_telemetry,
        profile_name=base.profile_name,
        scan_profile=base.scan_profile,
        extras=merged_extras,
        filename=base.filename,
    )


__all__ = ["GraphFinalizeArtifacts", "finalize_graph_plan"]
