"""Native Hamilton implementation for export_parquet target.

This module implements the export_parquet target as a pure Hamilton DAG,
exporting analytics data to Parquet format for efficient storage and analysis.
"""

from __future__ import annotations

import io
import logging

import ibis.expr.types as ir
from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.artifact_materializer import (
    ArtifactMaterializationContext,
    ArtifactMaterializationSpec,
    materialize_artifact,
)
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.runner import (
    NativeRunInfo,
    RunRecordInputs,
    create_run_record,
    save_manifest,
)
from codeintel.build.targets import TargetGraph
from codeintel.storage.ibis_types import and_predicates

LOG = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ir.Table)


@tag(domain="export", target="export_parquet", node_type="compute")
def t__export_parquet__compute(
    env: BuildEnv,
    q__analytics__function_metrics: ir.Table,
) -> ir.Table:
    """Compute the Parquet export table expression.

    This node prepares Ibis table expressions for export to Parquet format.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and config.
    q__analytics__function_metrics
        Ibis table expression for analytics.function_metrics.

    Returns
    -------
    ir.Table
        Ibis table expression for export.

    Examples
    --------
    >>> # This node is executed by Hamilton as part of the export_parquet target
    >>> # It produces an Ibis expression that is consumed by t__export_parquet materializer
    """
    LOG.info("Computing export_parquet: preparing tables for export")

    # Filter to current snapshot
    function_metrics = q__analytics__function_metrics.filter(
        and_predicates(
            q__analytics__function_metrics.repo == env.snapshot.repo,
            q__analytics__function_metrics.commit == env.snapshot.commit,
        )
    )

    LOG.info("export_parquet compute complete")
    return function_metrics


@tag(domain="export", target="export_parquet", node_type="materialize")
def t__export_parquet(
    env: BuildEnv,
    graph: TargetGraph,
    t__export_parquet__compute: ir.Table,
) -> TargetRunRecord:
    """Write Parquet export artifact and return record with ArtifactRef.

    This node takes the computed export tables and writes them to Parquet files
    in the export directory.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and config.
    graph
        Target graph for accessing OutputTarget contract.
    t__export_parquet__compute
        Export tables from compute node.

    Returns
    -------
    TargetRunRecord
        Record capturing execution status, duration, and artifact references.

    Examples
    --------
    >>> # This node is executed by Hamilton after the compute node succeeds
    >>> # It materializes the Ibis expressions to Parquet files
    """
    LOG.info("Materializing export_parquet to files")

    executor = NativeTargetExecutor.for_target(env, graph, "export_parquet")

    if executor.should_skip():
        return executor.skip()

    output_file = env.paths.document_output_dir / "codeintel.parquet"

    try:
        df = t__export_parquet__compute.execute()
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, engine="pyarrow")
        parquet_bytes = buffer.getvalue()
    except (OSError, ValueError, RuntimeError) as exc:
        return executor.fail(exc)

    try:
        artifact_ref = materialize_artifact(
            ArtifactMaterializationContext(
                snapshot=env.snapshot,
                gateway=env.gateway,
                owner_target="export_parquet",
                input_hash=executor.input_hash,
            ),
            ArtifactMaterializationSpec(
                artifact_name="parquet_export",
                artifact_type="file",
                content=parquet_bytes,
                output_path=output_file,
                metadata={"format": "parquet", "rows": len(df), "bytes": len(parquet_bytes)},
            ),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        return executor.fail(exc)

    LOG.info("export_parquet materialization complete: %s", output_file)

    # Create success record with artifact
    run = NativeRunInfo(
        input_hash=executor.input_hash,
        options_hash=executor.options_hash,
        duration_ms=0.0,
    )
    record = create_run_record(
        executor.target,
        "succeeded",
        executor.input_hash,
        inputs=RunRecordInputs(env=env, run=run),
    )

    # Add artifact to record
    record = TargetRunRecord(
        target=record.target,
        plugin_name=record.plugin_name,
        status=record.status,
        input_hash=record.input_hash,
        options_hash=record.options_hash,
        duration_ms=record.duration_ms,
        row_counts=record.row_counts,
        error=record.error,
        datasets=record.datasets,
        artifacts=(artifact_ref,),
    )
    save_manifest(env, record)
    return record


__all__ = ["t__export_parquet", "t__export_parquet__compute"]
