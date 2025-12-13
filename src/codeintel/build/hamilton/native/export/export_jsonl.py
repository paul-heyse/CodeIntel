"""Native Hamilton implementation for export_jsonl target.

This module implements the export_jsonl target as a pure Hamilton DAG,
exporting analytics data to JSONL format for external consumption.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, TypedDict, cast

import duckdb
from hamilton.function_modifiers import tag

from codeintel.build.hamilton.manifest_hook import TargetRunRecord, compute_target_input_hash
from codeintel.build.hamilton.native.artifact_materializer import (
    ArtifactMaterializationContext,
    ArtifactMaterializationSpec,
    materialize_artifact,
)
from codeintel.build.hamilton.native.runner import (
    NativeRunInfo,
    create_failed_record,
    create_skipped_record,
    create_success_record,
    save_manifest,
    should_skip_native_target,
)
from codeintel.storage.ibis_types import and_predicates

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    import ibis.expr.types as ir
    import pandas as pd

    from codeintel.build.env import BuildEnv
    from codeintel.build.targets import TargetGraph


class ExportJsonlComputeResult(TypedDict):
    modules_data: list[dict[str, Any]]
    function_metrics_data: list[dict[str, Any]]
    metadata: dict[str, Any]


@tag(domain="export", target="export_jsonl", node_kind="compute")
def t__export_jsonl__compute(
    env: BuildEnv,
    q__core__modules: ir.Table,
    q__analytics__function_metrics: ir.Table,
) -> ExportJsonlComputeResult:
    """Compute export manifest and gather data for JSONL export.

    This node collects data from multiple analytics tables and prepares
    it for export to JSONL format. The export includes modules and
    function metrics with full snapshot context.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and config.
    q__core__modules
        Ibis table expression for core.modules.
    q__analytics__function_metrics
        Ibis table expression for analytics.function_metrics.

    Returns
    -------
    dict[str, object]
        Export specification with:
        - "modules_data": List of module records
        - "function_metrics_data": List of function metric records
        - "metadata": Export metadata (snapshot info, timestamps, etc.)

    Examples
    --------
    >>> # This node is executed by Hamilton as part of the export_jsonl target
    >>> # It produces a dict that is consumed by t__export_jsonl materializer
    """
    LOG.info("Computing export_jsonl: gathering data for export")

    # Filter to current snapshot
    modules = q__core__modules.filter(
        and_predicates(
            q__core__modules.repo == env.snapshot.repo,
            q__core__modules.commit == env.snapshot.commit,
        )
    )

    function_metrics = q__analytics__function_metrics.filter(
        and_predicates(
            q__analytics__function_metrics.repo == env.snapshot.repo,
            q__analytics__function_metrics.commit == env.snapshot.commit,
        )
    )

    # Execute queries and convert to Python lists
    modules_df = cast("pd.DataFrame", modules.execute())
    function_metrics_df = cast("pd.DataFrame", function_metrics.execute())
    modules_data = cast("list[dict[str, Any]]", modules_df.to_dict(orient="records"))
    function_metrics_data = cast(
        "list[dict[str, Any]]", function_metrics_df.to_dict(orient="records")
    )

    # Build export metadata
    metadata = {
        "repo": env.snapshot.repo,
        "commit": env.snapshot.commit,
        "export_format": "jsonl",
        "modules_count": len(modules_data),
        "function_metrics_count": len(function_metrics_data),
    }

    LOG.info(
        "export_jsonl compute complete: %d modules, %d function metrics",
        len(modules_data),
        len(function_metrics_data),
    )

    return {
        "modules_data": modules_data,
        "function_metrics_data": function_metrics_data,
        "metadata": metadata,
    }


@tag(domain="export", target="export_jsonl", node_kind="materialize")
def t__export_jsonl(
    env: BuildEnv,
    graph: TargetGraph,
    t__export_jsonl__compute: ExportJsonlComputeResult,
) -> TargetRunRecord:
    """Write JSONL export artifact and return record with ArtifactRef.

    This node takes the computed export data and writes it to a JSONL file
    in the export directory, using atomic file write semantics.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and config.
    graph
        Target graph for accessing OutputTarget contract.
    t__export_jsonl__compute
        Export data from compute node.

    Returns
    -------
    TargetRunRecord
        Record capturing execution status, duration, and artifact references.

    Examples
    --------
    >>> # This node is executed by Hamilton after the compute node succeeds
    >>> # It materializes the export data to a JSONL file
    """
    LOG.info("Materializing export_jsonl to file")

    target = graph.get("export_jsonl")
    if target is None:
        return create_failed_record(
            target=graph.get("modules") or graph.all_targets[0],
            input_hash="",
            options_hash=None,
            duration_ms=0.0,
            error=ValueError("export_jsonl target not found in graph"),
        )

    input_hash = compute_target_input_hash(
        target=target,
        snapshot=env.snapshot,
        gateway=env.gateway,
        manifests=env.manifest_index,
    )

    if should_skip_native_target(env, target, input_hash):
        return create_skipped_record(
            target=target,
            env=env,
            run=NativeRunInfo(input_hash=input_hash, options_hash=None, duration_ms=0.0),
        )

    output_file = env.paths.document_output_dir / "codeintel.jsonl"

    # Extract data from compute result
    modules_data = t__export_jsonl__compute["modules_data"]
    function_metrics_data = t__export_jsonl__compute["function_metrics_data"]
    metadata = t__export_jsonl__compute["metadata"]

    # Format as JSONL (one JSON object per line)
    jsonl_lines: list[str] = []

    # Add metadata as first line
    jsonl_lines.append(json.dumps({"_metadata": metadata}, ensure_ascii=False))

    jsonl_lines.extend(
        [json.dumps({"_type": "module", **module}, ensure_ascii=False) for module in modules_data]
    )
    jsonl_lines.extend(
        [
            json.dumps({"_type": "function_metric", **metric}, ensure_ascii=False)
            for metric in function_metrics_data
        ]
    )

    jsonl_content = "\n".join(jsonl_lines) + "\n"

    # Materialize artifact
    try:
        artifact_ref = materialize_artifact(
            ArtifactMaterializationContext(
                snapshot=env.snapshot,
                gateway=env.gateway,
                owner_target=target.name,
                input_hash=input_hash,
            ),
            ArtifactMaterializationSpec(
                artifact_name="jsonl_export",
                artifact_type="file",
                content=jsonl_content,
                output_path=output_file,
                metadata={
                    "format": "jsonl",
                    "lines": len(jsonl_lines),
                    "modules_count": len(modules_data),
                    "function_metrics_count": len(function_metrics_data),
                },
            ),
        )
    except (OSError, ValueError, RuntimeError, duckdb.Error) as exc:
        return create_failed_record(
            target=target,
            input_hash=input_hash,
            options_hash=None,
            duration_ms=0.0,
            error=exc,
        )

    LOG.info("export_jsonl materialization complete: %s", output_file)

    record = create_success_record(
        target=target,
        env=env,
        run=NativeRunInfo(input_hash=input_hash, options_hash=None, duration_ms=0.0),
    )

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


__all__ = ["t__export_jsonl", "t__export_jsonl__compute"]
