"""Native Hamilton implementation for export_jsonl target.

This module implements the export_jsonl target as a pure Hamilton DAG,
exporting analytics data to JSONL format for external consumption.

Phase 5: Export domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, cast

import ibis.expr.types as ir
import pandas as pd
from hamilton.function_modifiers import source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import FileArtifactSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_file_artifact_materialization,
)
from codeintel.build.hamilton.native.runner import should_skip_native_target
from codeintel.build.hashing import compute_input_hash
from codeintel.build.targets import TargetGraph
from codeintel.storage.ibis_types import and_predicates

LOG = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ir.Table, pd.DataFrame)


@dataclass(frozen=True)
class ExportJsonlComputeResult:
    """Result of JSONL export computation.

    Attributes
    ----------
    modules_data
        List of module records for export.
    function_metrics_data
        List of function metric records for export.
    metadata
        Export metadata including snapshot info.
    """

    modules_data: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    function_metrics_data: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


@tag(domain="export", target="export_jsonl", node_type="compute")
def t__export_jsonl__compute(
    env: BuildEnv,
    graph: TargetGraph,
    q__core__modules: ir.Table,
    q__analytics__function_metrics: ir.Table,
) -> ExportJsonlComputeResult | None:
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
    ExportJsonlComputeResult
        Export specification with modules data, function metrics data,
        and export metadata.

    Examples
    --------
    >>> # This node is executed by Hamilton as part of the export_jsonl target
    >>> # It produces a result that is consumed by t__export_jsonl materializer
    """
    target = graph.get("export_jsonl")
    if target is not None:
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            options_hash=None,
            manifests=env.manifest_index,
        )
        if should_skip_native_target(env, target, input_hash):
            return None

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
    modules_data = tuple(cast("list[dict[str, Any]]", modules_df.to_dict(orient="records")))
    function_metrics_data = tuple(
        cast("list[dict[str, Any]]", function_metrics_df.to_dict(orient="records"))
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

    return ExportJsonlComputeResult(
        modules_data=modules_data,
        function_metrics_data=function_metrics_data,
        metadata=metadata,
    )


@SaveToDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node("artifact.jsonl_export"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("export_jsonl"),
    artifact_name=value("jsonl_export"),
)
@tag(domain="export", target="export_jsonl", node_type="compute", target_="export_jsonl__content")
def export_jsonl__content(t__export_jsonl__compute: ExportJsonlComputeResult | None) -> str | None:
    """Build JSONL content for export_jsonl artifact."""
    if t__export_jsonl__compute is None:
        return None

    modules_data = t__export_jsonl__compute.modules_data
    function_metrics_data = t__export_jsonl__compute.function_metrics_data
    metadata = t__export_jsonl__compute.metadata

    jsonl_lines: list[str] = []
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
    return "\n".join(jsonl_lines) + "\n"


@tag(domain="export", target="export_jsonl", node_type="materialize")
def t__export_jsonl(
    env: BuildEnv,
    graph: TargetGraph,
    m__artifact__jsonl_export: dict[str, Any],
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
    return record_from_file_artifact_materialization(
        env=env,
        graph=graph,
        target_name="export_jsonl",
        expected_artifact_name="jsonl_export",
        materialization=m__artifact__jsonl_export,
    )


__all__ = [
    "ExportJsonlComputeResult",
    "export_jsonl__content",
    "t__export_jsonl",
    "t__export_jsonl__compute",
]
