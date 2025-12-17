"""Native Hamilton implementation for export_parquet target.

This module implements the export_parquet target as a pure Hamilton DAG,
exporting analytics data to Parquet format for efficient storage and analysis.

Phase 5: Export domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import ibis.expr.types as ir
from hamilton.function_modifiers import check_output_custom, schema, source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import FileArtifactSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_file_artifact_materialization,
)
from codeintel.build.hamilton.native.runner import should_skip_native_target
from codeintel.build.hamilton.validators import build_table_contract
from codeintel.build.hashing import compute_input_hash
from codeintel.build.targets import TargetGraph
from codeintel.storage.ibis_types import and_predicates

LOG = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ir.Table)


@tag(domain="export", target="export_parquet", node_type="compute")
@check_output_custom(
    *build_table_contract(
        required_columns=["function_goid_h128", "repo", "commit"],
        no_nulls=["function_goid_h128", "repo", "commit"],
    ),
)
@schema.output(
    ("function_goid_h128", "string"),
    ("repo", "string"),
    ("commit", "string"),
    ("loc", "int"),
    ("complexity", "int"),
    ("parameter_count", "int"),
    ("return_count", "int"),
    ("has_docstring", "bool"),
)
def t__export_parquet__compute(
    env: BuildEnv,
    q__analytics__function_metrics: ir.Table,
) -> ir.Table:
    """Compute the Parquet export table expression.

    This node prepares Ibis table expressions for export to Parquet format.
    Filters the function_metrics table to the current snapshot.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and config.
    q__analytics__function_metrics
        Ibis table expression for analytics.function_metrics.

    Returns
    -------
    ir.Table
        Ibis table expression for export, filtered to current snapshot.

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


@SaveToDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node("artifact.parquet_export"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("export_parquet"),
    artifact_name=value("parquet_export"),
)
@tag(domain="export", target="export_parquet", node_type="compute", target_="export_parquet__bytes")
def export_parquet__bytes(
    env: BuildEnv,
    graph: TargetGraph,
    t__export_parquet__compute: ir.Table,
) -> bytes | None:
    """Serialize the Parquet export payload for file materialization."""
    target = graph.get("export_parquet")
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

    df = t__export_parquet__compute.execute()
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    return buffer.getvalue()


@tag(domain="export", target="export_parquet", node_type="materialize")
def t__export_parquet(
    env: BuildEnv,
    graph: TargetGraph,
    m__artifact__parquet_export: dict[str, Any],
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
    return record_from_file_artifact_materialization(
        env=env,
        graph=graph,
        target_name="export_parquet",
        expected_artifact_name="parquet_export",
        materialization=m__artifact__parquet_export,
    )


__all__ = ["t__export_parquet", "t__export_parquet__compute"]
