"""Native Hamilton implementations for export targets.

This module consolidates file-artifact export targets:
- ``export_jsonl``: JSONL export of selected analytics datasets
- ``export_parquet``: Parquet export of selected analytics datasets

Both targets use DAG-visible artifact I/O via ``FileArtifactSaver``.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from typing import Any, cast

import ibis.expr.types as ir
import pandas as pd
from hamilton.function_modifiers import check_output_custom, schema, source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.build.contracts import ArtifactSpec
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import FileArtifactSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_file_artifact_materialization,
)
from codeintel.build.hamilton.native.runner import should_skip_native_target
from codeintel.build.hamilton.native.target_spec_helpers import (
    TargetSpecOptions,
    make_output_target,
)
from codeintel.build.hamilton.validators import build_table_contract
from codeintel.build.hashing import compute_input_hash
from codeintel.build.targets import TargetGraph
from codeintel.storage.ibis_types import and_predicates

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ir.Table, pd.DataFrame)

EXPORT_JSONL_TARGET_NAME = "export_jsonl"
EXPORT_PARQUET_TARGET_NAME = "export_parquet"

JSONL_EXPORT_ARTIFACT_NAME = "jsonl_export"
PARQUET_EXPORT_ARTIFACT_NAME = "parquet_export"

EXPORT_JSONL_ARTIFACT_SPECS = (
    ArtifactSpec(
        JSONL_EXPORT_ARTIFACT_NAME,
        "{export_dir}/codeintel.jsonl",
        "JSONL export of analytics datasets",
    ),
)
EXPORT_PARQUET_ARTIFACT_SPECS = (
    ArtifactSpec(
        PARQUET_EXPORT_ARTIFACT_NAME,
        "{export_dir}/codeintel.parquet",
        "Parquet export of analytics datasets",
    ),
)

TARGET_SPECS = (
    make_output_target(
        name=EXPORT_JSONL_TARGET_NAME,
        module="export",
        description="Export datasets to JSONL format for Document Output.",
        options=TargetSpecOptions(
            artifacts=EXPORT_JSONL_ARTIFACT_SPECS,
        ),
    ),
    make_output_target(
        name=EXPORT_PARQUET_TARGET_NAME,
        module="export",
        description="Export datasets to Parquet format for Document Output.",
        options=TargetSpecOptions(
            artifacts=EXPORT_PARQUET_ARTIFACT_SPECS,
        ),
    ),
)


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


@tag(domain="export", target=EXPORT_JSONL_TARGET_NAME, node_type="tool")
def t__export_jsonl__compute(
    env: BuildEnv,
    graph: TargetGraph,
    q__core__modules: ir.Table,
    q__analytics__function_metrics: ir.Table,
) -> ExportJsonlComputeResult | None:
    """Compute export manifest and gather data for JSONL export.

    Returns
    -------
    ExportJsonlComputeResult | None
        Export payload components, or None when the target is skipped.
    """
    target = graph.get(EXPORT_JSONL_TARGET_NAME)
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

    modules_df = cast("pd.DataFrame", modules.execute())
    function_metrics_df = cast("pd.DataFrame", function_metrics.execute())
    modules_data = tuple(cast("list[dict[str, Any]]", modules_df.to_dict(orient="records")))
    function_metrics_data = tuple(
        cast("list[dict[str, Any]]", function_metrics_df.to_dict(orient="records"))
    )

    metadata = {
        "repo": env.snapshot.repo,
        "commit": env.snapshot.commit,
        "export_format": "jsonl",
        "modules_count": len(modules_data),
        "function_metrics_count": len(function_metrics_data),
    }

    return ExportJsonlComputeResult(
        modules_data=modules_data,
        function_metrics_data=function_metrics_data,
        metadata=metadata,
    )


@SaveToDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node(f"artifact.{JSONL_EXPORT_ARTIFACT_NAME}"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(EXPORT_JSONL_TARGET_NAME),
    artifact_name=value(JSONL_EXPORT_ARTIFACT_NAME),
)
@tag(
    domain="export",
    target=EXPORT_JSONL_TARGET_NAME,
    node_type="compute",
    target_="export_jsonl__content",
)
def export_jsonl__content(t__export_jsonl__compute: ExportJsonlComputeResult | None) -> str | None:
    """Build JSONL content for export_jsonl artifact.

    Returns
    -------
    str | None
        JSONL content, or None when the target is skipped.
    """
    if t__export_jsonl__compute is None:
        return None

    modules_data = t__export_jsonl__compute.modules_data
    function_metrics_data = t__export_jsonl__compute.function_metrics_data
    metadata = t__export_jsonl__compute.metadata

    jsonl_lines: list[str] = [json.dumps({"_metadata": metadata}, ensure_ascii=False)]
    jsonl_lines.extend(
        json.dumps({"_type": "module", **module}, ensure_ascii=False) for module in modules_data
    )
    jsonl_lines.extend(
        json.dumps({"_type": "function_metric", **metric}, ensure_ascii=False)
        for metric in function_metrics_data
    )
    return "\n".join(jsonl_lines) + "\n"


@tag(domain="export", target=EXPORT_JSONL_TARGET_NAME, node_type="materialize")
def t__export_jsonl(
    env: BuildEnv,
    graph: TargetGraph,
    m__artifact__jsonl_export: dict[str, Any],
) -> TargetRunRecord:
    """Write JSONL export artifact and return record with ArtifactRef.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    return record_from_file_artifact_materialization(
        env=env,
        graph=graph,
        target_name=EXPORT_JSONL_TARGET_NAME,
        expected_artifact_name=JSONL_EXPORT_ARTIFACT_NAME,
        materialization=m__artifact__jsonl_export,
    )


@tag(domain="export", target=EXPORT_PARQUET_TARGET_NAME, node_type="compute")
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

    Returns
    -------
    ir.Table
        Ibis expression producing rows for the Parquet export artifact.
    """
    return q__analytics__function_metrics.filter(
        and_predicates(
            q__analytics__function_metrics.repo == env.snapshot.repo,
            q__analytics__function_metrics.commit == env.snapshot.commit,
        )
    )


@SaveToDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node(f"artifact.{PARQUET_EXPORT_ARTIFACT_NAME}"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(EXPORT_PARQUET_TARGET_NAME),
    artifact_name=value(PARQUET_EXPORT_ARTIFACT_NAME),
)
@tag(
    domain="export",
    target=EXPORT_PARQUET_TARGET_NAME,
    node_type="tool",
    target_="export_parquet__bytes",
)
def export_parquet__bytes(
    env: BuildEnv,
    graph: TargetGraph,
    t__export_parquet__compute: ir.Table,
) -> bytes | None:
    """Serialize the Parquet export payload for file materialization.

    Returns
    -------
    bytes | None
        Serialized Parquet bytes, or None when the target is skipped.
    """
    target = graph.get(EXPORT_PARQUET_TARGET_NAME)
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

    df = cast("pd.DataFrame", t__export_parquet__compute.execute())
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    return buffer.getvalue()


@tag(domain="export", target=EXPORT_PARQUET_TARGET_NAME, node_type="materialize")
def t__export_parquet(
    env: BuildEnv,
    graph: TargetGraph,
    m__artifact__parquet_export: dict[str, Any],
) -> TargetRunRecord:
    """Write Parquet export artifact and return record with ArtifactRef.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    return record_from_file_artifact_materialization(
        env=env,
        graph=graph,
        target_name=EXPORT_PARQUET_TARGET_NAME,
        expected_artifact_name=PARQUET_EXPORT_ARTIFACT_NAME,
        materialization=m__artifact__parquet_export,
    )


__all__ = [
    "ExportJsonlComputeResult",
    "export_jsonl__content",
    "t__export_jsonl",
    "t__export_jsonl__compute",
    "t__export_parquet",
    "t__export_parquet__compute",
]
