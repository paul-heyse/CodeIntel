"""Native Hamilton implementations for export targets.

This module consolidates file-artifact export targets:
- ``export_jsonl``: JSONL export of selected dataset registry tables
- ``export_parquet``: Parquet export of selected dataset registry tables

Both targets delegate to the canonical export engine and record the shared
``datasets_manifest.json`` artifact for downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import ibis.expr.types as ir
from hamilton.function_modifiers import source, value

from codeintel.build.contracts import ArtifactSpec
from codeintel.build.exports.common import ExportCallOptions
from codeintel.build.exports.engine import ExportFormat, export_all_datasets
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import FileArtifactSaver
from codeintel.build.hamilton.materializers.artifact_saver import (
    ArtifactWritePlan,
    resolve_artifact_path,
)
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    FileArtifactRecordContext,
    record_from_file_artifact_materialization,
)
from codeintel.build.hamilton.native.target_spec_helpers import (
    TargetSpecOptions,
    make_output_target,
)
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.hamilton.run_records import (
    TargetRunRecord,
    options_hash_for_target,
    should_skip_native_target,
)
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_materialize, tag_tool
from codeintel.build.hashing import InputHashOptions, compute_input_hash
from codeintel.build.targets import TargetGraph

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, Path)

EXPORT_JSONL_TARGET_NAME = "export_jsonl"
EXPORT_PARQUET_TARGET_NAME = "export_parquet"

EXPORT_MANIFEST_FILENAME = "datasets_manifest.json"
EXPORT_JSONL_ARTIFACT_NAME = "datasets_manifest_jsonl"
EXPORT_PARQUET_ARTIFACT_NAME = "datasets_manifest_parquet"

DEFAULT_JSONL_DATASETS: tuple[str, ...] = ("modules", "function_metrics")
DEFAULT_PARQUET_DATASETS: tuple[str, ...] = ("function_metrics",)

EXPORT_JSONL_ARTIFACT_SPECS = (
    ArtifactSpec(
        EXPORT_JSONL_ARTIFACT_NAME,
        f"{{export_dir}}/{EXPORT_MANIFEST_FILENAME}",
        "Dataset manifest for JSONL exports.",
    ),
)
EXPORT_PARQUET_ARTIFACT_SPECS = (
    ArtifactSpec(
        EXPORT_PARQUET_ARTIFACT_NAME,
        f"{{export_dir}}/{EXPORT_MANIFEST_FILENAME}",
        "Dataset manifest for Parquet exports.",
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
class ExportManifestRequest:
    """Configuration for producing a dataset export manifest."""

    target_name: str
    artifact_name: str
    fmt: ExportFormat
    default_datasets: tuple[str, ...]


def _export_manifest_plan(
    env: BuildEnv,
    graph: TargetGraph,
    *,
    request: ExportManifestRequest,
) -> ArtifactWritePlan | None:
    target_name = request.target_name
    target = graph.get(target_name)
    if target is not None:
        options_hash = options_hash_for_target(env, target_name)
        hash_options = InputHashOptions(options_hash=options_hash, manifests=env.manifest_index)
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            settings=env.settings,
            options=hash_options,
        )
        if should_skip_native_target(env, target, input_hash):
            return None

    if (
        resolve_artifact_path(
            env,
            graph,
            target_name=target_name,
            artifact_name=request.artifact_name,
        )
        is None
    ):
        msg = f"Artifact path could not be resolved: {request.artifact_name}"
        raise ValueError(msg)

    export_options = load_target_options(
        env,
        target_name=target_name,
        options_type=ExportCallOptions,
    )
    if export_options.datasets is None:
        export_options = replace(export_options, datasets=list(request.default_datasets))

    def _write(output_path: Path) -> int:
        export_all_datasets(
            env.gateway,
            output_path.parent,
            fmt=request.fmt,
            settings=env.settings.export_audit,
            options=export_options,
        )
        if not output_path.exists():
            msg = f"Export manifest not written: {output_path}"
            raise ValueError(msg)
        return output_path.stat().st_size

    return ArtifactWritePlan(write_to=_write)


def _touch_dependencies(*_deps: object) -> None:
    """Touch inputs to keep DAG dependencies without extra work."""
    if not _deps:
        return


@tag_tool(domain="export", target=EXPORT_JSONL_TARGET_NAME)
def t__export_jsonl__compute(
    env: BuildEnv,
    graph: TargetGraph,
    q__core__modules: ir.Table,
    q__analytics__function_metrics: ir.Table,
) -> ArtifactWritePlan | None:
    """Compute export manifest and gather data for JSONL export.

    Returns
    -------
    ArtifactWritePlan | None
        Deferred export write plan, or None when the target is skipped.
    """
    _touch_dependencies(q__core__modules, q__analytics__function_metrics)
    return _export_manifest_plan(
        env,
        graph,
        request=ExportManifestRequest(
            target_name=EXPORT_JSONL_TARGET_NAME,
            artifact_name=EXPORT_JSONL_ARTIFACT_NAME,
            fmt="jsonl",
            default_datasets=DEFAULT_JSONL_DATASETS,
        ),
    )


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node(f"artifact.{EXPORT_JSONL_ARTIFACT_NAME}"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(EXPORT_JSONL_TARGET_NAME),
    artifact_name=value(EXPORT_JSONL_ARTIFACT_NAME),
)
@tag_compute(domain="export", target=EXPORT_JSONL_TARGET_NAME, target_="export_jsonl__content")
def export_jsonl__content(
    t__export_jsonl__compute: ArtifactWritePlan | None,
) -> ArtifactWritePlan | None:
    """Return the JSONL export write plan for materialization.

    Returns
    -------
    ArtifactWritePlan | None
        Export write plan, or None when the target is skipped.
    """
    return t__export_jsonl__compute


@tag_materialize(domain="export", target=EXPORT_JSONL_TARGET_NAME)
def t__export_jsonl(
    env: BuildEnv,
    graph: TargetGraph,
    m__artifact__datasets_manifest_jsonl: MaterializationMetadata,
) -> TargetRunRecord:
    """Write JSONL export artifacts and return record with ArtifactRef.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    context = FileArtifactRecordContext(
        env=env,
        graph=graph,
        target_name=EXPORT_JSONL_TARGET_NAME,
    )
    return record_from_file_artifact_materialization(
        context=context,
        expected_artifact_name=EXPORT_JSONL_ARTIFACT_NAME,
        materialization=m__artifact__datasets_manifest_jsonl,
    )


@tag_compute(domain="export", target=EXPORT_PARQUET_TARGET_NAME)
def t__export_parquet__compute(
    env: BuildEnv,
    graph: TargetGraph,
    q__analytics__function_metrics: ir.Table,
) -> ArtifactWritePlan | None:
    """Compute export manifest and gather data for Parquet export.

    Returns
    -------
    ArtifactWritePlan | None
        Deferred export write plan, or None when the target is skipped.
    """
    _touch_dependencies(q__analytics__function_metrics)
    return _export_manifest_plan(
        env,
        graph,
        request=ExportManifestRequest(
            target_name=EXPORT_PARQUET_TARGET_NAME,
            artifact_name=EXPORT_PARQUET_ARTIFACT_NAME,
            fmt="parquet",
            default_datasets=DEFAULT_PARQUET_DATASETS,
        ),
    )


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node(f"artifact.{EXPORT_PARQUET_ARTIFACT_NAME}"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(EXPORT_PARQUET_TARGET_NAME),
    artifact_name=value(EXPORT_PARQUET_ARTIFACT_NAME),
)
@tag_tool(domain="export", target=EXPORT_PARQUET_TARGET_NAME, target_="export_parquet__bytes")
def export_parquet__bytes(
    t__export_parquet__compute: ArtifactWritePlan | None,
) -> ArtifactWritePlan | None:
    """Return the Parquet export write plan for materialization.

    Returns
    -------
    ArtifactWritePlan | None
        Export write plan, or None when the target is skipped.
    """
    return t__export_parquet__compute


@tag_materialize(domain="export", target=EXPORT_PARQUET_TARGET_NAME)
def t__export_parquet(
    env: BuildEnv,
    graph: TargetGraph,
    m__artifact__datasets_manifest_parquet: MaterializationMetadata,
) -> TargetRunRecord:
    """Write Parquet export artifacts and return record with ArtifactRef.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    context = FileArtifactRecordContext(
        env=env,
        graph=graph,
        target_name=EXPORT_PARQUET_TARGET_NAME,
    )
    return record_from_file_artifact_materialization(
        context=context,
        expected_artifact_name=EXPORT_PARQUET_ARTIFACT_NAME,
        materialization=m__artifact__datasets_manifest_parquet,
    )


__all__ = [
    "export_jsonl__content",
    "export_parquet__bytes",
    "t__export_jsonl",
    "t__export_jsonl__compute",
    "t__export_parquet",
    "t__export_parquet__compute",
]
