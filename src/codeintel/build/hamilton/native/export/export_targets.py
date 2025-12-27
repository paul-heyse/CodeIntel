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

from codeintel.build.exports.common import ExportCallOptions
from codeintel.build.exports.engine import ExportFormat, export_all_datasets
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import FileArtifactSaver
from codeintel.build.hamilton.materializers.artifact_saver import ArtifactWritePlan
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    FileArtifactRecordContext,
    record_from_file_artifact_materialization,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_tool

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, Path)

EXPORT_JSONL_TARGET_NAME = "export_jsonl"
EXPORT_PARQUET_TARGET_NAME = "export_parquet"

EXPORT_JSONL_ARTIFACT_NAME = "datasets_manifest_jsonl"
EXPORT_PARQUET_ARTIFACT_NAME = "datasets_manifest_parquet"

DEFAULT_JSONL_DATASETS: tuple[str, ...] = ("modules", "function_metrics")
DEFAULT_PARQUET_DATASETS: tuple[str, ...] = ("function_metrics",)


@dataclass(frozen=True)
class ExportManifestRequest:
    """Configuration for producing a dataset export manifest."""

    target_name: str
    artifact_name: str
    fmt: ExportFormat
    default_datasets: tuple[str, ...]


def _export_manifest_plan(
    env: BuildEnv,
    catalog: DagCatalog,
    *,
    request: ExportManifestRequest,
) -> ArtifactWritePlan | None:
    target_name = request.target_name
    _ = catalog
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
    catalog: DagCatalog,
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
        catalog,
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
    catalog=source("catalog"),
    target_name=value(EXPORT_JSONL_TARGET_NAME),
    artifact_name=value(EXPORT_JSONL_ARTIFACT_NAME),
    path_template=value("{export_dir}/datasets_manifest.json"),
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


@codeintel_target(domain="export", target=EXPORT_JSONL_TARGET_NAME)
def t__export_jsonl(
    env: BuildEnv,
    catalog: DagCatalog,
    m__artifact__datasets_manifest_jsonl: MaterializationResult,
) -> TargetRunRecord:
    """Export datasets to JSONL format for Document Output.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    context = FileArtifactRecordContext(
        env=env,
        catalog=catalog,
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
    catalog: DagCatalog,
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
        catalog,
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
    catalog=source("catalog"),
    target_name=value(EXPORT_PARQUET_TARGET_NAME),
    artifact_name=value(EXPORT_PARQUET_ARTIFACT_NAME),
    path_template=value("{export_dir}/datasets_manifest.json"),
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


@codeintel_target(domain="export", target=EXPORT_PARQUET_TARGET_NAME)
def t__export_parquet(
    env: BuildEnv,
    catalog: DagCatalog,
    m__artifact__datasets_manifest_parquet: MaterializationResult,
) -> TargetRunRecord:
    """Export datasets to Parquet format for Document Output.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    context = FileArtifactRecordContext(
        env=env,
        catalog=catalog,
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
