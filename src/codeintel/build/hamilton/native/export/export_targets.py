"""Native Hamilton implementations for export targets.

This module consolidates file-artifact export targets:
- ``export_jsonl``: JSONL export of selected analytics datasets
- ``export_parquet``: Parquet export of selected analytics datasets

Both targets use DAG-visible artifact I/O via ``FileArtifactSaver``.
"""

from __future__ import annotations

import json
import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, SupportsInt, TextIO, cast, runtime_checkable

import ibis
import ibis.expr.types as ir
from hamilton.function_modifiers import check_output_custom, schema, source, value

from codeintel.build.contracts import ArtifactSpec
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import FileArtifactSaver
from codeintel.build.hamilton.materializers.artifact_saver import (
    ArtifactWritePlan,
    resolve_artifact_path,
)
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.ibis_helpers import filter_for_snapshot
from codeintel.build.hamilton.native.materialization_records import (
    record_from_file_artifact_materialization,
)
from codeintel.build.hamilton.native.target_spec_helpers import (
    TargetSpecOptions,
    make_output_target,
)
from codeintel.build.hamilton.run_records import TargetRunRecord, should_skip_native_target
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_materialize, tag_tool
from codeintel.build.hamilton.validators import build_table_contract
from codeintel.build.hashing import compute_input_hash
from codeintel.build.targets import TargetGraph

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ir.Table, Path)

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


@runtime_checkable
class _SupportsIsoformat(Protocol):
    def isoformat(self) -> str: ...


class _RecordBatch(Protocol):
    num_rows: int

    def to_pydict(self) -> dict[str, list[object]]: ...


class _DuckDBRelation(Protocol):
    def fetch_record_batch(self, rows_per_batch: SupportsInt = 1_000_000) -> object: ...


if TYPE_CHECKING:
    from collections.abc import Iterable


def _default_json_serializer(obj: object) -> object:
    if isinstance(obj, _SupportsIsoformat):
        return obj.isoformat()
    msg = f"Type {type(obj)} is not JSON serializable"
    raise TypeError(msg)


def _write_jsonl_records(
    handle: TextIO,
    *,
    rel: _DuckDBRelation,
    record_type: str,
) -> int:
    rows_written = 0
    reader = rel.fetch_record_batch(10_000)
    for batch in cast("Iterable[_RecordBatch]", reader):
        payload = batch.to_pydict()
        columns = list(payload.keys())
        for idx in range(batch.num_rows):
            record = {name: payload[name][idx] for name in columns}
            record["_type"] = record_type
            handle.write(json.dumps(record, ensure_ascii=False, default=_default_json_serializer))
            handle.write("\n")
            rows_written += 1
    return rows_written


def _write_export_jsonl(
    output_path: Path,
    *,
    env: BuildEnv,
    modules: ir.Table,
    function_metrics: ir.Table,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    modules_rel = env.gateway.con.sql(ibis.to_sql(modules, dialect="duckdb"))
    metrics_rel = env.gateway.con.sql(ibis.to_sql(function_metrics, dialect="duckdb"))

    modules_count_row = modules_rel.aggregate("count(*)").fetchone()
    metrics_count_row = metrics_rel.aggregate("count(*)").fetchone()
    modules_count = int(modules_count_row[0]) if modules_count_row else 0
    metrics_count = int(metrics_count_row[0]) if metrics_count_row else 0

    metadata = {
        "repo": env.snapshot.repo,
        "commit": env.snapshot.commit,
        "export_format": "jsonl",
        "modules_count": modules_count,
        "function_metrics_count": metrics_count,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"_metadata": metadata}, ensure_ascii=False))
        handle.write("\n")
        _ = _write_jsonl_records(handle, rel=modules_rel, record_type="module")
        _ = _write_jsonl_records(handle, rel=metrics_rel, record_type="function_metric")

    return output_path.stat().st_size


def _write_export_parquet(
    output_path: Path,
    *,
    env: BuildEnv,
    expr: ir.Table,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rel = env.gateway.con.sql(ibis.to_sql(expr, dialect="duckdb"))
    write_parquet = getattr(rel, "write_parquet", None)
    if write_parquet is None:
        msg = "DuckDB relation does not support write_parquet()"
        raise RuntimeError(msg)
    write_parquet(str(output_path))
    return output_path.stat().st_size


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

    Raises
    ------
    ValueError
        If the artifact path cannot be resolved from the target contract.
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

    modules = filter_for_snapshot(q__core__modules, env.snapshot)
    function_metrics = filter_for_snapshot(q__analytics__function_metrics, env.snapshot)
    if (
        resolve_artifact_path(
            env,
            graph,
            target_name=EXPORT_JSONL_TARGET_NAME,
            artifact_name=JSONL_EXPORT_ARTIFACT_NAME,
        )
        is None
    ):
        msg = f"Artifact path could not be resolved: {JSONL_EXPORT_ARTIFACT_NAME}"
        raise ValueError(msg)

    return ArtifactWritePlan(
        write_to=partial(
            _write_export_jsonl,
            env=env,
            modules=modules,
            function_metrics=function_metrics,
        )
    )


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node(f"artifact.{JSONL_EXPORT_ARTIFACT_NAME}"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(EXPORT_JSONL_TARGET_NAME),
    artifact_name=value(JSONL_EXPORT_ARTIFACT_NAME),
)
@tag_compute(domain="export", target=EXPORT_JSONL_TARGET_NAME, target_="export_jsonl__content")
def export_jsonl__content(t__export_jsonl__compute: ArtifactWritePlan | None) -> ArtifactWritePlan | None:
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
    m__artifact__jsonl_export: dict[str, object],
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


@tag_compute(domain="export", target=EXPORT_PARQUET_TARGET_NAME)
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
    return filter_for_snapshot(q__analytics__function_metrics, env.snapshot)


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node(f"artifact.{PARQUET_EXPORT_ARTIFACT_NAME}"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(EXPORT_PARQUET_TARGET_NAME),
    artifact_name=value(PARQUET_EXPORT_ARTIFACT_NAME),
)
@tag_tool(domain="export", target=EXPORT_PARQUET_TARGET_NAME, target_="export_parquet__bytes")
def export_parquet__bytes(
    env: BuildEnv,
    graph: TargetGraph,
    t__export_parquet__compute: ir.Table,
) -> ArtifactWritePlan | None:
    """Serialize the Parquet export payload for file materialization.

    Returns
    -------
    ArtifactWritePlan | None
        Deferred Parquet write plan, or None when the target is skipped.

    Raises
    ------
    ValueError
        If the artifact path cannot be resolved from the target contract.
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

    if (
        resolve_artifact_path(
            env,
            graph,
            target_name=EXPORT_PARQUET_TARGET_NAME,
            artifact_name=PARQUET_EXPORT_ARTIFACT_NAME,
        )
        is None
    ):
        msg = f"Artifact path could not be resolved: {PARQUET_EXPORT_ARTIFACT_NAME}"
        raise ValueError(msg)

    return ArtifactWritePlan(
        write_to=partial(
            _write_export_parquet,
            env=env,
            expr=t__export_parquet__compute,
        )
    )


@tag_materialize(domain="export", target=EXPORT_PARQUET_TARGET_NAME)
def t__export_parquet(
    env: BuildEnv,
    graph: TargetGraph,
    m__artifact__parquet_export: dict[str, object],
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
    "export_jsonl__content",
    "export_parquet__bytes",
    "t__export_jsonl",
    "t__export_jsonl__compute",
    "t__export_parquet",
    "t__export_parquet__compute",
]
