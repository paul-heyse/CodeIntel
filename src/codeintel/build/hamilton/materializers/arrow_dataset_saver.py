"""Arrow dataset saver for Hamilton materialization."""

from __future__ import annotations

import shutil
import types
import typing
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Literal, cast, get_args, get_origin

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
from hamilton.io.data_adapters import DataSaver
from polars.exceptions import PolarsError

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers.base import (
    MaterializationContextError,
    duration_ms,
    resolve_materialization_context,
)
from codeintel.build.schemas import get_schema_provider
from codeintel.core.execution.materialization import failed_table_result, succeeded_table_result
from codeintel.core.schemas.arrow_polars import (
    table_schema_from_arrow_schema,
    table_schema_from_polars_dataframe,
    table_schema_from_polars_lazyframe,
)
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.primitives import TableSchema
from codeintel.storage.datasets.arrow_store import (
    ArrowDatasetManifestRequest,
    ArrowDatasetWriteOptions,
    build_dataset_manifest,
    write_dataset,
)
from codeintel.storage.datasets.manifests import dataset_manifest_path, write_dataset_manifest
from codeintel.storage.datasets.paths import dataset_snapshot_dir

if TYPE_CHECKING:
    from pathlib import Path

    from pyarrow import RecordBatchReader, Table

    from codeintel.core.manifests import ArrowDatasetManifest

    type ArrowDatasetInput = Table | RecordBatchReader
else:
    type ArrowDatasetInput = object

type TabularData = ArrowDatasetInput | pl.DataFrame | pl.LazyFrame

_RECOVERABLE_EXCEPTIONS = (
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    OSError,
    pa.ArrowInvalid,
    PolarsError,
)

_TABULAR_TYPES: tuple[type, ...] = (
    pa.Table,
    pa.RecordBatchReader,
    pl.DataFrame,
    pl.LazyFrame,
)

_DEFAULT_PARTITION_COLUMNS: tuple[str, ...] = ("repo", "commit")


@dataclass(frozen=True)
class ArrowDatasetSaver(DataSaver):
    """Persist tabular outputs as Arrow datasets with manifest metadata."""

    env: BuildEnv
    catalog: DagCatalog
    target_name: str
    table_key: str
    partition_columns: tuple[str, ...] = ()
    output_role: Literal["contract", "internal"] | None = None

    @classmethod
    def name(cls) -> str:
        """Return a stable name for this saver adapter.

        Returns
        -------
        str
            Stable saver identifier.
        """
        return "codeintel.arrow_dataset"

    @classmethod
    def applicable_types(cls) -> list[type]:
        """Return types this saver can persist.

        Returns
        -------
        list[type]
            Supported output types.
        """
        return list(_TABULAR_TYPES)

    @classmethod
    def applies_to(cls, type_: type) -> bool:
        """Return True when this saver can handle the Hamilton node output type.

        Returns
        -------
        bool
            True when the saver applies to the provided type.
        """
        origin = get_origin(type_)
        if origin in {types.UnionType, typing.Union}:
            args = set(get_args(type_))
            if args.issubset(set(_TABULAR_TYPES) | {type(None)}):
                return True
        return super().applies_to(type_)

    def save_data(self, data: object) -> dict[str, object]:
        """Save the provided data and return metadata describing the write.

        Returns
        -------
        dict[str, object]
            Materialization metadata mapping.
        """
        start = perf_counter()
        input_hash: str | None = None
        result: MaterializationResult | None = None

        try:
            prepared = resolve_materialization_context(
                env=self.env,
                catalog=self.catalog,
                target_name=self.target_name,
            )
            if isinstance(prepared, MaterializationContextError):
                result = failed_table_result(
                    table_key=self.table_key,
                    duration_ms=duration_ms(start),
                    input_hash=prepared.input_hash or "",
                    error=prepared.message,
                )
            else:
                context = prepared
                input_hash = context.input_hash
                if data is None:
                    result = failed_table_result(
                        table_key=self.table_key,
                        duration_ms=duration_ms(start),
                        input_hash=input_hash or "",
                        error="Expected tabular data but received None",
                    )
                else:
                    manifest, manifest_path = _materialize_dataset(
                        env=self.env,
                        table_key=self.table_key,
                        data=cast("TabularData", data),
                        partition_columns=self.partition_columns,
                    )
                    row_count = manifest.row_count if manifest.row_count is not None else 0
                    result = succeeded_table_result(
                        table_key=self.table_key,
                        duration_ms=duration_ms(start),
                        input_hash=input_hash or "",
                        row_count=row_count,
                        dataset_manifest_path=str(manifest_path),
                    )
        except _RECOVERABLE_EXCEPTIONS as exc:
            result = failed_table_result(
                table_key=self.table_key,
                duration_ms=duration_ms(start),
                input_hash=input_hash or "",
                error=str(exc),
            )

        if result is None:
            result = failed_table_result(
                table_key=self.table_key,
                duration_ms=duration_ms(start),
                input_hash=input_hash or "",
                error="Unknown Arrow dataset materialization failure",
            )
        return result.to_mapping()


def _materialize_dataset(
    *,
    env: BuildEnv,
    table_key: str,
    data: TabularData,
    partition_columns: tuple[str, ...],
) -> tuple[ArrowDatasetManifest, Path]:
    table_schema = _table_schema_for_data(table_key=table_key, data=data)
    resolved_partitions = _resolve_partition_columns(
        table_schema=table_schema,
        requested=partition_columns,
    )
    schema_hash_value = schema_hash(table_schema)
    extras = _manifest_extras(table_schema=table_schema, table_key=table_key)
    options = ArrowDatasetWriteOptions(
        partition_columns=resolved_partitions,
        schema_hash=schema_hash_value,
        manifest_extras=extras,
    )
    snapshot_id = _snapshot_id(env)
    dataset_root = env.paths.dataset_root_dir

    if isinstance(data, pl.LazyFrame):
        manifest = _write_lazyframe_dataset(
            data=data,
            dataset_root=dataset_root,
            table_key=table_key,
            snapshot_id=snapshot_id,
            options=options,
        )
        manifest_path = dataset_manifest_path(
            dataset_root=dataset_root,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
        return manifest, manifest_path

    arrow_input = _coerce_arrow_input(data)
    manifest = write_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        data=arrow_input,
        options=options,
    )
    manifest_path = dataset_manifest_path(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    return manifest, manifest_path


def _write_lazyframe_dataset(
    *,
    data: pl.LazyFrame,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    options: ArrowDatasetWriteOptions,
) -> ArrowDatasetManifest:
    snapshot_dir = dataset_snapshot_dir(
        dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    _prepare_snapshot_dir(snapshot_dir, behavior=options.existing_data_behavior)
    partition_by = list(options.partition_columns) if options.partition_columns else None
    if partition_by:
        frame = data.collect()
        frame.write_parquet(
            str(snapshot_dir),
            partition_by=partition_by,
            mkdir=True,
        )
    else:
        data.sink_parquet(str(snapshot_dir / "data.parquet"))
    dataset = ds.dataset(str(snapshot_dir), format="parquet")
    request = ArrowDatasetManifestRequest(
        table_key=table_key,
        snapshot_id=snapshot_id,
        partition_columns=options.partition_columns,
        schema_hash=options.schema_hash,
        extras=options.manifest_extras,
    )
    manifest = build_dataset_manifest(
        dataset=dataset,
        snapshot_dir=snapshot_dir,
        request=request,
    )
    if options.persist_manifest:
        path = dataset_manifest_path(
            dataset_root=dataset_root,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
        write_dataset_manifest(path, manifest)
    return manifest


def _prepare_snapshot_dir(snapshot_dir: Path, *, behavior: object) -> None:
    if snapshot_dir.exists():
        if behavior == "error":
            msg = f"Dataset snapshot already exists: {snapshot_dir}"
            raise FileExistsError(msg)
        if behavior in {"delete_matching", "overwrite_or_ignore"}:
            shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)


def _table_schema_for_data(*, table_key: str, data: TabularData) -> TableSchema:
    if isinstance(data, pl.LazyFrame):
        return table_schema_from_polars_lazyframe(frame=data, table_key=table_key)
    if isinstance(data, pl.DataFrame):
        return table_schema_from_polars_dataframe(frame=data, table_key=table_key)
    if isinstance(data, pa.Table):
        arrow_table = cast("Table", data)
        return table_schema_from_arrow_schema(arrow_schema=arrow_table.schema, table_key=table_key)
    if isinstance(data, pa.RecordBatchReader):
        arrow_reader = cast("RecordBatchReader", data)
        return table_schema_from_arrow_schema(
            arrow_schema=arrow_reader.schema,
            table_key=table_key,
        )
    msg = f"Unsupported Arrow dataset input type: {type(data).__name__}"
    raise TypeError(msg)


def _resolve_partition_columns(
    *, table_schema: TableSchema, requested: tuple[str, ...]
) -> tuple[str, ...]:
    if requested:
        _validate_partition_columns(table_schema, requested)
        return requested
    return tuple(
        column for column in _DEFAULT_PARTITION_COLUMNS if column in table_schema.column_names()
    )


def _validate_partition_columns(table_schema: TableSchema, columns: tuple[str, ...]) -> None:
    if not columns:
        return
    column_set = set(table_schema.column_names())
    missing = [column for column in columns if column not in column_set]
    if missing:
        msg = f"Partition columns missing from {table_schema.table_key}: {missing}"
        raise ValueError(msg)


def _manifest_extras(*, table_schema: TableSchema, table_key: str) -> dict[str, object]:
    extras: dict[str, object] = {"table_schema": table_schema.to_json_obj()}
    provenance = _schema_provenance(table_key)
    if provenance:
        extras["provenance"] = provenance
    return extras


def _schema_provenance(table_key: str) -> dict[str, str] | None:
    provider = get_schema_provider()
    derivation_fn = getattr(provider, "derivation", None)
    if not callable(derivation_fn):
        return None
    derivation = derivation_fn(table_key)
    if derivation is None:
        return None
    source_kind = getattr(derivation, "source_kind", None)
    source_ref = getattr(derivation, "source_ref", None)
    if isinstance(source_kind, str) and isinstance(source_ref, str):
        return {"derivation_kind": source_kind, "derivation_source": source_ref}
    return None


def _snapshot_id(env: BuildEnv) -> str:
    snapshot_id = env.snapshot.commit.strip()
    if snapshot_id:
        return snapshot_id
    msg = "Snapshot commit is required for Arrow dataset materialization"
    raise ValueError(msg)


def _coerce_arrow_input(data: TabularData) -> ArrowDatasetInput:
    if isinstance(data, pa.Table):
        return cast("Table", data)
    if isinstance(data, pa.RecordBatchReader):
        return cast("RecordBatchReader", data)
    if isinstance(data, pl.DataFrame):
        return data.to_arrow()
    msg = f"Unsupported Arrow dataset input type: {type(data).__name__}"
    raise TypeError(msg)


__all__ = ["ArrowDatasetSaver"]
