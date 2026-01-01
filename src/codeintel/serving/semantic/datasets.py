"""Arrow dataset manifest helpers for serving engines."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.serving.semantic.filter_compiler import (
    FilterCompilerError,
    arrow_filter_expression,
    compile_filter_predicates,
)
from codeintel.storage.datasets.contracts import (
    DatasetTuningMetadata,
    WriteSettingsPayload,
    inferred_settings_from_manifest,
    tuning_metadata_from_manifest,
    write_settings_from_manifest,
)
from codeintel.storage.datasets.manifests import read_dataset_manifest
from codeintel.storage.datasets.parquet_metadata import metadata_from_schema
from codeintel.storage.tracking.schema_catalog_models import DerivedSettingsPayload

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.core.manifests import ArrowDatasetManifest, ServingSnapshotManifest
    from codeintel.core.schemas.primitives import ColumnType
    from codeintel.serving.semantic.models import FilterSpec

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DatasetManifestEntry:
    """Dataset manifest plus its on-disk location."""

    manifest: ArrowDatasetManifest
    manifest_path: Path

    @property
    def dataset_dir(self) -> Path:
        """Return the dataset directory for this manifest.

        Returns
        -------
        pathlib.Path
            Directory containing the dataset files.
        """
        return self.manifest_path.parent

    @property
    def inferred_settings(self) -> DerivedSettingsPayload | None:
        """Return inferred tuning settings from the manifest.

        Returns
        -------
        DerivedSettingsPayload | None
            Inferred settings payload, or None when absent.
        """
        return inferred_settings_from_manifest(self.manifest)

    @property
    def write_settings(self) -> WriteSettingsPayload | None:
        """Return persisted write settings from the manifest.

        Returns
        -------
        WriteSettingsPayload | None
            Write settings payload, or None when absent.
        """
        return write_settings_from_manifest(self.manifest)

    @property
    def tuning_metadata(self) -> DatasetTuningMetadata | None:
        """Return tuning metadata parsed from the manifest.

        Returns
        -------
        DatasetTuningMetadata | None
            Parsed tuning metadata, or None when no settings are present.
        """
        return tuning_metadata_from_manifest(self.manifest)


@dataclass(frozen=True, slots=True)
class DatasetManifestIndex:
    """Index of dataset manifests keyed by table_key."""

    by_table_key: Mapping[str, DatasetManifestEntry]

    def get(self, table_key: str) -> DatasetManifestEntry | None:
        """Return the dataset manifest entry for a table key, if present.

        Returns
        -------
        DatasetManifestEntry | None
            Manifest entry for the table key, if registered.
        """
        return self.by_table_key.get(table_key)

    def table_keys(self) -> tuple[str, ...]:
        """Return all table keys with dataset manifests.

        Returns
        -------
        tuple[str, ...]
            Table keys backed by dataset manifests.
        """
        return tuple(self.by_table_key.keys())


@dataclass(frozen=True, slots=True)
class DatasetScannerOptions:
    """Options for streaming dataset scans."""

    batch_size: int
    fragment_readahead: int | None = None
    filter_expression: ds.Expression | None = None
    metrics_enabled: bool = False
    schema: pa.Schema | None = None
    columns: Sequence[str] | None = None


def dataset_for_entry(entry: DatasetManifestEntry) -> ds.Dataset:
    """Return a PyArrow dataset for a manifest entry.

    Returns
    -------
    pyarrow.dataset.Dataset
        Dataset handle for the manifest entry.
    """
    partitioning: str | None = "hive" if entry.manifest.partition_columns else None
    if entry.manifest.files:
        paths = [str(entry.dataset_dir / path) for path in entry.manifest.files]
        return ds.dataset(paths, format="parquet", partitioning=partitioning)
    return ds.dataset(str(entry.dataset_dir), format="parquet", partitioning=partitioning)


def dataset_schema_for_entry(entry: DatasetManifestEntry) -> pa.Schema | None:
    """Return the Arrow schema for a dataset entry.

    Returns
    -------
    pyarrow.Schema | None
        Arrow schema for the dataset entry, or None if it cannot be read.
    """
    try:
        dataset = dataset_for_entry(entry)
    except (OSError, pa.ArrowInvalid, ValueError):
        return None
    return dataset.schema


def dataset_metadata_for_entry(entry: DatasetManifestEntry) -> dict[str, object]:
    """Return decoded schema metadata for a dataset entry.

    Returns
    -------
    dict[str, object]
        Metadata dictionary decoded from the dataset schema.
    """
    schema = dataset_schema_for_entry(entry)
    if schema is None:
        return {}
    return metadata_from_schema(schema)


def dataset_filter_expression(
    *,
    filters: list[FilterSpec],
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None = None,
) -> ds.Expression | None:
    """Build a dataset filter expression for pushdown.

    Parameters
    ----------
    filters
        Filter specs to translate into dataset expressions.
    allowed_columns
        Allowed columns for filtering.
    column_types
        Optional column type mapping for operator validation.

    Returns
    -------
    pyarrow.dataset.Expression | None
        Filter expression or None when unsupported.
    """
    if not filters:
        return None
    try:
        predicates = compile_filter_predicates(
            filters,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    except FilterCompilerError as exc:
        LOG.debug("Dataset filter compilation failed: %s", exc)
        return None
    return arrow_filter_expression(predicates)


def dataset_scanner_for_entry(
    entry: DatasetManifestEntry,
    *,
    options: DatasetScannerOptions,
) -> ds.Scanner:
    """Return a dataset scanner configured for streaming reads.

    Returns
    -------
    pyarrow.dataset.Scanner
        Scanner configured for batched streaming reads.
    """
    tuned_options = apply_tuning_options(entry, options=options)
    start = perf_counter()
    dataset = dataset_for_entry(entry)
    if tuned_options.metrics_enabled:
        _log_scan_metrics(
            entry,
            dataset=dataset,
            filter_expression=tuned_options.filter_expression,
            duration_ms=(perf_counter() - start) * 1000,
            memory_bytes=_total_allocated_bytes(),
        )
    scan_kwargs: dict[str, object] = {"batch_size": tuned_options.batch_size}
    if tuned_options.fragment_readahead is not None:
        scan_kwargs["fragment_readahead"] = tuned_options.fragment_readahead
    if tuned_options.columns is not None:
        scan_kwargs["columns"] = list(tuned_options.columns)
    return _build_scanner(
        dataset,
        filter_expression=tuned_options.filter_expression,
        scan_kwargs=scan_kwargs,
        schema=tuned_options.schema,
    )


def apply_tuning_options(
    entry: DatasetManifestEntry,
    *,
    options: DatasetScannerOptions,
) -> DatasetScannerOptions:
    """Apply dataset tuning metadata to scanner options.

    Parameters
    ----------
    entry
        Dataset manifest entry with tuning metadata.
    options
        Base scanner options to adjust.

    Returns
    -------
    DatasetScannerOptions
        Scanner options updated with tuning overrides when present.
    """
    batch_size = _tuned_batch_size(entry, default=options.batch_size)
    return DatasetScannerOptions(
        batch_size=batch_size,
        fragment_readahead=options.fragment_readahead,
        filter_expression=options.filter_expression,
        metrics_enabled=options.metrics_enabled,
        schema=options.schema,
        columns=options.columns,
    )


def _tuned_batch_size(entry: DatasetManifestEntry, *, default: int) -> int:
    tuning = entry.tuning_metadata
    if tuning is None:
        return default
    candidates = [
        tuning.write_settings,
        tuning.inferred_settings,
    ]
    for payload in candidates:
        if not payload:
            continue
        row_group_size = _coerce_int(payload.get("row_group_size"))
        if row_group_size is not None and row_group_size > 0:
            return min(default, row_group_size)
    return default


def load_dataset_manifests(
    snapshot_manifest: ServingSnapshotManifest,
) -> DatasetManifestIndex:
    """Load dataset manifests from a snapshot manifest.

    Returns
    -------
    DatasetManifestIndex
        Loaded dataset manifest index keyed by table key.

    Raises
    ------
    ValueError
        If manifest metadata is inconsistent with the snapshot manifest.
    """
    by_table: dict[str, DatasetManifestEntry] = {}
    for table_key, entry in snapshot_manifest.datasets.items():
        manifest_path = Path(entry.manifest_path)
        manifest = read_dataset_manifest(manifest_path)
        if manifest.table_key != table_key:
            msg = f"Dataset manifest table_key mismatch: {table_key} != {manifest.table_key}"
            raise ValueError(msg)
        if entry.schema_hash is None:
            msg = f"Snapshot dataset entry missing schema_hash for {table_key}"
            raise ValueError(msg)
        if manifest.schema_hash is None:
            msg = f"Dataset manifest missing schema_hash for {table_key}"
            raise ValueError(msg)
        if entry.schema_hash != manifest.schema_hash:
            msg = (
                "Dataset manifest schema hash mismatch for "
                f"{table_key}: {entry.schema_hash} != {manifest.schema_hash}"
            )
            raise ValueError(msg)
        by_table[table_key] = DatasetManifestEntry(
            manifest=manifest,
            manifest_path=manifest_path,
        )
    return DatasetManifestIndex(by_table_key=by_table)


def _log_scan_metrics(
    entry: DatasetManifestEntry,
    *,
    dataset: ds.Dataset,
    filter_expression: ds.Expression | None,
    duration_ms: float,
    memory_bytes: int | None,
) -> None:
    stats = entry.manifest.stats or {}
    row_groups = stats.get("row_groups")
    row_count = entry.manifest.row_count or stats.get("rows_from_metadata")
    total_bytes = stats.get("total_bytes")
    total_fragments = _count_fragments(dataset, filter_expression=None)
    filtered_fragments = (
        _count_fragments(dataset, filter_expression=filter_expression)
        if filter_expression is not None
        else None
    )
    LOG.info(
        "dataset_scan_metrics table=%s files=%s fragments=%s fragments_filtered=%s "
        "row_groups=%s rows=%s bytes=%s duration_ms=%.2f memory_bytes=%s",
        entry.manifest.table_key,
        len(entry.manifest.files),
        total_fragments,
        filtered_fragments,
        row_groups,
        row_count,
        total_bytes,
        duration_ms,
        memory_bytes,
    )


def _count_fragments(
    dataset: ds.Dataset,
    *,
    filter_expression: ds.Expression | None,
) -> int | None:
    get_fragments = getattr(dataset, "get_fragments", None)
    if not callable(get_fragments):
        return None
    try:
        fragments = (
            get_fragments(filter=filter_expression)
            if filter_expression is not None
            else get_fragments()
        )
        if not isinstance(fragments, Iterable):
            return None
        return sum(1 for _ in fragments)
    except (TypeError, ValueError, pa.ArrowInvalid):
        return None


def _total_allocated_bytes() -> int | None:
    total_fn = getattr(pa, "total_allocated_bytes", None)
    if not callable(total_fn):
        return None
    try:
        total = total_fn()
    except (TypeError, ValueError, pa.ArrowInvalid):
        return None
    return _coerce_int(total)


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return _coerce_int_from_str(value)
    return _coerce_int_from_intlike(value)


def _coerce_int_from_str(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _coerce_int_from_intlike(value: object) -> int | None:
    converter = getattr(value, "__int__", None)
    if not callable(converter):
        return None
    try:
        converted = converter()
    except (TypeError, ValueError):
        return None
    if isinstance(converted, bool):
        return None
    if isinstance(converted, int):
        return converted
    return None


def _build_scanner(
    dataset: ds.Dataset,
    *,
    filter_expression: ds.Expression | None,
    scan_kwargs: Mapping[str, object],
    schema: pa.Schema | None,
) -> ds.Scanner:
    resolved_schema = schema or dataset.schema
    scan_kwargs = dict(scan_kwargs)
    if schema is not None:
        scan_kwargs["schema"] = resolved_schema
    if filter_expression is None:
        return _scanner_with_schema(dataset, scan_kwargs)
    fragments = _fragments_for_filter(dataset, filter_expression)
    from_fragments = getattr(ds.Scanner, "from_fragments", None)
    if fragments is not None and callable(from_fragments):
        try:
            return from_fragments(fragments, schema=resolved_schema, **scan_kwargs)
        except (TypeError, ValueError, pa.ArrowInvalid):
            pass
    scan_kwargs["filter"] = filter_expression
    return _scanner_with_schema(dataset, scan_kwargs)


def _scanner_with_schema(dataset: ds.Dataset, scan_kwargs: dict[str, object]) -> ds.Scanner:
    try:
        return dataset.scanner(**scan_kwargs)
    except TypeError:
        scan_kwargs.pop("schema", None)
        return dataset.scanner(**scan_kwargs)


def _fragments_for_filter(
    dataset: ds.Dataset,
    filter_expression: ds.Expression,
) -> tuple[ds.Fragment, ...] | None:
    get_fragments = getattr(dataset, "get_fragments", None)
    if not callable(get_fragments):
        return None
    try:
        fragments = get_fragments(filter=filter_expression)
        if not isinstance(fragments, Iterable):
            return None
        return tuple(fragments)
    except (TypeError, ValueError, pa.ArrowInvalid):
        return None


__all__ = [
    "DatasetManifestEntry",
    "DatasetManifestIndex",
    "DatasetScannerOptions",
    "apply_tuning_options",
    "dataset_filter_expression",
    "dataset_for_entry",
    "dataset_scanner_for_entry",
    "load_dataset_manifests",
]
