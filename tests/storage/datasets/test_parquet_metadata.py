"""Tests for Parquet dataset metadata decoding."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.serving.semantic.datasets import DatasetManifestEntry, DatasetManifestIndex
from codeintel.serving.semantic.inventory import SchemaInventory
from codeintel.storage.datasets.arrow_store import ArrowDatasetWriteOptions, write_dataset
from codeintel.storage.datasets.manifests import dataset_manifest_path, read_dataset_manifest
from codeintel.storage.datasets.parquet_metadata import (
    column_types_from_metadata,
    metadata_from_schema,
    table_schema_from_dataset,
)
from codeintel.storage.datasets.paths import dataset_snapshot_dir


def _write_metadata_dataset(tmp_path: Path) -> tuple[str, str, ds.Dataset]:
    table_key = "analytics.demo"
    snapshot_id = "snap-1"
    table = pa.table({"id": [1, 2], "label": ["a", "b"]})
    metadata = {
        "codeintel.table_key": table_key,
        "codeintel.domain": "analytics",
        "codeintel.target": "demo_target",
        "codeintel.schema_hash": "demo_hash",
        "codeintel.columns_json": {"id": "INTEGER", "label": "VARCHAR"},
        "codeintel.nullability_json": {"id": False, "label": True},
        "codeintel.primary_keys_json": ["id"],
        "codeintel.partition_columns_json": [],
        "codeintel.build_id": "demo-build",
        "codeintel.repo": "demo-repo",
        "codeintel.commit": "demo-commit",
        "codeintel.snapshot_id": snapshot_id,
        "codeintel.generated_at": "2025-01-01T00:00:00Z",
        "codeintel.hamilton.node": "demo_node",
        "codeintel.hamilton.graph_version": "demo-version",
        "codeintel.inputs_json": [],
    }
    write_dataset(
        dataset_root=tmp_path,
        table_key=table_key,
        snapshot_id=snapshot_id,
        data=table,
        options=ArrowDatasetWriteOptions(
            schema_metadata=metadata,
            persist_manifest=True,
        ),
    )
    snapshot_dir = dataset_snapshot_dir(
        tmp_path,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    return table_key, snapshot_id, ds.dataset(str(snapshot_dir), format="parquet")


def test_parquet_metadata_roundtrip(tmp_path: Path) -> None:
    """Schema metadata should round-trip into a TableSchema."""
    table_key, _, dataset = _write_metadata_dataset(tmp_path)
    metadata = metadata_from_schema(dataset.schema)

    assert metadata.get("codeintel.table_key") == table_key
    assert metadata.get("codeintel.columns_json") == {"id": "INTEGER", "label": "VARCHAR"}

    table_schema = table_schema_from_dataset(dataset)
    assert table_schema is not None
    assert table_schema.table_key == table_key
    assert table_schema.primary_key == ("id",)
    assert [col.name for col in table_schema.columns] == ["id", "label"]
    assert [col.type for col in table_schema.columns] == ["INTEGER", "VARCHAR"]
    assert [col.nullable for col in table_schema.columns] == [False, True]

    types = column_types_from_metadata(dataset.schema)
    assert types == {"id": "INTEGER", "label": "VARCHAR"}


def test_schema_inventory_uses_parquet_metadata(tmp_path: Path) -> None:
    """SchemaInventory should load schemas from dataset metadata."""
    table_key, snapshot_id, _ = _write_metadata_dataset(tmp_path)
    manifest_path = dataset_manifest_path(
        dataset_root=tmp_path,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    manifest = read_dataset_manifest(manifest_path)
    entry = DatasetManifestEntry(manifest=manifest, manifest_path=manifest_path)
    inventory = SchemaInventory.from_dataset_manifests(
        DatasetManifestIndex(by_table_key={table_key: entry})
    )

    schema = inventory.require(table_key)
    assert schema.table_key == table_key
    assert schema.primary_key == ("id",)
