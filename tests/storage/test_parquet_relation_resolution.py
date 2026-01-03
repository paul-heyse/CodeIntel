"""Tests for parquet-backed relation resolution."""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import pytest

from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.storage.datasets.arrow_store import ArrowDatasetWriteOptions, write_dataset
from codeintel.storage.datasets.registry import DatasetRegistry
from codeintel.storage.gateway.accessors import DuckDBGateway
from codeintel.storage.gateway.config import StorageConfig
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true


def test_relation_from_table_key_reads_parquet_manifest(tmp_path: Path) -> None:
    """Verify relation resolution reads parquet manifests when tables are absent."""
    dataset_root_dir = tmp_path / "datasets"
    snapshot_id = "snap-1"
    table_schema = TableSchema(
        schema="test",
        name="metrics",
        columns=[
            Column(name="repo", type="VARCHAR", nullable=False),
            Column(name="commit", type="VARCHAR", nullable=False),
            Column(name="value", type="BIGINT", nullable=False),
        ],
        primary_key=(),
    )
    contract = DatasetContract(
        table_key=table_schema.table_key,
        name="metrics",
        schema=table_schema,
        owner_package="core",
    )

    frame = pl.DataFrame(
        [
            {"repo": "repo-1", "commit": "commit-1", "value": 7},
        ]
    )
    options = ArrowDatasetWriteOptions(
        partition_columns=("repo", "commit"),
        schema_hash=schema_hash(table_schema),
        manifest_extras={"table_schema": table_schema.to_json_obj()},
    )
    manifest = write_dataset(
        dataset_root=dataset_root_dir,
        table_key=contract.table_key,
        snapshot_id=snapshot_id,
        data=frame.to_arrow(),
        options=options,
    )

    registry = DatasetRegistry(
        by_name={contract.name: contract},
        by_table_key={contract.table_key: contract},
        jsonl_datasets={},
        parquet_datasets={},
        dataset_root_dir=dataset_root_dir,
        dataset_manifests={contract.table_key: manifest},
    )
    con = duckdb.connect()
    config = StorageConfig(
        db_path=Path(":memory:"),
        dataset_root_dir=dataset_root_dir,
        read_only=True,
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
        attach_meta=False,
    )
    gateway = DuckDBGateway(config=config, datasets=registry, con=con)
    try:
        expect_true(
            not gateway.policy.table_exists(
                schema=table_schema.schema,
                table=table_schema.name,
            )
        )
        relation = gateway.relation_from_table_key(contract.table_key)
        rows = relation.fetchall()
        expect_equal(len(rows), 1)
        expect_equal(rows[0], ("repo-1", "commit-1", 7))
    finally:
        gateway.close()


def test_relation_from_table_key_missing_manifest_errors(tmp_path: Path) -> None:
    """Verify parquet-only relation resolution errors without manifest files."""
    dataset_root_dir = tmp_path / "datasets"
    snapshot_id = "snap-2"
    table_schema = TableSchema(
        schema="test",
        name="missing_manifest",
        columns=[
            Column(name="repo", type="VARCHAR", nullable=False),
            Column(name="commit", type="VARCHAR", nullable=False),
            Column(name="value", type="BIGINT", nullable=False),
        ],
        primary_key=(),
    )
    contract = DatasetContract(
        table_key=table_schema.table_key,
        name="missing_manifest",
        schema=table_schema,
        owner_package="core",
    )

    registry = DatasetRegistry(
        by_name={contract.name: contract},
        by_table_key={contract.table_key: contract},
        jsonl_datasets={},
        parquet_datasets={},
        dataset_root_dir=dataset_root_dir,
        dataset_manifests={},
    )
    con = duckdb.connect()
    config = StorageConfig(
        db_path=Path(":memory:"),
        dataset_root_dir=dataset_root_dir,
        read_only=True,
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
        attach_meta=False,
        commit=snapshot_id,
    )
    gateway = DuckDBGateway(config=config, datasets=registry, con=con)
    try:
        with pytest.raises(FileNotFoundError, match="Dataset manifest missing"):
            gateway.relation_from_table_key(contract.table_key)
    finally:
        gateway.close()
