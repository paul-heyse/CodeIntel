"""Integration tests for parquet-backed warehouse reads."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import duckdb
import pyarrow as pa
import pytest

from codeintel.core.datasets.arrow_store import write_dataset
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.storage.datasets.registry import DatasetRegistry
from codeintel.storage.gateway.accessors import DuckDBGateway
from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.warehouse import Warehouse
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true

pytestmark = pytest.mark.no_runtime_env


def _test_contract() -> DatasetContract:
    table_schema = TableSchema(
        schema="test",
        name="warehouse_metrics",
        columns=[
            Column(name="repo", type="VARCHAR", nullable=False),
            Column(name="commit", type="VARCHAR", nullable=False),
            Column(name="value", type="BIGINT", nullable=False),
        ],
        primary_key=(),
    )
    return DatasetContract(
        table_key=table_schema.table_key,
        name="warehouse_metrics",
        schema=table_schema,
        owner_package="core",
    )


def test_warehouse_reads_parquet_without_duckdb_tables(tmp_path: Path) -> None:
    """Warehouse read/exists/count should work from parquet datasets."""
    dataset_root_dir = tmp_path / "datasets"
    snapshot_id = "snap-1"
    contract = _test_contract()
    table_schema = cast("TableSchema", contract.schema)
    table = pa.table(
        {
            "repo": ["repo-1"],
            "commit": ["commit-1"],
            "value": [11],
        }
    )
    manifest = write_dataset(
        dataset_root=dataset_root_dir,
        table_key=contract.table_key,
        snapshot_id=snapshot_id,
        data=table,
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
        commit=snapshot_id,
    )
    gateway = DuckDBGateway(config=config, datasets=registry, con=con)
    warehouse = Warehouse(gateway=gateway)
    try:
        expect_true(
            not gateway.policy.table_exists(
                schema=table_schema.schema,
                table=table_schema.name,
            )
        )
        relation = warehouse.read(contract.table_key)
        expect_equal(relation.fetchall(), [("repo-1", "commit-1", 11)])
        expect_true(warehouse.exists(contract.table_key))
        expect_equal(warehouse.count(contract.table_key), 1)
    finally:
        gateway.close()
