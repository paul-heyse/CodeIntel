"""Integration test for parquet-backed build exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import duckdb
import pyarrow as pa
import pytest

from codeintel.build.exports.engine import export_jsonl_for_table
from codeintel.core.config.settings import ExportAuditSettings
from codeintel.core.datasets.arrow_store import write_dataset
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.storage.datasets.registry import DatasetRegistry
from codeintel.storage.gateway.accessors import DuckDBGateway
from codeintel.storage.gateway.config import StorageConfig
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true

pytestmark = pytest.mark.no_runtime_env


def _test_contract() -> DatasetContract:
    table_schema = TableSchema(
        schema="test",
        name="export_metrics",
        columns=[
            Column(name="repo", type="VARCHAR", nullable=False),
            Column(name="commit", type="VARCHAR", nullable=False),
            Column(name="value", type="BIGINT", nullable=False),
        ],
        primary_key=(),
    )
    return DatasetContract(
        table_key=table_schema.table_key,
        name="export_metrics",
        schema=table_schema,
        owner_package="core",
    )


def test_export_jsonl_reads_parquet_manifest(tmp_path: Path) -> None:
    """Export JSONL via parquet-backed relation when no DuckDB table exists."""
    dataset_root_dir = tmp_path / "datasets"
    snapshot_id = "snap-1"
    contract = _test_contract()
    table_schema = cast("TableSchema", contract.schema)
    table = pa.table(
        {
            "repo": ["repo-1"],
            "commit": ["commit-1"],
            "value": [3],
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
        jsonl_datasets={contract.name: "export_metrics.jsonl"},
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
    try:
        expect_true(
            not gateway.policy.table_exists(
                schema=table_schema.schema,
                table=table_schema.name,
            )
        )
        output_path = tmp_path / "export.jsonl"
        rows_written = export_jsonl_for_table(
            gateway,
            contract.table_key,
            output_path,
            ExportAuditSettings(),
        )
        expect_equal(rows_written, 1)
        expect_true(output_path.is_file())
        payloads = [
            json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
        ]
        expect_equal(payloads, [{"repo": "repo-1", "commit": "commit-1", "value": 3}])
    finally:
        gateway.close()
