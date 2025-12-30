"""Tests for Iceberg metadata cache refresh helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.core.iceberg.catalog import IcebergCatalogProvider
from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.storage.iceberg.cache import refresh_iceberg_metadata_cache
from codeintel.storage.metadata.meta_catalog import meta_table_ref
from tests._helpers.assertions.expectation_assertions import (
    expect_is_not_none,
    expect_true,
)
from tests._helpers.gateway import seed_contract_catalog
from tests._helpers.serving_snapshot_factory import ServingSnapshotFactory

pytestmark = pytest.mark.no_runtime_env


def test_refresh_iceberg_metadata_cache_populates_meta_tables(tmp_path: Path) -> None:
    """Refreshing the cache should populate Iceberg metadata tables."""
    snapshot = ServingSnapshotFactory(tmp_path).demo_snapshot(row_count=1)
    provider = IcebergCatalogProvider(snapshot.iceberg_settings)
    table = provider.load_table("docs.v_demo")
    gateway = open_gateway(
        StorageConfig.for_ingest(snapshot.db_path),
        seed_contract_catalog=seed_contract_catalog,
    )
    try:
        refresh_iceberg_metadata_cache(
            gateway=gateway,
            table_key="docs.v_demo",
            table=table,
        )
        tables_ref = meta_table_ref("metadata.iceberg_tables")
        row = gateway.con.execute(
            f"SELECT current_snapshot_id, current_schema_id FROM {tables_ref} WHERE table_key = ?",
            ["docs.v_demo"],
        ).fetchone()
        row = expect_is_not_none(row, message="Expected iceberg_tables row")
        expect_true(isinstance(row[0], int), message="Expected current_snapshot_id")
        expect_true(isinstance(row[1], int), message="Expected current_schema_id")

        arrow_ref = meta_table_ref("metadata.iceberg_arrow_schema")
        arrow_row = gateway.con.execute(
            f"""
            SELECT arrow_schema_ipc, arrow_schema_json
            FROM {arrow_ref}
            WHERE table_key = ?
            LIMIT 1
            """,
            ["docs.v_demo"],
        ).fetchone()
        arrow_row = expect_is_not_none(arrow_row, message="Expected iceberg_arrow_schema row")
        expect_true(arrow_row[0] is not None, message="Expected arrow_schema_ipc")
        expect_true(arrow_row[1] is not None, message="Expected arrow_schema_json")
    finally:
        gateway.close()
