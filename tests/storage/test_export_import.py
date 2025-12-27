"""Tests for storage export/import APIs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.storage.gateway import StorageConfig, open_gateway
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true
from tests._helpers.gateway import seed_contract_catalog

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import StorageGateway


def _make_gateway(db_path: Path) -> StorageGateway:
    return open_gateway(
        StorageConfig(
            db_path=db_path,
            read_only=False,
            apply_schema=False,
            ensure_views=False,
            validate_schema=False,
        ),
        seed_contract_catalog=seed_contract_catalog,
    )


def test_export_import_roundtrip(tmp_path: Path) -> None:
    """StorageGateway export/import roundtrips a simple table."""
    src_db = tmp_path / "source.duckdb"
    dst_db = tmp_path / "target.duckdb"
    export_dir = tmp_path / "export"

    source = _make_gateway(src_db)
    try:
        source.con.execute("CREATE TABLE sample (id INTEGER)")
        source.con.execute("INSERT INTO sample VALUES (1), (2)")
        source.export_database(directory=export_dir)
    finally:
        source.close()

    expect_true(export_dir.is_dir(), message="export directory exists")

    target = _make_gateway(dst_db)
    try:
        target.con.execute("DROP SCHEMA metadata CASCADE")
        target.import_database(directory=export_dir)
        row = target.con.execute("SELECT COUNT(*) FROM sample").fetchone()
    finally:
        target.close()

    expect_equal(row, (2,))
