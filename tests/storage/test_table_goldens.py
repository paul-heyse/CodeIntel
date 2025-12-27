"""Golden table regression tests for serving snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.storage.gateway import StorageConfig, open_gateway
from tests._helpers.gateway import seed_contract_catalog
from tests._helpers.goldens import assert_table_matches_golden
from tests._helpers.serving_snapshot_factory import ServingSnapshot, ServingSnapshotFactory

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def _open_snapshot_gateway(snapshot: ServingSnapshot) -> StorageGateway:
    config = StorageConfig(
        db_path=snapshot.db_path,
        read_only=False,
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    return open_gateway(config, seed_contract_catalog=seed_contract_catalog)


def test_core_modules_table_matches_golden(tmp_path: Path) -> None:
    """Ensure core.modules output stays stable for demo snapshots."""
    snapshot = ServingSnapshotFactory(tmp_path, serve_dir=tmp_path).demo_snapshot(row_count=1)
    gateway = _open_snapshot_gateway(snapshot)
    try:
        golden_path = Path("tests/fixtures/goldens/tables/core_modules.json")
        assert_table_matches_golden(gateway, "core.modules", golden_path=golden_path)
    finally:
        gateway.close()
