"""Golden artifact regression tests for serving snapshots."""

from __future__ import annotations

from pathlib import Path

from tests._helpers.goldens import assert_json_artifact_matches_golden
from tests._helpers.serving_snapshot_factory import ServingSnapshotFactory


def test_semantic_registry_matches_golden(tmp_path: Path) -> None:
    """Ensure semantic registry artifacts stay stable for demo snapshots."""
    snapshot = ServingSnapshotFactory(tmp_path, serve_dir=tmp_path).demo_snapshot(row_count=1)
    golden_path = Path("tests/fixtures/goldens/serving/semantic_registry.json")
    assert_json_artifact_matches_golden(
        actual_path=snapshot.registry_path,
        golden_path=golden_path,
    )
