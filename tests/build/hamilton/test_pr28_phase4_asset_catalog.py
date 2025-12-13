"""Phase 4 asset catalog: schema + persistence smoke tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.storage.gateway import open_memory_gateway
from codeintel.storage.tracking.asset_tracking import (
    AssetAliasRecord,
    AssetDiffRecord,
    AssetLineageEdgeRecord,
    AssetVersionRecord,
    RunAssetVersionRecord,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_length,
    expect_true,
)


def test_pr28_phase4_asset_catalog_tables_exist() -> None:
    """Verify Phase 4 asset catalog tables are created by schema bootstrap."""
    gateway = open_memory_gateway(validate_schema=False)
    try:
        expected = {
            ("build", "asset_versions"),
            ("build", "run_asset_versions"),
            ("build", "asset_lineage"),
            ("build", "asset_aliases"),
            ("build", "asset_diffs"),
        }
        rows = gateway.con.execute(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema = 'build'
            ORDER BY table_name
            """
        ).fetchall()
        actual = {(str(r[0]), str(r[1])) for r in rows}
        missing = sorted(expected - actual)
        if missing:
            pytest.fail(f"Missing Phase 4 tables: {missing}")
    finally:
        gateway.close()


def test_pr28_phase4_asset_catalog_insert_and_resolve() -> None:
    """Verify version insert, run mapping, alias resolve, and diff caching."""
    gateway = open_memory_gateway(validate_schema=False)
    try:
        now = datetime.now(tz=UTC)
        version = AssetVersionRecord(
            asset_kind="table",
            asset_key="analytics.function_metrics",
            version_hash="0123456789abcdef",
            repo="test/repo",
            commit="abc123",
            status="materialized",
            run_id="run-1",
            target="function_metrics",
            impl_kind="native",
            location="analytics.function_metrics",
            input_hash="inputhash01234567",
            options_hash=None,
            schema_hash="schemahash",
            row_count=10,
            bytes=None,
            created_at=now,
            meta={"fingerprint": "fast"},
        )
        written = gateway.assets.record_asset_versions_batch([version])
        expect_true(written >= 1)

        run_map = RunAssetVersionRecord(
            run_id="run-1",
            repo="test/repo",
            commit="abc123",
            asset_kind="table",
            asset_key="analytics.function_metrics",
            version_hash="0123456789abcdef",
            resolution_kind="materialized",
            recorded_at=now,
            target="function_metrics",
            meta={"why": "test"},
        )
        gateway.assets.record_run_asset_versions_batch([run_map])
        mappings = gateway.assets.get_run_asset_versions(run_id="run-1")
        expect_length(mappings, 1)
        expect_equal(mappings[0].version_hash, "0123456789abcdef")

        gateway.assets.set_alias(
            AssetAliasRecord(
                alias="latest",
                asset_kind="table",
                asset_key="analytics.function_metrics",
                version_hash="0123456789abcdef",
                set_at=now,
                set_by_run_id="run-1",
                note="test",
            )
        )
        resolved = gateway.assets.resolve_alias(
            alias="latest", asset_kind="table", asset_key="analytics.function_metrics"
        )
        expect_equal(resolved, "0123456789abcdef")

        gateway.assets.save_cached_diff(
            AssetDiffRecord(
                asset_kind="table",
                asset_key="analytics.function_metrics",
                from_version_hash="0123456789abcdef",
                to_version_hash="fedcba9876543210",
                diff_kind="schema_rowcount",
                computed_at=now,
                computed_by_run_id="run-1",
                summary={"row_count": {"from": 10, "to": 11, "delta": 1}},
            )
        )
        cached = gateway.assets.get_cached_diff(
            asset_kind="table",
            asset_key="analytics.function_metrics",
            from_version_hash="0123456789abcdef",
            to_version_hash="fedcba9876543210",
            diff_kind="schema_rowcount",
        )
        if cached is None:
            pytest.fail("Expected cached diff to be present")
        if cached.summary is None:
            pytest.fail("Expected cached diff summary to be present")
        expect_equal(cached.summary.get("row_count", {}).get("delta"), 1)
    finally:
        gateway.close()


def test_pr28_phase4_asset_catalog_lineage_edges_upsert() -> None:
    """Verify lineage edges can be recorded and are queryable."""
    gateway = open_memory_gateway(validate_schema=False)
    try:
        now = datetime.now(tz=UTC)
        edges_written = gateway.assets.record_lineage_edges_batch(
            [
                AssetLineageEdgeRecord(
                    downstream_kind="table",
                    downstream_key="analytics.down",
                    downstream_version="aaaaaaaaaaaaaaaa",
                    upstream_kind="table",
                    upstream_key="analytics.up",
                    upstream_version="bbbbbbbbbbbbbbbb",
                    edge_kind="depends_on",
                    created_at=now,
                    meta={"reason": "test"},
                )
            ]
        )
        expect_true(edges_written >= 1)

        row = gateway.con.execute(
            """
            SELECT edge_kind
            FROM build.asset_lineage
            WHERE downstream_key = 'analytics.down'
            """
        ).fetchone()
        if row is None:
            pytest.fail("Expected lineage row")
        expect_equal(str(row[0]), "depends_on")
    finally:
        gateway.close()
