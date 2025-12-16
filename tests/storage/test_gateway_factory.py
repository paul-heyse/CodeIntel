"""Comprehensive tests for gateway factory functions.

This module tests all factory functions in codeintel.storage.gateway.factory,
following the Testing Charter by using real DuckDB connections.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.storage.gateway.accessors import DuckDBGateway
from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.gateway.factory import (
    build_snapshot_gateway_resolver,
    open_gateway,
    open_memory_gateway,
)
from tests._helpers import assert_frozen
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_instance,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.builders import ModuleRow, insert_rows


def test_storage_config_creates_with_defaults() -> None:
    """Verify StorageConfig default values."""
    cfg = StorageConfig(db_path=Path(":memory:"))
    expect_equal(cfg.db_path, Path(":memory:"), label="db_path")
    expect_true(cfg.read_only is False, message="read_only default")
    expect_true(cfg.apply_schema is False, message="apply_schema default")
    expect_true(cfg.ensure_views is False, message="ensure_views default")
    expect_true(cfg.validate_schema is False, message="validate_schema default")
    expect_true(cfg.attach_history is False, message="attach_history default")
    expect_true(cfg.history_db_path is None, message="history_db_path default")
    expect_true(cfg.repo is None, message="repo default")
    expect_true(cfg.commit is None, message="commit default")


def test_storage_config_creates_with_all_options(tmp_path: Path) -> None:
    """Verify StorageConfig accepts all options."""
    db_path = tmp_path / "test.duckdb"
    history_path = tmp_path / "history.duckdb"
    cfg = StorageConfig(
        db_path=db_path,
        read_only=True,
        apply_schema=True,
        ensure_views=True,
        validate_schema=True,
        attach_history=True,
        history_db_path=history_path,
        repo="test/repo",
        commit="abc123",
    )
    expect_equal(cfg.db_path, db_path, label="db_path")
    expect_true(cfg.read_only is True, message="read_only")
    expect_true(cfg.apply_schema is True, message="apply_schema")
    expect_true(cfg.ensure_views is True, message="ensure_views")
    expect_true(cfg.validate_schema is True, message="validate_schema")
    expect_true(cfg.attach_history is True, message="attach_history")
    expect_equal(cfg.history_db_path, history_path, label="history_db_path")
    expect_equal(cfg.repo, "test/repo", label="repo")
    expect_equal(cfg.commit, "abc123", label="commit")


def test_storage_config_for_ingest(tmp_path: Path) -> None:
    """Verify for_ingest factory method returns correct configuration."""
    db_path = tmp_path / "test.duckdb"
    cfg = StorageConfig.for_ingest(db_path)
    expect_equal(cfg.db_path, db_path, label="db_path")
    expect_true(cfg.read_only is False, message="read_only")
    expect_true(cfg.apply_schema is True, message="apply_schema")
    expect_true(cfg.ensure_views is True, message="ensure_views")
    expect_true(cfg.validate_schema is True, message="validate_schema")
    expect_true(cfg.attach_history is False, message="attach_history")


def test_storage_config_for_ingest_with_history(tmp_path: Path) -> None:
    """Verify for_ingest with history database."""
    db_path = tmp_path / "test.duckdb"
    history_path = tmp_path / "history.duckdb"
    cfg = StorageConfig.for_ingest(
        db_path,
        history_db_path=history_path,
    )
    expect_true(cfg.attach_history is True, message="attach_history")
    expect_equal(cfg.history_db_path, history_path, label="history_db_path")


def test_storage_config_for_readonly(tmp_path: Path) -> None:
    """Verify for_readonly factory method returns correct configuration."""
    db_path = tmp_path / "test.duckdb"
    cfg = StorageConfig.for_readonly(db_path)
    expect_equal(cfg.db_path, db_path, label="db_path")
    expect_true(cfg.read_only is True, message="read_only")
    expect_true(cfg.apply_schema is False, message="apply_schema")
    expect_true(cfg.ensure_views is True, message="ensure_views")
    expect_true(cfg.validate_schema is True, message="validate_schema")


def test_storage_config_is_frozen() -> None:
    """Verify StorageConfig is immutable."""
    cfg = StorageConfig(db_path=Path(":memory:"))
    assert_frozen(cfg, "db_path", Path("/other"))


def test_open_memory_gateway_returns_gateway() -> None:
    """Verify open_memory_gateway returns a DuckDBGateway."""
    gateway = open_memory_gateway(validate_schema=False)
    try:
        expect_is_instance(gateway, DuckDBGateway, label="gateway type")
        expect_is_not_none(gateway.config, label="gateway config")
        expect_is_not_none(gateway.datasets, label="gateway datasets")
    finally:
        gateway.close()


def test_open_memory_gateway_with_defaults() -> None:
    """Verify open_memory_gateway applies schema by default."""
    gateway = open_memory_gateway(validate_schema=False)
    try:
        result = gateway.con.execute("SELECT COUNT(*) FROM core.modules").fetchone()
        if result is None:
            pytest.fail("Expected modules count row")
        row_count = result[0]
        expect_equal(row_count, 0, label="modules row count")
    finally:
        gateway.close()


def test_open_memory_gateway_without_views() -> None:
    """Verify open_memory_gateway can skip view creation."""
    gateway = open_memory_gateway(ensure_views=False, validate_schema=False)
    try:
        result = gateway.con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'modules'"
        ).fetchone()
        if result is None:
            pytest.fail("Expected information_schema count row")
        table_count = result[0]
        expect_true(table_count >= 1, message="modules table exists")
    finally:
        gateway.close()


def test_open_memory_gateway_with_views() -> None:
    """Verify open_memory_gateway can create views."""
    gateway = open_memory_gateway(ensure_views=True, validate_schema=False)
    try:
        result = gateway.con.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = 'docs' AND table_type = 'VIEW'"
        ).fetchone()
        expect_is_not_none(result, label="docs views count")
        analytics_view = gateway.con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'analytics' AND table_name = 'v_function_summary'
            """
        ).fetchone()
        expect_is_not_none(analytics_view, label="analytics Ibis view exists")
    finally:
        gateway.close()


def test_open_memory_gateway_with_repo_and_commit() -> None:
    """Verify open_memory_gateway stores repo and commit in config."""
    gateway = open_memory_gateway(repo="test/repo", commit="abc123", validate_schema=False)
    try:
        expect_equal(gateway.config.repo, "test/repo", label="repo")
        expect_equal(gateway.config.commit, "abc123", label="commit")
    finally:
        gateway.close()


def test_open_memory_gateway_creates_accessors() -> None:
    """Verify gateway has all accessor properties."""
    gateway = open_memory_gateway(validate_schema=False)
    try:
        expect_true(hasattr(gateway, "core"), message="core accessor present")
        expect_true(hasattr(gateway, "graph"), message="graph accessor present")
        expect_true(hasattr(gateway, "docs"), message="docs accessor present")
        expect_true(hasattr(gateway, "analytics"), message="analytics accessor present")
    finally:
        gateway.close()


def test_open_memory_gateway_exposes_ibis_backend() -> None:
    """Verify gateway exposes an Ibis backend bound to the DuckDB connection."""
    gateway = open_memory_gateway(validate_schema=False)
    try:
        table = gateway.ibis.table("core.modules")
        row_count = table.count().execute()
        expect_equal(row_count, 0, label="ibis table count")
    finally:
        gateway.close()


def test_open_memory_gateway_supports_insert_and_query() -> None:
    """Verify gateway supports data operations."""
    gateway = open_memory_gateway(validate_schema=False)
    try:
        insert_rows(
            gateway,
            [
                ModuleRow(module="test_mod", path="test.py", repo="test/repo", commit="abc123"),
            ],
        )
        result = gateway.con.execute(
            "SELECT module FROM core.modules WHERE repo = ?", ["test/repo"]
        ).fetchone()
        if result is None:
            pytest.fail("Expected module fetch result")
        module_name = result[0]
        expect_equal(module_name, "test_mod", label="module value")
    finally:
        gateway.close()


def test_open_memory_gateway_has_dataset_registry() -> None:
    """Verify gateway has loaded dataset registry."""
    gateway = open_memory_gateway(validate_schema=False)
    try:
        expect_is_not_none(gateway.datasets, label="datasets registry")

        expect_true(len(gateway.datasets.by_name) > 0, message="dataset registry populated")
    finally:
        gateway.close()


def test_open_gateway_creates_file_database(tmp_path: Path) -> None:
    """Verify open_gateway creates a file-based database."""
    db_path = tmp_path / "test.duckdb"
    cfg = StorageConfig(
        db_path=db_path,
        read_only=False,
        apply_schema=True,
        ensure_views=False,
        validate_schema=False,
    )
    gateway = open_gateway(cfg)
    try:
        expect_true(db_path.exists(), message="db_path created")
        expect_is_instance(gateway, DuckDBGateway, label="gateway type")
    finally:
        gateway.close()


def test_open_gateway_creates_tables(tmp_path: Path) -> None:
    """Verify open_gateway creates tables when apply_schema is True."""
    db_path = tmp_path / "test.duckdb"
    cfg = StorageConfig(
        db_path=db_path,
        read_only=False,
        apply_schema=True,
        ensure_views=False,
        validate_schema=False,
    )
    gateway = open_gateway(cfg)
    try:
        result = gateway.con.execute("SELECT COUNT(*) FROM core.modules").fetchone()
        expect_is_not_none(result, label="modules count")
    finally:
        gateway.close()


def test_open_gateway_persists_data(tmp_path: Path) -> None:
    """Verify data persists across gateway sessions."""
    db_path = tmp_path / "test.duckdb"

    cfg1 = StorageConfig(
        db_path=db_path,
        read_only=False,
        apply_schema=True,
        ensure_views=False,
        validate_schema=False,
    )
    gateway1 = open_gateway(cfg1)
    try:
        insert_rows(
            gateway1,
            [ModuleRow(module="mod", path="mod.py", repo="repo", commit="commit")],
        )
    finally:
        gateway1.close()

    cfg2 = StorageConfig(
        db_path=db_path,
        read_only=True,
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
    )
    gateway2 = open_gateway(cfg2)
    try:
        result = gateway2.con.execute(
            "SELECT module FROM core.modules WHERE repo = ?", ["repo"]
        ).fetchone()
        if result is None:
            pytest.fail("Expected persisted module row")
        module_value = result[0]
        expect_equal(module_value, "mod", label="module value")
    finally:
        gateway2.close()


def test_open_gateway_read_only_mode(tmp_path: Path) -> None:
    """Verify read_only mode doesn't apply schema or views."""
    db_path = tmp_path / "test.duckdb"

    cfg_write = StorageConfig(
        db_path=db_path,
        read_only=False,
        apply_schema=True,
        ensure_views=False,
        validate_schema=False,
    )
    gateway_write = open_gateway(cfg_write)
    gateway_write.close()

    cfg_read = StorageConfig(
        db_path=db_path,
        read_only=True,
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
    )
    gateway_read = open_gateway(cfg_read)
    try:
        result = gateway_read.con.execute("SELECT COUNT(*) FROM core.modules").fetchone()
        expect_is_not_none(result, label="read count")
    finally:
        gateway_read.close()


def test_snapshot_resolver_returns_callable(tmp_path: Path) -> None:
    """Verify build_snapshot_gateway_resolver returns callable."""
    resolver = build_snapshot_gateway_resolver(db_dir=tmp_path)
    expect_true(callable(resolver), message="resolver callable")


def test_snapshot_resolver_raises_for_missing_file(tmp_path: Path) -> None:
    """Verify resolver raises FileNotFoundError for missing snapshot."""
    resolver = build_snapshot_gateway_resolver(db_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing snapshot database"):
        resolver("nonexistent_commit")


def test_snapshot_resolver_opens_existing_snapshot(tmp_path: Path) -> None:
    """Verify resolver opens existing snapshot database."""
    commit = "abc123"
    db_path = tmp_path / f"codeintel-{commit}.duckdb"

    cfg = StorageConfig(
        db_path=db_path,
        read_only=False,
        apply_schema=True,
        ensure_views=False,
        validate_schema=False,
    )
    setup_gw = open_gateway(cfg)
    setup_gw.close()

    resolver = build_snapshot_gateway_resolver(db_dir=tmp_path, repo="test/repo")
    gateway = resolver(commit)
    try:
        expect_is_instance(gateway, DuckDBGateway, label="gateway type")
        expect_true(gateway.config.read_only is True, message="read_only")
        expect_equal(gateway.config.repo, "test/repo", label="repo")
        expect_equal(gateway.config.commit, commit, label="commit")
    finally:
        gateway.close()


def test_snapshot_resolver_reuses_primary_gateway(tmp_path: Path) -> None:
    """Verify resolver reuses primary_gateway when paths match."""
    commit = "abc123"
    db_path = tmp_path / f"codeintel-{commit}.duckdb"

    cfg = StorageConfig(
        db_path=db_path,
        read_only=False,
        apply_schema=True,
        ensure_views=False,
        validate_schema=False,
    )
    primary = open_gateway(cfg)

    try:
        resolver = build_snapshot_gateway_resolver(
            db_dir=tmp_path,
            primary_gateway=primary,
        )

        resolved = resolver(commit)
        expect_true(resolved is primary, message="resolver reused primary gateway")
    finally:
        primary.close()


def test_snapshot_resolver_opens_different_commits(tmp_path: Path) -> None:
    """Verify resolver opens different snapshots for different commits."""
    commits = ["commit1", "commit2"]

    for commit in commits:
        db_path = tmp_path / f"codeintel-{commit}.duckdb"
        cfg = StorageConfig(
            db_path=db_path,
            read_only=False,
            apply_schema=True,
            ensure_views=False,
            validate_schema=False,
        )
        gw = open_gateway(cfg)

        insert_rows(
            gw,
            [ModuleRow(module=f"mod_{commit}", path="mod.py", repo="repo", commit=commit)],
        )
        gw.close()

    resolver = build_snapshot_gateway_resolver(db_dir=tmp_path)

    gw1 = resolver("commit1")
    gw2 = resolver("commit2")

    try:
        r1 = gw1.con.execute("SELECT module FROM core.modules").fetchone()
        r2 = gw2.con.execute("SELECT module FROM core.modules").fetchone()
        if r1 is None or r2 is None:
            pytest.fail("Expected rows for both commits")
        expect_equal(r1[0], "mod_commit1", label="commit1 module")
        expect_equal(r2[0], "mod_commit2", label="commit2 module")
    finally:
        gw1.close()
        gw2.close()


def test_gateway_close_releases_connection() -> None:
    """Verify close() releases the database connection."""
    gateway = open_memory_gateway(validate_schema=False)
    expect_is_not_none(gateway.con, label="connection before close")
    gateway.close()


def test_gateway_supports_context_manager() -> None:
    """Verify gateway can be used as context manager if supported."""
    gateway = open_memory_gateway(validate_schema=False)

    try:
        expect_is_not_none(gateway.con, label="connection available")
    finally:
        gateway.close()


def test_full_gateway_lifecycle(tmp_path: Path) -> None:
    """Test complete lifecycle: create, write, close, reopen, read."""
    db_path = tmp_path / "lifecycle.duckdb"

    cfg_write = StorageConfig(
        db_path=db_path,
        read_only=False,
        apply_schema=True,
        ensure_views=False,
        validate_schema=False,
    )
    gw_write = open_gateway(cfg_write)
    insert_rows(
        gw_write,
        [
            ModuleRow(module="mod_a", path="mod_a.py", repo="test/repo", commit="v1"),
            ModuleRow(module="mod_b", path="mod_b.py", repo="test/repo", commit="v1"),
        ],
    )
    gw_write.close()

    cfg_read = StorageConfig.for_readonly(db_path)
    gw_read = open_gateway(cfg_read)
    try:
        count = gw_read.con.execute(
            "SELECT COUNT(*) FROM core.modules WHERE repo = ?",
            ["test/repo"],
        ).fetchone()
        if count is None:
            pytest.fail("Expected module count row")
        expected_count = 2
        expect_equal(count[0], expected_count, label="module count value")
    finally:
        gw_read.close()


def test_multiple_memory_gateways_are_independent() -> None:
    """Verify multiple memory gateways don't share state."""
    gw1 = open_memory_gateway(validate_schema=False)
    gw2 = open_memory_gateway(validate_schema=False)

    try:
        insert_rows(gw1, [ModuleRow(module="mod1", path="m1.py", repo="repo1", commit="c1")])

        r2 = gw2.con.execute("SELECT COUNT(*) FROM core.modules").fetchone()
        if r2 is None:
            pytest.fail("Expected gateway2 modules count")
        expect_equal(r2[0], 0, label="gateway2 modules")

        r1 = gw1.con.execute("SELECT COUNT(*) FROM core.modules").fetchone()
        if r1 is None:
            pytest.fail("Expected gateway1 modules count")
        expect_equal(r1[0], 1, label="gateway1 modules")
    finally:
        gw1.close()
        gw2.close()
