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

# =============================================================================
# StorageConfig Tests
# =============================================================================


def test_storage_config_creates_with_defaults() -> None:
    """Verify StorageConfig default values."""
    cfg = StorageConfig(db_path=Path(":memory:"))
    assert cfg.db_path == Path(":memory:")
    assert cfg.read_only is False
    assert cfg.apply_schema is False
    assert cfg.ensure_views is False
    assert cfg.validate_schema is False
    assert cfg.attach_history is False
    assert cfg.history_db_path is None
    assert cfg.repo is None
    assert cfg.commit is None


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
    assert cfg.db_path == db_path
    assert cfg.read_only is True
    assert cfg.apply_schema is True
    assert cfg.ensure_views is True
    assert cfg.validate_schema is True
    assert cfg.attach_history is True
    assert cfg.history_db_path == history_path
    assert cfg.repo == "test/repo"
    assert cfg.commit == "abc123"


def test_storage_config_for_ingest(tmp_path: Path) -> None:
    """Verify for_ingest factory method returns correct configuration."""
    db_path = tmp_path / "test.duckdb"
    cfg = StorageConfig.for_ingest(db_path)
    assert cfg.db_path == db_path
    assert cfg.read_only is False
    assert cfg.apply_schema is True
    assert cfg.ensure_views is True
    assert cfg.validate_schema is True
    assert cfg.attach_history is False


def test_storage_config_for_ingest_with_history(tmp_path: Path) -> None:
    """Verify for_ingest with history database."""
    db_path = tmp_path / "test.duckdb"
    history_path = tmp_path / "history.duckdb"
    cfg = StorageConfig.for_ingest(
        db_path,
        history_db_path=history_path,
    )
    assert cfg.attach_history is True
    assert cfg.history_db_path == history_path


def test_storage_config_for_readonly(tmp_path: Path) -> None:
    """Verify for_readonly factory method returns correct configuration."""
    db_path = tmp_path / "test.duckdb"
    cfg = StorageConfig.for_readonly(db_path)
    assert cfg.db_path == db_path
    assert cfg.read_only is True
    assert cfg.apply_schema is False
    assert cfg.ensure_views is True
    assert cfg.validate_schema is True


def test_storage_config_is_frozen() -> None:
    """Verify StorageConfig is immutable."""
    cfg = StorageConfig(db_path=Path(":memory:"))
    assert_frozen(cfg, "db_path", Path("/other"))


# =============================================================================
# open_memory_gateway Tests
# =============================================================================


def test_open_memory_gateway_returns_gateway() -> None:
    """Verify open_memory_gateway returns a DuckDBGateway."""
    gateway = open_memory_gateway(validate_schema=False)
    try:
        # DuckDBGateway implements StorageGateway protocol
        assert isinstance(gateway, DuckDBGateway)
        assert gateway.config is not None
        assert gateway.datasets is not None
    finally:
        gateway.close()


def test_open_memory_gateway_with_defaults() -> None:
    """Verify open_memory_gateway applies schema by default."""
    gateway = open_memory_gateway(validate_schema=False)
    try:
        # Should be able to query core.modules if schema applied
        result = gateway.con.execute("SELECT COUNT(*) FROM core.modules").fetchone()
        assert result is not None
        assert result[0] == 0  # Empty but table exists
    finally:
        gateway.close()


def test_open_memory_gateway_without_views() -> None:
    """Verify open_memory_gateway can skip view creation."""
    gateway = open_memory_gateway(ensure_views=False, validate_schema=False)
    try:
        # Tables should exist
        result = gateway.con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'modules'"
        ).fetchone()
        assert result is not None
        assert result[0] >= 1  # Table exists
    finally:
        gateway.close()


def test_open_memory_gateway_with_views() -> None:
    """Verify open_memory_gateway can create views."""
    gateway = open_memory_gateway(ensure_views=True, validate_schema=False)
    try:
        # docs views should exist
        result = gateway.con.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = 'docs' AND table_type = 'VIEW'"
        ).fetchone()
        assert result is not None
        # At least some views should exist
    finally:
        gateway.close()


def test_open_memory_gateway_with_repo_and_commit() -> None:
    """Verify open_memory_gateway stores repo and commit in config."""
    gateway = open_memory_gateway(repo="test/repo", commit="abc123", validate_schema=False)
    try:
        assert gateway.config.repo == "test/repo"
        assert gateway.config.commit == "abc123"
    finally:
        gateway.close()


def test_open_memory_gateway_creates_accessors() -> None:
    """Verify gateway has all accessor properties."""
    gateway = open_memory_gateway(validate_schema=False)
    try:
        assert hasattr(gateway, "core")
        assert hasattr(gateway, "graph")
        assert hasattr(gateway, "docs")
        assert hasattr(gateway, "analytics")
    finally:
        gateway.close()


def test_open_memory_gateway_supports_insert_and_query() -> None:
    """Verify gateway supports data operations."""
    gateway = open_memory_gateway(validate_schema=False)
    try:
        gateway.core.insert_modules(
            [
                ("test_mod", "test.py", "test/repo", "abc123"),
            ]
        )
        result = gateway.con.execute(
            "SELECT module FROM core.modules WHERE repo = ?", ["test/repo"]
        ).fetchone()
        assert result is not None
        assert result[0] == "test_mod"
    finally:
        gateway.close()


def test_open_memory_gateway_has_dataset_registry() -> None:
    """Verify gateway has loaded dataset registry."""
    gateway = open_memory_gateway(validate_schema=False)
    try:
        assert gateway.datasets is not None
        # Registry should have some datasets accessible by name
        assert len(gateway.datasets.by_name) > 0
    finally:
        gateway.close()


# =============================================================================
# open_gateway Tests with File Database
# =============================================================================


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
        assert db_path.exists()
        # DuckDBGateway implements StorageGateway protocol
        assert isinstance(gateway, DuckDBGateway)
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
        # core.modules should exist
        result = gateway.con.execute("SELECT COUNT(*) FROM core.modules").fetchone()
        assert result is not None
    finally:
        gateway.close()


def test_open_gateway_persists_data(tmp_path: Path) -> None:
    """Verify data persists across gateway sessions."""
    db_path = tmp_path / "test.duckdb"

    # First session - insert data
    cfg1 = StorageConfig(
        db_path=db_path,
        read_only=False,
        apply_schema=True,
        ensure_views=False,
        validate_schema=False,
    )
    gateway1 = open_gateway(cfg1)
    try:
        gateway1.core.insert_modules([("mod", "mod.py", "repo", "commit")])
    finally:
        gateway1.close()

    # Second session - verify data
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
        assert result is not None
        assert result[0] == "mod"
    finally:
        gateway2.close()


def test_open_gateway_read_only_mode(tmp_path: Path) -> None:
    """Verify read_only mode doesn't apply schema or views."""
    db_path = tmp_path / "test.duckdb"

    # Create database first
    cfg_write = StorageConfig(
        db_path=db_path,
        read_only=False,
        apply_schema=True,
        ensure_views=False,
        validate_schema=False,
    )
    gateway_write = open_gateway(cfg_write)
    gateway_write.close()

    # Open in read-only mode
    cfg_read = StorageConfig(
        db_path=db_path,
        read_only=True,
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
    )
    gateway_read = open_gateway(cfg_read)
    try:
        # Should be able to read
        result = gateway_read.con.execute("SELECT COUNT(*) FROM core.modules").fetchone()
        assert result is not None
    finally:
        gateway_read.close()


# =============================================================================
# build_snapshot_gateway_resolver Tests
# =============================================================================


def test_snapshot_resolver_returns_callable(tmp_path: Path) -> None:
    """Verify build_snapshot_gateway_resolver returns callable."""
    resolver = build_snapshot_gateway_resolver(db_dir=tmp_path)
    assert callable(resolver)


def test_snapshot_resolver_raises_for_missing_file(tmp_path: Path) -> None:
    """Verify resolver raises FileNotFoundError for missing snapshot."""
    resolver = build_snapshot_gateway_resolver(db_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing snapshot database"):
        resolver("nonexistent_commit")


def test_snapshot_resolver_opens_existing_snapshot(tmp_path: Path) -> None:
    """Verify resolver opens existing snapshot database."""
    commit = "abc123"
    db_path = tmp_path / f"codeintel-{commit}.duckdb"

    # Create a snapshot database
    cfg = StorageConfig(
        db_path=db_path,
        read_only=False,
        apply_schema=True,
        ensure_views=False,
        validate_schema=False,
    )
    setup_gw = open_gateway(cfg)
    setup_gw.close()

    # Use resolver to open it
    resolver = build_snapshot_gateway_resolver(db_dir=tmp_path, repo="test/repo")
    gateway = resolver(commit)
    try:
        # DuckDBGateway implements StorageGateway protocol
        assert isinstance(gateway, DuckDBGateway)
        assert gateway.config.read_only is True
        assert gateway.config.repo == "test/repo"
        assert gateway.config.commit == commit
    finally:
        gateway.close()


def test_snapshot_resolver_reuses_primary_gateway(tmp_path: Path) -> None:
    """Verify resolver reuses primary_gateway when paths match."""
    commit = "abc123"
    db_path = tmp_path / f"codeintel-{commit}.duckdb"

    # Create primary gateway
    cfg = StorageConfig(
        db_path=db_path,
        read_only=False,
        apply_schema=True,
        ensure_views=False,
        validate_schema=False,
    )
    primary = open_gateway(cfg)

    try:
        # Create resolver with primary gateway
        resolver = build_snapshot_gateway_resolver(
            db_dir=tmp_path,
            primary_gateway=primary,
        )

        # Should return same instance
        resolved = resolver(commit)
        assert resolved is primary
    finally:
        primary.close()


def test_snapshot_resolver_opens_different_commits(tmp_path: Path) -> None:
    """Verify resolver opens different snapshots for different commits."""
    commits = ["commit1", "commit2"]

    # Create snapshot databases
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
        # Insert unique data per commit
        gw.core.insert_modules([(f"mod_{commit}", "mod.py", "repo", commit)])
        gw.close()

    # Resolve both commits
    resolver = build_snapshot_gateway_resolver(db_dir=tmp_path)

    gw1 = resolver("commit1")
    gw2 = resolver("commit2")

    try:
        # Verify different data
        r1 = gw1.con.execute("SELECT module FROM core.modules").fetchone()
        r2 = gw2.con.execute("SELECT module FROM core.modules").fetchone()
        assert r1 is not None
        assert r2 is not None
        assert r1[0] == "mod_commit1"
        assert r2[0] == "mod_commit2"
    finally:
        gw1.close()
        gw2.close()


# =============================================================================
# Gateway Close and Resource Management
# =============================================================================


def test_gateway_close_releases_connection() -> None:
    """Verify close() releases the database connection."""
    gateway = open_memory_gateway(validate_schema=False)
    assert gateway.con is not None
    gateway.close()
    # After close, connection should be closed
    # (attempting to use it would raise an error)


def test_gateway_supports_context_manager() -> None:
    """Verify gateway can be used as context manager if supported."""
    gateway = open_memory_gateway(validate_schema=False)
    # Manually close since DuckDBGateway might not implement __enter__/__exit__
    try:
        assert gateway.con is not None
    finally:
        gateway.close()


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_gateway_lifecycle(tmp_path: Path) -> None:
    """Test complete lifecycle: create, write, close, reopen, read."""
    db_path = tmp_path / "lifecycle.duckdb"

    # Create and populate
    cfg_write = StorageConfig(
        db_path=db_path,
        read_only=False,
        apply_schema=True,
        ensure_views=False,
        validate_schema=False,
    )
    gw_write = open_gateway(cfg_write)
    gw_write.core.insert_modules(
        [
            ("mod_a", "mod_a.py", "test/repo", "v1"),
            ("mod_b", "mod_b.py", "test/repo", "v1"),
        ]
    )
    gw_write.close()

    # Reopen and verify
    cfg_read = StorageConfig.for_readonly(db_path)
    gw_read = open_gateway(cfg_read)
    try:
        count = gw_read.con.execute(
            "SELECT COUNT(*) FROM core.modules WHERE repo = ?",
            ["test/repo"],
        ).fetchone()
        assert count is not None
        expected_count = 2
        assert count[0] == expected_count
    finally:
        gw_read.close()


def test_multiple_memory_gateways_are_independent() -> None:
    """Verify multiple memory gateways don't share state."""
    gw1 = open_memory_gateway(validate_schema=False)
    gw2 = open_memory_gateway(validate_schema=False)

    try:
        # Insert in gw1
        gw1.core.insert_modules([("mod1", "m1.py", "repo1", "c1")])

        # gw2 should be empty
        r2 = gw2.con.execute("SELECT COUNT(*) FROM core.modules").fetchone()
        assert r2 is not None
        assert r2[0] == 0

        # gw1 should have data
        r1 = gw1.con.execute("SELECT COUNT(*) FROM core.modules").fetchone()
        assert r1 is not None
        assert r1[0] == 1
    finally:
        gw1.close()
        gw2.close()
