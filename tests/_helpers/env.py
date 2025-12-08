"""Shared environment builder for tests.

Provides a single entry point for constructing gateways and ``TestContext``
instances with production-parity defaults (schema, views, macros). Exports
canonical test defaults for repo/commit/run identifiers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.macros import ensure_ingest_macros
from codeintel.storage.schema import apply_all_schemas
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO, DEFAULT_RUN_ID
from tests._helpers.gateway import GatewayFactory

if TYPE_CHECKING:
    from tests._helpers.context import TestContext


def build_test_gateway(
    *,
    file_backed: bool = False,
    db_path: Path | None = None,
    repo: str | None = None,
    commit: str | None = None,
    apply_schema: bool = True,
    ensure_views: bool = True,
    validate_schema: bool = True,
) -> StorageGateway:
    """Create a StorageGateway with schema/views/macros ensured.

    Parameters
    ----------
    file_backed
        Whether to use a file-backed DuckDB database.
    db_path
        Optional database path (required when file_backed=True).
    repo
        Optional repository identifier to record on the gateway.
    commit
        Optional commit identifier to record on the gateway.
    apply_schema
        Whether to apply database schema.
    ensure_views
        Whether to create views.
    validate_schema
        Whether to validate applied schema.

    Returns
    -------
    StorageGateway
        Gateway ready for test use with macros ensured.
    """
    factory = GatewayFactory()
    factory = factory.with_schema() if apply_schema else factory.without_schema()
    factory = factory.with_views() if ensure_views else factory.without_views()
    factory = factory.with_validation() if validate_schema else factory.without_validation()
    if repo is not None and commit is not None:
        factory = factory.with_snapshot(repo, commit)

    if file_backed:
        if db_path is None:
            message = "db_path must be provided for file_backed gateways"
            raise ValueError(message)
        factory = factory.file_backed(db_path)

    gateway = factory.open()
    apply_all_schemas(gateway.con)
    ensure_ingest_macros(gateway.con)
    return gateway


def create_test_env(
    tmp_path: Path,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    file_backed: bool = False,
    repo_root: Path | None = None,
    build_dir: Path | None = None,
    db_path: Path | None = None,
) -> TestContext:
    """Build a TestContext with consistent defaults via GatewayFactory.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.
    repo
        Repository identifier.
    commit
        Commit hash.
    file_backed
        Whether to use a file-backed DuckDB database.
    repo_root
        Optional repository root override (defaults to tmp_path / \"repo\").
    build_dir
        Optional build directory override (defaults to tmp_path / \"build\").
    db_path
        Optional database path when file_backed is True. Defaults under build_dir.

    Returns
    -------
    TestContext
        Configured test context with gateway, snapshot, and build paths.
    """
    from tests._helpers.context import TestContext

    repo_root_path = repo_root or (tmp_path / "repo")
    build_dir_path = build_dir or (tmp_path / "build")
    repo_root_path.mkdir(parents=True, exist_ok=True)
    build_dir_path.mkdir(parents=True, exist_ok=True)
    db_path_resolved = db_path
    if file_backed and db_path_resolved is None:
        db_path_resolved = build_dir_path / "db" / "codeintel.duckdb"
        db_path_resolved.parent.mkdir(parents=True, exist_ok=True)

    gateway = build_test_gateway(
        file_backed=file_backed,
        db_path=db_path_resolved,
        repo=repo,
        commit=commit,
    )
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=repo_root_path)
    build_paths = BuildPaths.from_repo_root(repo_root_path, build_dir=build_dir_path)

    return TestContext(
        snapshot=snapshot,
        gateway=gateway,
        build_paths=build_paths,
    )


__all__ = [
    "DEFAULT_COMMIT",
    "DEFAULT_REPO",
    "DEFAULT_RUN_ID",
    "build_test_gateway",
    "create_test_env",
]
