"""Integration tests for serving snapshot parquet-backed views."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from codeintel.core.hashing import stable_hash
from codeintel.core.manifests import ServingSnapshotManifest
from codeintel.storage.serving.search_index import build_search_documents_table
from codeintel.storage.serving.snapshot_service import ServingSnapshotService
from tests._helpers.assertions.expectation_assertions import expect_equal
from tests._helpers.schemas import ensure_production_schemas
from tests._helpers.serving_snapshot_factory import ServingSnapshotFactory, SnapshotArtifacts

pytestmark = pytest.mark.no_runtime_env


def _metrics_table_entry() -> dict[str, object]:
    return {
        "schema": "test",
        "name": "metrics",
        "table_key": "test.metrics",
        "primary_key": [],
        "indexes": [],
        "columns": [
            {"name": "repo", "type": "VARCHAR", "nullable": False},
            {"name": "commit", "type": "VARCHAR", "nullable": False},
            {"name": "value", "type": "BIGINT", "nullable": False},
        ],
    }


def _seed_core_modules(con: duckdb.DuckDBPyConnection, *, repo: str, commit: str) -> None:
    module_payload = {
        "module": "pkg.mod",
        "path": "pkg/mod.py",
        "repo": repo,
        "commit": commit,
        "language": "python",
        "tags": None,
        "owners": None,
    }
    module_payload["row_hash"] = stable_hash(module_payload)
    con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners, row_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            module_payload["module"],
            module_payload["path"],
            module_payload["repo"],
            module_payload["commit"],
            module_payload["language"],
            module_payload["tags"],
            module_payload["owners"],
            module_payload["row_hash"],
        ],
    )


def _ensure_search_documents(
    con: duckdb.DuckDBPyConnection,
    *,
    repo: str,
    commit: str,
) -> None:
    row = con.execute("SELECT COUNT(*) FROM docs.search_documents").fetchone()
    count = int(row[0]) if row is not None and row[0] is not None else 0
    if count > 0:
        return
    con.execute(
        """
        INSERT INTO docs.search_documents
            (doc_id, kind, name, module, rel_path, text, ref_goid_h128, repo, commit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "doc-1",
            "module",
            "pkg.mod",
            "pkg.mod",
            "pkg/mod.py",
            "Demo module",
            None,
            repo,
            commit,
        ],
    )


def _db_setup(db_path: Path, *, repo: str, commit: str) -> None:
    con = duckdb.connect(str(db_path))
    try:
        ensure_production_schemas(con)
        con.execute("CREATE SCHEMA IF NOT EXISTS test")
        con.execute("CREATE TABLE test.metrics (repo VARCHAR, commit VARCHAR, value BIGINT)")
        con.execute(
            "INSERT INTO test.metrics VALUES (?, ?, ?)",
            [repo, commit, 42],
        )
        _seed_core_modules(con, repo=repo, commit=commit)
        build_search_documents_table(con)
        _ensure_search_documents(con, repo=repo, commit=commit)
    finally:
        con.close()


def test_prepare_snapshot_registers_parquet_views(tmp_path: Path) -> None:
    """Snapshot preparation should register parquet-backed views when tables are missing."""
    factory = ServingSnapshotFactory(tmp_path, serve_dir=tmp_path / "serve")
    artifacts = SnapshotArtifacts(
        views=[],
        tables=[_metrics_table_entry()],
        db_setup=lambda db_path: _db_setup(
            db_path,
            repo=factory.repo,
            commit=factory.commit,
        ),
    )
    snapshot = factory.make_snapshot(run_id="run-1", artifacts=artifacts)
    con = duckdb.connect(str(snapshot.db_path))
    try:
        con.execute("DROP TABLE test.metrics")
    finally:
        con.close()

    snapshot_manifest = ServingSnapshotManifest.from_path(snapshot.snapshot_manifest_path)
    service = ServingSnapshotService()
    service.prepare_snapshot(
        db_path=snapshot.db_path,
        snapshot_manifest=snapshot_manifest,
    )

    con = duckdb.connect(str(snapshot.db_path))
    try:
        rows = con.execute("SELECT repo, commit, value FROM test.metrics").fetchall()
        expect_equal(rows, [(factory.repo, factory.commit, 42)])
    finally:
        con.close()
