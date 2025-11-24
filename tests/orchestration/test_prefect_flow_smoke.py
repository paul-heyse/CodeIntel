"""Smoke test to ensure Prefect flow scaffolding imports and runs no-op path."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pytest
from prefect import flow

from codeintel.docs_export.export_jsonl import export_all_jsonl
from codeintel.docs_export.export_parquet import export_all_parquet
from codeintel.orchestration.prefect_flow import (
    ExportArgs,
    _close_gateways,  # noqa: PLC2701
    _get_gateway,  # noqa: PLC2701
    _resolve_validation_settings,  # noqa: PLC2701
    export_docs_flow,
    gateway_cache_stats,
)


def test_prefect_flow_imports() -> None:
    """Prefect flow can be imported without side effects."""
    if not callable(export_docs_flow):
        pytest.fail("export_docs_flow is not callable")


def test_prefect_flow_minimal(tmp_path: Path, prefect_quiet_env: None) -> None:
    """
    Run the flow with empty inputs to ensure it executes without raising.

    This is a lightweight guardrail; full data population is covered elsewhere.
    """
    _ = prefect_quiet_env  # ensure harness/quiet logging fixtures are applied
    prev_skip = os.environ.get("CODEINTEL_SKIP_SCIP")
    os.environ["CODEINTEL_SKIP_SCIP"] = "true"
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / ".git").mkdir(parents=True, exist_ok=True)
    db_path = repo_root / "build" / "db" / "codeintel.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    build_dir = repo_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    scip_dir = build_dir / "scip"
    scip_dir.mkdir(parents=True, exist_ok=True)
    (scip_dir / "index.scip").write_text("dummy", encoding="utf8")
    (scip_dir / "index.scip.json").write_text("[]", encoding="utf8")

    export_docs_flow(
        args=ExportArgs(
            repo_root=repo_root,
            repo="demo/repo",
            commit="deadbeef",
            db_path=db_path,
            build_dir=build_dir,
        )
    )
    if prev_skip is None:
        os.environ.pop("CODEINTEL_SKIP_SCIP", None)
    else:
        os.environ["CODEINTEL_SKIP_SCIP"] = prev_skip


def test_resolve_validation_settings_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure export validation toggles honor environment variables."""
    monkeypatch.setenv("CODEINTEL_VALIDATE_EXPORTS", "true")
    monkeypatch.setenv("CODEINTEL_VALIDATE_SCHEMAS", "function_profile,call_graph_edges")
    args = ExportArgs(
        repo_root=Path(),
        repo="demo/repo",
        commit="deadbeef",
        db_path=Path("db.duckdb"),
        build_dir=Path("build"),
        validate_exports=False,
        export_schemas=None,
    )
    validate, schemas = _resolve_validation_settings(args)
    if validate is not True:
        pytest.fail("Expected validation to be enabled from environment toggle")
    if schemas != ["function_profile", "call_graph_edges"]:
        pytest.fail(f"Unexpected schema list: {schemas}")


def test_prefect_export_docs_with_validation(tmp_path: Path, prefect_quiet_env: None) -> None:
    """Ensure export_docs task can run with validation enabled against minimal data."""
    _ = prefect_quiet_env
    db_path = tmp_path / "db.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA IF NOT EXISTS core;")
    con.execute("CREATE SCHEMA IF NOT EXISTS analytics;")
    con.execute(
        """
        CREATE TABLE core.repo_map (
            repo TEXT,
            commit TEXT,
            modules JSON,
            overlays JSON,
            generated_at TIMESTAMP
        );
        """
    )
    con.execute(
        """
        INSERT INTO core.repo_map VALUES ('demo/repo', 'deadbeef', '{}', '{}', CURRENT_TIMESTAMP);
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.function_profile (
            function_goid_h128 BIGINT,
            urn TEXT,
            repo TEXT,
            commit TEXT,
            rel_path TEXT,
            module TEXT
        );
        """
    )
    con.execute(
        """
        INSERT INTO analytics.function_profile VALUES (1, 'urn:foo', 'demo/repo', 'deadbeef', 'foo.py', 'pkg.foo');
        """
    )
    con.close()

    output_dir = tmp_path / "Document Output"
    output_dir.mkdir(parents=True, exist_ok=True)

    @flow
    def _run_export() -> None:
        con_inner = duckdb.connect(str(db_path))
        export_all_parquet(
            con_inner,
            output_dir,
            validate_exports=True,
            schemas=["function_profile"],
        )
        export_all_jsonl(
            con_inner,
            output_dir,
            validate_exports=True,
            schemas=["function_profile"],
        )
        con_inner.close()

    _run_export()
    manifest = output_dir / "index.json"
    if not manifest.exists():
        pytest.fail("Expected manifest not written with validation enabled")


def test_gateway_cache_reuses_instance(tmp_path: Path) -> None:
    """Gateway cache should reuse the same instance for identical configs."""
    _close_gateways()
    db_path = tmp_path / "gw.duckdb"
    gw1 = _get_gateway(
        db_path,
        read_only=False,
        apply_schema=True,
        ensure_views=False,
        validate_schema=False,
    )
    stats_after_first = gateway_cache_stats()
    if gw1 is None:
        pytest.fail("Expected gateway instance from cache builder")
    if stats_after_first["opens"] != 1 or stats_after_first["hits"] != 0:
        pytest.fail(f"Unexpected stats after first open: {stats_after_first}")

    gw2 = _get_gateway(
        db_path,
        read_only=False,
        apply_schema=True,
        ensure_views=False,
        validate_schema=False,
    )
    stats_after_second = gateway_cache_stats()
    if gw1 is not gw2:
        pytest.fail("Expected gateway cache to return the same instance")
    if stats_after_second["hits"] != 1:
        pytest.fail(f"Unexpected stats after reuse: {stats_after_second}")

    _close_gateways()
