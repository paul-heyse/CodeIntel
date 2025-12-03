"""Reusable helpers for test coverage edge computations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from coverage import Coverage

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

from codeintel.analytics.tests import compute_test_coverage_edges
from codeintel.config import ConfigBuilder, TestCoverageStepConfig
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway
from tests._helpers.tooling import CoverageArtifact, generate_coverage_for_function

REPO = "demo/repo"
COMMIT = "deadbeef"
MODULE_IMPORT = "pkg.mod"
FUNCTION_NAME = "func"
TEST_ID = "pkg/mod.py::test_func"


@dataclass
class CoverageEdgeEnv:
    """Environment for computing test coverage edges end-to-end."""

    repo_root: Path
    gateway: StorageGateway
    builder: ConfigBuilder
    module_import: str
    function_name: str
    test_id: str
    function_goid: int
    test_goid: int


@dataclass(frozen=True)
class CoverageSeedConfig:
    """Configuration for seeding coverage edge fixtures."""

    module_import: str = MODULE_IMPORT
    function_name: str = FUNCTION_NAME
    function_urn: str | None = None
    function_qualname: str | None = None
    test_id: str = TEST_ID
    test_urn: str | None = None
    test_qualname: str | None = None
    repo: str = REPO
    commit: str = COMMIT
    function_goid: int = 1
    test_goid: int = 99


def create_coverage_edge_env(
    tmp_path: Path,
    seed: CoverageSeedConfig | None = None,
) -> CoverageEdgeEnv:
    """
    Create repo, gateway, and GOID/catalog seeds for coverage edge tests.

    Returns
    -------
    CoverageEdgeEnv
        Prepared environment with repo, gateway, builder, and seeded GOIDs.
    """
    seed_cfg = seed or CoverageSeedConfig()
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    rel_path = Path(seed_cfg.module_import.replace(".", "/")).with_suffix(".py")
    target_file = repo_root / rel_path
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(
        f"def {seed_cfg.function_name}():\n    return 1\n",
        encoding="utf8",
    )

    gateway = open_gateway(
        StorageConfig(
            db_path=tmp_path / "db.duckdb",
            apply_schema=True,
            ensure_views=True,
            validate_schema=True,
        )
    )
    builder = ConfigBuilder.from_snapshot(
        repo=seed_cfg.repo,
        commit=seed_cfg.commit,
        repo_root=repo_root,
    )
    seed_coverage_rows(gateway=gateway, rel_path=rel_path.as_posix(), seed=seed_cfg)
    return CoverageEdgeEnv(
        repo_root=repo_root,
        gateway=gateway,
        builder=builder,
        module_import=seed_cfg.module_import,
        function_name=seed_cfg.function_name,
        test_id=seed_cfg.test_id,
        function_goid=seed_cfg.function_goid,
        test_goid=seed_cfg.test_goid,
    )


def generate_coverage_artifact(
    env: CoverageEdgeEnv,
    *,
    coverage_file: Path | None = None,
) -> CoverageArtifact:
    """
    Run coverage for the target function under the given test id.

    Returns
    -------
    CoverageArtifact
        Coverage file artifact for use with analytics coverage edges.
    """
    return generate_coverage_for_function(
        repo_root=env.repo_root,
        module_import=env.module_import,
        function_name=env.function_name,
        test_id=env.test_id,
        coverage_file=coverage_file,
    )


def compute_coverage_edges(
    env: CoverageEdgeEnv,
    *,
    coverage_file: Path,
    coverage_loader: Callable[[TestCoverageStepConfig], Coverage | None] | None = None,
) -> None:
    """Invoke coverage edge computation with the seeded catalog and GOIDs."""
    cfg = env.builder.test_coverage(coverage_file=coverage_file)
    compute_test_coverage_edges(
        env.gateway,
        cfg,
        coverage_loader=coverage_loader,
    )


def assert_single_edge(con: DuckDBPyConnection) -> None:
    """
    Assert a single populated test coverage edge exists.

    Parameters
    ----------
    con
        DuckDB connection.

    Raises
    ------
    AssertionError
        If the edge count or contents do not match expectations.
    """
    rows = con.execute(
        "SELECT test_goid_h128, coverage_ratio, last_status FROM analytics.test_coverage_edges"
    ).fetchall()
    if len(rows) != 1:
        message = f"Expected 1 edge row, got {len(rows)}"
        raise AssertionError(message)
    test_goid, cov_ratio, status = rows[0]
    if test_goid is None:
        message = "Expected test_goid_h128 to be populated"
        raise AssertionError(message)
    tolerance = 1e-6
    if abs(float(cov_ratio) - 1.0) > tolerance:
        message = f"Unexpected coverage_ratio {cov_ratio}"
        raise AssertionError(message)
    if status != "passed":
        message = f"Unexpected last_status {status}"
        raise AssertionError(message)


def seed_coverage_rows(
    *,
    gateway: StorageGateway,
    rel_path: str,
    seed: CoverageSeedConfig,
    include_test_catalog: bool = True,
) -> None:
    """
    Seed minimal modules, GOIDs, test catalog, and coverage edges.

    Parameters
    ----------
    gateway
        Gateway whose connection will be mutated.
    rel_path
        Repository-relative path for the target module.
    seed
        Seed configuration controlling GOID/test identifiers.
    include_test_catalog
        When True, insert a matching row into ``analytics.test_catalog``.
    """
    now = datetime.now(UTC)
    function_urn = seed.function_urn or (
        f"goid:{seed.repo}#python:function:{seed.module_import}.{seed.function_name}"
    )
    function_qualname = seed.function_qualname or f"{seed.module_import}.{seed.function_name}"
    test_urn = seed.test_urn or f"goid:{seed.repo}#python:function:{seed.module_import}.test_func"
    test_qualname = seed.test_qualname or f"{seed.module_import}.test_func"
    gateway.con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES (?, ?, ?, ?, 'python', '[]', '[]')
        """,
        [seed.module_import, rel_path, seed.repo, seed.commit],
    )
    gateway.con.executemany(
        """
        INSERT INTO core.goids (
            goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
            start_line, end_line, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                seed.function_goid,
                function_urn,
                seed.repo,
                seed.commit,
                rel_path,
                "python",
                "function",
                function_qualname,
                1,
                2,
                now,
            ),
            (
                seed.test_goid,
                test_urn,
                seed.repo,
                seed.commit,
                rel_path,
                "python",
                "test",
                test_qualname,
                1,
                2,
                now,
            ),
        ],
    )
    if include_test_catalog:
        gateway.con.execute(
            """
            INSERT INTO analytics.test_catalog (
                test_id, rel_path, qualname, repo, commit, status, created_at
            ) VALUES (?, ?, ?, ?, ?, 'passed', ?)
            """,
            [
                seed.test_id,
                rel_path,
                test_qualname,
                seed.repo,
                seed.commit,
                now,
            ],
        )


__all__ = [
    "CoverageEdgeEnv",
    "CoverageSeedConfig",
    "assert_single_edge",
    "compute_coverage_edges",
    "create_coverage_edge_env",
    "generate_coverage_artifact",
    "seed_coverage_rows",
]
