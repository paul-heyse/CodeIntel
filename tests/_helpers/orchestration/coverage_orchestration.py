"""Coverage test environment orchestration functions."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.analytics.testing import compute_test_coverage_edges
from codeintel.analytics.testing.coverage.edges import TestCoverageOptions
from codeintel.config import ConfigBuilder, SnapshotInit
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageConfig, open_gateway
from tests._helpers.builders import GoidRow, ModuleRow, TestCatalogRow, insert_rows
from tests._helpers.configs.coverage_config import CoverageEdgeEnv, CoverageSeedConfig
from tests._helpers.orchestration.tooling import generate_coverage_for_function

if TYPE_CHECKING:
    from collections.abc import Callable

    from coverage import Coverage

    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.orchestration.tooling import CoverageArtifact


def create_coverage_edge_env(
    tmp_path: Path,
    seed: CoverageSeedConfig | None = None,
) -> CoverageEdgeEnv:
    """Create repo, gateway, and GOID/catalog seeds for coverage edge tests.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.
    seed
        Optional seed configuration; uses defaults if not provided.

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
        snapshot=SnapshotInit(repo=seed_cfg.repo, commit=seed_cfg.commit, repo_root=repo_root),
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
    """Run coverage for the target function under the given test id.

    Parameters
    ----------
    env
        Coverage edge environment.
    coverage_file
        Optional path for coverage output file.

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
    coverage_loader: Callable[[SnapshotRef, Path | None], Coverage | None] | None = None,
) -> None:
    """Invoke coverage edge computation with the seeded catalog and GOIDs.

    Parameters
    ----------
    env
        Coverage edge environment.
    coverage_file
        Path to the coverage data file.
    coverage_loader
        Optional custom coverage loader function.
    """
    snapshot = env.builder.snapshot
    options = TestCoverageOptions(
        coverage_file=coverage_file,
        coverage_loader=coverage_loader,
    )
    compute_test_coverage_edges(
        env.gateway,
        snapshot,
        options=options,
    )


def seed_coverage_rows(
    *,
    gateway: StorageGateway,
    rel_path: str,
    seed: CoverageSeedConfig,
    include_test_catalog: bool = True,
) -> None:
    """Seed minimal modules, GOIDs, test catalog, and coverage edges.

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

    # Seed modules
    insert_rows(
        gateway,
        [
            ModuleRow(
                module=seed.module_import,
                path=rel_path,
                repo=seed.repo,
                commit=seed.commit,
            )
        ],
    )

    # Seed GOIDs for function and test
    insert_rows(
        gateway,
        [
            GoidRow(
                goid_h128=seed.function_goid,
                urn=function_urn,
                repo=seed.repo,
                commit=seed.commit,
                rel_path=rel_path,
                kind="function",
                qualname=function_qualname,
                start_line=1,
                end_line=2,
                created_at=now,
            ),
            GoidRow(
                goid_h128=seed.test_goid,
                urn=test_urn,
                repo=seed.repo,
                commit=seed.commit,
                rel_path=rel_path,
                kind="test",
                qualname=test_qualname,
                start_line=1,
                end_line=2,
                created_at=now,
            ),
        ],
    )

    # Seed test catalog if requested
    if include_test_catalog:
        insert_rows(
            gateway,
            [
                TestCatalogRow(
                    test_id=seed.test_id,
                    repo=seed.repo,
                    commit=seed.commit,
                    rel_path=rel_path,
                    qualname=test_qualname,
                    status="passed",
                    created_at=now,
                )
            ],
        )


__all__ = [
    "compute_coverage_edges",
    "create_coverage_edge_env",
    "generate_coverage_artifact",
    "seed_coverage_rows",
]
