"""Entrypoints test orchestration functions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.primitives import SnapshotRef
from tests._helpers.assertions import ModulesAssertions
from tests._helpers.configs.coverage_config import CoverageSeedConfig
from tests._helpers.fixtures.rows import GoidRow, ModuleRow, RepoMapRow, insert_rows
from tests._helpers.fixtures.snapshots import SnapshotVariant
from tests._helpers.modules_expectations import modules_expected_from_repo_tree

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class AppSeeds:
    """Seed data for pkg.app entrypoints and coverage."""

    hello_goid: int
    cli_goid: int
    hello_urn: str
    cli_urn: str
    repo: str
    commit: str


@dataclass(frozen=True)
class AppSeedSpec:
    """Configuration for seeding pkg.app entrypoints."""

    repo: str
    commit: str
    hello_goid: int = 1001
    cli_goid: int = 1002
    repo_root: Path | None = None
    module_map: dict[str, str] | None = None


def seed_app_modules_and_goids(
    gateway: StorageGateway,
    spec: AppSeedSpec,
) -> AppSeeds:
    """Insert modules and GOIDs for pkg.app entrypoints.

    Parameters
    ----------
    gateway
        Storage gateway for database operations.
    spec
        Entrypoint seed configuration.

    Returns
    -------
    AppSeeds
        Seed metadata including GOIDs/URNs for hello and cli.
    """
    repo = spec.repo
    commit = spec.commit
    hello_goid = spec.hello_goid
    cli_goid = spec.cli_goid
    repo_root = spec.repo_root
    module_map = spec.module_map
    now = datetime.now(UTC)
    resolved_module_map = module_map
    if resolved_module_map is None and repo_root is not None:
        path_map = modules_expected_from_repo_tree(repo_root)
        resolved_module_map = {module: path for path, module in path_map.items()}
    if resolved_module_map is None:
        resolved_module_map = {"pkg.app": "pkg/app.py"}
    insert_rows(
        gateway,
        [
            ModuleRow(module=module, path=path, repo=repo, commit=commit)
            for module, path in sorted(resolved_module_map.items())
        ],
    )
    gateway.con.execute(
        "DELETE FROM core.repo_map WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    insert_rows(
        gateway,
        [
            RepoMapRow(
                repo=repo,
                commit=commit,
                modules=resolved_module_map,
            )
        ],
    )
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=repo_root or Path.cwd())
    ModulesAssertions(gateway, snapshot).inventory_consistent()
    insert_rows(
        gateway,
        [
            GoidRow(
                goid_h128=hello_goid,
                urn=f"goid:{repo}#python:function:pkg.app.hello",
                repo=repo,
                commit=commit,
                rel_path="pkg/app.py",
                kind="function",
                qualname="pkg.app.hello",
                start_line=9,
                end_line=15,
                created_at=now,
            ),
            GoidRow(
                goid_h128=cli_goid,
                urn=f"goid:{repo}#python:function:pkg.app.cli",
                repo=repo,
                commit=commit,
                rel_path="pkg/app.py",
                kind="function",
                qualname="pkg.app.cli",
                start_line=17,
                end_line=23,
                created_at=now,
            ),
        ],
    )
    return AppSeeds(
        hello_goid=hello_goid,
        cli_goid=cli_goid,
        hello_urn=f"goid:{repo}#python:function:pkg.app.hello",
        cli_urn=f"goid:{repo}#python:function:pkg.app.cli",
        repo=repo,
        commit=commit,
    )


def make_coverage_seed_from_app(seeds: AppSeeds) -> CoverageSeedConfig:
    """Build a CoverageSeedConfig for pkg.app.hello with a standard test id.

    Parameters
    ----------
    seeds
        Application seed data.

    Returns
    -------
    CoverageSeedConfig
        Coverage seed configuration for the hello function.
    """
    return CoverageSeedConfig(
        module_import="pkg.app",
        function_name="hello",
        test_id="tests.test_app::test_hello",
        snapshot_variant=SnapshotVariant(repo=seeds.repo, commit=seeds.commit),
        function_goid=seeds.hello_goid,
        test_goid=seeds.hello_goid + 1000,
    )


__all__ = [
    "AppSeeds",
    "make_coverage_seed_from_app",
    "seed_app_modules_and_goids",
]
