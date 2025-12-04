"""Entrypoints test orchestration functions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from codeintel.storage.gateway import StorageGateway
from tests._helpers.builders import GoidRow, ModuleRow
from tests._helpers.configs.coverage_config import CoverageSeedConfig
from tests._helpers.row_protocol import insert_rows


@dataclass(frozen=True)
class AppSeeds:
    """Seed data for pkg.app entrypoints and coverage."""

    hello_goid: int
    cli_goid: int
    hello_urn: str
    cli_urn: str
    repo: str
    commit: str


def seed_app_modules_and_goids(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    hello_goid: int = 1001,
    cli_goid: int = 1002,
) -> AppSeeds:
    """Insert modules and GOIDs for pkg.app entrypoints.

    Parameters
    ----------
    gateway
        Storage gateway for database operations.
    repo
        Repository identifier.
    commit
        Commit hash.
    hello_goid
        GOID for the hello function.
    cli_goid
        GOID for the cli function.

    Returns
    -------
    AppSeeds
        Seed metadata including GOIDs/URNs for hello and cli.
    """
    now = datetime.now(UTC)
    insert_rows(
        gateway,
        [ModuleRow(module="pkg.app", path="pkg/app.py", repo=repo, commit=commit)],
    )
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
        repo=seeds.repo,
        commit=seeds.commit,
        function_goid=seeds.hello_goid,
        test_goid=seeds.hello_goid + 1000,
    )


__all__ = [
    "AppSeeds",
    "make_coverage_seed_from_app",
    "seed_app_modules_and_goids",
]
