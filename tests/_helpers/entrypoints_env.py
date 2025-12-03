"""Helpers for entrypoint/dependency analytics tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from codeintel.storage.gateway import StorageGateway
from tests._helpers.coverage_env import CoverageSeedConfig


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

    Returns
    -------
    AppSeeds
        Seed metadata including GOIDs/URNs for hello and cli.
    """
    now = datetime.now(UTC)
    gateway.con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES ('pkg.app', 'pkg/app.py', ?, ?, 'python', '[]', '[]')
        """,
        [repo, commit],
    )
    gateway.con.executemany(
        """
        INSERT INTO core.goids (
            goid_h128, urn, repo, commit, rel_path, language, kind, qualname, start_line, end_line,
            created_at
        ) VALUES (?, ?, ?, ?, ?, 'python', 'function', ?, ?, ?, ?)
        """,
        [
            (
                hello_goid,
                f"goid:{repo}#python:function:pkg.app.hello",
                repo,
                commit,
                "pkg/app.py",
                "pkg.app.hello",
                9,
                15,
                now,
            ),
            (
                cli_goid,
                f"goid:{repo}#python:function:pkg.app.cli",
                repo,
                commit,
                "pkg/app.py",
                "pkg.app.cli",
                17,
                23,
                now,
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
    "CoverageSeedConfig",
    "make_coverage_seed_from_app",
    "seed_app_modules_and_goids",
]
