"""Helpers for realistic CLI project contexts in tests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.storage import gateway as gateway_pkg
from tests.cli._harness import CliTestHarness

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.storage.gateway import StorageGateway
    from tests.cli._harness import CliInvocationResult

PROJECT_FILENAME = "codeintel.yaml"


@dataclass
class CLIProjectContext:
    """Context for a temporary CLI project used in tests."""

    repo_root: Path
    build_dir: Path
    db_path: Path
    cfg_path: Path
    env: dict[str, str]
    gateway: StorageGateway | None


def _write_project_file(repo_root: Path, repo: str, commit: str, db_rel_path: Path) -> Path:
    cfg_path = repo_root / PROJECT_FILENAME
    cfg_path.write_text(
        "\n".join(
            [
                f"repo: {repo}",
                f"commit: {commit}",
                "default_profile: default",
                "storage:",
                f"  db_path: {db_rel_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    return cfg_path


def create_cli_project(tmp_path: Path, *, repo: str, commit: str) -> CLIProjectContext:
    """Create a temporary CLI project with real DuckDB backing.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.
    repo
        Repository slug to record in the project config.
    commit
        Commit hash to record in the project config.

    Returns
    -------
    CLIProjectContext
        Paths, environment variables, and an open storage gateway with schemas applied.
    """
    repo_root = tmp_path / "repo"
    build_dir = repo_root / "build"
    db_path = build_dir / "db" / "codeintel.duckdb"
    repo_root.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    cfg_path = _write_project_file(repo_root, repo, commit, db_path.relative_to(repo_root))

    env = {
        "CODEINTEL_REPO_ROOT": str(repo_root),
        "CODEINTEL_BUILD_DIR": str(build_dir),
    }

    gateway = gateway_pkg.open_gateway(gateway_pkg.StorageConfig.for_ingest(db_path))

    return CLIProjectContext(
        repo_root=repo_root,
        build_dir=build_dir,
        db_path=db_path,
        cfg_path=cfg_path,
        env=env,
        gateway=gateway,
    )


@dataclass
class CLIProjectHarness:
    """Harness wrapper for CLI project-backed invocations."""

    ctx: CLIProjectContext
    harness: CliTestHarness

    def _prepare(self) -> None:
        """Ensure gateways are closed before invocation to avoid cache leakage."""
        if self.ctx.gateway is not None:
            self.ctx.gateway.close()
            self.ctx.gateway = None

    def invoke(self, args: list[str]) -> CliInvocationResult:
        """Invoke CLI with project env and cwd configured.

        Returns
        -------
        CliInvocationResult
            Captured invocation result.
        """
        self._prepare()
        return self.harness.invoke(args)

    def invoke_json(self, args: list[str]) -> dict[str, object]:
        """Invoke CLI and parse JSON output.

        Returns
        -------
        dict[str, object]
            Parsed JSON response.
        """
        self._prepare()
        return self.harness.invoke_json(args)


@contextmanager
def cli_project_harness(
    tmp_path: Path,
    *,
    repo: str = "demo/repo",
    commit: str = "deadbeef",
) -> Iterator[CLIProjectHarness]:
    """Context manager yielding a CLIProjectHarness.

    Parameters
    ----------
    tmp_path
        Temporary directory root for the project.
    repo
        Repository slug.
    commit
        Commit hash.

    Yields
    ------
    CLIProjectHarness
        Harness configured for the temporary project.
    """
    ctx = create_cli_project(tmp_path, repo=repo, commit=commit)
    harness = CliTestHarness().with_env(**ctx.env).with_cwd(ctx.repo_root)
    project_harness = CLIProjectHarness(ctx=ctx, harness=harness)
    try:
        yield project_harness
    finally:
        if ctx.gateway is not None:
            ctx.gateway.close()
