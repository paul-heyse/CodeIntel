"""Helpers for realistic CLI project contexts in tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codeintel.storage import gateway as gateway_pkg
from codeintel.storage.gateway import StorageGateway

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
