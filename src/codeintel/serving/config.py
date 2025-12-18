"""Serving identity configuration.

This module defines a small, serving-owned configuration surface used by CLI/runtime
resolution and by entrypoints that need repo/db identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ServingMode = Literal["local_db", "remote_api"]


def normalize_optional_path(path: str | Path | None) -> Path | None:
    """Normalize a possibly missing path into a resolved Path.

    Parameters
    ----------
    path
        Raw path value or None.

    Returns
    -------
    Path | None
        Resolved Path when provided; otherwise None.
    """
    if path is None:
        return None
    return Path(path).expanduser().resolve()


class ServingConfig(BaseModel):
    """Configuration required to perform serving/API operations.

    Parameters
    ----------
    mode
        Backend mode: ``local_db`` for local DuckDB access, ``remote_api`` for a remote HTTP API.
    repo_root
        Repository root on disk.
    repo
        Repository slug (e.g. ``org/repo``).
    commit
        Commit SHA represented by the snapshot/database.
    db_path
        DuckDB database path when in ``local_db`` mode.
    api_base_url
        Base URL when in ``remote_api`` mode.
    read_only
        Whether the local DB connection should be treated as read-only.
    """

    model_config = ConfigDict(extra="forbid")

    mode: ServingMode = Field(
        default="local_db",
        description="Backend mode: 'local_db' for DuckDB or 'remote_api' for HTTP passthrough.",
    )
    repo_root: Path = Field(
        default_factory=lambda: Path().resolve(),
        description="Absolute path to the repository root on disk.",
    )
    repo: str = Field(
        default="",
        description="Repository slug, e.g. 'my-org/my-repo'. Defaults to repo_root name.",
    )
    commit: str = Field(default="HEAD", description="Commit SHA represented by this database.")
    db_path: Path | None = None
    api_base_url: str | None = None
    read_only: bool = True

    @model_validator(mode="after")
    def _validate_backend(self) -> ServingConfig:
        """Apply backend-specific defaults and validation.

        Returns
        -------
        ServingConfig
            Normalized configuration with required fields set.

        Raises
        ------
        ValueError
            When required values are missing for the chosen mode.
        """
        if not self.repo:
            self.repo = self.repo_root.name

        if self.mode == "local_db":
            if self.db_path is None:
                self.db_path = (self.repo_root / "build" / "db" / "codeintel.duckdb").resolve()
            else:
                self.db_path = normalize_optional_path(self.db_path)
        elif self.mode == "remote_api":
            if not self.api_base_url:
                msg = "api_base_url is required when mode='remote_api'"
                raise ValueError(msg)
        else:
            msg = f"Unsupported serving mode: {self.mode}"
            raise ValueError(msg)

        return self


__all__ = ["ServingConfig", "ServingMode", "normalize_optional_path"]
