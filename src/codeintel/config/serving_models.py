"""Shared serving configuration for FastAPI and MCP surfaces."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import duckdb
from pydantic import BaseModel, Field, model_validator

from codeintel.storage.gateway import StorageGateway

ServingMode = Literal["local_db", "remote_api"]


def _parse_env_flag(value: str | None, *, default: bool) -> bool:
    """
    Interpret a string environment value as a boolean.

    Parameters
    ----------
    value:
        Raw environment variable value or None.
    default:
        Value to return when the environment variable is unset.

    Returns
    -------
    bool
        Parsed boolean flag.
    """
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


class ServingConfig(BaseModel):
    """
    Runtime settings shared by FastAPI and MCP serving layers.

    This model centralizes environment loading and validation for both surfaces to
    ensure consistent repo identity, limits, and backend selection.
    """

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
    commit: str = Field(
        default="HEAD",
        description="Commit SHA represented by this database.",
    )
    db_path: Path | None = Field(
        default=None,
        description="Path to codeintel.duckdb (required when mode='local_db').",
    )
    api_base_url: str | None = Field(
        default=None,
        description="Base URL for remote API mode (required when mode='remote_api').",
    )
    default_limit: int = Field(
        default=50,
        description="Default row limit for tool responses.",
    )
    max_rows_per_call: int = Field(
        default=500,
        description="Hard cap on rows returned in a single call.",
    )
    timeout_seconds: float = Field(
        default=10.0,
        description="Timeout in seconds for remote HTTP calls.",
    )
    read_only: bool = Field(
        default=True,
        description="Whether to open the DuckDB connection in read-only mode.",
    )

    @classmethod
    def from_env(cls) -> ServingConfig:
        """
        Construct a ServingConfig from environment variables.

        Returns
        -------
        ServingConfig
            Validated configuration populated from environment values.
        """
        repo_root = Path(os.environ.get("CODEINTEL_REPO_ROOT", ".")).expanduser().resolve()
        repo = os.environ.get("CODEINTEL_REPO", repo_root.name)
        commit = os.environ.get("CODEINTEL_COMMIT", "HEAD")

        mode_env = os.environ.get("CODEINTEL_MCP_MODE", "local_db").lower()
        mode: ServingMode = "remote_api" if mode_env == "remote_api" else "local_db"

        db_path_env = os.environ.get("CODEINTEL_DB_PATH")
        db_path = Path(db_path_env).expanduser().resolve() if db_path_env else None

        api_base_url = os.environ.get("CODEINTEL_API_BASE_URL")

        default_limit = int(os.environ.get("CODEINTEL_MCP_DEFAULT_LIMIT", "50"))
        max_rows_per_call = int(os.environ.get("CODEINTEL_MCP_MAX_ROWS", "500"))
        timeout_seconds = float(os.environ.get("CODEINTEL_MCP_TIMEOUT_SEC", "10.0"))
        read_only = _parse_env_flag(os.environ.get("CODEINTEL_API_READ_ONLY"), default=True)

        return cls(
            mode=mode,
            repo_root=repo_root,
            repo=repo,
            commit=commit,
            db_path=db_path,
            api_base_url=api_base_url,
            default_limit=default_limit,
            max_rows_per_call=max_rows_per_call,
            timeout_seconds=timeout_seconds,
            read_only=read_only,
        )

    @model_validator(mode="after")
    def _validate_backend(self) -> ServingConfig:
        """
        Apply backend-specific defaults and validation.

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
                self.db_path = self.db_path.expanduser().resolve()
        elif self.mode == "remote_api":
            if not self.api_base_url:
                message = "api_base_url is required when mode='remote_api'"
                raise ValueError(message)
        else:
            message = f"Unsupported serving mode: {self.mode}"
            raise ValueError(message)

        if self.default_limit < 0:
            message = "default_limit must be non-negative"
            raise ValueError(message)
        if self.max_rows_per_call <= 0:
            message = "max_rows_per_call must be positive"
            raise ValueError(message)

        return self


def verify_db_identity(gateway: StorageGateway, cfg: ServingConfig) -> None:
    """
    Ensure the connected DuckDB matches the configured repo and commit.

    Parameters
    ----------
    gateway:
        Storage gateway providing access to the DuckDB connection.
    cfg:
        Serving configuration describing the expected repo and commit.

    Raises
    ------
    ValueError
        If the repo/commit cannot be read or does not match the configuration.
    """
    try:
        row = gateway.con.execute(
            "SELECT repo, commit FROM core.repo_map LIMIT 1",
        ).fetchone()
    except duckdb.Error as exc:
        message = "Failed to read repo identity from core.repo_map"
        raise ValueError(message) from exc

    if row is None:
        message = "core.repo_map is empty; cannot verify repo identity"
        raise ValueError(message)

    db_repo, db_commit = row[0], row[1]
    if db_repo != cfg.repo or db_commit != cfg.commit:
        message = (
            "DuckDB identity mismatch: "
            f"expected {cfg.repo}@{cfg.commit}, found {db_repo}@{db_commit}"
        )
        raise ValueError(message)


__all__ = ["ServingConfig", "ServingMode", "verify_db_identity"]
