"""MCP server configuration models and environment loading helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, root_validator

McpMode = Literal["local_db", "remote_api"]


class McpServerConfig(BaseModel):
    """
    Configuration for the CodeIntel MCP server.

    This determines:
      - whether we talk directly to DuckDB (`local_db`) or a remote HTTP API
      - which repo/commit the server represents
      - safety limits for tool responses (to avoid flooding context windows)
    """

    mode: McpMode = Field(
        default="local_db",
        description="Backend mode: 'local_db' (DuckDB) or 'remote_api' (FastAPI, future)",
    )

    # Repository identity (mirrors GOID embedding + Document Output) :contentReference[oaicite:1]{index=1}
    repo_root: Path = Field(..., description="Path to the repository root on disk")
    repo: str = Field(..., description="Repository slug, e.g. 'my-org/my-repo'")
    commit: str = Field(..., description="Commit SHA represented by this database")

    # Local DuckDB configuration
    db_path: Path | None = Field(
        default=None,
        description="Path to codeintel.duckdb (required when mode='local_db')",
    )

    # Remote HTTP API configuration (future)
    api_base_url: str | None = Field(
        default=None,
        description="Base URL for the remote CodeIntel API (required when mode='remote_api')",
    )

    # Safety / ergonomics knobs for tools
    default_limit: int = Field(
        default=50,
        description="Default row limit for tools when the caller does not specify one",
    )
    max_rows_per_call: int = Field(
        default=200,
        description="Hard cap on rows returned by a single MCP tool call",
    )
    timeout_seconds: float = Field(
        default=10.0,
        description="Backend timeout (seconds) for HTTP calls in remote_api mode",
    )

    @root_validator(pre=True)
    def _expand_paths(cls, values):
        """Normalize repo_root and db_path to absolute paths early."""
        repo_root = values.get("repo_root") or "."
        if not isinstance(repo_root, Path):
            repo_root = Path(str(repo_root))
        repo_root = repo_root.expanduser().resolve()
        values["repo_root"] = repo_root

        db_path = values.get("db_path")
        if db_path is not None:
            if not isinstance(db_path, Path):
                db_path = Path(str(db_path))
            values["db_path"] = db_path.expanduser().resolve()

        return values

    @root_validator
    def _check_backend(cls, values):
        """Ensure db_path or api_base_url is present for the configured backend mode."""
        mode: McpMode = values.get("mode", "local_db")
        db_path: Path | None = values.get("db_path")
        api_base_url: str | None = values.get("api_base_url")

        if mode == "local_db":
            if db_path is None:
                # If missing, default to <repo_root>/build/db/codeintel.duckdb
                repo_root: Path = values["repo_root"]
                values["db_path"] = (repo_root / "build" / "db" / "codeintel.duckdb").resolve()
        elif mode == "remote_api":
            if not api_base_url:
                raise ValueError("api_base_url is required when mode='remote_api'")
        else:
            raise ValueError(f"Unsupported McpMode: {mode}")

        return values

    @classmethod
    def from_env(cls) -> McpServerConfig:
        """
        Build a server config from environment variables.

        Environment variables:
          - CODEINTEL_REPO_ROOT      (default: ".")
          - CODEINTEL_REPO           (default: basename(CODEINTEL_REPO_ROOT))
          - CODEINTEL_COMMIT         (default: "HEAD")
          - CODEINTEL_DB_PATH        (default: <repo_root>/build/db/codeintel.duckdb)
          - CODEINTEL_API_BASE_URL   (for remote_api mode)
          - CODEINTEL_MCP_MODE       ("local_db" | "remote_api", default "local_db")
          - CODEINTEL_MCP_DEFAULT_LIMIT (int, default 50)
          - CODEINTEL_MCP_MAX_ROWS      (int, default 200)
          - CODEINTEL_MCP_TIMEOUT_SEC   (float, default 10.0)
        """
        repo_root = Path(
            os.environ.get("CODEINTEL_REPO_ROOT", ".")
        ).expanduser().resolve()

        repo = os.environ.get("CODEINTEL_REPO", repo_root.name)
        commit = os.environ.get("CODEINTEL_COMMIT", "HEAD")

        mode_env = os.environ.get("CODEINTEL_MCP_MODE", "local_db")
        mode: McpMode = "remote_api" if mode_env == "remote_api" else "local_db"

        db_path_env = os.environ.get("CODEINTEL_DB_PATH")
        db_path = Path(db_path_env).expanduser().resolve() if db_path_env else None

        api_base_url = os.environ.get("CODEINTEL_API_BASE_URL")

        default_limit = int(os.environ.get("CODEINTEL_MCP_DEFAULT_LIMIT", "50"))
        max_rows_per_call = int(os.environ.get("CODEINTEL_MCP_MAX_ROWS", "200"))
        timeout_seconds = float(os.environ.get("CODEINTEL_MCP_TIMEOUT_SEC", "10.0"))

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
        )
