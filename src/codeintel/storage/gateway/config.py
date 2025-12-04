"""Configuration helpers for opening DuckDB storage gateways."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["StorageConfig"]


@dataclass(frozen=True)
class StorageConfig:
    """Define configuration for opening a CodeIntel DuckDB database."""

    db_path: Path
    read_only: bool = False
    apply_schema: bool = False
    ensure_views: bool = False
    validate_schema: bool = False
    attach_history: bool = False
    history_db_path: Path | None = None
    repo: str | None = None
    commit: str | None = None

    @classmethod
    def for_ingest(
        cls,
        db_path: Path,
        *,
        history_db_path: Path | None = None,
    ) -> StorageConfig:
        """
        Build a write-capable configuration used by ingestion and analytics runs.

        Parameters
        ----------
        db_path
            Primary DuckDB database path.
        history_db_path
            Optional history database to attach for cross-commit analytics.

        Returns
        -------
        StorageConfig
            Preconfigured ingest-ready storage configuration.
        """
        return cls(
            db_path=db_path,
            read_only=False,
            apply_schema=True,
            ensure_views=True,
            validate_schema=True,
            attach_history=history_db_path is not None,
            history_db_path=history_db_path,
        )

    @classmethod
    def for_readonly(cls, db_path: Path) -> StorageConfig:
        """
        Build a read-only configuration for serving/inspection surfaces.

        Parameters
        ----------
        db_path
            DuckDB database path to open read-only.

        Returns
        -------
        StorageConfig
            Preconfigured read-only storage configuration.
        """
        return cls(
            db_path=db_path,
            read_only=True,
            apply_schema=False,
            ensure_views=True,
            validate_schema=True,
        )
