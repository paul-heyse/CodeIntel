"""Configuration helpers for opening DuckDB storage gateways."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codeintel.storage.validation.mode import ContractValidationMode

__all__ = ["StorageConfig"]


def _default_validation_summary_path(db_path: Path) -> Path | None:
    if str(db_path) == ":memory:":
        return None
    name = f"{db_path.stem}.contract_validation.json"
    return db_path.with_name(name)


@dataclass(frozen=True)
class StorageConfig:
    """Define configuration for opening a CodeIntel DuckDB database."""

    db_path: Path
    read_only: bool = False
    apply_schema: bool = False
    ensure_views: bool = False
    validate_schema: bool = False
    validation_mode: ContractValidationMode = ContractValidationMode.LENIENT
    validation_summary_path: Path | None = None
    attach_history: bool = False
    history_db_path: Path | None = None
    attach_meta: bool = True
    meta_db_path: Path | None = None
    repo: str | None = None
    commit: str | None = None

    @classmethod
    def for_ingest(
        cls,
        db_path: Path,
        *,
        history_db_path: Path | None = None,
        validation_mode: ContractValidationMode = ContractValidationMode.LENIENT,
        validation_summary_path: Path | None = None,
        attach_meta: bool = True,
    ) -> StorageConfig:
        """
        Build a write-capable configuration used by ingestion and analytics runs.

        Parameters
        ----------
        db_path
            Primary DuckDB database path.
        history_db_path
            Optional history database to attach for cross-commit analytics.
        validation_mode
            Contract validation behavior when opening the gateway.
        validation_summary_path
            Optional path to write validation summaries.
        attach_meta
            Whether to attach the meta database for metadata tables.

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
            validation_mode=validation_mode,
            validation_summary_path=validation_summary_path
            or _default_validation_summary_path(db_path),
            attach_history=history_db_path is not None,
            history_db_path=history_db_path,
            attach_meta=attach_meta,
        )

    @classmethod
    def for_readonly(
        cls,
        db_path: Path,
        *,
        validation_mode: ContractValidationMode = ContractValidationMode.LENIENT,
        validation_summary_path: Path | None = None,
        attach_meta: bool = True,
    ) -> StorageConfig:
        """
        Build a read-only configuration for serving/inspection surfaces.

        Parameters
        ----------
        db_path
            DuckDB database path to open read-only.
        validation_mode
            Contract validation behavior when opening the gateway.
        validation_summary_path
            Optional path to write validation summaries.
        attach_meta
            Whether to attach the meta database for metadata tables.

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
            validation_mode=validation_mode,
            validation_summary_path=validation_summary_path
            or _default_validation_summary_path(db_path),
            attach_meta=attach_meta,
        )
