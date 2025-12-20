"""Orchestration helpers for validated Document Output exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from codeintel.build.exports.common import ExportCallOptions
from codeintel.build.exports.jsonl import export_all_jsonl
from codeintel.build.exports.parquet import export_all_parquet
from codeintel.core.config.settings import ExportAuditSettings
from codeintel.storage.validation import validate_contract_or_raise

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from codeintel.storage.gateway import StorageGateway


def _validate_dataset_contract(gateway: StorageGateway) -> None:
    """Validate dataset contract using gateway connection.

    Parameters
    ----------
    gateway
        StorageGateway with active connection.
    """
    validate_contract_or_raise(gateway.con)


class Exporter(Protocol):
    """Protocol for export callables used by the runner."""

    def __call__(
        self,
        gateway: StorageGateway,
        document_output_dir: Path,
        *,
        settings: ExportAuditSettings,
        options: ExportCallOptions | None = None,
    ) -> None:
        """Execute an export call.

        Parameters
        ----------
        gateway
            StorageGateway for data access.
        document_output_dir
            Target directory for exports.
        settings
            Export audit settings.
        options
            Export options.
        """
        ...


class JsonlExporter(Protocol):
    """Protocol for JSONL export callables that return written paths."""

    def __call__(
        self,
        gateway: StorageGateway,
        document_output_dir: Path,
        *,
        settings: ExportAuditSettings,
        options: ExportCallOptions | None = None,
    ) -> list[Path]:
        """Execute an export call and return emitted file paths.

        Parameters
        ----------
        gateway
            StorageGateway for data access.
        document_output_dir
            Target directory for exports.
        settings
            Export audit settings.
        options
            Export options.

        Returns
        -------
        list[Path]
            List of written file paths.
        """
        ...


class ExportRunner(Protocol):
    """Protocol for higher-level export runners invoked by CLI or pipeline."""

    def __call__(
        self,
        *,
        gateway: StorageGateway,
        output_dir: Path,
        options: ExportOptions | None = None,
    ) -> list[Path]:
        """Run exports and return emitted file paths.

        Parameters
        ----------
        gateway
            StorageGateway for data access.
        output_dir
            Target directory for exports.
        options
            Export options.

        Returns
        -------
        list[Path]
            List of written file paths.
        """
        ...


@dataclass
class ExportOptions:
    """Options controlling export validation and dataset selection."""

    export: ExportCallOptions = field(default_factory=ExportCallOptions)
    settings: ExportAuditSettings = field(default_factory=ExportAuditSettings)
    validator: Callable[[StorageGateway], None] = _validate_dataset_contract
    export_parquet_fn: Exporter = field(default=export_all_parquet)
    export_jsonl_fn: JsonlExporter = field(default=export_all_jsonl)


def run_validated_exports(
    *,
    gateway: StorageGateway,
    output_dir: Path,
    options: ExportOptions | None = None,
) -> list[Path]:
    """Validate registry and emit Parquet/JSONL exports.

    Parameters
    ----------
    gateway
        StorageGateway providing datasets and connection metadata.
    output_dir
        Document Output directory for emitted artifacts.
    options
        ExportOptions controlling validation, schemas, and dataset selection.

    Returns
    -------
    list[Path]
        Paths written by the JSONL export (Parquet exports are written for
        side effects).
    """
    opts = options or ExportOptions()
    opts.validator(gateway)
    opts.export_parquet_fn(
        gateway,
        output_dir,
        settings=opts.settings,
        options=opts.export,
    )
    return opts.export_jsonl_fn(
        gateway,
        output_dir,
        settings=opts.settings,
        options=opts.export,
    )


__all__ = [
    "ExportOptions",
    "ExportRunner",
    "Exporter",
    "JsonlExporter",
    "run_validated_exports",
]
