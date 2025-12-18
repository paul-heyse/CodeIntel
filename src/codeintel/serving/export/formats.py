"""Canonical export format registry for CodeIntel serving.

This module is the single source of truth for:
- Supported export formats
- MIME types per format
- File suffixes per format

It is safe to import from both HTTP and FastMCP surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, cast

ExportFormat = Literal["json", "ndjson", "parquet", "arrow"]


@dataclass(frozen=True, slots=True)
class ExportFormatSpec:
    """Specification for a supported export format.

    Parameters
    ----------
    format
        Export format identifier.
    mime_type
        MIME type for the serialized payload.
    suffix
        File suffix (including leading dot).
    """

    format: ExportFormat
    mime_type: str
    suffix: str


EXPORT_FORMATS: Final[dict[ExportFormat, ExportFormatSpec]] = {
    "json": ExportFormatSpec(format="json", mime_type="application/json", suffix=".json"),
    "ndjson": ExportFormatSpec(format="ndjson", mime_type="application/x-ndjson", suffix=".ndjson"),
    "parquet": ExportFormatSpec(
        format="parquet", mime_type="application/vnd.apache.parquet", suffix=".parquet"
    ),
    "arrow": ExportFormatSpec(
        format="arrow", mime_type="application/vnd.apache.arrow.file", suffix=".arrow"
    ),
}

_EXPORT_FORMAT_ORDER: Final[tuple[ExportFormat, ...]] = ("ndjson", "json", "parquet", "arrow")
_TEXT_EXPORT_FORMATS: Final[frozenset[ExportFormat]] = frozenset({"json", "ndjson"})
_BINARY_EXPORT_FORMATS: Final[frozenset[ExportFormat]] = frozenset({"parquet", "arrow"})


def mime_type_for_export_format(fmt: ExportFormat) -> str:
    """Return the MIME type for an export format.

    Parameters
    ----------
    fmt
        Export format.

    Returns
    -------
    str
        MIME type for the format.
    """
    return EXPORT_FORMATS[fmt].mime_type


def suffix_for_export_format(fmt: ExportFormat) -> str:
    """Return the file suffix for an export format.

    Parameters
    ----------
    fmt
        Export format.

    Returns
    -------
    str
        File suffix for the format.
    """
    return EXPORT_FORMATS[fmt].suffix


def normalize_export_format(fmt: str) -> ExportFormat:
    """Validate and normalize an export format string.

    Parameters
    ----------
    fmt
        Raw export format string.

    Returns
    -------
    ExportFormat
        Normalized format value.

    Raises
    ------
    ValueError
        If the format is unsupported.
    """
    normalized = fmt.strip().lower()
    if normalized in EXPORT_FORMATS:
        return cast("ExportFormat", normalized)
    msg = f"Unsupported export format: {fmt}"
    raise ValueError(msg)


def export_format_choices() -> tuple[ExportFormat, ...]:
    """Return supported export formats in a stable, UX-friendly order."""
    return _EXPORT_FORMAT_ORDER


def default_export_format() -> ExportFormat:
    """Return the default export format for interactive clients."""
    return "ndjson"


def is_text_export_format(fmt: ExportFormat) -> bool:
    """Return True when the export format is a text payload (JSON/NDJSON)."""
    return fmt in _TEXT_EXPORT_FORMATS


def supports_preview(fmt: ExportFormat) -> bool:
    """Return True when `codeintel://exports/{id}/preview` is supported for the format."""
    return is_text_export_format(fmt)


def supports_line_chunks(fmt: ExportFormat) -> bool:
    """Return True when line-chunk resources are supported for the format.

    Notes
    -----
    Line chunking is row-based and therefore only supported for NDJSON.
    """
    return fmt == "ndjson"


def supports_byte_chunks(fmt: ExportFormat) -> bool:
    """Return True when byte-range chunk resources are supported for the format."""
    return fmt in _BINARY_EXPORT_FORMATS


def is_binary_export_format(fmt: ExportFormat) -> bool:
    """Return True when the export format is binary (Parquet/Arrow)."""
    return fmt in _BINARY_EXPORT_FORMATS


__all__ = [
    "EXPORT_FORMATS",
    "ExportFormat",
    "ExportFormatSpec",
    "default_export_format",
    "export_format_choices",
    "is_binary_export_format",
    "is_text_export_format",
    "mime_type_for_export_format",
    "normalize_export_format",
    "suffix_for_export_format",
    "supports_byte_chunks",
    "supports_line_chunks",
    "supports_preview",
]
