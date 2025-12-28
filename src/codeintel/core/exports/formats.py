"""Canonical export format registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, cast

ExportFormat = Literal["jsonl", "json", "parquet", "arrow"]
CanonicalExportFormat = ExportFormat


@dataclass(frozen=True, slots=True)
class ExportFormatSpec:
    """Specification for a supported export format.

    Parameters
    ----------
    format
        Canonical export format identifier.
    mime_type
        MIME type for the serialized payload.
    suffix
        File suffix (including leading dot).
    aliases
        Optional alias strings that normalize to this format.
    """

    format: CanonicalExportFormat
    mime_type: str
    suffix: str
    aliases: tuple[str, ...] = ()


EXPORT_FORMATS: Final[dict[CanonicalExportFormat, ExportFormatSpec]] = {
    "jsonl": ExportFormatSpec(
        format="jsonl",
        mime_type="application/x-ndjson",
        suffix=".jsonl",
    ),
    "json": ExportFormatSpec(format="json", mime_type="application/json", suffix=".json"),
    "parquet": ExportFormatSpec(
        format="parquet",
        mime_type="application/vnd.apache.parquet",
        suffix=".parquet",
    ),
    "arrow": ExportFormatSpec(
        format="arrow",
        mime_type="application/vnd.apache.arrow.stream",
        suffix=".arrow",
    ),
}

_EXPORT_FORMAT_ORDER: Final[tuple[CanonicalExportFormat, ...]] = (
    "arrow",
    "parquet",
    "jsonl",
    "json",
)
_TEXT_EXPORT_FORMATS: Final[frozenset[CanonicalExportFormat]] = frozenset({"jsonl", "json"})
_BINARY_EXPORT_FORMATS: Final[frozenset[CanonicalExportFormat]] = frozenset({"parquet", "arrow"})


def normalize_export_format(fmt: str) -> CanonicalExportFormat:
    """Validate and normalize an export format string.

    Parameters
    ----------
    fmt
        Raw export format string.

    Returns
    -------
    CanonicalExportFormat
        Normalized canonical format value.

    Raises
    ------
    ValueError
        If the format is unsupported.
    """
    normalized = fmt.strip().lower()
    if normalized in EXPORT_FORMATS:
        return cast("CanonicalExportFormat", normalized)
    msg = f"Unsupported export format: {fmt}"
    raise ValueError(msg)


def resolve_export_format_spec(fmt: str) -> ExportFormatSpec:
    """Return the ExportFormatSpec for a raw format string.

    Returns
    -------
    ExportFormatSpec
        Specification for the normalized export format.
    """
    return EXPORT_FORMATS[normalize_export_format(fmt)]


def mime_type_for_export_format(fmt: str) -> str:
    """Return the MIME type for an export format.

    Returns
    -------
    str
        MIME type for the export format.
    """
    return resolve_export_format_spec(fmt).mime_type


def suffix_for_export_format(fmt: str) -> str:
    """Return the file suffix for an export format.

    Returns
    -------
    str
        File suffix for the export format.
    """
    return resolve_export_format_spec(fmt).suffix


def export_format_choices() -> tuple[CanonicalExportFormat, ...]:
    """Return supported export formats in a stable, UX-friendly order.

    Returns
    -------
    tuple[CanonicalExportFormat, ...]
        Ordered export format identifiers.
    """
    return _EXPORT_FORMAT_ORDER


def default_export_format() -> CanonicalExportFormat:
    """Return the canonical default export format.

    Returns
    -------
    CanonicalExportFormat
        Default export format identifier.
    """
    return "arrow"


def is_text_export_format(fmt: str) -> bool:
    """Return True when the export format is a text payload (JSON/JSONL).

    Returns
    -------
    bool
        True when the format is text-based.
    """
    return normalize_export_format(fmt) in _TEXT_EXPORT_FORMATS


def supports_preview(fmt: str) -> bool:
    """Return True when preview endpoints are available for the format.

    Returns
    -------
    bool
        True when preview endpoints are supported.
    """
    return is_text_export_format(fmt)


def supports_line_chunks(fmt: str) -> bool:
    """Return True when line-chunk resources are supported for the format.

    Returns
    -------
    bool
        True when line-chunk resources are supported.
    """
    return normalize_export_format(fmt) == "jsonl"


def supports_byte_chunks(fmt: str) -> bool:
    """Return True when byte-range chunk resources are supported.

    Returns
    -------
    bool
        True when byte-range chunk resources are supported.
    """
    return normalize_export_format(fmt) in _BINARY_EXPORT_FORMATS


def is_binary_export_format(fmt: str) -> bool:
    """Return True when the export format is binary (Parquet/Arrow).

    Returns
    -------
    bool
        True when the format is binary.
    """
    return normalize_export_format(fmt) in _BINARY_EXPORT_FORMATS


__all__ = [
    "EXPORT_FORMATS",
    "CanonicalExportFormat",
    "ExportFormat",
    "ExportFormatSpec",
    "default_export_format",
    "export_format_choices",
    "is_binary_export_format",
    "is_text_export_format",
    "mime_type_for_export_format",
    "normalize_export_format",
    "resolve_export_format_spec",
    "suffix_for_export_format",
    "supports_byte_chunks",
    "supports_line_chunks",
    "supports_preview",
]
