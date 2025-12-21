"""Canonical export format registry for CodeIntel serving."""

from __future__ import annotations

from codeintel.core.exports import formats as core_formats

ExportFormat = core_formats.ExportFormat
ExportFormatSpec = core_formats.ExportFormatSpec
resolve_export_format_spec = core_formats.resolve_export_format_spec
mime_type_for_export_format = core_formats.mime_type_for_export_format
suffix_for_export_format = core_formats.suffix_for_export_format
is_text_export_format = core_formats.is_text_export_format
is_binary_export_format = core_formats.is_binary_export_format
supports_preview = core_formats.supports_preview
supports_line_chunks = core_formats.supports_line_chunks
supports_byte_chunks = core_formats.supports_byte_chunks

_CORE_EXPORT_FORMATS = core_formats.EXPORT_FORMATS
_NDJSON_SPEC = ExportFormatSpec(
    format=_CORE_EXPORT_FORMATS["jsonl"].format,
    mime_type=_CORE_EXPORT_FORMATS["jsonl"].mime_type,
    suffix=".ndjson",
    aliases=_CORE_EXPORT_FORMATS["jsonl"].aliases,
)

EXPORT_FORMATS: dict[ExportFormat, ExportFormatSpec] = {
    **_CORE_EXPORT_FORMATS,
    "ndjson": _NDJSON_SPEC,
}


def normalize_export_format(fmt: str) -> ExportFormat:
    """Normalize export format for serving (keep ndjson as canonical).

    Returns
    -------
    ExportFormat
        Canonical export format identifier for serving.
    """
    normalized = core_formats.normalize_export_format(fmt)
    if normalized == "jsonl":
        return "ndjson"
    return normalized


def default_export_format() -> ExportFormat:
    """Return the serving-default export format.

    Returns
    -------
    ExportFormat
        Default export format identifier.
    """
    return "ndjson"


def export_format_choices() -> tuple[ExportFormat, ...]:
    """Return supported export formats in a serving-friendly order.

    Returns
    -------
    tuple[ExportFormat, ...]
        Ordered export format identifiers.
    """
    return ("ndjson", "json", "parquet", "arrow")


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
    "resolve_export_format_spec",
    "suffix_for_export_format",
    "supports_byte_chunks",
    "supports_line_chunks",
    "supports_preview",
]
