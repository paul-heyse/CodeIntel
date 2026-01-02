"""Span normalization helpers."""

from __future__ import annotations


def normalize_line_span(start_line: int, end_line: int | None = None) -> tuple[int, int]:
    """Normalize line spans to inclusive ranges.

    Parameters
    ----------
    start_line
        Inclusive start line (0-based or 1-based depending on caller contract).
    end_line
        Inclusive end line. Defaults to start_line when omitted.

    Returns
    -------
    tuple[int, int]
        Normalized (start_line, end_line) with end_line >= start_line.
    """
    start_value = int(start_line)
    end_value = int(end_line) if end_line is not None else start_value
    end_value = max(end_value, start_value)
    return start_value, end_value


def normalize_byte_span(
    start_byte: int | None,
    end_byte: int | None,
) -> tuple[int, int] | None:
    """Normalize byte spans to half-open ranges.

    Parameters
    ----------
    start_byte
        Inclusive start byte offset.
    end_byte
        Exclusive end byte offset.

    Returns
    -------
    tuple[int, int] | None
        Normalized half-open span [start, end), or None when inputs are invalid.
    """
    if start_byte is None or end_byte is None:
        return None
    start_value = int(start_byte)
    end_value = int(end_byte)
    if start_value < 0 or end_value < 0:
        return None
    if end_value < start_value:
        return None
    return start_value, end_value


def to_half_open_span(start: int, end_inclusive: int | None = None) -> tuple[int, int]:
    """Normalize inclusive spans to half-open ranges.

    Parameters
    ----------
    start
        Inclusive start position.
    end_inclusive
        Inclusive end position. Defaults to start when omitted.

    Returns
    -------
    tuple[int, int]
        Normalized half-open span [start, end).
    """
    start_value = int(start)
    end_value = int(end_inclusive) if end_inclusive is not None else start_value
    end_value = max(end_value, start_value)
    return start_value, end_value + 1


__all__ = ["normalize_byte_span", "normalize_line_span", "to_half_open_span"]
