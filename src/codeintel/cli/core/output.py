"""Structured output protocols for pipeable CLI composition.

This module provides stdin reading helpers and structured output envelopes
for CLI command composition via pipes.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from codeintel.cli.rendering.types import OutputFormat


@dataclass
class OutputEnvelope:
    """Structured CLI output envelope for pipeable composition.

    Wraps command output with metadata for downstream processing.

    Parameters
    ----------
    data
        Result data payload (list of records or single object).
    metadata
        Additional metadata about the operation.
    """

    data: list[dict[str, object]] | dict[str, object]
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary with data and metadata.
        """
        result: dict[str, object] = {"data": self.data}
        if self.metadata:
            result["meta"] = self.metadata
        return result

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to JSON string.

        Parameters
        ----------
        indent
            JSON indentation level (None for compact output).

        Returns
        -------
        str
            JSON representation.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def write(
        self,
        output_format: OutputFormat,
        writer: TextIO = sys.stdout,
    ) -> None:
        """Write the envelope to the specified writer.

        Parameters
        ----------
        output_format
            Output format (TEXT or JSON).
        writer
            Text writer for output (default: stdout).
        """
        if output_format.value == "json":
            writer.write(self.to_json())
            writer.write("\n")
        else:
            self._write_text(writer)

    def _write_text(self, writer: TextIO) -> None:
        """Render data as human-readable text."""
        if isinstance(self.data, list):
            for item in self.data:
                if isinstance(item, dict):
                    for key, value in item.items():
                        writer.write(f"{key}: {value}\n")
                    writer.write("\n")
                else:
                    writer.write(f"{item}\n")
        elif isinstance(self.data, dict):
            for key, value in self.data.items():
                writer.write(f"{key}: {value}\n")
        else:
            writer.write(f"{self.data}\n")


def read_stdin_records() -> list[dict[str, object]]:
    """Read JSON or JSONL records from stdin.

    Supports both a single JSON array/object and JSONL (one JSON per line).

    Returns
    -------
    list[dict[str, object]]
        List of parsed records.
    """
    content = sys.stdin.read().strip()
    if not content:
        return []

    # Try JSONL first (one JSON object per line)
    if "\n" in content:
        return _parse_jsonl(content)

    # Fall back to single JSON array/object
    data = json.loads(content)
    if isinstance(data, list):
        return [_ensure_dict(item) for item in data]
    return [_ensure_dict(data)]


def iter_stdin_records() -> Iterator[dict[str, object]]:
    """Iterate over JSON/JSONL records from stdin.

    Memory-efficient streaming parser for large inputs.

    Yields
    ------
    dict[str, object]
        Each parsed record.
    """
    for raw_line in sys.stdin:
        stripped_line = raw_line.strip()
        if stripped_line:
            data = json.loads(stripped_line)
            yield _ensure_dict(data)


def _parse_jsonl(content: str) -> list[dict[str, object]]:
    """Parse JSONL content (one JSON object per line).

    Returns
    -------
    list[dict[str, object]]
        List of parsed records.
    """
    records = []
    for raw_line in content.splitlines():
        stripped_line = raw_line.strip()
        if stripped_line:
            data = json.loads(stripped_line)
            records.append(_ensure_dict(data))
    return records


def _ensure_dict(data: object) -> dict[str, object]:
    """Ensure data is a dictionary.

    Returns
    -------
    dict[str, object]
        The data as a dictionary.
    """
    if isinstance(data, dict):
        return data
    return {"value": data}


def merge_stdin_with_args(
    stdin_record: dict[str, object],
    cli_args: dict[str, object],
) -> dict[str, object]:
    """Merge stdin record with CLI arguments.

    CLI arguments take precedence over stdin values.

    Parameters
    ----------
    stdin_record
        Record read from stdin.
    cli_args
        Arguments provided via CLI flags.

    Returns
    -------
    dict[str, object]
        Merged arguments.
    """
    result = dict(stdin_record)
    # CLI args override stdin values (filter out None values)
    result.update({k: v for k, v in cli_args.items() if v is not None})
    return result


__all__ = [
    "OutputEnvelope",
    "iter_stdin_records",
    "merge_stdin_with_args",
    "read_stdin_records",
]
