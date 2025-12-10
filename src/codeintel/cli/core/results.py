"""Structured result types for CLI handlers.

This module provides the CliResult protocol for handlers that return
structured results, enabling composition, testing, and consistent output.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from codeintel.cli.errors import ProblemDetail
    from codeintel.cli.rendering.types import OutputFormat


# Text renderer protocol: callable that takes data and writer
TextRenderer = Callable[[object, TextIO], None]


@dataclass
class CliResult[T]:
    """Structured result from a CLI handler.

    Encapsulates success/failure status, data payload, and warnings
    for consistent rendering and composition.

    Parameters
    ----------
    success
        Whether the operation completed successfully.
    data
        Result data payload (type varies by handler).
    error
        Problem details if the operation failed.
    warnings
        Non-fatal warnings to display.
    metadata
        Additional metadata about the operation.
    """

    success: bool
    data: T | None = None
    error: ProblemDetail | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation with data, metadata, and warnings.
        """
        result: dict[str, object] = {
            "success": self.success,
        }

        if self.data is not None:
            result["data"] = self._serialize_data(self.data)

        if self.error is not None:
            result["error"] = self.error.to_dict()

        if self.warnings:
            result["warnings"] = self.warnings

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @staticmethod
    def _serialize_data(data: object) -> object:
        """Serialize data for JSON output.

        Returns
        -------
        object
            Serialized representation of the data.
        """
        to_dict = getattr(data, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        if hasattr(data, "__dict__") and not isinstance(data, type):
            return data.__dict__
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to JSON string.

        Parameters
        ----------
        indent
            JSON indentation level (None for compact output).

        Returns
        -------
        str
            JSON representation of the result.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def render(
        self,
        output_format: OutputFormat,
        writer: TextIO = sys.stdout,
        *,
        text_renderer: TextRenderer | None = None,
    ) -> None:
        """Render the result to the specified writer.

        Parameters
        ----------
        output_format
            Output format (TEXT or JSON).
        writer
            Text writer for output (default: stdout).
        text_renderer
            Optional callable for custom text rendering.
        """
        self._write_warnings()

        if self._is_json_format(output_format):
            self._render_json(writer)
        elif text_renderer is not None:
            text_renderer(self.data, writer)
        elif self.data is not None:
            self._render_text(writer)

    def _write_warnings(self) -> None:
        """Write warnings to stderr."""
        for warning in self.warnings:
            sys.stderr.write(f"Warning: {warning}\n")

    @staticmethod
    def _is_json_format(output_format: OutputFormat) -> bool:
        """Check if format is JSON (avoiding circular import).

        Returns
        -------
        bool
            True if format is JSON.
        """
        return output_format.value == "json"

    def _render_json(self, writer: TextIO) -> None:
        """Render as JSON."""
        writer.write(self.to_json())
        writer.write("\n")

    def _render_text(self, writer: TextIO) -> None:
        """Render data as text."""
        data = self.data
        if isinstance(data, str):
            writer.write(data)
            if not data.endswith("\n"):
                writer.write("\n")
        elif isinstance(data, list):
            for item in data:
                writer.write(f"{item}\n")
        elif isinstance(data, dict):
            for key, value in data.items():
                writer.write(f"{key}: {value}\n")
        else:
            writer.write(f"{data}\n")

    @classmethod
    def ok(cls, data: T, *, metadata: dict[str, object] | None = None) -> CliResult[T]:
        """Create a successful result.

        Parameters
        ----------
        data
            Result data payload.
        metadata
            Optional metadata about the operation.

        Returns
        -------
        CliResult[T]
            Successful result with the given data.
        """
        return cls(
            success=True,
            data=data,
            metadata=metadata or {},
        )

    @classmethod
    def fail(
        cls,
        error: ProblemDetail,
        *,
        warnings: list[str] | None = None,
    ) -> CliResult[T]:
        """Create a failed result.

        Parameters
        ----------
        error
            Problem details describing the failure.
        warnings
            Optional warnings to include.

        Returns
        -------
        CliResult[T]
            Failed result with the given error.
        """
        return cls(
            success=False,
            error=error,
            warnings=warnings or [],
        )


__all__ = [
    "CliResult",
    "TextRenderer",
]
