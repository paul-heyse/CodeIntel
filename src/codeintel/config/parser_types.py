"""Shared parser type enums for function analytics."""

from __future__ import annotations

from enum import StrEnum


class FunctionParserKind(StrEnum):
    """Supported parsers for function analytics."""

    PYTHON = "python"
