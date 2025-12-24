"""Canonical tool identifiers used across the repo."""

from __future__ import annotations

from enum import StrEnum


class ToolName(StrEnum):
    """Supported external tools invoked by the pipeline."""

    PYRIGHT = "pyright"
    PYREFLY = "pyrefly"
    COVERAGE = "coverage"
    RUFF = "ruff"
    PYTEST = "pytest"
    GIT = "git"
    SCIP_PYTHON = "scip-python"
    SCIP = "scip"
    PROTOC = "protoc"


__all__ = ["ToolName"]
