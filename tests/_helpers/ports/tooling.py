"""Tooling port protocol aligned with ingestion tool adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from codeintel.ingestion.ports.tools import DiagnosticResult, ScipResult, ScipRunRequest

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@runtime_checkable
class ToolingPort(Protocol):
    """Protocol for external tool execution used in tests."""

    async def run_pyright(
        self,
        repo_root: Path,
        *,
        paths: Sequence[Path] | None = None,
    ) -> DiagnosticResult:
        """Run pyright and return diagnostics."""
        ...

    async def run_pyrefly(
        self,
        repo_root: Path,
        *,
        paths: Sequence[Path] | None = None,
    ) -> DiagnosticResult:
        """Run pyrefly and return diagnostics."""
        ...

    async def run_ruff(
        self,
        repo_root: Path,
        *,
        paths: Sequence[Path] | None = None,
    ) -> DiagnosticResult:
        """Run ruff and return diagnostics."""
        ...

    async def run_scip(self, request: ScipRunRequest) -> ScipResult:
        """Run SCIP indexing and return parsed results."""
        ...


__all__ = ["ToolingPort"]
