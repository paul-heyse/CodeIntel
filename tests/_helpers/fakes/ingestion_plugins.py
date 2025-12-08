"""Fakes for ingestion plugins and steps.

These helpers provide typed doubles for the ingestion plugins' adapter and
step dependencies so tests can assert wiring without monkeypatching.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from codeintel.build.protocols import TypeCheckResult, TypeDiagnostic
from codeintel.ingestion.compute.base import StepResult
from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord
from codeintel.ingestion.ports.storage import IngestStoragePort
from codeintel.ingestion.ports.tools import (
    CoverageResult,
    DiagnosticEntry,
    DiagnosticResult,
    IngestToolPort,
    ToolStatus,
)
from tests._helpers.fakes.storage import FakeIngestStorage


@dataclass
class StepCallCapture:
    """Capture object for recording step invocations."""

    storage: object | None = None
    discovery: object | None = None
    modules: list[ModuleRecord] = field(default_factory=list)
    repo: str | None = None
    commit: str | None = None
    repo_root: Path | None = None
    run_diagnostics: bool | None = None
    tool_port: IngestToolPort | None = None


class RecordingStorageAdapter(FakeIngestStorage):
    """IngestStoragePort implementation that records the gateway passed in."""

    def __init__(self, gateway: object, capture: StepCallCapture | None = None) -> None:
        super().__init__()
        self.gateway = gateway
        self.capture = capture
        if capture is not None:
            capture.storage = self


class RecordingDiscoveryAdapter(ModuleDiscoveryPort):
    """Discovery adapter that records the repo root and serves canned sources."""

    def __init__(
        self,
        repo_root: Path,
        module_sources: dict[str, str] | None = None,
        capture: StepCallCapture | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.module_sources = module_sources or {}
        self.capture = capture
        if capture is not None:
            capture.discovery = self
            capture.repo_root = repo_root

    def discover_modules(self, repo_root: Path, profile: object) -> Sequence[ModuleRecord]:
        _ = profile
        if repo_root != self.repo_root:
            return ()
        return [
            ModuleRecord(
                rel_path=rel_path,
                module_name=rel_path.replace("/", ".").removesuffix(".py"),
                file_path=self.repo_root / rel_path,
                index=idx + 1,
                total=len(self.module_sources),
            )
            for idx, rel_path in enumerate(self.module_sources)
        ]

    def read_module_source(self, record: ModuleRecord) -> str | None:
        return self.module_sources.get(record.rel_path)

    @staticmethod
    def file_exists(path: Path) -> bool:
        return path.exists()

    @staticmethod
    def read_text(path: Path, encoding: str = "utf-8") -> str | None:
        try:
            return path.read_text(encoding=encoding)
        except OSError:
            return None


class RecordingStep:
    """Synchronous step double that captures calls and returns canned results."""

    def __init__(
        self,
        storage: IngestStoragePort,
        discovery: ModuleDiscoveryPort,
        *,
        capture: StepCallCapture | None = None,
        table_key: str = "core.fake",
        result: StepResult | None = None,
    ) -> None:
        self.storage = storage
        self.discovery = discovery
        self.capture = capture or StepCallCapture()
        self.table_key = table_key
        self._result = result
        self.capture.storage = storage
        self.capture.discovery = discovery

    def execute(
        self,
        modules: Sequence[ModuleRecord],
        *,
        repo: str,
        commit: str,
    ) -> StepResult:
        self.capture.modules = list(modules)
        self.capture.repo = repo
        self.capture.commit = commit
        if self._result is not None:
            return self._result
        return StepResult.ok(table_counts={self.table_key: len(modules)})


class RecordingAsyncStep:
    """Async step double for TypingIngestPlugin."""

    def __init__(
        self,
        storage: IngestStoragePort,
        discovery: ModuleDiscoveryPort,
        tools: IngestToolPort | None,
        *,
        capture: StepCallCapture | None = None,
        result: StepResult | None = None,
    ) -> None:
        self.storage = storage
        self.discovery = discovery
        self.tools = tools
        self.capture = capture or StepCallCapture()
        self.table_key = "analytics.typedness"
        self._result = result
        self.capture.storage = storage
        self.capture.discovery = discovery
        self.capture.tool_port = tools

    async def execute_async(
        self,
        modules: Sequence[ModuleRecord],
        *,
        repo: str,
        commit: str,
        repo_root: str,
        run_diagnostics: bool = True,
    ) -> StepResult:
        self.capture.modules = list(modules)
        self.capture.repo = repo
        self.capture.commit = commit
        self.capture.repo_root = Path(repo_root)
        self.capture.run_diagnostics = run_diagnostics
        if self._result is not None:
            return self._result
        return StepResult.ok(table_counts={self.table_key: len(modules)})


class RecordingTypeChecker:
    """Minimal TypeChecker that returns a canned result."""

    def __init__(self, *, success: bool = True) -> None:
        self.success = success

    async def check(
        self,
        repo_root: Path,
        *,
        paths: Sequence[Path] | None = None,
        config_path: Path | None = None,
    ) -> object:
        _ = repo_root, paths, config_path

        return TypeCheckResult(
            success=self.success,
            diagnostics=(
                TypeDiagnostic(
                    path="pkg/sample.py",
                    line=1,
                    character=0,
                    severity="info",
                    code="info",
                    message="ok",
                    source="pyright",
                ),
            ),
        )


class RecordingToolPort(IngestToolPort):
    """Tool port double that can return deterministic results."""

    def __init__(self, status: ToolStatus = ToolStatus.OK) -> None:
        self.status = status

    async def run_pyright(self, repo_root: Path) -> object:
        _ = repo_root

        return DiagnosticResult(
            status=self.status,
            diagnostics=(
                DiagnosticEntry(
                    path="pkg/sample.py",
                    line=1,
                    column=0,
                    severity="info",
                    code="info",
                    message="ok",
                ),
            ),
        )

    async def run_pyrefly(self, repo_root: Path) -> object:
        return await self.run_pyright(repo_root)

    async def run_ruff(self, repo_root: Path) -> object:
        return await self.run_pyright(repo_root)

    async def run_coverage(
        self,
        repo_root: Path,
        *,
        coverage_file: Path | None = None,
        output_path: Path | None = None,
    ) -> object:
        _ = repo_root, coverage_file, output_path

        return CoverageResult(status=self.status, files=())


def make_type_checker_factory(
    checker: TypeChecker | None,
) -> Callable[[TypeChecker | None], TypeChecker | None]:
    """Return a factory suitable for TypingIngestPlugin.

    Returns
    -------
    Callable[[TypeChecker | None], TypeChecker | None]
        Factory that always yields the provided checker.
    """

    def _factory(_: TypeChecker | None) -> TypeChecker | None:
        return checker

    return _factory


__all__ = [
    "RecordingAsyncStep",
    "RecordingDiscoveryAdapter",
    "RecordingStep",
    "RecordingStorageAdapter",
    "RecordingToolPort",
    "RecordingTypeChecker",
    "StepCallCapture",
    "make_type_checker_factory",
]
