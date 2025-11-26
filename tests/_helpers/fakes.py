"""Typed fakes for ingestion and analytics tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Protocol

from anyio import to_thread
from coverage import Coverage

from codeintel.config.models import TestCoverageConfig
from codeintel.ingestion.tool_runner import ToolName, ToolResult, ToolRunner


def _mkdir_parents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf8")


class FakeCoverageData:
    """Lightweight coverage data implementing measured_files/contexts_by_lineno."""

    def __init__(self, contexts_by_file: dict[str, dict[int, set[str]]]) -> None:
        self._contexts_by_file = contexts_by_file

    def measured_files(self) -> list[str]:
        """
        Return measured file paths.

        Returns
        -------
        list[str]
            File paths observed in coverage data.
        """
        return list(self._contexts_by_file.keys())

    def contexts_by_lineno(self, filename: str) -> dict[int, set[str]]:
        """
        Return contexts keyed by line number for a file.

        Parameters
        ----------
        filename
            File path to resolve contexts for.

        Returns
        -------
        dict[int, set[str]]
            Mapping of line numbers to context identifiers.
        """
        return self._contexts_by_file.get(filename, {})


class FakeCoverage:
    """Coverage shim providing deterministic statements/contexts."""

    def __init__(
        self,
        statements: dict[str, list[int]],
        contexts: dict[str, dict[int, set[str]]],
    ) -> None:
        self._statements = statements
        self._contexts = contexts

    def analysis2(self, filename: str) -> tuple[str, list[int], list[int], list[int], list[int]]:
        stmts = self._statements.get(filename, [])
        return filename, stmts, [], [], stmts

    def get_data(self) -> FakeCoverageData:
        """
        Return deterministic coverage data wrapper.

        Returns
        -------
        FakeCoverageData
            Coverage data exposing measured files and contexts.
        """
        return FakeCoverageData(self._contexts)


class CoverageLoader(Protocol):
    """Protocol for injecting coverage loaders."""

    def __call__(self, cfg: TestCoverageConfig | object) -> Coverage:
        """Return a Coverage-compatible object."""
        raise NotImplementedError


class FakeToolRunner(ToolRunner):
    """ToolRunner stub returning canned payloads."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        payloads: dict[str, Any] | None = None,
        on_run: Callable[[ToolName, list[str]], None] | None = None,
    ) -> None:
        super().__init__(cache_dir=cache_dir)
        self.payloads = payloads or {}
        self.calls: list[tuple[ToolName, list[str]]] = []
        self.on_run = on_run

    async def run_async(
        self,
        tool: ToolName | str,
        args: list[str],
        *,
        cwd: Path | None = None,
        output_path: Path | None = None,
        timeout_s: float | None = None,
    ) -> ToolResult:
        """
        Execute a tool invocation with canned outputs.

        Returns
        -------
        ToolResult
            Structured result capturing stdout/stderr and codes.
        """
        _ = cwd
        _ = timeout_s
        tool_enum = tool if isinstance(tool, ToolName) else ToolName(str(tool))
        args_list = list(args)
        self.calls.append((tool_enum, args_list))
        if self.on_run is not None:
            self.on_run(tool_enum, args_list)
        payload_stdout = self.payloads.get(tool_enum.value, "")
        if output_path is not None and tool_enum in {ToolName.COVERAGE, ToolName.PYREFLY}:
            json_payload = self.payloads.get(
                f"{tool_enum.value}_json",
                self.payloads.get("json", {}),
            )
            await to_thread.run_sync(_mkdir_parents, output_path.parent)
            await to_thread.run_sync(_write_text, output_path, json.dumps(json_payload))
        return ToolResult(
            tool=tool_enum,
            args=tuple(args_list),
            returncode=0,
            stdout=str(payload_stdout),
            stderr="",
            output_path=output_path,
            duration_s=0.0,
        )


@dataclass(frozen=True)
class FakeScipResult:
    """SCIP result stand-in mirroring dataclass fields."""

    status: str = "success"
    index_scip: Path | None = None
    index_json: Path | None = None
    reason: str | None = None


def write_dummy_scip_files(base_dir: Path, *, index_content: str = "[]") -> tuple[Path, Path]:
    """
    Create minimal SCIP artifacts for tests.

    Returns
    -------
    tuple[Path, Path]
        Paths to index.scip and index.scip.json.
    """
    scip_dir = base_dir / "scip"
    scip_dir.mkdir(parents=True, exist_ok=True)
    index_scip = scip_dir / "index.scip"
    index_json = scip_dir / "index.scip.json"
    index_scip.write_text("scip-binary", encoding="utf8")
    index_json.write_text(index_content, encoding="utf8")
    return index_scip, index_json


def utcnow() -> datetime:
    """
    Return timezone-aware now for deterministic tests.

    Returns
    -------
    datetime
        Current timezone-aware datetime.
    """
    return datetime.now().astimezone()
