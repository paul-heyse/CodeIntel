"""Typed fakes for ingestion and analytics tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from coverage import Coverage

from codeintel.config.models import TestCoverageConfig
from codeintel.ingestion.tool_runner import ToolName, ToolResult, ToolRunner


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


class FakeCoverage(Coverage):
    """Coverage shim providing deterministic statements/contexts."""

    def __init__(
        self,
        statements: dict[str, list[int]],
        contexts: dict[str, dict[int, set[str]]],
    ) -> None:
        super().__init__()
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

    def run(
        self,
        tool: ToolName,
        args: list[str],
        *,
        cwd: Path | None = None,
        output_path: Path | None = None,
    ) -> ToolResult:
        """
        Execute a tool invocation with canned outputs.

        Returns
        -------
        ToolResult
            Structured result capturing stdout/stderr and codes.
        """
        _ = cwd
        self.calls.append((tool, args))
        if self.on_run is not None:
            self.on_run(tool, args)
        payload_stdout = self.payloads.get(tool, "")
        if tool == "coverage" and output_path is not None:
            json_payload = self.payloads.get("json", {})
            output_path.write_text(json.dumps(json_payload), encoding="utf8")
        return ToolResult(
            tool=tool,
            returncode=0,
            stdout=str(payload_stdout),
            stderr="",
            output_path=output_path,
            duration_s=0.0,
        )

    @staticmethod
    def load_json(path: Path) -> dict[str, object] | None:
        """
        Load JSON payload from a path when present.

        Returns
        -------
        dict[str, object] | None
            Parsed JSON content or None when missing.
        """
        if not path.is_file():
            return None
        content = path.read_text(encoding="utf8")
        if not content:
            return {}
        return ToolRunner.load_json(path)


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
