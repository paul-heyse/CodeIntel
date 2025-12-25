"""Capability probes for tool integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.tools import ToolName
from codeintel.ingestion.engine.infrastructure import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolRunOptions,
)

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.config.models import ToolsConfig
    from codeintel.ingestion.engine.infrastructure import ToolRunner


_PYTEST_JSON_REPORT_FLAG = "--json-report"


@dataclass(frozen=True)
class ToolCapability:
    """Capability probe result."""

    name: str
    supported: bool
    state: str
    detail: str | None = None


class ToolCapabilityProbe:
    """Probe optional tool capabilities with caching."""

    def __init__(self, *, runner: ToolRunner, tools_config: ToolsConfig) -> None:
        self._runner = runner
        self._tools_config = tools_config
        self._cache: dict[str, ToolCapability] = {}

    async def pytest_json_report(self, *, repo_root: Path) -> ToolCapability:
        """Probe pytest for json-report support.

        Returns
        -------
        ToolCapability
            Capability result describing pytest json-report support.
        """
        key = "pytest_json_report"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        capability = await self._probe_pytest_json_report(repo_root=repo_root)
        if capability.state != "unknown":
            self._cache[key] = capability
        return capability

    async def _probe_pytest_json_report(self, *, repo_root: Path) -> ToolCapability:
        try:
            result = await self._runner.run_async(
                ToolName.PYTEST,
                ["--help"],
                options=ToolRunOptions(
                    cwd=repo_root,
                    timeout_s=self._tools_config.default_timeout_s,
                ),
            )
        except ToolNotFoundError as exc:
            return ToolCapability(
                name="pytest_json_report",
                supported=False,
                state="missing_tool",
                detail=str(exc),
            )
        except ToolExecutionError as exc:
            return ToolCapability(
                name="pytest_json_report",
                supported=False,
                state="unknown",
                detail=exc.result.stderr or str(exc),
            )
        if not result.ok:
            return ToolCapability(
                name="pytest_json_report",
                supported=False,
                state="unknown",
                detail=result.stderr,
            )
        if not result.stdout.strip():
            return ToolCapability(
                name="pytest_json_report",
                supported=False,
                state="unknown",
                detail="pytest --help returned empty output",
            )
        if _PYTEST_JSON_REPORT_FLAG in result.stdout:
            return ToolCapability(
                name="pytest_json_report",
                supported=True,
                state="supported",
                detail=None,
            )
        return ToolCapability(
            name="pytest_json_report",
            supported=False,
            state="unsupported",
            detail="pytest-json-report flag not present",
        )


__all__ = [
    "ToolCapability",
    "ToolCapabilityProbe",
]
