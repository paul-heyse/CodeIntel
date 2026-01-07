"""Tool runner fakes for tests that avoid subprocess execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.engine.infrastructure import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunner,
    ToolRunOptions,
    ToolRunResult,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class PresetRunner(ToolRunner):
    """ToolRunner that returns a preset result without invoking subprocesses."""

    def __init__(self, result: ToolRunResult | Exception) -> None:
        super().__init__(tools_config=ToolsConfig.default(), cache_dir=Path("build/.tool_cache"))
        self._result = result

    async def run_async(
        self,
        tool: ToolName | str,
        args: Sequence[str],
        *,
        options: ToolRunOptions | None = None,
    ) -> ToolRunResult:
        run_options = options or ToolRunOptions()
        tool_enum = tool if isinstance(tool, ToolName) else ToolName(str(tool))
        args_tuple = tuple(args)
        if isinstance(self._result, ToolNotFoundError):
            raise ToolNotFoundError(self._result.tool, self._result.configured_path)
        if isinstance(self._result, Exception):
            raise ToolExecutionError(
                make_tool_run_result(
                    tool_enum,
                    args=args_tuple,
                    options=ToolRunResultOptions(
                        returncode=1,
                        stderr="dummy error",
                        output_path=run_options.output_path,
                        duration_s=0.1,
                    ),
                )
            ) from self._result
        return self._result


@dataclass(frozen=True)
class ToolRunResultOptions:
    """Configuration for a fake ToolRunResult."""

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    output_path: Path | None = None
    duration_s: float = 0.0


def make_tool_run_result(
    tool: ToolName | str,
    *,
    args: Sequence[str] | None = None,
    options: ToolRunResultOptions | None = None,
) -> ToolRunResult:
    """Build a ToolRunResult with sensible defaults for tests.

    Returns
    -------
    ToolRunResult
        Constructed tool run result.
    """
    tool_enum = tool if isinstance(tool, ToolName) else ToolName(str(tool))
    opts = options or ToolRunResultOptions()
    return ToolRunResult(
        tool=tool_enum,
        args=tuple(args or ()),
        returncode=opts.returncode,
        stdout=opts.stdout,
        stderr=opts.stderr,
        output_path=opts.output_path,
        duration_s=opts.duration_s,
    )


__all__ = [
    "PresetRunner",
    "ToolRunResultOptions",
    "make_tool_run_result",
]
