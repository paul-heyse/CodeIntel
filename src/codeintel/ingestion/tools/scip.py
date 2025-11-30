"""SCIP plugin for the ingestion tool runtime."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from anyio import to_thread

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.tool_runner import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunner,
    ToolRunResult,
)
from codeintel.ingestion.tools.plugins import (
    ToolPlugin,
    ToolPluginMetadata,
    ToolPluginResult,
    ToolStatus,
)

log = logging.getLogger(__name__)


def _mkdir_parents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf8")


def _resolve_target_base(repo_root: Path, target_dir: Path | None) -> Path:
    if target_dir is not None:
        return target_dir
    src_dir = repo_root / "src"
    return src_dir if src_dir.is_dir() else repo_root


@dataclass
class ScipPlugin(ToolPlugin):
    """Plugin for SCIP indexing via scip-python + scip CLI."""

    runner: ToolRunner
    tools_config: ToolsConfig
    metadata: ToolPluginMetadata = field(
        default_factory=lambda: ToolPluginMetadata(
            name="scip",
            produces_artifacts=("index_scip", "index_json"),
            consumes_configs=("scip_python_bin", "scip_bin"),
            datasets=("core.scip_symbols", "core.goid_crosswalk"),
        )
    )

    async def run(
        self,
        *,
        repo_root: Path,
        **kwargs: object,
    ) -> ToolPluginResult:
        """
        Run scip-python index and scip print to produce SCIP + JSON.

        When rel_paths is provided, only those paths are targeted; otherwise
        the full repo (or target_dir/src) is indexed.

        Returns
        -------
        ToolPluginResult
            Normalized execution result from the SCIP plugin.
        """
        output_scip_obj = kwargs.get("output_scip")
        output_json_obj = kwargs.get("output_json")
        target_dir_obj = kwargs.get("target_dir")
        rel_paths_obj = kwargs.get("rel_paths")

        if not isinstance(output_scip_obj, Path):
            message = "scip plugin requires output_scip of type Path"
            raise TypeError(message)
        if not isinstance(output_json_obj, Path):
            message = "scip plugin requires output_json of type Path"
            raise TypeError(message)
        if target_dir_obj is not None and not isinstance(target_dir_obj, Path):
            message = "scip plugin requires target_dir to be Path or None"
            raise TypeError(message)
        if rel_paths_obj is not None and not isinstance(rel_paths_obj, Sequence):
            message = "scip plugin requires rel_paths to be a sequence of strings"
            raise TypeError(message)

        output_scip = output_scip_obj
        output_json = output_json_obj
        target_dir = target_dir_obj
        rel_paths = tuple(rel_paths_obj) if rel_paths_obj is not None else None
        try:
            await self._run_scip_python(
                repo_root,
                output_scip=output_scip,
                target_dir=target_dir,
                rel_paths=rel_paths,
            )
        except ToolNotFoundError as exc:
            log.warning("scip-python binary not found; SCIP index cannot be built")
            return ToolPluginResult(
                tool=ToolName.SCIP_PYTHON,
                status=ToolStatus.NOT_FOUND,
                artifacts={},
                run=None,
                error=exc,
            )
        except ToolExecutionError as exc:
            return ToolPluginResult(
                tool=ToolName.SCIP_PYTHON,
                status=ToolStatus.ERROR,
                artifacts={"index_scip": output_scip},
                run=exc.result,
                error=exc,
            )

        try:
            print_result = await self._run_scip_print(output_scip, output_json)
        except ToolNotFoundError as exc:
            log.warning("scip binary not found; JSON export cannot be built")
            return ToolPluginResult(
                tool=ToolName.SCIP,
                status=ToolStatus.NOT_FOUND,
                artifacts={"index_scip": output_scip},
                run=None,
                error=exc,
            )
        except ToolExecutionError as exc:
            return ToolPluginResult(
                tool=ToolName.SCIP,
                status=ToolStatus.ERROR,
                artifacts={"index_scip": output_scip, "index_json": output_json},
                run=exc.result,
                error=exc,
            )

        artifacts = {
            "index_scip": output_scip,
            "index_json": output_json,
        }
        return ToolPluginResult(
            tool=ToolName.SCIP,
            status=ToolStatus.OK,
            artifacts=artifacts,
            run=print_result,
            error=None,
        )

    async def _run_scip_python(
        self,
        repo_root: Path,
        *,
        output_scip: Path,
        target_dir: Path | None,
        rel_paths: Sequence[str] | None,
    ) -> ToolRunResult:
        target_base = await to_thread.run_sync(_resolve_target_base, repo_root, target_dir)
        await to_thread.run_sync(_mkdir_parents, output_scip.parent)

        args: list[str] = ["index", str(target_base), "--output", str(output_scip)]
        for rel_path in rel_paths or ():
            args.extend(["--target-only", rel_path])

        result = await self.runner.run_async(
            ToolName.SCIP_PYTHON,
            args,
            cwd=repo_root,
            output_path=output_scip,
            timeout_s=self.tools_config.default_timeout_s,
        )
        if not result.ok:
            raise ToolExecutionError(result)
        return result

    async def _run_scip_print(self, scip_path: Path, output_json: Path) -> ToolRunResult:
        args = ["print", "--json", str(scip_path)]
        await to_thread.run_sync(_mkdir_parents, output_json.parent)
        result = await self.runner.run_async(
            ToolName.SCIP,
            args,
            cwd=scip_path.parent,
            output_path=output_json,
            timeout_s=self.tools_config.default_timeout_s,
        )
        if not result.ok:
            raise ToolExecutionError(result)
        await to_thread.run_sync(_write_text, output_json, result.stdout or "")
        return result
