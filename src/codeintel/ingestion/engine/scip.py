"""SCIP plugin for the ingestion tool runtime."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, cast

from anyio import to_thread

from codeintel.ingestion.engine.infrastructure import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunOptions,
    ToolSpec,
)
from codeintel.ingestion.engine.plugins import (
    ToolPlugin,
    ToolPluginMetadata,
    ToolPluginResult,
    ToolStatus,
)
from codeintel.ingestion.engine.results import ScipDocument, ScipIndexResult, ScipOccurrence
from codeintel.ingestion.scip.cli import build_scip_python_args, ensure_pip_available
from codeintel.ingestion.scip.index_store import write_index_proto
from codeintel.ingestion.scip.paths import resolve_target_base
from codeintel.ingestion.scip.proto import load_generated_module
from codeintel.ingestion.scip.proto_types import ScipProtoModule
from codeintel.ingestion.scip.protobuf_parser import parse_index, rebase_parsed_index

if TYPE_CHECKING:
    from codeintel.config.models import ToolsConfig
    from codeintel.ingestion.engine.infrastructure import (
        ToolRunner,
        ToolRunResult,
    )

log = logging.getLogger(__name__)


def _mkdir_parents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_scip_index(
    scip_path: Path,
    proto_module_path: Path,
    repo_root: Path,
) -> ScipIndexResult:
    """Parse index.scip via protobuf into a ScipIndexResult.

    Returns
    -------
    ScipIndexResult
        Parsed index result with documents and occurrences.
    """
    parsed = parse_index(scip_path, proto_module_path)
    parsed = rebase_parsed_index(parsed, repo_root)
    documents = []
    for doc in parsed.documents:
        occurrences = tuple(
            ScipOccurrence(
                symbol=occ.symbol,
                range_=(
                    occ.range_start_line,
                    occ.range_start_col,
                    occ.range_end_line,
                    occ.range_end_col,
                ),
                symbol_roles=occ.symbol_roles,
                position_encoding=occ.position_encoding,
                text_document_encoding=occ.text_document_encoding,
                start_byte=occ.start_byte,
                end_byte=occ.end_byte,
            )
            for occ in doc.occurrences
        )
        documents.append(
            ScipDocument(
                relative_path=doc.relative_path,
                symbols=doc.symbols,
                occurrences=occurrences,
                position_encoding=doc.position_encoding,
                text_document_encoding=doc.text_document_encoding,
            )
        )
    return ScipIndexResult.from_documents(tuple(documents), index_scip_path=scip_path)


def _is_no_git_failure(result: ToolRunResult) -> bool:
    combined = f"{result.stdout}\n{result.stderr}".lower()
    return "not a git repository" in combined or "git: not found" in combined


def _write_empty_scip_index(output_scip: Path, proto_module_path: Path) -> None:
    module = cast("ScipProtoModule", load_generated_module(proto_module_path))
    empty_index = module.Index()
    write_index_proto(empty_index, output_scip)


@dataclass
class ScipPlugin(ToolPlugin):
    """Plugin for SCIP indexing via scip-python."""

    runner: ToolRunner
    tools_config: ToolsConfig
    metadata: ToolPluginMetadata = field(
        default_factory=lambda: ToolPluginMetadata(
            name="scip-python",
            produces_artifacts=("index_scip",),
            consumes_configs=("scip_python_bin",),
            datasets=("core.scip_symbols", "core.goid_crosswalk"),
            spec=ToolSpec(
                required_kwargs=("output_scip", "proto_module_path"),
                optional_kwargs=("target_dir", "rel_paths", "environment_json", "timeout_s"),
            ),
        )
    )

    def _validate_kwargs(
        self,
        kwargs: dict[str, object],
    ) -> tuple[
        Path,
        Path,
        Path | None,
        tuple[str, ...] | None,
        Path | None,
        float | None,
    ]:
        """Validate and extract keyword arguments for run method.

        Parameters
        ----------
        kwargs
            Keyword arguments to validate.

        Returns
        -------
        tuple[Path, Path, Path | None, tuple[str, ...] | None, Path | None, float | None]
            Tuple of (output_scip, proto_module_path, target_dir, rel_paths,
            environment_json, timeout_s).

        Raises
        ------
        TypeError
            If required arguments are missing or of wrong type.
        """
        _ = self
        output_scip_obj = kwargs.get("output_scip")
        target_dir_obj = kwargs.get("target_dir")
        rel_paths_obj = kwargs.get("rel_paths")
        proto_module_obj = kwargs.get("proto_module_path")
        environment_obj = kwargs.get("environment_json")
        timeout_obj = kwargs.get("timeout_s")

        if not isinstance(output_scip_obj, Path):
            message = "scip-python plugin requires output_scip of type Path"
            raise TypeError(message)
        if target_dir_obj is not None and not isinstance(target_dir_obj, Path):
            message = "scip-python plugin requires target_dir to be Path or None"
            raise TypeError(message)
        if rel_paths_obj is not None and not isinstance(rel_paths_obj, Sequence):
            message = "scip-python plugin requires rel_paths to be a sequence of strings"
            raise TypeError(message)
        if not isinstance(proto_module_obj, Path):
            message = "scip-python plugin requires proto_module_path of type Path"
            raise TypeError(message)
        if environment_obj is not None and not isinstance(environment_obj, Path):
            message = "scip-python plugin requires environment_json to be Path or None"
            raise TypeError(message)
        if timeout_obj is not None and not isinstance(timeout_obj, Real):
            message = "scip-python plugin requires timeout_s to be a number or None"
            raise TypeError(message)

        rel_paths = tuple(rel_paths_obj) if rel_paths_obj is not None else None
        timeout_s = float(timeout_obj) if isinstance(timeout_obj, Real) else None
        return (
            output_scip_obj,
            proto_module_obj,
            target_dir_obj,
            rel_paths,
            environment_obj,
            timeout_s,
        )

    async def run(
        self,
        *,
        repo_root: Path,
        **kwargs: object,
    ) -> ToolPluginResult:
        """
        Run scip-python index to produce a parsed index.

        When rel_paths is provided, only those paths are targeted; otherwise
        the full repo (or target_dir) is indexed.

        Returns
        -------
        ToolPluginResult
            Normalized execution result with parsed ScipIndexResult.
        """
        (
            output_scip,
            proto_module_path,
            target_dir,
            rel_paths,
            environment_json,
            timeout_s,
        ) = self._validate_kwargs(dict(kwargs))

        result: ToolPluginResult | None = None
        try:
            await self._run_scip_python(
                repo_root,
                output_scip=output_scip,
                target_dir=target_dir,
                rel_paths=rel_paths,
                environment_json=environment_json,
                timeout_s=timeout_s,
            )
        except ToolNotFoundError as exc:
            log.warning("scip-python binary not found; SCIP index cannot be built")
            result = ToolPluginResult(
                tool=ToolName.SCIP_PYTHON,
                status=ToolStatus.NOT_FOUND,
                artifacts={},
                run=None,
                error=exc,
                parsed=ScipIndexResult.empty(),
            )
        except ToolExecutionError as exc:
            if _is_no_git_failure(exc.result):
                try:
                    _write_empty_scip_index(output_scip, proto_module_path)
                except (OSError, ValueError, RuntimeError) as write_exc:
                    log.warning(
                        "Failed to write empty SCIP index for no-git fallback: %s",
                        write_exc,
                    )
                else:
                    log.warning("scip-python failed due to missing git metadata; using empty index")
                    result = ToolPluginResult(
                        tool=ToolName.SCIP_PYTHON,
                        status=ToolStatus.OK,
                        artifacts={"index_scip": output_scip},
                        run=exc.result,
                        error=None,
                        parsed=ScipIndexResult.empty(),
                    )
            if result is None:
                result = ToolPluginResult(
                    tool=ToolName.SCIP_PYTHON,
                    status=ToolStatus.FAILED,
                    artifacts={"index_scip": output_scip},
                    run=exc.result,
                    error=exc,
                    parsed=ScipIndexResult.empty(),
                )
        except ValueError as exc:
            result = ToolPluginResult(
                tool=ToolName.SCIP_PYTHON,
                status=ToolStatus.FAILED,
                artifacts={"index_scip": output_scip},
                run=None,
                error=exc,
                parsed=ScipIndexResult.empty(),
            )

        if result is None:
            parsed = await to_thread.run_sync(
                _parse_scip_index,
                output_scip,
                proto_module_path,
                repo_root,
            )
            result = ToolPluginResult(
                tool=ToolName.SCIP_PYTHON,
                status=ToolStatus.OK,
                artifacts={"index_scip": output_scip},
                run=None,
                error=None,
                parsed=parsed,
            )

        return result

    async def _run_scip_python(
        self,
        repo_root: Path,
        *,
        output_scip: Path,
        target_dir: Path | None,
        rel_paths: Sequence[str] | None,
        environment_json: Path | None,
        timeout_s: float | None,
    ) -> ToolRunResult:
        target_base = await to_thread.run_sync(resolve_target_base, repo_root, target_dir)
        await to_thread.run_sync(_mkdir_parents, output_scip.parent)

        if environment_json is None:
            ensure_pip_available()
        args = build_scip_python_args(
            target_base=target_base,
            output_scip=output_scip,
            project_name=self.tools_config.scip_project_name,
            rel_paths=rel_paths,
            scope_paths=None,
            environment_json=environment_json,
        )

        result = await self.runner.run_async(
            ToolName.SCIP_PYTHON,
            args,
            options=ToolRunOptions(
                cwd=repo_root,
                output_path=output_scip,
                timeout_s=timeout_s
                if timeout_s is not None
                else self.tools_config.default_timeout_s,
            ),
        )
        if not result.ok:
            raise ToolExecutionError(result)
        return result
