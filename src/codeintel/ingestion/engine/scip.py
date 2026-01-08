"""SCIP plugin for the ingestion tool runtime."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, cast

from anyio import to_thread

from codeintel.core.execution.ids import new_run_id
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
from codeintel.ingestion.scip.cli import (
    ScipPythonArgs,
    build_scip_python_args,
    stage_pyright_config,
)
from codeintel.ingestion.scip.environment import resolve_environment_json
from codeintel.ingestion.scip.index_store import write_index_proto
from codeintel.ingestion.scip.paths import resolve_target_base
from codeintel.ingestion.scip.proto import load_generated_module
from codeintel.ingestion.scip.proto_types import ScipProtoModule
from codeintel.ingestion.scip.protobuf_parser import parse_index, rebase_parsed_index
from codeintel.ingestion.scip.telemetry import write_tool_logs

if TYPE_CHECKING:
    from codeintel.config.models import ToolsConfig
    from codeintel.ingestion.engine.infrastructure import (
        ToolRunner,
        ToolRunResult,
    )

log = logging.getLogger(__name__)


def _mkdir_parents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _require_path(kwargs: dict[str, object], key: str) -> Path:
    value = kwargs.get(key)
    if not isinstance(value, Path):
        message = f"scip-python plugin requires {key} of type Path"
        raise TypeError(message)
    return value


def _optional_path(value: object, key: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, Path):
        message = f"scip-python plugin requires {key} to be Path or None"
        raise TypeError(message)
    return value


def _optional_str(value: object, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        message = f"scip-python plugin requires {key} to be str or None"
        raise TypeError(message)
    return value


def _optional_int(value: object, key: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        message = f"scip-python plugin requires {key} to be int or None"
        raise TypeError(message)
    return value


def _optional_real(value: object, key: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, Real):
        message = f"scip-python plugin requires {key} to be a number or None"
        raise TypeError(message)
    return float(value)


def _optional_rel_paths(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        message = "scip-python plugin requires rel_paths to be a sequence of strings"
        raise TypeError(message)
    if not all(isinstance(item, str) for item in value):
        message = "scip-python plugin requires rel_paths to contain strings"
        raise TypeError(message)
    return tuple(value)


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
                syntax_kind=occ.syntax_kind,
                enclosing_range=(
                    (
                        occ.enclosing_start_line,
                        occ.enclosing_start_col,
                        occ.enclosing_end_line,
                        occ.enclosing_end_col,
                    )
                    if (
                        occ.enclosing_start_line is not None
                        and occ.enclosing_start_col is not None
                        and occ.enclosing_end_line is not None
                        and occ.enclosing_end_col is not None
                    )
                    else None
                ),
                override_documentation=occ.override_documentation,
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


def _persist_tool_logs(
    result: ToolRunResult,
    *,
    output_scip: Path,
    run_id: str,
    label: str,
) -> None:
    try:
        write_tool_logs(
            scip_dir=output_scip.parent,
            run_id=run_id,
            label=label,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        log.warning("Failed to persist scip-python logs: %s", exc)


@dataclass(frozen=True)
class _ScipToolRun:
    output_scip: Path
    target_dir: Path | None
    rel_paths: Sequence[str] | None
    environment_json: Path | None
    pyright_config_path: Path | None
    project_version: str | None
    project_namespace: str | None
    scip_node_max_old_space_mb: int | None
    timeout_s: float | None
    run_id: str


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
                optional_kwargs=(
                    "target_dir",
                    "rel_paths",
                    "environment_json",
                    "pyright_config_path",
                    "project_version",
                    "project_namespace",
                    "scip_node_max_old_space_mb",
                    "timeout_s",
                    "run_id",
                ),
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
        Path | None,
        str | None,
        str | None,
        int | None,
        float | None,
        str | None,
    ]:
        """Validate and extract keyword arguments for run method.

        Parameters
        ----------
        kwargs
            Keyword arguments to validate.

        Returns
        -------
        tuple[
            Path,
            Path,
            Path | None,
            tuple[str, ...] | None,
            Path | None,
            Path | None,
            str | None,
            str | None,
            int | None,
            float | None,
            str | None,
        ]
            Tuple of (output_scip, proto_module_path, target_dir, rel_paths,
            environment_json, pyright_config_path, project_version, project_namespace,
            scip_node_max_old_space_mb, timeout_s, run_id).
        """
        _ = self
        output_scip_obj = _require_path(kwargs, "output_scip")
        target_dir_obj = _optional_path(kwargs.get("target_dir"), "target_dir")
        rel_paths = _optional_rel_paths(kwargs.get("rel_paths"))
        proto_module_obj = _require_path(kwargs, "proto_module_path")
        environment_obj = _optional_path(
            kwargs.get("environment_json"),
            "environment_json",
        )
        pyright_config_obj = _optional_path(
            kwargs.get("pyright_config_path"),
            "pyright_config_path",
        )
        project_version_obj = _optional_str(
            kwargs.get("project_version"),
            "project_version",
        )
        project_namespace_obj = _optional_str(
            kwargs.get("project_namespace"),
            "project_namespace",
        )
        node_max_obj = _optional_int(
            kwargs.get("scip_node_max_old_space_mb"),
            "scip_node_max_old_space_mb",
        )
        timeout_s = _optional_real(kwargs.get("timeout_s"), "timeout_s")
        run_id = _optional_str(kwargs.get("run_id"), "run_id")
        return (
            output_scip_obj,
            proto_module_obj,
            target_dir_obj,
            rel_paths,
            environment_obj,
            pyright_config_obj,
            project_version_obj,
            project_namespace_obj,
            node_max_obj,
            timeout_s,
            run_id,
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
            pyright_config_path,
            project_version,
            project_namespace,
            scip_node_max_old_space_mb,
            timeout_s,
            run_id,
        ) = self._validate_kwargs(dict(kwargs))
        resolved_run_id = run_id or new_run_id("scip")
        run_args = _ScipToolRun(
            output_scip=output_scip,
            target_dir=target_dir,
            rel_paths=rel_paths,
            environment_json=environment_json,
            pyright_config_path=pyright_config_path,
            project_version=project_version,
            project_namespace=project_namespace,
            scip_node_max_old_space_mb=scip_node_max_old_space_mb,
            timeout_s=timeout_s,
            run_id=resolved_run_id,
        )

        result: ToolPluginResult | None = None
        try:
            await self._run_scip_python(repo_root, run_args=run_args)
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
        run_args: _ScipToolRun,
    ) -> ToolRunResult:
        output_scip = run_args.output_scip
        target_base = await to_thread.run_sync(
            resolve_target_base,
            repo_root,
            run_args.target_dir,
        )
        await to_thread.run_sync(_mkdir_parents, output_scip.parent)
        env_resolution = resolve_environment_json(
            environment_json=run_args.environment_json,
            scip_dir=output_scip.parent,
        )
        environment_json = env_resolution.environment_json
        args = build_scip_python_args(
            ScipPythonArgs(
                target_base=target_base,
                output_scip=output_scip,
                project_name=self.tools_config.scip_project_name,
                target_paths=run_args.rel_paths,
                environment_json=environment_json,
                project_version=run_args.project_version,
                project_namespace=run_args.project_namespace,
            )
        )
        env: dict[str, str] | None = None
        if (
            run_args.scip_node_max_old_space_mb is not None
            and run_args.scip_node_max_old_space_mb > 0
        ):
            env = {"NODE_OPTIONS": f"--max-old-space-size={run_args.scip_node_max_old_space_mb}"}

        with stage_pyright_config(
            target_base=target_base,
            pyright_config_path=run_args.pyright_config_path,
        ):
            result = await self.runner.run_async(
                ToolName.SCIP_PYTHON,
                args,
                options=ToolRunOptions(
                    cwd=repo_root,
                    output_path=output_scip,
                    timeout_s=run_args.timeout_s
                    if run_args.timeout_s is not None
                    else self.tools_config.default_timeout_s,
                    env=env,
                ),
            )
        _persist_tool_logs(
            result,
            output_scip=output_scip,
            run_id=run_args.run_id,
            label="scip-python",
        )
        if not result.ok:
            raise ToolExecutionError(result)
        return result
