"""Hamilton-based build executor.

This module provides HamiltonBuildExecutor, a DAG-based executor for build
targets using Hamilton's Driver.

Design Principles
-----------------
1. HamiltonBuildExecutor.run() is the main entry point for execution.
2. It maps target names to Hamilton node names via runtime mappings.
3. Results are returned in a structured HamiltonBuildResult.
4. The executor integrates with existing manifest/tracking infrastructure.
5. Executes the full dependency closure, not just requested targets.
"""

from __future__ import annotations

import faulthandler
import importlib
import inspect
import json
import logging
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TextIO, cast
from urllib.parse import urljoin, urlparse

import hamilton.base as h_base
import requests
from hamilton import htypes
from hamilton.caching.adapter import HamiltonCacheAdapter
from hamilton.lifecycle import GracefulErrorAdapter, PDBDebugger, PrintLn
from hamilton.lifecycle import base as lifecycle_base

from codeintel.build.config import BuildConfig
from codeintel.build.execution_policy import effective_max_workers_for_graph
from codeintel.build.hamilton.build_log import (
    drain_build_log,
    record_build_event,
    start_build_log,
)
from codeintel.build.hamilton.cache_adapter import (
    ArrowFileResultStore,
    CacheAdapterOptions,
    ManifestBackedCacheAdapter,
)
from codeintel.build.hamilton.cache_key_resolver import CacheKeyResolver
from codeintel.build.hamilton.decision_trace import (
    DECISION_TRACE_ARTIFACT_NAME,
    DECISION_TRACE_PATH_TEMPLATE,
)
from codeintel.build.hamilton.diagnostics import (
    DiagnosticsInputs,
    DiagnosticsTargets,
    diagnostics_dir,
    emit_diagnostics,
)
from codeintel.build.hamilton.driver_factory import target_to_node_name
from codeintel.build.hamilton.driver_options import BuildDriverOptions
from codeintel.build.hamilton.execution_options import BuildExecutionOptions
from codeintel.build.hamilton.execution_profiles import (
    apply_dynamic_execution_config,
    build_execution_profile,
    build_parallel_adapter,
)
from codeintel.build.hamilton.hooks import (
    LifecycleEventStreamHook,
    NodeTelemetryHook,
    ProgressBarHook,
    build_hooks,
)
from codeintel.build.hamilton.native.views.view_outputs import view_lineage_payload
from codeintel.build.hamilton.optional_inputs import optional_inputs_for_target
from codeintel.build.hamilton.result_builder import BuildResultBuilder
from codeintel.build.hamilton.run_records import (
    NativeRunInfo,
    RunRecordInputs,
    TargetRunRecord,
    create_run_record,
)
from codeintel.build.hamilton.run_writer import BuildRunWriter, RunReportInputs
from codeintel.build.meta.bundle import (
    BuildMetadataBundleWriter,
    DerivedLineageContext,
    dataflow_from_contracts,
    derived_lineage_from_catalog,
    schema_registry_from_manifest,
)
from codeintel.build.meta.contract_catalog import build_contract_catalog_payload
from codeintel.build.schemas import get_schema_provider
from codeintel.build.schemas.compile import (
    SchemaManifestContext,
    SchemaManifestRequest,
    compile_schema_manifest,
)
from codeintel.build.schemas.contract_service import iter_contracts
from codeintel.core.config.settings import HamiltonTrackerSettings
from codeintel.core.datasets.manifests import dataset_manifest_path
from codeintel.core.duckdb_types import DuckDBError
from codeintel.core.execution.ids import new_run_id
from codeintel.core.runtime.loader import load_runtime_settings
from codeintel.observability.telemetry_context import (
    RepoCommitContext,
    telemetry_context,
)
from codeintel.runtime.compose import compose_runtime, set_execution_active
from codeintel.runtime.inputs import ExecutionInputs, execution_input_mapping
from codeintel.runtime.runtime_bundle import HamiltonRuntimeBundle

if TYPE_CHECKING:
    from typing import TypedDict

    from hamilton.io.materialization import ExtractorFactory, MaterializerFactory
    from hamilton.lifecycle.base import LifecycleAdapter
    from hamilton.node import Node

    from codeintel.build.hamilton.build_log import BuildLogContext
    from codeintel.build.hamilton.dag_catalog import DagCatalog, TargetDescriptor
    from codeintel.build.hamilton.env import BuildEnv

    class BuildExecutionOverrides(TypedDict, total=False):
        profile: str | None
        parallel_backend: str
        max_workers: int | None
        enable_cache: bool
        cache_dir: str | None
        plugins_enabled: tuple[str, ...] | None
        plugins_disabled: tuple[str, ...] | None
        allow_workspace_modules: bool | None

    class BuildExecutionOptionsData(TypedDict, total=False):
        profile: str | None
        parallel_backend: str
        max_workers: int | None
        enable_hamilton_cache: bool
        cache_dir: str | None
        plugins_enabled: tuple[str, ...] | None
        plugins_disabled: tuple[str, ...] | None
        allow_workspace_modules: bool | None


log = logging.getLogger(__name__)

_EXECUTOR_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "profile",
        "parallel_backend",
        "max_workers",
        "enable_cache",
        "cache_dir",
        "plugins_enabled",
        "plugins_disabled",
        "allow_workspace_modules",
    }
)
_EXECUTOR_OVERRIDE_MAP: tuple[tuple[str, str], ...] = (
    ("profile", "profile"),
    ("parallel_backend", "parallel_backend"),
    ("max_workers", "max_workers"),
    ("enable_cache", "enable_hamilton_cache"),
    ("cache_dir", "cache_dir"),
    ("plugins_enabled", "plugins_enabled"),
    ("plugins_disabled", "plugins_disabled"),
    ("allow_workspace_modules", "allow_workspace_modules"),
)

_INTRINSIC_TARGETS: tuple[str, ...] = ("scip",)


@dataclass(frozen=True)
class _RunState:
    """Execution state shared across run steps."""

    env: BuildEnv
    targets: tuple[str, ...]
    runtime: HamiltonRuntimeBundle
    run_id: str
    cache_dir: Path
    telemetry: _TelemetryHooksSettings
    start_time: float
    started_at: datetime
    domain: str | None

    @property
    def duration_ms(self) -> float:
        """Return elapsed milliseconds for the run."""
        return (time.perf_counter() - self.start_time) * 1000


@dataclass(frozen=True)
class _RunSetup:
    env: BuildEnv
    run_id: str
    resolved_targets: tuple[str, ...]
    writer: BuildRunWriter
    cache_dir: Path
    log_handler: logging.Handler | None
    diagnostics_path: Path
    telemetry_settings: _TelemetryHooksSettings
    cache_logger_handle: _CacheLoggerHandle | None
    hang_watchdog: _HangWatchdog | None
    ui_handle: _HamiltonUiHandle | None


@dataclass(frozen=True)
class _ExecutionPlan:
    closure: tuple[str, ...]
    final_vars: list[str]
    preflight_records: dict[str, Any]


@dataclass(frozen=True)
class _FailureSnapshotContext:
    run_id: str
    repo: str
    commit: str
    domain: str | None
    requested_targets: tuple[str, ...]
    build_dir: Path


@dataclass(frozen=True)
class _MissingInputs:
    required: tuple[str, ...]
    optional: tuple[str, ...]


@dataclass(frozen=True)
class _TrackerTagContext:
    env: BuildEnv
    run_id: str
    domain: str | None
    deployment_environment: str | None
    cache_dir: Path | None
    diagnostics_path: Path | None


@dataclass(frozen=True)
class _TelemetryHooksSettings:
    enable_progress: bool
    progress_style: str
    progress_desc: str
    println_enabled: bool
    println_verbosity: int
    println_node_filter: tuple[str, ...] | None
    typecheck_enabled: bool
    typecheck_inputs: bool
    typecheck_outputs: bool
    graceful_errors_enabled: bool
    graceful_try_all_parallel: bool
    graceful_allow_injection: bool
    pdb_enabled: bool
    pdb_before: bool
    pdb_during: bool
    pdb_after: bool
    pdb_node_filter: tuple[str, ...] | None
    event_stream_enabled: bool
    event_stream_path: Path
    cache_logger_level: str | None
    cache_logger_path: Path | None
    hang_watchdog_enabled: bool
    hang_watchdog_timeout_s: float
    hang_watchdog_repeat: bool
    hang_watchdog_path: Path
    display_all_functions_enabled: bool
    display_all_functions_path: Path
    visualize_execution_enabled: bool
    visualize_execution_path: Path
    ddog_enabled: bool
    ddog_root_name: str | None
    ddog_service: str | None
    ddog_include_causal_links: bool


@dataclass(frozen=True)
class _HamiltonUiSettings:
    enabled: bool
    command: tuple[str, ...]
    api_url: str
    healthcheck_path: str
    startup_timeout_s: float
    shutdown_timeout_s: float
    log_path: Path


@dataclass(frozen=True)
class _HamiltonUiHandle:
    process: subprocess.Popen[str]
    log_handle: TextIO | None
    shutdown_timeout_s: float

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=self.shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=self.shutdown_timeout_s)
        if self.log_handle is not None:
            self.log_handle.close()


class _HangWatchdog:
    def __init__(
        self,
        *,
        timeout_s: float,
        output_path: Path,
        repeat: bool,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = output_path.open("a", encoding="utf-8")
        faulthandler.enable(file=self._file)
        faulthandler.dump_traceback_later(timeout_s, repeat=repeat, file=self._file)

    def close(self) -> None:
        faulthandler.cancel_dump_traceback_later()
        self._file.close()


@dataclass(frozen=True)
class _CacheLoggerHandle:
    logger: logging.Logger
    handler: logging.Handler | None
    previous_level: int
    previous_propagate: bool

    def close(self) -> None:
        if self.handler is not None:
            self.logger.removeHandler(self.handler)
            self.handler.close()
        self.logger.setLevel(self.previous_level)
        self.logger.propagate = self.previous_propagate


class _SafeTypeCheckHook(
    lifecycle_base.BasePreNodeExecute,
    lifecycle_base.BasePostNodeExecute,
):
    def __init__(
        self,
        *,
        check_input: bool,
        check_output: bool,
        logger: logging.Logger,
    ) -> None:
        self._check_input = check_input
        self._check_output = check_output
        self._logger = logger

    def pre_node_execute(
        self,
        *,
        run_id: str,
        node_: Node,
        kwargs: dict[str, object],
        task_id: str | None = None,
    ) -> None:
        _ = (run_id, task_id)
        if not self._check_input:
            return
        for input_name, input_value in kwargs.items():
            input_type = node_.input_types.get(input_name)
            if input_type is None:
                continue
            declared_type = input_type[0]
            try:
                if not htypes.check_instance(input_value, declared_type):
                    self._logger.warning(
                        "build.telemetry.typecheck_input_mismatch node=%s input=%s "
                        "expected=%s actual=%s",
                        node_.name,
                        input_name,
                        declared_type,
                        type(input_value),
                    )
            except TypeError as exc:
                self._logger.warning(
                    "build.telemetry.typecheck_input_failed node=%s input=%s error=%s",
                    node_.name,
                    input_name,
                    exc,
                )

    def post_node_execute(
        self,
        *,
        run_id: str,
        node_: Node,
        kwargs: dict[str, object],
        success: bool,
        task_id: str | None = None,
        **context: object,
    ) -> None:
        _ = (run_id, kwargs, task_id)
        if not self._check_output or not success:
            return
        result = context.get("result")
        try:
            if not htypes.check_instance(result, node_.type):
                self._logger.warning(
                    "build.telemetry.typecheck_output_mismatch node=%s expected=%s actual=%s",
                    node_.name,
                    node_.type,
                    type(result),
                )
        except TypeError as exc:
            self._logger.warning(
                "build.telemetry.typecheck_output_failed node=%s error=%s",
                node_.name,
                exc,
            )


@dataclass(frozen=True)
class _TrackerOverrides:
    enabled: bool | None = None
    project_id: str | None = None
    username: str | None = None
    dag_name: str | None = None
    api_url: str | None = None
    ui_url: str | None = None
    capture_data_statistics: bool | None = None
    max_list_length: int | None = None
    max_dict_length: int | None = None
    config_uri: str | None = None
    tags: tuple[tuple[str, str], ...] | None = None


class _TrackingConstants(Protocol):
    CAPTURE_DATA_STATISTICS: bool
    MAX_LIST_LENGTH_CAPTURE: int
    MAX_DICT_LENGTH_CAPTURE: int
    DEFAULT_CONFIG_URI: str


def _generate_run_id() -> str:
    """Generate a unique run ID for build tracking.

    Returns
    -------
    str
        Unique run identifier for this Hamilton execution.
    """
    return new_run_id("hamilton")


def _ensure_intrinsic_targets(targets: list[str]) -> list[str]:
    resolved = list(targets)
    for target in _INTRINSIC_TARGETS:
        if target not in resolved:
            resolved.append(target)
    return resolved


def _coerce_project_id(value: str) -> int | str:
    if value.isdigit():
        return int(value)
    return value


def _telemetry_section(config: BuildConfig) -> Mapping[str, object]:
    raw = config.get("telemetry.hooks")
    if isinstance(raw, Mapping):
        return raw
    return {}


def _telemetry_bool(config: Mapping[str, object], key: str, *, default: bool) -> bool:
    value = config.get(key)
    if isinstance(value, bool):
        return value
    return default


def _telemetry_int(config: Mapping[str, object], key: str, *, default: int, min_value: int) -> int:
    value = config.get(key)
    if isinstance(value, int) and value >= min_value:
        return value
    return default


def _telemetry_float(
    config: Mapping[str, object],
    key: str,
    *,
    default: float,
    min_value: float,
) -> float:
    value = config.get(key)
    if isinstance(value, (int, float)) and float(value) >= min_value:
        return float(value)
    return default


def _telemetry_str(config: Mapping[str, object], key: str, *, default: str) -> str:
    value = config.get(key)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return default


def _telemetry_optional_str(config: Mapping[str, object], key: str) -> str | None:
    value = config.get(key)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _telemetry_filter_list(value: object) -> tuple[str, ...] | None:
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else None
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        cleaned = tuple(item.strip() for item in value if item.strip())
        return cleaned or None
    return None


def _node_filter_from_prefixes(
    prefixes: tuple[str, ...] | None,
) -> Callable[[str, dict[str, Any]], bool] | None:
    if not prefixes:
        return None
    normalized = tuple(prefixes)

    def _filter(node_name: str, _tags: dict[str, Any]) -> bool:
        return any(node_name == prefix or node_name.startswith(prefix) for prefix in normalized)

    return _filter


def _load_telemetry_hooks_settings(
    *,
    env: BuildEnv,
    diagnostics_path: Path,
) -> _TelemetryHooksSettings:
    raw = _telemetry_section(env.config)
    progress_style = _telemetry_str(raw, "progress_style", default="internal").lower()
    enable_progress = _telemetry_bool(raw, "enable_progress", default=False)
    return _TelemetryHooksSettings(
        enable_progress=enable_progress,
        progress_style=progress_style,
        progress_desc=_telemetry_str(raw, "progress_desc", default="Building targets"),
        println_enabled=_telemetry_bool(raw, "println_enabled", default=False),
        println_verbosity=_telemetry_int(raw, "println_verbosity", default=1, min_value=0),
        println_node_filter=_telemetry_filter_list(raw.get("println_node_filter")),
        typecheck_enabled=_telemetry_bool(raw, "typecheck_enabled", default=False),
        typecheck_inputs=_telemetry_bool(raw, "typecheck_inputs", default=True),
        typecheck_outputs=_telemetry_bool(raw, "typecheck_outputs", default=True),
        graceful_errors_enabled=_telemetry_bool(raw, "graceful_errors_enabled", default=False),
        graceful_try_all_parallel=_telemetry_bool(
            raw,
            "graceful_try_all_parallel",
            default=True,
        ),
        graceful_allow_injection=_telemetry_bool(
            raw,
            "graceful_allow_injection",
            default=True,
        ),
        pdb_enabled=_telemetry_bool(raw, "pdb_enabled", default=False),
        pdb_before=_telemetry_bool(raw, "pdb_before", default=False),
        pdb_during=_telemetry_bool(raw, "pdb_during", default=False),
        pdb_after=_telemetry_bool(raw, "pdb_after", default=False),
        pdb_node_filter=_telemetry_filter_list(raw.get("pdb_node_filter")),
        event_stream_enabled=_telemetry_bool(raw, "event_stream_enabled", default=False),
        event_stream_path=_resolve_telemetry_path(
            raw.get("event_stream_path"),
            repo_root=env.snapshot.repo_root,
            default_path=diagnostics_path / "hamilton_event_stream.jsonl",
        ),
        cache_logger_level=_telemetry_optional_str(raw, "cache_logger_level"),
        cache_logger_path=_resolve_optional_path(
            raw.get("cache_logger_path"),
            repo_root=env.snapshot.repo_root,
        ),
        hang_watchdog_enabled=_telemetry_bool(raw, "hang_watchdog_enabled", default=False),
        hang_watchdog_timeout_s=_telemetry_float(
            raw,
            "hang_watchdog_timeout_s",
            default=600.0,
            min_value=1.0,
        ),
        hang_watchdog_repeat=_telemetry_bool(raw, "hang_watchdog_repeat", default=True),
        hang_watchdog_path=_resolve_telemetry_path(
            raw.get("hang_watchdog_path"),
            repo_root=env.snapshot.repo_root,
            default_path=diagnostics_path / "hamilton_hang_dump.log",
        ),
        display_all_functions_enabled=_telemetry_bool(
            raw,
            "display_all_functions_enabled",
            default=False,
        ),
        display_all_functions_path=_resolve_telemetry_path(
            raw.get("display_all_functions_path"),
            repo_root=env.snapshot.repo_root,
            default_path=diagnostics_path / "hamilton_graph_all.svg",
        ),
        visualize_execution_enabled=_telemetry_bool(
            raw,
            "visualize_execution_enabled",
            default=False,
        ),
        visualize_execution_path=_resolve_telemetry_path(
            raw.get("visualize_execution_path"),
            repo_root=env.snapshot.repo_root,
            default_path=diagnostics_path / "hamilton_graph_execution.svg",
        ),
        ddog_enabled=_telemetry_bool(raw, "ddog_enabled", default=False),
        ddog_root_name=_telemetry_optional_str(raw, "ddog_root_name"),
        ddog_service=_telemetry_optional_str(raw, "ddog_service"),
        ddog_include_causal_links=_telemetry_bool(
            raw,
            "ddog_include_causal_links",
            default=False,
        ),
    )


def _resolve_log_level(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    cleaned = value.strip().upper()
    if not cleaned:
        return default
    if cleaned.isdigit():
        return int(cleaned)
    level_map = logging.getLevelNamesMapping()
    resolved = level_map.get(cleaned)
    if isinstance(resolved, int):
        return resolved
    return default


def _configure_cache_logger(
    settings: _TelemetryHooksSettings,
) -> _CacheLoggerHandle | None:
    if settings.cache_logger_level is None and settings.cache_logger_path is None:
        return None
    logger = logging.getLogger("hamilton.caching")
    previous_level = logger.level
    previous_propagate = logger.propagate
    level = _resolve_log_level(settings.cache_logger_level, default=logging.INFO)
    if (
        settings.cache_logger_level is not None
        and level == logging.INFO
        and settings.cache_logger_level.strip().upper() not in {"INFO", "20"}
    ):
        log.warning(
            "telemetry.cache_logger_level_invalid value=%s default=INFO",
            settings.cache_logger_level,
        )
    logger.setLevel(level)
    handler: logging.Handler | None = None
    if settings.cache_logger_path is not None:
        settings.cache_logger_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(settings.cache_logger_path, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    return _CacheLoggerHandle(
        logger=logger,
        handler=handler,
        previous_level=previous_level,
        previous_propagate=previous_propagate,
    )


def _start_hang_watchdog(settings: _TelemetryHooksSettings) -> _HangWatchdog | None:
    if not settings.hang_watchdog_enabled:
        return None
    try:
        return _HangWatchdog(
            timeout_s=settings.hang_watchdog_timeout_s,
            output_path=settings.hang_watchdog_path,
            repeat=settings.hang_watchdog_repeat,
        )
    except OSError as exc:
        log.warning("build.telemetry.hang_watchdog_start_failed error=%s", exc)
        return None


def _build_println_adapter(settings: _TelemetryHooksSettings) -> PrintLn | None:
    if not settings.println_enabled:
        return None
    node_filter = _node_filter_from_prefixes(settings.println_node_filter)
    println_logger = logging.getLogger("hamilton.println")
    return PrintLn(
        verbosity=settings.println_verbosity,
        print_fn=println_logger.info,
        node_filter=node_filter,
    )


def _build_typecheck_adapter(settings: _TelemetryHooksSettings) -> LifecycleAdapter | None:
    if not settings.typecheck_enabled:
        return None
    return _SafeTypeCheckHook(
        check_input=settings.typecheck_inputs,
        check_output=settings.typecheck_outputs,
        logger=log,
    )


def _build_graceful_adapter(settings: _TelemetryHooksSettings) -> GracefulErrorAdapter | None:
    if not settings.graceful_errors_enabled:
        return None
    return GracefulErrorAdapter(
        Exception,
        sentinel_value=None,
        try_all_parallel=settings.graceful_try_all_parallel,
        allow_injection=settings.graceful_allow_injection,
    )


def _build_pdb_adapter(settings: _TelemetryHooksSettings) -> PDBDebugger | None:
    if not settings.pdb_enabled:
        return None
    node_filter = _node_filter_from_prefixes(settings.pdb_node_filter)
    return PDBDebugger(
        node_filter=node_filter,
        before=settings.pdb_before,
        during=settings.pdb_during,
        after=settings.pdb_after,
    )


def _build_progress_adapter(settings: _TelemetryHooksSettings) -> object | None:
    if not settings.enable_progress:
        return None
    style = settings.progress_style
    adapter: object | None = None
    if style == "tqdm":
        adapter = ProgressBarHook(desc=settings.progress_desc)
    elif style == "rich":
        try:
            progress_module = importlib.import_module("hamilton.plugins.h_rich")
        except ModuleNotFoundError as exc:
            log.warning("Rich progress unavailable; install sf-hamilton[rich]: %s", exc)
        else:
            progress_cls = getattr(progress_module, "RichProgressBar", None)
            if not isinstance(progress_cls, type):
                log.warning("Rich progress adapter missing in hamilton.plugins.h_rich")
            else:
                try:
                    adapter = progress_cls(run_desc=settings.progress_desc)
                except (TypeError, ValueError) as exc:
                    log.warning("Failed to initialize RichProgressBar: %s", exc)
    return adapter


def _build_ddog_adapter(
    settings: _TelemetryHooksSettings,
    *,
    default_root_name: str,
) -> object | None:
    if not settings.ddog_enabled:
        return None
    try:
        ddog_module = importlib.import_module("hamilton.plugins.h_ddog")
    except ModuleNotFoundError as exc:
        log.warning("DDOG tracer unavailable; install sf-hamilton[datadog]: %s", exc)
        return None
    tracer_cls = getattr(ddog_module, "DDOGTracer", None)
    if not isinstance(tracer_cls, type):
        log.warning("DDOG tracer adapter missing in hamilton.plugins.h_ddog")
        return None
    root_name = settings.ddog_root_name or default_root_name
    try:
        return tracer_cls(
            root_name=root_name,
            include_causal_links=settings.ddog_include_causal_links,
            service=settings.ddog_service,
        )
    except (TypeError, ValueError) as exc:
        log.warning("Failed to initialize DDOGTracer: %s", exc)
        return None


def _build_diagnostic_adapters(
    *,
    settings: _TelemetryHooksSettings,
    run_id: str,
    default_ddog_root: str,
    cache_enabled: bool,
) -> list[LifecycleAdapter]:
    adapters: list[LifecycleAdapter] = []
    println_adapter = _build_println_adapter(settings)
    if println_adapter is not None:
        adapters.append(println_adapter)
    progress_adapter = _build_progress_adapter(settings)
    if progress_adapter is not None:
        adapters.append(cast("LifecycleAdapter", progress_adapter))
    typecheck_adapter = _build_typecheck_adapter(settings)
    if typecheck_adapter is not None:
        adapters.append(typecheck_adapter)
    if cache_enabled and settings.graceful_errors_enabled:
        log.warning(
            "build.telemetry.graceful_disabled cache_enabled=true; disable cache to enable "
            "GracefulErrorAdapter",
        )
    else:
        graceful_adapter = _build_graceful_adapter(settings)
        if graceful_adapter is not None:
            adapters.append(graceful_adapter)
    pdb_adapter = _build_pdb_adapter(settings)
    if pdb_adapter is not None:
        adapters.append(pdb_adapter)
    if settings.event_stream_enabled:
        adapters.append(
            LifecycleEventStreamHook(
                run_id=run_id,
                output_path=settings.event_stream_path,
            )
        )
    ddog_adapter = _build_ddog_adapter(settings, default_root_name=default_ddog_root)
    if ddog_adapter is not None:
        adapters.append(cast("LifecycleAdapter", ddog_adapter))
    return adapters


def _default_hamilton_ui_command() -> tuple[str, ...]:
    return (sys.executable, "-m", "hamilton.cli.__main__", "ui")


def _resolve_hamilton_ui_command(raw: object | None) -> tuple[str, ...]:
    default_command = _default_hamilton_ui_command()
    if raw is None:
        return default_command
    if isinstance(raw, str):
        parsed = tuple(part for part in shlex.split(raw) if part.strip())
        return parsed or default_command
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        cleaned = tuple(item.strip() for item in raw if item.strip())
        return cleaned or default_command
    return default_command


def _normalize_healthcheck_path(path: str) -> str:
    cleaned = path.strip()
    if not cleaned:
        return "/api/v1/phone_home"
    if not cleaned.startswith("/"):
        return f"/{cleaned}"
    return cleaned


def _resolve_tracker_settings(config: BuildConfig) -> HamiltonTrackerSettings:
    runtime_settings = load_runtime_settings().observability.hamilton_tracker
    overrides = _tracker_overrides_from_config(config)
    if overrides is not None:
        return _merge_tracker_settings(runtime_settings, overrides)
    return runtime_settings


def _tracker_api_url(config: BuildConfig) -> str:
    settings = _resolve_tracker_settings(config)
    return settings.api_url or "http://localhost:8241"


def _load_hamilton_ui_settings(*, env: BuildEnv) -> _HamiltonUiSettings:
    raw = env.config.get("telemetry.hamilton_ui")
    section = raw if isinstance(raw, Mapping) else {}
    tracker_settings = _resolve_tracker_settings(env.config)
    default_enabled = bool(tracker_settings.enabled)
    return _HamiltonUiSettings(
        enabled=_telemetry_bool(section, "enabled", default=default_enabled),
        command=_resolve_hamilton_ui_command(section.get("command")),
        api_url=tracker_settings.api_url or "http://localhost:8241",
        healthcheck_path=_normalize_healthcheck_path(
            _telemetry_str(section, "healthcheck_path", default="/api/v1/phone_home")
        ),
        startup_timeout_s=_telemetry_float(
            section,
            "startup_timeout_s",
            default=30.0,
            min_value=0.0,
        ),
        shutdown_timeout_s=_telemetry_float(
            section,
            "shutdown_timeout_s",
            default=5.0,
            min_value=0.0,
        ),
        log_path=_resolve_telemetry_path(
            section.get("log_path"),
            repo_root=env.snapshot.repo_root,
            default_path=env.paths.build_dir / "logs" / "hamilton_ui.log",
        ),
    )


def _is_local_ui_url(api_url: str) -> bool:
    parsed = urlparse(api_url)
    hostname = (parsed.hostname or "").lower()
    return hostname in {"localhost", "127.0.0.1", "::1"}


def _hamilton_ui_healthcheck_url(settings: _HamiltonUiSettings) -> str:
    return urljoin(settings.api_url.rstrip("/") + "/", settings.healthcheck_path.lstrip("/"))


def _hamilton_ui_is_ready(settings: _HamiltonUiSettings) -> bool:
    url = _hamilton_ui_healthcheck_url(settings)
    try:
        response = requests.get(url, timeout=1.0)
    except requests.RequestException:
        return False
    return response is not None


def _wait_for_hamilton_ui(handle: _HamiltonUiHandle, settings: _HamiltonUiSettings) -> bool:
    if settings.startup_timeout_s <= 0:
        return True
    deadline = time.monotonic() + settings.startup_timeout_s
    while time.monotonic() < deadline:
        if handle.process.poll() is not None:
            log.warning("hamilton.ui.exited_early code=%s", handle.process.returncode)
            handle.stop()
            return False
        if _hamilton_ui_is_ready(settings):
            log.info("hamilton.ui.ready api_url=%s", settings.api_url)
            return True
        time.sleep(0.5)
    log.warning("hamilton.ui.start_timeout api_url=%s", settings.api_url)
    return True


def _start_hamilton_ui(
    settings: _HamiltonUiSettings, *, repo_root: Path
) -> _HamiltonUiHandle | None:
    if not settings.enabled:
        return None
    if not _is_local_ui_url(settings.api_url):
        log.warning("hamilton.ui.autostart_skipped api_url=%s", settings.api_url)
        return None
    if _hamilton_ui_is_ready(settings):
        log.info("hamilton.ui.already_running api_url=%s", settings.api_url)
        return None
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = settings.log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            list(settings.command),
            cwd=str(repo_root),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=log_handle,
            text=True,
        )
    except (OSError, ValueError) as exc:
        log.warning("hamilton.ui.start_failed error=%s", exc)
        log_handle.close()
        return None
    handle = _HamiltonUiHandle(
        process=process,
        log_handle=log_handle,
        shutdown_timeout_s=settings.shutdown_timeout_s,
    )
    if not _wait_for_hamilton_ui(handle, settings):
        return None
    return handle


def _apply_tracker_constants(
    settings: HamiltonTrackerSettings,
    *,
    config_uri: str | None = None,
) -> None:
    try:
        tracking_constants = importlib.import_module("hamilton_sdk.tracking.constants")
    except ModuleNotFoundError as exc:
        log.warning("Hamilton tracker constants unavailable: %s", exc)
        return
    constants = cast("_TrackingConstants", tracking_constants)
    if settings.capture_data_statistics is not None:
        constants.CAPTURE_DATA_STATISTICS = bool(settings.capture_data_statistics)
    if settings.max_list_length is not None:
        constants.MAX_LIST_LENGTH_CAPTURE = settings.max_list_length
    if settings.max_dict_length is not None:
        constants.MAX_DICT_LENGTH_CAPTURE = settings.max_dict_length
    if config_uri is not None:
        try:
            constants.DEFAULT_CONFIG_URI = config_uri
        except (AttributeError, TypeError) as exc:
            log.warning("tracker.constant_set_failed DEFAULT_CONFIG_URI: %s", exc)


def _run_preflight(
    *,
    context: _RunState,
    catalog: DagCatalog,
) -> tuple[bool, str | None]:
    log.info(
        "build.dag.preflight.start run_id=%s repo=%s commit=%s target_count=%d table_count=%d",
        context.run_id,
        context.env.repo,
        context.env.commit,
        len(catalog.targets),
        len(catalog.table_outputs),
    )
    report = catalog.preflight_report(repo_root=context.env.snapshot.repo_root)
    if report.ok:
        log.info(
            "build.dag.preflight.ok run_id=%s repo=%s commit=%s duration_ms=%.1f",
            context.run_id,
            context.env.repo,
            context.env.commit,
            report.duration_ms,
        )
        return True, None
    log.error(
        "build.dag.preflight.fail run_id=%s repo=%s commit=%s failures=%s",
        context.run_id,
        context.env.repo,
        context.env.commit,
        report.log_entries(),
    )
    return False, report.summary()


def _materializer_import_path(raw: str) -> tuple[str, str]:
    module_name, sep, attr = raw.partition(":")
    if not sep:
        module_name, _, attr = raw.rpartition(".")
    if not module_name or not attr:
        msg = f"Invalid materializer import path: {raw}"
        raise ValueError(msg)
    return module_name, attr


def _requires_factory_args(factory: object) -> bool:
    if not callable(factory):
        return True
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return True
    for param in signature.parameters.values():
        if param.kind in {param.VAR_POSITIONAL, param.VAR_KEYWORD}:
            continue
        if param.default is param.empty:
            return True
    return False


def _materializer_from_object(obj: object, *, source: str) -> object:
    if inspect.isfunction(obj) or inspect.isclass(obj):
        if _requires_factory_args(obj):
            msg = f"Materializer {source} requires arguments; supply an instance or no-arg factory."
            raise ValueError(msg)
        return obj()
    return obj


def _resolve_materializers(
    materializer_paths: tuple[str, ...],
) -> tuple[ExtractorFactory | MaterializerFactory, ...]:
    if not materializer_paths:
        return ()
    resolved: list[ExtractorFactory | MaterializerFactory] = []
    for raw in materializer_paths:
        module_name, attr = _materializer_import_path(raw)
        module = importlib.import_module(module_name)
        obj = getattr(module, attr, None)
        if obj is None:
            msg = f"Materializer not found: {raw}"
            raise ValueError(msg)
        resolved.append(
            cast(
                "ExtractorFactory | MaterializerFactory",
                _materializer_from_object(obj, source=raw),
            )
        )
    return tuple(resolved)


def _base_hamilton_config(
    *,
    env: BuildEnv,
    options: BuildExecutionOptions,
) -> dict[str, Any]:
    config: dict[str, Any] = {"profile": options.resolved_profile(env=env)}
    config.update(env.variants.as_hamilton_config())
    config["variant_fingerprint"] = env.variants.variant_fingerprint
    config.update(options.plugin_overrides())
    graph_backend = env.config.get("hamilton.graph_backend")
    if graph_backend is not None:
        if not isinstance(graph_backend, str):
            msg = "hamilton.graph_backend must be a string"
            raise TypeError(msg)
        allowed_backends = {"compute", "existing", "empty"}
        if graph_backend not in allowed_backends:
            msg = f"Unsupported hamilton.graph_backend={graph_backend!r}"
            raise ValueError(msg)
        config["graph_backend"] = graph_backend
    config["ci_validate_outputs"] = bool(env.validate_outputs)
    config["ci_validation_mode"] = env.validation_mode.value
    return config


def _build_cache_adapter(
    *,
    run_id: str,
    cache_dir: Path,
    enable_cache: bool,
) -> ManifestBackedCacheAdapter | None:
    if not enable_cache:
        return None
    cache_options = CacheAdapterOptions(
        default_behavior="disable",
        default_loader_behavior="disable",
        default_saver_behavior="disable",
        log_to_file=True,
        result_store=ArrowFileResultStore(path=str(cache_dir)),
    )
    return ManifestBackedCacheAdapter(
        path=cache_dir,
        manifest_writer=None,
        manifest_run_id=run_id,
        options=cache_options,
    )


def _build_tracker_tags(
    *,
    settings: HamiltonTrackerSettings,
    context: _TrackerTagContext,
) -> dict[str, str]:
    tags = dict(settings.tags)
    if context.deployment_environment and "environment" not in tags:
        tags["environment"] = context.deployment_environment
    tags.setdefault("repo", context.env.snapshot.repo)
    tags.setdefault("commit", context.env.snapshot.commit)
    tags.setdefault("run_id", context.run_id)
    if context.domain and "domain" not in tags:
        tags["domain"] = context.domain
    tags.setdefault("build.decision_trace_artifact", DECISION_TRACE_ARTIFACT_NAME)
    tags.setdefault(
        "build.decision_trace_path",
        DECISION_TRACE_PATH_TEMPLATE.format(build_dir=context.env.paths.build_dir.name),
    )
    if context.cache_dir is not None:
        tags.setdefault("build.cache_dir", str(context.cache_dir))
    if context.diagnostics_path is not None:
        tags.setdefault("build.diagnostics_dir", str(context.diagnostics_path))
    return tags


def _tracker_overrides_from_config(config: BuildConfig) -> _TrackerOverrides | None:
    raw = config.get("telemetry.hamilton_tracker")
    if not isinstance(raw, Mapping):
        return None
    overrides = _TrackerOverrides(
        enabled=_coerce_bool(raw.get("enabled")),
        project_id=_coerce_project_id_override(raw.get("project_id")),
        username=_coerce_optional_str(raw.get("username")),
        dag_name=_coerce_optional_str(raw.get("dag_name")),
        api_url=_coerce_optional_str(raw.get("api_url")),
        ui_url=_coerce_optional_str(raw.get("ui_url")),
        capture_data_statistics=_coerce_bool(raw.get("capture_data_statistics")),
        max_list_length=_coerce_optional_int(raw.get("max_list_length")),
        max_dict_length=_coerce_optional_int(raw.get("max_dict_length")),
        config_uri=_coerce_optional_str(raw.get("config_uri")),
        tags=_coerce_tags(raw.get("tags")),
    )
    return overrides if _has_tracker_overrides(overrides) else None


def _coerce_optional_str(value: object | None) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _coerce_project_id_override(value: object | None) -> str | None:
    if not isinstance(value, (str, int)):
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _coerce_optional_int(value: object | None) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _coerce_bool(value: object | None) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _coerce_tags(value: object | None) -> tuple[tuple[str, str], ...] | None:
    if not isinstance(value, Mapping):
        return None
    return tuple((str(key), str(item)) for key, item in value.items())


def _has_tracker_overrides(overrides: _TrackerOverrides) -> bool:
    return any(
        value is not None
        for value in (
            overrides.enabled,
            overrides.project_id,
            overrides.username,
            overrides.dag_name,
            overrides.api_url,
            overrides.ui_url,
            overrides.capture_data_statistics,
            overrides.max_list_length,
            overrides.max_dict_length,
            overrides.config_uri,
            overrides.tags,
        )
    )


def _merge_tracker_settings(
    base: HamiltonTrackerSettings,
    overrides: _TrackerOverrides,
) -> HamiltonTrackerSettings:
    return HamiltonTrackerSettings(
        enabled=overrides.enabled if overrides.enabled is not None else base.enabled,
        project_id=overrides.project_id if overrides.project_id is not None else base.project_id,
        username=overrides.username if overrides.username is not None else base.username,
        dag_name=overrides.dag_name if overrides.dag_name is not None else base.dag_name,
        api_url=overrides.api_url if overrides.api_url is not None else base.api_url,
        ui_url=overrides.ui_url if overrides.ui_url is not None else base.ui_url,
        capture_data_statistics=(
            overrides.capture_data_statistics
            if overrides.capture_data_statistics is not None
            else base.capture_data_statistics
        ),
        max_list_length=(
            overrides.max_list_length
            if overrides.max_list_length is not None
            else base.max_list_length
        ),
        max_dict_length=(
            overrides.max_dict_length
            if overrides.max_dict_length is not None
            else base.max_dict_length
        ),
        tags=overrides.tags if overrides.tags is not None else base.tags,
    )


def _build_hamilton_tracker_adapter(
    *,
    env: BuildEnv,
    run_id: str,
    domain: str | None,
    cache_dir: Path | None,
    diagnostics_path: Path | None,
) -> object | None:
    runtime_settings = load_runtime_settings().observability
    tracker_settings = runtime_settings.hamilton_tracker
    overrides = _tracker_overrides_from_config(env.config)
    config_uri = overrides.config_uri if overrides is not None else None
    if overrides is not None:
        tracker_settings = _merge_tracker_settings(tracker_settings, overrides)
    if not tracker_settings.enabled:
        return None
    if not tracker_settings.project_id or not tracker_settings.username:
        log.warning("Hamilton tracker enabled but project_id/username are not configured")
        return None
    try:
        hamilton_adapters = importlib.import_module("hamilton_sdk.adapters")
    except ModuleNotFoundError as exc:
        log.warning("Hamilton tracker enabled but hamilton_sdk is missing: %s", exc)
        return None

    tracker_cls = getattr(hamilton_adapters, "HamiltonTracker", None)
    if tracker_cls is None:
        log.warning("HamiltonTracker adapter is unavailable in hamilton_sdk.adapters")
        return None

    _apply_tracker_constants(tracker_settings, config_uri=config_uri)
    tag_context = _TrackerTagContext(
        env=env,
        run_id=run_id,
        domain=domain,
        deployment_environment=runtime_settings.deployment_environment,
        cache_dir=cache_dir,
        diagnostics_path=diagnostics_path,
    )
    tags = _build_tracker_tags(
        settings=tracker_settings,
        context=tag_context,
    )
    dag_name = tracker_settings.dag_name or env.snapshot.repo
    kwargs = {
        "project_id": _coerce_project_id(tracker_settings.project_id),
        "username": tracker_settings.username,
        "dag_name": dag_name,
        "tags": tags,
    }
    if tracker_settings.api_url:
        kwargs["hamilton_api_url"] = tracker_settings.api_url
    if tracker_settings.ui_url:
        kwargs["hamilton_ui_url"] = tracker_settings.ui_url
    tracker_adapter: object | None = None
    try:
        tracker_adapter = tracker_cls(**kwargs)
    except requests.exceptions.RequestException as exc:
        log.warning("Hamilton tracker unreachable; disabling tracker. error=%s", exc)
    except (TypeError, ValueError) as exc:
        log.warning("Failed to initialize HamiltonTracker: %s", exc)
    return tracker_adapter


def _resolve_telemetry_path(
    raw: object | None,
    *,
    repo_root: Path,
    default_path: Path,
) -> Path:
    if raw is None:
        return default_path
    if not isinstance(raw, str):
        msg = "telemetry.hooks output paths must be strings"
        raise TypeError(msg)
    cleaned = raw.strip()
    if not cleaned:
        return default_path
    path = Path(cleaned)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _resolve_optional_path(
    raw: object | None,
    *,
    repo_root: Path,
) -> Path | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        msg = "telemetry.hooks output paths must be strings"
        raise TypeError(msg)
    cleaned = raw.strip()
    if not cleaned:
        return None
    path = Path(cleaned)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _categorize_outputs(
    closure: tuple[str, ...],
    outputs: dict[str, Any],
    runtime: HamiltonRuntimeBundle,
) -> tuple[list[str], list[str], list[str]]:
    """Categorize outputs into computed/skipped/failed lists.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        Computed, skipped, and failed targets in that order.
    """
    computed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for target in closure:
        node_name = target_to_node_name(target, catalog=runtime.catalog)
        if node_name is None:
            failed.append(target)
            continue

        record = outputs.get(node_name)
        if not isinstance(record, TargetRunRecord):
            failed.append(target)
        elif record.status == "succeeded":
            computed.append(target)
        elif record.status == "skipped":
            skipped.append(target)
        else:
            failed.append(target)

    return computed, skipped, failed


def _skip_record(
    *,
    target_name: str,
    env: BuildEnv,
    catalog: DagCatalog,
    reason: str | None,
) -> TargetRunRecord:
    try:
        target = catalog.get(target_name)
    except KeyError:
        return TargetRunRecord(
            target=target_name,
            impl_kind="native",
            status="skipped",
            input_hash=None,
            error=reason,
        )
    run = NativeRunInfo(
        input_hash=None,
        options_hash=None,
        duration_ms=0.0,
        row_counts=None,
    )
    record = create_run_record(
        target,
        "skipped",
        None,
        inputs=RunRecordInputs(env=env, run=run, catalog=catalog),
    )
    if reason:
        return replace(record, error=reason)
    return record


def _failure_record(
    *,
    target_name: str,
    runtime: HamiltonRuntimeBundle,
    cache_keys: Mapping[str, str],
    error: str,
) -> TargetRunRecord:
    exc = RuntimeError(error)
    try:
        target = runtime.catalog.get(target_name)
    except KeyError:
        return TargetRunRecord(
            target=target_name,
            impl_kind="native",
            status="failed",
            input_hash=None,
            error=str(exc),
        )
    input_hash = _safe_input_hash(target, cache_keys=cache_keys, runtime=runtime)
    return create_run_record(
        target,
        "failed",
        input_hash,
        inputs=RunRecordInputs(error=exc),
    )


def _safe_input_hash(
    target: TargetDescriptor,
    *,
    cache_keys: Mapping[str, str],
    runtime: HamiltonRuntimeBundle,
) -> str | None:
    node_name = runtime.catalog.target_nodes.get(target.name)
    if node_name is None:
        return None
    return cache_keys.get(node_name)


def _ensure_failure_records(
    *,
    env: BuildEnv,
    runtime: HamiltonRuntimeBundle,
    closure: tuple[str, ...],
    outputs: dict[str, Any],
    error: str,
) -> None:
    cache_keys = _resolve_cache_keys(env=env, runtime=runtime)
    for target in closure:
        node_name = target_to_node_name(target, catalog=runtime.catalog)
        existing = outputs.get(node_name) if node_name is not None else None
        if isinstance(existing, TargetRunRecord):
            continue
        record = _failure_record(
            target_name=target,
            runtime=runtime,
            cache_keys=cache_keys,
            error=error,
        )
        key = node_name or f"__failed__{target}"
        outputs[key] = record


def _resolve_cache_keys(*, env: BuildEnv, runtime: HamiltonRuntimeBundle) -> dict[str, str]:
    resolver = runtime.cache_key_resolver
    if resolver is None:
        return {}
    inputs = ExecutionInputs(
        env=env,
        catalog=runtime.catalog,
        tag_query=runtime.tag_query,
        cache_index=runtime.cache_index,
        cache_key_resolver=runtime.cache_key_resolver,
        schema_index=runtime.schema_index,
        semantic_registry=runtime.semantic_registry,
        runtime_fingerprint=runtime.fingerprint,
    )
    input_values = _execution_input_mapping(inputs)
    node_set = set(resolver.node_dependencies)
    node_set.difference_update(input_values)
    return resolver.resolve_node_versions(
        nodes=node_set,
        input_values=input_values,
    )


def _apply_cache_keys(
    *,
    outputs: dict[str, Any],
    runtime: HamiltonRuntimeBundle,
) -> None:
    cache_adapter = runtime.cache_adapter
    if cache_adapter is None:
        cache_adapter = getattr(runtime.dr, "cache", None)
    if not isinstance(cache_adapter, HamiltonCacheAdapter):
        return
    if not cache_adapter.run_ids:
        return
    cache_run_id = cache_adapter.last_run_id
    resolver = CacheKeyResolver()
    for node_name in runtime.catalog.target_nodes.values():
        record = outputs.get(node_name)
        if not isinstance(record, TargetRunRecord):
            continue
        snapshot = resolver.resolve(
            cache_adapter,
            run_id=cache_run_id,
            node_name=node_name,
        )
        if snapshot.cache_key is None:
            continue
        outputs[node_name] = replace(record, input_hash=snapshot.cache_key)


def _map_closure_to_nodes(
    closure: tuple[str, ...],
    runtime: HamiltonRuntimeBundle,
) -> tuple[list[str], list[str]]:
    """Map closure targets to Hamilton node names.

    Returns
    -------
    tuple[list[str], list[str]]
        Node names for final execution variables, and missing targets.
    """
    final_vars: list[str] = []
    missing: list[str] = []

    for target in closure:
        node_name = target_to_node_name(target, catalog=runtime.catalog)
        if node_name is None:
            missing.append(target)
        else:
            final_vars.append(node_name)

    return final_vars, missing


def _execution_input_mapping(inputs: ExecutionInputs) -> dict[str, object]:
    return execution_input_mapping(inputs)


def _build_execution_inputs(context: _RunState) -> dict[str, object]:
    inputs = ExecutionInputs(
        env=context.env,
        catalog=context.runtime.catalog,
        tag_query=context.runtime.tag_query,
        cache_index=context.runtime.cache_index,
        cache_key_resolver=context.runtime.cache_key_resolver,
        schema_index=context.runtime.schema_index,
        semantic_registry=context.runtime.semantic_registry,
        runtime_fingerprint=context.runtime.fingerprint,
    )
    return _execution_input_mapping(inputs)


def _emit_execution_visualizations(
    *,
    context: _RunState,
    final_vars: list[str],
) -> None:
    settings = context.telemetry
    driver = context.runtime.driver
    if settings.display_all_functions_enabled:
        try:
            settings.display_all_functions_path.parent.mkdir(parents=True, exist_ok=True)
            driver.display_all_functions(
                output_file_path=str(settings.display_all_functions_path),
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            log.warning("build.telemetry.display_all_functions_failed error=%s", exc)
    if settings.visualize_execution_enabled:
        try:
            settings.visualize_execution_path.parent.mkdir(parents=True, exist_ok=True)
            driver.visualize_execution(
                list(final_vars),
                inputs=_build_execution_inputs(context),
                overrides={},
                output_file_path=str(settings.visualize_execution_path),
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            log.warning("build.telemetry.visualize_execution_failed error=%s", exc)


def _dataset_manifest_exists(env: BuildEnv, table_key: str) -> bool:
    dataset_root_dir = env.paths.dataset_root_dir
    if dataset_root_dir is None:
        return False
    manifest_path = dataset_manifest_path(
        dataset_root=dataset_root_dir,
        table_key=table_key,
        snapshot_id=env.commit,
    )
    return manifest_path.is_file()


def _table_key_exists_in_gateway(env: BuildEnv, table_key: str) -> bool | None:
    gateway = env.gateway
    if gateway is None:
        return None
    if "." not in table_key:
        return None
    schema, table = table_key.split(".", maxsplit=1)
    if not schema or not table:
        return None
    policy = getattr(gateway, "policy", None)
    if policy is None:
        return None
    try:
        return policy.table_exists(schema=schema, table=table)
    except (AttributeError, DuckDBError, RuntimeError, TypeError, ValueError):
        return None


def _table_key_exists(env: BuildEnv, table_key: str) -> bool:
    gateway_exists = _table_key_exists_in_gateway(env, table_key)
    if gateway_exists is not None:
        return gateway_exists
    return _dataset_manifest_exists(env, table_key)


def _preflight_missing_inputs(
    *,
    env: BuildEnv,
    runtime: HamiltonRuntimeBundle,
    closure: tuple[str, ...],
) -> dict[str, _MissingInputs]:
    catalog = runtime.catalog
    produced_table_keys = {
        output.key
        for target in closure
        for output in catalog.table_outputs_by_target.get(target, ())
    }
    surfaces = runtime.catalog.io_surfaces
    missing_by_target: dict[str, _MissingInputs] = {}

    for target in closure:
        surface = surfaces.get(target)
        if surface is None:
            continue
        missing_required: set[str] = set()
        missing_optional: set[str] = set()
        optional_inputs = optional_inputs_for_target(target)
        for read in surface.reads:
            table_key = read.table_key
            if table_key in produced_table_keys:
                continue
            if not _table_key_exists(env, table_key):
                if table_key in optional_inputs:
                    missing_optional.add(table_key)
                else:
                    missing_required.add(table_key)
        if missing_required or missing_optional:
            missing_by_target[target] = _MissingInputs(
                required=tuple(sorted(missing_required)),
                optional=tuple(sorted(missing_optional)),
            )
    return missing_by_target


def _blocked_targets(catalog: DagCatalog, roots: set[str]) -> set[str]:
    blocked = set(roots)
    queue = list(roots)
    while queue:
        current = queue.pop()
        for dependent in catalog.dependents_of(current):
            if dependent in blocked:
                continue
            blocked.add(dependent)
            queue.append(dependent)
    return blocked


def _preflight_blocked_records(
    *,
    env: BuildEnv,
    runtime: HamiltonRuntimeBundle,
    closure: tuple[str, ...],
    missing_by_target: Mapping[str, _MissingInputs],
) -> tuple[dict[str, TargetRunRecord], set[str]]:
    catalog = runtime.catalog
    cache_keys = _resolve_cache_keys(env=env, runtime=runtime)
    required_roots = {target for target, missing in missing_by_target.items() if missing.required}
    optional_roots = {
        target
        for target, missing in missing_by_target.items()
        if missing.optional and not missing.required
    }
    blocked_failed = _blocked_targets(catalog, required_roots)
    blocked_skipped = _blocked_targets(catalog, optional_roots) - blocked_failed
    blocked_records: dict[str, TargetRunRecord] = {}
    required_list = ", ".join(sorted(required_roots))
    optional_list = ", ".join(sorted(optional_roots))

    for target in closure:
        if target in blocked_failed:
            if target in missing_by_target and missing_by_target[target].required:
                missing_tables = ", ".join(missing_by_target[target].required)
                error = f"Missing input tables: {missing_tables}"
            else:
                error = f"Missing upstream inputs: {required_list}"
            record = _failure_record(
                target_name=target,
                runtime=runtime,
                cache_keys=cache_keys,
                error=error,
            )
        elif target in blocked_skipped:
            if target in missing_by_target and missing_by_target[target].optional:
                missing_tables = ", ".join(missing_by_target[target].optional)
                reason = f"Missing optional input tables: {missing_tables}"
            else:
                reason = f"Missing upstream inputs: {optional_list}"
            record = _skip_record(
                target_name=target,
                env=env,
                catalog=catalog,
                reason=reason,
            )
        else:
            continue
        node_name = target_to_node_name(target, catalog=runtime.catalog)
        key = node_name or f"__preflight__{target}"
        blocked_records[key] = record

    return blocked_records, blocked_failed | blocked_skipped


def _apply_preflight(
    *,
    context: _RunState,
    closure: tuple[str, ...],
    final_vars: list[str],
) -> tuple[list[str], dict[str, TargetRunRecord]]:
    preflight_missing = _preflight_missing_inputs(
        env=context.env,
        runtime=context.runtime,
        closure=closure,
    )
    if not preflight_missing:
        return final_vars, {}

    preflight_records, blocked_targets = _preflight_blocked_records(
        env=context.env,
        runtime=context.runtime,
        closure=closure,
        missing_by_target=preflight_missing,
    )
    if not blocked_targets:
        return final_vars, preflight_records

    adjusted: list[str] = []
    for target in closure:
        if target in blocked_targets:
            continue
        node_name = target_to_node_name(target, catalog=context.runtime.catalog)
        if node_name is None:
            continue
        adjusted.append(node_name)
    return adjusted, preflight_records


@dataclass(frozen=True)
class _FinalizeInputs:
    writer: BuildRunWriter
    closure: tuple[str, ...]
    outputs: dict[str, Any]
    error: str | None


def _finalize_run(
    *,
    context: _RunState,
    inputs: _FinalizeInputs,
) -> HamiltonBuildResult:
    catalog = context.runtime.catalog
    computed, skipped, failed = _categorize_outputs(inputs.closure, inputs.outputs, context.runtime)
    duration_ms = context.duration_ms
    success = not failed and inputs.error is None

    _write_failed_targets_diagnostic(
        context=context,
        failed_targets=failed,
        outputs=inputs.outputs,
    )

    records: list[TargetRunRecord] = [
        value for value in inputs.outputs.values() if isinstance(value, TargetRunRecord)
    ]
    inputs.writer.save_run_targets(env=context.env, run_id=context.run_id, records=records)
    inputs.writer.persist_asset_catalog(
        env=context.env,
        run_id=context.run_id,
        catalog=catalog,
        records=records,
    )

    error_summary = inputs.error or (f"{len(failed)} targets failed" if failed else None)
    inputs.writer.complete_run(
        run_id=context.run_id,
        success=success,
        computed_targets=computed,
        skipped_targets=skipped,
        error_summary=error_summary,
    )
    inputs.writer.write_run_report(
        inputs=RunReportInputs(
            env=context.env,
            run_id=context.run_id,
            catalog=catalog,
            records=records,
            computed_targets=computed,
            skipped_targets=skipped,
            failed_targets=failed,
            started_at=context.started_at,
            duration_ms=duration_ms,
            success=success,
            error_summary=error_summary,
        )
    )

    log.info(
        "build.hamilton.executor.complete run_id=%s success=%s duration_ms=%.1f",
        context.run_id,
        success,
        duration_ms,
    )
    record_build_event(
        "build.run.complete",
        success=success,
        duration_ms=duration_ms,
        computed_targets_count=len(computed),
        skipped_targets_count=len(skipped),
        failed_targets_count=len(failed),
        error=error_summary,
    )
    _emit_metadata_bundle(
        env=context.env,
        runtime=context.runtime,
        run_id=context.run_id,
    )

    return HamiltonBuildResult(
        requested=context.targets,
        closure=inputs.closure,
        computed_targets=tuple(computed),
        skipped_targets=tuple(skipped),
        failed_targets=tuple(failed),
        outputs=inputs.outputs,
        success=success,
        duration_ms=duration_ms,
        error=inputs.error,
        run_id=context.run_id,
        runtime=context.runtime,
    )


def _emit_metadata_bundle(
    *,
    env: BuildEnv,
    runtime: HamiltonRuntimeBundle,
    run_id: str,
) -> None:
    bundle = env.metadata_bundle
    if bundle is None:
        return
    generated_at = bundle.generated_at
    _emit_contract_catalog(bundle=bundle, env=env, run_id=run_id, generated_at=generated_at)
    _emit_schema_manifest(bundle=bundle, runtime=runtime, run_id=run_id, generated_at=generated_at)
    _emit_dataflow(bundle=bundle, run_id=run_id)
    _emit_lineage(
        bundle=bundle,
        env=env,
        runtime=runtime,
        run_id=run_id,
        generated_at=generated_at,
    )
    _finalize_bundle(bundle=bundle, run_id=run_id)


def _emit_contract_catalog(
    *,
    bundle: BuildMetadataBundleWriter,
    env: BuildEnv,
    run_id: str,
    generated_at: datetime,
) -> None:
    try:
        contract_payload = build_contract_catalog_payload(include_views=True)
        contract_payload = {
            **contract_payload,
            "generated_at": generated_at.isoformat(),
            "repo": env.repo,
            "commit": env.commit,
        }
        contract_record = bundle.write_json(
            "contracts/contract_catalog.json",
            contract_payload,
            schema_version="v1",
            indent=2,
        )
        bundle.write_text("contracts/contract_catalog.hash", f"{contract_record.sha256}\n")
    except (OSError, TypeError, ValueError) as exc:
        log.warning("build.metadata.contract_catalog_failed run_id=%s error=%s", run_id, exc)


def _emit_schema_manifest(
    *,
    bundle: BuildMetadataBundleWriter,
    runtime: HamiltonRuntimeBundle,
    run_id: str,
    generated_at: datetime,
) -> None:
    schema_index = runtime.schema_index
    if schema_index is None:
        log.warning(
            "build.metadata.schema_manifest_skipped run_id=%s reason=missing_schema_index",
            run_id,
        )
        return
    try:
        manifest = compile_schema_manifest(
            provider=get_schema_provider(),
            context=SchemaManifestContext(
                catalog=runtime.catalog,
                schema_index=schema_index,
                tag_query=runtime.tag_query,
            ),
            request=SchemaManifestRequest(
                all_targets=True,
                stable=True,
                version="v2",
                include_views=True,
                include_artifacts=True,
                include_provenance=True,
                infer_native=False,
                batch_infer_native=False,
            ),
        )
        bundle.write_json(
            "schema/schema_manifest.json",
            manifest.to_json_obj(),
            schema_version=manifest.version,
            indent=2,
        )
        catalog_hash = bundle.catalog_hash_for_manifest(manifest)
        versions, registry = schema_registry_from_manifest(
            manifest,
            catalog_hash=catalog_hash,
            generated_at=generated_at,
        )
        bundle.write_schema_registry(
            "schema/schema_registry.json",
            registry,
            schema_version="v1",
        )
        bundle.write_schema_versions(
            "schema/schema_versions.jsonl",
            versions,
            schema_version="v1",
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        log.warning("build.metadata.schema_manifest_failed run_id=%s error=%s", run_id, exc)


def _emit_dataflow(*, bundle: BuildMetadataBundleWriter, run_id: str) -> None:
    try:
        contracts = list(iter_contracts())
        nodes, edges = dataflow_from_contracts(contracts)
        for node in nodes:
            bundle.append_jsonl("dataflow/dataset_nodes.jsonl", node, schema_version="v1")
        for edge in edges:
            bundle.append_jsonl("dataflow/dataset_edges.jsonl", edge, schema_version="v1")
    except (RuntimeError, TypeError, ValueError) as exc:
        log.warning("build.metadata.dataflow_failed run_id=%s error=%s", run_id, exc)


def _emit_lineage(
    *,
    bundle: BuildMetadataBundleWriter,
    env: BuildEnv,
    runtime: HamiltonRuntimeBundle,
    run_id: str,
    generated_at: datetime,
) -> None:
    view_lineage, column_lineage = _safe_view_lineage(env=env, runtime=runtime, run_id=run_id)
    try:
        lineage_context = DerivedLineageContext(
            repo=env.repo,
            commit=env.commit,
            created_at=generated_at,
            view_lineage=view_lineage,
            column_lineage=column_lineage,
        )
        edges, columns = derived_lineage_from_catalog(
            runtime.catalog,
            context=lineage_context,
        )
        for edge in edges:
            bundle.append_jsonl("lineage/derived_edges.jsonl", edge, schema_version="v1")
        for column in columns:
            bundle.append_jsonl("lineage/derived_columns.jsonl", column, schema_version="v1")
    except (RuntimeError, TypeError, ValueError) as exc:
        log.warning("build.metadata.lineage_failed run_id=%s error=%s", run_id, exc)


def _safe_view_lineage(
    *,
    env: BuildEnv,
    runtime: HamiltonRuntimeBundle,
    run_id: str,
) -> tuple[dict[str, frozenset[str]] | None, dict[str, dict[str, frozenset[str]]] | None]:
    try:
        return view_lineage_payload(env=env, catalog=runtime.catalog)
    except (RuntimeError, TypeError, ValueError) as exc:
        log.warning("build.metadata.view_lineage_failed run_id=%s error=%s", run_id, exc)
        return None, None


def _finalize_bundle(*, bundle: BuildMetadataBundleWriter, run_id: str) -> None:
    try:
        bundle.finalize()
    except (OSError, RuntimeError, ValueError) as exc:
        log.warning("build.metadata.bundle_finalize_failed run_id=%s error=%s", run_id, exc)


@dataclass(frozen=True)
class HamiltonBuildResult:
    """Result of a Hamilton-based build execution.

    Attributes
    ----------
    requested
        Tuple of target names that were requested by the user.
    closure
        Tuple of target names in the full dependency closure.
    computed_targets
        Targets that were actually computed (status="succeeded").
    skipped_targets
        Targets that were skipped (status="skipped").
    failed_targets
        Targets that failed during execution (status="failed").
    outputs
        Dictionary mapping Hamilton node names to their outputs.
    success
        Whether all requested targets succeeded.
    duration_ms
        Total execution duration in milliseconds.
    error
        Error message if the entire execution failed.
    run_id
        Unique identifier for this build run.
    runtime
        Reference to the HamiltonRuntimeBundle for mapping lookups.
    """

    requested: tuple[str, ...]
    closure: tuple[str, ...] = ()
    computed_targets: tuple[str, ...] = ()
    skipped_targets: tuple[str, ...] = ()
    failed_targets: tuple[str, ...] = ()
    outputs: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    duration_ms: float = 0.0
    error: str | None = None
    run_id: str = ""
    runtime: HamiltonRuntimeBundle | None = None

    def get_record(self, target_name: str) -> TargetRunRecord | None:
        """Get the execution record for a target.

        Returns
        -------
        TargetRunRecord | None
            Execution record for the target, if present.
        """
        catalog = self.runtime.catalog if self.runtime is not None else None
        node_name = target_to_node_name(target_name, catalog=catalog)
        if node_name is not None:
            value = self.outputs.get(node_name)
            if isinstance(value, TargetRunRecord):
                return value
        for value in self.outputs.values():
            if isinstance(value, TargetRunRecord) and value.target == target_name:
                return value
        return None


def _options_from_overrides(
    *,
    options: BuildExecutionOptions | None,
    overrides: Mapping[str, object],
) -> BuildExecutionOptions:
    if options is not None:
        resolved = _ensure_no_overrides(options, overrides)
        resolved.validate()
        return resolved
    if not overrides:
        resolved = BuildExecutionOptions()
        resolved.validate()
        return resolved
    _ensure_known_overrides(overrides)
    data = _build_override_data(overrides)
    resolved = BuildExecutionOptions(**cast("BuildExecutionOptionsData", data))
    resolved.validate()
    return resolved


def _ensure_no_overrides(
    options: BuildExecutionOptions,
    overrides: Mapping[str, object],
) -> BuildExecutionOptions:
    if overrides:
        msg = "Pass either options or keyword overrides, not both."
        raise TypeError(msg)
    return options


def _ensure_known_overrides(overrides: Mapping[str, object]) -> None:
    unknown = sorted(set(overrides) - _EXECUTOR_OVERRIDE_KEYS)
    if unknown:
        msg = f"Unknown BuildExecutionOptions overrides: {', '.join(unknown)}"
        raise TypeError(msg)


def _build_override_data(overrides: Mapping[str, object]) -> dict[str, object]:
    return {
        target_key: overrides[source_key]
        for source_key, target_key in _EXECUTOR_OVERRIDE_MAP
        if source_key in overrides
    }


def _prepare_run_setup(
    *,
    env: BuildEnv,
    targets: list[str],
    options: BuildExecutionOptions,
) -> _RunSetup:
    resolved_targets = tuple(_ensure_intrinsic_targets(targets))
    run_id = _generate_run_id()
    start_build_log(env=env, run_id=run_id)
    record_build_event(
        "build.run.start",
        requested_targets_count=len(resolved_targets),
    )
    metadata_bundle = BuildMetadataBundleWriter(
        env.paths.build_dir / "metadata",
        run_id=run_id,
        repo=env.repo,
        commit=env.commit,
    )
    env = replace(env, metadata_bundle=metadata_bundle)
    writer = BuildRunWriter(metadata_bundle=metadata_bundle)
    cache_dir = options.resolved_cache_dir(env=env)
    log_handler = _install_run_log_handler(env.paths.build_dir, run_id)
    diagnostics_path = diagnostics_dir(env.paths.build_dir)
    telemetry_settings = _load_telemetry_hooks_settings(
        env=env,
        diagnostics_path=diagnostics_path,
    )
    ui_settings = _load_hamilton_ui_settings(env=env)
    ui_handle = _start_hamilton_ui(ui_settings, repo_root=env.snapshot.repo_root)
    cache_logger_handle = _configure_cache_logger(telemetry_settings)
    hang_watchdog = _start_hang_watchdog(telemetry_settings)
    return _RunSetup(
        env=env,
        run_id=run_id,
        resolved_targets=resolved_targets,
        writer=writer,
        cache_dir=cache_dir,
        log_handler=log_handler,
        diagnostics_path=diagnostics_path,
        telemetry_settings=telemetry_settings,
        cache_logger_handle=cache_logger_handle,
        hang_watchdog=hang_watchdog,
        ui_handle=ui_handle,
    )


class HamiltonBuildExecutor:
    """Execute build targets using Hamilton Driver.

    Parameters
    ----------
    options
        Optional ``BuildExecutionOptions`` instance configuring execution behavior.
    **overrides
        Keyword overrides matching ``BuildExecutionOptions`` fields (profile, cache, plugins).
        Only one of ``options`` or keyword overrides may be supplied.
    """

    def __init__(
        self,
        *,
        options: BuildExecutionOptions | None = None,
        **overrides: object,
    ) -> None:
        """Initialize the Hamilton executor."""
        self._options = _options_from_overrides(options=options, overrides=overrides)

    @property
    def profile(self) -> str | None:
        """Return the configured profile name."""
        return self._options.profile

    def run(
        self,
        *,
        env: BuildEnv,
        targets: list[str],
        domain: str | None = None,
    ) -> HamiltonBuildResult:
        """Execute build targets using Hamilton.

        Parameters
        ----------
        env
            Build environment for this execution.
        targets
            Target names to execute.
        domain
            Optional domain identifier for telemetry context.

        Returns
        -------
        HamiltonBuildResult
            Structured result containing outputs and status details.
        """
        logging.captureWarnings(capture=True)
        setup = _prepare_run_setup(
            env=env,
            targets=targets,
            options=self._options,
        )
        try:
            runtime, telemetry_hook = self._build_runtime(
                setup=setup,
                domain=domain,
            )

            context = _RunState(
                env=setup.env,
                targets=setup.resolved_targets,
                runtime=runtime,
                run_id=setup.run_id,
                cache_dir=setup.cache_dir,
                telemetry=setup.telemetry_settings,
                start_time=time.perf_counter(),
                started_at=datetime.now(tz=UTC),
                domain=domain,
            )
            return self._run_with_state(
                context=context,
                writer=setup.writer,
                telemetry_hook=telemetry_hook,
            )
        except Exception as exc:
            error_summary = str(exc)
            exception_type = type(exc).__name__
            record_build_event(
                "build.runtime.exception",
                exception_type=exception_type,
                error=error_summary,
            )
            events = _persist_build_log_from_buffer(writer=setup.writer)
            _write_failure_snapshot_from_context(
                context=_FailureSnapshotContext(
                    run_id=setup.run_id,
                    repo=setup.env.repo,
                    commit=setup.env.commit,
                    domain=domain,
                    requested_targets=setup.resolved_targets,
                    build_dir=setup.env.paths.build_dir,
                ),
                error_summary=error_summary,
                exception_type=exception_type,
                events=events,
            )
            log.exception("build.hamilton.executor.bootstrap_error run_id=%s", setup.run_id)
            raise
        finally:
            if setup.hang_watchdog is not None:
                setup.hang_watchdog.close()
            if setup.cache_logger_handle is not None:
                setup.cache_logger_handle.close()
            if setup.ui_handle is not None:
                setup.ui_handle.stop()
            _teardown_run_logging(setup.log_handler)

    def _run_with_state(
        self,
        *,
        context: _RunState,
        writer: BuildRunWriter,
        telemetry_hook: NodeTelemetryHook | None,
    ) -> HamiltonBuildResult:
        error_summary: str | None = None
        exception_type: str | None = None

        try:
            result, error_summary = self._execute_run(
                context=context,
                writer=writer,
                telemetry_hook=telemetry_hook,
            )
        except Exception as exc:
            exception_type = type(exc).__name__
            error_summary = str(exc)
            record_build_event(
                "build.runtime.exception",
                exception_type=exception_type,
                error=error_summary,
            )
            log.exception("build.hamilton.executor.error run_id=%s", context.run_id)
            raise
        else:
            return result
        finally:
            events = _persist_build_log(writer=writer)
            _write_failure_snapshot(
                context=context,
                error_summary=error_summary,
                exception_type=exception_type,
                events=events,
            )

    def _execute_run(
        self,
        *,
        context: _RunState,
        writer: BuildRunWriter,
        telemetry_hook: NodeTelemetryHook | None,
    ) -> tuple[HamiltonBuildResult, str | None]:
        requested_targets = list(context.targets)
        log.info(
            "build.hamilton.executor.start run_id=%s targets=%s",
            context.run_id,
            requested_targets,
        )

        writer.start_run(
            env=context.env,
            run_id=context.run_id,
            requested_targets=requested_targets,
            started_at=context.started_at,
        )

        plan, failure_result, error_summary = self._prepare_execution(
            context=context,
            writer=writer,
            requested_targets=requested_targets,
        )
        if failure_result is not None:
            return failure_result, error_summary
        if plan is None:
            return self._make_error_result(context, "Build planning failed"), error_summary

        _emit_execution_visualizations(context=context, final_vars=plan.final_vars)
        outputs, error = self._execute_final_vars(
            context=context,
            final_vars=plan.final_vars,
            telemetry_hook=telemetry_hook,
        )
        outputs.update(plan.preflight_records)
        if error:
            _ensure_failure_records(
                env=context.env,
                runtime=context.runtime,
                closure=plan.closure,
                outputs=outputs,
                error=error,
            )

        _apply_cache_keys(outputs=outputs, runtime=context.runtime)
        result = _finalize_run(
            context=context,
            inputs=_FinalizeInputs(
                writer=writer,
                closure=plan.closure,
                outputs=outputs,
                error=error,
            ),
        )
        if not result.success:
            error_summary = result.error

        _emit_diagnostics_safe(
            context=context,
            result=result,
            telemetry_hook=telemetry_hook,
        )
        return result, error_summary

    def _prepare_execution(
        self,
        *,
        context: _RunState,
        writer: BuildRunWriter,
        requested_targets: list[str],
    ) -> tuple[_ExecutionPlan | None, HamiltonBuildResult | None, str | None]:
        catalog = context.runtime.catalog
        closure = self._compute_closure(catalog, requested_targets, context.run_id)
        if closure is None:
            error_summary = "Failed to compute closure"
            result = self._complete_failure_result(
                context=context,
                writer=writer,
                error_summary=error_summary,
            )
            return None, result, error_summary

        preflight_ok, preflight_error = _run_preflight(context=context, catalog=catalog)
        if not preflight_ok:
            error_summary = preflight_error or "DAG preflight failed"
            result = self._complete_failure_result(
                context=context,
                writer=writer,
                error_summary=error_summary,
            )
            return None, result, error_summary

        final_vars, missing = _map_closure_to_nodes(closure, context.runtime)
        if missing:
            error_summary = f"Missing node mappings for: {missing}"
            result = self._complete_failure_result(
                context=context,
                writer=writer,
                error_summary=error_summary,
                closure=closure,
                missing=missing,
            )
            return None, result, error_summary

        final_vars, preflight_records = _apply_preflight(
            context=context,
            closure=closure,
            final_vars=final_vars,
        )
        plan = _ExecutionPlan(
            closure=closure,
            final_vars=final_vars,
            preflight_records=preflight_records,
        )
        return plan, None, None

    def _complete_failure_result(
        self,
        *,
        context: _RunState,
        writer: BuildRunWriter,
        error_summary: str,
        closure: tuple[str, ...] | None = None,
        missing: list[str] | None = None,
    ) -> HamiltonBuildResult:
        writer.complete_run(
            run_id=context.run_id,
            success=False,
            computed_targets=(),
            skipped_targets=(),
            error_summary=error_summary,
        )
        if missing is not None and closure is not None:
            return self._make_missing_result(context, closure, missing)
        return self._make_error_result(context, error_summary)

    def _execute_final_vars(
        self,
        *,
        context: _RunState,
        final_vars: list[str],
        telemetry_hook: NodeTelemetryHook | None,
    ) -> tuple[dict[str, Any], str | None]:
        try:
            if final_vars:
                outputs, error = self._execute_dag(context, final_vars)
            else:
                outputs, error = {}, None
        finally:
            if telemetry_hook is not None:
                telemetry_hook.flush()
        return outputs, error

    def _effective_max_workers(self, catalog: DagCatalog) -> int | None:
        return effective_max_workers_for_graph(run_options=self._options, catalog=catalog)

    def _build_runtime(
        self,
        *,
        setup: _RunSetup,
        domain: str | None,
    ) -> tuple[HamiltonRuntimeBundle, NodeTelemetryHook | None]:
        """Build Hamilton runtime with configured mode and lifecycle adapters.

        Returns
        -------
        HamiltonRuntimeBundle
            Configured runtime bundle with driver and catalog.
        """
        config = _base_hamilton_config(env=setup.env, options=self._options)
        thread_name_prefix = "codeintel-build"
        execution_profile = build_execution_profile(
            env=setup.env,
            options=self._options,
            max_workers=self._options.max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        dynamic_config = apply_dynamic_execution_config(
            config=config,
            profile=execution_profile,
        )
        materializers = _resolve_materializers(setup.env.execution_settings.materializers)
        telemetry_hook: NodeTelemetryHook | None = None

        telemetry_output = _resolve_telemetry_path(
            setup.env.config.get("telemetry.hooks.telemetry_output_path"),
            repo_root=setup.env.snapshot.repo_root,
            default_path=setup.diagnostics_path / "node_telemetry.jsonl",
        )
        io_telemetry_output = _resolve_telemetry_path(
            setup.env.config.get("telemetry.hooks.io_telemetry_output_path"),
            repo_root=setup.env.snapshot.repo_root,
            default_path=setup.diagnostics_path / "node_io_telemetry.jsonl",
        )
        hook_options = self._options.hook_options(
            telemetry_output_path=telemetry_output,
            io_telemetry_output_path=io_telemetry_output,
            progress_desc=setup.telemetry_settings.progress_desc,
        )
        cache_adapter = _build_cache_adapter(
            run_id=setup.run_id,
            cache_dir=setup.cache_dir,
            enable_cache=self._options.enable_hamilton_cache,
        )

        def _adapter_factory(catalog: DagCatalog) -> list[LifecycleAdapter]:
            nonlocal telemetry_hook
            adapters: list[LifecycleAdapter] = []
            effective_max_workers = self._effective_max_workers(catalog)
            adapter_profile = build_execution_profile(
                env=setup.env,
                options=self._options,
                max_workers=effective_max_workers,
                thread_name_prefix=thread_name_prefix,
            )
            result_builder = BuildResultBuilder(
                allowed_nodes=tuple(catalog.target_nodes.values()),
            )
            parallel_adapter = build_parallel_adapter(
                adapter_profile,
                result_builder=result_builder,
                dynamic_enabled=dynamic_config.enabled,
            )
            if parallel_adapter is not None:
                adapters.append(parallel_adapter)
            else:
                adapters.append(h_base.DictResult())

            hooks = build_hooks(setup.run_id, setup.writer, options=hook_options)
            for hook in hooks:
                if isinstance(hook, NodeTelemetryHook):
                    telemetry_hook = hook
                adapters.append(cast("LifecycleAdapter", hook))

            adapters.extend(
                _build_diagnostic_adapters(
                    settings=setup.telemetry_settings,
                    run_id=setup.run_id,
                    default_ddog_root=setup.env.snapshot.repo,
                    cache_enabled=cache_adapter is not None,
                )
            )

            tracker_adapter = _build_hamilton_tracker_adapter(
                env=setup.env,
                run_id=setup.run_id,
                domain=domain,
                cache_dir=setup.cache_dir,
                diagnostics_path=setup.diagnostics_path,
            )
            if tracker_adapter is not None:
                adapters.append(cast("LifecycleAdapter", tracker_adapter))
            return adapters

        composition = compose_runtime(
            env=setup.env,
            config=config,
            options=BuildDriverOptions(
                adapter_factory=_adapter_factory,
                materializers=materializers or None,
                enable_cache=self._options.enable_hamilton_cache,
                cache_dir=str(setup.cache_dir),
                cache_adapter=cache_adapter,
            ),
        )
        return composition.bundle, telemetry_hook

    @staticmethod
    def _compute_closure(
        catalog: DagCatalog,
        targets: list[str],
        run_id: str,
    ) -> tuple[str, ...] | None:
        """Compute dependency closure, returning None on error.

        Returns
        -------
        tuple[str, ...] | None
            Ordered dependency closure, or None if computation failed.
        """
        try:
            return catalog.closure(targets)
        except (KeyError, ValueError):
            log.exception("build.hamilton.executor.closure_error run_id=%s", run_id)
            return None

    @staticmethod
    def _make_error_result(
        context: _RunState,
        error: str,
    ) -> HamiltonBuildResult:
        """Create error result for closure computation failure.

        Returns
        -------
        HamiltonBuildResult
            Error result indicating failed closure computation.
        """
        outputs: dict[str, Any] = {}
        cache_keys = _resolve_cache_keys(env=context.env, runtime=context.runtime)
        for target in context.targets:
            record = _failure_record(
                target_name=target,
                runtime=context.runtime,
                cache_keys=cache_keys,
                error=error,
            )
            outputs[f"__failed__{target}"] = record
        return HamiltonBuildResult(
            requested=context.targets,
            outputs=outputs,
            success=False,
            failed_targets=context.targets,
            duration_ms=context.duration_ms,
            error=error,
            run_id=context.run_id,
            runtime=context.runtime,
        )

    @staticmethod
    def _make_missing_result(
        context: _RunState,
        closure: tuple[str, ...],
        missing: list[str],
    ) -> HamiltonBuildResult:
        """Create error result for missing node mappings.

        Returns
        -------
        HamiltonBuildResult
            Error result indicating missing node mappings.
        """
        log.error("build.hamilton.executor.missing_targets targets=%s", missing)
        outputs: dict[str, Any] = {}
        cache_keys = _resolve_cache_keys(env=context.env, runtime=context.runtime)
        for target in missing:
            record = _failure_record(
                target_name=target,
                runtime=context.runtime,
                cache_keys=cache_keys,
                error=f"Missing node mappings for: {missing}",
            )
            outputs[f"__missing__{target}"] = record
        return HamiltonBuildResult(
            requested=context.targets,
            closure=closure,
            outputs=outputs,
            success=False,
            failed_targets=tuple(missing),
            duration_ms=context.duration_ms,
            error=f"Missing node mappings for: {missing}",
            run_id=context.run_id,
            runtime=context.runtime,
        )

    @staticmethod
    def _execute_dag(
        context: _RunState,
        final_vars: list[str],
    ) -> tuple[dict[str, Any], str | None]:
        """Execute the Hamilton DAG, returning (outputs, error).

        Parameters
        ----------
        context
            Execution state for this run.
        final_vars
            List of node names to execute.

        Returns
        -------
        tuple[dict[str, Any], str | None]
            Outputs keyed by node name, and optional error string.
        """
        try:
            with telemetry_context(
                run_id=context.run_id,
                domain=context.domain,
                repo_commit=RepoCommitContext(
                    repo=context.env.repo,
                    commit=context.env.commit,
                ),
            ):
                input_mapping = _build_execution_inputs(context)
                set_execution_active(active=True)
                try:
                    outputs = context.runtime.driver.execute(
                        list(final_vars),
                        inputs=input_mapping,
                    )
                finally:
                    set_execution_active(active=False)
        except Exception as exc:
            record_build_event(
                "build.runtime.error",
                exception_type=type(exc).__name__,
                error=str(exc),
            )
            log.exception("build.hamilton.executor.error run_id=%s", context.run_id)
            return {}, str(exc)
        else:
            return outputs, None


def _persist_build_log(
    *,
    writer: BuildRunWriter,
) -> list[dict[str, object]] | None:
    drained = drain_build_log()
    if drained is None:
        return None
    log_context, events = drained
    return _write_build_log_events(
        writer=writer,
        log_context=log_context,
        events=events,
    )


def _persist_build_log_from_buffer(
    *,
    writer: BuildRunWriter,
) -> list[dict[str, object]] | None:
    drained = drain_build_log()
    if drained is None:
        return None
    log_context, events = drained
    return _write_build_log_events(
        writer=writer,
        log_context=log_context,
        events=events,
    )


def _write_build_log_events(
    *,
    writer: BuildRunWriter,
    log_context: BuildLogContext,
    events: list[dict[str, object]],
) -> list[dict[str, object]]:
    path = writer.write_build_log(context=log_context, events=events)
    if path is None:
        return events
    log.info(
        "build.hamilton.executor.build_log_written run_id=%s event_count=%d path=%s",
        log_context.run_id,
        len(events),
        path,
    )
    return events


def _emit_diagnostics_safe(
    *,
    context: _RunState,
    result: HamiltonBuildResult,
    telemetry_hook: NodeTelemetryHook | None,
) -> None:
    try:
        diagnostics_targets = DiagnosticsTargets(
            requested=context.targets,
            computed=result.computed_targets,
            skipped=result.skipped_targets,
            failed=result.failed_targets,
        )
        diagnostics_inputs = DiagnosticsInputs(
            env=context.env,
            runtime=context.runtime,
            run_id=context.run_id,
            cache_dir=context.cache_dir,
            cache_adapter=context.runtime.cache_adapter,
            telemetry_records=telemetry_hook.last_flushed_records()
            if telemetry_hook is not None
            else None,
            targets=diagnostics_targets,
            duration_ms=result.duration_ms,
            domain=context.domain,
        )
        emit_diagnostics(diagnostics_inputs)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        log.warning(
            "build.hamilton.diagnostics_failed run_id=%s error=%s",
            context.run_id,
            exc,
        )


def _write_failed_targets_diagnostic(
    *,
    context: _RunState,
    failed_targets: Sequence[str],
    outputs: Mapping[str, Any],
) -> None:
    if not failed_targets:
        return
    diag_dir = diagnostics_dir(context.env.paths.build_dir)
    try:
        diag_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, RuntimeError) as exc:
        log.warning(
            "build.hamilton.failed_targets_dir_failed run_id=%s error=%s", context.run_id, exc
        )
        return

    record_map = {
        record.target: record for record in outputs.values() if isinstance(record, TargetRunRecord)
    }
    entries: list[dict[str, object]] = []
    for target in failed_targets:
        record = record_map.get(target)
        if record is None:
            entries.append(
                {
                    "target": target,
                    "status": "missing_record",
                    "error": "Missing TargetRunRecord for failed target",
                }
            )
            continue
        entries.append(
            {
                "target": target,
                "status": record.status,
                "error": record.error,
                "row_counts": record.row_counts,
                "dataset_count": len(record.datasets),
                "artifact_count": len(record.artifacts),
            }
        )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": context.run_id,
        "repo": context.env.repo,
        "commit": context.env.commit,
        "failed_targets": entries,
    }
    path = diag_dir / "failed_targets.json"
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        log.warning(
            "build.hamilton.failed_targets_write_failed run_id=%s error=%s",
            context.run_id,
            exc,
        )


def _install_run_log_handler(build_dir: Path, run_id: str) -> logging.Handler | None:
    log_dir = build_dir / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, RuntimeError):
        log.warning("build.hamilton.executor.log_dir_failed run_id=%s", run_id)
        return None
    handler = logging.FileHandler(log_dir / f"build_run_{run_id}.log", encoding="utf-8")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    return handler


def _teardown_run_logging(handler: logging.Handler | None) -> None:
    if handler is None:
        return
    root_logger = logging.getLogger()
    root_logger.removeHandler(handler)
    handler.flush()
    handler.close()


def _write_failure_snapshot(
    *,
    context: _RunState,
    error_summary: str | None,
    exception_type: str | None,
    events: list[dict[str, object]] | None,
) -> None:
    _write_failure_snapshot_from_context(
        context=_FailureSnapshotContext(
            run_id=context.run_id,
            repo=context.env.repo,
            commit=context.env.commit,
            domain=context.domain,
            requested_targets=context.targets,
            build_dir=context.env.paths.build_dir,
        ),
        error_summary=error_summary,
        exception_type=exception_type,
        events=events,
    )


def _write_failure_snapshot_from_context(
    *,
    context: _FailureSnapshotContext,
    error_summary: str | None,
    exception_type: str | None,
    events: list[dict[str, object]] | None,
) -> None:
    if not error_summary:
        return
    diag_dir = diagnostics_dir(context.build_dir)
    try:
        diag_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, RuntimeError):
        log.warning(
            "build.hamilton.executor.failure_dir_failed run_id=%s",
            context.run_id,
        )
        return
    payload = _failure_snapshot_payload(
        context=context,
        error_summary=error_summary,
        exception_type=exception_type,
        events=events,
    )
    _write_failure_snapshot_files(
        diag_dir=diag_dir,
        run_id=context.run_id,
        payload=payload,
    )


def _failure_snapshot_payload(
    *,
    context: _FailureSnapshotContext,
    error_summary: str,
    exception_type: str | None,
    events: list[dict[str, object]] | None,
) -> dict[str, object]:
    log_path = context.build_dir / "logs" / f"build_run_{context.run_id}.log"
    return {
        "run_id": context.run_id,
        "repo": context.repo,
        "commit": context.commit,
        "domain": context.domain,
        "requested_targets": list(context.requested_targets),
        "error": error_summary,
        "exception_type": exception_type,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "log_path": str(log_path),
        "build_log_event_count": len(events) if events is not None else 0,
        "build_log_tail": (events[-200:] if events else []),
    }


def _write_failure_snapshot_files(
    *,
    diag_dir: Path,
    run_id: str,
    payload: dict[str, object],
) -> None:
    snapshot_path = diag_dir / f"failure_{run_id}.json"
    latest_path = diag_dir / "failure_latest.json"
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    try:
        snapshot_path.write_text(encoded + "\n", encoding="utf-8")
        latest_path.write_text(encoded + "\n", encoding="utf-8")
    except (OSError, RuntimeError) as exc:
        log.warning(
            "build.hamilton.executor.failure_write_failed run_id=%s error=%s",
            run_id,
            exc,
        )


__all__ = [
    "HamiltonBuildExecutor",
    "HamiltonBuildResult",
]
