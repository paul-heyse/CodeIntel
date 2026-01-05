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

import importlib
import inspect
import logging
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import hamilton.base as h_base
from hamilton.caching.adapter import HamiltonCacheAdapter

from codeintel.build.execution_policy import effective_max_workers_for_graph
from codeintel.build.hamilton.adapters.parallel import create_parallel_adapter
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
from codeintel.build.hamilton.hooks import NodeTelemetryHook, build_hooks
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
from codeintel.runtime.runtime_bundle import RuntimeBundle

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import TypedDict

    from hamilton.io.materialization import ExtractorFactory, MaterializerFactory
    from hamilton.lifecycle.base import LifecycleAdapter

    from codeintel.build.hamilton.dag_catalog import DagCatalog, TargetDescriptor
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.core.config.settings import HamiltonTrackerSettings

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

_EXECUTOR_ALIASES: dict[str, str] = {
    "synchronous": "sync",
    "sync": "sync",
    "local": "sync",
    "thread": "thread",
    "threads": "thread",
    "threading": "thread",
    "process": "process",
    "processes": "process",
    "multiprocessing": "process",
    "mp": "process",
    "none": "none",
    "off": "none",
    "disabled": "none",
}

_EXECUTOR_CLASS_NAMES: dict[str, str] = {
    "sync": "SynchronousLocalTaskExecutor",
    "thread": "MultiThreadingExecutor",
    "process": "MultiProcessingExecutor",
}
_INTRINSIC_TARGETS: tuple[str, ...] = ("scip",)


@dataclass(frozen=True)
class _RunState:
    """Execution state shared across run steps."""

    env: BuildEnv
    targets: tuple[str, ...]
    runtime: RuntimeBundle
    run_id: str
    cache_dir: Path
    start_time: float
    started_at: datetime
    domain: str | None

    @property
    def duration_ms(self) -> float:
        """Return elapsed milliseconds for the run."""
        return (time.perf_counter() - self.start_time) * 1000


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


class _TrackingConstants(Protocol):
    CAPTURE_DATA_STATISTICS: bool
    MAX_LIST_LENGTH_CAPTURE: int
    MAX_DICT_LENGTH_CAPTURE: int


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


def _apply_tracker_constants(settings: HamiltonTrackerSettings) -> None:
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


def _normalize_executor_name(value: str | None, *, default: str) -> str:
    if value is None:
        return default
    normalized = value.strip().lower()
    if not normalized:
        return default
    return _EXECUTOR_ALIASES.get(normalized, normalized)


def _executor_kwargs(executor_cls: type[object], *, max_tasks: int | None) -> dict[str, object]:
    if max_tasks is None:
        return {}
    params = inspect.signature(executor_cls).parameters
    if "max_tasks" in params:
        return {"max_tasks": max_tasks}
    if "max_workers" in params:
        return {"max_workers": max_tasks}
    if "max_concurrent_tasks" in params:
        return {"max_concurrent_tasks": max_tasks}
    return {}


def _instantiate_task_executor(
    executor_cls: type[object],
    *,
    max_tasks: int | None,
    label: str,
) -> object | None:
    kwargs = _executor_kwargs(executor_cls, max_tasks=max_tasks)
    try:
        return executor_cls(**kwargs)
    except (TypeError, ValueError) as exc:
        log.warning("Failed to instantiate %s executor: %s", label, exc)
        return None


def _resolve_task_executor(
    *,
    name: str | None,
    default: str,
    max_tasks: int | None,
) -> object | None:
    normalized = _normalize_executor_name(name, default=default)
    if normalized == "none":
        return None
    class_name = _EXECUTOR_CLASS_NAMES.get(normalized)
    if class_name is None:
        log.warning("Unknown dynamic executor '%s', defaulting to %s", normalized, default)
        class_name = _EXECUTOR_CLASS_NAMES.get(default)
        if class_name is None:
            return None
    try:
        executors = importlib.import_module("hamilton.execution.executors")
    except ModuleNotFoundError as exc:
        log.warning("Dynamic execution requested but executors module missing: %s", exc)
        return None
    executor_cls = getattr(executors, class_name, None)
    if executor_cls is None:
        log.warning("Dynamic executor class %s is unavailable", class_name)
        return None
    return _instantiate_task_executor(
        executor_cls,
        max_tasks=max_tasks,
        label=normalized,
    )


def _resolve_dynamic_executors(
    *,
    enabled: bool,
    local_name: str | None,
    remote_name: str | None,
    max_tasks: int | None,
) -> tuple[bool, object | None, object | None]:
    if not enabled:
        return False, None, None
    local_executor = _resolve_task_executor(
        name=local_name,
        default="sync",
        max_tasks=None,
    )
    remote_executor = _resolve_task_executor(
        name=remote_name,
        default="thread",
        max_tasks=max_tasks,
    )
    if local_executor is None and remote_executor is None:
        log.warning("Dynamic execution enabled but no executors resolved; disabling")
        return False, None, None
    if local_executor is None:
        local_executor = _resolve_task_executor(
            name="sync",
            default="sync",
            max_tasks=None,
        )
    return True, local_executor, remote_executor


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


def _apply_dynamic_execution_config(
    *,
    config: dict[str, Any],
    env: BuildEnv,
    options: BuildExecutionOptions,
) -> tuple[bool, object | None, object | None]:
    dynamic_enabled, local_executor, remote_executor = _resolve_dynamic_executors(
        enabled=bool(env.execution_settings.dynamic_execution),
        local_name=env.execution_settings.dynamic_local_executor,
        remote_name=env.execution_settings.dynamic_remote_executor,
        max_tasks=env.execution_settings.dynamic_remote_max_tasks
        or options.max_workers
        or env.execution_settings.max_workers,
    )
    config["ci.dynamic_execution"] = dynamic_enabled
    config["ci_dynamic_module_records"] = dynamic_enabled
    if dynamic_enabled:
        if local_executor is not None:
            config["ci.dynamic_execution.local_executor"] = local_executor
        if remote_executor is not None:
            config["ci.dynamic_execution.remote_executor"] = remote_executor
    return dynamic_enabled, local_executor, remote_executor


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

    _apply_tracker_constants(tracker_settings)
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
    try:
        return tracker_cls(**kwargs)
    except (TypeError, ValueError) as exc:
        log.warning("Failed to initialize HamiltonTracker: %s", exc)
        return None


def _categorize_outputs(
    closure: tuple[str, ...],
    outputs: dict[str, Any],
    runtime: RuntimeBundle,
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
    runtime: RuntimeBundle,
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
    runtime: RuntimeBundle,
) -> str | None:
    node_name = runtime.catalog.target_nodes.get(target.name)
    if node_name is None:
        return None
    return cache_keys.get(node_name)


def _ensure_failure_records(
    *,
    env: BuildEnv,
    runtime: RuntimeBundle,
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


def _resolve_cache_keys(*, env: BuildEnv, runtime: RuntimeBundle) -> dict[str, str]:
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
    runtime: RuntimeBundle,
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
    runtime: RuntimeBundle,
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
    runtime: RuntimeBundle,
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
    runtime: RuntimeBundle,
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
    runtime: RuntimeBundle,
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
    runtime: RuntimeBundle,
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
    runtime: RuntimeBundle,
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
    runtime: RuntimeBundle,
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
        Reference to the RuntimeBundle for mapping lookups.
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
    runtime: RuntimeBundle | None = None

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
        resolved_targets = _ensure_intrinsic_targets(targets)
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
        cache_dir = self._options.resolved_cache_dir(env=env)
        runtime, telemetry_hook = self._build_runtime(
            env=env,
            run_id=run_id,
            writer=writer,
            cache_dir=cache_dir,
            domain=domain,
        )

        context = _RunState(
            env=env,
            targets=tuple(resolved_targets),
            runtime=runtime,
            run_id=run_id,
            cache_dir=cache_dir,
            start_time=time.perf_counter(),
            started_at=datetime.now(tz=UTC),
            domain=domain,
        )
        return self._run_with_state(
            context=context,
            writer=writer,
            telemetry_hook=telemetry_hook,
        )

    def _run_with_state(
        self,
        *,
        context: _RunState,
        writer: BuildRunWriter,
        telemetry_hook: NodeTelemetryHook | None,
    ) -> HamiltonBuildResult:
        catalog = context.runtime.catalog
        requested_targets = list(context.targets)

        try:
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

            closure = self._compute_closure(catalog, requested_targets, context.run_id)
            if closure is None:
                writer.complete_run(
                    run_id=context.run_id,
                    success=False,
                    computed_targets=(),
                    skipped_targets=(),
                    error_summary="Failed to compute closure",
                )
                return self._make_error_result(context, "Failed to compute closure")

            preflight_ok, preflight_error = _run_preflight(context=context, catalog=catalog)
            if not preflight_ok:
                writer.complete_run(
                    run_id=context.run_id,
                    success=False,
                    computed_targets=(),
                    skipped_targets=(),
                    error_summary=preflight_error,
                )
                return self._make_error_result(context, preflight_error or "DAG preflight failed")

            final_vars, missing = _map_closure_to_nodes(closure, context.runtime)
            if missing:
                writer.complete_run(
                    run_id=context.run_id,
                    success=False,
                    computed_targets=(),
                    skipped_targets=(),
                    error_summary=f"Missing node mappings for: {missing}",
                )
                return self._make_missing_result(context, closure, missing)

            final_vars, preflight_records = _apply_preflight(
                context=context,
                closure=closure,
                final_vars=final_vars,
            )

            try:
                if final_vars:
                    outputs, error = self._execute_dag(context, final_vars)
                else:
                    outputs, error = {}, None
            finally:
                if telemetry_hook is not None:
                    telemetry_hook.flush()

            outputs.update(preflight_records)

            if error:
                _ensure_failure_records(
                    env=context.env,
                    runtime=context.runtime,
                    closure=closure,
                    outputs=outputs,
                    error=error,
                )

            _apply_cache_keys(outputs=outputs, runtime=context.runtime)

            result = _finalize_run(
                context=context,
                inputs=_FinalizeInputs(
                    writer=writer,
                    closure=closure,
                    outputs=outputs,
                    error=error,
                ),
            )
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
            return result
        finally:
            _persist_build_log(context=context, writer=writer)

    def _effective_max_workers(self, catalog: DagCatalog) -> int | None:
        return effective_max_workers_for_graph(run_options=self._options, catalog=catalog)

    def _build_runtime(
        self,
        *,
        env: BuildEnv,
        run_id: str,
        writer: BuildRunWriter,
        cache_dir: Path,
        domain: str | None,
    ) -> tuple[RuntimeBundle, NodeTelemetryHook | None]:
        """Build Hamilton runtime with configured mode and lifecycle adapters.

        Returns
        -------
        RuntimeBundle
            Configured runtime bundle with driver and catalog.
        """
        config = _base_hamilton_config(env=env, options=self._options)
        dynamic_enabled, _, _ = _apply_dynamic_execution_config(
            config=config,
            env=env,
            options=self._options,
        )
        materializers = _resolve_materializers(env.execution_settings.materializers)
        telemetry_hook: NodeTelemetryHook | None = None

        diagnostics_path = diagnostics_dir(env.paths.build_dir)
        telemetry_output = diagnostics_path / "node_telemetry.jsonl"
        hook_options = self._options.hook_options(telemetry_output_path=telemetry_output)
        cache_adapter = _build_cache_adapter(
            run_id=run_id,
            cache_dir=cache_dir,
            enable_cache=self._options.enable_hamilton_cache,
        )

        def _adapter_factory(catalog: DagCatalog) -> list[LifecycleAdapter]:
            nonlocal telemetry_hook
            adapters: list[LifecycleAdapter] = []
            effective_max_workers = self._effective_max_workers(catalog)
            result_builder = BuildResultBuilder(
                allowed_nodes=tuple(catalog.target_nodes.values()),
            )
            parallel_adapter = None
            if not dynamic_enabled:
                parallel_adapter = create_parallel_adapter(
                    self._options.parallel_backend,
                    max_workers=effective_max_workers,
                    thread_name_prefix="codeintel-build",
                    result_builder=result_builder,
                )
            if parallel_adapter is not None:
                adapters.append(parallel_adapter)
            else:
                adapters.append(h_base.DictResult())

            hooks = build_hooks(run_id, writer, options=hook_options)
            for hook in hooks:
                if isinstance(hook, NodeTelemetryHook):
                    telemetry_hook = hook
                adapters.append(cast("LifecycleAdapter", hook))

            tracker_adapter = _build_hamilton_tracker_adapter(
                env=env,
                run_id=run_id,
                domain=domain,
                cache_dir=cache_dir,
                diagnostics_path=diagnostics_path,
            )
            if tracker_adapter is not None:
                adapters.append(cast("LifecycleAdapter", tracker_adapter))
            return adapters

        composition = compose_runtime(
            env=env,
            config=config,
            options=BuildDriverOptions(
                adapter_factory=_adapter_factory,
                materializers=materializers or None,
                enable_cache=self._options.enable_hamilton_cache,
                cache_dir=str(cache_dir),
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
            execution_env = context.env

            with telemetry_context(
                run_id=context.run_id,
                domain=context.domain,
                repo_commit=RepoCommitContext(
                    repo=context.env.repo,
                    commit=context.env.commit,
                ),
            ):
                inputs = ExecutionInputs(
                    env=execution_env,
                    catalog=context.runtime.catalog,
                    tag_query=context.runtime.tag_query,
                    cache_index=context.runtime.cache_index,
                    cache_key_resolver=context.runtime.cache_key_resolver,
                    schema_index=context.runtime.schema_index,
                    semantic_registry=context.runtime.semantic_registry,
                    runtime_fingerprint=context.runtime.fingerprint,
                )
                input_mapping = _execution_input_mapping(inputs)
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


def _persist_build_log(*, context: _RunState, writer: BuildRunWriter) -> None:
    drained = drain_build_log()
    if drained is None:
        return
    log_context, events = drained
    path = writer.write_build_log(context=log_context, events=events)
    if path is None:
        return
    log.info(
        "build.hamilton.executor.build_log_written run_id=%s event_count=%d path=%s",
        context.run_id,
        len(events),
        path,
    )


__all__ = [
    "HamiltonBuildExecutor",
    "HamiltonBuildResult",
]
