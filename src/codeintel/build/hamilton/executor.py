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
import logging
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

import hamilton.base as h_base
from hamilton.caching.adapter import HamiltonCacheAdapter
from opentelemetry import trace as otel_trace

from codeintel.build.execution_policy import effective_max_workers_for_graph
from codeintel.build.hamilton.adapters.parallel import create_parallel_adapter
from codeintel.build.hamilton.cache_adapter import ManifestBackedCacheAdapter
from codeintel.build.hamilton.cache_key_resolver import CacheKeyResolver
from codeintel.build.hamilton.contracts.enforced_gateway import ContractEnforcingStorageGateway
from codeintel.build.hamilton.decision_trace import (
    DECISION_TRACE_ARTIFACT_NAME,
    DECISION_TRACE_PATH_TEMPLATE,
)
from codeintel.build.hamilton.driver_factory import (
    BuildDriverOptions,
    build_driver,
    target_to_node_name,
)
from codeintel.build.hamilton.execution_options import BuildExecutionOptions
from codeintel.build.hamilton.hooks import (
    ContractEnforcementHook,
    NodeTelemetryHook,
    ValidationSummary,
    build_hooks,
)
from codeintel.build.hamilton.optional_inputs import optional_inputs_for_target
from codeintel.build.hamilton.run_records import (
    NativeRunInfo,
    RunRecordInputs,
    TargetRunRecord,
    create_run_record,
)
from codeintel.build.hamilton.run_writer import BuildRunWriter
from codeintel.build.manifest.writer import CacheManifestWriter
from codeintel.build.schemas import get_schema_provider
from codeintel.build.schemas.compile import SchemaManifestRequest, compile_schema_manifest
from codeintel.core.execution.ids import new_run_id
from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.runtime.loader import load_runtime_settings
from codeintel.observability.telemetry_context import (
    RepoCommitContext,
    telemetry_context,
)
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.metadata.catalogs import load_latest_canonical_catalog_from_connection
from codeintel.storage.tracking.schema_catalog import SchemaCatalogRequest
from codeintel.storage.tracking.schema_catalog_models import DEFAULT_SCHEMA_MANIFEST_KIND

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hamilton.lifecycle.base import LifecycleAdapter

    from codeintel.build.hamilton.dag_catalog import DagCatalog
    from codeintel.build.hamilton.driver_factory import HamiltonRuntime
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.core.config.settings import HamiltonTrackerSettings
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RunState:
    """Execution state shared across run steps."""

    env: BuildEnv
    targets: tuple[str, ...]
    runtime: HamiltonRuntime
    run_id: str
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


def _coerce_project_id(value: str) -> int | str:
    if value.isdigit():
        return int(value)
    return value


def _current_trace_id() -> str | None:
    span = otel_trace.get_current_span()
    context = span.get_span_context()
    if not context.is_valid:
        return None
    if context.trace_id == 0:
        return None
    return f"{context.trace_id:032x}"


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


def _build_tracker_tags(
    *,
    settings: HamiltonTrackerSettings,
    env: BuildEnv,
    run_id: str,
    domain: str | None,
    deployment_environment: str | None,
) -> dict[str, str]:
    tags = dict(settings.tags)
    if deployment_environment and "environment" not in tags:
        tags["environment"] = deployment_environment
    tags.setdefault("repo", env.snapshot.repo)
    tags.setdefault("commit", env.snapshot.commit)
    tags.setdefault("run_id", run_id)
    if domain and "domain" not in tags:
        tags["domain"] = domain
    tags.setdefault("build.decision_trace_artifact", DECISION_TRACE_ARTIFACT_NAME)
    tags.setdefault(
        "build.decision_trace_path",
        DECISION_TRACE_PATH_TEMPLATE.format(build_dir=env.paths.build_dir.name),
    )
    return tags


def _build_hamilton_tracker_adapter(
    *,
    env: BuildEnv,
    run_id: str,
    domain: str | None,
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
    tags = _build_tracker_tags(
        settings=tracker_settings,
        env=env,
        run_id=run_id,
        domain=domain,
        deployment_environment=runtime_settings.deployment_environment,
    )
    trace_id = _current_trace_id()
    if trace_id and "trace_id" not in tags:
        tags["trace_id"] = trace_id
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
    runtime: HamiltonRuntime,
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
        node_name = target_to_node_name(target, runtime=runtime)
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
        inputs=RunRecordInputs(env=env, run=run),
    )
    if reason:
        return replace(record, error=reason)
    return record


def _failure_record(
    *,
    target_name: str,
    env: BuildEnv,
    catalog: DagCatalog,
    error: str,
) -> TargetRunRecord:
    exc = RuntimeError(error)
    try:
        target = catalog.get(target_name)
    except KeyError:
        return TargetRunRecord(
            target=target_name,
            impl_kind="native",
            status="failed",
            input_hash=None,
            error=str(exc),
        )
    input_hash = _safe_input_hash(target, env)
    return create_run_record(
        target,
        "failed",
        input_hash,
        inputs=RunRecordInputs(error=exc),
    )


def _safe_input_hash(target: TargetDescriptor, env: BuildEnv) -> str | None:
    manifest_index = env.manifest_index
    if manifest_index is None:
        return None
    manifest = manifest_index.get(target.name)
    if manifest is None:
        return None
    return manifest.input_hash


def _ensure_failure_records(
    *,
    env: BuildEnv,
    runtime: HamiltonRuntime,
    closure: tuple[str, ...],
    outputs: dict[str, Any],
    error: str,
) -> None:
    for target in closure:
        node_name = target_to_node_name(target, runtime=runtime)
        existing = outputs.get(node_name) if node_name is not None else None
        if isinstance(existing, TargetRunRecord):
            continue
        record = _failure_record(
            target_name=target,
            env=env,
            catalog=runtime.catalog,
            error=error,
        )
        key = node_name or f"__failed__{target}"
        outputs[key] = record


def _apply_cache_keys(
    *,
    outputs: dict[str, Any],
    runtime: HamiltonRuntime,
) -> None:
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
    runtime: HamiltonRuntime,
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
        node_name = target_to_node_name(target, runtime=runtime)
        if node_name is None:
            missing.append(target)
        else:
            final_vars.append(node_name)

    return final_vars, missing


def _table_key_exists(env: BuildEnv, table_key: str) -> bool:
    schema, table = split_table_key(table_key)
    return env.gateway.policy.table_exists(schema=schema, table=table)


def _maybe_persist_schema_manifest(
    *,
    env: BuildEnv,
    run_id: str,
    domain: str | None,
) -> None:
    """Persist the latest schema manifest before executing targets.

    Raises
    ------
    ValueError
        If repo or commit metadata is missing.
    """
    if env.gateway.config.read_only:
        return
    if not env.repo or not env.commit:
        msg = "Schema manifest persistence requires repo and commit metadata"
        raise ValueError(msg)

    request = SchemaManifestRequest(
        all_targets=True,
        stable=True,
        version="v2",
        include_views=True,
        include_artifacts=True,
        include_provenance=True,
    )
    provider = get_schema_provider()
    manifest = compile_schema_manifest(
        provider=provider,
        request=request,
        con=env.gateway.con,
    )
    catalog_hash = fingerprint(manifest.to_json_obj())
    latest = load_latest_canonical_catalog_from_connection(
        env.gateway.con,
        catalog_kind=DEFAULT_SCHEMA_MANIFEST_KIND,
    )
    if latest is not None and latest.catalog_hash == catalog_hash:
        log.info("build.hamilton.executor.schema_manifest.skip hash=%s", catalog_hash)
        return

    catalog_inputs: dict[str, object] = {"source": "build.execute"}
    if domain is not None:
        catalog_inputs["domain"] = domain

    result = env.gateway.schemas.persist_schema_manifest(
        manifest,
        request=SchemaCatalogRequest(
            run_id=run_id,
            repo=env.repo,
            commit=env.commit,
            catalog_inputs=catalog_inputs,
        ),
    )
    log.info(
        "build.hamilton.executor.schema_manifest.persisted hash=%s tables=%d views=%d",
        result.catalog_hash,
        result.tables,
        result.views,
    )


def _preflight_missing_inputs(
    *,
    env: BuildEnv,
    runtime: HamiltonRuntime,
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
    runtime: HamiltonRuntime,
    closure: tuple[str, ...],
    missing_by_target: Mapping[str, _MissingInputs],
) -> tuple[dict[str, TargetRunRecord], set[str]]:
    catalog = runtime.catalog
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
                env=env,
                catalog=catalog,
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
        node_name = target_to_node_name(target, runtime=runtime)
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
        node_name = target_to_node_name(target, runtime=context.runtime)
        if node_name is None:
            continue
        adjusted.append(node_name)
    return adjusted, preflight_records


@dataclass(frozen=True)
class _FinalizeInputs:
    writer: BuildRunWriter
    contract_hook: ContractEnforcementHook | None
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

    inputs.writer.complete_run(
        run_id=context.run_id,
        success=success,
        computed_targets=computed,
        skipped_targets=skipped,
        error_summary=inputs.error or (f"{len(failed)} targets failed" if failed else None),
    )

    log.info(
        "build.hamilton.executor.complete run_id=%s success=%s duration_ms=%.1f",
        context.run_id,
        success,
        duration_ms,
    )

    validation_summary = (
        inputs.contract_hook.get_validation_summary() if inputs.contract_hook else None
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
        validation_summary=validation_summary,
    )


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
        Reference to the HamiltonRuntime for mapping lookups.
    validation_summary
        Optional validation summary produced by ContractEnforcementHook.
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
    runtime: HamiltonRuntime | None = None
    validation_summary: ValidationSummary | None = None

    def get_record(self, target_name: str) -> TargetRunRecord | None:
        """Get the execution record for a target.

        Returns
        -------
        TargetRunRecord | None
            Execution record for the target, if present.
        """
        node_name = target_to_node_name(target_name, runtime=self.runtime)
        if node_name is not None:
            value = self.outputs.get(node_name)
            if isinstance(value, TargetRunRecord):
                return value
        for value in self.outputs.values():
            if isinstance(value, TargetRunRecord) and value.target == target_name:
                return value
        return None


class HamiltonBuildExecutor:
    """Execute build targets using Hamilton Driver.

    Parameters
    ----------
    profile
        Optional policy profile name (e.g., "fast", "full", "ci").
    enable_cache
        When True, enable Hamilton's on-disk caching adapter for nodes decorated with
        ``@cache``.
    cache_dir
        Optional override for the Hamilton cache directory. When omitted, uses
        ``{build_dir}/.hamilton_cache`` from the provided BuildEnv.
    """

    def __init__(
        self,
        *,
        profile: str | None = None,
        parallel_backend: str = "sequential",
        max_workers: int | None = None,
        enable_cache: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize the Hamilton executor."""
        self._options = BuildExecutionOptions(
            profile=profile,
            parallel_backend=parallel_backend,
            max_workers=max_workers,
            enable_hamilton_cache=enable_cache,
            cache_dir=cache_dir,
        )

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
        run_id = _generate_run_id()
        writer = BuildRunWriter(env.gateway)
        runtime, telemetry_hook, contract_hook = self._build_runtime(
            env=env,
            run_id=run_id,
            writer=writer,
            domain=domain,
        )
        _maybe_persist_schema_manifest(env=env, run_id=run_id, domain=domain)

        context = _RunState(
            env=env,
            targets=tuple(targets),
            runtime=runtime,
            run_id=run_id,
            start_time=time.perf_counter(),
            started_at=datetime.now(tz=UTC),
            domain=domain,
        )
        return self._run_with_state(
            context=context,
            writer=writer,
            telemetry_hook=telemetry_hook,
            contract_hook=contract_hook,
        )

    def _run_with_state(
        self,
        *,
        context: _RunState,
        writer: BuildRunWriter,
        telemetry_hook: NodeTelemetryHook | None,
        contract_hook: ContractEnforcementHook | None,
    ) -> HamiltonBuildResult:
        catalog = context.runtime.catalog
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

        return _finalize_run(
            context=context,
            inputs=_FinalizeInputs(
                writer=writer,
                contract_hook=contract_hook,
                closure=closure,
                outputs=outputs,
                error=error,
            ),
        )

    def _effective_max_workers(self, catalog: DagCatalog) -> int | None:
        return effective_max_workers_for_graph(run_options=self._options, catalog=catalog)

    def _build_runtime(
        self,
        *,
        env: BuildEnv,
        run_id: str,
        writer: BuildRunWriter,
        domain: str | None,
    ) -> tuple[HamiltonRuntime, NodeTelemetryHook | None, ContractEnforcementHook | None]:
        """Build Hamilton runtime with configured mode and lifecycle adapters.

        Returns
        -------
        HamiltonRuntime
            Configured runtime with driver and catalog.
        """
        config: dict[str, Any] = {"profile": self._options.resolved_profile(env=env)}
        config.update(env.variants.as_hamilton_config())
        config["variant_fingerprint"] = env.variants.variant_fingerprint
        telemetry_hook: NodeTelemetryHook | None = None
        contract_hook: ContractEnforcementHook | None = None

        hook_options = self._options.hook_options(env=env)
        cache_adapter: ManifestBackedCacheAdapter | None = None
        cache_dir = self._options.resolved_cache_dir(env=env)
        if self._options.enable_hamilton_cache:
            cache_adapter = ManifestBackedCacheAdapter(
                path=cache_dir,
                manifest_writer=CacheManifestWriter(env.gateway),
                manifest_run_id=run_id,
                default_behavior="default",
                default_loader_behavior="disable",
                default_saver_behavior="disable",
            )

        def _adapter_factory(catalog: DagCatalog) -> list[LifecycleAdapter]:
            nonlocal telemetry_hook
            nonlocal contract_hook
            adapters: list[LifecycleAdapter] = []
            effective_max_workers = self._effective_max_workers(catalog)
            parallel_adapter = create_parallel_adapter(
                self._options.parallel_backend,
                max_workers=effective_max_workers,
                thread_name_prefix="codeintel-build",
            )
            if parallel_adapter is not None:
                adapters.append(parallel_adapter)
            else:
                adapters.append(h_base.DictResult())

            hooks = build_hooks(run_id, writer, catalog, options=hook_options)
            for hook in hooks:
                if isinstance(hook, NodeTelemetryHook):
                    telemetry_hook = hook
                if isinstance(hook, ContractEnforcementHook):
                    contract_hook = hook
                adapters.append(cast("LifecycleAdapter", hook))

            tracker_adapter = _build_hamilton_tracker_adapter(
                env=env,
                run_id=run_id,
                domain=domain,
            )
            if tracker_adapter is not None:
                adapters.append(cast("LifecycleAdapter", tracker_adapter))
            return adapters

        runtime = build_driver(
            config=config,
            options=BuildDriverOptions(
                adapter_factory=_adapter_factory,
                enable_cache=self._options.enable_hamilton_cache,
                cache_dir=str(cache_dir),
                cache_adapter=cache_adapter,
            ),
        )
        return runtime, telemetry_hook, contract_hook

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
        for target in context.targets:
            record = _failure_record(
                target_name=target,
                env=context.env,
                catalog=context.runtime.catalog,
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
        for target in missing:
            record = _failure_record(
                target_name=target,
                env=context.env,
                catalog=context.runtime.catalog,
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
            if context.env.strict_contracts:
                wrapped_gateway = ContractEnforcingStorageGateway(context.env.gateway)
                execution_env = replace(
                    context.env,
                    gateway=cast("StorageGateway", wrapped_gateway),
                )

            with telemetry_context(
                run_id=context.run_id,
                domain=context.domain,
                repo_commit=RepoCommitContext(
                    repo=context.env.repo,
                    commit=context.env.commit,
                ),
            ):
                outputs = context.runtime.dr.execute(
                    list(final_vars),
                    inputs={
                        "env": execution_env,
                        "catalog": context.runtime.catalog,
                        "tag_query": context.runtime.tag_query,
                    },
                )
        except Exception as exc:
            log.exception("build.hamilton.executor.error run_id=%s", context.run_id)
            return {}, str(exc)
        else:
            return outputs, None


__all__ = [
    "HamiltonBuildExecutor",
    "HamiltonBuildResult",
]
