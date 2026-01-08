"""Composition root for runtime bundle construction."""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import local
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import hamilton.driver as h_driver
from hamilton import graph_types
from hamilton.caching.stores.file import FileResultStore
from hamilton.caching.stores.sqlite import SQLiteMetadataStore

from codeintel.build.contracts.policy_registry import configure_contract_policy_registry
from codeintel.build.contracts.runtime import configure_contract_runtime
from codeintel.build.hamilton.cache_adapter import (
    CacheAdapterOptions,
    CacheStore,
    ManifestBackedCacheAdapter,
)
from codeintel.build.hamilton.cache_key_resolver import CacheKeyResolver
from codeintel.build.hamilton.cache_policy import (
    cache_salt,
    default_cache_policy,
    is_salt_sensitive,
)
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.dag_catalog_compiler import compile_dag_catalog
from codeintel.build.hamilton.driver_options import BuildDriverOptions
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.graph_validation import (
    validate_graph,
    validation_result_to_json,
)
from codeintel.build.hamilton.nodes.support_spec import support_spec_from_catalog
from codeintel.build.schemas.contract_service import configure_contract_service
from codeintel.build.schemas.inference_service import (
    SeedDatasetConfig,
    get_schema_inference_service,
)
from codeintel.build.schemas.observation_provider import observation_provider_for_env
from codeintel.build.schemas.schema_index import SchemaIndex, build_schema_index
from codeintel.build.schemas.service import configure_schema_service
from codeintel.core.config.settings import HamiltonTrackerSettings
from codeintel.core.hamilton.tag_query import TagQuery
from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.runtime.loader import load_runtime_settings
from codeintel.core.schemas import (
    SchemaService,
    get_schema_service,
    set_schema_service,
    table_schema_from_json_obj,
)
from codeintel.core.schemas.declared import source_declared_schema_provider
from codeintel.core.schemas.provider import MappingSchemaProvider, SchemaProvider
from codeintel.core.schemas.table_registry import TABLE_SCHEMAS
from codeintel.runtime.module_resolver import ResolvedModuleSet, resolve_module_set
from codeintel.runtime.plugins.config import (
    PluginConfig,
    plugin_config_from_build_config,
    plugin_config_from_mapping,
)
from codeintel.runtime.plugins.spec import TargetPack
from codeintel.runtime.runtime_bundle import HamiltonRuntimeBundle, RuntimeKey
from codeintel.serving.semantic_compile import compile_semantic_registry_from_tag_query
from codeintel.storage.duckdb_types import DuckDBError
from codeintel.storage.gateway import open_inference_gateway

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import ModuleType
    from typing import Self

    from hamilton.caching.adapter import HamiltonCacheAdapter
    from hamilton.execution.executors import TaskExecutor
    from hamilton.io.materialization import ExtractorFactory, MaterializerFactory
    from hamilton.lifecycle.base import LifecycleAdapter

    from codeintel.core.schemas.primitives import TableSchema

log = logging.getLogger(__name__)

_DEFAULT_HAMILTON_CACHE_DIR = Path.cwd() / "build" / ".hamilton_cache"
_ALLOWED_GRAPH_VALIDATION_MODES: frozenset[str] = frozenset({"strict", "warn", "off"})
CacheBehavior = Literal["default", "recompute", "disable", "ignore"]

_ALLOWED_CACHE_BEHAVIORS: frozenset[CacheBehavior] = frozenset(
    {"default", "recompute", "disable", "ignore"}
)

_STATE = local()


class _SupportSpec(Protocol):
    def validate(self, *, catalog: DagCatalog | None = None) -> None: ...

    def to_hamilton_config(self) -> dict[str, object]: ...


class _CacheCodeVersionAdapter(Protocol):
    code_version: str


class _HamiltonTrackingConstants(Protocol):
    CAPTURE_DATA_STATISTICS: bool
    MAX_LIST_LENGTH_CAPTURE: int
    MAX_DICT_LENGTH_CAPTURE: int


class _DynamicExecutionBuilder(Protocol):
    def enable_dynamic_execution(self, *, allow_experimental_mode: bool = False) -> Self: ...

    def with_local_executor(self, local_executor: TaskExecutor) -> Self: ...

    def with_remote_executor(self, remote_executor: TaskExecutor) -> Self: ...


@dataclass(frozen=True, slots=True)
class RuntimeComposition:
    """Bundle returned by compose_runtime with the derived runtime key."""

    key: RuntimeKey
    bundle: HamiltonRuntimeBundle


@dataclass(frozen=True, slots=True)
class _RuntimeIdentity:
    env: BuildEnv
    config: Mapping[str, Any]
    module_paths: tuple[str, ...]
    modules_fingerprint: str


@dataclass(frozen=True, slots=True)
class _DriverResourceInputs:
    options: BuildDriverOptions
    config: Mapping[str, Any]
    base_catalog: DagCatalog
    modules_fingerprint: str
    cache_profile: str | None
    tracker: object | None


@dataclass(frozen=True, slots=True)
class _DynamicExecutionConfig:
    enabled: bool
    local_executor: object | None
    remote_executor: object | None


@dataclass(frozen=True, slots=True)
class _ResolvedRuntimeConfig:
    hamilton_config: dict[str, Any]
    plugin_config: PluginConfig
    dynamic_execution: _DynamicExecutionConfig


@dataclass(frozen=True, slots=True)
class _BundleSchemaRegistryEntry:
    table_key: str
    schema_digest: str
    derivation_kind: str | None


@contextmanager
def _composition_guard() -> Iterator[None]:
    if getattr(_STATE, "execution_active", False):
        msg = "compose_runtime cannot run during DAG execution"
        raise RuntimeError(msg)
    if getattr(_STATE, "composition_active", False):
        msg = "compose_runtime re-entry detected"
        raise RuntimeError(msg)
    _STATE.composition_active = True
    try:
        yield
    finally:
        _STATE.composition_active = False


def set_execution_active(*, active: bool) -> None:
    """Mark whether DAG execution is in progress for guard checks."""
    _STATE.execution_active = active


def _ensure_schema_service_for_module_imports(*, env: BuildEnv) -> None:
    try:
        schema_service = get_schema_service()
    except RuntimeError:
        provider = _override_schema_provider(env=env)
        schema_service = SchemaService(table_provider=provider)
        set_schema_service(schema_service)
    policy_registry = configure_contract_policy_registry(config=env.config)
    configure_contract_runtime(
        schema_service=schema_service,
        policy_registry=policy_registry,
    )
    schema_service = get_schema_service()
    required_keys = (
        "analytics.scip_diagnostics_summary",
        "analytics.scip_diagnostics_by_file",
        "analytics.scip_diagnostics_top_messages",
    )
    if any(schema_service.get_table_schema(key) is None for key in required_keys):
        provider = _override_schema_provider(env=env)
        set_schema_service(SchemaService(table_provider=provider))


def _resolve_runtime_config(
    *,
    env: BuildEnv,
    config: Mapping[str, Any] | None,
) -> _ResolvedRuntimeConfig:
    normalized = _normalize_config(config)
    normalized, plugin_overrides = _split_plugin_overrides(normalized)
    plugin_config = plugin_config_from_build_config(env.config)
    if plugin_overrides:
        plugin_config = _apply_plugin_overrides(plugin_config, plugin_overrides)
    hamilton_config = _merge_hamilton_config(normalized, plugin_config.hamilton_config)
    dynamic_execution = _parse_dynamic_execution_config(hamilton_config)
    return _ResolvedRuntimeConfig(
        hamilton_config=hamilton_config,
        plugin_config=plugin_config,
        dynamic_execution=dynamic_execution,
    )


def compose_runtime(
    *,
    env: BuildEnv,
    config: Mapping[str, Any] | None = None,
    options: BuildDriverOptions | None = None,
) -> RuntimeComposition:
    """Compose the runtime bundle for Hamilton execution.

    Returns
    -------
    RuntimeComposition
        Composed runtime bundle with driver, catalog, and settings.
    """
    with _composition_guard():
        resolved_options = options or BuildDriverOptions()
        resolved_config = _resolve_runtime_config(env=env, config=config)
        _ensure_schema_service_for_module_imports(env=env)
        resolved_modules = _resolve_modules_for_runtime(
            env=env,
            resolved_config=resolved_config,
        )
        identity = _RuntimeIdentity(
            env=env,
            config=resolved_config.hamilton_config,
            module_paths=resolved_modules.module_paths,
            modules_fingerprint=resolved_modules.fingerprint,
        )
        base_catalog = _build_base_catalog(
            config=identity.config,
            modules=resolved_modules.modules,
        )
        merged_config = _build_support_config(
            config=identity.config,
            base_catalog=base_catalog,
        )
        cache_profile = _cache_profile(merged_config)
        adapters, cache_adapter, cache_store = _resolve_driver_resources(
            inputs=_DriverResourceInputs(
                options=resolved_options,
                config=merged_config,
                base_catalog=base_catalog,
                modules_fingerprint=resolved_modules.fingerprint,
                cache_profile=cache_profile,
                tracker=_build_tracker_adapter(
                    env=env,
                    modules_fingerprint=resolved_modules.fingerprint,
                ),
            )
        )
        driver = _build_driver_with_adapters(
            config=merged_config,
            modules=resolved_modules.modules,
            adapters=adapters,
            materializers=resolved_options.materializers,
            dynamic_config=resolved_config.dynamic_execution,
        )
        runtime_bundle = _build_runtime_bundle(
            identity=identity,
            resolved_modules=resolved_modules,
            driver=driver,
            cache_adapter=cache_adapter,
            cache_store=cache_store,
        )
        observation_provider = observation_provider_for_env(env)
        schema_service = configure_schema_service(
            runtime=runtime_bundle,
            observation_provider=observation_provider,
        )
        configure_contract_service(runtime=runtime_bundle)
        configure_contract_runtime(
            schema_service=schema_service,
            policy_registry=configure_contract_policy_registry(config=env.config),
        )
        _validate_graph_invariants(
            runtime=runtime_bundle,
            mode=_graph_validation_mode(identity.config),
        )
        runtime_key = _runtime_key(
            env=identity.env,
            config=identity.config,
            modules_fingerprint=identity.modules_fingerprint,
        )

        log.info(
            "runtime.compose completed modules=%d targets=%d",
            len(identity.module_paths),
            len(runtime_bundle.catalog.targets),
        )
        return RuntimeComposition(key=runtime_key, bundle=runtime_bundle)


def _resolve_modules_for_runtime(
    *,
    env: BuildEnv,
    resolved_config: _ResolvedRuntimeConfig,
) -> ResolvedModuleSet:
    include_planning_nodes = _planning_enabled(resolved_config.hamilton_config)
    resolved = resolve_module_set(
        env=env,
        plugin_config=resolved_config.plugin_config,
        hamilton_config=resolved_config.hamilton_config,
        include_planning=include_planning_nodes,
    )
    _enforce_pack_namespaces(resolved.packs, resolved_config.plugin_config)
    return resolved


def _split_plugin_overrides(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    overrides: dict[str, Any] = {}
    prefix = "ci.plugins."
    for key in list(config):
        if key.startswith(prefix):
            overrides[key[len(prefix) :]] = config.pop(key)
    return config, overrides


def _apply_plugin_overrides(
    base: PluginConfig,
    overrides: Mapping[str, Any],
) -> PluginConfig:
    if not overrides:
        return base
    base_data = base.as_dict()
    known_keys = {
        "enabled",
        "disabled",
        "strict",
        "namespace_enforcement",
        "allow_workspace_modules",
        "hamilton_config",
    }
    for key in sorted(set(overrides) - known_keys):
        log.warning("plugins.override.unknown key=%s", key)
    if "hamilton_config" in overrides:
        override_value = overrides["hamilton_config"]
        if not isinstance(override_value, Mapping):
            msg = "plugins.hamilton_config override must be a mapping"
            raise TypeError(msg)
        merged = {**base.hamilton_config, **dict(override_value)}
        base_data["hamilton_config"] = merged
    for key in known_keys - {"hamilton_config"}:
        if key in overrides:
            base_data[key] = overrides[key]
    return plugin_config_from_mapping(base_data)


def _merge_hamilton_config(
    base_config: Mapping[str, Any],
    plugin_config: Mapping[str, object],
) -> dict[str, Any]:
    merged = dict(base_config)
    merged.update(plugin_config)
    return merged


def _parse_dynamic_execution_config(config: Mapping[str, Any]) -> _DynamicExecutionConfig:
    enabled = False
    local_executor: object | None = None
    remote_executor: object | None = None
    dynamic = config.get("ci.dynamic_execution")
    if isinstance(dynamic, bool):
        enabled = dynamic
    elif isinstance(dynamic, Mapping):
        enabled = bool(dynamic.get("enabled", False))
        local_executor = dynamic.get("local_executor")
        remote_executor = dynamic.get("remote_executor")
    if "ci.dynamic_execution.local_executor" in config:
        local_executor = config.get("ci.dynamic_execution.local_executor")
    if "ci.dynamic_execution.remote_executor" in config:
        remote_executor = config.get("ci.dynamic_execution.remote_executor")
    return _DynamicExecutionConfig(
        enabled=enabled,
        local_executor=local_executor,
        remote_executor=remote_executor,
    )


def _apply_dynamic_execution[BuilderT: _DynamicExecutionBuilder](
    *,
    builder: BuilderT,
    config: _DynamicExecutionConfig,
) -> BuilderT:
    if not config.enabled:
        return builder
    try:
        builder = builder.enable_dynamic_execution(allow_experimental_mode=True)
    except AttributeError as exc:
        msg = "Dynamic execution requested but builder lacks enable_dynamic_execution"
        raise RuntimeError(msg) from exc
    if config.remote_executor is not None:
        try:
            remote_executor = cast("TaskExecutor", config.remote_executor)
            builder = builder.with_remote_executor(remote_executor)
        except AttributeError as exc:
            msg = "Dynamic execution requested but builder lacks with_remote_executor"
            raise RuntimeError(msg) from exc
    if config.local_executor is not None:
        try:
            local_executor = cast("TaskExecutor", config.local_executor)
            builder = builder.with_local_executor(local_executor)
        except AttributeError as exc:
            msg = "Dynamic execution requested but builder lacks with_local_executor"
            raise RuntimeError(msg) from exc
    return builder


def apply_dynamic_execution[BuilderT: _DynamicExecutionBuilder](
    *,
    builder: BuilderT,
    config: _DynamicExecutionConfig,
) -> BuilderT:
    """Apply dynamic execution configuration to a Hamilton builder.

    Returns
    -------
    BuilderT
        Builder instance with dynamic execution settings applied.
    """
    return _apply_dynamic_execution(builder=builder, config=config)


DynamicExecutionConfig = _DynamicExecutionConfig


def _cache_profile(config: Mapping[str, Any]) -> str | None:
    value = config.get("ci.cache.profile")
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _cache_options_from_profile(
    *,
    cache_profile: str | None,
    cache_store: CacheStore,
    config: Mapping[str, Any],
) -> CacheAdapterOptions:
    default_behavior: CacheBehavior = "disable"
    log_to_file = True
    if cache_profile:
        normalized = cache_profile.lower()
        if normalized in {"default", "optout"}:
            default_behavior = "default"
        elif normalized in {"audit", "jsonl"}:
            default_behavior = "disable"
            log_to_file = True
        elif normalized in {"off", "disable"}:
            default_behavior = "disable"
            log_to_file = False
        else:
            log.warning("Unknown cache profile %s; falling back to defaults", cache_profile)
    behavior_override = _parse_cache_behavior(config.get("ci.cache.default_behavior"))
    if behavior_override is not None:
        default_behavior = behavior_override
    log_override = config.get("ci.cache.log_to_file")
    if isinstance(log_override, bool):
        log_to_file = log_override
    return CacheAdapterOptions(
        cache_store=cache_store,
        default_behavior=default_behavior,
        default_loader_behavior="disable",
        default_saver_behavior="disable",
        log_to_file=log_to_file,
    )


def _has_tracker_adapter(adapters: Sequence[LifecycleAdapter]) -> bool:
    return any(adapter.__class__.__name__ == "HamiltonTracker" for adapter in adapters)


def _set_cache_code_version(
    cache_adapter: HamiltonCacheAdapter,
    modules_fingerprint: str,
) -> None:
    if not modules_fingerprint:
        return
    if not hasattr(cache_adapter, "code_version"):
        return
    adapter = cast("_CacheCodeVersionAdapter", cache_adapter)
    try:
        adapter.code_version = modules_fingerprint
    except (AttributeError, TypeError) as exc:
        log.warning("cache.code_version_set_failed: %s", exc)


def _enforce_pack_namespaces(
    packs: Sequence[TargetPack],
    plugin_config: PluginConfig,
) -> None:
    if not plugin_config.namespace_enforcement:
        return
    namespaces = [pack.config_namespace for pack in packs if pack.config_namespace]
    if not namespaces:
        return
    duplicates = _find_duplicates(namespaces)
    if duplicates:
        msg = f"Duplicate plugin config namespaces detected: {', '.join(sorted(duplicates))}"
        raise ValueError(msg)
    if not plugin_config.hamilton_config:
        return
    prefixes = tuple(sorted(namespaces))
    for key in plugin_config.hamilton_config:
        key_str = str(key)
        if _matches_namespace(key_str, prefixes):
            continue
        msg = f"Plugin config key {key_str!r} is not namespaced under {prefixes}"
        raise ValueError(msg)


def _matches_namespace(key: str, namespaces: Sequence[str]) -> bool:
    return any(key == namespace or key.startswith(f"{namespace}.") for namespace in namespaces)


def _find_duplicates(values: Sequence[str]) -> set[str]:
    seen: set[str] = set()
    dupes: set[str] = set()
    for value in values:
        if value in seen:
            dupes.add(value)
            continue
        seen.add(value)
    return dupes


def _build_tracker_adapter(
    *,
    env: BuildEnv,
    modules_fingerprint: str,
) -> object | None:
    tracker_settings: HamiltonTrackerSettings = (
        load_runtime_settings().observability.hamilton_tracker
    )
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
        env=env,
        settings=tracker_settings,
        modules_fingerprint=modules_fingerprint,
    )
    dag_name = tracker_settings.dag_name or env.snapshot.repo
    kwargs: dict[str, object] = {
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


def _coerce_project_id(value: str) -> int | str:
    if value.isdigit():
        return int(value)
    return value


def _apply_tracker_constants(settings: HamiltonTrackerSettings) -> None:
    if not isinstance(settings, HamiltonTrackerSettings):
        log.warning("Hamilton tracker settings unavailable: %s", type(settings))
        return
    try:
        tracking_constants = importlib.import_module("hamilton_sdk.tracking.constants")
    except ModuleNotFoundError as exc:
        log.warning("Hamilton tracker constants unavailable: %s", exc)
        return
    constants = cast("_HamiltonTrackingConstants", tracking_constants)
    if settings.capture_data_statistics is not None:
        try:
            constants.CAPTURE_DATA_STATISTICS = bool(settings.capture_data_statistics)
        except (AttributeError, TypeError) as exc:
            log.warning("tracker.constant_set_failed CAPTURE_DATA_STATISTICS: %s", exc)
    if settings.max_list_length is not None:
        try:
            constants.MAX_LIST_LENGTH_CAPTURE = settings.max_list_length
        except (AttributeError, TypeError) as exc:
            log.warning("tracker.constant_set_failed MAX_LIST_LENGTH_CAPTURE: %s", exc)
    if settings.max_dict_length is not None:
        try:
            constants.MAX_DICT_LENGTH_CAPTURE = settings.max_dict_length
        except (AttributeError, TypeError) as exc:
            log.warning("tracker.constant_set_failed MAX_DICT_LENGTH_CAPTURE: %s", exc)


def _build_tracker_tags(
    *,
    env: BuildEnv,
    settings: HamiltonTrackerSettings,
    modules_fingerprint: str,
) -> dict[str, str]:
    tags: dict[str, str] = dict(settings.tags)
    tags.setdefault("repo", env.snapshot.repo)
    tags.setdefault("commit", env.snapshot.commit)
    if modules_fingerprint:
        tags.setdefault("modules_fingerprint", modules_fingerprint)
    run_kind = env.execution_context.run.kind if env.execution_context else None
    if isinstance(run_kind, str) and run_kind:
        tags.setdefault("run_kind", run_kind)
    return tags


def _build_base_catalog(
    *,
    config: Mapping[str, Any],
    modules: Sequence[ModuleType],
) -> DagCatalog:
    base_driver = _build_driver(
        config=config,
        modules=modules,
    )
    return compile_dag_catalog(base_driver, strict=True)


def _build_support_config(
    *,
    config: Mapping[str, Any],
    base_catalog: DagCatalog,
) -> dict[str, Any]:
    support_spec = support_spec_from_catalog(base_catalog)
    support_spec.validate(catalog=base_catalog)
    return _merge_support_config(
        config=config,
        support_spec=support_spec,
    )


def _resolve_driver_resources(
    *,
    inputs: _DriverResourceInputs,
) -> tuple[list[LifecycleAdapter], HamiltonCacheAdapter | None, CacheStore | None]:
    adapter_list = list(inputs.options.adapters) if inputs.options.adapters else []
    if inputs.options.adapter_factory is not None:
        adapter_list.extend(inputs.options.adapter_factory(inputs.base_catalog))

    cache_adapter = inputs.options.cache_adapter
    cache_store = _cache_store_from_adapter(cache_adapter)
    if cache_adapter is not None:
        _set_cache_code_version(cache_adapter, inputs.modules_fingerprint)
    if _has_tracker_adapter(adapter_list) is False and inputs.tracker is not None:
        adapter_list.append(cast("LifecycleAdapter", inputs.tracker))
    enable_cache = inputs.options.enable_cache or inputs.cache_profile is not None
    if enable_cache and cache_adapter is None:
        cache_dir = _cache_dir(inputs.options.cache_dir)
        cache_store = _cache_store_from_path(cache_dir)
        cache_adapter = ManifestBackedCacheAdapter(
            path=cache_dir,
            options=_cache_options_from_profile(
                cache_profile=inputs.cache_profile,
                cache_store=cache_store,
                config=inputs.config,
            ),
        )
        _set_cache_code_version(cache_adapter, inputs.modules_fingerprint)
    if cache_adapter is not None:
        adapter_list.append(cache_adapter)

    return adapter_list, cache_adapter, cache_store


def _build_driver_with_adapters(
    *,
    config: Mapping[str, Any],
    modules: Sequence[ModuleType],
    adapters: Sequence[LifecycleAdapter],
    materializers: Sequence[ExtractorFactory | MaterializerFactory] | None,
    dynamic_config: _DynamicExecutionConfig,
) -> h_driver.Driver:
    builder = h_driver.Builder().with_config(dict(config)).with_modules(*modules)
    allow_overrides = _allow_module_overrides(config)
    builder = _apply_module_overrides(builder=builder, allow_overrides=allow_overrides)
    if materializers:
        builder = builder.with_materializers(*materializers)
    builder = _apply_dynamic_execution(builder=builder, config=dynamic_config)
    return _build_or_raise(
        builder=builder.with_adapters(*adapters),
        allow_overrides=allow_overrides,
    )


def _build_runtime_bundle(
    *,
    identity: _RuntimeIdentity,
    resolved_modules: ResolvedModuleSet,
    driver: h_driver.Driver,
    cache_adapter: HamiltonCacheAdapter | None,
    cache_store: CacheStore | None,
) -> HamiltonRuntimeBundle:
    catalog = compile_dag_catalog(driver, strict=True)
    tag_query = TagQuery(driver)
    cache_index = cache_store
    cache_key_resolver = None
    runtime_fingerprint = _runtime_fingerprint(
        env=identity.env,
        config=identity.config,
        modules_fingerprint=identity.modules_fingerprint,
    )
    if cache_store is not None:
        cache_key_resolver = _build_cache_key_resolver(
            driver=driver,
            cache_store=cache_store,
            modules_fingerprint=identity.modules_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
        )

    schema_index = _build_schema_index(driver=driver, catalog=catalog, env=identity.env)
    semantic_registry = compile_semantic_registry_from_tag_query(
        schema_provider=schema_index.schema_provider(allow_inference=False),
        tag_query=tag_query,
        version="v1",
    )
    created_at_utc = datetime.now(tz=UTC).isoformat()
    return HamiltonRuntimeBundle(
        driver=driver,
        catalog=catalog,
        tag_query=tag_query,
        variants=identity.env.variants,
        cache_adapter=cache_adapter,
        cache_index=cache_index,
        cache_key_resolver=cache_key_resolver,
        schema_index=schema_index,
        semantic_registry=semantic_registry,
        packs=resolved_modules.packs,
        module_provenance=resolved_modules.provenance,
        modules_fingerprint=identity.modules_fingerprint,
        fingerprint=runtime_fingerprint,
        created_at_utc=created_at_utc,
    )


def _normalize_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = dict(config or {})
    normalized.setdefault("hamilton.enable_power_user_mode", True)
    return normalized


def _planning_enabled(config: Mapping[str, Any]) -> bool:
    value = config.get("ci.enable_planning_nodes")
    if isinstance(value, bool):
        return value
    return True


def _merge_support_config(
    *,
    config: Mapping[str, Any],
    support_spec: _SupportSpec,
) -> dict[str, Any]:
    merged = dict(config)
    support_config = support_spec.to_hamilton_config()
    for key, value in support_config.items():
        if key in merged and key.startswith("ci_support_include_"):
            continue
        merged[key] = value
    return merged


def _build_driver(
    *,
    config: Mapping[str, Any],
    modules: Sequence[ModuleType],
) -> h_driver.Driver:
    builder = h_driver.Builder().with_config(dict(config)).with_modules(*modules)
    allow_overrides = _allow_module_overrides(config)
    builder = _apply_module_overrides(builder=builder, allow_overrides=allow_overrides)
    return _build_or_raise(builder=builder, allow_overrides=allow_overrides)


def _cache_dir(path: str | Path | None) -> Path:
    if path is None:
        return _DEFAULT_HAMILTON_CACHE_DIR
    return Path(path)


def _cache_store_from_path(path: Path) -> CacheStore:
    metadata_store = SQLiteMetadataStore(path=str(path))
    result_store = FileResultStore(path=str(path))
    return CacheStore(metadata_store=metadata_store, result_store=result_store)


def _cache_store_from_adapter(
    cache_adapter: HamiltonCacheAdapter | None,
) -> CacheStore | None:
    if isinstance(cache_adapter, ManifestBackedCacheAdapter):
        return cache_adapter.cache_store
    return None


def _allow_module_overrides(config: Mapping[str, Any]) -> bool:
    value = config.get("ci.allow_module_overrides")
    if isinstance(value, bool):
        return value
    return False


def _apply_module_overrides(
    *,
    builder: h_driver.Builder,
    allow_overrides: bool,
) -> h_driver.Builder:
    if allow_overrides:
        return builder.allow_module_overrides()
    return builder


def _build_or_raise(
    *,
    builder: h_driver.Builder,
    allow_overrides: bool,
) -> h_driver.Driver:
    try:
        return builder.build()
    except Exception as exc:
        if allow_overrides:
            raise
        msg = (
            "Hamilton driver build failed. If this is due to duplicate node names, "
            "set ci.allow_module_overrides=true to enable overrides."
        )
        raise RuntimeError(msg) from exc


def _graph_validation_mode(config: Mapping[str, Any]) -> str:
    value = config.get("ci.graph_validation")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _ALLOWED_GRAPH_VALIDATION_MODES:
            return normalized
    if value is not None:
        log.warning("Unknown graph validation mode %r; defaulting to strict", value)
    return "strict"


def _validate_graph_invariants(*, runtime: HamiltonRuntimeBundle, mode: str) -> None:
    if mode == "off":
        return
    result = validate_graph(runtime=runtime, validate_schema=True)
    if not result.errors and not result.warnings:
        return
    payload = validation_result_to_json(result)
    if result.errors and mode == "strict":
        raise RuntimeError(payload)
    log.warning("graph.validation.warn %s", payload)


def _parse_cache_behavior(value: object) -> CacheBehavior | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in _ALLOWED_CACHE_BEHAVIORS:
        return cast("CacheBehavior", normalized)
    log.warning("Unknown cache behavior override %r; ignoring", value)
    return None


def _build_cache_key_resolver(
    *,
    driver: h_driver.Driver,
    cache_store: CacheStore,
    modules_fingerprint: str,
    runtime_fingerprint: str,
) -> CacheKeyResolver:
    code_versions: dict[str, str] = {}
    node_dependencies: dict[str, tuple[str, ...]] = {}
    policy = default_cache_policy()
    salted_nodes: set[str] = set()

    for name, node in driver.graph.nodes.items():
        h_node = graph_types.HamiltonNode.from_node(node)
        version_prefix = f"{modules_fingerprint}:" if modules_fingerprint else ""
        if h_node.is_external_input:
            code_versions[name] = f"{version_prefix}input__{name}"
        elif h_node.version is not None:
            code_versions[name] = f"{version_prefix}{h_node.version}"
        node_dependencies[name] = tuple(dep.name for dep in node.dependencies)
        if is_salt_sensitive(node, policy):
            salted_nodes.add(name)

    return CacheKeyResolver(
        code_versions=code_versions,
        node_dependencies=node_dependencies,
        cache_store=cache_store,
        cache_salt=cache_salt(runtime_fingerprint) if salted_nodes else None,
        salted_nodes=frozenset(salted_nodes),
    )


def _build_schema_index(
    *,
    driver: h_driver.Driver,
    catalog: DagCatalog,
    env: BuildEnv,
) -> SchemaIndex:
    inference_service = get_schema_inference_service(
        driver=driver,
        catalog=catalog,
        env=env,
        seed_dataset=_seed_dataset_config(env),
        gateway_factory=open_inference_gateway,
    )
    declared_provider = source_declared_schema_provider(
        exclude_table_keys=catalog.table_outputs,
    )
    override_provider = _override_schema_provider(env=env)
    _ensure_schema_service_for_inference(provider=override_provider)
    schema_index = build_schema_index(
        system=catalog,
        declared_provider=declared_provider,
        override_provider=override_provider,
        inference_service=inference_service,
        env_provider=lambda: env,
    )
    _prefill_schema_index(env=env, schema_index=schema_index)
    _require_schema_authority(schema_index=schema_index, catalog=catalog)
    return schema_index


def _seed_dataset_config(env: BuildEnv) -> SeedDatasetConfig:
    snapshot_id = env.snapshot.commit.strip()
    return SeedDatasetConfig(
        dataset_root_dir=env.paths.dataset_root_dir,
        snapshot_id=snapshot_id or None,
    )


def _ensure_schema_service_for_inference(*, provider: SchemaProvider) -> None:
    try:
        get_schema_service()
    except RuntimeError:
        set_schema_service(SchemaService(table_provider=provider))


def _bundle_root_for_schema_cache(env: BuildEnv) -> Path | None:
    if env.metadata_bundle is not None:
        bundle_root = env.metadata_bundle.bundle_root
    else:
        bundle_root = env.paths.build_dir / "metadata"
    registry_path = bundle_root / "schema" / "schema_registry.json"
    versions_path = bundle_root / "schema" / "schema_versions.jsonl"
    if registry_path.is_file() and versions_path.is_file():
        return bundle_root
    return None


def _optional_str(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _load_bundle_schema_versions(bundle_root: Path) -> dict[str, TableSchema]:
    path = bundle_root / "schema" / "schema_versions.jsonl"
    if not path.is_file():
        return {}
    versions: dict[str, TableSchema] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, Mapping):
                continue
            schema_digest = _optional_str(payload.get("schema_digest"))
            schema_json = payload.get("schema_json")
            if not schema_digest or not isinstance(schema_json, Mapping):
                continue
            try:
                table_schema = table_schema_from_json_obj(schema_json)
            except (KeyError, TypeError, ValueError) as exc:
                log.warning(
                    "schema.bundle.version_parse_failed digest=%s error=%s",
                    schema_digest,
                    exc,
                )
                continue
            versions.setdefault(schema_digest, table_schema)
    return versions


def _load_bundle_schema_registry_entries(
    bundle_root: Path,
) -> list[_BundleSchemaRegistryEntry]:
    path = bundle_root / "schema" / "schema_registry.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "schema.bundle.registry_load_failed path=%s error=%s",
            path,
            exc,
        )
        return []
    if not isinstance(payload, Mapping):
        return []
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return []
    resolved: list[_BundleSchemaRegistryEntry] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        table_key = _optional_str(entry.get("table_key"))
        schema_digest = _optional_str(entry.get("schema_digest"))
        if not table_key or not schema_digest:
            continue
        derivation_kind = _optional_str(entry.get("derivation_kind"))
        resolved.append(
            _BundleSchemaRegistryEntry(
                table_key=table_key,
                schema_digest=schema_digest,
                derivation_kind=derivation_kind,
            )
        )
    return resolved


def _bundle_schema_cache(
    bundle_root: Path,
) -> tuple[dict[str, TableSchema], dict[str, TableSchema]]:
    versions = _load_bundle_schema_versions(bundle_root)
    if not versions:
        return {}, {}
    entries = _load_bundle_schema_registry_entries(bundle_root)
    if not entries:
        return {}, {}
    prefetched: dict[str, TableSchema] = {}
    overrides: dict[str, TableSchema] = {}
    for entry in entries:
        table_schema = versions.get(entry.schema_digest)
        if table_schema is None:
            continue
        prefetched[entry.table_key] = table_schema
        if entry.derivation_kind == "explicit_override":
            overrides[entry.table_key] = table_schema
    return prefetched, overrides


def _override_schema_provider(*, env: BuildEnv) -> SchemaProvider:
    override_schemas = dict(TABLE_SCHEMAS)
    bundle_root = _bundle_root_for_schema_cache(env)
    if bundle_root is not None:
        _, bundle_overrides = _bundle_schema_cache(bundle_root)
        for table_key, table_schema in bundle_overrides.items():
            override_schemas.setdefault(table_key, table_schema)
        return MappingSchemaProvider(override_schemas)
    if env.gateway is not None:
        try:
            override_schemas.update(env.gateway.schemas.load_override_registry())
        except (DuckDBError, RuntimeError, TypeError, ValueError) as exc:
            log.warning("schema.override_registry.load failed: %s", exc)
    return MappingSchemaProvider(override_schemas)


def _prefill_schema_index(*, env: BuildEnv, schema_index: SchemaIndex) -> None:
    bundle_root = _bundle_root_for_schema_cache(env)
    if bundle_root is not None:
        prefetched, _ = _bundle_schema_cache(bundle_root)
        if prefetched:
            schema_index.prefill_cache(prefetched)
            log.info("schema.index.prefill bundle_loaded=%d", len(prefetched))
            return
    if env.gateway is None:
        return
    try:
        prefetched_gateway = env.gateway.schemas.prefill_schema_index(schema_index)
    except (DuckDBError, RuntimeError, TypeError, ValueError) as exc:
        log.warning("schema.index.prefill failed: %s", exc)
        return
    if prefetched_gateway:
        log.info("schema.index.prefill loaded %d schemas", prefetched_gateway)


def _require_schema_authority(*, schema_index: SchemaIndex, catalog: DagCatalog) -> None:
    missing: list[str] = []
    details: list[str] = []
    for table_key in sorted(catalog.table_outputs):
        schema = schema_index.get_table_schema(
            table_key,
            allow_inference=True,
            perform_inference=True,
        )
        if schema is not None:
            continue
        missing.append(table_key)
        error = schema_index.get_inference_error(table_key)
        if error:
            details.append(f"{table_key}: {error}")
        else:
            details.append(table_key)
    if not missing:
        return
    detail_lines = "\n".join(f"- {line}" for line in details)
    msg = "Missing TableSchema definitions for DAG outputs:\n" + detail_lines
    raise ValueError(msg)


def _runtime_key(
    *,
    env: BuildEnv,
    config: Mapping[str, Any],
    modules_fingerprint: str,
) -> RuntimeKey:
    repo_fingerprint = fingerprint(
        {
            "repo": env.snapshot.repo,
            "commit": env.snapshot.commit,
        }
    )
    return RuntimeKey(
        repo_fingerprint=repo_fingerprint,
        config_fingerprint=fingerprint(config),
        modules_fingerprint=modules_fingerprint,
        build_profile=config.get("profile") if isinstance(config.get("profile"), str) else None,
    )


def _runtime_fingerprint(
    *,
    env: BuildEnv,
    config: Mapping[str, Any],
    modules_fingerprint: str,
) -> str:
    return fingerprint(
        {
            "repo": env.snapshot.repo,
            "commit": env.snapshot.commit,
            "config": config,
            "modules": modules_fingerprint,
            "variant_fingerprint": env.variants.variant_fingerprint,
        }
    )


__all__ = [
    "RuntimeComposition",
    "compose_runtime",
    "set_execution_active",
]
