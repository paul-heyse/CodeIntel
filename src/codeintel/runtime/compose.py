"""Composition root for runtime bundle construction."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Collection, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import local
from typing import TYPE_CHECKING, Any, Protocol, cast

import hamilton.driver as h_driver
from hamilton import graph_types
from hamilton.caching.stores.file import FileResultStore
from hamilton.caching.stores.sqlite import SQLiteMetadataStore

from codeintel.build.hamilton.cache_adapter import (
    CacheAdapterOptions,
    CacheStore,
    ManifestBackedCacheAdapter,
)
from codeintel.build.hamilton.cache_key_resolver import CacheKeyResolver
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.dag_catalog_compiler import compile_dag_catalog
from codeintel.build.hamilton.driver_options import BuildDriverOptions
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.nodes.support_spec import support_spec_from_catalog
from codeintel.build.schemas.contract_service import configure_contract_service
from codeintel.build.schemas.inference_service import get_schema_inference_service
from codeintel.build.schemas.schema_index import SchemaIndex, build_schema_index
from codeintel.build.schemas.service import configure_schema_service
from codeintel.build.serving.semantic_compile import compile_semantic_registry_from_tag_query
from codeintel.core.config.settings import HamiltonTrackerSettings
from codeintel.core.hamilton.tag_query import TagQuery
from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.runtime.loader import load_runtime_settings
from codeintel.core.schemas.declared import source_declared_schema_provider
from codeintel.core.schemas.output_registry import non_inferable_output_schemas
from codeintel.core.schemas.provider import MappingSchemaProvider, SchemaProvider
from codeintel.runtime.module_resolver import ResolvedModuleSet, resolve_module_set
from codeintel.runtime.plugins.config import (
    PluginConfig,
    plugin_config_from_build_config,
    plugin_config_from_mapping,
)
from codeintel.runtime.plugins.spec import TargetPack
from codeintel.runtime.runtime_bundle import RuntimeBundle, RuntimeKey
from codeintel.storage.duckdb_types import DuckDBError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import ModuleType
    from typing import Self

    from hamilton.caching.adapter import HamiltonCacheAdapter
    from hamilton.execution.executors import TaskExecutor
    from hamilton.io.materialization import ExtractorFactory, MaterializerFactory
    from hamilton.lifecycle.base import LifecycleAdapter

log = logging.getLogger(__name__)

_DEFAULT_HAMILTON_CACHE_DIR = Path.cwd() / "build" / ".hamilton_cache"

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
    bundle: RuntimeBundle


@dataclass(frozen=True, slots=True)
class _RuntimeIdentity:
    env: BuildEnv
    config: Mapping[str, Any]
    module_paths: tuple[str, ...]
    modules_fingerprint: str


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
            options=resolved_options,
            base_catalog=base_catalog,
            modules_fingerprint=resolved_modules.fingerprint,
            cache_profile=cache_profile,
            tracker=_build_tracker_adapter(
                env=env,
                modules_fingerprint=resolved_modules.fingerprint,
            ),
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
        configure_schema_service(runtime=runtime_bundle)
        configure_contract_service(runtime=runtime_bundle)
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
    include_plan_materialization = _plan_materialization_enabled(resolved_config.hamilton_config)
    resolved = resolve_module_set(
        env=env,
        plugin_config=resolved_config.plugin_config,
        hamilton_config=resolved_config.hamilton_config,
        include_planning=include_planning_nodes,
        include_plan_materialization=include_plan_materialization,
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
    for key, value in plugin_config.items():
        if key.startswith("ci_support_include_") and key in merged:
            continue
        merged[key] = value
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
) -> CacheAdapterOptions:
    default_behavior = "default"
    log_to_file = False
    if cache_profile:
        normalized = cache_profile.lower()
        if normalized in {"audit", "jsonl"}:
            default_behavior = "disable"
            log_to_file = True
        elif normalized != "default":
            log.warning("Unknown cache profile %s; falling back to defaults", cache_profile)
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
    options: BuildDriverOptions | None,
    base_catalog: DagCatalog,
    modules_fingerprint: str,
    cache_profile: str | None,
    tracker: object | None,
) -> tuple[list[LifecycleAdapter], HamiltonCacheAdapter | None, CacheStore | None]:
    resolved_options = options or BuildDriverOptions()
    adapter_list = list(resolved_options.adapters) if resolved_options.adapters else []
    if resolved_options.adapter_factory is not None:
        adapter_list.extend(resolved_options.adapter_factory(base_catalog))

    cache_adapter = resolved_options.cache_adapter
    cache_store = _cache_store_from_adapter(cache_adapter)
    if cache_adapter is not None:
        _set_cache_code_version(cache_adapter, modules_fingerprint)
    if _has_tracker_adapter(adapter_list) is False and tracker is not None:
        adapter_list.append(cast("LifecycleAdapter", tracker))
    enable_cache = resolved_options.enable_cache or cache_profile is not None
    if enable_cache and cache_adapter is None:
        cache_dir = _cache_dir(resolved_options.cache_dir)
        cache_store = _cache_store_from_path(cache_dir)
        cache_adapter = ManifestBackedCacheAdapter(
            path=cache_dir,
            options=_cache_options_from_profile(
                cache_profile=cache_profile,
                cache_store=cache_store,
            ),
        )
        _set_cache_code_version(cache_adapter, modules_fingerprint)
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
    builder = (
        h_driver.Builder().with_config(dict(config)).with_modules(*modules).allow_module_overrides()
    )
    if materializers:
        builder = builder.with_materializers(*materializers)
    builder = _apply_dynamic_execution(builder=builder, config=dynamic_config)
    return builder.with_adapters(*adapters).build()


def _build_runtime_bundle(
    *,
    identity: _RuntimeIdentity,
    resolved_modules: ResolvedModuleSet,
    driver: h_driver.Driver,
    cache_adapter: HamiltonCacheAdapter | None,
    cache_store: CacheStore | None,
) -> RuntimeBundle:
    catalog = compile_dag_catalog(driver, strict=True)
    tag_query = TagQuery(driver)
    cache_index = cache_store
    cache_key_resolver = None
    if cache_store is not None:
        cache_key_resolver = _build_cache_key_resolver(
            driver=driver,
            cache_store=cache_store,
            modules_fingerprint=identity.modules_fingerprint,
        )

    schema_index = _build_schema_index(driver=driver, catalog=catalog, env=identity.env)
    semantic_registry = compile_semantic_registry_from_tag_query(
        schema_provider=schema_index.schema_provider(),
        tag_query=tag_query,
        version="v1",
    )
    created_at_utc = datetime.now(tz=UTC).isoformat()
    return RuntimeBundle(
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
        fingerprint=_runtime_fingerprint(
            env=identity.env,
            config=identity.config,
            modules_fingerprint=identity.modules_fingerprint,
        ),
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


def _plan_materialization_enabled(config: Mapping[str, Any]) -> bool:
    value = config.get("ci.plan_materialization")
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
    return (
        h_driver.Builder()
        .with_config(dict(config))
        .with_modules(*modules)
        .allow_module_overrides()
        .build()
    )


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


def _build_cache_key_resolver(
    *,
    driver: h_driver.Driver,
    cache_store: CacheStore,
    modules_fingerprint: str,
) -> CacheKeyResolver:
    code_versions: dict[str, str] = {}
    node_dependencies: dict[str, tuple[str, ...]] = {}

    for name, node in driver.graph.nodes.items():
        h_node = graph_types.HamiltonNode.from_node(node)
        version_prefix = f"{modules_fingerprint}:" if modules_fingerprint else ""
        if h_node.is_external_input:
            code_versions[name] = f"{version_prefix}input__{name}"
        elif h_node.version is not None:
            code_versions[name] = f"{version_prefix}{h_node.version}"
        node_dependencies[name] = tuple(dep.name for dep in node.dependencies)

    return CacheKeyResolver(
        code_versions=code_versions,
        node_dependencies=node_dependencies,
        cache_store=cache_store,
    )


def _build_schema_index(
    *,
    driver: h_driver.Driver,
    catalog: DagCatalog,
    env: BuildEnv,
) -> SchemaIndex:
    inference_service = get_schema_inference_service(driver=driver, catalog=catalog)
    inferable_table_keys = inference_service.inferable_table_keys()
    declared_provider = source_declared_schema_provider(
        exclude_table_keys=catalog.table_outputs,
    )
    override_provider = _override_schema_provider(
        env=env,
        inferable_table_keys=inferable_table_keys,
    )
    schema_index = build_schema_index(
        system=catalog,
        declared_provider=declared_provider,
        override_provider=override_provider,
        inference_service=inference_service,
    )
    _prefill_schema_index(env=env, schema_index=schema_index)
    return schema_index


def _override_schema_provider(
    *,
    env: BuildEnv,
    inferable_table_keys: Collection[str] | None = None,
) -> SchemaProvider:
    override_schemas = non_inferable_output_schemas(
        inferable_table_keys=inferable_table_keys,
    )
    try:
        override_schemas.update(env.gateway.schemas.load_override_registry())
    except (DuckDBError, RuntimeError, TypeError, ValueError) as exc:
        log.warning("schema.override_registry.load failed: %s", exc)
    return MappingSchemaProvider(override_schemas)


def _prefill_schema_index(*, env: BuildEnv, schema_index: SchemaIndex) -> None:
    try:
        prefetched = env.gateway.schemas.prefill_schema_index(schema_index)
    except (DuckDBError, RuntimeError, TypeError, ValueError) as exc:
        log.warning("schema.index.prefill failed: %s", exc)
        return
    if prefetched:
        log.info("schema.index.prefill loaded %d schemas", prefetched)


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
