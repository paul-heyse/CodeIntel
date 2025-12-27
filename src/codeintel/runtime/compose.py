"""Composition root for runtime bundle construction."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import local
from typing import TYPE_CHECKING, Any, Protocol

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
from codeintel.build.hamilton.nodes import support_nodes
from codeintel.build.hamilton.nodes.support_spec import support_spec_from_catalog
from codeintel.build.schemas.inference_service import get_schema_inference_service
from codeintel.build.schemas.schema_index import SchemaIndex, build_schema_index
from codeintel.build.serving.semantic_compile import compile_semantic_registry_from_tag_query
from codeintel.core.hamilton.tag_query import TagQuery
from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.schemas.declared import source_declared_schema_provider
from codeintel.runtime.module_resolver import resolve_module_paths, resolve_modules
from codeintel.runtime.runtime_bundle import RuntimeBundle, RuntimeKey

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from types import ModuleType

    from hamilton.caching.adapter import HamiltonCacheAdapter
    from hamilton.lifecycle.base import LifecycleAdapter

log = logging.getLogger(__name__)

_DEFAULT_HAMILTON_CACHE_DIR = Path.cwd() / "build" / ".hamilton_cache"

_STATE = local()


class _SupportSpec(Protocol):
    def validate(self, *, catalog: DagCatalog | None = None) -> None: ...

    def to_hamilton_config(self) -> dict[str, object]: ...


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
        normalized_config = _normalize_config(config)
        module_paths, modules = _resolve_modules_and_paths(normalized_config)
        identity = _RuntimeIdentity(
            env=env,
            config=normalized_config,
            module_paths=module_paths,
        )
        base_catalog = _build_base_catalog(
            config=identity.config,
            modules=modules,
        )
        merged_config = _build_support_config(
            config=identity.config,
            base_catalog=base_catalog,
        )
        adapters, cache_adapter, cache_store = _resolve_driver_resources(
            options=options,
            base_catalog=base_catalog,
        )
        driver = _build_driver_with_adapters(
            config=merged_config,
            modules=modules,
            adapters=adapters,
        )
        runtime_bundle = _build_runtime_bundle(
            identity=identity,
            driver=driver,
            cache_adapter=cache_adapter,
            cache_store=cache_store,
        )
        runtime_key = _runtime_key(
            env=identity.env,
            config=identity.config,
            module_paths=identity.module_paths,
        )

        log.info(
            "runtime.compose completed modules=%d targets=%d",
            len(module_paths),
            len(runtime_bundle.catalog.targets),
        )
        return RuntimeComposition(key=runtime_key, bundle=runtime_bundle)


def _resolve_modules_and_paths(
    config: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[ModuleType, ...]]:
    include_planning_nodes = _planning_enabled(config)
    module_paths = resolve_module_paths(include_planning=include_planning_nodes)
    modules = resolve_modules(include_planning=include_planning_nodes)
    return module_paths, modules


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
) -> tuple[list[LifecycleAdapter], HamiltonCacheAdapter | None, CacheStore | None]:
    resolved_options = options or BuildDriverOptions()
    adapter_list = list(resolved_options.adapters) if resolved_options.adapters else []
    if resolved_options.adapter_factory is not None:
        adapter_list.extend(resolved_options.adapter_factory(base_catalog))

    cache_adapter = resolved_options.cache_adapter
    cache_store = _cache_store_from_adapter(cache_adapter)
    if resolved_options.enable_cache and cache_adapter is None:
        cache_dir = _cache_dir(resolved_options.cache_dir)
        cache_store = _cache_store_from_path(cache_dir)
        cache_adapter = ManifestBackedCacheAdapter(
            path=cache_dir,
            options=CacheAdapterOptions(
                cache_store=cache_store,
                default_behavior="default",
                default_loader_behavior="disable",
                default_saver_behavior="disable",
            ),
        )
    if cache_adapter is not None:
        adapter_list.append(cache_adapter)

    return adapter_list, cache_adapter, cache_store


def _build_driver_with_adapters(
    *,
    config: Mapping[str, Any],
    modules: Sequence[ModuleType],
    adapters: Sequence[LifecycleAdapter],
) -> h_driver.Driver:
    builder = (
        h_driver.Builder()
        .with_config(dict(config))
        .with_modules(*modules, support_nodes)
        .allow_module_overrides()
    )
    return builder.with_adapters(*adapters).build()


def _build_runtime_bundle(
    *,
    identity: _RuntimeIdentity,
    driver: h_driver.Driver,
    cache_adapter: HamiltonCacheAdapter | None,
    cache_store: CacheStore | None,
) -> RuntimeBundle:
    catalog = compile_dag_catalog(driver, strict=True)
    tag_query = TagQuery(driver)
    cache_index = cache_store
    cache_key_resolver = (
        _build_cache_key_resolver(driver, cache_store)
        if cache_store is not None
        else None
    )

    schema_index = _build_schema_index(driver=driver, catalog=catalog)
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
        fingerprint=_runtime_fingerprint(
            env=identity.env,
            config=identity.config,
            module_paths=identity.module_paths,
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
    dr: h_driver.Driver,
    cache_store: CacheStore,
) -> CacheKeyResolver:
    code_versions: dict[str, str] = {}
    node_dependencies: dict[str, tuple[str, ...]] = {}

    for name, node in dr.graph.nodes.items():
        h_node = graph_types.HamiltonNode.from_node(node)
        if h_node.is_external_input:
            code_versions[name] = f"input__{name}"
        elif h_node.version is not None:
            code_versions[name] = h_node.version
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
) -> SchemaIndex:
    inference_service = get_schema_inference_service(driver=driver, catalog=catalog)
    inferable_table_keys = inference_service.inferable_table_keys()
    declared_provider = source_declared_schema_provider(
        exclude_table_keys=inferable_table_keys,
    )
    return build_schema_index(
        system=catalog,
        declared_provider=declared_provider,
        inference_service=inference_service,
    )


def _runtime_key(
    *,
    env: BuildEnv,
    config: Mapping[str, Any],
    module_paths: tuple[str, ...],
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
        modules_fingerprint=fingerprint(module_paths),
        build_profile=config.get("profile") if isinstance(config.get("profile"), str) else None,
    )


def _runtime_fingerprint(
    *,
    env: BuildEnv,
    config: Mapping[str, Any],
    module_paths: tuple[str, ...],
) -> str:
    return fingerprint(
        {
            "repo": env.snapshot.repo,
            "commit": env.snapshot.commit,
            "config": config,
            "modules": module_paths,
            "variant_fingerprint": env.variants.variant_fingerprint,
        }
    )


__all__ = [
    "RuntimeComposition",
    "compose_runtime",
    "set_execution_active",
]
