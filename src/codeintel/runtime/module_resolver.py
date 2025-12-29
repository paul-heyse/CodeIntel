"""Module discovery for Hamilton runtime composition."""

from __future__ import annotations

import hashlib
import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Literal

from codeintel.build.hamilton.native.discovery import native_module_paths
from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.runtime.plugins.config import PluginConfig
from codeintel.runtime.plugins.loader import TargetPackEntry, discover_target_pack_entries
from codeintel.runtime.plugins.spec import TargetPack

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.build.hamilton.env import BuildEnv

log = logging.getLogger(__name__)

_SUPPORT_NODES_MODULE = "codeintel.build.hamilton.nodes.support_nodes"


@dataclass(frozen=True, slots=True)
class ModuleProvenance:
    """Provenance metadata for a resolved module."""

    origin: Literal["core", "workspace", "plugin"]
    module_import: str
    file_path: str | None
    plugin_name: str | None
    dist_name: str | None
    dist_version: str | None


@dataclass(frozen=True, slots=True)
class ResolvedModuleSet:
    """Resolved runtime modules with provenance and fingerprint."""

    modules: tuple[ModuleType, ...]
    provenance: dict[str, ModuleProvenance]
    fingerprint: str
    packs: tuple[TargetPack, ...]

    @property
    def module_paths(self) -> tuple[str, ...]:
        """Return import paths for the resolved modules."""
        return tuple(module.__name__ for module in self.modules)


def resolve_module_set(
    *,
    env: BuildEnv,
    plugin_config: PluginConfig,
    hamilton_config: Mapping[str, object],
    include_planning: bool = True,
    codeintel_version: str | None = None,
) -> ResolvedModuleSet:
    """Resolve core, workspace, and plugin modules deterministically.

    Returns
    -------
    ResolvedModuleSet
        Resolved modules with provenance and fingerprint metadata.
    """
    resolved_version = codeintel_version or env.settings.engine_version
    pack_entries = _resolve_pack_entries(
        plugin_config=plugin_config,
        codeintel_version=resolved_version,
    )
    modules = _resolve_modules(
        env=env,
        pack_entries=pack_entries,
        plugin_config=plugin_config,
        include_planning=include_planning,
    )
    config_digest = fingerprint(
        {
            "hamilton_config": dict(hamilton_config),
            "plugins": plugin_config.as_dict(),
        }
    )
    modules_fingerprint = _modules_fingerprint(
        modules=modules.modules,
        packs=pack_entries,
        config_digest=config_digest,
    )
    packs = tuple(entry.pack for entry in pack_entries)
    return ResolvedModuleSet(
        modules=modules.modules,
        provenance=modules.provenance,
        fingerprint=modules_fingerprint,
        packs=packs,
    )


def resolve_module_paths(
    *,
    include_planning: bool = True,
    env: BuildEnv | None = None,
    plugin_config: PluginConfig | None = None,
    hamilton_config: Mapping[str, object] | None = None,
    codeintel_version: str | None = None,
) -> tuple[str, ...]:
    """Return module import paths for runtime composition.

    Returns
    -------
    tuple[str, ...]
        Ordered module import paths for the runtime.
    """
    if env is None:
        return _core_module_paths(include_planning=include_planning)
    resolved = resolve_module_set(
        env=env,
        plugin_config=plugin_config or PluginConfig(),
        hamilton_config=hamilton_config or {},
        include_planning=include_planning,
        codeintel_version=codeintel_version,
    )
    return resolved.module_paths


def resolve_modules(
    *,
    include_planning: bool = True,
    env: BuildEnv | None = None,
    plugin_config: PluginConfig | None = None,
    hamilton_config: Mapping[str, object] | None = None,
    codeintel_version: str | None = None,
) -> tuple[ModuleType, ...]:
    """Return imported modules for runtime composition.

    Returns
    -------
    tuple[ModuleType, ...]
        Imported modules for runtime composition.
    """
    if env is None:
        core_spec = _ModuleDescriptorSpec(
            origin="core",
            plugin_name=None,
            dist_name=None,
            dist_version=None,
        )
        return _import_modules(
            _core_module_paths(include_planning=include_planning),
            spec=core_spec,
            strict=True,
        ).modules
    resolved = resolve_module_set(
        env=env,
        plugin_config=plugin_config or PluginConfig(),
        hamilton_config=hamilton_config or {},
        include_planning=include_planning,
        codeintel_version=codeintel_version,
    )
    return resolved.modules


@dataclass(frozen=True, slots=True)
class _ResolvedModules:
    modules: tuple[ModuleType, ...]
    provenance: dict[str, ModuleProvenance]


@dataclass(frozen=True, slots=True)
class _ModuleDescriptor:
    import_path: str
    origin: Literal["core", "workspace", "plugin"]
    plugin_name: str | None
    dist_name: str | None
    dist_version: str | None


@dataclass(frozen=True, slots=True)
class _ModuleDescriptorSpec:
    origin: Literal["core", "workspace", "plugin"]
    plugin_name: str | None
    dist_name: str | None
    dist_version: str | None


def _resolve_pack_entries(
    *,
    plugin_config: PluginConfig,
    codeintel_version: str,
) -> tuple[TargetPackEntry, ...]:
    entries = discover_target_pack_entries(
        codeintel_version=codeintel_version,
        strict=plugin_config.strict,
    )
    enabled_set = set(plugin_config.enabled) if plugin_config.enabled is not None else None
    disabled_set = set(plugin_config.disabled)
    selected: list[TargetPackEntry] = []
    for entry in entries:
        pack = entry.pack
        if enabled_set is not None and pack.name not in enabled_set:
            continue
        if pack.name in disabled_set:
            continue
        if enabled_set is None and not pack.default_enabled:
            continue
        selected.append(entry)
    selected.sort(key=lambda entry: (entry.pack.name, entry.pack.version))
    return tuple(selected)


def _resolve_modules(
    *,
    env: BuildEnv,
    pack_entries: Sequence[TargetPackEntry],
    plugin_config: PluginConfig,
    include_planning: bool,
) -> _ResolvedModules:
    descriptors: list[_ModuleDescriptor] = []
    seen: set[str] = set()
    core_spec = _ModuleDescriptorSpec(
        origin="core",
        plugin_name=None,
        dist_name=None,
        dist_version=None,
    )
    workspace_spec = _ModuleDescriptorSpec(
        origin="workspace",
        plugin_name=None,
        dist_name=None,
        dist_version=None,
    )

    for path in _core_module_paths(include_planning=include_planning):
        _append_descriptor(
            descriptors=descriptors,
            seen=seen,
            path=path,
            spec=core_spec,
            strict=plugin_config.strict,
        )

    if plugin_config.allow_workspace_modules:
        for path in _workspace_module_paths(env=env):
            if not include_planning and ".planning." in path:
                continue
            _append_descriptor(
                descriptors=descriptors,
                seen=seen,
                path=path,
                spec=workspace_spec,
                strict=plugin_config.strict,
            )

    for entry in pack_entries:
        plugin_spec = _ModuleDescriptorSpec(
            origin="plugin",
            plugin_name=entry.pack.name,
            dist_name=entry.dist_name,
            dist_version=entry.dist_version,
        )
        for module in entry.pack.modules:
            path = module.import_path
            if not include_planning and ".planning." in path:
                continue
            _append_descriptor(
                descriptors=descriptors,
                seen=seen,
                path=path,
                spec=plugin_spec,
                strict=plugin_config.strict,
            )

    resolved = _import_descriptors(descriptors, strict=plugin_config.strict)
    log.info(
        "runtime.module_resolver resolved modules=%d packs=%d",
        len(resolved.modules),
        len(pack_entries),
    )
    return resolved


def _append_descriptor(
    *,
    descriptors: list[_ModuleDescriptor],
    seen: set[str],
    path: str,
    spec: _ModuleDescriptorSpec,
    strict: bool,
) -> None:
    if path in seen:
        msg = f"Duplicate module import path detected: {path}"
        if strict:
            raise ValueError(msg)
        log.warning(msg)
        return
    seen.add(path)
    descriptors.append(
        _ModuleDescriptor(
            import_path=path,
            origin=spec.origin,
            plugin_name=spec.plugin_name,
            dist_name=spec.dist_name,
            dist_version=spec.dist_version,
        )
    )


def _core_module_paths(*, include_planning: bool) -> tuple[str, ...]:
    paths = list(native_module_paths())
    paths.append(_SUPPORT_NODES_MODULE)
    if not include_planning:
        return tuple(path for path in paths if ".planning." not in path)
    return tuple(paths)


def _workspace_module_paths(*, env: BuildEnv) -> tuple[str, ...]:
    repo_root = env.snapshot.repo_root
    source_root = repo_root / "src"
    workspace_root = source_root / "codeintel_targets"
    if not workspace_root.exists():
        return ()

    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    module_paths: list[str] = []
    for path in sorted(workspace_root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        rel = path.relative_to(source_root)
        module_paths.append(".".join(rel.with_suffix("").parts))
    return tuple(module_paths)


def _import_descriptors(
    descriptors: Sequence[_ModuleDescriptor],
    *,
    strict: bool,
) -> _ResolvedModules:
    modules: list[ModuleType] = []
    provenance: dict[str, ModuleProvenance] = {}
    for desc in descriptors:
        try:
            module = importlib.import_module(desc.import_path)
        except Exception as exc:
            if strict:
                msg = f"Failed to import module {desc.import_path}: {exc}"
                raise RuntimeError(msg) from exc
            log.warning("Skipping module %s: %s", desc.import_path, exc)
            continue
        modules.append(module)
        provenance[module.__name__] = ModuleProvenance(
            origin=desc.origin,
            module_import=desc.import_path,
            file_path=getattr(module, "__file__", None),
            plugin_name=desc.plugin_name,
            dist_name=desc.dist_name,
            dist_version=desc.dist_version,
        )
    return _ResolvedModules(modules=tuple(modules), provenance=provenance)


def _import_modules(
    module_paths: Sequence[str],
    *,
    spec: _ModuleDescriptorSpec,
    strict: bool,
) -> _ResolvedModules:
    descriptors = [
        _ModuleDescriptor(
            import_path=path,
            origin=spec.origin,
            plugin_name=spec.plugin_name,
            dist_name=spec.dist_name,
            dist_version=spec.dist_version,
        )
        for path in module_paths
    ]
    return _import_descriptors(descriptors, strict=strict)


def _modules_fingerprint(
    *,
    modules: Sequence[ModuleType],
    packs: Sequence[TargetPackEntry],
    config_digest: str,
) -> str:
    hasher = hashlib.sha256(usedforsecurity=False)
    hasher.update(config_digest.encode())
    for entry in sorted(packs, key=lambda item: (item.pack.name, item.pack.version)):
        hasher.update(f"pack:{entry.pack.name}:{entry.pack.version}".encode())
        if entry.dist_name and entry.dist_version:
            hasher.update(f"dist:{entry.dist_name}:{entry.dist_version}".encode())

    for module in sorted(modules, key=lambda mod: mod.__name__):
        hasher.update(module.__name__.encode())
        hasher.update(_module_content_hash(module).encode())

    return hasher.hexdigest()


def _module_content_hash(module: ModuleType) -> str:
    file_path = getattr(module, "__file__", None)
    if not file_path:
        return "nofile"
    try:
        data = Path(file_path).read_bytes()
    except OSError as exc:
        log.warning("module.hash_failed module=%s error=%s", module.__name__, exc)
        return "nofile"
    return hashlib.sha256(data, usedforsecurity=False).hexdigest()


__all__ = [
    "ModuleProvenance",
    "ResolvedModuleSet",
    "resolve_module_paths",
    "resolve_module_set",
    "resolve_modules",
]
