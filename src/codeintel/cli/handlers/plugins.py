"""Plugin inspection handlers for the CLI."""

from __future__ import annotations

import hashlib
import importlib.util as importlib_util
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.config import load_build_config
from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import (
    ListResult,
    PluginModuleInfo,
    PluginPackDetail,
    PluginPackInfo,
)
from codeintel.cli.errors.results import fail_not_found
from codeintel.core.runtime.loader import load_runtime_settings
from codeintel.runtime.plugins.config import plugin_config_from_build_config
from codeintel.runtime.plugins.loader import TargetPackEntry, discover_target_pack_entries

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext
    from codeintel.runtime.plugins.config import PluginConfig

LOG = logging.getLogger(__name__)


def plugins_list_handler(ctx: CommandContext) -> CliResult[ListResult[PluginPackInfo]]:
    """List discovered plugin packs.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    CliResult[ListResult[PluginPackInfo]]
        CLI result containing plugin pack summaries.
    """
    runtime = ctx.runtime
    config = load_build_config(runtime.snapshot.repo_root)
    plugin_config = plugin_config_from_build_config(config)
    engine_version = load_runtime_settings().build.engine_version
    entries = discover_target_pack_entries(
        codeintel_version=engine_version,
        strict=plugin_config.strict,
    )
    items = [
        _pack_summary(entry=entry, enabled=_pack_enabled(entry=entry, plugin_config=plugin_config))
        for entry in entries
    ]
    items.sort(key=lambda pack: (pack.name, pack.version))
    return CliResult.ok(ListResult.from_items(items))


def plugins_info_handler(ctx: CommandContext) -> CliResult[PluginPackDetail]:
    """Show detailed information for a plugin pack.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    CliResult[PluginPackDetail]
        CLI result containing detailed plugin pack metadata.
    """
    runtime = ctx.runtime
    name = ctx.params.require_str("name")
    config = load_build_config(runtime.snapshot.repo_root)
    plugin_config = plugin_config_from_build_config(config)
    engine_version = load_runtime_settings().build.engine_version
    entries = discover_target_pack_entries(
        codeintel_version=engine_version,
        strict=plugin_config.strict,
    )
    entry = next((item for item in entries if item.pack.name == name), None)
    if entry is None:
        return fail_not_found("plugin", name)
    enabled = _pack_enabled(entry=entry, plugin_config=plugin_config)
    module_infos = _pack_module_infos(entry)
    return CliResult.ok(
        PluginPackDetail(
            name=entry.pack.name,
            version=entry.pack.version,
            enabled=enabled,
            default_enabled=entry.pack.default_enabled,
            modules=module_infos,
            requires_codeintel=entry.pack.requires_codeintel,
            config_namespace=entry.pack.config_namespace,
            dist_name=entry.dist_name,
            dist_version=entry.dist_version,
            capabilities=sorted(entry.pack.capabilities),
            entry_point=str(entry.entry_point.value),
        )
    )


def _pack_summary(*, entry: TargetPackEntry, enabled: bool) -> PluginPackInfo:
    modules = [module.import_path for module in entry.pack.modules]
    return PluginPackInfo(
        name=entry.pack.name,
        version=entry.pack.version,
        enabled=enabled,
        default_enabled=entry.pack.default_enabled,
        modules=modules,
        requires_codeintel=entry.pack.requires_codeintel,
        config_namespace=entry.pack.config_namespace,
        dist_name=entry.dist_name,
        dist_version=entry.dist_version,
        capabilities=sorted(entry.pack.capabilities),
    )


def _pack_enabled(*, entry: TargetPackEntry, plugin_config: PluginConfig) -> bool:
    enabled_set = set(plugin_config.enabled) if plugin_config.enabled is not None else None
    disabled_set = set(plugin_config.disabled)
    if enabled_set is not None:
        return entry.pack.name in enabled_set
    if entry.pack.name in disabled_set:
        return False
    return entry.pack.default_enabled


def _pack_module_infos(entry: TargetPackEntry) -> list[PluginModuleInfo]:
    infos: list[PluginModuleInfo] = []
    for module in entry.pack.modules:
        path = _module_file_path(module.import_path)
        content_hash = _hash_file(path)
        infos.append(
            PluginModuleInfo(
                import_path=module.import_path,
                file_path=str(path) if path is not None else None,
                content_hash=content_hash,
            )
        )
    return infos


def _module_file_path(import_path: str) -> Path | None:
    try:
        spec = importlib_util.find_spec(import_path)
    except ImportError as exc:
        LOG.warning("plugins.import_failed module=%s error=%s", import_path, exc)
        return None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin)


def _hash_file(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        data = path.read_bytes()
    except OSError as exc:
        LOG.warning("plugins.hash_failed path=%s error=%s", path, exc)
        return None
    return hashlib.sha256(data, usedforsecurity=False).hexdigest()


__all__ = [
    "plugins_info_handler",
    "plugins_list_handler",
]
