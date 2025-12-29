"""Entry point discovery and validation for CodeIntel target packs."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from packaging.specifiers import InvalidSpecifier, SpecifierSet

from codeintel.runtime.plugins.spec import TargetPack, TargetPackModule

if TYPE_CHECKING:
    from collections.abc import Iterable
    from importlib.metadata import EntryPoint

log = logging.getLogger(__name__)

TARGET_PACK_ENTRYPOINT_GROUP = "codeintel.target_packs"


@dataclass(frozen=True, slots=True)
class TargetPackEntry:
    """Loaded target pack with distribution metadata."""

    pack: TargetPack
    entry_point: EntryPoint
    dist_name: str | None
    dist_version: str | None


def discover_target_packs(
    *,
    codeintel_version: str,
    group: str = TARGET_PACK_ENTRYPOINT_GROUP,
    strict: bool = True,
) -> tuple[TargetPack, ...]:
    """Discover target packs exposed via entry points.

    Parameters
    ----------
    codeintel_version
        CodeIntel runtime version used for compatibility checks.
    group
        Entry point group to inspect.
    strict
        Whether to raise on invalid packs instead of skipping.

    Returns
    -------
    tuple[TargetPack, ...]
        Loaded target packs sorted by name and version.
    """
    entries = _discover_target_pack_entries(
        codeintel_version=codeintel_version,
        group=group,
        strict=strict,
    )
    return tuple(entry.pack for entry in entries)


def discover_target_pack_entries(
    *,
    codeintel_version: str,
    group: str = TARGET_PACK_ENTRYPOINT_GROUP,
    strict: bool = True,
) -> tuple[TargetPackEntry, ...]:
    """Discover target packs with distribution metadata.

    Parameters
    ----------
    codeintel_version
        CodeIntel runtime version used for compatibility checks.
    group
        Entry point group to inspect.
    strict
        Whether to raise on invalid packs instead of skipping.

    Returns
    -------
    tuple[TargetPackEntry, ...]
        Loaded target packs with distribution metadata.
    """
    return _discover_target_pack_entries(
        codeintel_version=codeintel_version,
        group=group,
        strict=strict,
    )


def load_pack(
    entry_point: EntryPoint,
    *,
    codeintel_version: str,
    strict: bool = True,
) -> TargetPack | None:
    """Load and validate a pack from an entry point.

    Parameters
    ----------
    entry_point
        Entry point providing the target pack factory.
    codeintel_version
        CodeIntel runtime version used for compatibility checks.
    strict
        Whether to raise on invalid packs instead of skipping.

    Returns
    -------
    TargetPack | None
        Normalized pack when load succeeds, otherwise None when non-strict.

    Raises
    ------
    RuntimeError
        When loading the entry point or pack fails in strict mode.
    TypeError
        When the entry point or pack payload is invalid in strict mode.
    ValueError
        When the pack metadata is invalid in strict mode.
    """
    factory = _load_entry_point(entry_point, strict=strict)
    if factory is None:
        return None
    pack = _load_factory(entry_point=entry_point, factory=factory, strict=strict)
    if pack is None:
        return None
    try:
        normalized = _normalize_pack(pack)
        validate_pack(normalized, codeintel_version=codeintel_version)
    except (TypeError, ValueError, RuntimeError) as exc:
        if strict:
            raise
        log.warning("Skipping invalid target pack %s: %s", entry_point.name, exc)
        return None
    return normalized


def validate_pack(pack: TargetPack, *, codeintel_version: str) -> None:
    """Validate pack metadata and compatibility.

    Parameters
    ----------
    pack
        Target pack to validate.
    codeintel_version
        CodeIntel runtime version used for compatibility checks.
    """
    _validate_pack_metadata(pack)
    _validate_pack_version(pack=pack, codeintel_version=codeintel_version)


def _validate_pack_metadata(pack: TargetPack) -> None:
    """Validate pack metadata structure.

    Parameters
    ----------
    pack
        Target pack to validate.

    Raises
    ------
    ValueError
        When required metadata is missing or malformed.
    """
    if not pack.name:
        msg = "Pack name must be non-empty"
        raise ValueError(msg)
    if not pack.version:
        msg = f"Pack {pack.name} must define a non-empty version"
        raise ValueError(msg)
    if not pack.requires_codeintel:
        msg = f"Pack {pack.name} must define requires_codeintel"
        raise ValueError(msg)
    if not pack.modules:
        msg = f"Pack {pack.name} has no modules"
        raise ValueError(msg)
    for module in pack.modules:
        if not module.import_path:
            msg = f"Pack {pack.name} declares empty module import path"
            raise ValueError(msg)
        if module.kind != "hamilton":
            msg = f"Pack {pack.name} module {module.import_path} has invalid kind={module.kind}"
            raise ValueError(msg)


def _validate_pack_version(*, pack: TargetPack, codeintel_version: str) -> None:
    """Validate pack compatibility with the runtime version.

    Parameters
    ----------
    pack
        Target pack to validate.
    codeintel_version
        CodeIntel runtime version used for compatibility checks.

    Raises
    ------
    RuntimeError
        When the pack version is incompatible with the runtime version.
    ValueError
        When the pack version specifier is invalid.
    """
    if codeintel_version == "unknown":
        log.warning(
            "Pack %s: skipping version check because CodeIntel version is unknown",
            pack.name,
        )
        return
    if not codeintel_version:
        return
    try:
        spec = SpecifierSet(pack.requires_codeintel)
    except InvalidSpecifier as exc:
        msg = f"Pack {pack.name} has invalid requires_codeintel spec: {exc}"
        raise ValueError(msg) from exc
    if not spec.contains(codeintel_version, prereleases=True):
        msg = (
            f"Pack {pack.name} requires CodeIntel {pack.requires_codeintel} "
            f"(installed {codeintel_version})"
        )
        raise RuntimeError(msg)


def _discover_target_pack_entries(
    *,
    codeintel_version: str,
    group: str,
    strict: bool,
) -> tuple[TargetPackEntry, ...]:
    entry_points = _select_entry_points(group)
    loaded: list[TargetPackEntry] = []
    for entry_point in entry_points:
        pack = load_pack(entry_point, codeintel_version=codeintel_version, strict=strict)
        if pack is None:
            continue
        dist_name, dist_version = _entry_point_dist_info(entry_point)
        loaded.append(
            TargetPackEntry(
                pack=pack,
                entry_point=entry_point,
                dist_name=dist_name,
                dist_version=dist_version,
            )
        )
    loaded.sort(key=lambda entry: (entry.pack.name, entry.pack.version))
    return tuple(loaded)


def _select_entry_points(group: str) -> list[EntryPoint]:
    all_eps = importlib_metadata.entry_points()
    try:
        selected = all_eps.select(group=group)
    except AttributeError:
        selected = [ep for ep in all_eps if ep.group == group]
    return sorted(selected, key=lambda ep: (ep.name, ep.value))


def _entry_point_dist_info(entry_point: EntryPoint) -> tuple[str | None, str | None]:
    dist = getattr(entry_point, "dist", None)
    if dist is None:
        return None, None
    dist_name = getattr(dist, "metadata", {}).get("Name") if hasattr(dist, "metadata") else None
    dist_version = getattr(dist, "version", None)
    if dist_name is None and hasattr(dist, "name"):
        dist_name = dist.name
    return dist_name, dist_version


def _load_entry_point(entry_point: EntryPoint, *, strict: bool) -> Callable[[], TargetPack] | None:
    try:
        factory = entry_point.load()
    except Exception as exc:
        if strict:
            msg = f"Failed to load target pack entry point {entry_point.name}: {exc}"
            raise RuntimeError(msg) from exc
        log.warning("Skipping target pack entry point %s: %s", entry_point.name, exc)
        return None

    if not callable(factory):
        msg = f"Target pack entry point {entry_point.name} did not return a callable"
        raise TypeError(msg)
    return cast("Callable[[], TargetPack]", factory)


def _load_factory(
    *,
    entry_point: EntryPoint,
    factory: Callable[[], TargetPack],
    strict: bool,
) -> TargetPack | None:
    try:
        pack = factory()
    except Exception as exc:
        if strict:
            msg = f"Target pack factory {entry_point.name} raised: {exc}"
            raise RuntimeError(msg) from exc
        log.warning("Skipping target pack factory %s: %s", entry_point.name, exc)
        return None
    if not isinstance(pack, TargetPack):
        msg = f"Target pack factory {entry_point.name} returned {type(pack)!r}, expected TargetPack"
        raise TypeError(msg)
    return pack


def _normalize_pack(pack: TargetPack) -> TargetPack:
    modules = _normalize_modules(pack.modules)
    if pack.config_namespace is not None and not pack.config_namespace:
        msg = f"Pack {pack.name} config_namespace must be non-empty when set"
        raise ValueError(msg)
    capabilities = _normalize_capabilities(pack.capabilities)
    return TargetPack(
        name=pack.name,
        version=pack.version,
        modules=modules,
        requires_codeintel=pack.requires_codeintel,
        default_enabled=pack.default_enabled,
        config_namespace=pack.config_namespace,
        capabilities=capabilities,
    )


def _normalize_modules(
    modules: Iterable[TargetPackModule | str],
) -> tuple[TargetPackModule, ...]:
    resolved: list[TargetPackModule] = []
    for module in modules:
        if isinstance(module, TargetPackModule):
            resolved.append(module)
            continue
        if isinstance(module, str):
            resolved.append(TargetPackModule(import_path=module))
            continue
        msg = f"Invalid module entry: {module!r}"
        raise TypeError(msg)
    unique: dict[str, TargetPackModule] = {m.import_path: m for m in resolved}
    return tuple(sorted(unique.values(), key=lambda m: m.import_path))


def _normalize_capabilities(capabilities: Iterable[str] | frozenset[str]) -> frozenset[str]:
    if isinstance(capabilities, frozenset):
        return capabilities
    return frozenset(str(value) for value in capabilities)


__all__ = [
    "TARGET_PACK_ENTRYPOINT_GROUP",
    "TargetPackEntry",
    "discover_target_pack_entries",
    "discover_target_packs",
    "load_pack",
    "validate_pack",
]
