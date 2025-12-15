"""Schema compilation utilities for producing schema manifests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.hamilton.native.registry import native_target_names
from codeintel.build.registry import get_target_graph
from codeintel.build.schemas.manifest import SchemaManifest
from codeintel.build.schemas.provider_hamilton import (
    HamiltonSchemaProvider,
    infer_schema_for_table_key,
    inferable_native_table_keys,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.targets import TargetModule
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.provider import SchemaProvider

DEFAULT_SCHEMA_MANIFEST_VERSION = "v1"


@dataclass(frozen=True)
class SchemaManifestRequest:
    """Selection and options for compiling a SchemaManifest.

    Attributes
    ----------
    targets
        Explicit target names to include.
    module
        Optional target module to include.
    all_targets
        When True, include all targets across all modules.
    only_native
        When True, restrict selection to targets with native implementations.
    infer_native
        When True, infer schemas for inferable native outputs (fallback to declared on error).
    stable
        When True, produce deterministic output ordering and de-duplication.
    version
        Manifest version identifier.
    """

    targets: tuple[str, ...] | None = None
    module: TargetModule | None = None
    all_targets: bool = False
    only_native: bool = False
    infer_native: bool = False
    stable: bool = True
    version: str = DEFAULT_SCHEMA_MANIFEST_VERSION


def _table_keys_for_selection(
    *,
    targets: list[str] | None,
    module: TargetModule | None,
    all_targets: bool,
    only_native: bool,
    stable: bool,
) -> tuple[str, ...]:
    graph = get_target_graph()

    if targets:
        missing = sorted(t for t in targets if t not in graph)
        if missing:
            msg = f"Unknown targets: {missing}"
            raise KeyError(msg)
        selected = [graph.get(t) for t in targets]
    elif module is not None:
        selected = list(graph.targets_for_module(module))
    else:
        selected = list(graph.all_targets) if all_targets or not (targets or module) else []

    if only_native:
        native_names = native_target_names()
        selected = [t for t in selected if t.name in native_names]
        if not selected:
            msg = "No native targets matched selection"
            raise ValueError(msg)

    table_keys: list[str] = []
    for target in selected:
        table_keys.extend(target.contract.table_keys)

    if stable:
        return tuple(sorted(set(table_keys)))

    seen: set[str] = set()
    ordered: list[str] = []
    for table_key in table_keys:
        if table_key in seen:
            continue
        seen.add(table_key)
        ordered.append(table_key)
    return tuple(ordered)


def compile_schema_manifest_for_table_keys(
    table_keys: Iterable[str],
    *,
    provider: SchemaProvider,
    version: str = DEFAULT_SCHEMA_MANIFEST_VERSION,
    stable: bool = True,
) -> SchemaManifest:
    """Compile a deterministic schema manifest for specific table keys.

    Parameters
    ----------
    table_keys
        Table keys (schema.table) to include.
    provider
        Schema provider used to resolve TableSchema definitions.
    version
        Manifest version identifier.
    stable
        When True, sort tables deterministically by table_key.

    Returns
    -------
    SchemaManifest
        Compiled schema manifest.
    """
    schemas = [provider.require_table_schema(key) for key in table_keys]
    if stable:
        schemas = sorted(schemas, key=lambda s: s.table_key)
    return SchemaManifest(version=version, tables=tuple(schemas))


def compile_schema_manifest(
    *,
    provider: SchemaProvider,
    request: SchemaManifestRequest | None = None,
) -> SchemaManifest:
    """Compile a schema manifest for a build target selection.

    Parameters
    ----------
    provider
        Base schema provider used for declared schemas.
    request
        Selection and options for manifest compilation. When None, uses defaults.

    Returns
    -------
    SchemaManifest
        Compiled schema manifest.
    """
    req = request or SchemaManifestRequest()
    graph = get_target_graph()
    table_keys = _table_keys_for_selection(
        targets=list(req.targets) if req.targets else None,
        module=req.module,
        all_targets=req.all_targets,
        only_native=req.only_native,
        stable=req.stable,
    )

    active_provider = provider
    if req.infer_native:
        inferable = set(inferable_native_table_keys(graph=graph))
        selected_inferable = frozenset(k for k in table_keys if k in inferable)

        def _infer(table_key: str) -> TableSchema:
            return infer_schema_for_table_key(table_key=table_key, declared_provider=provider)

        active_provider = HamiltonSchemaProvider(
            declared=provider,
            inferer=_infer,
            inferable_table_keys=selected_inferable,
            fallback_to_declared_on_error=True,
        )

    return compile_schema_manifest_for_table_keys(
        table_keys,
        provider=active_provider,
        version=req.version,
        stable=req.stable,
    )


__all__ = [
    "DEFAULT_SCHEMA_MANIFEST_VERSION",
    "SchemaManifest",
    "SchemaManifestRequest",
    "compile_schema_manifest",
    "compile_schema_manifest_for_table_keys",
]
