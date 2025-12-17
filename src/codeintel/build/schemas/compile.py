"""Schema compilation utilities for producing schema manifests.

This module provides functions for compiling SchemaManifest objects from
build target selections. The v2 format extends compilation to include
view schemas and export artifact specifications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from codeintel.build.hamilton.native.registry import native_target_names
from codeintel.build.schemas.contract_provider import iter_contracts
from codeintel.build.schemas.infer_duckdb import infer_view_schema
from codeintel.build.schemas.manifest import ExportArtifact, SchemaManifest
from codeintel.build.schemas.provider_hamilton import (
    HamiltonSchemaProvider,
    infer_schema_for_table_key,
    infer_table_schemas,
    inferable_native_table_keys,
)
from codeintel.build.target_system import load_target_system
from codeintel.storage.views.inventory import discover_derived_docs_views

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.targets import TargetModule
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.provider import SchemaProvider
    from codeintel.storage.gateway.protocol import DuckDBConnection

_logger = logging.getLogger(__name__)

DEFAULT_SCHEMA_MANIFEST_VERSION = "v1"
V2_SCHEMA_MANIFEST_VERSION = "v2"


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
    batch_infer_native
        When True, pre-infer all selected inferable native schemas in a single ephemeral session.
    stable
        When True, produce deterministic output ordering and de-duplication.
    version
        Manifest version identifier.
    include_views
        When True, include DuckDB view schemas in the manifest (v2).
    include_artifacts
        When True, include export artifact specifications in the manifest (v2).
    """

    targets: tuple[str, ...] | None = None
    module: TargetModule | None = None
    all_targets: bool = False
    only_native: bool = False
    infer_native: bool = False
    batch_infer_native: bool = True
    stable: bool = True
    version: str = DEFAULT_SCHEMA_MANIFEST_VERSION
    include_views: bool = False
    include_artifacts: bool = False


class NativeBatchInferer(Protocol):
    """Callable protocol for batch native schema inference."""

    def __call__(
        self,
        table_keys: Iterable[str],
        *,
        declared_provider: SchemaProvider,
    ) -> dict[str, TableSchema]:
        """Infer schemas for multiple table keys."""
        ...


def _table_keys_for_selection(
    *,
    targets: list[str] | None,
    module: TargetModule | None,
    all_targets: bool,
    only_native: bool,
    stable: bool,
) -> tuple[str, ...]:
    graph = load_target_system().graph

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


def _collect_view_schemas(
    *,
    con: DuckDBConnection,
    stable: bool,
) -> tuple[TableSchema, ...]:
    """Collect schemas for all known DuckDB views.

    Iterates through discovered docs views and infers schemas for views that
    exist in the database. Views that don't exist are silently skipped.

    Parameters
    ----------
    con
        DuckDB connection with views created.
    stable
        When True, sort views deterministically by table_key.

    Returns
    -------
    tuple[TableSchema, ...]
        Inferred view schemas.
    """
    views: list[TableSchema] = []
    for view_key in discover_derived_docs_views():
        try:
            view_schema = infer_view_schema(con=con, view_key=view_key)
            views.append(view_schema)
        except (RuntimeError, OSError, ValueError):
            # Skip views that don't exist or can't be described
            _logger.debug("Skipping view %s (not found or error)", view_key)
            continue

    if stable:
        views = sorted(views, key=lambda v: v.table_key)
    return tuple(views)


def _collect_export_artifacts(*, stable: bool) -> tuple[ExportArtifact, ...]:
    """Collect export artifact specifications from contracts metadata.

    Uses derived DatasetContracts (via the build-owned contract provider) to
    collect export filenames for JSONL and Parquet artifacts.

    Parameters
    ----------
    stable
        When True, sort artifacts deterministically.

    Returns
    -------
    tuple[ExportArtifact, ...]
        Export artifact specifications.
    """
    artifacts: list[ExportArtifact] = []

    for contract in iter_contracts():
        if contract.jsonl_filename is not None:
            artifacts.append(
                ExportArtifact(
                    kind="jsonl",
                    filename=contract.jsonl_filename,
                    table_key=contract.table_key,
                )
            )
        if contract.parquet_filename is not None:
            artifacts.append(
                ExportArtifact(
                    kind="parquet",
                    filename=contract.parquet_filename,
                    table_key=contract.table_key,
                )
            )

    if stable:
        artifacts = sorted(artifacts, key=lambda a: (a.kind, a.table_key or ""))
    return tuple(artifacts)


@dataclass(frozen=True)
class V2Extras:
    """Optional v2 manifest extras (views and artifacts).

    Parameters
    ----------
    views
        View schemas to include (v2 feature).
    artifacts
        Export artifacts to include (v2 feature).
    """

    views: tuple[TableSchema, ...] = ()
    artifacts: tuple[ExportArtifact, ...] = ()


def compile_schema_manifest_for_table_keys(
    table_keys: Iterable[str],
    *,
    provider: SchemaProvider,
    version: str = DEFAULT_SCHEMA_MANIFEST_VERSION,
    stable: bool = True,
    extras: V2Extras | None = None,
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
    extras
        Optional v2 extras (views and artifacts).

    Returns
    -------
    SchemaManifest
        Compiled schema manifest.
    """
    schemas = [provider.require_table_schema(key) for key in table_keys]
    if stable:
        schemas = sorted(schemas, key=lambda s: s.table_key)
    v2 = extras or V2Extras()
    return SchemaManifest(
        version=version,
        tables=tuple(schemas),
        views=v2.views,
        artifacts=v2.artifacts,
    )


def compile_schema_manifest(
    *,
    provider: SchemaProvider,
    request: SchemaManifestRequest | None = None,
    con: DuckDBConnection | None = None,
    batch_inferer: NativeBatchInferer | None = None,
) -> SchemaManifest:
    """Compile a schema manifest for a build target selection.

    Parameters
    ----------
    provider
        Base schema provider used for declared schemas.
    request
        Selection and options for manifest compilation. When None, uses defaults.
    con
        Optional DuckDB connection required for view schema inference.
        Must be provided if request.include_views is True.
    batch_inferer
        Optional callable used to batch-infer native table schemas in a single pass.

    Returns
    -------
    SchemaManifest
        Compiled schema manifest.

    Raises
    ------
    ValueError
        If include_views is True but no connection is provided.
    """
    req = request or SchemaManifestRequest()
    graph = load_target_system().graph
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

        hamilton_provider = HamiltonSchemaProvider(
            declared=provider,
            inferer=_infer,
            inferable_table_keys=selected_inferable,
            fallback_to_declared_on_error=True,
        )
        active_provider = hamilton_provider
        if req.batch_infer_native and selected_inferable:
            try:
                batch = infer_table_schemas if batch_inferer is None else batch_inferer
                inferred = batch(selected_inferable, declared_provider=provider)
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                _logger.warning(
                    "Batch inference failed; falling back to per-table inference: %s", exc
                )
            else:
                hamilton_provider.prefill_cache(inferred)

    # Collect views if requested (v2 feature)
    views: tuple[TableSchema, ...] = ()
    if req.include_views:
        if con is None:
            msg = "DuckDB connection required for view schema inference"
            raise ValueError(msg)
        views = _collect_view_schemas(con=con, stable=req.stable)

    # Collect artifacts if requested (v2 feature)
    artifacts: tuple[ExportArtifact, ...] = ()
    if req.include_artifacts:
        artifacts = _collect_export_artifacts(stable=req.stable)

    # Determine version: use v2 if views or artifacts are included
    version = req.version
    if (views or artifacts) and version == DEFAULT_SCHEMA_MANIFEST_VERSION:
        version = V2_SCHEMA_MANIFEST_VERSION

    extras = V2Extras(views=views, artifacts=artifacts) if views or artifacts else None

    return compile_schema_manifest_for_table_keys(
        table_keys,
        provider=active_provider,
        version=version,
        stable=req.stable,
        extras=extras,
    )


__all__ = [
    "DEFAULT_SCHEMA_MANIFEST_VERSION",
    "V2_SCHEMA_MANIFEST_VERSION",
    "SchemaManifest",
    "SchemaManifestRequest",
    "V2Extras",
    "compile_schema_manifest",
    "compile_schema_manifest_for_table_keys",
]
