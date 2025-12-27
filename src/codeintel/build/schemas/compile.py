"""Schema compilation utilities for producing schema manifests.

This module provides functions for compiling SchemaManifest objects from
build target selections. The v2 format extends compilation to include
view schemas and export artifact specifications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from codeintel.build.hamilton.impl_kind import native_target_names
from codeintel.build.schemas.contract_service import (
    ContractResolutionMode,
    ContractResolutionSettings,
)
from codeintel.build.schemas.infer_duckdb import infer_view_schema
from codeintel.build.schemas.inference_service import infer_table_schemas
from codeintel.build.schemas.manifest import (
    ArtifactProvenance,
    ExportArtifact,
    SchemaManifest,
    TableProvenance,
)
from codeintel.build.schemas.provider_unified import (
    UnifiedSchemaProvider,
)
from codeintel.build.target_metadata import get_target_metadata_service
from codeintel.core.schemas.contract_service import iter_contracts
from codeintel.core.schemas.hashing import schema_hash
from codeintel.storage.views.inventory import discover_derived_docs_views

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.hamilton.dag_catalog import DagCatalog
    from codeintel.build.hamilton.driver_factory import HamiltonRuntime
    from codeintel.build.schemas.schema_index import SchemaIndex
    from codeintel.build.targets import TargetModule
    from codeintel.core.hamilton.tag_query import TagQuery
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.provider import SchemaProvider
    from codeintel.storage.gateway.protocol import DuckDBConnection

_logger = logging.getLogger(__name__)

V2_SCHEMA_MANIFEST_VERSION = "v2"
DEFAULT_SCHEMA_MANIFEST_VERSION = V2_SCHEMA_MANIFEST_VERSION
DECLARED_SOURCE_KIND = "declared_source"
DECLARED_SOURCE_NAME = "declared"
VIEW_DERIVATION_KIND = "view_inferred"
VIEW_DERIVATION_SOURCE = "duckdb"


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
    infer_native
        When True, infer schemas for inferable native outputs.
    batch_infer_native
        When True, pre-infer all selected inferable native schemas in a single ephemeral session.
    stable
        When True, produce deterministic output ordering and de-duplication.
    version
        Manifest version identifier (v2 only).
    include_views
        When True, include DuckDB view schemas in the manifest.
    include_artifacts
        When True, include export artifact specifications in the manifest.
    include_provenance
        When True, include per-entry provenance fields in the manifest.
    """

    targets: tuple[str, ...] | None = None
    module: TargetModule | None = None
    all_targets: bool = False
    infer_native: bool = True
    batch_infer_native: bool = True
    stable: bool = True
    version: str = DEFAULT_SCHEMA_MANIFEST_VERSION
    include_views: bool = False
    include_artifacts: bool = False
    include_provenance: bool = False


@dataclass(frozen=True)
class TableKeySelection:
    """Normalized selection for target table keys.

    Attributes
    ----------
    targets
        Explicit target names to include.
    module
        Optional target module to include.
    all_targets
        When True, include all targets across all modules.
    stable
        When True, preserve deterministic ordering.
    """

    targets: tuple[str, ...] | None
    module: TargetModule | None
    all_targets: bool
    stable: bool

    @classmethod
    def from_request(cls, request: SchemaManifestRequest) -> TableKeySelection:
        """Build a selection from a manifest request.

        Parameters
        ----------
        request
            Manifest request to derive selection from.

        Returns
        -------
        TableKeySelection
            Normalized selection derived from the request.
        """
        return cls(
            targets=request.targets,
            module=request.module,
            all_targets=request.all_targets,
            stable=request.stable,
        )


class NativeBatchInferer(Protocol):
    """Callable protocol for batch native schema inference."""

    def __call__(
        self,
        table_keys: Iterable[str],
        *,
        declared_provider: SchemaProvider,
    ) -> dict[str, TableSchema]:
        """Infer schemas for multiple table keys.

        Parameters
        ----------
        table_keys
            Table keys to infer.
        declared_provider
            Declared schema provider for fallback.

        Returns
        -------
        dict[str, TableSchema]
            Mapping of table key to inferred schema.
        """
        ...


def _table_keys_for_selection(
    *,
    catalog: DagCatalog,
    runtime: HamiltonRuntime,
    selection: TableKeySelection,
) -> tuple[str, ...]:
    """Return table keys for the selected targets.

    Parameters
    ----------
    catalog
        DAG catalog containing target definitions.
    runtime
        Hamilton runtime for resolving native targets.
    selection
        Normalized selection criteria.

    Returns
    -------
    tuple[str, ...]
        Selected table keys.

    Raises
    ------
    KeyError
        If explicit targets are requested but missing.
    ValueError
        If any selected target lacks a native implementation.
    """
    targets = list(selection.targets) if selection.targets else None
    module = selection.module
    all_targets = selection.all_targets
    stable = selection.stable

    if targets:
        missing = sorted(t for t in targets if t not in catalog.targets)
        if missing:
            msg = f"Unknown targets: {missing}"
            raise KeyError(msg)
        selected = [catalog.get(t) for t in targets]
    elif module is not None:
        selected = list(catalog.targets_for_module(module))
    else:
        selected = list(catalog.all_targets) if all_targets or not (targets or module) else []

    native_names = native_target_names(runtime)
    missing_native = [t.name for t in selected if t.name not in native_names]
    if missing_native:
        msg = "Targets lack native implementations: " + ", ".join(sorted(missing_native))
        raise ValueError(msg)

    table_keys: list[str] = []
    for target in selected:
        table_keys.extend(
            output.key for output in catalog.table_outputs_by_target.get(target.name, ())
        )

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


def _apply_native_inference(
    *,
    provider: SchemaProvider,
    request: SchemaManifestRequest,
    table_keys: Iterable[str],
    batch_inferer: NativeBatchInferer | None,
    schema_index: SchemaIndex,
) -> SchemaProvider:
    """Wrap the schema provider with native inference when requested.

    Parameters
    ----------
    provider
        Base schema provider used for declared schemas.
    request
        Manifest request controlling inference options.
    table_keys
        Selected table keys for the manifest.
    batch_inferer
        Optional batch inference implementation.
    schema_index
        Schema index used to prioritize DAG-derived outputs.

    Returns
    -------
    SchemaProvider
        Provider potentially wrapped with native inference.
    """
    if isinstance(provider, UnifiedSchemaProvider):
        declared_provider = provider.declared
        schema_index = provider.schema_index
        unified_provider = provider.with_inference(allow_inference=request.infer_native)
    else:
        declared_provider = provider
        unified_provider = UnifiedSchemaProvider(
            declared=declared_provider,
            schema_index=schema_index,
            allow_inference=request.infer_native,
        )

    if not request.infer_native:
        return unified_provider

    selected_table_keys = tuple(sorted(set(table_keys)))
    if not selected_table_keys:
        return unified_provider

    if request.batch_infer_native:
        try:
            batch = infer_table_schemas if batch_inferer is None else batch_inferer
            inferred = batch(selected_table_keys, declared_provider=declared_provider)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            _logger.warning(
                "Batch inference failed; falling back to per-table inference: %s",
                exc,
            )
        else:
            schema_index.prefill_cache(inferred)

    return unified_provider


def _ensure_v2(version: str) -> None:
    if version != V2_SCHEMA_MANIFEST_VERSION:
        msg = f"Unsupported schema manifest version: {version}"
        raise ValueError(msg)


def _resolve_v2_extras(
    *,
    request: SchemaManifestRequest,
    con: DuckDBConnection | None,
    tag_query: TagQuery | None,
) -> tuple[V2Extras | None, str]:
    """Resolve optional v2 manifest extras and the effective version.

    Parameters
    ----------
    request
        Manifest request controlling inclusion of views and artifacts.
    con
        Optional DuckDB connection required for view inference.

    Returns
    -------
    tuple[V2Extras | None, str]
        Extras bundle (or None) and the effective manifest version.

    Raises
    ------
    ValueError
        If view inference is requested without a DuckDB connection.
    """
    views: tuple[TableSchema, ...] = ()
    if request.include_views:
        if con is None:
            msg = "DuckDB connection required for view schema inference"
            raise ValueError(msg)
        views = _collect_view_schemas(con=con, stable=request.stable, tag_query=tag_query)

    artifacts: tuple[ExportArtifact, ...] = ()
    if request.include_artifacts:
        artifacts = _collect_export_artifacts(stable=request.stable)

    _ensure_v2(request.version)
    version = request.version

    extras = V2Extras(views=views, artifacts=artifacts) if views or artifacts else None
    return extras, version


def _collect_view_schemas(
    *,
    con: DuckDBConnection,
    stable: bool,
    tag_query: TagQuery | None,
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
    for view_key in discover_derived_docs_views(tag_query=tag_query):
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

    for contract in iter_contracts(
        settings=ContractResolutionSettings(mode=ContractResolutionMode.FULL)
    ):
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


@dataclass(frozen=True)
class ManifestProvenance:
    """Container for manifest provenance mappings.

    Parameters
    ----------
    table_provenance
        Per-table provenance metadata.
    view_provenance
        Per-view provenance metadata.
    artifact_provenance
        Per-artifact provenance metadata.
    """

    table_provenance: dict[str, TableProvenance]
    view_provenance: dict[str, TableProvenance]
    artifact_provenance: dict[str, ArtifactProvenance]


def _schema_hash_for_table_key(
    table_key: str,
    *,
    known_hashes: dict[str, str],
    provider: SchemaProvider,
) -> str:
    """Resolve the schema hash for a table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    known_hashes
        Cache of precomputed schema hashes keyed by table key.
    provider
        Schema provider used to resolve the table schema if needed.

    Returns
    -------
    str
        Schema hash for the table.

    Raises
    ------
    KeyError
        If the table schema is unknown to the provider.
    """
    cached = known_hashes.get(table_key)
    if cached is not None:
        return cached
    schema = provider.get_table_schema(table_key)
    if schema is None:
        msg = f"Unknown table schema for provenance: {table_key}"
        raise KeyError(msg)
    computed = schema_hash(schema)
    known_hashes[table_key] = computed
    return computed


def _collect_manifest_provenance(
    *,
    tables: tuple[TableSchema, ...],
    views: tuple[TableSchema, ...],
    artifacts: tuple[ExportArtifact, ...],
    provider: SchemaProvider,
    schema_index: SchemaIndex,
) -> ManifestProvenance:
    """Collect provenance metadata for manifest entries.

    Parameters
    ----------
    tables
        Table schemas included in the manifest.
    views
        View schemas included in the manifest.
    artifacts
        Export artifacts included in the manifest.
    provider
        Schema provider used for fallback table resolution.
    schema_index
        Schema index derived from the global target system.

    Returns
    -------
    ManifestProvenance
        Provenance mappings for tables, views, and artifacts.
    """
    table_provenance: dict[str, TableProvenance] = {}
    view_provenance: dict[str, TableProvenance] = {}
    artifact_provenance: dict[str, ArtifactProvenance] = {}
    known_hashes: dict[str, str] = {}

    allow_inference = True
    if isinstance(provider, UnifiedSchemaProvider):
        allow_inference = provider.allow_inference

    for table in tables:
        derivation = schema_index.derivations.get(table.table_key)
        if derivation is None:
            derivation_kind = DECLARED_SOURCE_KIND
            derivation_source = DECLARED_SOURCE_NAME
        else:
            derivation_kind = derivation.kind
            derivation_source = derivation.source
        inference_status = schema_index.inference_status_for(
            table.table_key,
            allow_inference=allow_inference,
        )
        inference_error = (
            schema_index.get_inference_error(table.table_key)
            if inference_status == "error"
            else None
        )
        table_hash = schema_hash(table)
        known_hashes[table.table_key] = table_hash
        table_provenance[table.table_key] = TableProvenance(
            schema_hash=table_hash,
            derivation_kind=derivation_kind,
            derivation_source=derivation_source,
            inference_status=inference_status,
            inference_error=inference_error,
        )

    for view in views:
        view_hash = schema_hash(view)
        view_provenance[view.table_key] = TableProvenance(
            schema_hash=view_hash,
            derivation_kind=VIEW_DERIVATION_KIND,
            derivation_source=VIEW_DERIVATION_SOURCE,
        )

    for artifact in artifacts:
        source_table_keys = (artifact.table_key,) if artifact.table_key is not None else ()
        source_schema_hashes = tuple(
            _schema_hash_for_table_key(
                table_key,
                known_hashes=known_hashes,
                provider=provider,
            )
            for table_key in source_table_keys
        )
        artifact_provenance[artifact.filename] = ArtifactProvenance(
            source_table_keys=source_table_keys,
            source_schema_hashes=source_schema_hashes,
        )

    return ManifestProvenance(
        table_provenance=table_provenance,
        view_provenance=view_provenance,
        artifact_provenance=artifact_provenance,
    )


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
    _ensure_v2(version)
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
    """
    req = request or SchemaManifestRequest()
    _ensure_v2(req.version)
    service = get_target_metadata_service()
    selection = TableKeySelection.from_request(req)
    table_keys = _table_keys_for_selection(
        catalog=service.system.catalog,
        runtime=service.system.runtime,
        selection=selection,
    )
    active_provider = _apply_native_inference(
        provider=provider,
        request=req,
        table_keys=table_keys,
        batch_inferer=batch_inferer,
        schema_index=service.schema_index,
    )
    extras, version = _resolve_v2_extras(
        request=req,
        con=con,
        tag_query=service.system.runtime.tag_query,
    )

    manifest = compile_schema_manifest_for_table_keys(
        table_keys,
        provider=active_provider,
        version=version,
        stable=req.stable,
        extras=extras,
    )
    if not req.include_provenance:
        return manifest

    provenance = _collect_manifest_provenance(
        tables=manifest.tables,
        views=manifest.views,
        artifacts=manifest.artifacts,
        provider=active_provider,
        schema_index=service.schema_index,
    )
    return SchemaManifest(
        version=manifest.version,
        tables=manifest.tables,
        views=manifest.views,
        artifacts=manifest.artifacts,
        table_provenance=provenance.table_provenance,
        view_provenance=provenance.view_provenance,
        artifact_provenance=provenance.artifact_provenance,
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
