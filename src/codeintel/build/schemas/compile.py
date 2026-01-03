"""Schema compilation utilities for producing schema manifests.

This module provides functions for compiling SchemaManifest objects from
build target selections. The v2 format extends compilation to include
view schemas and export artifact specifications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from codeintel.build.schemas.contract_service import (
    ContractResolutionMode,
    ContractResolutionSettings,
    iter_contracts,
)
from codeintel.build.schemas.manifest import (
    ArtifactProvenance,
    ExportArtifact,
    SchemaManifest,
    TableProvenance,
)
from codeintel.build.schemas.provider_unified import (
    UnifiedSchemaProvider,
)
from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.tag_filters import tf_schema_tables
from codeintel.core.schemas.declared import declared_schema_provider
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.resolution import ResolvedSchemaProvider
from codeintel.core.views.inventory import discover_derived_docs_views

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.hamilton.dag_catalog import DagCatalog
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.schemas.schema_index import SchemaIndex
    from codeintel.build.targets import TargetModule
    from codeintel.core.hamilton.tag_query import TagQuery
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.provider import SchemaProvider

_logger = logging.getLogger(__name__)

V2_SCHEMA_MANIFEST_VERSION = "v2"
DEFAULT_SCHEMA_MANIFEST_VERSION = V2_SCHEMA_MANIFEST_VERSION
DECLARED_SOURCE_KIND = "declared_source"
DECLARED_SOURCE_NAME = "declared"
VIEW_DERIVATION_KIND = "view_inferred"
VIEW_DERIVATION_SOURCE = "schema_provider"


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
        When True, include view schemas discovered from the schema provider.
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


@dataclass(frozen=True)
class SchemaManifestContext:
    """Context required to compile schema manifests.

    Attributes
    ----------
    catalog
        DAG catalog containing target metadata.
    schema_index
        Schema index used for inference and provenance.
    tag_query
        TagQuery helper used for table discovery.
    """

    catalog: DagCatalog
    schema_index: SchemaIndex
    tag_query: TagQuery


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


@runtime_checkable
class _SupportsInference(Protocol):
    """Protocol for schema providers that can toggle inference."""

    def with_inference(self, *, allow_inference: bool) -> SchemaProvider:
        """Return a schema provider with inference enabled or disabled."""
        ...


def _table_keys_for_selection(
    *,
    catalog: DagCatalog,
    selection: TableKeySelection,
    tag_query: TagQuery,
) -> tuple[str, ...]:
    """Return table keys for the selected targets.

    Parameters
    ----------
    catalog
        DAG catalog containing target definitions.
    selection
        Normalized selection criteria.
    tag_query
        TagQuery helper for tag-filter-based table discovery.

    Returns
    -------
    tuple[str, ...]
        Selected table keys.

    Raises
    ------
    KeyError
        If explicit targets are requested but missing.
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

    target_names = {target.name for target in selected}
    return _table_keys_from_tag_query(
        tag_query=tag_query,
        target_names=target_names,
        stable=stable,
    )


def _table_keys_from_tag_query(
    *,
    tag_query: TagQuery,
    target_names: set[str],
    stable: bool,
) -> tuple[str, ...]:
    table_keys: list[str] = []
    for variable in tag_query.query(tf_schema_tables()):
        tags = getattr(variable, "tags", None)
        if not isinstance(tags, dict):
            continue
        table_key = tags.get(ht.TAG_TABLE_KEY)
        if not isinstance(table_key, str) or not table_key:
            continue
        target = tags.get(ht.TAG_TARGET)
        if not isinstance(target, str) or target not in target_names:
            continue
        table_keys.append(table_key)

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


@dataclass(frozen=True)
class _InferenceInputs:
    """Inputs for native schema inference during manifest compilation."""

    provider: SchemaProvider
    request: SchemaManifestRequest
    table_keys: Iterable[str]
    schema_index: SchemaIndex


def _apply_native_inference(
    inputs: _InferenceInputs,
    *,
    batch_inferer: NativeBatchInferer | None,
) -> SchemaProvider:
    """Wrap the schema provider with native inference when requested.

    Parameters
    ----------
    inputs
        Bundled inputs required for inference decisions.
    batch_inferer
        Optional batch inference implementation.

    Returns
    -------
    SchemaProvider
        Provider potentially wrapped with native inference.
    """
    provider = inputs.provider
    request = inputs.request
    schema_index = inputs.schema_index
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

    selected_table_keys = tuple(sorted(set(inputs.table_keys)))
    if not selected_table_keys:
        return unified_provider

    if request.batch_infer_native:
        try:
            batch = (
                _schema_index_batch_inferer(schema_index)
                if batch_inferer is None
                else batch_inferer
            )
            inferred = batch(selected_table_keys, declared_provider=declared_provider)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            _logger.warning(
                "Batch inference failed; falling back to per-table inference: %s",
                exc,
            )
        else:
            schema_index.prefill_cache(inferred)

    return unified_provider


def _schema_index_batch_inferer(schema_index: SchemaIndex) -> NativeBatchInferer:
    def _resolve_env() -> BuildEnv | None:
        provider = schema_index.env_provider
        if provider is None:
            return None
        return provider()

    def _infer(
        table_keys: Iterable[str],
        *,
        declared_provider: SchemaProvider,
    ) -> dict[str, TableSchema]:
        return schema_index.inference_service.infer_table_schemas(
            table_keys,
            declared_provider=declared_provider,
            env=_resolve_env(),
        )

    return _infer


def _ensure_v2(version: str) -> None:
    if version != V2_SCHEMA_MANIFEST_VERSION:
        msg = f"Unsupported schema manifest version: {version}"
        raise ValueError(msg)


def _resolve_v2_extras(
    *,
    request: SchemaManifestRequest,
    provider: SchemaProvider,
    tag_query: TagQuery | None,
) -> tuple[V2Extras | None, str]:
    """Resolve optional v2 manifest extras and the effective version.

    Parameters
    ----------
    request
        Manifest request controlling inclusion of views and artifacts.
    provider
        Schema provider used to resolve view schemas.
    tag_query
        Optional TagQuery helper for view discovery.

    Returns
    -------
    tuple[V2Extras | None, str]
        Extras bundle (or None) and the effective manifest version.

    """
    views: tuple[TableSchema, ...] = ()
    if request.include_views:
        views = _collect_view_schemas(
            provider=provider,
            stable=request.stable,
            tag_query=tag_query,
        )

    artifacts: tuple[ExportArtifact, ...] = ()
    if request.include_artifacts:
        artifacts = _collect_export_artifacts(stable=request.stable)

    _ensure_v2(request.version)
    version = request.version

    extras = V2Extras(views=views, artifacts=artifacts) if views or artifacts else None
    return extras, version


def _collect_view_schemas(
    *,
    provider: SchemaProvider,
    stable: bool,
    tag_query: TagQuery | None,
) -> tuple[TableSchema, ...]:
    """Collect schemas for all known docs views.

    Iterates through discovered docs views and resolves any registered schemas
    from the provider. Views without schemas are skipped.

    Parameters
    ----------
    provider
        Schema provider used to resolve view schemas.
    stable
        When True, sort views deterministically by table_key.
    tag_query
        Optional TagQuery helper for view discovery.

    Returns
    -------
    tuple[TableSchema, ...]
        Inferred view schemas.
    """
    views: list[TableSchema] = []
    provider = _non_inferable_provider(provider)
    for view_key in discover_derived_docs_views(tag_query=tag_query):
        view_schema = provider.get_table_schema(view_key)
        if view_schema is None:
            _logger.debug("Skipping view %s (no registered schema)", view_key)
            continue
        views.append(view_schema)

    if stable:
        views = sorted(views, key=lambda v: v.table_key)
    return tuple(views)


def _non_inferable_provider(provider: SchemaProvider) -> SchemaProvider:
    if isinstance(provider, ResolvedSchemaProvider):
        fallback = _non_inferable_provider(provider.fallback_provider)
        if fallback is provider.fallback_provider:
            return provider
        return ResolvedSchemaProvider(
            observation_provider=provider.observation_provider,
            fallback_provider=fallback,
        )
    if isinstance(provider, UnifiedSchemaProvider):
        return UnifiedSchemaProvider(
            declared=declared_schema_provider(),
            schema_index=provider.schema_index,
            allow_inference=False,
            fallback_to_override_on_error=provider.fallback_to_override_on_error,
        )
    if isinstance(provider, _SupportsInference):
        return provider.with_inference(allow_inference=False)
    return provider


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

    allow_inference = (
        provider.allow_inference if isinstance(provider, UnifiedSchemaProvider) else True
    )

    for table in tables:
        table_provenance[table.table_key] = _table_provenance_entry(
            table,
            schema_index=schema_index,
            allow_inference=allow_inference,
            known_hashes=known_hashes,
        )

    for view in views:
        view_provenance[view.table_key] = _view_provenance_entry(view)

    for artifact in artifacts:
        artifact_provenance[artifact.filename] = _artifact_provenance_entry(
            artifact,
            known_hashes=known_hashes,
            provider=provider,
        )

    return ManifestProvenance(
        table_provenance=table_provenance,
        view_provenance=view_provenance,
        artifact_provenance=artifact_provenance,
    )


def _table_provenance_entry(
    table: TableSchema,
    *,
    schema_index: SchemaIndex,
    allow_inference: bool,
    known_hashes: dict[str, str],
) -> TableProvenance:
    derivation = schema_index.derivations.get(table.table_key)
    if derivation is None:
        derivation_kind = DECLARED_SOURCE_KIND
        derivation_source = DECLARED_SOURCE_NAME
        producer_target = None
        producer_module = None
        producer_version = None
    else:
        derivation_kind = derivation.kind
        derivation_source = derivation.source
        producer_target = derivation.source
        producer_module = derivation.source_module
        producer_version = derivation.source_version
    inference_status = schema_index.inference_status_for(
        table.table_key,
        allow_inference=allow_inference,
    )
    inference_error = (
        schema_index.get_inference_error(table.table_key) if inference_status == "error" else None
    )
    table_hash = schema_hash(table)
    known_hashes[table.table_key] = table_hash
    return TableProvenance(
        schema_hash=table_hash,
        derivation_kind=derivation_kind,
        derivation_source=derivation_source,
        inference_status=inference_status,
        inference_error=inference_error,
        producer_target=producer_target,
        producer_module=producer_module,
        producer_version=producer_version,
    )


def _view_provenance_entry(view: TableSchema) -> TableProvenance:
    view_hash = schema_hash(view)
    return TableProvenance(
        schema_hash=view_hash,
        derivation_kind=VIEW_DERIVATION_KIND,
        derivation_source=VIEW_DERIVATION_SOURCE,
    )


def _artifact_provenance_entry(
    artifact: ExportArtifact,
    *,
    known_hashes: dict[str, str],
    provider: SchemaProvider,
) -> ArtifactProvenance:
    source_table_keys = (artifact.table_key,) if artifact.table_key is not None else ()
    source_schema_hashes = tuple(
        _schema_hash_for_table_key(
            table_key,
            known_hashes=known_hashes,
            provider=provider,
        )
        for table_key in source_table_keys
    )
    return ArtifactProvenance(
        source_table_keys=source_table_keys,
        source_schema_hashes=source_schema_hashes,
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
    context: SchemaManifestContext,
    request: SchemaManifestRequest | None = None,
    batch_inferer: NativeBatchInferer | None = None,
) -> SchemaManifest:
    """Compile a schema manifest for a build target selection.

    Parameters
    ----------
    provider
        Base schema provider used for declared schemas.
    context
        Manifest compilation context (catalog, schema index, tag query).
    request
        Selection and options for manifest compilation. When None, uses defaults.
    batch_inferer
        Optional callable used to batch-infer native table schemas in a single pass.

    Returns
    -------
    SchemaManifest
        Compiled schema manifest.
    """
    req = request or SchemaManifestRequest()
    _ensure_v2(req.version)
    selection = TableKeySelection.from_request(req)
    table_keys = _table_keys_for_selection(
        catalog=context.catalog,
        selection=selection,
        tag_query=context.tag_query,
    )
    active_provider = _apply_native_inference(
        _InferenceInputs(
            provider=provider,
            request=req,
            table_keys=table_keys,
            schema_index=context.schema_index,
        ),
        batch_inferer=batch_inferer,
    )
    extras, version = _resolve_v2_extras(
        request=req,
        provider=active_provider,
        tag_query=context.tag_query,
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
        schema_index=context.schema_index,
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
    "SchemaManifestContext",
    "SchemaManifestRequest",
    "V2Extras",
    "compile_schema_manifest",
    "compile_schema_manifest_for_table_keys",
]
