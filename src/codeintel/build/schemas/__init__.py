"""Build-owned schema authority helpers.

This package provides the canonical schema resolution interface for the
build system. Use `get_schema_provider()` as the single entry point for
all schema access.

Examples
--------
>>> from codeintel.build.schemas import get_schema_provider
>>> provider = get_schema_provider()
>>> schema = provider.require_table_schema("analytics.function_metrics")

For row bindings, use `get_row_binding()`:

>>> from codeintel.build.schemas import get_row_binding
>>> binding = get_row_binding("analytics.function_metrics")
>>> binding.table_key
'analytics.function_metrics'

For dataset contracts, use `get_contract_for_table_key()`:

>>> from codeintel.build.schemas import get_contract_for_table_key, is_view
>>> contract = get_contract_for_table_key("analytics.function_metrics")
>>> contract.table_key
'analytics.function_metrics'
>>> is_view("docs.v_function_profile")
True
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.schemas.column_resolution import (
    DeferredColumns,
    deferred_columns_for_table_key,
)
from codeintel.build.schemas.contract_service import (
    ContractProvider,
    ContractResolutionMode,
    ContractResolutionSettings,
    SchemaContractService,
    clear_contract_cache,
    get_contract_provider,
)
from codeintel.build.schemas.diff import (
    ColumnDiff,
    ManifestDiffResult,
    SchemaDiff,
    compute_manifest_diffs,
    compute_schema_diff,
)
from codeintel.build.schemas.infer_duckdb import (
    infer_table_schema_from_ibis,
    infer_view_schema,
    normalize_duckdb_type,
)
from codeintel.build.schemas.json_schema_registry import (
    clear_json_schema_cache,
    compute_json_schema_digest,
    get_json_schema,
    get_json_schema_for_table_schema,
)
from codeintel.build.schemas.manifest import (
    ExportArtifact,
    ExportArtifactKind,
    SchemaManifest,
)
from codeintel.build.schemas.provider_declared import declared_schema_provider
from codeintel.build.schemas.registry import (
    clear_schema_provider_cache,
    get_schema_provider,
    iter_table_schemas,
    require_table_schema,
)
from codeintel.build.schemas.row_registry import (
    clear_row_binding_cache,
    column_names_for_table_key,
    get_row_binding,
    iter_row_bindings,
)
from codeintel.build.schemas.service import (
    clear_schema_service_cache,
    get_schema_service,
)
from codeintel.config.datasets.composites import get_composite_schemas
from codeintel.core.imports.lazy import lazy_import
from codeintel.core.schemas.contract_service import (
    column_order_for_table_key,
    get_contract_for_table_key,
    get_contract_service,
    get_enriched_contract_service,
    is_view,
    iter_contracts,
    iter_contracts_by_table_key,
    overrides_from_output_contract,
)

if TYPE_CHECKING:
    from codeintel.build.schemas.provider_unified import (
        UnifiedSchemaProvider,
        clear_unified_provider_cache,
        unified_schema_provider,
    )

# Lazy imports for unified provider to avoid circular imports.
# provider_unified imports inference_service which triggers a long import chain.
_LAZY_IMPORTS = {
    "UnifiedSchemaProvider": "codeintel.build.schemas.provider_unified",
    "clear_unified_provider_cache": "codeintel.build.schemas.provider_unified",
    "unified_schema_provider": "codeintel.build.schemas.provider_unified",
}


def __getattr__(name: str) -> object:
    """Lazy import for unified provider exports.

    Parameters
    ----------
    name
        Name of the attribute to resolve.

    Returns
    -------
    object
        The requested attribute from the lazy-loaded module.

    Raises
    ------
    AttributeError
        If the attribute is not found in this module or lazy imports.
    """
    if name in _LAZY_IMPORTS:
        module = lazy_import(_LAZY_IMPORTS[name])
        return getattr(module, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "ColumnDiff",
    "ContractProvider",
    "ContractResolutionMode",
    "ContractResolutionSettings",
    "DeferredColumns",
    "ExportArtifact",
    "ExportArtifactKind",
    "ManifestDiffResult",
    "SchemaContractService",
    "SchemaDiff",
    "SchemaManifest",
    "UnifiedSchemaProvider",
    "clear_contract_cache",
    "clear_json_schema_cache",
    "clear_row_binding_cache",
    "clear_schema_provider_cache",
    "clear_schema_service_cache",
    "clear_unified_provider_cache",
    "column_names_for_table_key",
    "column_order_for_table_key",
    "compute_json_schema_digest",
    "compute_manifest_diffs",
    "compute_schema_diff",
    "declared_schema_provider",
    "deferred_columns_for_table_key",
    "get_composite_schemas",
    "get_contract_for_table_key",
    "get_contract_provider",
    "get_contract_service",
    "get_enriched_contract_service",
    "get_json_schema",
    "get_json_schema_for_table_schema",
    "get_row_binding",
    "get_schema_provider",
    "get_schema_service",
    "infer_table_schema_from_ibis",
    "infer_view_schema",
    "is_view",
    "iter_contracts",
    "iter_contracts_by_table_key",
    "iter_row_bindings",
    "iter_table_schemas",
    "normalize_duckdb_type",
    "overrides_from_output_contract",
    "require_table_schema",
    "unified_schema_provider",
]
