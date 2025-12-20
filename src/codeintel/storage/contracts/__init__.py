"""Storage-owned contract and schema provider surface.

This package exists to keep `codeintel.storage` free of runtime dependencies on
`codeintel.build.*` while still providing access to dataset contracts, schemas,
and validation utilities.
"""

from __future__ import annotations

from codeintel.storage.contracts.json_schema import get_json_schema_for_table_key
from codeintel.storage.contracts.provider import (
    ContractProvider,
    clear_contract_cache,
    get_contract_for_table_key,
    get_contract_provider,
    is_view,
    iter_contracts,
    iter_contracts_by_table_key,
)
from codeintel.storage.contracts.schema_provider import (
    clear_schema_provider_cache,
    get_schema_provider,
    iter_table_schemas,
    require_table_schema,
)

__all__ = [
    "ContractProvider",
    "clear_contract_cache",
    "clear_schema_provider_cache",
    "get_contract_for_table_key",
    "get_contract_provider",
    "get_json_schema_for_table_key",
    "get_schema_provider",
    "is_view",
    "iter_contracts",
    "iter_contracts_by_table_key",
    "iter_table_schemas",
    "require_table_schema",
]
