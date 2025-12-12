"""Hamilton contract integration for data quality validation.

This package provides:
- Pandera schema integration with SCHEMA_REGISTRY
- @check_output integration for Hamilton nodes
- Contract validation utilities
"""

from __future__ import annotations

from codeintel.build.hamilton.contracts.pandera_hook import (
    get_pandera_schema,
    validate_dataframe,
    validate_dataset_ref,
    with_contract,
)

__all__ = [
    "get_pandera_schema",
    "validate_dataframe",
    "validate_dataset_ref",
    "with_contract",
]
