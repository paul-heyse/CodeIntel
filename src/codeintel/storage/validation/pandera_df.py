"""Schema-backed Pandera validation helpers for storage boundaries."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pandera.errors import SchemaErrors

from codeintel.core.schemas.pandera_gen import pandera_schema_from_table_schema
from codeintel.storage.contracts.schema_provider import get_schema_provider

if TYPE_CHECKING:
    import pandas as pd
    from pandera import DataFrameSchema

ValidationMode = str

_log = logging.getLogger(__name__)

__all__ = ["ValidationMode", "get_pandera_schema", "validate_df"]


def get_pandera_schema(table_key: str) -> DataFrameSchema | None:
    """Return a generated Pandera schema for a dataset when available.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    DataFrameSchema | None
        Generated schema for the dataset, or None when no declared schema exists.
    """
    schema = get_schema_provider().get_table_schema(table_key)
    if schema is None:
        return None
    return pandera_schema_from_table_schema(table_key=table_key, table_schema=schema)


def validate_df(
    table_key: str,
    df: pd.DataFrame,
    *,
    mode: ValidationMode = "strict",
) -> pd.DataFrame:
    """Validate a DataFrame against the generated Pandera schema when present.

    Parameters
    ----------
    table_key
        Fully qualified dataset name.
    df
        DataFrame to validate.
    mode
        Validation handling: ``"strict"`` raises on errors, ``"warn"`` logs and
        returns the original frame, ``"skip"`` bypasses validation.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame, or the original frame when validation is skipped.

    Raises
    ------
    SchemaErrors
        When validation fails in ``"strict"`` mode.
    """
    schema = get_pandera_schema(table_key)
    if schema is None or mode == "skip":
        return df

    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as exc:
        if mode == "warn":
            _log.warning("Validation warning for %s: %s", table_key, str(exc)[:200])
            return df
        raise
