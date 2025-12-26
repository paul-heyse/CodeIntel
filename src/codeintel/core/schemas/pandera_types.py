"""Shared Pandera dtype mapping for CodeIntel schema validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from pandas.api.extensions import ExtensionDtype

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.core.schemas.primitives import ColumnType

PanderaDtype = type | str | ExtensionDtype

_STRING_DTYPE: PanderaDtype = pd.StringDtype()
_INT_DTYPE: PanderaDtype = pd.Int64Dtype()
_FLOAT_DTYPE: PanderaDtype = pd.Float64Dtype()
_BOOL_DTYPE: PanderaDtype = pd.BooleanDtype()
# DECIMAL(38,0) stores 128-bit IDs; pandas Int64 cannot represent them safely.
_DECIMAL_38_DTYPE: PanderaDtype = object

_COLUMN_TYPE_TO_DTYPE: Mapping[str, PanderaDtype] = {
    "BOOLEAN": _BOOL_DTYPE,
    "INTEGER": _INT_DTYPE,
    "BIGINT": _INT_DTYPE,
    "DOUBLE": _FLOAT_DTYPE,
    "DECIMAL": _FLOAT_DTYPE,
    "DECIMAL(38,0)": _DECIMAL_38_DTYPE,
    "VARCHAR": _STRING_DTYPE,
    "JSON": _STRING_DTYPE,
    "TIMESTAMP": "datetime64[ns]",
    "TIMESTAMPTZ": "datetime64[ns]",
}


def dtype_for_column_type(col_type: ColumnType | str) -> PanderaDtype:
    """Return the Pandera dtype for a canonical column type.

    Parameters
    ----------
    col_type
        Column type literal or string.

    Returns
    -------
    PanderaDtype
        Pandera dtype mapped from the column type.
    """
    normalized = str(col_type).upper()
    if normalized.startswith("DECIMAL("):
        return _COLUMN_TYPE_TO_DTYPE.get("DECIMAL(38,0)", _INT_DTYPE)
    return _COLUMN_TYPE_TO_DTYPE.get(normalized, _STRING_DTYPE)


__all__ = ["PanderaDtype", "dtype_for_column_type"]
