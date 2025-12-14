"""Dataset schema describing reflected serving operation contracts."""

from __future__ import annotations

from typing import Final

import pandas as pd
from pandera import Check, Column, DataFrameSchema

from codeintel.config.datasets.schema import DatasetMetadata, DatasetSchema

OPERATION_CONTRACT_TABLE_KEY: Final = "analytics.operation_contracts"
TRANSPORT_KINDS: Final = (
    "protocol_service",
    "protocol_backend",
    "service",
    "backend",
)


def _is_str_tuple(value: object) -> bool:
    """Return True when ``value`` is a tuple containing only strings.

    Parameters
    ----------
    value
        Value to inspect.

    Returns
    -------
    bool
        ``True`` when value is a tuple containing only strings.
    """
    return isinstance(value, tuple) and all(isinstance(item, str) for item in value)


def _is_tuple_of_str(series: pd.Series) -> bool:
    """Return True when all values are tuples of strings.

    Parameters
    ----------
    series
        Pandas Series holding candidate tuple values.

    Returns
    -------
    bool
        ``True`` when every value is a tuple containing only strings.
    """
    return bool(series.apply(_is_str_tuple).all())


def build_operation_contract_schema() -> DatasetSchema:
    """Create the DatasetSchema describing operation contract rows.

    Returns
    -------
    DatasetSchema
        Schema describing the operation contract table.
    """
    schema = DataFrameSchema(
        {
            "component": Column(
                pd.StringDtype(),
                checks=Check(lambda series: series.str.len() > 0),
                coerce=True,
            ),
            "transport": Column(
                pd.StringDtype(),
                checks=Check.isin(TRANSPORT_KINDS),
                coerce=True,
            ),
            "method": Column(
                pd.StringDtype(),
                checks=Check(lambda series: series.str.len() > 0),
                coerce=True,
            ),
            "args": Column(
                object,
                checks=Check(_is_tuple_of_str, element_wise=False),
                coerce=False,
            ),
            "arg_types": Column(
                object,
                checks=Check(_is_tuple_of_str, element_wise=False),
                coerce=False,
            ),
            "return_type": Column(
                pd.StringDtype(),
                checks=Check(lambda series: series.str.len() > 0),
                coerce=True,
            ),
            "scope_type": Column(
                pd.StringDtype(),
                nullable=True,
                coerce=True,
            ),
        },
        strict=True,
        coerce=True,
        name=OPERATION_CONTRACT_TABLE_KEY,
    )

    metadata = DatasetMetadata(
        description="Canonical and reflected serving operation contracts.",
        owner="serving",
        family="analytics",
        tags=frozenset({"contracts", "serving", "tooling"}),
    )

    return DatasetSchema(
        name=OPERATION_CONTRACT_TABLE_KEY,
        pandera_schema=schema,
        metadata=metadata,
    )


__all__ = [
    "OPERATION_CONTRACT_TABLE_KEY",
    "TRANSPORT_KINDS",
    "build_operation_contract_schema",
]
