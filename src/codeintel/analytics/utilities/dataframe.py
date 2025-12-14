"""DataFrame utility functions for analytics pipelines.

This module provides common DataFrame operations used across analytics modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import pandas as pd


def to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert DataFrame rows into a list of dictionaries.

    Parameters
    ----------
    df
        The pandas DataFrame to convert.

    Returns
    -------
    list[dict[str, Any]]
        Records returned by ``DataFrame.to_dict(orient="records")``.
    """
    return cast("list[dict[str, Any]]", df.to_dict(orient="records"))


__all__ = ["to_records"]
