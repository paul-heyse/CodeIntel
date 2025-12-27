"""Column operations for function-level feature tables."""

from __future__ import annotations

import pandas as pd
import polars as pl

Column = pd.Series | pl.Series | pl.Expr


def loc_squared(loc: Column) -> Column:
    """Compute squared lines-of-code as a simple feature.

    Returns
    -------
    Column
        Squared LOC feature column.
    """
    return loc * loc


def cyclo_weighted(cyclomatic_complexity: Column) -> Column:
    """Compute weighted cyclomatic complexity.

    Returns
    -------
    Column
        Weighted cyclomatic complexity column.
    """
    return cyclomatic_complexity * 1.0


def loc_cyclo_sum(loc: Column, cyclomatic_complexity: Column) -> Column:
    """Compute combined LOC + cyclo feature.

    Returns
    -------
    Column
        Summed LOC and cyclomatic complexity column.
    """
    return loc + cyclomatic_complexity


def loc_per_cyclo(loc: Column, cyclomatic_complexity: Column) -> Column:
    """Compute LOC per cyclomatic complexity (stabilized).

    Returns
    -------
    Column
        LOC divided by cyclomatic complexity with a +1 stabilizer.
    """
    return loc / (cyclomatic_complexity + 1)


__all__ = [
    "cyclo_weighted",
    "loc_cyclo_sum",
    "loc_per_cyclo",
    "loc_squared",
]
