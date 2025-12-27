"""Column operations for module-level feature tables."""

from __future__ import annotations

import pandas as pd
import polars as pl

Column = pd.Series | pl.Series | pl.Expr


def module_loc_density(total_loc: Column, function_count: Column) -> Column:
    """Compute LOC per function for modules.

    Returns
    -------
    Column
        LOC divided by function count with a +1 stabilizer.
    """
    return total_loc / (function_count + 1)


def module_risk_scaled(avg_risk_score: Column) -> Column:
    """Compute a scaled risk score for modules.

    Returns
    -------
    Column
        Scaled risk score column.
    """
    return avg_risk_score * 1.0


def module_coverage_gap(module_coverage_ratio: Column) -> Column:
    """Compute coverage gap (1 - coverage_ratio).

    Returns
    -------
    Column
        Coverage gap column.
    """
    return 1 - module_coverage_ratio


__all__ = ["module_coverage_gap", "module_loc_density", "module_risk_scaled"]
