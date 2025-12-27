"""Column operations for risk-related feature tables."""

from __future__ import annotations

import pandas as pd
import polars as pl

Column = pd.Series | pl.Series | pl.Expr


def risk_scaled(risk_score: Column) -> Column:
    """Compute a scaled risk score for downstream use.

    Returns
    -------
    Column
        Scaled risk score column.
    """
    return risk_score * 1.0


def risk_gap(risk_score: Column, cyclomatic_complexity: Column) -> Column:
    """Compute the gap between complexity and current risk score.

    Returns
    -------
    Column
        Gap between max score and risk score.
    """
    return cyclomatic_complexity - risk_score


__all__ = ["risk_gap", "risk_scaled"]
