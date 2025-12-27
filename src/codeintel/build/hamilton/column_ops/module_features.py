"""Column operations for module-level feature tables."""

from __future__ import annotations

import pandas as pd
import polars as pl

Column = pd.Series | pl.Series | pl.Expr


def module_loc_density(loc: Column, function_count: Column) -> Column:
    """Compute LOC per function for modules."""
    return loc / (function_count + 1)


def module_risk_scaled(risk_score: Column) -> Column:
    """Compute a scaled risk score for modules."""
    return risk_score * 1.0


def module_coverage_gap(coverage_ratio: Column) -> Column:
    """Compute coverage gap (1 - coverage_ratio)."""
    return 1 - coverage_ratio


__all__ = ["module_coverage_gap", "module_loc_density", "module_risk_scaled"]
