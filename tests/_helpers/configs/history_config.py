"""Configuration dataclasses for history and timeseries testing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SnapshotSpec:
    """Specification for a minimal function snapshot.

    Attributes
    ----------
    repo : str
        Repository identifier.
    commit : str
        Commit hash.
    goid : int
        GOID hash value.
    rel_path : str
        Relative file path.
    module : str
        Module name.
    qualname : str
        Qualified function name.
    risk_score : float
        Risk score (default 0.5).
    coverage_ratio : float
        Coverage ratio (default 0.5).
    risk_level : str
        Risk level string (default "medium").
    cyclomatic_complexity : int
        Cyclomatic complexity (default 1).
    loc : int
        Lines of code (default 10).
    """

    repo: str
    commit: str
    goid: int
    rel_path: str
    module: str
    qualname: str
    risk_score: float = 0.5
    coverage_ratio: float = 0.5
    risk_level: str = "medium"
    cyclomatic_complexity: int = 1
    loc: int = 10


__all__ = ["SnapshotSpec"]
