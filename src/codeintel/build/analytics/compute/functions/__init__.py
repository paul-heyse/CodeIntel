"""Pure computation modules for function-level analytics.

This package provides pure functions for analyzing Python function AST nodes.
All functions in this package:
- Take AST nodes or primitive data as input
- Return immutable dataclasses as output
- Have no side effects (no I/O, no logging, no exceptions for control flow)

Modules
-------
complexity
    Cyclomatic complexity, nesting depth, statement counts.
typedness
    Parameter annotation statistics.
signatures
    Parameter extraction and signature analysis.
loc
    Lines of code computation (physical and logical).
"""

from __future__ import annotations

from codeintel.build.analytics.compute.functions.complexity import (
    ComplexityMetrics,
    compute_complexity,
)
from codeintel.build.analytics.compute.functions.goids import (
    FunctionGoid,
    FunctionGoidLoader,
    GoidRow,
)
from codeintel.build.analytics.compute.functions.loc import (
    LinesOfCode,
    compute_loc,
)
from codeintel.build.analytics.compute.functions.signatures import (
    FunctionSignature,
    extract_signature,
)
from codeintel.build.analytics.compute.functions.typedness import (
    ParamStats,
    compute_param_stats,
)

__all__ = [
    "ComplexityMetrics",
    "FunctionGoid",
    "FunctionGoidLoader",
    "FunctionSignature",
    "GoidRow",
    "LinesOfCode",
    "ParamStats",
    "compute_complexity",
    "compute_loc",
    "compute_param_stats",
    "extract_signature",
]
