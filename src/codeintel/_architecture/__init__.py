"""Architecture boundary definitions and enforcement utilities.

This module defines the layered architecture boundaries and provides
utilities for enforcing them at test time. Import boundaries ensure
that database-specific code (like duckdb), graph libraries (like networkx),
and other infrastructure concerns remain isolated in their designated layers.

Usage
-----
The boundaries are primarily used in architecture tests:

    from codeintel._architecture import ALL_BOUNDARIES, check_boundary

    @pytest.mark.parametrize("boundary", ALL_BOUNDARIES, ids=lambda b: b.name)
    def test_import_boundary_respected(boundary) -> None:
        violations = check_boundary(boundary, Path("src"))
        assert not violations, f"{boundary.description}: {violations}"

Adding New Boundaries
---------------------
To add a new boundary, create an ``ImportBoundary`` instance and add it
to ``ALL_BOUNDARIES``:

    NEW_BOUNDARY = ImportBoundary(
        name="new_library",
        restricted_modules=frozenset({"new_library"}),
        allowed_paths=frozenset({"src/codeintel/allowed_layer"}),
        description="new_library imports must be confined to allowed_layer",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImportBoundary:
    """Define an import boundary constraint.

    An import boundary specifies which modules are restricted to certain
    paths within the codebase. Any import of a restricted module outside
    the allowed paths is a violation.

    Parameters
    ----------
    name
        Short identifier for the boundary (used in test output).
    restricted_modules
        Set of module names that are restricted.
    allowed_paths
        Set of path prefixes where restricted modules are allowed.
    description
        Human-readable description of the boundary rule.
    """

    name: str
    restricted_modules: frozenset[str]
    allowed_paths: frozenset[str]
    description: str


# ============================================================================
# Boundary Definitions
# ============================================================================

DUCKDB_BOUNDARY = ImportBoundary(
    name="duckdb",
    restricted_modules=frozenset({"duckdb"}),
    allowed_paths=frozenset({"src/codeintel/storage"}),
    description="DuckDB imports must be confined to storage layer",
)

NETWORKX_BOUNDARY = ImportBoundary(
    name="networkx",
    restricted_modules=frozenset({"networkx", "nx_cugraph"}),
    allowed_paths=frozenset(
        {
            "src/codeintel/graphs",
            "src/codeintel/analytics",
            "src/codeintel/cli/nx_backend",
        }
    ),
    description="NetworkX imports must be confined to graphs/analytics layers",
)

FAISS_BOUNDARY = ImportBoundary(
    name="faiss",
    restricted_modules=frozenset({"faiss"}),
    allowed_paths=frozenset(
        {
            "src/codeintel/serving",
            "src/codeintel/cli",
        }
    ),
    description="FAISS imports must be confined to serving layer",
)

ALL_BOUNDARIES = (
    DUCKDB_BOUNDARY,
    NETWORKX_BOUNDARY,
    FAISS_BOUNDARY,
)


def check_boundary(boundary: ImportBoundary, root: Path) -> list[str]:
    """Check for violations of an import boundary.

    Scan all Python files under ``root`` and check if any import restricted
    modules outside their allowed paths.

    Parameters
    ----------
    boundary
        The boundary constraint to check.
    root
        Root directory to scan.

    Returns
    -------
    list[str]
        List of file paths that violate the boundary.

    Examples
    --------
    >>> from pathlib import Path
    >>> violations = check_boundary(DUCKDB_BOUNDARY, Path("src"))
    >>> assert not violations, f"DuckDB boundary violations: {violations}"
    """
    violations: list[str] = []

    for path in root.rglob("*.py"):
        str_path = str(path)

        # Check if this path is in an allowed location
        if any(allowed in str_path for allowed in boundary.allowed_paths):
            continue

        # Check for restricted imports
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue

        for module in boundary.restricted_modules:
            if f"import {module}" in text:
                violations.append(str_path)
                break

    return violations


__all__ = [
    "ALL_BOUNDARIES",
    "DUCKDB_BOUNDARY",
    "FAISS_BOUNDARY",
    "NETWORKX_BOUNDARY",
    "ImportBoundary",
    "check_boundary",
]
