"""Mock function catalog implementations for testing.

This module provides mock implementations of FunctionCatalog and
FunctionCatalogProvider that satisfy the catalog protocols, enabling
tests without real database connections or complex setup.

The mocks follow the Testing Charter:
- They implement the same interface as production code
- They preserve key invariants (span lookups, URN resolution)
- They can be used in dev/staging environments

Example
-------
>>> from tests._helpers.fakes.function_catalogs import MockFunctionCatalog
>>>
>>> # Create a mock with sample functions
>>> catalog = MockFunctionCatalog(
...     functions=[
...         MockFunctionMeta(goid=1, urn="urn:test:func1", rel_path="mod.py"),
...     ]
... )
>>>
>>> # Use with CatalogProvider
>>> from codeintel.analytics.resources.catalog import CatalogProvider
>>> provider = CatalogProvider()
>>> provider.set_preloaded(catalog)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from codeintel.graphs.catalog import FunctionCatalog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_LANGUAGE: Final[str] = "python"
DEFAULT_KIND: Final[str] = "function"


# ---------------------------------------------------------------------------
# Mock Data Classes
# ---------------------------------------------------------------------------


@dataclass
class MockFunctionMeta:
    """Mock function metadata for testing.

    Provides a simplified representation of function metadata
    for constructing test catalogs.

    Attributes
    ----------
    goid
        Global object identifier.
    urn
        Unique resource name for the function.
    rel_path
        Relative path to the source file.
    qualname
        Qualified name of the function.
    start_line
        Starting line number in source.
    end_line
        Ending line number in source.
    language
        Programming language (default: python).
    kind
        Function kind (default: function).
    """

    goid: int
    urn: str = ""
    rel_path: str = "module.py"
    qualname: str = "function"
    start_line: int = 1
    end_line: int = 10
    language: str = DEFAULT_LANGUAGE
    kind: str = DEFAULT_KIND

    def __post_init__(self) -> None:
        """Set default URN if not provided."""
        if not self.urn:
            self.urn = f"urn:test:{self.rel_path}#{self.qualname}"


@dataclass
class MockFunctionSpan:
    """Mock function span for testing.

    Represents a function's location in source code.

    Attributes
    ----------
    goid
        Global object identifier.
    rel_path
        Relative path to the source file.
    qualname
        Qualified name of the function.
    start_line
        Starting line number.
    end_line
        Ending line number.
    """

    goid: int
    rel_path: str = "module.py"
    qualname: str = "function"
    start_line: int = 1
    end_line: int = 10


# ---------------------------------------------------------------------------
# Mock Catalog Classes
# ---------------------------------------------------------------------------


@dataclass
class MockFunctionCatalog:
    """Mock FunctionCatalog for testing analytics plugins.

    Provides configurable function catalog responses for testing
    catalog access patterns and function lookups without database I/O.

    Implements the FunctionCatalogProvider protocol interface so it can
    be used directly or wrapped by CatalogProvider.

    Attributes
    ----------
    functions
        List of mock function metadata.
    module_by_path
        Mapping of file paths to module names.
    function_spans
        List of function spans (derived from functions if not provided).

    Examples
    --------
    Create a mock with sample functions:

    >>> catalog = MockFunctionCatalog(
    ...     functions=[
    ...         MockFunctionMeta(goid=1, qualname="main"),
    ...         MockFunctionMeta(goid=2, qualname="helper"),
    ...     ]
    ... )
    >>> catalog.urn_for_goid(1)
    'urn:test:module.py#main'

    Create with module mappings:

    >>> catalog = MockFunctionCatalog(
    ...     module_by_path={"src/utils.py": "src.utils"},
    ... )
    """

    functions: list[MockFunctionMeta] = field(default_factory=list)
    module_by_path: dict[str, str] = field(default_factory=dict)
    function_spans: list[MockFunctionSpan] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Build spans from functions if not explicitly provided."""
        if not self.function_spans and self.functions:
            self.function_spans = [
                MockFunctionSpan(
                    goid=fn.goid,
                    rel_path=fn.rel_path,
                    qualname=fn.qualname,
                    start_line=fn.start_line,
                    end_line=fn.end_line,
                )
                for fn in self.functions
            ]

        # Build internal indexes
        self._goid_to_urn: dict[int, str] = {fn.goid: fn.urn for fn in self.functions}
        self._urn_to_goid: dict[str, int] = {fn.urn: fn.goid for fn in self.functions}

    def catalog(self) -> FunctionCatalog:
        """Return self as the catalog.

        This cast is safe because MockFunctionCatalog implements all the
        attributes that FunctionCatalog consumers need (function_spans,
        urn_for_goid, lookup_goid, module_by_path).

        Returns
        -------
        FunctionCatalog
            Self, cast to FunctionCatalog for protocol compatibility.
        """
        return cast("FunctionCatalog", self)

    def get(self) -> MockFunctionCatalog:
        """Return self for LazyResource-like interface.

        Returns
        -------
        MockFunctionCatalog
            Self reference.
        """
        return self

    def urn_for_goid(self, goid: int) -> str | None:
        """Return URN for a GOID.

        Parameters
        ----------
        goid
            Global object identifier to look up.

        Returns
        -------
        str | None
            URN if found, None otherwise.
        """
        return self._goid_to_urn.get(goid)

    def goid_for_urn(self, urn: str) -> int | None:
        """Return GOID for a URN.

        Parameters
        ----------
        urn
            URN to look up.

        Returns
        -------
        int | None
            GOID if found, None otherwise.
        """
        return self._urn_to_goid.get(urn)

    def lookup_goid(
        self,
        rel_path: str,
        start_line: int,
        end_line: int | None,
        qualname: str | None,
    ) -> int | None:
        """Resolve GOID from span information.

        Parameters
        ----------
        rel_path
            Relative path to the source file.
        start_line
            Starting line number.
        end_line
            Ending line number (optional, unused in mock).
        qualname
            Qualified name (optional).

        Returns
        -------
        int | None
            GOID if a matching function is found, None otherwise.
        """
        for fn in self.functions:
            path_matches = fn.rel_path == rel_path and fn.start_line == start_line
            end_matches = end_line is None or fn.end_line == end_line
            name_matches = qualname is None or fn.qualname == qualname
            if path_matches and end_matches and name_matches:
                return fn.goid
        return None

    def get_all_goids(self) -> list[int]:
        """Return all GOIDs in the catalog.

        Returns
        -------
        list[int]
            List of all function GOIDs.
        """
        return [fn.goid for fn in self.functions]

    def get_functions_by_path(self, rel_path: str) -> list[MockFunctionMeta]:
        """Return all functions in a specific file.

        Parameters
        ----------
        rel_path
            Relative path to the source file.

        Returns
        -------
        list[MockFunctionMeta]
            Functions in the specified file.
        """
        return [fn for fn in self.functions if fn.rel_path == rel_path]


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------


def create_mock_catalog_empty() -> MockFunctionCatalog:
    """Create an empty MockFunctionCatalog.

    Returns
    -------
    MockFunctionCatalog
        Empty catalog with no functions.
    """
    return MockFunctionCatalog()


def create_mock_catalog_with_functions(
    count: int = 3,
    *,
    rel_path: str = "module.py",
    module_name: str = "module",
) -> MockFunctionCatalog:
    """Create a MockFunctionCatalog with sample functions.

    Parameters
    ----------
    count
        Number of functions to create.
    rel_path
        Relative path for all functions.
    module_name
        Module name for path mapping.

    Returns
    -------
    MockFunctionCatalog
        Catalog with sample functions.

    Examples
    --------
    >>> catalog = create_mock_catalog_with_functions(3)
    >>> len(catalog.functions)
    3
    """
    functions = [
        MockFunctionMeta(
            goid=1000 + i,
            urn=f"urn:test:{rel_path}#func_{i}",
            rel_path=rel_path,
            qualname=f"func_{i}",
            start_line=10 * i + 1,
            end_line=10 * i + 9,
        )
        for i in range(count)
    ]

    return MockFunctionCatalog(
        functions=functions,
        module_by_path={rel_path: module_name},
    )


def create_mock_catalog_multi_file(
    files: dict[str, int] | None = None,
) -> MockFunctionCatalog:
    """Create a MockFunctionCatalog with functions across multiple files.

    Parameters
    ----------
    files
        Mapping of relative paths to function counts.
        Defaults to {"src/main.py": 2, "src/utils.py": 3}.

    Returns
    -------
    MockFunctionCatalog
        Catalog with functions in multiple files.

    Examples
    --------
    >>> catalog = create_mock_catalog_multi_file({"a.py": 1, "b.py": 2})
    >>> len(catalog.functions)
    3
    """
    if files is None:
        files = {"src/main.py": 2, "src/utils.py": 3}

    functions: list[MockFunctionMeta] = []
    module_by_path: dict[str, str] = {}
    goid_counter = 1000

    for rel_path, count in files.items():
        # Derive module name from path
        module_name = rel_path.replace("/", ".").replace(".py", "")
        module_by_path[rel_path] = module_name

        for i in range(count):
            functions.append(
                MockFunctionMeta(
                    goid=goid_counter,
                    urn=f"urn:test:{rel_path}#func_{i}",
                    rel_path=rel_path,
                    qualname=f"func_{i}",
                    start_line=10 * i + 1,
                    end_line=10 * i + 9,
                )
            )
            goid_counter += 1

    return MockFunctionCatalog(
        functions=functions,
        module_by_path=module_by_path,
    )


def create_mock_catalog_realistic() -> MockFunctionCatalog:
    """Create a MockFunctionCatalog with realistic test data.

    Returns a catalog with varied function types commonly seen in
    real codebases: public functions, private helpers, class methods,
    async functions, etc.

    Returns
    -------
    MockFunctionCatalog
        Catalog with realistic function patterns.
    """
    functions = [
        # Public entry point
        MockFunctionMeta(
            goid=1001,
            urn="urn:test:main.py#main",
            rel_path="main.py",
            qualname="main",
            start_line=10,
            end_line=25,
        ),
        # Public function
        MockFunctionMeta(
            goid=1002,
            urn="urn:test:utils.py#process_data",
            rel_path="utils.py",
            qualname="process_data",
            start_line=5,
            end_line=20,
        ),
        # Private helper
        MockFunctionMeta(
            goid=1003,
            urn="urn:test:utils.py#_validate",
            rel_path="utils.py",
            qualname="_validate",
            start_line=25,
            end_line=35,
        ),
        # Class method
        MockFunctionMeta(
            goid=1004,
            urn="urn:test:models.py#User.save",
            rel_path="models.py",
            qualname="User.save",
            start_line=50,
            end_line=70,
        ),
        # Static method
        MockFunctionMeta(
            goid=1005,
            urn="urn:test:models.py#User.from_dict",
            rel_path="models.py",
            qualname="User.from_dict",
            start_line=75,
            end_line=85,
        ),
        # Async function
        MockFunctionMeta(
            goid=1006,
            urn="urn:test:api.py#fetch_data",
            rel_path="api.py",
            qualname="fetch_data",
            start_line=10,
            end_line=30,
        ),
    ]

    return MockFunctionCatalog(
        functions=functions,
        module_by_path={
            "main.py": "main",
            "utils.py": "utils",
            "models.py": "models",
            "api.py": "api",
        },
    )


__all__ = [
    "MockFunctionCatalog",
    "MockFunctionMeta",
    "MockFunctionSpan",
    "create_mock_catalog_empty",
    "create_mock_catalog_multi_file",
    "create_mock_catalog_realistic",
    "create_mock_catalog_with_functions",
]
