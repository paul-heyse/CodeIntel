"""Tests for catalog resource provider.

This module tests the CatalogResource provider from
`codeintel.graphs.resources.catalog`, including:

- Resource protocol compliance
- Catalog delegation
- Caching behavior
- FunctionSpanData conversion
"""

from __future__ import annotations

from typing import Final

import pytest

from codeintel.graphs.catalog import FunctionCatalog, FunctionMeta
from codeintel.graphs.ports.catalog import CatalogPort, FunctionSpanData
from codeintel.graphs.resources.catalog import CatalogResource

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESOURCE_NAME: Final[str] = "catalog"
TEST_PATH_A: Final[str] = "pkg/module_a.py"
TEST_PATH_B: Final[str] = "pkg/module_b.py"
TEST_MODULE_A: Final[str] = "pkg.module_a"
TEST_MODULE_B: Final[str] = "pkg.module_b"
GOID_1: Final[int] = 1001
GOID_2: Final[int] = 1002
GOID_3: Final[int] = 1003
URN_1: Final[str] = "urn:test:func1"
URN_2: Final[str] = "urn:test:func2"
URN_3: Final[str] = "urn:test:func3"
QUALNAME_1: Final[str] = "func1"
QUALNAME_2: Final[str] = "ClassA.method1"
QUALNAME_3: Final[str] = "func2"
START_LINE_1: Final[int] = 10
START_LINE_2: Final[int] = 20
START_LINE_3: Final[int] = 5
END_LINE_1: Final[int] = 15
END_LINE_2: Final[int] = 30
END_LINE_3: Final[int] = 12
EXPECTED_TOTAL_SPANS: Final[int] = 3
EXPECTED_PATH_COUNT: Final[int] = 2
EXPECTED_SPANS_IN_PATH_A: Final[int] = 2
EXPECTED_SPANS_IN_PATH_B: Final[int] = 1


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_functions() -> list[FunctionMeta]:
    """Create sample function metadata for testing.

    Returns
    -------
    list[FunctionMeta]
        List of sample function metadata.
    """
    return [
        FunctionMeta(
            goid=GOID_1,
            rel_path=TEST_PATH_A,
            qualname=QUALNAME_1,
            start_line=START_LINE_1,
            end_line=END_LINE_1,
            urn=URN_1,
        ),
        FunctionMeta(
            goid=GOID_2,
            rel_path=TEST_PATH_A,
            qualname=QUALNAME_2,
            start_line=START_LINE_2,
            end_line=END_LINE_2,
            urn=URN_2,
        ),
        FunctionMeta(
            goid=GOID_3,
            rel_path=TEST_PATH_B,
            qualname=QUALNAME_3,
            start_line=START_LINE_3,
            end_line=END_LINE_3,
            urn=URN_3,
        ),
    ]


@pytest.fixture
def module_by_path() -> dict[str, str]:
    """Create sample module mapping.

    Returns
    -------
    dict[str, str]
        Sample module path mapping.
    """
    return {
        TEST_PATH_A: TEST_MODULE_A,
        TEST_PATH_B: TEST_MODULE_B,
    }


@pytest.fixture
def sample_catalog(
    sample_functions: list[FunctionMeta],
    module_by_path: dict[str, str],
) -> FunctionCatalog:
    """Create a sample FunctionCatalog.

    Returns
    -------
    FunctionCatalog
        Sample function catalog.
    """
    return FunctionCatalog(functions=sample_functions, module_by_path=module_by_path)


@pytest.fixture
def catalog_resource(sample_catalog: FunctionCatalog) -> CatalogResource:
    """Create a CatalogResource wrapping the sample catalog.

    Returns
    -------
    CatalogResource
        Resource wrapping sample catalog.
    """
    return CatalogResource(catalog=sample_catalog)


@pytest.fixture
def empty_catalog() -> FunctionCatalog:
    """Create an empty FunctionCatalog.

    Returns
    -------
    FunctionCatalog
        Empty function catalog.
    """
    return FunctionCatalog(functions=[], module_by_path={})


@pytest.fixture
def empty_catalog_resource(empty_catalog: FunctionCatalog) -> CatalogResource:
    """Create a CatalogResource wrapping an empty catalog.

    Returns
    -------
    CatalogResource
        Resource wrapping empty catalog.
    """
    return CatalogResource(catalog=empty_catalog)


# ===========================================================================
# Resource Protocol Tests
# ===========================================================================


def test_catalog_resource_name_constant() -> None:
    """CatalogResource.RESOURCE_NAME is 'catalog'."""
    assert CatalogResource.RESOURCE_NAME == RESOURCE_NAME


def test_catalog_resource_name_property(catalog_resource: CatalogResource) -> None:
    """CatalogResource.resource_name returns RESOURCE_NAME."""
    assert catalog_resource.resource_name == RESOURCE_NAME


def test_catalog_resource_get_returns_self(catalog_resource: CatalogResource) -> None:
    """CatalogResource.get() returns self as CatalogPort."""
    result = catalog_resource.get()
    assert result is catalog_resource


def test_catalog_resource_implements_catalog_port(catalog_resource: CatalogResource) -> None:
    """CatalogResource implements CatalogPort protocol."""
    assert isinstance(catalog_resource, CatalogPort)


# ===========================================================================
# Function Spans Tests
# ===========================================================================


def test_function_spans_returns_all_spans(catalog_resource: CatalogResource) -> None:
    """Function_spans returns all function spans."""
    spans = catalog_resource.function_spans
    assert len(spans) == EXPECTED_TOTAL_SPANS


def test_function_spans_returns_span_data(catalog_resource: CatalogResource) -> None:
    """function_spans returns FunctionSpanData objects."""
    spans = catalog_resource.function_spans
    for span in spans:
        assert isinstance(span, FunctionSpanData)


def test_function_spans_contains_expected_goids(catalog_resource: CatalogResource) -> None:
    """function_spans contains expected GOIDs."""
    spans = catalog_resource.function_spans
    goids = {span.goid for span in spans}
    assert goids == {GOID_1, GOID_2, GOID_3}


def test_function_spans_contains_urns(catalog_resource: CatalogResource) -> None:
    """function_spans includes URNs."""
    spans = catalog_resource.function_spans
    urns = {span.urn for span in spans}
    assert URN_1 in urns
    assert URN_2 in urns
    assert URN_3 in urns


def test_function_spans_empty_catalog(empty_catalog_resource: CatalogResource) -> None:
    """function_spans returns empty for empty catalog."""
    spans = empty_catalog_resource.function_spans
    assert len(spans) == 0


# ===========================================================================
# Caching Tests
# ===========================================================================


def test_function_spans_caches_result(catalog_resource: CatalogResource) -> None:
    """function_spans caches the result."""
    spans1 = catalog_resource.function_spans
    spans2 = catalog_resource.function_spans
    assert spans1 is spans2


def test_invalidate_clears_cache(catalog_resource: CatalogResource) -> None:
    """invalidate() clears the cached spans."""
    _ = catalog_resource.function_spans  # Populate cache
    catalog_resource.invalidate()
    assert catalog_resource._cached_spans is None  # noqa: SLF001


def test_function_spans_repopulates_after_invalidate(
    catalog_resource: CatalogResource,
) -> None:
    """function_spans repopulates cache after invalidate."""
    spans1 = catalog_resource.function_spans
    catalog_resource.invalidate()
    spans2 = catalog_resource.function_spans
    # New tuple is created but with same content
    assert spans1 is not spans2
    assert len(spans1) == len(spans2)


# ===========================================================================
# Paths Tests
# ===========================================================================


def test_paths_returns_all_paths(catalog_resource: CatalogResource) -> None:
    """Paths returns all unique file paths."""
    paths = catalog_resource.paths
    assert len(paths) == EXPECTED_PATH_COUNT
    assert TEST_PATH_A in paths
    assert TEST_PATH_B in paths


def test_paths_empty_catalog(empty_catalog_resource: CatalogResource) -> None:
    """Paths returns empty for empty catalog."""
    paths = empty_catalog_resource.paths
    assert len(paths) == 0


# ===========================================================================
# Module By Path Tests
# ===========================================================================


def test_module_by_path_returns_mapping(catalog_resource: CatalogResource) -> None:
    """module_by_path returns module mapping."""
    mapping = catalog_resource.module_by_path
    assert mapping[TEST_PATH_A] == TEST_MODULE_A
    assert mapping[TEST_PATH_B] == TEST_MODULE_B


def test_module_by_path_empty_catalog(empty_catalog_resource: CatalogResource) -> None:
    """module_by_path returns empty for empty catalog."""
    mapping = empty_catalog_resource.module_by_path
    assert len(mapping) == 0


# ===========================================================================
# Spans For Path Tests
# ===========================================================================


def test_spans_for_path_returns_file_spans(catalog_resource: CatalogResource) -> None:
    """Spans_for_path returns spans for specific file."""
    spans = catalog_resource.spans_for_path(TEST_PATH_A)
    assert len(spans) == EXPECTED_SPANS_IN_PATH_A


def test_spans_for_path_returns_span_data(catalog_resource: CatalogResource) -> None:
    """spans_for_path returns FunctionSpanData objects."""
    spans = catalog_resource.spans_for_path(TEST_PATH_A)
    for span in spans:
        assert isinstance(span, FunctionSpanData)


def test_spans_for_path_single_span_file(catalog_resource: CatalogResource) -> None:
    """spans_for_path returns single span for single-function file."""
    spans = catalog_resource.spans_for_path(TEST_PATH_B)
    assert len(spans) == 1
    assert spans[0].goid == GOID_3


def test_spans_for_path_nonexistent_file(catalog_resource: CatalogResource) -> None:
    """spans_for_path returns empty for nonexistent file."""
    spans = catalog_resource.spans_for_path("nonexistent/file.py")
    assert len(spans) == 0


def test_spans_for_path_includes_urns(catalog_resource: CatalogResource) -> None:
    """spans_for_path includes URNs in span data."""
    spans = catalog_resource.spans_for_path(TEST_PATH_A)
    urns = {span.urn for span in spans}
    assert URN_1 in urns
    assert URN_2 in urns


# ===========================================================================
# Local Name Map Tests
# ===========================================================================


def test_local_name_map_returns_mapping(catalog_resource: CatalogResource) -> None:
    """local_name_map returns local name to GOID mapping."""
    name_map = catalog_resource.local_name_map(TEST_PATH_A)
    # Should have entries for local names
    assert isinstance(name_map, dict)


def test_local_name_map_nonexistent_file(catalog_resource: CatalogResource) -> None:
    """local_name_map returns empty for nonexistent file."""
    name_map = catalog_resource.local_name_map("nonexistent/file.py")
    assert len(name_map) == 0


# ===========================================================================
# Lookup GOID Tests
# ===========================================================================


def test_lookup_goid_finds_function(catalog_resource: CatalogResource) -> None:
    """lookup_goid finds function by path and line."""
    goid = catalog_resource.lookup_goid(TEST_PATH_A, START_LINE_1, END_LINE_1, QUALNAME_1)
    assert goid == GOID_1


def test_lookup_goid_finds_method(catalog_resource: CatalogResource) -> None:
    """lookup_goid finds method by path and line."""
    goid = catalog_resource.lookup_goid(TEST_PATH_A, START_LINE_2, END_LINE_2, QUALNAME_2)
    assert goid == GOID_2


def test_lookup_goid_not_found(catalog_resource: CatalogResource) -> None:
    """lookup_goid returns None when not found."""
    goid = catalog_resource.lookup_goid(TEST_PATH_A, 999, 1000, "nonexistent")
    assert goid is None


def test_lookup_goid_wrong_file(catalog_resource: CatalogResource) -> None:
    """lookup_goid returns None for wrong file."""
    goid = catalog_resource.lookup_goid("wrong/file.py", START_LINE_1, END_LINE_1, QUALNAME_1)
    assert goid is None


def test_lookup_goid_empty_catalog(empty_catalog_resource: CatalogResource) -> None:
    """lookup_goid returns None for empty catalog."""
    goid = empty_catalog_resource.lookup_goid(TEST_PATH_A, 10, 20, "func")
    assert goid is None


# ===========================================================================
# URN For GOID Tests
# ===========================================================================


def test_urn_for_goid_returns_urn(catalog_resource: CatalogResource) -> None:
    """urn_for_goid returns URN for known GOID."""
    urn = catalog_resource.urn_for_goid(GOID_1)
    assert urn == URN_1


def test_urn_for_goid_all_goids(catalog_resource: CatalogResource) -> None:
    """urn_for_goid returns correct URN for all GOIDs."""
    assert catalog_resource.urn_for_goid(GOID_1) == URN_1
    assert catalog_resource.urn_for_goid(GOID_2) == URN_2
    assert catalog_resource.urn_for_goid(GOID_3) == URN_3


def test_urn_for_goid_unknown_returns_none(catalog_resource: CatalogResource) -> None:
    """urn_for_goid returns None for unknown GOID."""
    urn = catalog_resource.urn_for_goid(9999)
    assert urn is None


def test_urn_for_goid_empty_catalog(empty_catalog_resource: CatalogResource) -> None:
    """urn_for_goid returns None for empty catalog."""
    urn = empty_catalog_resource.urn_for_goid(1)
    assert urn is None


# ===========================================================================
# FunctionSpanData Properties Tests
# ===========================================================================


def test_span_data_has_all_properties(catalog_resource: CatalogResource) -> None:
    """FunctionSpanData has all expected properties."""
    spans = catalog_resource.function_spans
    span = next(s for s in spans if s.goid == GOID_1)

    assert span.goid == GOID_1
    assert span.rel_path == TEST_PATH_A
    assert span.qualname == QUALNAME_1
    assert span.start_line == START_LINE_1
    assert span.end_line == END_LINE_1
    assert span.urn == URN_1


# ===========================================================================
# Parametrized Tests
# ===========================================================================


@pytest.mark.parametrize(
    ("goid", "expected_urn"),
    [
        (GOID_1, URN_1),
        (GOID_2, URN_2),
        (GOID_3, URN_3),
        (9999, None),
    ],
)
def test_urn_lookup_parametrized(
    catalog_resource: CatalogResource, goid: int, expected_urn: str | None
) -> None:
    """URN lookup returns expected values."""
    result = catalog_resource.urn_for_goid(goid)
    assert result == expected_urn


@pytest.mark.parametrize(
    ("path", "expected_count"),
    [
        (TEST_PATH_A, 2),
        (TEST_PATH_B, 1),
        ("nonexistent.py", 0),
    ],
)
def test_spans_for_path_count(
    catalog_resource: CatalogResource, path: str, expected_count: int
) -> None:
    """Spans for path returns expected count."""
    spans = catalog_resource.spans_for_path(path)
    assert len(spans) == expected_count
