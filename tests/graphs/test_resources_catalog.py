"""Tests for CatalogService resource provider.

This module tests the CatalogService from
`codeintel.storage.catalog`, including:

- Resource protocol compliance
- Catalog delegation
- Caching behavior
- FunctionSpan data
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from codeintel.core.catalog import FunctionSpan
from codeintel.core.resources import ResourceRegistry
from codeintel.storage.catalog import CatalogService, FunctionCatalog
from tests._helpers.assertions import (
    expect_equal,
    expect_is_instance,
    expect_length,
    expect_true,
)

if TYPE_CHECKING:
    from tests.graphs.conftest import CatalogSampleData


RESOURCE_NAME: Final[str] = "catalog"


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
def empty_catalog_resource(empty_catalog: FunctionCatalog) -> CatalogService:
    """Create a CatalogService wrapping an empty catalog.

    Returns
    -------
    CatalogService
        Service wrapping empty catalog.
    """
    return CatalogService(empty_catalog)


@pytest.fixture
def resource_registry() -> ResourceRegistry:
    """Provide a fresh ResourceRegistry for catalog tests.

    Returns
    -------
    ResourceRegistry
        Registry instance for tests.
    """
    return ResourceRegistry()


def test_catalog_service_name_constant() -> None:
    """CatalogService.RESOURCE_NAME is 'catalog'."""
    expect_equal(CatalogService.RESOURCE_NAME, RESOURCE_NAME)


def test_catalog_service_name_property(catalog_resource: CatalogService) -> None:
    """CatalogService.resource_name returns RESOURCE_NAME."""
    expect_equal(catalog_resource.resource_name, RESOURCE_NAME)


def test_catalog_service_get_returns_self(catalog_resource: CatalogService) -> None:
    """CatalogService.get() returns self as CatalogPort."""
    result = catalog_resource.get()
    expect_true(result is catalog_resource)


def test_catalog_service_has_expected_methods(catalog_resource: CatalogService) -> None:
    """CatalogService has expected catalog interface methods."""
    expect_true(hasattr(catalog_resource, "function_spans"))
    expect_true(hasattr(catalog_resource, "spans_for_path"))
    expect_true(hasattr(catalog_resource, "lookup_goid"))
    expect_true(hasattr(catalog_resource, "urn_for_goid"))


def test_catalog_service_registers_with_registry(
    resource_registry: ResourceRegistry, catalog_resource: CatalogService
) -> None:
    """CatalogService can be registered and retrieved from registry."""
    resource_registry.register_provider(catalog_resource)

    retrieved = resource_registry.get_by_name(CatalogService.RESOURCE_NAME)
    expect_true(retrieved is catalog_resource)


def test_catalog_service_require_returns_registered(
    resource_registry: ResourceRegistry, catalog_resource: CatalogService
) -> None:
    """Require returns the registered CatalogService."""
    resource_registry.register_provider(catalog_resource)

    required = resource_registry.require_by_name(CatalogService.RESOURCE_NAME)
    expect_true(required is catalog_resource)


def test_function_spans_returns_all_spans(
    catalog_resource: CatalogService, catalog_sample_data: CatalogSampleData
) -> None:
    """Function_spans returns all function spans."""
    spans = catalog_resource.function_spans
    expect_length(spans, len(catalog_sample_data.functions))


def test_function_spans_returns_span_objects(catalog_resource: CatalogService) -> None:
    """function_spans returns FunctionSpan objects."""
    spans = catalog_resource.function_spans
    for span in spans:
        expect_is_instance(span, FunctionSpan)


def test_function_spans_contains_expected_goids(
    catalog_resource: CatalogService, catalog_sample_data: CatalogSampleData
) -> None:
    """function_spans contains expected GOIDs."""
    spans = catalog_resource.function_spans
    goids = {span.goid for span in spans}
    expect_equal(goids, {meta.goid for meta in catalog_sample_data.functions})


def test_function_spans_contains_urns(
    catalog_resource: CatalogService, catalog_sample_data: CatalogSampleData
) -> None:
    """function_spans includes URNs."""
    spans = catalog_resource.function_spans
    urns = {span.urn for span in spans}
    expected_urns = {meta.urn for meta in catalog_sample_data.functions}
    expect_equal(urns, expected_urns)


def test_function_spans_empty_catalog(empty_catalog_resource: CatalogService) -> None:
    """function_spans returns empty for empty catalog."""
    spans = empty_catalog_resource.function_spans
    expect_length(spans, 0)


def test_function_spans_caches_result(catalog_resource: CatalogService) -> None:
    """function_spans caches the result."""
    spans1 = catalog_resource.function_spans
    spans2 = catalog_resource.function_spans
    expect_true(spans1 is spans2)


def test_invalidate_clears_cache(catalog_resource: CatalogService) -> None:
    """invalidate() clears the cached spans."""
    spans_before = catalog_resource.function_spans
    catalog_resource.invalidate()
    spans_after = catalog_resource.function_spans
    expect_true(spans_before is not spans_after)


def test_function_spans_repopulates_after_invalidate(
    catalog_resource: CatalogService,
) -> None:
    """function_spans repopulates cache after invalidate."""
    spans1 = catalog_resource.function_spans
    catalog_resource.invalidate()
    spans2 = catalog_resource.function_spans

    expect_true(spans1 is not spans2)
    expect_equal(len(spans1), len(spans2))


def test_paths_returns_all_paths(
    catalog_resource: CatalogService, catalog_sample_data: CatalogSampleData
) -> None:
    """Paths returns all unique file paths."""
    paths = catalog_resource.paths
    expected_paths = set(catalog_sample_data.module_by_path.keys())
    expect_equal(set(paths), expected_paths)


def test_paths_empty_catalog(empty_catalog_resource: CatalogService) -> None:
    """Paths returns empty for empty catalog."""
    paths = empty_catalog_resource.paths
    expect_length(paths, 0)


def test_module_by_path_returns_mapping(
    catalog_resource: CatalogService, catalog_sample_data: CatalogSampleData
) -> None:
    """module_by_path returns module mapping."""
    mapping = catalog_resource.module_by_path
    expect_equal(mapping, catalog_sample_data.module_by_path)


def test_module_by_path_empty_catalog(empty_catalog_resource: CatalogService) -> None:
    """module_by_path returns empty for empty catalog."""
    mapping = empty_catalog_resource.module_by_path
    expect_length(mapping, 0)


def test_spans_for_path_returns_file_spans(
    catalog_resource: CatalogService, catalog_sample_data: CatalogSampleData
) -> None:
    """Spans_for_path returns spans for specific file."""
    target_path = "pkg/module_a.py"
    spans = catalog_resource.spans_for_path(target_path)
    expected_count = sum(
        1 for meta in catalog_sample_data.functions if meta.rel_path == target_path
    )
    expect_length(spans, expected_count)


def test_spans_for_path_returns_span_objects(catalog_resource: CatalogService) -> None:
    """spans_for_path returns FunctionSpan objects."""
    spans = catalog_resource.spans_for_path("pkg/module_a.py")
    for span in spans:
        expect_is_instance(span, FunctionSpan)


def test_spans_for_path_single_span_file(
    catalog_resource: CatalogService, catalog_sample_data: CatalogSampleData
) -> None:
    """spans_for_path returns single span for single-function file."""
    target_path = "pkg/module_b.py"
    spans = catalog_resource.spans_for_path(target_path)
    expected_goids = [
        meta.goid for meta in catalog_sample_data.functions if meta.rel_path == target_path
    ]
    expect_length(spans, len(expected_goids))
    expect_equal({span.goid for span in spans}, set(expected_goids))


def test_spans_for_path_nonexistent_file(catalog_resource: CatalogService) -> None:
    """spans_for_path returns empty for nonexistent file."""
    spans = catalog_resource.spans_for_path("nonexistent/file.py")
    expect_length(spans, 0)


def test_spans_for_path_includes_urns(
    catalog_resource: CatalogService, catalog_sample_data: CatalogSampleData
) -> None:
    """spans_for_path includes URNs in span data."""
    target_path = "pkg/module_a.py"
    spans = catalog_resource.spans_for_path(target_path)
    urns = {span.urn for span in spans}
    expected_urns = {
        meta.urn for meta in catalog_sample_data.functions if meta.rel_path == target_path
    }
    expect_equal(urns, expected_urns)


def test_local_name_map_returns_mapping(
    catalog_resource: CatalogService, catalog_sample_data: CatalogSampleData
) -> None:
    """local_name_map returns local name to GOID mapping."""
    target_path = "pkg/module_a.py"
    name_map = catalog_resource.local_name_map(target_path)
    expected_names = {
        meta.qualname for meta in catalog_sample_data.functions if meta.rel_path == target_path
    }
    expect_true(expected_names.issubset(name_map.keys()))


def test_local_name_map_nonexistent_file(catalog_resource: CatalogService) -> None:
    """local_name_map returns empty for nonexistent file."""
    name_map = catalog_resource.local_name_map("nonexistent/file.py")
    expect_length(name_map, 0)


def test_lookup_goid_finds_function(
    catalog_resource: CatalogService, catalog_sample_data: CatalogSampleData
) -> None:
    """lookup_goid finds function by path and line."""
    target = catalog_sample_data.functions[0]
    goid = catalog_resource.lookup_goid(
        target.rel_path, target.start_line, target.end_line, target.qualname
    )
    expect_equal(goid, target.goid)


def test_lookup_goid_finds_method(
    catalog_resource: CatalogService, catalog_sample_data: CatalogSampleData
) -> None:
    """lookup_goid finds method by path and line."""
    target = catalog_sample_data.functions[1]
    goid = catalog_resource.lookup_goid(
        target.rel_path, target.start_line, target.end_line, target.qualname
    )
    expect_equal(goid, target.goid)


def test_lookup_goid_not_found(catalog_resource: CatalogService) -> None:
    """lookup_goid returns None when not found."""
    goid = catalog_resource.lookup_goid("pkg/module_a.py", 999, 1000, "nonexistent")
    expect_true(goid is None)


def test_lookup_goid_wrong_file(catalog_resource: CatalogService) -> None:
    """lookup_goid returns None for wrong file."""
    goid = catalog_resource.lookup_goid("wrong/file.py", 10, 15, "func1")
    expect_true(goid is None)


def test_lookup_goid_empty_catalog(empty_catalog_resource: CatalogService) -> None:
    """lookup_goid returns None for empty catalog."""
    goid = empty_catalog_resource.lookup_goid("pkg/module_a.py", 10, 20, "func")
    expect_true(goid is None)


def test_urn_for_goid_returns_urn(
    catalog_resource: CatalogService, catalog_sample_data: CatalogSampleData
) -> None:
    """urn_for_goid returns URN for known GOID."""
    target = catalog_sample_data.functions[0]
    urn = catalog_resource.urn_for_goid(target.goid)
    expect_equal(urn, target.urn)


def test_urn_for_goid_all_goids(
    catalog_resource: CatalogService, catalog_sample_data: CatalogSampleData
) -> None:
    """urn_for_goid returns correct URN for all GOIDs."""
    for meta in catalog_sample_data.functions:
        expect_equal(catalog_resource.urn_for_goid(meta.goid), meta.urn)


def test_urn_for_goid_unknown_returns_none(catalog_resource: CatalogService) -> None:
    """urn_for_goid returns None for unknown GOID."""
    urn = catalog_resource.urn_for_goid(9999)
    expect_true(urn is None)


def test_urn_for_goid_empty_catalog(empty_catalog_resource: CatalogService) -> None:
    """urn_for_goid returns None for empty catalog."""
    urn = empty_catalog_resource.urn_for_goid(1)
    expect_true(urn is None)


def test_span_has_all_properties(
    catalog_resource: CatalogService, catalog_sample_data: CatalogSampleData
) -> None:
    """FunctionSpan has all expected properties."""
    spans = catalog_resource.function_spans
    target_meta = catalog_sample_data.functions[0]
    span = next(s for s in spans if s.goid == target_meta.goid)

    expect_equal(span.goid, target_meta.goid)
    expect_equal(span.rel_path, target_meta.rel_path)
    expect_equal(span.qualname, target_meta.qualname)
    expect_equal(span.start_line, target_meta.start_line)
    expect_equal(span.end_line, target_meta.end_line)
    expect_equal(span.urn, target_meta.urn)


def test_urn_lookup_parametrized(
    catalog_resource: CatalogService, catalog_sample_data: CatalogSampleData
) -> None:
    """URN lookup returns expected values."""
    for meta in catalog_sample_data.functions:
        expect_equal(catalog_resource.urn_for_goid(meta.goid), meta.urn)
    expect_true(catalog_resource.urn_for_goid(9999) is None)


def test_spans_for_path_count(
    catalog_resource: CatalogService, catalog_sample_data: CatalogSampleData
) -> None:
    """Spans for path returns expected count."""
    path_counts: dict[str, int] = {}
    for meta in catalog_sample_data.functions:
        path_counts[meta.rel_path] = path_counts.get(meta.rel_path, 0) + 1

    for path, expected_count in path_counts.items():
        spans = catalog_resource.spans_for_path(path)
        expect_length(spans, expected_count)

    spans_missing = catalog_resource.spans_for_path("nonexistent.py")
    expect_length(spans_missing, 0)
