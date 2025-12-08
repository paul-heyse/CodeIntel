"""Demonstration tests for MockFunctionCatalog usage patterns.

This module showcases comprehensive usage of the MockFunctionCatalog
helper for analytics testing scenarios. These patterns should be followed
when testing analytics components that depend on function catalogs.

Testing Charter Compliance:
- Uses real DuckDB via gateway fixtures where appropriate
- MockFunctionCatalog implements the same interface as production
- No monkeypatching or test-only code paths
- Realistic test data via factory functions
"""

from __future__ import annotations

import pytest

from codeintel.analytics.resources.catalog import CatalogProvider
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_not_none,
    expect_length,
    expect_true,
)
from tests._helpers.fakes.function_catalogs import (
    MockFunctionCatalog,
    MockFunctionMeta,
    create_mock_catalog_multi_file,
    create_mock_catalog_realistic,
    create_mock_catalog_with_functions,
)
from tests._helpers.rows import function_meta


class TestMockFunctionCatalogBasics:
    """Basic MockFunctionCatalog usage patterns."""

    @staticmethod
    def test_empty_catalog_returns_none_for_lookups() -> None:
        """Empty catalog returns None for all lookups."""
        catalog = MockFunctionCatalog()

        expect_true(catalog.urn_for_goid(1000) is None)
        expect_true(catalog.goid_for_urn("urn:test:module.py#func") is None)
        expect_true(catalog.lookup_goid("module.py", 1, 10, "func") is None)
        expect_equal(catalog.get_all_goids(), [])
        expect_equal(catalog.get_functions_by_path("module.py"), [])

    @staticmethod
    def test_catalog_with_custom_functions() -> None:
        """Custom functions are accessible via all lookup methods."""
        catalog = MockFunctionCatalog(
            functions=[
                function_meta(
                    goid=1001,
                    rel_path="main.py",
                    qualname="main",
                    snapshot=("test", "test"),
                    line_span=(10, 25),
                ),
                function_meta(
                    goid=1002,
                    rel_path="utils.py",
                    qualname="helper",
                    snapshot=("test", "test"),
                    line_span=(5, 15),
                ),
            ]
        )

        # URN lookups
        expect_equal(catalog.urn_for_goid(1001), "urn:test:test:main.py#main")
        expect_equal(catalog.urn_for_goid(1002), "urn:test:test:utils.py#helper")

        # Reverse lookups
        expect_equal(catalog.goid_for_urn("urn:test:test:main.py#main"), 1001)

        # Span lookups
        expect_equal(catalog.lookup_goid("main.py", 10, 25, "main"), 1001)

        # All GOIDs
        expect_equal(set(catalog.get_all_goids()), {1001, 1002})

        # Functions by path
        main_funcs = catalog.get_functions_by_path("main.py")
        expect_length(main_funcs, 1)
        expect_equal(main_funcs[0].qualname, "main")

    @staticmethod
    def test_catalog_auto_generates_urn_if_not_provided() -> None:
        """URN is auto-generated from rel_path and qualname."""
        catalog = MockFunctionCatalog(
            functions=[
                function_meta(
                    goid=1001,
                    rel_path="mod.py",
                    qualname="process",
                    snapshot=("test", "test"),
                )
            ]
        )

        urn = catalog.urn_for_goid(1001)
        expect_is_not_none(urn)
        if urn is None:
            pytest.fail("URN should be generated")
        expect_in("mod.py", urn)
        expect_in("process", urn)

    @staticmethod
    def test_catalog_builds_spans_from_functions() -> None:
        """Function spans are automatically derived from function metadata."""
        catalog = MockFunctionCatalog(
            functions=[
                function_meta(
                    goid=1001,
                    rel_path="main.py",
                    qualname="main",
                    snapshot=("test", "test"),
                    line_span=(10, 25),
                ),
            ]
        )

        expect_length(catalog.function_spans, 1)
        span = catalog.function_spans[0]
        expect_equal(span.goid, 1001)
        expect_equal(span.rel_path, "main.py")
        expect_equal(span.start_line, 10)
        expect_equal(span.end_line, 25)


class TestMockCatalogFactoryFunctions:
    """Tests for factory function usage patterns."""

    @staticmethod
    def test_create_with_functions_basic() -> None:
        """create_mock_catalog_with_functions creates populated catalog."""
        catalog = create_mock_catalog_with_functions(5)

        expect_length(catalog.functions, 5)
        expect_length(catalog.get_all_goids(), 5)

        # All functions are in the same module
        funcs_in_module = catalog.get_functions_by_path("module.py")
        expect_length(funcs_in_module, 5)

    @staticmethod
    def test_create_with_functions_custom_path() -> None:
        """create_mock_catalog_with_functions accepts custom path."""
        catalog = create_mock_catalog_with_functions(
            3,
            rel_path="src/utils.py",
            module_name="src.utils",
        )

        expect_length(catalog.functions, 3)
        expect_in("src/utils.py", catalog.module_by_path)
        expect_equal(catalog.module_by_path["src/utils.py"], "src.utils")

    @staticmethod
    def test_create_multi_file_default() -> None:
        """create_mock_catalog_multi_file creates multi-file catalog."""
        catalog = create_mock_catalog_multi_file()

        # Default: 2 in main.py, 3 in utils.py
        expected_total = 5
        expect_length(catalog.functions, expected_total)

        main_funcs = catalog.get_functions_by_path("src/main.py")
        utils_funcs = catalog.get_functions_by_path("src/utils.py")
        expected_main = 2
        expected_utils = 3
        expect_length(main_funcs, expected_main)
        expect_length(utils_funcs, expected_utils)

    @staticmethod
    def test_create_multi_file_custom() -> None:
        """create_mock_catalog_multi_file accepts custom file layout."""
        catalog = create_mock_catalog_multi_file(
            {
                "api/routes.py": 4,
                "api/models.py": 2,
                "tests/test_api.py": 3,
            }
        )

        expect_length(catalog.functions, 9)

        routes = catalog.get_functions_by_path("api/routes.py")
        expect_length(routes, 4)

        # Module names are derived from paths
        expect_equal(catalog.module_by_path["api/routes.py"], "api.routes")

    @staticmethod
    def test_create_realistic_patterns() -> None:
        """create_mock_catalog_realistic provides varied function types."""
        catalog = create_mock_catalog_realistic()

        # Has functions across multiple files
        expected_file_count = 4
        expect_length(catalog.module_by_path, expected_file_count)

        # Has varied function patterns
        all_qualnames = [fn.qualname for fn in catalog.functions]

        # Public entry point
        expect_in("main", all_qualnames)

        # Public function
        expect_in("process_data", all_qualnames)

        # Private helper
        expect_in("_validate", all_qualnames)

        # Class methods
        expect_in("User.save", all_qualnames)
        expect_in("User.from_dict", all_qualnames)

        # Async function
        expect_in("fetch_data", all_qualnames)


class TestMockCatalogWithCatalogProvider:
    """Tests for MockFunctionCatalog integration with CatalogProvider."""

    @staticmethod
    def test_catalog_provider_accepts_mock() -> None:
        """CatalogProvider can wrap MockFunctionCatalog."""
        mock = create_mock_catalog_with_functions(3)

        provider = CatalogProvider()
        provider.set_preloaded(mock)

        # Provider returns the mock
        result = provider.get()
        expect_true(result is mock)

    @staticmethod
    def test_catalog_provider_caches_result() -> None:
        """CatalogProvider caches the mock on subsequent calls."""
        mock = create_mock_catalog_with_functions(3)

        provider = CatalogProvider()
        provider.set_preloaded(mock)

        result1 = provider.get()
        result2 = provider.get()

        expect_true(result1 is result2)


class TestMockCatalogWithFixtures:
    """Tests demonstrating fixture usage patterns."""

    @staticmethod
    def test_fixture_empty_catalog(mock_function_catalog: MockFunctionCatalog) -> None:
        """mock_function_catalog fixture provides empty catalog."""
        expect_length(mock_function_catalog.functions, 0)
        expect_true(mock_function_catalog.urn_for_goid(1000) is None)

    @staticmethod
    def test_fixture_with_functions(mock_catalog_with_functions: MockFunctionCatalog) -> None:
        """mock_catalog_with_functions fixture provides populated catalog."""
        expect_length(mock_catalog_with_functions.functions, 3)
        expect_length(mock_catalog_with_functions.get_all_goids(), 3)

    @staticmethod
    def test_fixture_multi_file(mock_catalog_multi_file: MockFunctionCatalog) -> None:
        """mock_catalog_multi_file fixture provides multi-file catalog."""
        expected_total = 5
        expect_length(mock_catalog_multi_file.functions, expected_total)
        expect_true(len(mock_catalog_multi_file.module_by_path) > 1)

    @staticmethod
    def test_fixture_realistic(mock_catalog_realistic: MockFunctionCatalog) -> None:
        """mock_catalog_realistic fixture provides varied function patterns."""
        # Has public, private, class methods, async
        qualnames = [fn.qualname for fn in mock_catalog_realistic.functions]

        expect_true(any(q.startswith("_") for q in qualnames))  # Private
        expect_true(any("." in q for q in qualnames))  # Methods


class TestMockCatalogWithMockRuntime:
    """Tests combining MockFunctionCatalog with MockGraphRuntime."""

    @staticmethod
    def test_combined_mocks_for_analytics_context(
        mock_catalog_realistic: MockFunctionCatalog,
        mock_runtime_all_graphs: object,  # MockGraphRuntime from conftest
    ) -> None:
        """Both mocks can be used together for comprehensive testing."""
        # Catalog provides function metadata
        expect_true(len(mock_catalog_realistic.functions) > 0)

        # Runtime provides graph data
        expect_true(mock_runtime_all_graphs is not None)


class TestMockCatalogEdgeCases:
    """Edge case tests for MockFunctionCatalog."""

    @staticmethod
    def test_lookup_goid_partial_match() -> None:
        """lookup_goid matches on rel_path and start_line."""
        catalog = MockFunctionCatalog(
            functions=[
                MockFunctionMeta(
                    goid=1001,
                    rel_path="main.py",
                    qualname="main",
                    start_line=10,
                    end_line=25,
                ),
            ]
        )

        # Match without qualname
        result = catalog.lookup_goid("main.py", 10, 25, None)
        expect_equal(result, 1001)

        # Match with qualname
        result = catalog.lookup_goid("main.py", 10, 25, "main")
        expect_equal(result, 1001)

        # No match on wrong line
        result = catalog.lookup_goid("main.py", 11, 25, "main")
        expect_true(result is None)

    @staticmethod
    def test_multiple_functions_same_file() -> None:
        """Multiple functions in same file are all accessible."""
        catalog = MockFunctionCatalog(
            functions=[
                MockFunctionMeta(goid=1001, rel_path="mod.py", qualname="func_a", start_line=1),
                MockFunctionMeta(goid=1002, rel_path="mod.py", qualname="func_b", start_line=20),
                MockFunctionMeta(goid=1003, rel_path="mod.py", qualname="func_c", start_line=40),
            ]
        )

        funcs = catalog.get_functions_by_path("mod.py")
        expected_count = 3
        expect_length(funcs, expected_count)

        goids = catalog.get_all_goids()
        expect_equal(set(goids), {1001, 1002, 1003})

    @staticmethod
    def test_empty_module_by_path() -> None:
        """Catalog works without module_by_path."""
        catalog = MockFunctionCatalog(
            functions=[MockFunctionMeta(goid=1001)],
            module_by_path={},
        )

        expect_true(catalog.urn_for_goid(1001) is not None)
        expect_length(catalog.module_by_path, 0)


class TestMockCatalogDocumentation:
    """Tests that serve as documentation for common patterns."""

    @staticmethod
    def test_pattern_testing_catalog_access() -> None:
        """Pattern: Testing code that accesses catalog metadata.

        When your code needs to look up function URNs or GOIDs,
        use MockFunctionCatalog with known test data.
        """
        # Setup: Create catalog with known functions
        catalog = MockFunctionCatalog(
            functions=[
                MockFunctionMeta(
                    goid=12345,
                    urn="urn:test:app.py#main",
                    rel_path="app.py",
                    qualname="main",
                ),
            ]
        )

        # Test: Code that looks up URN by GOID
        urn = catalog.urn_for_goid(12345)

        # Assert: Known result
        expect_equal(urn, "urn:test:app.py#main")

    @staticmethod
    def test_pattern_testing_function_discovery() -> None:
        """Pattern: Testing code that discovers functions in a file.

        When your code needs to find all functions in a file,
        use MockFunctionCatalog with multi-file layout.
        """
        # Setup: Create catalog with functions in multiple files
        catalog = create_mock_catalog_multi_file(
            {
                "src/core.py": 5,
                "src/utils.py": 3,
            }
        )

        # Test: Discovery code
        core_funcs = catalog.get_functions_by_path("src/core.py")
        utils_funcs = catalog.get_functions_by_path("src/utils.py")

        # Assert: Expected counts
        expect_length(core_funcs, 5)
        expect_length(utils_funcs, 3)

    @staticmethod
    def test_pattern_testing_with_realistic_data() -> None:
        """Pattern: Integration tests with varied function types.

        When testing analytics that handles different function types,
        use the realistic catalog factory.
        """
        # Setup: Realistic catalog with varied patterns
        catalog = create_mock_catalog_realistic()

        # Test: Analytics processing (simulated)
        public_funcs = [fn for fn in catalog.functions if not fn.qualname.startswith("_")]
        private_funcs = [fn for fn in catalog.functions if fn.qualname.startswith("_")]
        methods = [fn for fn in catalog.functions if "." in fn.qualname]

        # Assert: Realistic distribution
        expect_true(len(public_funcs) > 0)
        expect_true(len(private_funcs) > 0)
        expect_true(len(methods) > 0)
