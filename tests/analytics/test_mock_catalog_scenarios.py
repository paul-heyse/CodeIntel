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

from codeintel.analytics.resources.catalog import CatalogProvider
from tests._helpers.fakes.function_catalogs import (
    MockFunctionCatalog,
    MockFunctionMeta,
    create_mock_catalog_multi_file,
    create_mock_catalog_realistic,
    create_mock_catalog_with_functions,
)


class TestMockFunctionCatalogBasics:
    """Basic MockFunctionCatalog usage patterns."""

    def test_empty_catalog_returns_none_for_lookups(self) -> None:
        """Empty catalog returns None for all lookups."""
        catalog = MockFunctionCatalog()

        assert catalog.urn_for_goid(1000) is None
        assert catalog.goid_for_urn("urn:test:module.py#func") is None
        assert catalog.lookup_goid("module.py", 1, 10, "func") is None
        assert catalog.get_all_goids() == []
        assert catalog.get_functions_by_path("module.py") == []

    def test_catalog_with_custom_functions(self) -> None:
        """Custom functions are accessible via all lookup methods."""
        catalog = MockFunctionCatalog(
            functions=[
                MockFunctionMeta(
                    goid=1001,
                    urn="urn:test:main.py#main",
                    rel_path="main.py",
                    qualname="main",
                    start_line=10,
                    end_line=25,
                ),
                MockFunctionMeta(
                    goid=1002,
                    urn="urn:test:utils.py#helper",
                    rel_path="utils.py",
                    qualname="helper",
                    start_line=5,
                    end_line=15,
                ),
            ]
        )

        # URN lookups
        assert catalog.urn_for_goid(1001) == "urn:test:main.py#main"
        assert catalog.urn_for_goid(1002) == "urn:test:utils.py#helper"

        # Reverse lookups
        assert catalog.goid_for_urn("urn:test:main.py#main") == 1001

        # Span lookups
        assert catalog.lookup_goid("main.py", 10, 25, "main") == 1001

        # All GOIDs
        assert set(catalog.get_all_goids()) == {1001, 1002}

        # Functions by path
        main_funcs = catalog.get_functions_by_path("main.py")
        assert len(main_funcs) == 1
        assert main_funcs[0].qualname == "main"

    def test_catalog_auto_generates_urn_if_not_provided(self) -> None:
        """URN is auto-generated from rel_path and qualname."""
        catalog = MockFunctionCatalog(
            functions=[
                MockFunctionMeta(
                    goid=1001,
                    rel_path="mod.py",
                    qualname="process",
                )
            ]
        )

        urn = catalog.urn_for_goid(1001)
        assert urn is not None
        assert "mod.py" in urn
        assert "process" in urn

    def test_catalog_builds_spans_from_functions(self) -> None:
        """Function spans are automatically derived from function metadata."""
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

        assert len(catalog.function_spans) == 1
        span = catalog.function_spans[0]
        assert span.goid == 1001
        assert span.rel_path == "main.py"
        assert span.start_line == 10
        assert span.end_line == 25


class TestMockCatalogFactoryFunctions:
    """Tests for factory function usage patterns."""

    def test_create_with_functions_basic(self) -> None:
        """create_mock_catalog_with_functions creates populated catalog."""
        catalog = create_mock_catalog_with_functions(5)

        assert len(catalog.functions) == 5
        assert len(catalog.get_all_goids()) == 5

        # All functions are in the same module
        funcs_in_module = catalog.get_functions_by_path("module.py")
        assert len(funcs_in_module) == 5

    def test_create_with_functions_custom_path(self) -> None:
        """create_mock_catalog_with_functions accepts custom path."""
        catalog = create_mock_catalog_with_functions(
            3,
            rel_path="src/utils.py",
            module_name="src.utils",
        )

        assert len(catalog.functions) == 3
        assert "src/utils.py" in catalog.module_by_path
        assert catalog.module_by_path["src/utils.py"] == "src.utils"

    def test_create_multi_file_default(self) -> None:
        """create_mock_catalog_multi_file creates multi-file catalog."""
        catalog = create_mock_catalog_multi_file()

        # Default: 2 in main.py, 3 in utils.py
        expected_total = 5
        assert len(catalog.functions) == expected_total

        main_funcs = catalog.get_functions_by_path("src/main.py")
        utils_funcs = catalog.get_functions_by_path("src/utils.py")
        expected_main = 2
        expected_utils = 3
        assert len(main_funcs) == expected_main
        assert len(utils_funcs) == expected_utils

    def test_create_multi_file_custom(self) -> None:
        """create_mock_catalog_multi_file accepts custom file layout."""
        catalog = create_mock_catalog_multi_file(
            {
                "api/routes.py": 4,
                "api/models.py": 2,
                "tests/test_api.py": 3,
            }
        )

        assert len(catalog.functions) == 9

        routes = catalog.get_functions_by_path("api/routes.py")
        assert len(routes) == 4

        # Module names are derived from paths
        assert catalog.module_by_path["api/routes.py"] == "api.routes"

    def test_create_realistic_patterns(self) -> None:
        """create_mock_catalog_realistic provides varied function types."""
        catalog = create_mock_catalog_realistic()

        # Has functions across multiple files
        expected_file_count = 4
        assert len(catalog.module_by_path) == expected_file_count

        # Has varied function patterns
        all_qualnames = [fn.qualname for fn in catalog.functions]

        # Public entry point
        assert "main" in all_qualnames

        # Public function
        assert "process_data" in all_qualnames

        # Private helper
        assert "_validate" in all_qualnames

        # Class methods
        assert "User.save" in all_qualnames
        assert "User.from_dict" in all_qualnames

        # Async function
        assert "fetch_data" in all_qualnames


class TestMockCatalogWithCatalogProvider:
    """Tests for MockFunctionCatalog integration with CatalogProvider."""

    def test_catalog_provider_accepts_mock(self) -> None:
        """CatalogProvider can wrap MockFunctionCatalog."""
        mock = create_mock_catalog_with_functions(3)

        provider = CatalogProvider()
        provider.set_preloaded(mock)

        # Provider returns the mock
        result = provider.get()
        assert result is mock

    def test_catalog_provider_caches_result(self) -> None:
        """CatalogProvider caches the mock on subsequent calls."""
        mock = create_mock_catalog_with_functions(3)

        provider = CatalogProvider()
        provider.set_preloaded(mock)

        result1 = provider.get()
        result2 = provider.get()

        assert result1 is result2


class TestMockCatalogWithFixtures:
    """Tests demonstrating fixture usage patterns."""

    def test_fixture_empty_catalog(self, mock_function_catalog: MockFunctionCatalog) -> None:
        """mock_function_catalog fixture provides empty catalog."""
        assert len(mock_function_catalog.functions) == 0
        assert mock_function_catalog.urn_for_goid(1000) is None

    def test_fixture_with_functions(self, mock_catalog_with_functions: MockFunctionCatalog) -> None:
        """mock_catalog_with_functions fixture provides populated catalog."""
        assert len(mock_catalog_with_functions.functions) == 3
        assert len(mock_catalog_with_functions.get_all_goids()) == 3

    def test_fixture_multi_file(self, mock_catalog_multi_file: MockFunctionCatalog) -> None:
        """mock_catalog_multi_file fixture provides multi-file catalog."""
        expected_total = 5
        assert len(mock_catalog_multi_file.functions) == expected_total
        assert len(mock_catalog_multi_file.module_by_path) > 1

    def test_fixture_realistic(self, mock_catalog_realistic: MockFunctionCatalog) -> None:
        """mock_catalog_realistic fixture provides varied function patterns."""
        # Has public, private, class methods, async
        qualnames = [fn.qualname for fn in mock_catalog_realistic.functions]

        assert any(q.startswith("_") for q in qualnames)  # Private
        assert any("." in q for q in qualnames)  # Methods


class TestMockCatalogWithMockRuntime:
    """Tests combining MockFunctionCatalog with MockGraphRuntime."""

    def test_combined_mocks_for_analytics_context(
        self,
        mock_catalog_realistic: MockFunctionCatalog,
        mock_runtime_all_graphs: object,  # MockGraphRuntime from conftest
    ) -> None:
        """Both mocks can be used together for comprehensive testing."""
        # Catalog provides function metadata
        assert len(mock_catalog_realistic.functions) > 0

        # Runtime provides graph data
        assert mock_runtime_all_graphs is not None


class TestMockCatalogEdgeCases:
    """Edge case tests for MockFunctionCatalog."""

    def test_lookup_goid_partial_match(self) -> None:
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
        assert result == 1001

        # Match with qualname
        result = catalog.lookup_goid("main.py", 10, 25, "main")
        assert result == 1001

        # No match on wrong line
        result = catalog.lookup_goid("main.py", 11, 25, "main")
        assert result is None

    def test_multiple_functions_same_file(self) -> None:
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
        assert len(funcs) == expected_count

        goids = catalog.get_all_goids()
        assert set(goids) == {1001, 1002, 1003}

    def test_empty_module_by_path(self) -> None:
        """Catalog works without module_by_path."""
        catalog = MockFunctionCatalog(
            functions=[MockFunctionMeta(goid=1001)],
            module_by_path={},
        )

        assert catalog.urn_for_goid(1001) is not None
        assert len(catalog.module_by_path) == 0


class TestMockCatalogDocumentation:
    """Tests that serve as documentation for common patterns."""

    def test_pattern_testing_catalog_access(self) -> None:
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
        assert urn == "urn:test:app.py#main"

    def test_pattern_testing_function_discovery(self) -> None:
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
        assert len(core_funcs) == 5
        assert len(utils_funcs) == 3

    def test_pattern_testing_with_realistic_data(self) -> None:
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
        assert len(public_funcs) > 0
        assert len(private_funcs) > 0
        assert len(methods) > 0
