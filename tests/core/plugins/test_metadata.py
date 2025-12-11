"""Tests for CorePluginMetadata."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from tests._helpers.assertions import expect_equal, expect_true


class TestPluginDomain:
    """Tests for PluginDomain enum."""

    @staticmethod
    def test_domain_values() -> None:
        """Verify all domain values are strings."""
        expect_equal(PluginDomain.INGEST.value, "ingest")
        expect_equal(PluginDomain.GRAPH.value, "graph")
        expect_equal(PluginDomain.ANALYTICS.value, "analytics")
        expect_equal(PluginDomain.EXPORT.value, "export")
        expect_equal(PluginDomain.SERVING.value, "serving")
        expect_equal(PluginDomain.CLI.value, "cli")

    @staticmethod
    def test_domain_is_string_enum() -> None:
        """Verify domains can be used as strings."""
        domain = PluginDomain.ANALYTICS
        expect_equal(f"domain={domain}", "domain=analytics")


class TestCorePluginMetadata:
    """Tests for CorePluginMetadata."""

    @staticmethod
    def test_minimal_metadata() -> None:
        """Verify minimal valid metadata can be created."""
        meta = CorePluginMetadata(
            name="test.plugin",
            version="1.0.0",
            description="Test plugin.",
            domain=PluginDomain.ANALYTICS,
            kind="metric",
        )
        expect_equal(meta.name, "test.plugin")
        expect_equal(meta.version, "1.0.0")
        expect_equal(meta.domain, PluginDomain.ANALYTICS)
        expect_equal(meta.kind, "metric")
        expect_equal(meta.stage, None)
        expect_equal(meta.provides, ())
        expect_equal(meta.requires, ())

    @staticmethod
    def test_full_metadata() -> None:
        """Verify full metadata with all fields."""

        @dataclass
        class TestOptions:
            threshold: float = 0.5

        meta = CorePluginMetadata(
            name="analytics.function_metrics",
            version="3.0.0",
            description="Compute function metrics.",
            domain=PluginDomain.ANALYTICS,
            kind="metric",
            stage="function",
            provides=("analytics.function_metrics", "analytics.function_types"),
            requires=("core.goids",),
            produces_tables=("analytics.function_metrics",),
            consumes_tables=("core.goids",),
            supports_incremental=False,
            scope_aware=False,
            options_model=TestOptions,
            resource_hints={"max_memory_mb": 512},
            extra={"custom_key": "custom_value"},
        )
        expect_equal(meta.name, "analytics.function_metrics")
        expect_equal(meta.provides, ("analytics.function_metrics", "analytics.function_types"))
        expect_equal(meta.requires, ("core.goids",))
        expect_equal(meta.options_model, TestOptions)
        expect_true(meta.has_options)
        expect_equal(meta.extra["custom_key"], "custom_value")

    @staticmethod
    def test_has_options_without_model() -> None:
        """Verify has_options returns False when no model set."""
        meta = CorePluginMetadata(
            name="test.plugin",
            version="1.0.0",
            description="Test.",
            domain=PluginDomain.GRAPH,
            kind="builder",
        )
        expect_true(not meta.has_options)

    @staticmethod
    def test_capability_names_property() -> None:
        """Verify capability_names combines provides and requires."""
        meta = CorePluginMetadata(
            name="test.plugin",
            version="1.0.0",
            description="Test.",
            domain=PluginDomain.ANALYTICS,
            kind="metric",
            provides=("cap.a", "cap.b"),
            requires=("cap.c",),
        )
        expect_equal(meta.capability_names, ("cap.a", "cap.b", "cap.c"))

    @staticmethod
    def test_all_tables_property() -> None:
        """Verify all_tables combines produces and consumes."""
        meta = CorePluginMetadata(
            name="test.plugin",
            version="1.0.0",
            description="Test.",
            domain=PluginDomain.ANALYTICS,
            kind="metric",
            produces_tables=("table.a",),
            consumes_tables=("table.b", "table.c"),
        )
        expect_equal(meta.all_tables, ("table.a", "table.b", "table.c"))

    @staticmethod
    def test_empty_name_raises() -> None:
        """Verify empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            CorePluginMetadata(
                name="",
                version="1.0.0",
                description="Test.",
                domain=PluginDomain.ANALYTICS,
                kind="metric",
            )

    @staticmethod
    def test_empty_version_raises() -> None:
        """Verify empty version raises ValueError."""
        with pytest.raises(ValueError, match="version cannot be empty"):
            CorePluginMetadata(
                name="test.plugin",
                version="",
                description="Test.",
                domain=PluginDomain.ANALYTICS,
                kind="metric",
            )

    @staticmethod
    def test_metadata_is_frozen() -> None:
        """Verify metadata is immutable."""
        meta = CorePluginMetadata(
            name="test.plugin",
            version="1.0.0",
            description="Test.",
            domain=PluginDomain.ANALYTICS,
            kind="metric",
        )
        frozen_meta: object = meta
        attr = "name"
        with pytest.raises(AttributeError):
            setattr(frozen_meta, attr, "modified")
