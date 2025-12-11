"""Tests for CallGraphPlugin metadata and options integration."""

from __future__ import annotations

from typing import Any

from codeintel.core.plugins.execution.options import (
    ConfigSource,
    PluginOptionsResolver,
)
from codeintel.core.plugins.types.metadata import PluginDomain
from codeintel.graphs.plugins.builders.callgraph import (
    CALLGRAPH_METADATA,
    CallGraphPlugin,
)
from codeintel.graphs.plugins.builders.callgraph_options import CallGraphOptions
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_not_none,
    expect_true,
)


class DictConfigSource(ConfigSource):
    """Test config source backed by a dict."""

    def __init__(self, options: dict[str, dict[str, Any]]) -> None:
        self._options = options

    def get_plugin_options(self, plugin_name: str) -> dict[str, Any] | None:
        """Return configured options for the given plugin name.

        Returns
        -------
        dict[str, Any] | None
            Raw options mapping when present.
        """
        return self._options.get(plugin_name)


class TestCallGraphMetadata:
    """Tests for CALLGRAPH_METADATA constant."""

    @staticmethod
    def test_metadata_name() -> None:
        """Verify metadata name is canonical."""
        expect_equal(CALLGRAPH_METADATA.name, "graphs.callgraph")

    @staticmethod
    def test_metadata_domain() -> None:
        """Verify metadata domain is graph."""
        expect_equal(CALLGRAPH_METADATA.domain, PluginDomain.GRAPH)

    @staticmethod
    def test_metadata_kind_and_stage() -> None:
        """Verify kind and stage."""
        expect_equal(CALLGRAPH_METADATA.kind, "builder")
        expect_equal(CALLGRAPH_METADATA.stage, "edges")

    @staticmethod
    def test_metadata_capabilities() -> None:
        """Verify provides and requires."""
        expect_in("graph.callgraph", CALLGRAPH_METADATA.provides)
        expect_in("core.goids", CALLGRAPH_METADATA.requires)

    @staticmethod
    def test_metadata_tables() -> None:
        """Verify produces_tables."""
        expect_in("graph.call_graph_nodes", CALLGRAPH_METADATA.produces_tables)
        expect_in("graph.call_graph_edges", CALLGRAPH_METADATA.produces_tables)

    @staticmethod
    def test_metadata_is_scope_aware() -> None:
        """Verify plugin is marked as scope-aware."""
        expect_true(CALLGRAPH_METADATA.scope_aware)

    @staticmethod
    def test_metadata_extra_graph_kinds() -> None:
        """Verify extra.graph_kinds is set."""
        expect_equal(CALLGRAPH_METADATA.extra.get("graph_kinds"), ("callgraph",))


class TestCallGraphPluginOptionsIntegration:
    """Tests for options resolution integration."""

    @staticmethod
    def test_default_options_without_resolver() -> None:
        """Verify default options when no resolver provided."""
        plugin = CallGraphPlugin()
        opts = plugin.resolve_options()
        expect_true(opts.use_libcst)
        expect_true(opts.resolve_imports)
        expect_true(opts.include_external_calls)

    @staticmethod
    def test_options_with_fast_profile() -> None:
        """Verify fast profile options."""
        source = DictConfigSource(
            {
                "graphs.callgraph": {
                    "use_libcst": False,
                    "resolve_imports": False,
                    "include_external_calls": False,
                },
            }
        )
        resolver = PluginOptionsResolver(source)
        plugin = CallGraphPlugin(options_resolver=resolver)
        opts = plugin.resolve_options()
        expect_true(not opts.use_libcst)
        expect_true(not opts.resolve_imports)
        expect_true(not opts.include_external_calls)

    @staticmethod
    def test_scope_paths_filtering() -> None:
        """Verify scope_paths is passed through options."""
        source = DictConfigSource(
            {
                "graphs.callgraph": {
                    "scope_paths": ["src/", "lib/"],
                },
            }
        )
        resolver = PluginOptionsResolver(source)
        plugin = CallGraphPlugin(options_resolver=resolver)
        opts = plugin.resolve_options()
        expect_equal(opts.scope_paths, ["src/", "lib/"])


class TestCallGraphPluginMetadataProperty:
    """Tests for plugin.metadata property."""

    @staticmethod
    def test_metadata_property_returns_plugin_metadata() -> None:
        """Verify metadata property returns PluginMetadata type."""
        plugin = CallGraphPlugin()
        meta = plugin.metadata
        expect_equal(meta.name, "graphs.callgraph")
        expect_equal(meta.version, "3.0.0")

    @staticmethod
    def test_core_metadata_property() -> None:
        """Verify core_metadata returns CorePluginMetadata."""
        plugin = CallGraphPlugin()
        core = plugin.core_metadata
        expect_is_not_none(core)
        expect_equal(core, CALLGRAPH_METADATA)


class TestCallGraphOptionsModel:
    """Tests for CallGraphOptions dataclass."""

    @staticmethod
    def test_default_values() -> None:
        """Verify default values match expected configuration."""
        opts = CallGraphOptions()
        expect_true(opts.include_external_calls)
        expect_true(opts.resolve_imports)
        expect_true(opts.use_libcst)
