"""Tests for CfgDfgPlugin metadata and options integration."""

from __future__ import annotations

from typing import Any

from codeintel.build.plugins.graphs.builders.cfg_dfg import CFG_DFG_METADATA, CfgDfgPlugin
from codeintel.build.plugins.graphs.builders.cfg_dfg_options import CfgDfgOptions
from codeintel.core.plugins.execution.options import ConfigSource, PluginOptionsResolver
from codeintel.core.plugins.types.metadata import PluginDomain
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


class TestCfgDfgMetadata:
    """Tests for CFG_DFG_METADATA constant."""

    @staticmethod
    def test_metadata_name() -> None:
        """Verify metadata name is canonical."""
        expect_equal(CFG_DFG_METADATA.name, "graphs.cfg_dfg")

    @staticmethod
    def test_metadata_domain() -> None:
        """Verify metadata domain is graph."""
        expect_equal(CFG_DFG_METADATA.domain, PluginDomain.GRAPH)

    @staticmethod
    def test_metadata_kind_and_stage() -> None:
        """Verify kind and stage."""
        expect_equal(CFG_DFG_METADATA.kind, "builder")
        expect_equal(CFG_DFG_METADATA.stage, "edges")

    @staticmethod
    def test_metadata_capabilities() -> None:
        """Verify provides and requires."""
        expect_in("graph.cfg", CFG_DFG_METADATA.provides)
        expect_in("graph.dfg", CFG_DFG_METADATA.provides)
        expect_in("core.goids", CFG_DFG_METADATA.requires)

    @staticmethod
    def test_metadata_tables() -> None:
        """Verify produces_tables."""
        expect_in("graph.cfg_blocks", CFG_DFG_METADATA.produces_tables)
        expect_in("graph.cfg_edges", CFG_DFG_METADATA.produces_tables)
        expect_in("graph.dfg_edges", CFG_DFG_METADATA.produces_tables)

    @staticmethod
    def test_metadata_is_scope_aware() -> None:
        """Verify plugin is marked as scope-aware."""
        expect_true(CFG_DFG_METADATA.scope_aware)

    @staticmethod
    def test_metadata_extra_graph_kinds() -> None:
        """Verify extra.graph_kinds is set."""
        expect_equal(CFG_DFG_METADATA.extra.get("graph_kinds"), ("cfg", "dfg"))


class TestCfgDfgPluginOptionsIntegration:
    """Tests for options resolution integration."""

    @staticmethod
    def test_default_options_without_resolver() -> None:
        """Verify default options when no resolver provided."""
        plugin = CfgDfgPlugin()
        opts = plugin.resolve_options()
        expect_true(opts.include_test_files)
        expect_equal(opts.scope_paths, None)

    @staticmethod
    def test_options_with_profile() -> None:
        """Verify options from config source."""
        source = DictConfigSource(
            {
                "graphs.cfg_dfg": {
                    "scope_paths": ["src/graphs/"],
                    "include_test_files": False,
                },
            }
        )
        resolver = PluginOptionsResolver(source)
        plugin = CfgDfgPlugin(options_resolver=resolver)
        opts = plugin.resolve_options()
        expect_equal(opts.scope_paths, ["src/graphs/"])
        expect_true(not opts.include_test_files)


class TestCfgDfgPluginMetadataProperty:
    """Tests for plugin.metadata property."""

    @staticmethod
    def test_metadata_property_returns_plugin_metadata() -> None:
        """Verify metadata property returns PluginMetadata type."""
        plugin = CfgDfgPlugin()
        meta = plugin.metadata
        expect_equal(meta.name, "graphs.cfg_dfg")
        expect_equal(meta.version, "3.0.0")

    @staticmethod
    def test_core_metadata_property() -> None:
        """Verify core_metadata returns CorePluginMetadata."""
        plugin = CfgDfgPlugin()
        core = plugin.core_metadata
        expect_is_not_none(core)
        expect_equal(core, CFG_DFG_METADATA)


class TestCfgDfgOptionsModel:
    """Tests for CfgDfgOptions dataclass."""

    @staticmethod
    def test_default_values() -> None:
        """Verify default values match expected configuration."""
        opts = CfgDfgOptions()
        expect_true(opts.include_test_files)
        expect_equal(opts.scope_paths, None)
