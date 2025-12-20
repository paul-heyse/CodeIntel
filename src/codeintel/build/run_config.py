"""Build run configuration tying profiles to options resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from codeintel.core.plugins.execution.options import (
    PluginConfigBundle,
    PluginOptionsResolver,
    ProfiledConfigSource,
)
from codeintel.core.plugins.execution.profiles import get_profile

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from codeintel.core.plugins.execution.profiles import ExecutionProfile


@dataclass(frozen=True)
class BuildRunConfig:
    """Configuration for a single build/analytics run."""

    profile_name: str = "full"
    cli_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    scope_paths: list[str] | None = None
    output_dir: Path | None = None
    repo: str = ""
    commit: str = ""
    base_config: PluginConfigBundle | None = None

    def get_profile(self) -> ExecutionProfile:
        """Return the selected execution profile.

        Returns
        -------
        ExecutionProfile
            Selected profile instance.
        """
        return get_profile(self.profile_name)

    def build_options_resolver(self) -> PluginOptionsResolver:
        """Build a PluginOptionsResolver for this run configuration.

        Returns
        -------
        PluginOptionsResolver
            Resolver configured with base/profile/CLI layers.
        """
        profile = self.get_profile()
        profile_bundle = PluginConfigBundle(plugin_options=profile.plugin_options)
        cli_bundle = PluginConfigBundle(plugin_options=self.cli_overrides)

        config_source = ProfiledConfigSource(
            base=self.base_config,
            profile=profile_bundle,
            cli=cli_bundle,
            active_profile_name=profile.name,
        )
        return PluginOptionsResolver(config_source)

    def config_overrides_for_target(self, target_name: str) -> Mapping[str, Any]:
        """Return merged config overrides for a target name.

        Parameters
        ----------
        target_name
            Target name used as the plugin identifier in config layers.

        Returns
        -------
        Mapping[str, Any]
            Merged configuration mapping for the target, or empty mapping.
        """
        resolver = self.build_options_resolver()
        options = resolver.config_source.get_plugin_options(target_name)
        return options or {}


__all__ = ["BuildRunConfig"]
