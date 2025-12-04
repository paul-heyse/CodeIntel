"""Config seed pack for configuration reference data.

This module provides the ConfigPack which seeds the analytics.config_values
table with configuration file references and their usage patterns.

The pack depends on CORE_PACK and uses its module definitions to create
realistic configuration reference patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tests._helpers.builders import ConfigValueRow, insert_rows
from tests._helpers.seeds.core import (
    CORE_PACK,
    MOD_A_FQN,
    MOD_A_PATH,
    MOD_B_FQN,
    MOD_B_PATH,
    MOD_UTIL_FQN,
)

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


# =============================================================================
# Config Data Constants
# =============================================================================

# Configuration file paths
CONFIG_PYPROJECT = "pyproject.toml"
CONFIG_SETTINGS = "settings.yaml"
CONFIG_ENV = ".env"


# =============================================================================
# Config Pack Implementation
# =============================================================================


@dataclass
class ConfigPack:
    """Seed pack for configuration reference data.

    Seeds analytics.config_values table with configuration file references
    showing which modules use which configuration keys.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    include_env : bool
        Whether to include .env file references.
    """

    name: str = "config"
    include_env: bool = True

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one.

        Returns
        -------
        tuple[SeedPack, ...]
            CorePack is required for module data.
        """
        return (CORE_PACK,)

    def apply(self, ctx: TestContext) -> None:
        """Apply config seeds to the test context.

        Seeds analytics.config_values with configuration references.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        self._seed_config_values(ctx)

    def _seed_config_values(self, ctx: TestContext) -> None:
        """Seed the config_values table.

        Parameters
        ----------
        ctx
            Test context with gateway.
        """
        rows = [
            # pyproject.toml references
            ConfigValueRow(
                repo=ctx.repo,
                commit=ctx.commit,
                config_path=CONFIG_PYPROJECT,
                format="toml",
                key="tool.ruff.line-length",
                reference_paths=[MOD_A_PATH, MOD_B_PATH],
                reference_modules=[MOD_A_FQN, MOD_B_FQN],
                reference_count=2,
            ),
            ConfigValueRow(
                repo=ctx.repo,
                commit=ctx.commit,
                config_path=CONFIG_PYPROJECT,
                format="toml",
                key="project.name",
                reference_paths=[],
                reference_modules=[],
                reference_count=0,
            ),
            # settings.yaml references
            ConfigValueRow(
                repo=ctx.repo,
                commit=ctx.commit,
                config_path=CONFIG_SETTINGS,
                format="yaml",
                key="database.host",
                reference_paths=[MOD_A_PATH],
                reference_modules=[MOD_A_FQN],
                reference_count=1,
            ),
            ConfigValueRow(
                repo=ctx.repo,
                commit=ctx.commit,
                config_path=CONFIG_SETTINGS,
                format="yaml",
                key="cache.ttl",
                reference_paths=[MOD_B_PATH],
                reference_modules=[MOD_B_FQN],
                reference_count=1,
            ),
        ]

        # Optionally add .env references
        if self.include_env:
            rows.extend(
                [
                    ConfigValueRow(
                        repo=ctx.repo,
                        commit=ctx.commit,
                        config_path=CONFIG_ENV,
                        format="env",
                        key="DEBUG",
                        reference_paths=[MOD_A_PATH, MOD_B_PATH],
                        reference_modules=[MOD_A_FQN, MOD_B_FQN, MOD_UTIL_FQN],
                        reference_count=3,
                    ),
                    ConfigValueRow(
                        repo=ctx.repo,
                        commit=ctx.commit,
                        config_path=CONFIG_ENV,
                        format="env",
                        key="API_KEY",
                        reference_paths=[MOD_A_PATH],
                        reference_modules=[MOD_A_FQN],
                        reference_count=1,
                    ),
                ]
            )

        insert_rows(ctx.gateway, rows)


# Default instance for common usage
CONFIG_PACK = ConfigPack()


__all__ = [
    "CONFIG_ENV",
    "CONFIG_PACK",
    "CONFIG_PYPROJECT",
    "CONFIG_SETTINGS",
    "ConfigPack",
]
