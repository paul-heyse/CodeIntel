"""Unified configuration service.

This module provides ConfigService, the single source of truth for all
CLI configuration loading, validation, and access.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from cyclopts import config as cyclopts_config

from codeintel.cli.config.loader import apply_overrides, load_config
from codeintel.cli.config.model import CliConfig

if TYPE_CHECKING:
    from cyclopts import App


# Environment variable prefix (canonical location)
CONFIG_ENV_PREFIX = "CODEINTEL_"

# Environment variable for explicit config path
CONFIG_PATH_ENV_VAR = "CODEINTEL_CONFIG_PATH"

# TOML config file search paths
TOML_CONFIG_PATHS = [
    Path("codeintel.toml"),
    Path.home() / ".codeintel" / "config.toml",
]


def _resolve_toml_config_path() -> Path | None:
    """Return the configured TOML path if it exists.

    Check environment override first, then search default locations.

    Returns
    -------
    Path | None
        Path to the config file if found, None otherwise.
    """
    # Check environment override
    env_path = os.environ.get(CONFIG_PATH_ENV_VAR)
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # Search default locations
    for path in TOML_CONFIG_PATHS:
        if path.exists():
            return path

    return None


def _make_optional_toml_config() -> Callable[[object, tuple[str, ...], object], object]:
    """Create TOML config callable for Cyclopts.

    Returns
    -------
    Callable
        Config callable that applies TOML config if present.
    """

    def _apply_toml(apps: object, commands: tuple[str, ...], arguments: object) -> object:
        """Apply TOML config if present; otherwise return unchanged.

        Parameters
        ----------
        apps
            Cyclopts App instance.
        commands
            Command path tuple.
        arguments
            Current arguments to potentially override.

        Returns
        -------
        object
            Possibly updated arguments after applying TOML overrides.
        """
        path = _resolve_toml_config_path()
        if path is None:
            return arguments

        toml_loader = cast("Any", cyclopts_config.Toml(str(path)))
        app_arg = cast("App", apps)
        args_arg = cast("Any", arguments)
        return toml_loader(app_arg, commands, args_arg)

    return _apply_toml


@dataclass(frozen=True)
class ConfigService:
    """Unified configuration service.

    Precedence (highest to lowest):

    1. CLI flags (explicit overrides)
    2. Environment variables (CODEINTEL_*)
    3. Config file (codeintel.toml or ~/.codeintel/config.yaml)
    4. Built-in defaults from CliConfig

    Parameters
    ----------
    config
        The resolved, validated configuration.
    sources
        Ordered list of sources that contributed to the config.

    Examples
    --------
    >>> service = ConfigService.load(validate=False)
    >>> service.config.output_format
    'text'
    >>> "defaults" in service.sources
    True
    """

    config: CliConfig
    sources: tuple[str, ...]

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
        cli_overrides: dict[str, Any] | None = None,
        *,
        env_prefix: str = CONFIG_ENV_PREFIX,
        validate: bool = True,
    ) -> ConfigService:
        """Load configuration from all sources with precedence.

        Parameters
        ----------
        config_path
            Explicit config file path. If None, searches default locations.
        cli_overrides
            Overrides from CLI flags (highest precedence).
        env_prefix
            Environment variable prefix.
        validate
            If True, validate config and raise ConfigLoadError on failure.

        Returns
        -------
        ConfigService
            Service with loaded configuration.

        Examples
        --------
        >>> service = ConfigService.load(validate=False)
        >>> service.config.color
        True
        """
        # Use the existing load_config with its precedence
        config = load_config(
            config_file=config_path,
            env_prefix=env_prefix,
            cli_overrides=cli_overrides,
            validate=validate,
        )

        return cls(
            config=config,
            sources=tuple(config.config_sources),
        )

    @staticmethod
    def get_cyclopts_config_chain() -> list[object]:
        """Return Cyclopts-compatible config callables.

        Integrate with Cyclopts' config parameter while maintaining
        our unified precedence. The returned chain:

        1. Applies TOML config if present (codeintel.toml)
        2. Applies environment variable overrides (CODEINTEL_*)

        Returns
        -------
        list[object]
            Config callables for Cyclopts App.config parameter.
            Contains a TOML config loader and an environment variable loader.

        Examples
        --------
        >>> chain = ConfigService.get_cyclopts_config_chain()
        >>> len(chain) == 2
        True
        """
        return [
            _make_optional_toml_config(),
            cyclopts_config.Env(CONFIG_ENV_PREFIX),
        ]

    @staticmethod
    def get_toml_config_path() -> Path | None:
        """Get the resolved TOML config path if it exists.

        Returns
        -------
        Path | None
            Path to TOML config file or None.

        Examples
        --------
        >>> path = ConfigService.get_toml_config_path()
        >>> path is None or path.suffix == ".toml"
        True
        """
        return _resolve_toml_config_path()

    def with_overrides(self, **overrides: object) -> ConfigService:
        """Create new service with overrides applied.

        Useful for testing or command-specific modifications.

        Parameters
        ----------
        **overrides
            Field overrides to apply.

        Returns
        -------
        ConfigService
            New service with overrides applied.

        Examples
        --------
        >>> service = ConfigService.load(validate=False)
        >>> modified = service.with_overrides(color=False)
        >>> modified.config.color
        False
        """
        new_config = apply_overrides(self.config, overrides)
        return ConfigService(
            config=new_config,
            sources=(*self.sources, "overrides"),
        )


__all__ = [
    "CONFIG_ENV_PREFIX",
    "CONFIG_PATH_ENV_VAR",
    "TOML_CONFIG_PATHS",
    "ConfigService",
]
