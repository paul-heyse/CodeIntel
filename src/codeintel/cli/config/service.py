"""Unified configuration service.

This module provides ConfigService, the single source of truth for all
CLI configuration loading, validation, and access.

Also provides config building utilities for constructing CodeIntelConfig
from CLI options.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from cyclopts import config as cyclopts_config

from codeintel.cli.config.env_vars import CodeIntelEnv
from codeintel.cli.config.loader import TOML_CONFIG_PATHS, apply_overrides, load_config
from codeintel.config.models import CliConfigOptions, CodeIntelConfig, RepoConfig
from codeintel.config.primitives import GraphBackendConfig, GraphFeatureFlags
from codeintel.core.runtime.loader import load_runtime_settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from cyclopts import App

    from codeintel.cli.config.model import CliConfig
    from codeintel.cli.resolution.params import BackendFlags
    from codeintel.config.models import CliPathsInput


LOG = logging.getLogger(__name__)


CONFIG_PATH_ENV_VAR = "CODEINTEL_CONFIG_PATH"


def _resolve_toml_config_path() -> Path | None:
    """Return the configured TOML path if it exists.

    Check environment override first, then search default locations.

    Returns
    -------
    Path | None
        Path to the config file if found, None otherwise.
    """
    env_path = load_runtime_settings().cli.config_path
    if env_path is not None and env_path.exists():
        return env_path

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

        toml_loader_obj = cyclopts_config.Toml(str(path))
        toml_loader = cast(
            "Callable[[App, tuple[str, ...], object], object]",
            toml_loader_obj,
        )
        app_arg = cast("App", apps)
        return toml_loader(app_arg, commands, arguments)

    return _apply_toml


def _make_env_config() -> object:
    """Create an environment config loader for Cyclopts.

    Returns
    -------
    object
        Cyclopts config loader that reads CODEINTEL_* environment variables.
    """
    return CodeIntelEnv(prefix="CODEINTEL_", command=True, show=True)


@dataclass(frozen=True)
class ConfigService:
    """Unified configuration service.

    Precedence (highest to lowest):

    1. CLI flags (explicit overrides)
    2. Config file (codeintel.toml or ~/.codeintel/config.toml)
    3. Built-in defaults from CliConfig

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
        validate: bool = True,
    ) -> ConfigService:
        """Load configuration from all sources with precedence.

        Parameters
        ----------
        config_path
            Explicit config file path. If None, searches default locations.
        cli_overrides
            Overrides from CLI flags (highest precedence).
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
        config = load_config(
            config_file=config_path,
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
        2. Applies CODEINTEL_* environment overrides

        Returns
        -------
        list[object]
            Config callables for Cyclopts App.config parameter.
            Contains TOML and environment config loaders.

        Examples
        --------
        >>> chain = ConfigService.get_cyclopts_config_chain()
        >>> len(chain) == 2
        True
        """
        return [
            _make_optional_toml_config(),
            _make_env_config(),
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


def build_graph_backend_config(flags: BackendFlags) -> GraphBackendConfig:
    """Build graph backend configuration from CLI options.

    Parameters
    ----------
    flags
        Backend preferences collected from CLI flags.

    Returns
    -------
    GraphBackendConfig
        Configured graph backend settings.
    """
    backend: Literal["auto", "cpu", "nx-cugraph"] = "auto"
    if flags.backend == "cpu":
        backend = "cpu"
    elif flags.backend == "nx-cugraph":
        backend = "nx-cugraph"
    return GraphBackendConfig(
        use_gpu=flags.use_gpu,
        backend=backend,
        strict=flags.strict,
    )


def build_config_from_options(
    repo: str,
    commit: str,
    paths_cfg: CliPathsInput,
    backend: BackendFlags,
) -> CodeIntelConfig:
    """Build CodeIntelConfig from explicit CLI options.

    Parameters
    ----------
    repo
        Repository slug.
    commit
        Commit SHA.
    paths_cfg
        CLI paths input describing repo root, build directory, and storage.
    backend
        Graph backend flags captured from CLI.

    Returns
    -------
    CodeIntelConfig
        Configured CodeIntel settings.
    """
    graph_backend = build_graph_backend_config(backend)
    graph_features = GraphFeatureFlags()
    LOG.info(
        "cli.runtime.config repo=%s commit=%s backend=%s use_gpu=%s features=%s",
        repo,
        commit,
        graph_backend.backend,
        graph_backend.use_gpu,
        graph_features,
    )
    repo_cfg = RepoConfig(repo=repo, commit=commit)
    return CodeIntelConfig.from_cli_args(
        repo_cfg=repo_cfg,
        paths_cfg=paths_cfg,
        options=CliConfigOptions(graph_backend=graph_backend, graph_features=graph_features),
    )


__all__ = [
    "CONFIG_PATH_ENV_VAR",
    "TOML_CONFIG_PATHS",
    "ConfigService",
    "build_config_from_options",
    "build_graph_backend_config",
]
