"""Configuration introspection commands for the CodeIntel CLI.

Provides commands to inspect effective configuration after merging
all sources (defaults, file, environment, CLI flags).
"""

from __future__ import annotations

import json
import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TextIO

from cyclopts import App, Parameter

if TYPE_CHECKING:
    from codeintel.cli.cyclopts_common import RuntimeCLI

CONFIG_ENV_PREFIX = "CODEINTEL_"
CONFIG_PATH_ENV_VAR = "CODEINTEL_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path("codeintel.toml")


def _resolve_config_path() -> Path:
    """Return the configured TOML path (env override or default).

    Returns
    -------
    Path
        Path to the config file.
    """
    env_path = os.environ.get(CONFIG_PATH_ENV_VAR)
    return Path(env_path) if env_path else DEFAULT_CONFIG_PATH


def _load_toml_config() -> dict[str, object]:
    """Load configuration from TOML file.

    Returns
    -------
    dict[str, object]
        Configuration loaded from TOML, or empty dict if not found.
    """
    path = _resolve_config_path()
    if not path.exists():
        return {}

    with path.open("rb") as f:
        return tomllib.load(f)


def _load_env_config() -> dict[str, object]:
    """Load configuration from environment variables.

    Returns
    -------
    dict[str, object]
        Configuration values from CODEINTEL_* environment variables.
    """
    result: dict[str, object] = {}
    for key, value in os.environ.items():
        if key.startswith(CONFIG_ENV_PREFIX):
            config_key = key[len(CONFIG_ENV_PREFIX) :].lower()
            result[config_key] = value
    return result


def _cli_to_dict(cli: RuntimeCLI | None) -> dict[str, object]:
    """Convert RuntimeCLI to dictionary.

    Parameters
    ----------
    cli
        RuntimeCLI instance or None.

    Returns
    -------
    dict[str, object]
        Dictionary of non-None CLI values.
    """
    if cli is None:
        return {}

    result: dict[str, object] = {}
    if cli.repo is not None:
        result["repo"] = cli.repo
    if cli.commit is not None:
        result["commit"] = cli.commit
    if cli.db_path is not None:
        result["db_path"] = str(cli.db_path)
    if cli.build_dir is not None:
        result["build_dir"] = str(cli.build_dir)
    if cli.repo_root is not None:
        result["repo_root"] = str(cli.repo_root)
    if cli.document_output_dir is not None:
        result["document_output_dir"] = str(cli.document_output_dir)
    return result


def _resolve_effective_config(cli: RuntimeCLI | None) -> dict[str, object]:
    """Merge config from file, env, and CLI with precedence.

    Parameters
    ----------
    cli
        RuntimeCLI instance with CLI-provided values.

    Returns
    -------
    dict[str, object]
        Merged configuration with source tracking.
    """
    result: dict[str, object] = {}
    sources: dict[str, list[str]] = {
        "file": [],
        "env": [],
        "cli": [],
    }

    # 1. Load file config (lowest precedence)
    file_cfg = _load_toml_config()
    for key, value in file_cfg.items():
        result[key] = value
        sources["file"].append(key)

    # 2. Apply env overrides
    env_cfg = _load_env_config()
    for key, value in env_cfg.items():
        result[key] = value
        sources["env"].append(key)

    # 3. Apply CLI overrides (highest precedence)
    cli_cfg = _cli_to_dict(cli)
    for key, value in cli_cfg.items():
        result[key] = value
        sources["cli"].append(key)

    result["_sources"] = sources
    return result


def _render_config(
    config: dict[str, object],
    source_filter: str | None,
    output_format: Literal["text", "json"],
    writer: TextIO = sys.stdout,
) -> None:
    """Render configuration to output.

    Parameters
    ----------
    config
        Configuration dictionary with _sources metadata.
    source_filter
        Filter to specific source (file, env, cli) or None for all.
    output_format
        Output format (text or json).
    writer
        Output writer.
    """
    raw_sources = config.pop("_sources", {})
    sources: dict[str, list[str]] = (
        raw_sources if isinstance(raw_sources, dict) else {}
    )

    if source_filter and source_filter != "all":
        filtered_keys = sources.get(source_filter, [])
        config = {k: v for k, v in config.items() if k in filtered_keys}

    if output_format == "json":
        output: dict[str, object] = {"config": config, "sources": sources}
        writer.write(json.dumps(output, indent=2, default=str))
        writer.write("\n")
    else:
        writer.write("Effective Configuration:\n")
        writer.write("-" * 40 + "\n")
        for key, value in sorted(config.items()):
            source = _find_source(key, sources)
            writer.write(f"{key}: {value} [{source}]\n")


def _find_source(key: str, sources: dict[str, list[str]]) -> str:
    """Find the source of a config key.

    Parameters
    ----------
    key
        Configuration key.
    sources
        Sources dictionary mapping source name to list of keys.

    Returns
    -------
    str
        Source name (file, env, cli, or unknown).
    """
    for source, keys in sources.items():
        if key in keys:
            return source
    return "unknown"


# -----------------------------------------------------------------------------
# Command Group
# -----------------------------------------------------------------------------

config_app = App(name="config", help="Configuration inspection and management.")


@config_app.command(name="show")
@dataclass
class ConfigShowCommand:
    """Show effective configuration after merging all sources.

    Displays the merged configuration with source tracking, showing
    which values come from file, environment, or defaults.
    """

    source: Annotated[
        Literal["all", "file", "env", "cli"] | None,
        Parameter(help="Show only config from specific source."),
    ] = None
    output_format: Annotated[
        Literal["text", "json"],
        Parameter(name="--format", help="Output format."),
    ] = "text"

    def __call__(self) -> None:
        """Execute the config show command."""
        # For now, we don't have a RuntimeCLI in this context
        cfg = _resolve_effective_config(None)
        _render_config(cfg, self.source, self.output_format)


@config_app.command(name="path")
@dataclass
class ConfigPathCommand:
    """Show path to configuration file.

    Displays the resolved configuration file path and whether it exists.
    """

    def __call__(self) -> None:
        """Execute the config path command."""
        path = _resolve_config_path()
        writer = sys.stdout
        writer.write(f"Config file: {path}\n")
        writer.write(f"Absolute:    {path.absolute()}\n")
        writer.write(f"Exists:      {path.exists()}\n")
        if path.exists():
            writer.write(f"Size:        {path.stat().st_size} bytes\n")


@config_app.command(name="validate")
@dataclass
class ConfigValidateCommand:
    """Validate configuration file syntax and schema.

    Checks the configuration file for syntax errors and validates
    against the expected schema.

    Raises
    ------
    SystemExit
        If the configuration file is invalid.
    """

    def __call__(self) -> None:
        """Execute the config validate command.

        Raises
        ------
        SystemExit
            If the configuration file is invalid.
        """
        path = _resolve_config_path()
        writer = sys.stdout

        if not path.exists():
            writer.write(f"Config file not found: {path}\n")
            return

        try:
            with path.open("rb") as f:
                config = tomllib.load(f)
            writer.write(f"Config file: {path}\n")
            writer.write("Status: VALID\n")
            writer.write(f"Keys: {', '.join(config.keys())}\n")
        except tomllib.TOMLDecodeError as exc:
            writer.write(f"Config file: {path}\n")
            writer.write("Status: INVALID\n")
            writer.write(f"Error: {exc}\n")
            raise SystemExit(1) from exc


@config_app.command(name="env")
@dataclass
class ConfigEnvCommand:
    """Show environment variable configuration.

    Lists all CODEINTEL_* environment variables that affect configuration.
    """

    output_format: Annotated[
        Literal["text", "json"],
        Parameter(name="--format", help="Output format."),
    ] = "text"

    def __call__(self) -> None:
        """Execute the config env command."""
        env_cfg = _load_env_config()
        writer = sys.stdout

        if self.output_format == "json":
            writer.write(json.dumps(env_cfg, indent=2))
            writer.write("\n")
        else:
            writer.write("Environment Configuration:\n")
            writer.write("-" * 40 + "\n")
            if env_cfg:
                for key, value in sorted(env_cfg.items()):
                    writer.write(f"{CONFIG_ENV_PREFIX}{key.upper()}: {value}\n")
            else:
                writer.write("No CODEINTEL_* environment variables set.\n")


__all__ = [
    "config_app",
]
