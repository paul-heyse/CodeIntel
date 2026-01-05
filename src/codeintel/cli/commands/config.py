"""Configuration introspection commands for the CodeIntel CLI.

Provide commands to inspect effective configuration after merging
defaults, file-based config, and CLI flags.
"""

from __future__ import annotations

import json
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App

from codeintel.cli.config import TOML_CONFIG_PATHS, ConfigService, config_to_dict
from codeintel.cli.options.registry import CONFIG_FORMAT, CONFIG_SOURCE, CONFIG_TARGET
from codeintel.cli.options.types import CommandPath, option_param
from codeintel.core.runtime.loader import load_runtime_settings


def _resolve_config_path() -> Path:
    """Return the configured TOML path (env override or default).

    Check environment override first, then search default TOML locations.

    Returns
    -------
    Path
        Path to the config file.
    """
    env_path = load_runtime_settings().cli.config_path
    if env_path is not None:
        return env_path

    for path in TOML_CONFIG_PATHS:
        if path.exists():
            return path

    return Path("config/codeintel.toml")


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


config_app = App(name="config", help="Configuration inspection and management.")
CONFIG_SHOW_PATH: CommandPath = ("config", "show")
CONFIG_ENV_PATH: CommandPath = ("config", "env")
CONFIG_INIT_PATH: CommandPath = ("config", "init")


@config_app.command(name="show")
@dataclass
class ConfigShowCommand:
    """Show effective configuration after merging all sources.

    Display the merged configuration with source tracking, showing
    which values come from file, CLI overrides, or defaults.
    """

    source: Annotated[
        Literal["all", "file", "cli-flags", "defaults"] | None,
        option_param(CONFIG_SOURCE, command_path=CONFIG_SHOW_PATH),
    ] = None
    output_format: Annotated[
        Literal["text", "json"],
        option_param(CONFIG_FORMAT, command_path=CONFIG_SHOW_PATH),
    ] = "text"

    def __call__(self) -> None:
        """Execute the config show command."""
        service = ConfigService.load(validate=False)
        cfg_dict = config_to_dict(service.config)
        writer = sys.stdout

        if self.output_format == "json":
            output: dict[str, object] = {
                "config": cfg_dict,
                "sources": list(service.sources),
            }
            writer.write(json.dumps(output, indent=2, default=str))
            writer.write("\n")
        else:
            writer.write("Effective Configuration:\n")
            writer.write("-" * 40 + "\n")
            for key, value in sorted(cfg_dict.items()):
                writer.write(f"{key}: {value}\n")
            writer.write("\nSources: " + " → ".join(service.sources) + "\n")


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
    """Show runtime settings derived from environment variables."""

    output_format: Annotated[
        Literal["text", "json"],
        option_param(CONFIG_FORMAT, command_path=CONFIG_ENV_PATH),
    ] = "text"

    def __call__(self) -> None:
        """Execute the config env command."""
        settings = load_runtime_settings()
        settings_dict = asdict(settings)
        writer = sys.stdout

        if self.output_format == "json":
            writer.write(json.dumps(settings_dict, indent=2, default=str))
            writer.write("\n")
        else:
            writer.write("Runtime Settings (Environment-derived):\n")
            writer.write("-" * 40 + "\n")
            for section, values in settings_dict.items():
                writer.write(f"{section}:\n")
                if isinstance(values, dict):
                    for key, value in sorted(values.items()):
                        writer.write(f"  {key}: {value}\n")
                else:
                    writer.write(f"  value: {values}\n")


@config_app.command(name="init")
@dataclass
class ConfigInitCommand:
    """Create a default configuration file.

    Generate a configuration file with default values and documentation
    at the specified path or default location.
    """

    target: Annotated[
        Path | None,
        option_param(CONFIG_TARGET, command_path=CONFIG_INIT_PATH),
    ] = None

    def __call__(self) -> None:
        """Execute the config init command.

        Raises
        ------
        SystemExit
            If the config file already exists.
        """
        target = self.target or (Path("config") / "codeintel.toml")
        writer = sys.stdout

        if target.exists():
            writer.write(f"Config file already exists: {target}\n")
            writer.write("Use --target to specify a different path.\n")
            raise SystemExit(1)

        target.parent.mkdir(parents=True, exist_ok=True)

        default_config = """
output_format = "text"
color = true
log_level = "WARNING"

[progress]
enabled = true
threshold = 2.0

[telemetry]
enabled = true
service_name = "codeintel-cli"

[retry]
max_attempts = 3
initial_delay = 0.5
backoff_factor = 2.0
max_delay = 30.0
"""

        target.write_text(default_config)
        writer.write(f"Created configuration file: {target}\n")


@config_app.command(name="paths")
@dataclass
class ConfigPathsCommand:
    """Show configuration file search paths.

    Display all paths where configuration files are searched,
    with indicators for which paths exist.
    """

    def __call__(self) -> None:
        """Execute the config paths command."""
        writer = sys.stdout
        writer.write("Configuration File Search Paths:\n")
        writer.write("-" * 40 + "\n")

        all_paths = list(TOML_CONFIG_PATHS)
        seen: set[str] = set()

        for path in all_paths:
            path_str = str(path.absolute())
            if path_str in seen:
                continue
            seen.add(path_str)

            exists = "✓" if path.exists() else "✗"
            writer.write(f"  {exists} {path}\n")

        active_path = ConfigService.get_toml_config_path()
        if active_path:
            writer.write(f"\nActive config: {active_path}\n")


__all__ = [
    "config_app",
]
