"""Shared option metadata types for the CLI registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cyclopts import Parameter

from codeintel.cli.config.env_vars import build_env_var_name

if TYPE_CHECKING:
    from collections.abc import Mapping

CommandPath = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OptionSpec:
    """Canonical metadata for a CLI option."""

    arg_name: str
    names: tuple[str, ...] | None = None
    help: str | None = None
    show_default: bool | None = None
    show_choices: bool | None = None
    count: bool | None = None
    negative: tuple[str, ...] | None = None
    parse: bool | None = None
    env_name: str | None = None

    def env_var_for(self, command_path: CommandPath) -> str:
        """Return the explicit env var name for this option.

        Returns
        -------
        str
            Fully-qualified env var name for this option.
        """
        suffix = self.env_name or self.arg_name
        return build_env_var_name(command_path, suffix)

    def parameter(self, *, command_path: CommandPath) -> Parameter:
        """Build a Cyclopts Parameter with explicit env var metadata.

        Returns
        -------
        Parameter
            Configured Cyclopts parameter.
        """
        env_var = self.env_var_for(command_path)
        name = list(self.names) if self.names is not None else None
        return Parameter(
            name=name,
            env_var=env_var,
            help=self.help,
            show_default=self.show_default,
            show_choices=self.show_choices,
            count=self.count,
            negative=self.negative,
            parse=self.parse,
        )


@dataclass(frozen=True, slots=True)
class OptionGroup:
    """Bundle a set of related options for reuse."""

    name: str
    options: Mapping[str, OptionSpec]

    def spec(self, key: str) -> OptionSpec:
        """Return the OptionSpec for the given key.

        Returns
        -------
        OptionSpec
            Option specification for the key.
        """
        return self.options[key]


def option_param(spec: OptionSpec, *, command_path: CommandPath) -> Parameter:
    """Return a Parameter configured with explicit env vars.

    Returns
    -------
    Parameter
        Configured Cyclopts parameter.
    """
    return spec.parameter(command_path=command_path)


__all__ = [
    "CommandPath",
    "OptionGroup",
    "OptionSpec",
    "option_param",
]
