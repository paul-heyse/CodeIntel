"""Explicit env var naming for CLI options."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from cyclopts import config as cyclopts_config

if TYPE_CHECKING:
    from cyclopts.argument import Argument


_ENV_PREFIX = "CODEINTEL_"


def _normalize_env_token(value: str) -> str:
    """Normalize a command/argument token to env-var format.

    Returns
    -------
    str
        Normalized token suitable for env-var naming.
    """
    return value.upper().replace("-", "_").replace(".", "_").lstrip("_")


def build_env_var_name(
    command_chain: Iterable[str],
    argument_name: str,
    *,
    prefix: str = _ENV_PREFIX,
) -> str:
    """Build a stable env var name for a CLI argument.

    Parameters
    ----------
    command_chain
        Command path segments (e.g., ("build", "run")).
    argument_name
        Raw argument name from Cyclopts.
    prefix
        Env var prefix to apply (defaults to CODEINTEL_).

    Returns
    -------
    str
        Fully-qualified env var name (e.g., CODEINTEL_BUILD_RUN_TARGETS).
    """
    chain = "_".join(_normalize_env_token(part) for part in command_chain if part)
    suffix = _normalize_env_token(argument_name)
    if chain:
        return f"{prefix}{chain}_{suffix}"
    return f"{prefix}{suffix}"


class CodeIntelEnv(cyclopts_config.Env):
    """Env config that uses explicit CodeIntel env-var naming."""

    def _convert_argument(
        self,
        commands: tuple[str, ...],
        argument: Argument,
    ) -> str:
        prefix = self.prefix or _ENV_PREFIX
        return build_env_var_name(commands, str(argument.name), prefix=prefix)


__all__ = ["CodeIntelEnv", "build_env_var_name"]
