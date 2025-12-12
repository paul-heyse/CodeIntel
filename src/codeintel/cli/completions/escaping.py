"""Unified shell escaping utilities for completion generators.

Provide consistent string escaping for all shell completion backends,
ensuring proper handling of special characters across different shells.
"""

from __future__ import annotations

from typing import Literal

ShellType = Literal["bash", "zsh", "fish", "powershell"]


def escape_for_shell(text: str, shell: ShellType) -> str:
    r"""Escape text for a specific shell.

    Route to the appropriate escape function based on the shell type.

    Parameters
    ----------
    text
        Text to escape.
    shell
        Target shell type.

    Returns
    -------
    str
        Escaped text suitable for the specified shell.

    Examples
    --------
    >>> escape_for_shell("it's a test", "bash")
    "it's a test"
    >>> escape_for_shell("it's a test", "zsh")
    "it'\\''s a test"
    """
    match shell:
        case "bash":
            return escape_bash(text)
        case "zsh":
            return escape_zsh(text)
        case "fish":
            return escape_fish(text)
        case "powershell":
            return escape_powershell(text)
        case _:
            return text


def escape_bash(text: str) -> str:
    r"""Escape text for bash completion descriptions.

    Bash completion descriptions in double-quoted strings need minimal escaping
    since most special characters are safe within double quotes.

    Parameters
    ----------
    text
        Text to escape.

    Returns
    -------
    str
        Escaped text for bash.

    Examples
    --------
    >>> escape_bash("simple text")
    'simple text'
    >>> escape_bash("path with $var")
    'path with \\$var'
    """
    return text.replace("\\", "\\\\").replace("$", "\\$").replace("`", "\\`").replace('"', '\\"')


def escape_zsh(text: str) -> str:
    r"""Escape text for zsh completion descriptions.

    Zsh completion descriptions in single-quoted strings require escaping
    of single quotes using the shell-standard approach.

    Parameters
    ----------
    text
        Text to escape.

    Returns
    -------
    str
        Escaped text for zsh.

    Examples
    --------
    >>> escape_zsh("simple text")
    'simple text'
    >>> escape_zsh("it's a test")
    "it'\\''s a test"
    """
    return text.replace("'", "'\\''")


def escape_fish(text: str) -> str:
    r"""Escape text for fish completion descriptions.

    Fish completion descriptions in single-quoted strings require escaping
    of single quotes with backslash.

    Parameters
    ----------
    text
        Text to escape.

    Returns
    -------
    str
        Escaped text for fish.

    Examples
    --------
    >>> escape_fish("simple text")
    'simple text'
    >>> escape_fish("it's a test")
    "it\\'s a test"
    """
    return text.replace("'", "\\'")


def escape_powershell(text: str) -> str:
    """Escape text for PowerShell completion descriptions.

    PowerShell completion descriptions in single-quoted strings require
    escaping of single quotes by doubling them.

    Parameters
    ----------
    text
        Text to escape.

    Returns
    -------
    str
        Escaped text for PowerShell.

    Examples
    --------
    >>> escape_powershell("simple text")
    'simple text'
    >>> escape_powershell("it's a test")
    "it''s a test"
    """
    return text.replace("'", "''")


class IndentManager:
    """Shared indentation management for all shells.

    Provide consistent indentation across completion generators.

    Parameters
    ----------
    base
        Base indentation string (default: 4 spaces).

    Examples
    --------
    >>> im = IndentManager()
    >>> im.indent(2)
    '        '
    >>> im = IndentManager(base="  ")
    >>> im.indent(3)
    '      '
    """

    def __init__(self, base: str = "    ") -> None:
        """Initialize indent manager with base indentation string."""
        self.base = base

    def indent(self, level: int) -> str:
        """Return indentation string for the specified level.

        Parameters
        ----------
        level
            Indentation level (0 = no indent, 1 = one base, etc).

        Returns
        -------
        str
            Indentation string for the level.
        """
        return self.base * level


__all__ = [
    "IndentManager",
    "ShellType",
    "escape_bash",
    "escape_fish",
    "escape_for_shell",
    "escape_powershell",
    "escape_zsh",
]
