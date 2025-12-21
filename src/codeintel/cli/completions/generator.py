"""Unified completion generator with shell-specific backends.

Provide a protocol-based approach to shell completion generation,
allowing shell-specific rendering while sharing common traversal logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from codeintel.cli.completions.completion_model import CommandSpec, CompletionModel


class ShellBackend(Protocol):
    """Protocol for shell-specific completion rendering.

    Each shell implementation provides methods that render completion
    script fragments for different parts of the CLI structure.
    """

    def header(self, model: CompletionModel) -> list[str]:
        """Generate header lines for the completion script.

        Parameters
        ----------
        model
            Completion model.

        Returns
        -------
        list[str]
            Header lines.
        """
        ...

    def global_flags(self, model: CompletionModel) -> list[str]:
        """Generate completion lines for global flags.

        Parameters
        ----------
        model
            Completion model.

        Returns
        -------
        list[str]
            Global flag completion lines.
        """
        ...

    def command(self, cmd: CommandSpec) -> list[str]:
        """Generate completion lines for a command.

        Parameters
        ----------
        cmd
            Command specification.

        Returns
        -------
        list[str]
            Command completion lines.
        """
        ...

    def footer(self, model: CompletionModel) -> list[str]:
        """Generate footer lines for the completion script.

        Parameters
        ----------
        model
            Completion model.

        Returns
        -------
        list[str]
            Footer lines.
        """
        ...


def generate_with_backend(model: CompletionModel, backend: ShellBackend) -> str:
    """Generate completion script using the specified backend.

    Parameters
    ----------
    model
        Completion model.
    backend
        Shell-specific backend.

    Returns
    -------
    str
        Complete shell completion script.
    """
    lines: list[str] = []
    lines.extend(backend.header(model))
    lines.extend(backend.global_flags(model))
    for cmd in model.root_command.subcommands:
        lines.extend(backend.command(cmd))
    lines.extend(backend.footer(model))
    return "\n".join(lines)


__all__ = [
    "ShellBackend",
    "generate_with_backend",
]
