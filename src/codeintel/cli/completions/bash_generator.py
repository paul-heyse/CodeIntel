"""Bash completion generator.

Generate bash completion scripts from the completion model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.cli.completions.generator import ShellBackend, generate_with_backend

if TYPE_CHECKING:
    from codeintel.cli.completions.completion_model import CommandSpec, CompletionModel


class BashBackend(ShellBackend):
    """Bash-specific completion backend.

    Attributes
    ----------
    _program
        Program name for completion script.
    _indent
        Base indentation string.
    """

    def __init__(self) -> None:
        """Initialize bash backend with default configuration."""
        self._program = "codeintel"
        self._indent = "    "
        self._case_started = False

    def header(self, model: CompletionModel) -> list[str]:
        """Generate bash header.

        Returns
        -------
        list[str]
            Header lines for bash completion script.
        """
        self._program = model.program_name
        commands = " ".join(cmd.name for cmd in model.root_command.subcommands)
        return [
            "# Bash completion for codeintel",
            "# Generated automatically - do not edit",
            "",
            f"_{self._program}_completions() {{",
            f"{self._indent}local cur prev words cword",
            f"{self._indent}_init_completion || return",
            "",
            f'{self._indent}local commands="{commands}"',
            "",
        ]

    def global_flags(self, model: CompletionModel) -> list[str]:
        """Generate bash global flags.

        Returns
        -------
        list[str]
            Global flag definition lines.
        """
        flags = " ".join(
            f"--{f.name}" + (f" -{f.short}" if f.short else "") for f in model.global_flags
        )
        return [f'{self._indent}local global_flags="{flags}"', ""]

    def command(self, cmd: CommandSpec) -> list[str]:
        """Generate bash command completion.

        Returns
        -------
        list[str]
            Command completion lines.
        """
        lines: list[str] = []
        if not self._case_started:
            lines.append(f"{self._indent}case ${{words[1]}} in")
            self._case_started = True

        lines.extend(_generate_command_case(cmd, depth=2, indent=self._indent))
        return lines

    def footer(self, model: CompletionModel) -> list[str]:
        """Generate bash footer.

        Returns
        -------
        list[str]
            Footer lines including function close and complete command.
        """
        # Ensure program name is current
        if model.program_name != self._program:
            self._program = model.program_name

        return [
            f"{self._indent}*)",
            f'{self._indent}{self._indent}COMPREPLY=($(compgen -W "$commands $global_flags" -- $cur))',
            f"{self._indent}{self._indent};;",
            f"{self._indent}esac",
            "}",
            "",
            f"complete -F _{self._program}_completions {self._program}",
        ]


def _generate_command_case(
    cmd: CommandSpec,
    depth: int,
    *,
    indent: str = "    ",
) -> list[str]:
    """Generate case block for command.

    Parameters
    ----------
    cmd
        Command specification.
    depth
        Nesting depth.
    indent
        Base indentation string.

    Returns
    -------
    list[str]
        Case block lines.
    """
    lines: list[str] = []
    base = indent * depth

    lines.append(f"{base}{cmd.name})")

    if cmd.subcommands:
        subcommand_names = " ".join(sub.name for sub in cmd.subcommands)
        lines.append(f"{base}{indent}case ${{words[2]}} in")

        for sub in cmd.subcommands:
            sub_flags = " ".join(f"--{f.name}" for f in sub.flags)
            lines.extend(
                [
                    f"{base}{indent}{sub.name})",
                    f'{base}{indent}{indent}COMPREPLY=($(compgen -W "{sub_flags}" -- $cur))',
                    f"{base}{indent}{indent};;",
                ],
            )

        lines.extend(
            [
                f"{base}{indent}*)",
                f'{base}{indent}{indent}COMPREPLY=($(compgen -W "{subcommand_names}" -- $cur))',
                f"{base}{indent}{indent};;",
                f"{base}{indent}esac",
            ],
        )
    else:
        flags = " ".join(f"--{f.name}" for f in cmd.flags)
        lines.append(f'{base}{indent}COMPREPLY=($(compgen -W "{flags}" -- $cur))')

    lines.append(f"{base}{indent};;")
    return lines


def generate_bash_completion(model: CompletionModel) -> str:
    """Generate bash completion script.

    Parameters
    ----------
    model
        Completion model.

    Returns
    -------
    str
        Bash completion script.
    """
    return generate_with_backend(model, BashBackend())


__all__ = ["BashBackend", "generate_bash_completion"]
