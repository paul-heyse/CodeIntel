"""Zsh completion generator.

Generate zsh completion scripts with rich descriptions and grouping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.cli.completions.escaping import escape_zsh
from codeintel.cli.completions.generator import ShellBackend

if TYPE_CHECKING:
    from codeintel.cli.completions.completion_model import CommandSpec, CompletionModel


class ZshBackend(ShellBackend):
    """Zsh-specific completion backend.

    Attributes
    ----------
    _program
        Program name for completion script.
    _indent
        Base indentation string.
    """

    def __init__(self) -> None:
        """Initialize zsh backend with default configuration."""
        self._program = "codeintel"
        self._indent = "    "

    def header(self, model: CompletionModel) -> list[str]:
        """Generate zsh header.

        Returns
        -------
        list[str]
            Header lines for zsh completion script.
        """
        self._program = model.program_name
        return [
            f"#compdef {self._program}",
            "# Zsh completion for codeintel",
            "# Generated automatically - do not edit",
            "",
            "_codeintel() {",
            f"{self._indent}local context state state_descr line",
            f"{self._indent}typeset -A opt_args",
            "",
            f"{self._indent}_arguments -C \\",
        ]

    def global_flags(self, model: CompletionModel) -> list[str]:
        """Generate zsh global flags.

        Returns
        -------
        list[str]
            Global flag definition lines.
        """
        lines: list[str] = []
        double_indent = self._indent * 2
        for flag in model.global_flags:
            short = f"-{flag.short}" if flag.short else ""
            desc = escape_zsh(flag.description)
            if short:
                lines.append(f"{double_indent}'{short}[{desc}]' \\")
            lines.append(f"{double_indent}'--{flag.name}[{desc}]' \\")
        return lines

    def command(self, cmd: CommandSpec) -> list[str]:
        """Generate zsh command completion.

        Returns
        -------
        list[str]
            Command completion lines.
        """
        return _generate_zsh_command(cmd, indent=self._indent)

    def footer(self, model: CompletionModel) -> list[str]:
        """Generate zsh footer.

        Returns
        -------
        list[str]
            Footer lines including function structure and invocation.
        """
        double_indent = self._indent * 2
        triple_indent = self._indent * 3
        quad_indent = self._indent * 4
        lines: list[str] = [
            f"{double_indent}'1:command:->commands' \\",
            f"{double_indent}'*::arg:->args'",
            "",
            f"{self._indent}case $state in",
            f"{double_indent}commands)",
            f"{triple_indent}local -a commands",
            f"{triple_indent}commands=(",
        ]

        for cmd in model.root_command.subcommands:
            desc = escape_zsh(cmd.description)
            lines.append(f"{quad_indent}'{cmd.name}:{desc}'")

        lines.extend(
            [
                f"{triple_indent})",
                f"{triple_indent}_describe 'command' commands",
                f"{double_indent};;",
                f"{double_indent}args)",
                f"{triple_indent}case $words[1] in",
            ],
        )

        # Command cases are added via command() method, but we need to close the structure
        lines.extend(
            [
                f"{triple_indent}esac",
                f"{double_indent};;",
                f"{self._indent}esac",
                "}",
                "",
                "_codeintel",
            ],
        )
        return lines


def _generate_zsh_command(
    cmd: CommandSpec,
    *,
    indent: str = "    ",
) -> list[str]:
    """Generate zsh completion for command.

    Parameters
    ----------
    cmd
        Command specification.
    indent
        Base indentation string.
    Returns
    -------
    list[str]
        Zsh completion lines.
    """
    quad_indent = indent * 4
    five_indent = indent * 5
    six_indent = indent * 6
    lines: list[str] = [f"{quad_indent}{cmd.name})"]

    if cmd.subcommands:
        lines.extend(
            [
                f"{five_indent}local -a subcommands",
                f"{five_indent}subcommands=(",
            ],
        )
        for sub in cmd.subcommands:
            desc = escape_zsh(sub.description)
            lines.append(f"{six_indent}'{sub.name}:{desc}'")
        lines.extend(
            [
                f"{five_indent})",
                f"{five_indent}_describe 'subcommand' subcommands",
            ],
        )
    elif cmd.flags:
        lines.append(f"{five_indent}_arguments \\")
        for flag in cmd.flags:
            desc = escape_zsh(flag.description)
            if flag.takes_value:
                lines.append(f"{six_indent}'--{flag.name}=[{desc}]' \\")
            else:
                lines.append(f"{six_indent}'--{flag.name}[{desc}]' \\")

    lines.append(f"{five_indent};;")
    return lines


def generate_zsh_completion(model: CompletionModel) -> str:
    """Generate zsh completion script.

    Parameters
    ----------
    model
        Completion model.

    Returns
    -------
    str
        Zsh completion script.
    """
    # Note: ZshBackend uses a custom structure that doesn't fit the
    # simple header/commands/footer pattern well, so we use direct generation
    backend = ZshBackend()
    lines: list[str] = []
    lines.extend(backend.header(model))
    lines.extend(backend.global_flags(model))

    # Insert command cases between the header/flags and footer
    footer = backend.footer(model)
    # Insert commands before the "esac" in args case
    pre_footer = footer[:12]  # Up to "case $words[1] in"
    post_footer = footer[12:]  # After the args case starts

    lines.extend(pre_footer)
    for cmd in model.root_command.subcommands:
        lines.extend(backend.command(cmd))
    lines.extend(post_footer)

    return "\n".join(lines)


__all__ = ["ZshBackend", "generate_zsh_completion"]
