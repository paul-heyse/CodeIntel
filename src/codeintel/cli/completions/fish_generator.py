"""Fish completion generator.

Generate fish shell completion scripts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.cli.completions.escaping import escape_fish
from codeintel.cli.completions.generator import ShellBackend, generate_with_backend

if TYPE_CHECKING:
    from codeintel.cli.completions.completion_model import CommandSpec, CompletionModel


class FishBackend(ShellBackend):
    """Fish-specific completion backend.

    Attributes
    ----------
    _program
        Program name for completion script.
    _complete_prefix
        Prefix for complete commands.
    """

    def __init__(self, program_name: str = "codeintel") -> None:
        """Initialize with program name."""
        self._program = program_name
        self._complete_prefix = f"complete -c {program_name}"

    def header(self, model: CompletionModel) -> list[str]:
        """Generate fish header.

        Returns
        -------
        list[str]
            Header lines for fish completion script.
        """
        self._program = model.program_name
        self._complete_prefix = f"complete -c {self._program}"
        return [
            "# Fish completion for codeintel",
            "# Generated automatically - do not edit",
            "",
            "# Disable file completion by default",
            f"{self._complete_prefix} -f",
            "",
            "# Global flags",
        ]

    def global_flags(self, model: CompletionModel) -> list[str]:
        """Generate fish global flags.

        Returns
        -------
        list[str]
            Global flag definition lines.
        """
        lines: list[str] = []
        for flag in model.global_flags:
            parts = [self._complete_prefix]
            if flag.short:
                parts.append(f"-s {flag.short}")
            parts.append(f"-l {flag.name}")
            parts.append(f"-d '{escape_fish(flag.description)}'")
            lines.append(" ".join(parts))
        lines.extend(["", "# Subcommands"])

        # Top-level commands
        lines.extend(
            f"{self._complete_prefix} -n '__fish_use_subcommand' "
            f"-a {cmd.name} -d '{escape_fish(cmd.description)}'"
            for cmd in model.root_command.subcommands
        )
        lines.append("")
        return lines

    def command(self, cmd: CommandSpec, depth: int) -> list[str]:
        """Generate fish command completion.

        Returns
        -------
        list[str]
            Command completion lines.
        """
        return _generate_fish_command(self._program, cmd, depth=depth)

    def footer(self, model: CompletionModel) -> list[str]:
        """Generate fish footer (empty for fish).

        Returns
        -------
        list[str]
            Empty list (fish doesn't need a footer).
        """
        # Use model to validate program consistency
        if model.program_name != self._program:
            self._program = model.program_name
        return []


def _generate_fish_command(
    program: str,
    cmd: CommandSpec,
    *,
    depth: int = 0,
) -> list[str]:
    """Generate fish completion for command.

    Parameters
    ----------
    program
        Program name.
    cmd
        Command specification.
    depth
        Nesting depth (unused but kept for protocol consistency).

    Returns
    -------
    list[str]
        Fish completion lines.
    """
    _ = depth  # Protocol consistency
    lines: list[str] = [f"# {cmd.name} subcommands"]

    condition = f"__fish_seen_subcommand_from {cmd.name}"

    if cmd.subcommands:
        for sub in cmd.subcommands:
            lines.append(
                f"complete -c {program} -n '{condition}' "
                f"-a {sub.name} -d '{escape_fish(sub.description)}'",
            )

            # Subcommand flags
            sub_condition = f"{condition}; and __fish_seen_subcommand_from {sub.name}"
            lines.extend(
                f"complete -c {program} -n '{sub_condition}' "
                f"-l {flag.name} -d '{escape_fish(flag.description)}'"
                for flag in sub.flags
            )

    else:
        lines.extend(
            f"complete -c {program} -n '{condition}' "
            f"-l {flag.name} -d '{escape_fish(flag.description)}'"
            for flag in cmd.flags
        )

    lines.append("")
    return lines


def generate_fish_completion(model: CompletionModel) -> str:
    """Generate fish completion script.

    Parameters
    ----------
    model
        Completion model.

    Returns
    -------
    str
        Fish completion script.
    """
    return generate_with_backend(model, FishBackend(model.program_name))


__all__ = ["FishBackend", "generate_fish_completion"]
