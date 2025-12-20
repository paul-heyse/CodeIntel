"""PowerShell completion generator.

Generate PowerShell completion scripts for Windows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.cli.completions.generator import ShellBackend, generate_with_backend

if TYPE_CHECKING:
    from codeintel.cli.completions.completion_model import CommandSpec, CompletionModel


class PowerShellBackend(ShellBackend):
    """PowerShell-specific completion backend.

    Attributes
    ----------
    _program
        Program name for completion script.
    _indent
        Base indentation string.
    _var_prefix
        Variable prefix for PowerShell variables.
    """

    def __init__(self) -> None:
        """Initialize PowerShell backend with default configuration."""
        self._program = "codeintel"
        self._indent = "    "
        self._var_prefix = "CodeIntel"

    def header(self, model: CompletionModel) -> list[str]:
        """Generate PowerShell header.

        Returns
        -------
        list[str]
            Header lines for PowerShell completion script.
        """
        self._program = model.program_name
        return [
            f"# PowerShell completion for {self._program}",
            "# Generated automatically - do not edit",
            "",
            f"${self._var_prefix}Commands = @{{",
        ]

    def global_flags(self, model: CompletionModel) -> list[str]:
        """Generate PowerShell global flags.

        Returns
        -------
        list[str]
            Global flag definition lines.
        """
        lines: list[str] = [
            "}",
            "",
            f"${self._var_prefix}GlobalFlags = @(",
        ]
        lines.extend(f"{self._indent}'--{flag.name}'" for flag in model.global_flags)
        lines.append(")")
        return lines

    def command(self, cmd: CommandSpec, depth: int) -> list[str]:
        """Generate PowerShell command entry.

        Returns
        -------
        list[str]
            Command entry line for the hashtable.
        """
        base_indent = self._indent * (depth + 1)
        if cmd.subcommands:
            subcommands = ", ".join(f"'{sub.name}'" for sub in cmd.subcommands)
            return [f"{base_indent}'{cmd.name}' = @({subcommands})"]
        return [f"{base_indent}'{cmd.name}' = @()"]

    def footer(self, model: CompletionModel) -> list[str]:
        """Generate PowerShell footer with argument completer.

        Returns
        -------
        list[str]
            Footer lines including the argument completer script block.
        """
        # Ensure program name is current
        if model.program_name != self._program:
            self._program = model.program_name

        cmds_var = f"${self._var_prefix}Commands"
        flags_var = f"${self._var_prefix}GlobalFlags"
        return [
            "",
            f"Register-ArgumentCompleter -CommandName {self._program} -ScriptBlock {{",
            f"{self._indent}param(",
            f"{self._indent}{self._indent}$wordToComplete,",
            f"{self._indent}{self._indent}$commandAst,",
            f"{self._indent}{self._indent}$cursorPosition",
            f"{self._indent})",
            "",
            f"{self._indent}$words = $commandAst.CommandElements | ForEach-Object {{ $_.ToString() }}",
            "",
            f"{self._indent}if ($words.Count -eq 1) {{",
            f"{self._indent}{self._indent}# Complete top-level commands",
            f'{self._indent}{self._indent}{cmds_var}.Keys | Where-Object {{ $_ -like "$wordToComplete*" }} |',
            (
                f"{self._indent}{self._indent}{self._indent}ForEach-Object {{ "
                "[System.Management.Automation.CompletionResult]::new($_, $_, "
                "'ParameterValue', $_) }}"
            ),
            f"{self._indent}{self._indent}return",
            f"{self._indent}}}",
            "",
            f"{self._indent}$command = $words[1]",
            f"{self._indent}if ({cmds_var}.ContainsKey($command)) {{",
            f"{self._indent}{self._indent}if ($words.Count -eq 2) {{",
            f"{self._indent}{self._indent}{self._indent}# Complete subcommands",
            f'{self._indent}{self._indent}{self._indent}{cmds_var}[$command] | Where-Object {{ $_ -like "$wordToComplete*" }} |',
            (
                f"{self._indent}{self._indent}{self._indent}{self._indent}ForEach-Object {{ "
                "[System.Management.Automation.CompletionResult]::new($_, $_, "
                "'ParameterValue', $_) }}"
            ),
            f"{self._indent}{self._indent}}}",
            f"{self._indent}}}",
            "",
            f"{self._indent}# Complete global flags",
            f"{self._indent}if ($wordToComplete.StartsWith('-')) {{",
            f'{self._indent}{self._indent}{flags_var} | Where-Object {{ $_ -like "$wordToComplete*" }} |',
            (
                f"{self._indent}{self._indent}{self._indent}ForEach-Object {{ "
                "[System.Management.Automation.CompletionResult]::new($_, $_, "
                "'ParameterValue', $_) }}"
            ),
            f"{self._indent}}}",
            "}",
        ]


def generate_powershell_completion(model: CompletionModel) -> str:
    """Generate PowerShell completion script.

    Parameters
    ----------
    model
        Completion model.

    Returns
    -------
    str
        PowerShell completion script.
    """
    return generate_with_backend(model, PowerShellBackend())


__all__ = ["PowerShellBackend", "generate_powershell_completion"]
