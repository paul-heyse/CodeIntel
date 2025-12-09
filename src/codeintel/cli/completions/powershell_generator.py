"""PowerShell completion generator.

Generate PowerShell completion scripts for Windows.
"""

from __future__ import annotations

from codeintel.cli.completions.completion_model import CompletionModel


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
    lines: list[str] = [
        "# PowerShell completion for codeintel",
        "# Generated automatically - do not edit",
        "",
        "$CodeIntelCommands = @{",
    ]

    # Build command dictionary
    for cmd in model.root_command.subcommands:
        if cmd.subcommands:
            subcommands = ", ".join(f"'{sub.name}'" for sub in cmd.subcommands)
            lines.append(f"    '{cmd.name}' = @({subcommands})")
        else:
            lines.append(f"    '{cmd.name}' = @()")

    lines.extend(
        [
            "}",
            "",
            "$CodeIntelGlobalFlags = @(",
        ],
    )

    lines.extend(f"    '--{flag.name}'" for flag in model.global_flags)

    lines.extend(
        [
            ")",
            "",
            "Register-ArgumentCompleter -CommandName codeintel -ScriptBlock {",
            "    param(",
            "        $wordToComplete,",
            "        $commandAst,",
            "        $cursorPosition",
            "    )",
            "",
            "    $words = $commandAst.CommandElements | ForEach-Object { $_.ToString() }",
            "",
            "    if ($words.Count -eq 1) {",
            "        # Complete top-level commands",
            '        $CodeIntelCommands.Keys | Where-Object { $_ -like "$wordToComplete*" } |',
            "            ForEach-Object { "
            "[System.Management.Automation.CompletionResult]::new($_, $_, "
            "'ParameterValue', $_) }",
            "        return",
            "    }",
            "",
            "    $command = $words[1]",
            "    if ($CodeIntelCommands.ContainsKey($command)) {",
            "        if ($words.Count -eq 2) {",
            "            # Complete subcommands",
            '            $CodeIntelCommands[$command] | Where-Object { $_ -like "$wordToComplete*" } |',
            "                ForEach-Object { "
            "[System.Management.Automation.CompletionResult]::new($_, $_, "
            "'ParameterValue', $_) }",
            "        }",
            "    }",
            "",
            "    # Complete global flags",
            "    if ($wordToComplete.StartsWith('-')) {",
            '        $CodeIntelGlobalFlags | Where-Object { $_ -like "$wordToComplete*" } |',
            "            ForEach-Object { "
            "[System.Management.Automation.CompletionResult]::new($_, $_, "
            "'ParameterValue', $_) }",
            "    }",
            "}",
        ],
    )

    return "\n".join(lines)


__all__ = ["generate_powershell_completion"]
