"""Bash completion generator.

Generate bash completion scripts from the completion model.
"""

from __future__ import annotations

from codeintel.cli.completions.completion_model import CommandSpec, CompletionModel


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
    lines: list[str] = [
        "# Bash completion for codeintel",
        "# Generated automatically - do not edit",
        "",
        f"_{model.program_name}_completions() {{",
        "    local cur prev words cword",
        "    _init_completion || return",
        "",
        '    local commands="' + " ".join(cmd.name for cmd in model.root_command.subcommands) + '"',
        "",
    ]

    # Add global flags
    global_flags = " ".join(
        f"--{f.name}" + (f" -{f.short}" if f.short else "") for f in model.global_flags
    )
    lines.append(f'    local global_flags="{global_flags}"')
    lines.append("")

    # Generate command-specific completions
    lines.append("    case ${words[1]} in")

    for cmd in model.root_command.subcommands:
        lines.extend(_generate_command_case(cmd, depth=2))

    lines.extend(
        [
            "    *)",
            '        COMPREPLY=($(compgen -W "$commands $global_flags" -- $cur))',
            "        ;;",
            "    esac",
            "}",
            "",
            f"complete -F _{model.program_name}_completions {model.program_name}",
        ],
    )

    return "\n".join(lines)


def _generate_command_case(cmd: CommandSpec, depth: int) -> list[str]:
    """Generate case block for command.

    Parameters
    ----------
    cmd
        Command specification.
    depth
        Nesting depth.

    Returns
    -------
    list[str]
        Case block lines.
    """
    lines: list[str] = []
    indent = "    " * depth

    lines.append(f"{indent}{cmd.name})")

    if cmd.subcommands:
        subcommand_names = " ".join(sub.name for sub in cmd.subcommands)
        lines.append(f"{indent}    case ${{words[2]}} in")

        for sub in cmd.subcommands:
            sub_flags = " ".join(f"--{f.name}" for f in sub.flags)
            lines.extend(
                [
                    f"{indent}    {sub.name})",
                    f'{indent}        COMPREPLY=($(compgen -W "{sub_flags}" -- $cur))',
                    f"{indent}        ;;",
                ],
            )

        lines.extend(
            [
                f"{indent}    *)",
                f'{indent}        COMPREPLY=($(compgen -W "{subcommand_names}" -- $cur))',
                f"{indent}        ;;",
                f"{indent}    esac",
            ],
        )
    else:
        flags = " ".join(f"--{f.name}" for f in cmd.flags)
        lines.append(f'{indent}    COMPREPLY=($(compgen -W "{flags}" -- $cur))')

    lines.append(f"{indent}    ;;")
    return lines


__all__ = ["generate_bash_completion"]
