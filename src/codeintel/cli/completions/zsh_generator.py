"""Zsh completion generator.

Generate zsh completion scripts with rich descriptions and grouping.
"""

from __future__ import annotations

from codeintel.cli.completions.completion_model import CommandSpec, CompletionModel


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
    lines: list[str] = [
        f"#compdef {model.program_name}",
        "# Zsh completion for codeintel",
        "# Generated automatically - do not edit",
        "",
        "_codeintel() {",
        "    local context state state_descr line",
        "    typeset -A opt_args",
        "",
        "    _arguments -C \\",
    ]

    # Global flags
    for flag in model.global_flags:
        short = f"-{flag.short}" if flag.short else ""
        desc = _escape_zsh_description(flag.description)
        if short:
            lines.append(f"        '{short}[{desc}]' \\")
        lines.append(f"        '--{flag.name}[{desc}]' \\")

    # Subcommands
    lines.extend(
        [
            "        '1:command:->commands' \\",
            "        '*::arg:->args'",
            "",
            "    case $state in",
            "        commands)",
            "            local -a commands",
            "            commands=(",
        ],
    )

    for cmd in model.root_command.subcommands:
        desc = _escape_zsh_description(cmd.description)
        lines.append(f"                '{cmd.name}:{desc}'")

    lines.extend(
        [
            "            )",
            "            _describe 'command' commands",
            "            ;;",
            "        args)",
            "            case $words[1] in",
        ],
    )

    # Command-specific completions
    for cmd in model.root_command.subcommands:
        lines.extend(_generate_zsh_command(cmd))

    lines.extend(
        [
            "            esac",
            "            ;;",
            "    esac",
            "}",
            "",
            "_codeintel",
        ],
    )

    return "\n".join(lines)


def _escape_zsh_description(desc: str) -> str:
    """Escape description for zsh completion.

    Parameters
    ----------
    desc
        Description text.

    Returns
    -------
    str
        Escaped description.
    """
    return desc.replace("'", "'\\''")


def _generate_zsh_command(cmd: CommandSpec) -> list[str]:
    """Generate zsh completion for command.

    Parameters
    ----------
    cmd
        Command specification.

    Returns
    -------
    list[str]
        Zsh completion lines.
    """
    lines: list[str] = [f"                {cmd.name})"]

    if cmd.subcommands:
        lines.extend(
            [
                "                    local -a subcommands",
                "                    subcommands=(",
            ],
        )
        for sub in cmd.subcommands:
            desc = _escape_zsh_description(sub.description)
            lines.append(f"                        '{sub.name}:{desc}'")
        lines.extend(
            [
                "                    )",
                "                    _describe 'subcommand' subcommands",
            ],
        )
    elif cmd.flags:
        lines.append("                    _arguments \\")
        for flag in cmd.flags:
            desc = _escape_zsh_description(flag.description)
            if flag.takes_value:
                lines.append(f"                        '--{flag.name}=[{desc}]' \\")
            else:
                lines.append(f"                        '--{flag.name}[{desc}]' \\")

    lines.append("                    ;;")
    return lines


__all__ = ["generate_zsh_completion"]
