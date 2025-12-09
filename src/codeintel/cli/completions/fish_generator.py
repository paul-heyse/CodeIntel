"""Fish completion generator.

Generate fish shell completion scripts.
"""

from __future__ import annotations

from codeintel.cli.completions.completion_model import CommandSpec, CompletionModel


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
    lines: list[str] = [
        "# Fish completion for codeintel",
        "# Generated automatically - do not edit",
        "",
        "# Disable file completion by default",
        f"complete -c {model.program_name} -f",
        "",
        "# Global flags",
    ]

    # Global flags
    for flag in model.global_flags:
        parts = [f"complete -c {model.program_name}"]
        if flag.short:
            parts.append(f"-s {flag.short}")
        parts.append(f"-l {flag.name}")
        parts.append(f"-d '{_escape_fish_description(flag.description)}'")
        lines.append(" ".join(parts))

    lines.append("")
    lines.append("# Subcommands")

    # Top-level commands
    lines.extend(
        f"complete -c {model.program_name} -n '__fish_use_subcommand' "
        f"-a {cmd.name} -d '{_escape_fish_description(cmd.description)}'"
        for cmd in model.root_command.subcommands
    )

    lines.append("")

    # Subcommand completions
    for cmd in model.root_command.subcommands:
        lines.extend(_generate_fish_command(model.program_name, cmd))

    return "\n".join(lines)


def _escape_fish_description(desc: str) -> str:
    """Escape description for fish completion.

    Parameters
    ----------
    desc
        Description text.

    Returns
    -------
    str
        Escaped description.
    """
    return desc.replace("'", "\\'")


def _generate_fish_command(program: str, cmd: CommandSpec) -> list[str]:
    """Generate fish completion for command.

    Parameters
    ----------
    program
        Program name.
    cmd
        Command specification.

    Returns
    -------
    list[str]
        Fish completion lines.
    """
    lines: list[str] = [f"# {cmd.name} subcommands"]

    condition = f"__fish_seen_subcommand_from {cmd.name}"

    if cmd.subcommands:
        for sub in cmd.subcommands:
            lines.append(
                f"complete -c {program} -n '{condition}' "
                f"-a {sub.name} -d '{_escape_fish_description(sub.description)}'",
            )

            # Subcommand flags
            sub_condition = f"{condition}; and __fish_seen_subcommand_from {sub.name}"
            lines.extend(
                f"complete -c {program} -n '{sub_condition}' "
                f"-l {flag.name} -d '{_escape_fish_description(flag.description)}'"
                for flag in sub.flags
            )

    else:
        lines.extend(
            f"complete -c {program} -n '{condition}' "
            f"-l {flag.name} -d '{_escape_fish_description(flag.description)}'"
            for flag in cmd.flags
        )

    lines.append("")
    return lines


__all__ = ["generate_fish_completion"]
