"""CLI commands for completion generation."""

from __future__ import annotations

import sys
from typing import Annotated

import cyclopts

from codeintel.cli.completions import Shell, generate_completion, get_install_instructions

completions_app = cyclopts.App(name="completions", help="Shell completion generation")


@completions_app.command()
def bash() -> None:
    """Generate bash completion script.

    Print bash completion script to stdout. Redirect to a file
    or source directly for completion support.

    Examples
    --------
    codeintel completions bash > ~/.local/share/bash-completion/completions/codeintel
    source <(codeintel completions bash)
    """
    print(generate_completion(Shell.BASH))  # noqa: T201


@completions_app.command()
def zsh() -> None:
    """Generate zsh completion script.

    Print zsh completion script to stdout. Save to a file
    in your fpath for completion support.

    Examples
    --------
    codeintel completions zsh > ~/.zsh/completions/_codeintel
    """
    print(generate_completion(Shell.ZSH))  # noqa: T201


@completions_app.command()
def fish() -> None:
    """Generate fish completion script.

    Print fish completion script to stdout. Save to
    fish completions directory.

    Examples
    --------
    codeintel completions fish > ~/.config/fish/completions/codeintel.fish
    """
    print(generate_completion(Shell.FISH))  # noqa: T201


@completions_app.command()
def powershell() -> None:
    """Generate PowerShell completion script.

    Print PowerShell completion script to stdout.
    Can be invoked directly or saved to a module.

    Examples
    --------
    codeintel completions powershell | Out-String | Invoke-Expression
    """
    print(generate_completion(Shell.POWERSHELL))  # noqa: T201


@completions_app.command()
def install(
    shell: Annotated[str, cyclopts.Parameter(help="Shell to install for")],
) -> None:
    """Show installation instructions for shell.

    Display instructions for installing completions for the
    specified shell.

    Parameters
    ----------
    shell
        Target shell (bash, zsh, fish, powershell).

    Raises
    ------
    SystemExit
        If shell is unknown.

    Examples
    --------
    codeintel completions install bash
    """
    shell_lower = shell.lower()

    # Validate shell name
    valid_shells = {s.value for s in Shell}
    if shell_lower not in valid_shells:
        print(f"Unknown shell: {shell}", file=sys.stderr)  # noqa: T201
        print(f"Supported: {', '.join(sorted(valid_shells))}", file=sys.stderr)  # noqa: T201
        raise SystemExit(1)

    shell_enum = Shell(shell_lower)
    print(get_install_instructions(shell_enum))  # noqa: T201


__all__ = ["completions_app"]
