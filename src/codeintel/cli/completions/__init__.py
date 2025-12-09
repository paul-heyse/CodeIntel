"""Shell completion generation for CLI.

Provide auto-generated completions for bash, zsh, fish, and PowerShell.
"""

from __future__ import annotations

from enum import Enum

from codeintel.cli.completions.bash_generator import generate_bash_completion
from codeintel.cli.completions.completion_model import (
    CompletionModel,
    build_completion_model,
)
from codeintel.cli.completions.fish_generator import generate_fish_completion
from codeintel.cli.completions.powershell_generator import generate_powershell_completion
from codeintel.cli.completions.zsh_generator import generate_zsh_completion


class Shell(Enum):
    """Supported shells.

    Values
    ------
    BASH
        Bash shell.
    ZSH
        Zsh shell.
    FISH
        Fish shell.
    POWERSHELL
        PowerShell.
    """

    BASH = "bash"
    ZSH = "zsh"
    FISH = "fish"
    POWERSHELL = "powershell"


def generate_completion(shell: Shell) -> str:
    """Generate completion script for shell.

    Parameters
    ----------
    shell
        Target shell.

    Returns
    -------
    str
        Completion script.
    """
    model = build_completion_model()

    generators = {
        Shell.BASH: generate_bash_completion,
        Shell.ZSH: generate_zsh_completion,
        Shell.FISH: generate_fish_completion,
        Shell.POWERSHELL: generate_powershell_completion,
    }

    return generators[shell](model)


def get_install_instructions(shell: Shell) -> str:
    """Get installation instructions for shell.

    Parameters
    ----------
    shell
        Target shell.

    Returns
    -------
    str
        Installation instructions.
    """
    instructions = {
        Shell.BASH: """# Add to ~/.bashrc:
source <(codeintel completions bash)

# Or save to file:
codeintel completions bash > ~/.local/share/bash-completion/completions/codeintel""",
        Shell.ZSH: """# Add to ~/.zshrc:
source <(codeintel completions zsh)

# Or save to fpath:
codeintel completions zsh > ~/.zsh/completions/_codeintel""",
        Shell.FISH: """# Save to fish completions directory:
codeintel completions fish > ~/.config/fish/completions/codeintel.fish""",
        Shell.POWERSHELL: """# Add to $PROFILE:
codeintel completions powershell | Out-String | Invoke-Expression

# Or save to module:
codeintel completions powershell > $HOME/Documents/PowerShell/Modules/CodeIntel/CodeIntel.psm1""",
    }

    return instructions[shell]


__all__ = [
    "CompletionModel",
    "Shell",
    "build_completion_model",
    "generate_completion",
    "get_install_instructions",
]
