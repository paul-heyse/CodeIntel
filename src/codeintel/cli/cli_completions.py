"""Shell completion generation for CodeIntel CLI.

Provides completion scripts for Bash, Zsh, and Fish shells,
including dynamic completions for operations and datasets.
"""

from __future__ import annotations

import os
import sys
from enum import Enum
from pathlib import Path


class Shell(Enum):
    """Supported shells for completion."""

    BASH = "bash"
    ZSH = "zsh"
    FISH = "fish"


# Bash completion script template
BASH_COMPLETION_TEMPLATE = """
# CodeIntel CLI completion for Bash
# Generated automatically - do not edit

_codeintel_completions() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Top-level commands
    local commands="op dataset build graph docs config storage"

    # Subcommands by parent
    local op_commands="list call"
    local dataset_commands="list describe verify"
    local build_commands="status run clean"
    local graph_commands="stats query export"
    local docs_commands="build serve"
    local config_commands="show set"
    local storage_commands="init migrate status"

    # Dynamic completions
    case "${prev}" in
        codeintel)
            COMPREPLY=( $(compgen -W "${commands}" -- ${cur}) )
            return 0
            ;;
        op)
            COMPREPLY=( $(compgen -W "${op_commands}" -- ${cur}) )
            return 0
            ;;
        call)
            # Complete operation IDs dynamically
            local ops=$(codeintel op list --format=json 2>/dev/null | jq -r '.[].id' 2>/dev/null)
            COMPREPLY=( $(compgen -W "${ops}" -- ${cur}) )
            return 0
            ;;
        dataset)
            COMPREPLY=( $(compgen -W "${dataset_commands}" -- ${cur}) )
            return 0
            ;;
        describe|verify)
            # Complete table keys dynamically
            local tables=$(codeintel dataset list --format=json 2>/dev/null | jq -r '.[].table_key' 2>/dev/null)
            COMPREPLY=( $(compgen -W "${tables}" -- ${cur}) )
            return 0
            ;;
        build)
            COMPREPLY=( $(compgen -W "${build_commands}" -- ${cur}) )
            return 0
            ;;
        graph)
            COMPREPLY=( $(compgen -W "${graph_commands}" -- ${cur}) )
            return 0
            ;;
        docs)
            COMPREPLY=( $(compgen -W "${docs_commands}" -- ${cur}) )
            return 0
            ;;
        config)
            COMPREPLY=( $(compgen -W "${config_commands}" -- ${cur}) )
            return 0
            ;;
        storage)
            COMPREPLY=( $(compgen -W "${storage_commands}" -- ${cur}) )
            return 0
            ;;
        --db-path|--repo-root|--build-dir)
            # Path completion
            COMPREPLY=( $(compgen -f -- ${cur}) )
            return 0
            ;;
        --format)
            COMPREPLY=( $(compgen -W "text json" -- ${cur}) )
            return 0
            ;;
    esac

    # Global options
    local global_opts="--help --version --format --verbose --quiet"
    if [[ ${cur} == -* ]]; then
        COMPREPLY=( $(compgen -W "${global_opts}" -- ${cur}) )
        return 0
    fi
}

complete -F _codeintel_completions codeintel
"""

# Zsh completion script template
ZSH_COMPLETION_TEMPLATE = """#compdef codeintel

# CodeIntel CLI completion for Zsh
# Generated automatically - do not edit

_codeintel() {
    local -a commands
    commands=(
        'op:Operation management'
        'dataset:Dataset inspection'
        'build:Build management'
        'graph:Graph operations'
        'docs:Documentation'
        'config:Configuration'
        'storage:Storage management'
    )

    local -a op_commands
    op_commands=(
        'list:List available operations'
        'call:Call an operation'
    )

    local -a dataset_commands
    dataset_commands=(
        'list:List available datasets'
        'describe:Describe a dataset'
        'verify:Verify dataset integrity'
    )

    local -a build_commands
    build_commands=(
        'status:Show build status'
        'run:Run build targets'
        'clean:Clean build artifacts'
    )

    _arguments -C \\
        '1: :->command' \\
        '2: :->subcommand' \\
        '3: :->argument' \\
        '--help[Show help]' \\
        '--version[Show version]' \\
        '--format[Output format]:format:(text json)' \\
        '--verbose[Verbose output]' \\
        '--quiet[Quiet output]'

    case $state in
        command)
            _describe -t commands 'command' commands
            ;;
        subcommand)
            case $words[2] in
                op)
                    _describe -t op_commands 'op command' op_commands
                    ;;
                dataset)
                    _describe -t dataset_commands 'dataset command' dataset_commands
                    ;;
                build)
                    _describe -t build_commands 'build command' build_commands
                    ;;
            esac
            ;;
        argument)
            case $words[2]:$words[3] in
                op:call)
                    # Dynamic operation completion
                    local -a operations
                    operations=(${(f)"$(codeintel op list --format=json 2>/dev/null | jq -r '.[].id' 2>/dev/null)"})
                    _describe -t operations 'operation' operations
                    ;;
                dataset:describe|dataset:verify)
                    # Dynamic table completion
                    local -a tables
                    tables=(${(f)"$(codeintel dataset list --format=json 2>/dev/null | jq -r '.[].table_key' 2>/dev/null)"})
                    _describe -t tables 'table' tables
                    ;;
            esac
            ;;
    esac
}

_codeintel "$@"
"""

# Fish completion script template
FISH_COMPLETION_TEMPLATE = """# CodeIntel CLI completion for Fish
# Generated automatically - do not edit

# Disable file completion by default
complete -c codeintel -f

# Top-level commands
complete -c codeintel -n "__fish_use_subcommand" -a "op" -d "Operation management"
complete -c codeintel -n "__fish_use_subcommand" -a "dataset" -d "Dataset inspection"
complete -c codeintel -n "__fish_use_subcommand" -a "build" -d "Build management"
complete -c codeintel -n "__fish_use_subcommand" -a "graph" -d "Graph operations"
complete -c codeintel -n "__fish_use_subcommand" -a "docs" -d "Documentation"
complete -c codeintel -n "__fish_use_subcommand" -a "config" -d "Configuration"
complete -c codeintel -n "__fish_use_subcommand" -a "storage" -d "Storage management"

# Op subcommands
complete -c codeintel -n "__fish_seen_subcommand_from op" -a "list" -d "List operations"
complete -c codeintel -n "__fish_seen_subcommand_from op" -a "call" -d "Call an operation"

# Dataset subcommands
complete -c codeintel -n "__fish_seen_subcommand_from dataset" -a "list" -d "List datasets"
complete -c codeintel -n "__fish_seen_subcommand_from dataset" -a "describe" -d "Describe dataset"
complete -c codeintel -n "__fish_seen_subcommand_from dataset" -a "verify" -d "Verify dataset"

# Build subcommands
complete -c codeintel -n "__fish_seen_subcommand_from build" -a "status" -d "Build status"
complete -c codeintel -n "__fish_seen_subcommand_from build" -a "run" -d "Run build"
complete -c codeintel -n "__fish_seen_subcommand_from build" -a "clean" -d "Clean build"

# Dynamic operation completion for 'op call'
complete -c codeintel -n "__fish_seen_subcommand_from call" -a "(codeintel op list --format=json 2>/dev/null | jq -r '.[].id' 2>/dev/null)"

# Dynamic table completion for 'dataset describe/verify'
complete -c codeintel -n "__fish_seen_subcommand_from describe" -a "(codeintel dataset list --format=json 2>/dev/null | jq -r '.[].table_key' 2>/dev/null)"
complete -c codeintel -n "__fish_seen_subcommand_from verify" -a "(codeintel dataset list --format=json 2>/dev/null | jq -r '.[].table_key' 2>/dev/null)"

# Global options
complete -c codeintel -l help -d "Show help"
complete -c codeintel -l version -d "Show version"
complete -c codeintel -l format -d "Output format" -a "text json"
complete -c codeintel -l verbose -d "Verbose output"
complete -c codeintel -l quiet -d "Quiet output"
"""


def generate_completion(shell: Shell) -> str:
    """Generate completion script for a shell.

    Parameters
    ----------
    shell
        Target shell.

    Returns
    -------
    str
        Completion script content.
    """
    templates = {
        Shell.BASH: BASH_COMPLETION_TEMPLATE,
        Shell.ZSH: ZSH_COMPLETION_TEMPLATE,
        Shell.FISH: FISH_COMPLETION_TEMPLATE,
    }
    return templates[shell].strip()


def get_completion_install_path(shell: Shell) -> Path:
    """Get the recommended installation path for completions.

    Parameters
    ----------
    shell
        Target shell.

    Returns
    -------
    Path
        Recommended installation path.
    """
    home = Path.home()

    paths = {
        Shell.BASH: home / ".bash_completion.d" / "codeintel",
        Shell.ZSH: home / ".zsh" / "completions" / "_codeintel",
        Shell.FISH: home / ".config" / "fish" / "completions" / "codeintel.fish",
    }
    return paths[shell]


def install_completion(shell: Shell, *, force: bool = False) -> Path:
    """Install completion script for a shell.

    Parameters
    ----------
    shell
        Target shell.
    force
        Overwrite existing file.

    Returns
    -------
    Path
        Path where completion was installed.

    Raises
    ------
    FileExistsError
        If completion exists and force is False.
    """
    path = get_completion_install_path(shell)

    if path.exists() and not force:
        msg = f"Completion already exists: {path}"
        raise FileExistsError(msg)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_completion(shell))

    return path


def detect_shell() -> Shell | None:
    """Detect the current shell.

    Returns
    -------
    Shell | None
        Detected shell or None if unknown.
    """
    shell_path = os.environ.get("SHELL", "")

    if "bash" in shell_path:
        return Shell.BASH
    if "zsh" in shell_path:
        return Shell.ZSH
    if "fish" in shell_path:
        return Shell.FISH

    return None


def print_completion(shell: Shell) -> None:
    """Print completion script to stdout.

    Parameters
    ----------
    shell
        Target shell.
    """
    sys.stdout.write(generate_completion(shell))
    sys.stdout.write("\n")


__all__ = [
    "Shell",
    "detect_shell",
    "generate_completion",
    "get_completion_install_path",
    "install_completion",
    "print_completion",
]
