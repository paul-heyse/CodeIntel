# CLI Best-in-Class Implementation Plan (Phase 7 - Part 2)

> **Status**: Proposed  
> **Author**: AI Assistant  
> **Created**: 2025-12-09  
> **Depends On**: Phase 6 (Completed), Phase 7 Part 1

---

## Executive Summary

Phase 7 Part 2 focuses on **developer experience** and **extensibility** — shell completions that make the CLI discoverable and a hardened plugin system that enables ecosystem growth.

The two priorities addressed:

1. **Shell Completion Generation** — Auto-generated completions for bash, zsh, fish, PowerShell
2. **Plugin Ecosystem Hardening** — Versioning, permissions, sandboxing, and testing utilities

### Why These Priorities Matter

| Aspect | Current State | After Phase 7.3-7.4 |
|--------|---------------|---------------------|
| Shell Completions | Manual/none | Auto-generated from registry |
| Bash Support | Limited | Full subcommand + flag completion |
| Zsh Support | None | Rich completions with descriptions |
| Fish Support | None | Native fish completions |
| PowerShell Support | None | Windows-native completions |
| Plugin Versioning | None | Semantic versioning checks |
| Plugin Permissions | None | Capability-based access control |
| Plugin Testing | None | Built-in test harness |
| Plugin Templates | None | Scaffolding commands |

---

## Table of Contents

1. [Phase 7.3: Shell Completion Generation](#phase-73-shell-completion-generation)
2. [Phase 7.4: Plugin Ecosystem Hardening](#phase-74-plugin-ecosystem-hardening)
3. [Implementation Timeline](#implementation-timeline)
4. [Success Metrics](#success-metrics)

---

## Phase 7.3: Shell Completion Generation

### Value Proposition

Shell completions dramatically improve CLI usability:

- **Discoverability** — Users find commands without documentation
- **Accuracy** — Completions prevent typos and invalid values
- **Speed** — Tab completion is faster than typing
- **Context-aware** — Dynamic completions based on project state

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Shell Completion Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Operation Registry                              │  │
│  │                                                                    │  │
│  │   • operation_id: "build.status"                                  │  │
│  │   • description: "Show build target status"                        │  │
│  │   • param_schema: {project_root: PathValidator, ...}              │  │
│  └────────────────────────────────┬──────────────────────────────────┘  │
│                                   │                                      │
│                                   ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                   CompletionGenerator                              │  │
│  │                                                                    │  │
│  │   generate_bash()  → bash_completion.sh                           │  │
│  │   generate_zsh()   → _codeintel                                   │  │
│  │   generate_fish()  → codeintel.fish                               │  │
│  │   generate_pwsh()  → CodeIntel.psm1                               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  Dynamic        │  │  Static         │  │  Project        │         │
│  │  Completions    │  │  Completions    │  │  Context        │         │
│  │                 │  │                 │  │                 │         │
│  │ • File paths    │  │ • Operation IDs │  │ • Symbol names  │         │
│  │ • Op params     │  │ • Subcommands   │  │ • Module paths  │         │
│  │ • Config values │  │ • Global flags  │  │ • Target names  │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Functional Objectives

1. Create completion generator framework with shell-agnostic model
2. Implement Bash completion with subcommand and flag support
3. Implement Zsh completion with descriptions and groups
4. Implement Fish completion with native syntax
5. Implement PowerShell completion for Windows
6. Add dynamic completion support for context-aware values
7. Create CLI commands to generate and install completions

### Implementation

#### File: `src/codeintel/cli/completions/completion_model.py`

```python
"""Shell-agnostic completion model.

Provides a unified model for completions that can be rendered
to any shell's completion format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CompletionType(Enum):
    """Types of completion values."""

    LITERAL = "literal"  # Fixed string value
    FILE = "file"  # File path completion
    DIRECTORY = "directory"  # Directory path completion
    COMMAND = "command"  # Subcommand completion
    OPERATION = "operation"  # Operation ID completion
    DYNAMIC = "dynamic"  # Dynamic completion from function


@dataclass(frozen=True)
class CompletionValue:
    """Single completion value.

    Parameters
    ----------
    value
        The completion value.
    description
        Description for the value.
    """

    value: str
    description: str = ""


@dataclass
class CompletionSpec:
    """Specification for a completion context.

    Parameters
    ----------
    completion_type
        Type of completion.
    values
        Static completion values.
    dynamic_source
        Function name for dynamic values.
    """

    completion_type: CompletionType
    values: list[CompletionValue] = field(default_factory=list)
    dynamic_source: str | None = None


@dataclass
class FlagSpec:
    """Specification for a command flag.

    Parameters
    ----------
    name
        Flag name (without dashes).
    short
        Short flag (single character).
    description
        Flag description.
    takes_value
        Whether flag takes a value.
    value_completion
        Completion spec for value.
    required
        Whether flag is required.
    """

    name: str
    short: str | None = None
    description: str = ""
    takes_value: bool = False
    value_completion: CompletionSpec | None = None
    required: bool = False


@dataclass
class CommandSpec:
    """Specification for a command or subcommand.

    Parameters
    ----------
    name
        Command name.
    description
        Command description.
    subcommands
        Child subcommands.
    flags
        Command flags.
    positional_completion
        Completion for positional arguments.
    """

    name: str
    description: str = ""
    subcommands: list[CommandSpec] = field(default_factory=list)
    flags: list[FlagSpec] = field(default_factory=list)
    positional_completion: CompletionSpec | None = None


@dataclass
class CompletionModel:
    """Complete model for CLI completions.

    Parameters
    ----------
    program_name
        Name of the CLI program.
    root_command
        Root command specification.
    global_flags
        Flags available on all commands.
    """

    program_name: str
    root_command: CommandSpec
    global_flags: list[FlagSpec] = field(default_factory=list)


def build_completion_model() -> CompletionModel:
    """Build completion model from operation registry.

    Returns
    -------
    CompletionModel
        Complete completion model.
    """
    from codeintel.cli.operation_registry import get_operation_registry

    registry = get_operation_registry()

    # Build subcommands from registry
    command_groups: dict[str, list[CommandSpec]] = {}

    for spec in registry.list_operations():
        parts = spec.operation_id.split(".")
        if len(parts) >= 2:
            group = parts[0]
            subcommand = parts[1]

            if group not in command_groups:
                command_groups[group] = []

            # Build flags from param_schema
            flags: list[FlagSpec] = []
            if spec.param_schema:
                for name, validator in spec.param_schema.validators.items():
                    flags.append(FlagSpec(
                        name=name,
                        description=f"Parameter: {name}",
                        takes_value=True,
                    ))

            command_groups[group].append(CommandSpec(
                name=subcommand,
                description=spec.description,
                flags=flags,
            ))

    # Build top-level commands
    subcommands: list[CommandSpec] = []
    for group, commands in command_groups.items():
        subcommands.append(CommandSpec(
            name=group,
            description=f"{group.title()} commands",
            subcommands=commands,
        ))

    # Add special commands
    subcommands.extend([
        CommandSpec(name="config", description="Configuration management"),
        CommandSpec(name="health", description="Health checks"),
        CommandSpec(name="plugin", description="Plugin management"),
        CommandSpec(name="shell", description="Interactive shell"),
    ])

    # Global flags
    global_flags = [
        FlagSpec(
            name="format",
            short="f",
            description="Output format",
            takes_value=True,
            value_completion=CompletionSpec(
                completion_type=CompletionType.LITERAL,
                values=[
                    CompletionValue("text", "Human-readable text"),
                    CompletionValue("json", "JSON output"),
                ],
            ),
        ),
        FlagSpec(
            name="help",
            short="h",
            description="Show help",
        ),
        FlagSpec(
            name="version",
            short="V",
            description="Show version",
        ),
        FlagSpec(
            name="debug",
            description="Enable debug mode",
        ),
    ]

    return CompletionModel(
        program_name="codeintel",
        root_command=CommandSpec(
            name="codeintel",
            description="CodeIntel CLI",
            subcommands=subcommands,
        ),
        global_flags=global_flags,
    )


__all__ = [
    "CommandSpec",
    "CompletionModel",
    "CompletionSpec",
    "CompletionType",
    "CompletionValue",
    "FlagSpec",
    "build_completion_model",
]
```

#### File: `src/codeintel/cli/completions/bash_generator.py`

```python
"""Bash completion generator.

Generates bash completion scripts from the completion model.
"""

from __future__ import annotations

from codeintel.cli.completions.completion_model import (
    CommandSpec,
    CompletionModel,
    CompletionType,
    FlagSpec,
)


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
        '    _init_completion || return',
        "",
        '    local commands="' + " ".join(
            cmd.name for cmd in model.root_command.subcommands
        ) + '"',
        "",
    ]

    # Add global flags
    global_flags = " ".join(
        f"--{f.name}" + (f" -{f.short}" if f.short else "")
        for f in model.global_flags
    )
    lines.append(f'    local global_flags="{global_flags}"')
    lines.append("")

    # Generate command-specific completions
    lines.append("    case ${words[1]} in")

    for cmd in model.root_command.subcommands:
        lines.extend(_generate_command_case(cmd, 2))

    lines.extend([
        "    *)",
        "        COMPREPLY=($(compgen -W \"$commands $global_flags\" -- $cur))",
        "        ;;",
        "    esac",
        "}",
        "",
        f"complete -F _{model.program_name}_completions {model.program_name}",
    ])

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
            lines.extend([
                f"{indent}    {sub.name})",
                f'{indent}        COMPREPLY=($(compgen -W "{sub_flags}" -- $cur))',
                f"{indent}        ;;",
            ])

        lines.extend([
            f"{indent}    *)",
            f'{indent}        COMPREPLY=($(compgen -W "{subcommand_names}" -- $cur))',
            f"{indent}        ;;",
            f"{indent}    esac",
        ])
    else:
        flags = " ".join(f"--{f.name}" for f in cmd.flags)
        lines.append(f'{indent}    COMPREPLY=($(compgen -W "{flags}" -- $cur))')

    lines.append(f"{indent}    ;;")
    return lines


__all__ = ["generate_bash_completion"]
```

#### File: `src/codeintel/cli/completions/zsh_generator.py`

```python
"""Zsh completion generator.

Generates zsh completion scripts with rich descriptions and grouping.
"""

from __future__ import annotations

from codeintel.cli.completions.completion_model import (
    CommandSpec,
    CompletionModel,
    CompletionType,
    FlagSpec,
)


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
        desc = flag.description.replace("'", "'\\''")
        if short:
            lines.append(f"        '{short}[{desc}]' \\")
        lines.append(f"        '--{flag.name}[{desc}]' \\")

    # Subcommands
    lines.extend([
        "        '1:command:->commands' \\",
        "        '*::arg:->args'",
        "",
        "    case $state in",
        "        commands)",
        "            local -a commands",
        "            commands=(",
    ])

    for cmd in model.root_command.subcommands:
        desc = cmd.description.replace("'", "'\\''")
        lines.append(f"                '{cmd.name}:{desc}'")

    lines.extend([
        "            )",
        "            _describe 'command' commands",
        "            ;;",
        "        args)",
        "            case $words[1] in",
    ])

    # Command-specific completions
    for cmd in model.root_command.subcommands:
        lines.extend(_generate_zsh_command(cmd))

    lines.extend([
        "            esac",
        "            ;;",
        "    esac",
        "}",
        "",
        "_codeintel",
    ])

    return "\n".join(lines)


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
        lines.extend([
            "                    local -a subcommands",
            "                    subcommands=(",
        ])
        for sub in cmd.subcommands:
            desc = sub.description.replace("'", "'\\''")
            lines.append(f"                        '{sub.name}:{desc}'")
        lines.extend([
            "                    )",
            "                    _describe 'subcommand' subcommands",
        ])
    elif cmd.flags:
        lines.append("                    _arguments \\")
        for flag in cmd.flags:
            desc = flag.description.replace("'", "'\\''")
            if flag.takes_value:
                lines.append(f"                        '--{flag.name}=[{desc}]' \\")
            else:
                lines.append(f"                        '--{flag.name}[{desc}]' \\")

    lines.append("                    ;;")
    return lines


__all__ = ["generate_zsh_completion"]
```

#### File: `src/codeintel/cli/completions/fish_generator.py`

```python
"""Fish completion generator.

Generates fish shell completion scripts.
"""

from __future__ import annotations

from codeintel.cli.completions.completion_model import (
    CommandSpec,
    CompletionModel,
    FlagSpec,
)


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
        parts.append(f"-d '{flag.description}'")
        lines.append(" ".join(parts))

    lines.append("")
    lines.append("# Subcommands")

    # Top-level commands
    for cmd in model.root_command.subcommands:
        lines.append(
            f"complete -c {model.program_name} -n '__fish_use_subcommand' "
            f"-a {cmd.name} -d '{cmd.description}'"
        )

    lines.append("")

    # Subcommand completions
    for cmd in model.root_command.subcommands:
        lines.extend(_generate_fish_command(model.program_name, cmd))

    return "\n".join(lines)


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
                f"-a {sub.name} -d '{sub.description}'"
            )

            # Subcommand flags
            sub_condition = f"{condition}; and __fish_seen_subcommand_from {sub.name}"
            for flag in sub.flags:
                lines.append(
                    f"complete -c {program} -n '{sub_condition}' "
                    f"-l {flag.name} -d '{flag.description}'"
                )

    else:
        for flag in cmd.flags:
            lines.append(
                f"complete -c {program} -n '{condition}' "
                f"-l {flag.name} -d '{flag.description}'"
            )

    lines.append("")
    return lines


__all__ = ["generate_fish_completion"]
```

#### File: `src/codeintel/cli/completions/powershell_generator.py`

```python
"""PowerShell completion generator.

Generates PowerShell completion scripts for Windows.
"""

from __future__ import annotations

from codeintel.cli.completions.completion_model import (
    CommandSpec,
    CompletionModel,
    FlagSpec,
)


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

    lines.extend([
        "}",
        "",
        "$CodeIntelGlobalFlags = @(",
    ])

    for flag in model.global_flags:
        lines.append(f"    '--{flag.name}'")

    lines.extend([
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
        "        $CodeIntelCommands.Keys | Where-Object { $_ -like \"$wordToComplete*\" } |",
        "            ForEach-Object { [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_) }",
        "        return",
        "    }",
        "",
        "    $command = $words[1]",
        "    if ($CodeIntelCommands.ContainsKey($command)) {",
        "        if ($words.Count -eq 2) {",
        "            # Complete subcommands",
        "            $CodeIntelCommands[$command] | Where-Object { $_ -like \"$wordToComplete*\" } |",
        "                ForEach-Object { [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_) }",
        "        }",
        "    }",
        "",
        "    # Complete global flags",
        "    if ($wordToComplete.StartsWith('-')) {",
        "        $CodeIntelGlobalFlags | Where-Object { $_ -like \"$wordToComplete*\" } |",
        "            ForEach-Object { [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_) }",
        "    }",
        "}",
    ])

    return "\n".join(lines)


__all__ = ["generate_powershell_completion"]
```

#### File: `src/codeintel/cli/completions/__init__.py`

```python
"""Shell completion generation for CLI.

Provides auto-generated completions for bash, zsh, fish, and PowerShell.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from codeintel.cli.completions.bash_generator import generate_bash_completion
from codeintel.cli.completions.completion_model import (
    CompletionModel,
    build_completion_model,
)
from codeintel.cli.completions.fish_generator import generate_fish_completion
from codeintel.cli.completions.powershell_generator import generate_powershell_completion
from codeintel.cli.completions.zsh_generator import generate_zsh_completion


class Shell(Enum):
    """Supported shells."""

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
        Shell.BASH: """
# Add to ~/.bashrc:
source <(codeintel completions bash)

# Or save to file:
codeintel completions bash > ~/.local/share/bash-completion/completions/codeintel
""",
        Shell.ZSH: """
# Add to ~/.zshrc:
source <(codeintel completions zsh)

# Or save to fpath:
codeintel completions zsh > ~/.zsh/completions/_codeintel
""",
        Shell.FISH: """
# Save to fish completions directory:
codeintel completions fish > ~/.config/fish/completions/codeintel.fish
""",
        Shell.POWERSHELL: """
# Add to $PROFILE:
codeintel completions powershell | Out-String | Invoke-Expression

# Or save to module:
codeintel completions powershell > $HOME/Documents/PowerShell/Modules/CodeIntel/CodeIntel.psm1
""",
    }

    return instructions[shell].strip()


__all__ = [
    "Shell",
    "build_completion_model",
    "generate_completion",
    "get_install_instructions",
]
```

#### File: `src/codeintel/cli/cyclopts_completions.py`

```python
"""CLI commands for completion generation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import cyclopts

from codeintel.cli.completions import Shell, generate_completion, get_install_instructions

completions_app = cyclopts.App(name="completions", help="Shell completion generation")


@completions_app.command()
def bash() -> None:
    """Generate bash completion script.

    Examples
    --------
    codeintel completions bash > ~/.local/share/bash-completion/completions/codeintel
    source <(codeintel completions bash)
    """
    print(generate_completion(Shell.BASH))


@completions_app.command()
def zsh() -> None:
    """Generate zsh completion script.

    Examples
    --------
    codeintel completions zsh > ~/.zsh/completions/_codeintel
    """
    print(generate_completion(Shell.ZSH))


@completions_app.command()
def fish() -> None:
    """Generate fish completion script.

    Examples
    --------
    codeintel completions fish > ~/.config/fish/completions/codeintel.fish
    """
    print(generate_completion(Shell.FISH))


@completions_app.command()
def powershell() -> None:
    """Generate PowerShell completion script.

    Examples
    --------
    codeintel completions powershell | Out-String | Invoke-Expression
    """
    print(generate_completion(Shell.POWERSHELL))


@completions_app.command()
def install(
    shell: Annotated[str, cyclopts.Parameter(help="Shell to install for")],
) -> None:
    """Show installation instructions for shell.

    Parameters
    ----------
    shell
        Target shell (bash, zsh, fish, powershell).

    Examples
    --------
    codeintel completions install bash
    """
    try:
        shell_enum = Shell(shell.lower())
    except ValueError:
        print(f"Unknown shell: {shell}")
        print(f"Supported: {', '.join(s.value for s in Shell)}")
        raise SystemExit(1) from None

    print(get_install_instructions(shell_enum))


__all__ = ["completions_app"]
```

---

## Phase 7.4: Plugin Ecosystem Hardening

### Value Proposition

A mature plugin ecosystem enables:

- **Community extensions** — Third-party operations without core changes
- **Enterprise customization** — Organization-specific commands
- **Safe extension** — Sandboxed plugins can't break the CLI
- **Quality control** — Versioning prevents incompatibilities

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Plugin Ecosystem Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      Plugin Manager                                │  │
│  │                                                                    │  │
│  │   discover()  →  validate()  →  load()  →  register()             │  │
│  │       ↓            ↓             ↓            ↓                   │  │
│  │   Scan dirs   Check version  Import     Add to registry           │  │
│  │              Check caps     Sandbox                               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  Plugin        │  │   Capability     │  │   Sandbox       │         │
│  │  Manifest      │  │   System         │  │   Runtime       │         │
│  │                │  │                  │  │                 │         │
│  │ • name         │  │ • register_ops   │  │ • Limited imports│        │
│  │ • version      │  │ • read_config    │  │ • No file write │         │
│  │ • api_version  │  │ • write_storage  │  │ • No network    │         │
│  │ • capabilities │  │ • exec_external  │  │ • Timeout       │         │
│  │ • dependencies │  │                  │  │                 │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Plugin Testing Utilities                        │  │
│  │                                                                    │  │
│  │   PluginTestHarness  →  Isolated environment for testing          │  │
│  │   PluginValidator    →  Check manifest and capabilities           │  │
│  │   PluginScaffold     →  Generate plugin templates                 │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Functional Objectives

1. Define plugin manifest schema with versioning
2. Implement capability-based permission system
3. Create plugin sandbox with restricted access
4. Add dependency resolution between plugins
5. Build plugin testing harness
6. Create scaffolding command for new plugins

### Implementation

#### File: `src/codeintel/cli/plugin_manifest.py`

```python
"""Plugin manifest schema and validation.

Defines the structure for plugin manifests with semantic
versioning and capability declarations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Semantic version pattern
SEMVER_PATTERN = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)

# Current CLI API version
CLI_API_VERSION = "1.0.0"


class PluginCapability(Enum):
    """Capabilities a plugin can request."""

    REGISTER_OPERATIONS = "register_operations"  # Register new operations
    READ_CONFIG = "read_config"  # Read CLI configuration
    WRITE_CONFIG = "write_config"  # Modify CLI configuration
    READ_STORAGE = "read_storage"  # Read from storage
    WRITE_STORAGE = "write_storage"  # Write to storage
    EXECUTE_EXTERNAL = "execute_external"  # Run external commands
    NETWORK_ACCESS = "network_access"  # Make network requests
    FILE_READ = "file_read"  # Read arbitrary files
    FILE_WRITE = "file_write"  # Write arbitrary files


@dataclass(frozen=True)
class SemanticVersion:
    """Semantic version representation.

    Parameters
    ----------
    major
        Major version.
    minor
        Minor version.
    patch
        Patch version.
    prerelease
        Prerelease identifier.
    build
        Build metadata.
    """

    major: int
    minor: int
    patch: int
    prerelease: str | None = None
    build: str | None = None

    @classmethod
    def parse(cls, version: str) -> SemanticVersion:
        """Parse version string.

        Parameters
        ----------
        version
            Version string.

        Returns
        -------
        SemanticVersion
            Parsed version.

        Raises
        ------
        ValueError
            If version is invalid.
        """
        match = SEMVER_PATTERN.match(version)
        if not match:
            msg = f"Invalid semantic version: {version}"
            raise ValueError(msg)

        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
            prerelease=match.group("prerelease"),
            build=match.group("buildmetadata"),
        )

    def __str__(self) -> str:
        """Convert to string."""
        result = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            result += f"-{self.prerelease}"
        if self.build:
            result += f"+{self.build}"
        return result

    def is_compatible_with(self, required: SemanticVersion) -> bool:
        """Check compatibility with required version.

        Uses semver compatibility rules:
        - Major must match
        - Minor must be >= required
        - Patch can be any value

        Parameters
        ----------
        required
            Required version.

        Returns
        -------
        bool
            True if compatible.
        """
        if self.major != required.major:
            return False
        if self.minor < required.minor:
            return False
        return True


@dataclass
class PluginDependency:
    """Plugin dependency declaration.

    Parameters
    ----------
    name
        Dependency plugin name.
    version_requirement
        Version requirement string.
    optional
        Whether dependency is optional.
    """

    name: str
    version_requirement: str
    optional: bool = False


@dataclass
class PluginManifest:
    """Plugin manifest with metadata and capabilities.

    Parameters
    ----------
    name
        Plugin name.
    version
        Plugin version.
    api_version
        Required CLI API version.
    description
        Plugin description.
    author
        Plugin author.
    capabilities
        Requested capabilities.
    dependencies
        Plugin dependencies.
    entry_point
        Module entry point.
    """

    name: str
    version: str
    api_version: str
    description: str = ""
    author: str = ""
    capabilities: list[PluginCapability] = field(default_factory=list)
    dependencies: list[PluginDependency] = field(default_factory=list)
    entry_point: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginManifest:
        """Create manifest from dictionary.

        Parameters
        ----------
        data
            Manifest data.

        Returns
        -------
        PluginManifest
            Parsed manifest.
        """
        capabilities = [
            PluginCapability(cap) for cap in data.get("capabilities", [])
        ]
        dependencies = [
            PluginDependency(
                name=dep["name"],
                version_requirement=dep.get("version", "*"),
                optional=dep.get("optional", False),
            )
            for dep in data.get("dependencies", [])
        ]

        return cls(
            name=data["name"],
            version=data["version"],
            api_version=data.get("api_version", CLI_API_VERSION),
            description=data.get("description", ""),
            author=data.get("author", ""),
            capabilities=capabilities,
            dependencies=dependencies,
            entry_point=data.get("entry_point", ""),
        )

    @classmethod
    def load(cls, path: Path) -> PluginManifest:
        """Load manifest from file.

        Parameters
        ----------
        path
            Path to manifest file.

        Returns
        -------
        PluginManifest
            Loaded manifest.
        """
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, Any]
            Manifest data.
        """
        return {
            "name": self.name,
            "version": self.version,
            "api_version": self.api_version,
            "description": self.description,
            "author": self.author,
            "capabilities": [cap.value for cap in self.capabilities],
            "dependencies": [
                {
                    "name": dep.name,
                    "version": dep.version_requirement,
                    "optional": dep.optional,
                }
                for dep in self.dependencies
            ],
            "entry_point": self.entry_point,
        }

    def validate(self) -> list[str]:
        """Validate manifest.

        Returns
        -------
        list[str]
            Validation errors.
        """
        errors: list[str] = []

        # Validate version
        try:
            SemanticVersion.parse(self.version)
        except ValueError as e:
            errors.append(f"Invalid version: {e}")

        # Validate API version
        try:
            plugin_api = SemanticVersion.parse(self.api_version)
            cli_api = SemanticVersion.parse(CLI_API_VERSION)
            if not cli_api.is_compatible_with(plugin_api):
                errors.append(
                    f"Incompatible API version: plugin requires {self.api_version}, "
                    f"CLI provides {CLI_API_VERSION}"
                )
        except ValueError as e:
            errors.append(f"Invalid API version: {e}")

        # Validate name
        if not re.match(r"^[a-z][a-z0-9_-]*$", self.name):
            errors.append(
                "Invalid name: must be lowercase alphanumeric with hyphens/underscores"
            )

        # Validate entry point
        if not self.entry_point:
            errors.append("Missing entry_point")

        return errors


__all__ = [
    "CLI_API_VERSION",
    "PluginCapability",
    "PluginDependency",
    "PluginManifest",
    "SemanticVersion",
]
```

#### File: `src/codeintel/cli/plugin_sandbox.py`

```python
"""Plugin sandbox for restricted execution.

Provides a sandboxed environment for plugin execution with
limited access to system resources.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

from codeintel.cli.plugin_manifest import PluginCapability, PluginManifest

LOG = logging.getLogger(__name__)


# Modules that plugins can always import
ALLOWED_MODULES = frozenset({
    "abc",
    "collections",
    "dataclasses",
    "datetime",
    "enum",
    "functools",
    "itertools",
    "json",
    "logging",
    "pathlib",
    "re",
    "typing",
    "codeintel.cli.results",
    "codeintel.cli.executor",
})

# Modules that require specific capabilities
CAPABILITY_MODULES: dict[PluginCapability, frozenset[str]] = {
    PluginCapability.NETWORK_ACCESS: frozenset({"urllib", "http", "socket"}),
    PluginCapability.FILE_READ: frozenset({"io", "os.path"}),
    PluginCapability.FILE_WRITE: frozenset({"io", "os", "shutil"}),
    PluginCapability.EXECUTE_EXTERNAL: frozenset({"subprocess", "os"}),
}


@dataclass
class SandboxConfig:
    """Configuration for plugin sandbox.

    Parameters
    ----------
    allowed_capabilities
        Capabilities granted to plugin.
    timeout
        Execution timeout in seconds.
    memory_limit
        Memory limit in bytes (not enforced on all platforms).
    """

    allowed_capabilities: set[PluginCapability] = field(default_factory=set)
    timeout: float = 30.0
    memory_limit: int | None = None


class SandboxedImporter:
    """Custom importer that restricts module access.

    Parameters
    ----------
    manifest
        Plugin manifest.
    config
        Sandbox configuration.
    """

    def __init__(
        self,
        manifest: PluginManifest,
        config: SandboxConfig,
    ) -> None:
        """Initialize importer."""
        self._manifest = manifest
        self._config = config
        self._allowed = self._compute_allowed_modules()

    def _compute_allowed_modules(self) -> frozenset[str]:
        """Compute set of allowed modules.

        Returns
        -------
        frozenset[str]
            Allowed module names.
        """
        allowed = set(ALLOWED_MODULES)

        for capability in self._config.allowed_capabilities:
            if capability in CAPABILITY_MODULES:
                allowed.update(CAPABILITY_MODULES[capability])

        return frozenset(allowed)

    def find_module(self, name: str, path: Any = None) -> SandboxedImporter | None:
        """Check if module import is allowed.

        Parameters
        ----------
        name
            Module name.
        path
            Import path.

        Returns
        -------
        SandboxedImporter | None
            Self if handling, None otherwise.
        """
        # Allow importing the plugin itself
        if name.startswith(self._manifest.entry_point.split(".")[0]):
            return None

        # Check if module is allowed
        root_module = name.split(".")[0]
        if root_module in self._allowed or name in self._allowed:
            return None

        # Block import
        return self

    def load_module(self, name: str) -> None:
        """Block module load.

        Parameters
        ----------
        name
            Module name.

        Raises
        ------
        ImportError
            Always raised to block import.
        """
        msg = (
            f"Plugin '{self._manifest.name}' cannot import '{name}': "
            f"missing required capability"
        )
        raise ImportError(msg)


class PluginSandbox:
    """Sandbox environment for plugin execution.

    Parameters
    ----------
    manifest
        Plugin manifest.
    config
        Sandbox configuration.
    """

    def __init__(
        self,
        manifest: PluginManifest,
        config: SandboxConfig | None = None,
    ) -> None:
        """Initialize sandbox."""
        self._manifest = manifest
        self._config = config or SandboxConfig(
            allowed_capabilities=set(manifest.capabilities)
        )
        self._importer = SandboxedImporter(manifest, self._config)
        self._active = False

    def __enter__(self) -> PluginSandbox:
        """Enter sandbox context."""
        if self._active:
            msg = "Sandbox already active"
            raise RuntimeError(msg)

        # Install custom importer
        sys.meta_path.insert(0, self._importer)  # type: ignore[arg-type]
        self._active = True
        LOG.debug("Entered sandbox for plugin: %s", self._manifest.name)
        return self

    def __exit__(self, *args: object) -> None:
        """Exit sandbox context."""
        # Remove custom importer
        try:
            sys.meta_path.remove(self._importer)  # type: ignore[arg-type]
        except ValueError:
            pass
        self._active = False
        LOG.debug("Exited sandbox for plugin: %s", self._manifest.name)

    def load_plugin(self) -> ModuleType:
        """Load plugin module within sandbox.

        Returns
        -------
        ModuleType
            Loaded plugin module.

        Raises
        ------
        ImportError
            If plugin cannot be loaded.
        """
        if not self._active:
            msg = "Sandbox not active"
            raise RuntimeError(msg)

        return importlib.import_module(self._manifest.entry_point)

    def check_capability(self, capability: PluginCapability) -> bool:
        """Check if capability is granted.

        Parameters
        ----------
        capability
            Capability to check.

        Returns
        -------
        bool
            True if granted.
        """
        return capability in self._config.allowed_capabilities


__all__ = [
    "PluginSandbox",
    "SandboxConfig",
    "SandboxedImporter",
]
```

#### File: `src/codeintel/cli/plugin_testing.py`

```python
"""Plugin testing utilities.

Provides a test harness for plugin developers to test
their plugins in isolation.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeintel.cli.plugin_manifest import PluginCapability, PluginManifest
from codeintel.cli.plugin_sandbox import PluginSandbox, SandboxConfig
from codeintel.cli.results import CliResult


@dataclass
class PluginTestResult:
    """Result of a plugin test.

    Parameters
    ----------
    success
        Whether test passed.
    message
        Result message.
    errors
        List of errors.
    warnings
        List of warnings.
    """

    success: bool
    message: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class PluginTestHarness:
    """Test harness for plugin development.

    Provides isolated environment for testing plugins
    without affecting the global CLI state.

    Parameters
    ----------
    manifest
        Plugin manifest.
    capabilities
        Override capabilities for testing.
    """

    def __init__(
        self,
        manifest: PluginManifest,
        capabilities: set[PluginCapability] | None = None,
    ) -> None:
        """Initialize test harness."""
        self._manifest = manifest
        self._capabilities = capabilities or set(manifest.capabilities)
        self._registered_operations: list[str] = []
        self._test_results: list[PluginTestResult] = []

    def validate_manifest(self) -> PluginTestResult:
        """Validate plugin manifest.

        Returns
        -------
        PluginTestResult
            Validation result.
        """
        errors = self._manifest.validate()

        if errors:
            return PluginTestResult(
                success=False,
                message="Manifest validation failed",
                errors=errors,
            )

        return PluginTestResult(
            success=True,
            message="Manifest is valid",
        )

    def test_load(self) -> PluginTestResult:
        """Test plugin loading.

        Returns
        -------
        PluginTestResult
            Load test result.
        """
        config = SandboxConfig(allowed_capabilities=self._capabilities)

        try:
            with PluginSandbox(self._manifest, config) as sandbox:
                module = sandbox.load_plugin()

                # Check for required attributes
                warnings: list[str] = []
                if not hasattr(module, "register"):
                    warnings.append("Plugin has no 'register' function")

                return PluginTestResult(
                    success=True,
                    message=f"Plugin loaded successfully: {module.__name__}",
                    warnings=warnings,
                )

        except ImportError as e:
            return PluginTestResult(
                success=False,
                message="Plugin failed to load",
                errors=[str(e)],
            )
        except Exception as e:
            return PluginTestResult(
                success=False,
                message="Plugin loading raised exception",
                errors=[f"{type(e).__name__}: {e}"],
            )

    def test_operations(self) -> PluginTestResult:
        """Test plugin operations.

        Returns
        -------
        PluginTestResult
            Operations test result.
        """
        config = SandboxConfig(allowed_capabilities=self._capabilities)
        errors: list[str] = []
        warnings: list[str] = []

        try:
            with PluginSandbox(self._manifest, config) as sandbox:
                module = sandbox.load_plugin()

                if not hasattr(module, "register"):
                    return PluginTestResult(
                        success=True,
                        message="No operations to test",
                        warnings=["Plugin has no 'register' function"],
                    )

                # Create mock registry
                registered: list[Any] = []

                class MockRegistry:
                    def register(self, spec: Any) -> Any:
                        registered.append(spec)
                        return spec

                # Register operations
                module.register(MockRegistry())

                # Validate registered operations
                for spec in registered:
                    if not hasattr(spec, "operation_id"):
                        errors.append("Operation missing operation_id")
                        continue

                    if not spec.operation_id.startswith(f"{self._manifest.name}."):
                        warnings.append(
                            f"Operation '{spec.operation_id}' should be prefixed "
                            f"with plugin name '{self._manifest.name}.'"
                        )

                    self._registered_operations.append(spec.operation_id)

                return PluginTestResult(
                    success=len(errors) == 0,
                    message=f"Registered {len(registered)} operations",
                    errors=errors,
                    warnings=warnings,
                )

        except Exception as e:
            return PluginTestResult(
                success=False,
                message="Operation testing failed",
                errors=[f"{type(e).__name__}: {e}"],
            )

    def run_all_tests(self) -> list[PluginTestResult]:
        """Run all plugin tests.

        Returns
        -------
        list[PluginTestResult]
            All test results.
        """
        results = [
            self.validate_manifest(),
            self.test_load(),
            self.test_operations(),
        ]
        self._test_results = results
        return results

    def get_summary(self) -> dict[str, Any]:
        """Get test summary.

        Returns
        -------
        dict[str, Any]
            Test summary.
        """
        passed = sum(1 for r in self._test_results if r.success)
        failed = len(self._test_results) - passed

        return {
            "plugin": self._manifest.name,
            "version": self._manifest.version,
            "tests_run": len(self._test_results),
            "passed": passed,
            "failed": failed,
            "registered_operations": self._registered_operations,
        }


def create_plugin_scaffold(
    name: str,
    output_dir: Path,
    *,
    capabilities: list[PluginCapability] | None = None,
) -> Path:
    """Create plugin scaffold.

    Parameters
    ----------
    name
        Plugin name.
    output_dir
        Output directory.
    capabilities
        Initial capabilities.

    Returns
    -------
    Path
        Path to created plugin directory.
    """
    capabilities = capabilities or [PluginCapability.REGISTER_OPERATIONS]

    plugin_dir = output_dir / name
    plugin_dir.mkdir(parents=True, exist_ok=True)

    # Create manifest
    manifest = {
        "name": name,
        "version": "0.1.0",
        "api_version": "1.0.0",
        "description": f"{name.title()} plugin for CodeIntel CLI",
        "author": "",
        "capabilities": [cap.value for cap in capabilities],
        "dependencies": [],
        "entry_point": f"{name}.main",
    }

    manifest_path = plugin_dir / "plugin.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Create package
    pkg_dir = plugin_dir / name
    pkg_dir.mkdir(exist_ok=True)

    # Create __init__.py
    init_content = f'''"""
{name.title()} plugin for CodeIntel CLI.
"""

from {name}.main import register

__all__ = ["register"]
'''
    (pkg_dir / "__init__.py").write_text(init_content, encoding="utf-8")

    # Create main.py
    main_content = f'''"""
Main module for {name} plugin.
"""

from __future__ import annotations

from codeintel.cli.executor import OperationCategory, OperationSpec
from codeintel.cli.results import CliResult


def _example_handler() -> CliResult[dict[str, str]]:
    """Example operation handler."""
    return CliResult.ok({{"message": "Hello from {name}!"}})


def register(registry: object) -> None:
    """Register plugin operations.

    Parameters
    ----------
    registry
        Operation registry.
    """
    registry.register(
        OperationSpec(
            operation_id="{name}.hello",
            handler=_example_handler,
            category=OperationCategory.READ,
            description="Example operation from {name} plugin",
        )
    )
'''
    (pkg_dir / "main.py").write_text(main_content, encoding="utf-8")

    # Create test file
    test_content = f'''"""
Tests for {name} plugin.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.cli.plugin_manifest import PluginManifest
from codeintel.cli.plugin_testing import PluginTestHarness


@pytest.fixture
def manifest() -> PluginManifest:
    """Load plugin manifest."""
    manifest_path = Path(__file__).parent.parent / "plugin.json"
    return PluginManifest.load(manifest_path)


class Test{name.title().replace("-", "").replace("_", "")}Plugin:
    """Tests for {name} plugin."""

    def test_manifest_valid(self, manifest: PluginManifest) -> None:
        """Test manifest is valid."""
        harness = PluginTestHarness(manifest)
        result = harness.validate_manifest()
        assert result.success, result.errors

    def test_plugin_loads(self, manifest: PluginManifest) -> None:
        """Test plugin loads successfully."""
        harness = PluginTestHarness(manifest)
        result = harness.test_load()
        assert result.success, result.errors

    def test_operations_register(self, manifest: PluginManifest) -> None:
        """Test operations register correctly."""
        harness = PluginTestHarness(manifest)
        result = harness.test_operations()
        assert result.success, result.errors
'''
    tests_dir = plugin_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / f"test_{name}.py").write_text(test_content, encoding="utf-8")
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")

    # Create README
    readme_content = f"""# {name.title()} Plugin

A plugin for CodeIntel CLI.

## Installation

Copy this plugin to your plugins directory:

```bash
cp -r {name} ~/.codeintel/plugins/
```

## Usage

```bash
codeintel op call {name}.hello
```

## Development

Run tests:

```bash
pytest {name}/tests/
```
"""
    (plugin_dir / "README.md").write_text(readme_content, encoding="utf-8")

    return plugin_dir


__all__ = [
    "PluginTestHarness",
    "PluginTestResult",
    "create_plugin_scaffold",
]
```

#### File: `src/codeintel/cli/cyclopts_plugin_commands.py`

```python
"""CLI commands for plugin management."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import cyclopts

from codeintel.cli.plugin_manifest import PluginCapability, PluginManifest
from codeintel.cli.plugin_testing import PluginTestHarness, create_plugin_scaffold

plugin_cmd_app = cyclopts.App(name="plugin", help="Plugin management commands")


@plugin_cmd_app.command()
def new(
    name: Annotated[str, cyclopts.Parameter(help="Plugin name")],
    output: Annotated[
        Path | None,
        cyclopts.Parameter(help="Output directory"),
    ] = None,
) -> None:
    """Create new plugin from template.

    Parameters
    ----------
    name
        Plugin name (lowercase, alphanumeric with hyphens).
    output
        Output directory (default: current directory).

    Examples
    --------
    codeintel plugin new my-plugin
    codeintel plugin new my-plugin --output ~/projects/
    """
    output_dir = output or Path.cwd()

    # Validate name
    import re
    if not re.match(r"^[a-z][a-z0-9_-]*$", name):
        print("Error: Plugin name must be lowercase alphanumeric with hyphens/underscores")
        raise SystemExit(1)

    plugin_dir = create_plugin_scaffold(name, output_dir)
    print(f"Created plugin scaffold at: {plugin_dir}")
    print()
    print("Next steps:")
    print(f"  1. cd {plugin_dir}")
    print(f"  2. Edit {name}/main.py to add your operations")
    print(f"  3. Run tests: pytest tests/")
    print(f"  4. Install: cp -r {name} ~/.codeintel/plugins/")


@plugin_cmd_app.command()
def test(
    path: Annotated[Path, cyclopts.Parameter(help="Plugin directory")],
) -> None:
    """Test a plugin.

    Parameters
    ----------
    path
        Path to plugin directory.

    Examples
    --------
    codeintel plugin test ./my-plugin
    """
    manifest_path = path / "plugin.json"
    if not manifest_path.exists():
        print(f"Error: No plugin.json found in {path}")
        raise SystemExit(1)

    manifest = PluginManifest.load(manifest_path)
    harness = PluginTestHarness(manifest)
    results = harness.run_all_tests()

    print(f"Testing plugin: {manifest.name} v{manifest.version}")
    print()

    all_passed = True
    for result in results:
        status = "✓" if result.success else "✗"
        print(f"  {status} {result.message}")

        for error in result.errors:
            print(f"      Error: {error}")
            all_passed = False

        for warning in result.warnings:
            print(f"      Warning: {warning}")

    print()
    summary = harness.get_summary()
    print(f"Tests: {summary['passed']}/{summary['tests_run']} passed")

    if summary["registered_operations"]:
        print(f"Operations: {', '.join(summary['registered_operations'])}")

    if not all_passed:
        raise SystemExit(1)


@plugin_cmd_app.command()
def validate(
    path: Annotated[Path, cyclopts.Parameter(help="Plugin directory")],
) -> None:
    """Validate plugin manifest.

    Parameters
    ----------
    path
        Path to plugin directory.

    Examples
    --------
    codeintel plugin validate ./my-plugin
    """
    manifest_path = path / "plugin.json"
    if not manifest_path.exists():
        print(f"Error: No plugin.json found in {path}")
        raise SystemExit(1)

    try:
        manifest = PluginManifest.load(manifest_path)
    except Exception as e:
        print(f"Error loading manifest: {e}")
        raise SystemExit(1) from None

    errors = manifest.validate()

    if errors:
        print("Manifest validation failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print(f"✓ Manifest is valid")
    print(f"  Name: {manifest.name}")
    print(f"  Version: {manifest.version}")
    print(f"  API Version: {manifest.api_version}")
    print(f"  Capabilities: {', '.join(cap.value for cap in manifest.capabilities)}")


__all__ = ["plugin_cmd_app"]
```

---

## Implementation Timeline

| Phase | Duration | Dependencies | Priority | Effort |
|-------|----------|--------------|----------|--------|
| 7.3 Shell Completions | 3-4 days | Phase 6 complete | Medium-High | Medium |
| 7.4 Plugin Hardening | 4-5 days | Phase 6 complete | Medium-High | High |

**Total estimated time: 7-9 days**

### Recommended Order

```
Week 1:       [===== Phase 7.3: Completions =====]
Week 1-2:                    [====== 7.4: Plugins ======]
```

### Parallel with Part 1

- 7.3 and 7.4 can run in parallel with 7.1 and 7.2
- Completions (7.3) benefit from operation registry stability
- Plugin testing (7.4) can use harness from 7.1

---

## Success Metrics

### Phase 7.3: Shell Completions

- [ ] Bash completion with full subcommand support
- [ ] Zsh completion with descriptions
- [ ] Fish completion with native syntax
- [ ] PowerShell completion for Windows
- [ ] `codeintel completions <shell>` commands working
- [ ] Install instructions for each shell
- [ ] Dynamic completion for operation IDs
- [ ] Completion tests verify script generation

### Phase 7.4: Plugin Hardening

- [ ] Plugin manifest schema with semver validation
- [ ] Capability-based permission system
- [ ] Plugin sandbox restricts imports
- [ ] Dependency resolution between plugins
- [ ] `codeintel plugin new` scaffolds plugin
- [ ] `codeintel plugin test` runs test harness
- [ ] `codeintel plugin validate` checks manifest
- [ ] Plugin testing utilities documented

---

## Appendix: File Manifest

### New Files

| File | Purpose |
|------|---------|
| `src/codeintel/cli/completions/__init__.py` | Completion module init |
| `src/codeintel/cli/completions/completion_model.py` | Shell-agnostic model |
| `src/codeintel/cli/completions/bash_generator.py` | Bash completion |
| `src/codeintel/cli/completions/zsh_generator.py` | Zsh completion |
| `src/codeintel/cli/completions/fish_generator.py` | Fish completion |
| `src/codeintel/cli/completions/powershell_generator.py` | PowerShell completion |
| `src/codeintel/cli/cyclopts_completions.py` | Completion commands |
| `src/codeintel/cli/plugin_manifest.py` | Manifest schema |
| `src/codeintel/cli/plugin_sandbox.py` | Sandbox execution |
| `src/codeintel/cli/plugin_testing.py` | Test harness |
| `src/codeintel/cli/cyclopts_plugin_commands.py` | Plugin commands |

### Modified Files

| File | Changes |
|------|---------|
| `src/codeintel/cli/cyclopts_app.py` | Add completion and plugin subcommands |
| `src/codeintel/cli/plugins.py` | Integrate manifest validation |

---

*End of Phase 7 Part 2 Implementation Plan*

