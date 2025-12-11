"""Shell-agnostic completion model.

Provide a unified model for completions that can be rendered
to any shell's completion format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from codeintel.cli.execution.registry import OperationRegistry, get_registry

if TYPE_CHECKING:
    from codeintel.cli.introspection import ValidationSchema

# Minimum number of parts for a valid operation ID (group.subcommand)
_MIN_OPERATION_ID_PARTS = 2


class CompletionType(Enum):
    """Types of completion values.

    Values
    ------
    LITERAL
        Fixed string value.
    FILE
        File path completion.
    DIRECTORY
        Directory path completion.
    COMMAND
        Subcommand completion.
    OPERATION
        Operation ID completion.
    DYNAMIC
        Dynamic completion from function.
    """

    LITERAL = "literal"
    FILE = "file"
    DIRECTORY = "directory"
    COMMAND = "command"
    OPERATION = "operation"
    DYNAMIC = "dynamic"


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


def _extract_flags_from_schema(schema: ValidationSchema | None) -> list[FlagSpec]:
    """Extract flags from a validation schema.

    Parameters
    ----------
    schema
        Validation schema.

    Returns
    -------
    list[FlagSpec]
        List of flag specifications.
    """
    if schema is None:
        return []

    return [
        FlagSpec(
            name=name,
            description=f"Parameter: {name}",
            takes_value=True,
        )
        for name in schema.validators
    ]


def build_completion_model(registry: OperationRegistry | None = None) -> CompletionModel:
    """Build completion model from operation registry.

    Parameters
    ----------
    registry
        Optional registry instance; defaults to global registry.

    Returns
    -------
    CompletionModel
        Complete completion model.
    """
    registry = registry or get_registry()

    # Build subcommands from registry
    command_groups: dict[str, list[CommandSpec]] = {}

    for spec in registry.list_operations():
        parts = spec.operation_id.split(".", maxsplit=1)
        if len(parts) >= _MIN_OPERATION_ID_PARTS:
            group = parts[0]
            subcommand = parts[1]

            if group not in command_groups:
                command_groups[group] = []

            # Note: New OperationSpec doesn't have param_schema, so flags are empty
            # Command-line flags come from Cyclopts dataclass definitions instead
            flags: list[FlagSpec] = []

            command_groups[group].append(
                CommandSpec(
                    name=subcommand,
                    description=spec.description,
                    flags=flags,
                ),
            )

    # Build top-level commands
    subcommands: list[CommandSpec] = []
    for group, commands in sorted(command_groups.items()):
        subcommands.append(
            CommandSpec(
                name=group,
                description=f"{group.title()} commands",
                subcommands=commands,
            ),
        )

    # Add special commands
    subcommands.extend(
        [
            CommandSpec(name="config", description="Configuration management"),
            CommandSpec(name="health", description="Health checks"),
            CommandSpec(name="plugins", description="Plugin management"),
            CommandSpec(name="completions", description="Shell completion generation"),
        ],
    )

    # Global flags
    global_flags = [
        FlagSpec(
            name="output-format",
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
