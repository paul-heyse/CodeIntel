"""CLI option registry exports."""

from codeintel.cli.options.registry import (
    JSON_FLAG,
    OUTPUT_FORMAT,
    PROJECT_ROOT,
    SHARED_FLAGS,
    VERBOSE,
)
from codeintel.cli.options.types import CommandPath, OptionGroup, OptionSpec, option_param

__all__ = [
    "CommandPath",
    "JSON_FLAG",
    "OUTPUT_FORMAT",
    "PROJECT_ROOT",
    "SHARED_FLAGS",
    "VERBOSE",
    "OptionGroup",
    "OptionSpec",
    "option_param",
]
