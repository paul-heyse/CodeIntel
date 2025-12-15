"""Build schema commands.

These commands treat schemas as a first-class build artifact, enabling
deterministic manifest compilation and (later) Hamilton-native inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import SHARED_FLAGS_METADATA, SharedFlags
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.build_schema import (
    build_schema_compile_handler,
    build_schema_diff_handler,
    build_schema_migrate_handler,
)

build_schema_app = App(
    name="schema",
    help="Schema product commands (compile, diff, migrate, etc.).",
)

_SCHEMA_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)


@cli_command("build.schema.compile", handler=build_schema_compile_handler, config=_SCHEMA_CONFIG)
@build_schema_app.command(name="compile")
@dataclass
class BuildSchemaCompileCommand:
    """Compile a deterministic schema manifest for selected targets."""

    targets: Annotated[
        list[str] | None,
        Parameter(
            name=None,
            help="Target names to include (defaults to all targets).",
        ),
    ] = None
    module: Annotated[
        str | None,
        Parameter(
            name=["--module", "-m"],
            help="Compile schemas for all targets in a module.",
            show_choices=True,
        ),
    ] = None
    all_targets: Annotated[
        bool,
        Parameter(
            name=["--all", "-a"],
            help="Compile schemas for all targets across all modules.",
            negative=(),
        ),
    ] = False
    only_native: Annotated[
        bool,
        Parameter(
            name=["--only-native"],
            help="Restrict compilation to targets with native Hamilton implementations.",
            negative=(),
        ),
    ] = False
    infer_native: Annotated[
        bool,
        Parameter(
            name=["--infer-native", "--infer"],
            help="Infer schemas for inferable native targets (fallback to declared on errors).",
            negative=(),
        ),
    ] = False
    stable: Annotated[
        bool,
        Parameter(
            name=["--stable"],
            help="Force deterministic ordering and canonicalized output.",
            negative=(),
        ),
    ] = True
    output_format: Annotated[
        str,
        Parameter(
            name=["--format", "-f"],
            help="Output format: json (default).",
        ),
    ] = "json"
    output_file: Annotated[
        str | None,
        Parameter(
            name=["--output", "-o"],
            help="Output file path (stdout if not specified).",
        ),
    ] = None
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("build.schema.diff", handler=build_schema_diff_handler, config=_SCHEMA_CONFIG)
@build_schema_app.command(name="diff")
@dataclass
class BuildSchemaDiffCommand:
    """Diff a compiled schema manifest against an expected file."""

    expected_file: Annotated[
        str,
        Parameter(
            name=["--expected", "-e"],
            help="Path to an expected schema manifest JSON file.",
        ),
    ]
    targets: Annotated[
        list[str] | None,
        Parameter(
            name=None,
            help="Target names to include (defaults to all targets).",
        ),
    ] = None
    module: Annotated[
        str | None,
        Parameter(
            name=["--module", "-m"],
            help="Compile schemas for all targets in a module.",
            show_choices=True,
        ),
    ] = None
    all_targets: Annotated[
        bool,
        Parameter(
            name=["--all", "-a"],
            help="Compile schemas for all targets across all modules.",
            negative=(),
        ),
    ] = False
    only_native: Annotated[
        bool,
        Parameter(
            name=["--only-native"],
            help="Restrict compilation to targets with native Hamilton implementations.",
            negative=(),
        ),
    ] = False
    infer_native: Annotated[
        bool,
        Parameter(
            name=["--infer-native", "--infer"],
            help="Infer schemas for inferable native targets (fallback to declared on errors).",
            negative=(),
        ),
    ] = False
    stable: Annotated[
        bool,
        Parameter(
            name=["--stable"],
            help="Force deterministic ordering and canonicalized output.",
            negative=(),
        ),
    ] = True
    detailed: Annotated[
        bool,
        Parameter(
            name=["--detailed", "-d"],
            help="Show structured diff with breaking change detection.",
            negative=(),
        ),
    ] = False
    fail_on_breaking: Annotated[
        bool,
        Parameter(
            name=["--fail-on-breaking"],
            help="Exit with error if breaking changes detected (default: true).",
            negative=["--no-fail-on-breaking"],
        ),
    ] = True
    fail_on_any: Annotated[
        bool,
        Parameter(
            name=["--fail-on-any"],
            help="Exit with error on any schema drift, not just breaking changes.",
            negative=(),
        ),
    ] = False
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("build.schema.migrate", handler=build_schema_migrate_handler, config=_SCHEMA_CONFIG)
@build_schema_app.command(name="migrate")
@dataclass
class BuildSchemaMigrateCommand:
    """Update expected manifest to match current schemas."""

    expected_file: Annotated[
        str,
        Parameter(
            name=["--expected", "-e"],
            help="Path to the expected schema manifest JSON file to update.",
        ),
    ]
    targets: Annotated[
        list[str] | None,
        Parameter(
            name=None,
            help="Target names to include (defaults to all targets).",
        ),
    ] = None
    module: Annotated[
        str | None,
        Parameter(
            name=["--module", "-m"],
            help="Compile schemas for all targets in a module.",
            show_choices=True,
        ),
    ] = None
    all_targets: Annotated[
        bool,
        Parameter(
            name=["--all", "-a"],
            help="Compile schemas for all targets across all modules.",
            negative=(),
        ),
    ] = False
    only_native: Annotated[
        bool,
        Parameter(
            name=["--only-native"],
            help="Restrict compilation to targets with native Hamilton implementations.",
            negative=(),
        ),
    ] = False
    infer_native: Annotated[
        bool,
        Parameter(
            name=["--infer-native", "--infer"],
            help="Infer schemas for inferable native targets (fallback to declared on errors).",
            negative=(),
        ),
    ] = False
    stable: Annotated[
        bool,
        Parameter(
            name=["--stable"],
            help="Force deterministic ordering and canonicalized output.",
            negative=(),
        ),
    ] = True
    dry_run: Annotated[
        bool,
        Parameter(
            name=["--dry-run"],
            help="Show migration plan without writing changes (default: true).",
            negative=["--no-dry-run"],
        ),
    ] = True
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


__all__ = [
    "BuildSchemaCompileCommand",
    "BuildSchemaDiffCommand",
    "BuildSchemaMigrateCommand",
    "build_schema_app",
]
