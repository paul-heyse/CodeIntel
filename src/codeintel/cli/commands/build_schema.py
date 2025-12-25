"""Build schema commands.

These commands treat schemas as a first-class build artifact, enabling
deterministic manifest compilation and (later) Hamilton-native inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from cyclopts import App

from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.build_schema import (
    build_schema_compile_handler,
    build_schema_diff_handler,
    build_schema_migrate_handler,
)
from codeintel.cli.options.registry import (
    BUILD_SCHEMA_ALL,
    BUILD_SCHEMA_DRY_RUN,
    BUILD_SCHEMA_EXPECTED_FILE,
    BUILD_SCHEMA_FAIL_ON_ANY,
    BUILD_SCHEMA_FAIL_ON_BREAKING,
    BUILD_SCHEMA_FORMAT,
    BUILD_SCHEMA_INCLUDE_ARTIFACTS,
    BUILD_SCHEMA_INCLUDE_PROVENANCE,
    BUILD_SCHEMA_INCLUDE_VIEWS,
    BUILD_SCHEMA_INFER_NATIVE,
    BUILD_SCHEMA_MODULE,
    BUILD_SCHEMA_OUTPUT,
    BUILD_SCHEMA_STABLE,
    BUILD_SCHEMA_TARGETS,
)
from codeintel.cli.options.shared_flags import SharedFlags, shared_flags_field
from codeintel.cli.options.types import CommandPath, option_param

build_schema_app = App(
    name="schema",
    help="Schema product commands (compile, diff, migrate, etc.).",
)

_SCHEMA_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)

BUILD_SCHEMA_COMPILE_PATH: CommandPath = ("build", "schema", "compile")
BUILD_SCHEMA_DIFF_PATH: CommandPath = ("build", "schema", "diff")
BUILD_SCHEMA_MIGRATE_PATH: CommandPath = ("build", "schema", "migrate")

_BUILD_SCHEMA_COMPILE_FLAGS_FIELD = shared_flags_field(BUILD_SCHEMA_COMPILE_PATH)
_BUILD_SCHEMA_DIFF_FLAGS_FIELD = shared_flags_field(BUILD_SCHEMA_DIFF_PATH)
_BUILD_SCHEMA_MIGRATE_FLAGS_FIELD = shared_flags_field(BUILD_SCHEMA_MIGRATE_PATH)


@cli_command("build.schema.compile", handler=build_schema_compile_handler, config=_SCHEMA_CONFIG)
@build_schema_app.command(name="compile")
@dataclass
class BuildSchemaCompileCommand:
    """Compile a deterministic schema manifest for selected targets."""

    targets: Annotated[
        list[str] | None,
        option_param(BUILD_SCHEMA_TARGETS, command_path=BUILD_SCHEMA_COMPILE_PATH),
    ] = None
    module: Annotated[
        str | None,
        option_param(BUILD_SCHEMA_MODULE, command_path=BUILD_SCHEMA_COMPILE_PATH),
    ] = None
    all_targets: Annotated[
        bool,
        option_param(BUILD_SCHEMA_ALL, command_path=BUILD_SCHEMA_COMPILE_PATH),
    ] = False
    infer_native: Annotated[
        bool,
        option_param(BUILD_SCHEMA_INFER_NATIVE, command_path=BUILD_SCHEMA_COMPILE_PATH),
    ] = True
    stable: Annotated[
        bool,
        option_param(BUILD_SCHEMA_STABLE, command_path=BUILD_SCHEMA_COMPILE_PATH),
    ] = True
    output_format: Annotated[
        str,
        option_param(BUILD_SCHEMA_FORMAT, command_path=BUILD_SCHEMA_COMPILE_PATH),
    ] = "json"
    output_file: Annotated[
        str | None,
        option_param(BUILD_SCHEMA_OUTPUT, command_path=BUILD_SCHEMA_COMPILE_PATH),
    ] = None
    include_views: Annotated[
        bool,
        option_param(BUILD_SCHEMA_INCLUDE_VIEWS, command_path=BUILD_SCHEMA_COMPILE_PATH),
    ] = False
    include_artifacts: Annotated[
        bool,
        option_param(BUILD_SCHEMA_INCLUDE_ARTIFACTS, command_path=BUILD_SCHEMA_COMPILE_PATH),
    ] = False
    include_provenance: Annotated[
        bool,
        option_param(BUILD_SCHEMA_INCLUDE_PROVENANCE, command_path=BUILD_SCHEMA_COMPILE_PATH),
    ] = False
    flags: SharedFlags = _BUILD_SCHEMA_COMPILE_FLAGS_FIELD


@cli_command("build.schema.diff", handler=build_schema_diff_handler, config=_SCHEMA_CONFIG)
@build_schema_app.command(name="diff")
@dataclass
class BuildSchemaDiffCommand:
    """Diff a compiled schema manifest against an expected file."""

    expected_file: Annotated[
        str,
        option_param(BUILD_SCHEMA_EXPECTED_FILE, command_path=BUILD_SCHEMA_DIFF_PATH),
    ]
    targets: Annotated[
        list[str] | None,
        option_param(BUILD_SCHEMA_TARGETS, command_path=BUILD_SCHEMA_DIFF_PATH),
    ] = None
    module: Annotated[
        str | None,
        option_param(BUILD_SCHEMA_MODULE, command_path=BUILD_SCHEMA_DIFF_PATH),
    ] = None
    all_targets: Annotated[
        bool,
        option_param(BUILD_SCHEMA_ALL, command_path=BUILD_SCHEMA_DIFF_PATH),
    ] = False
    infer_native: Annotated[
        bool,
        option_param(BUILD_SCHEMA_INFER_NATIVE, command_path=BUILD_SCHEMA_DIFF_PATH),
    ] = True
    stable: Annotated[
        bool,
        option_param(BUILD_SCHEMA_STABLE, command_path=BUILD_SCHEMA_DIFF_PATH),
    ] = True
    fail_on_breaking: Annotated[
        bool,
        option_param(BUILD_SCHEMA_FAIL_ON_BREAKING, command_path=BUILD_SCHEMA_DIFF_PATH),
    ] = True
    fail_on_any: Annotated[
        bool,
        option_param(BUILD_SCHEMA_FAIL_ON_ANY, command_path=BUILD_SCHEMA_DIFF_PATH),
    ] = False
    include_views: Annotated[
        bool,
        option_param(BUILD_SCHEMA_INCLUDE_VIEWS, command_path=BUILD_SCHEMA_DIFF_PATH),
    ] = False
    include_artifacts: Annotated[
        bool,
        option_param(BUILD_SCHEMA_INCLUDE_ARTIFACTS, command_path=BUILD_SCHEMA_DIFF_PATH),
    ] = False
    include_provenance: Annotated[
        bool,
        option_param(BUILD_SCHEMA_INCLUDE_PROVENANCE, command_path=BUILD_SCHEMA_DIFF_PATH),
    ] = False
    flags: SharedFlags = _BUILD_SCHEMA_DIFF_FLAGS_FIELD


@cli_command("build.schema.migrate", handler=build_schema_migrate_handler, config=_SCHEMA_CONFIG)
@build_schema_app.command(name="migrate")
@dataclass
class BuildSchemaMigrateCommand:
    """Update expected manifest to match current schemas."""

    expected_file: Annotated[
        str,
        option_param(BUILD_SCHEMA_EXPECTED_FILE, command_path=BUILD_SCHEMA_MIGRATE_PATH),
    ]
    targets: Annotated[
        list[str] | None,
        option_param(BUILD_SCHEMA_TARGETS, command_path=BUILD_SCHEMA_MIGRATE_PATH),
    ] = None
    module: Annotated[
        str | None,
        option_param(BUILD_SCHEMA_MODULE, command_path=BUILD_SCHEMA_MIGRATE_PATH),
    ] = None
    all_targets: Annotated[
        bool,
        option_param(BUILD_SCHEMA_ALL, command_path=BUILD_SCHEMA_MIGRATE_PATH),
    ] = False
    infer_native: Annotated[
        bool,
        option_param(BUILD_SCHEMA_INFER_NATIVE, command_path=BUILD_SCHEMA_MIGRATE_PATH),
    ] = True
    stable: Annotated[
        bool,
        option_param(BUILD_SCHEMA_STABLE, command_path=BUILD_SCHEMA_MIGRATE_PATH),
    ] = True
    dry_run: Annotated[
        bool,
        option_param(BUILD_SCHEMA_DRY_RUN, command_path=BUILD_SCHEMA_MIGRATE_PATH),
    ] = True
    include_views: Annotated[
        bool,
        option_param(BUILD_SCHEMA_INCLUDE_VIEWS, command_path=BUILD_SCHEMA_MIGRATE_PATH),
    ] = False
    include_artifacts: Annotated[
        bool,
        option_param(BUILD_SCHEMA_INCLUDE_ARTIFACTS, command_path=BUILD_SCHEMA_MIGRATE_PATH),
    ] = False
    include_provenance: Annotated[
        bool,
        option_param(BUILD_SCHEMA_INCLUDE_PROVENANCE, command_path=BUILD_SCHEMA_MIGRATE_PATH),
    ] = False
    flags: SharedFlags = _BUILD_SCHEMA_MIGRATE_FLAGS_FIELD


__all__ = [
    "BuildSchemaCompileCommand",
    "BuildSchemaDiffCommand",
    "BuildSchemaMigrateCommand",
    "build_schema_app",
]
