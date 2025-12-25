"""Docs export commands.

Note: Docs commands require runtime/gateway access via handler pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from cyclopts import App

from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.docs import docs_export_handler
from codeintel.cli.options.registry import (
    DOCS_BUILD_DIR,
    DOCS_COMMIT,
    DOCS_DATASET,
    DOCS_DB_PATH,
    DOCS_DOCUMENT_OUTPUT_DIR,
    DOCS_DRY_RUN,
    DOCS_NX_BACKEND,
    DOCS_NX_GPU_MODE,
    DOCS_PREREQ_MODE,
    DOCS_REPO,
    DOCS_REPO_ROOT,
    DOCS_RUN_MODE,
    DOCS_SCHEMA,
    DOCS_SKIP_PREREQS,
    DOCS_VALIDATE,
    DOCS_VALIDATION_MODE,
)
from codeintel.cli.options.shared_flags import SharedFlagsProtocol, shared_flags_field
from codeintel.cli.options.types import CommandPath, option_param

docs_app = App(
    name="docs",
    help="Document export utilities.",
)

_CYCLOPTS_PATH_TYPE = Path


class NxBackend(StrEnum):
    """NetworkX backend selection."""

    AUTO = "auto"
    CPU = "cpu"
    NX_CUGRAPH = "nx-cugraph"


_DOCS_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)

DOCS_EXPORT_PATH: CommandPath = ("docs", "export")

_DOCS_EXPORT_FLAGS_FIELD = shared_flags_field(DOCS_EXPORT_PATH)


@cli_command("docs.export", handler=docs_export_handler, config=_DOCS_CONFIG)
@docs_app.command(name="export")
@dataclass
class DocsExportCommand:
    """Export datasets to Document Output/."""

    repo: Annotated[
        str | None,
        option_param(DOCS_REPO, command_path=DOCS_EXPORT_PATH),
    ] = None
    commit: Annotated[
        str | None,
        option_param(DOCS_COMMIT, command_path=DOCS_EXPORT_PATH),
    ] = None
    db_path: Annotated[
        Path | None,
        option_param(DOCS_DB_PATH, command_path=DOCS_EXPORT_PATH),
    ] = None
    build_dir: Annotated[
        Path | None,
        option_param(DOCS_BUILD_DIR, command_path=DOCS_EXPORT_PATH),
    ] = None
    repo_root: Annotated[
        Path | None,
        option_param(DOCS_REPO_ROOT, command_path=DOCS_EXPORT_PATH),
    ] = None
    document_output_dir: Annotated[
        Path | None,
        option_param(DOCS_DOCUMENT_OUTPUT_DIR, command_path=DOCS_EXPORT_PATH),
    ] = None

    nx_backend: Annotated[
        NxBackend,
        option_param(DOCS_NX_BACKEND, command_path=DOCS_EXPORT_PATH),
    ] = NxBackend.AUTO
    nx_gpu_mode: Annotated[
        str,
        option_param(DOCS_NX_GPU_MODE, command_path=DOCS_EXPORT_PATH),
    ] = "disabled"

    validation_mode: Annotated[
        str,
        option_param(DOCS_VALIDATION_MODE, command_path=DOCS_EXPORT_PATH),
    ] = "skip"
    validate: Annotated[
        bool,
        option_param(DOCS_VALIDATE, command_path=DOCS_EXPORT_PATH),
    ] = False
    skip_prereqs: Annotated[
        bool,
        option_param(DOCS_SKIP_PREREQS, command_path=DOCS_EXPORT_PATH),
    ] = False
    schemas: Annotated[
        list[str] | None,
        option_param(DOCS_SCHEMA, command_path=DOCS_EXPORT_PATH),
    ] = None
    datasets: Annotated[
        list[str] | None,
        option_param(DOCS_DATASET, command_path=DOCS_EXPORT_PATH),
    ] = None
    run_mode: Annotated[
        str,
        option_param(DOCS_RUN_MODE, command_path=DOCS_EXPORT_PATH),
    ] = "execute"
    dry_run: Annotated[
        bool,
        option_param(DOCS_DRY_RUN, command_path=DOCS_EXPORT_PATH),
    ] = False
    prereq_mode: Annotated[
        str,
        option_param(DOCS_PREREQ_MODE, command_path=DOCS_EXPORT_PATH),
    ] = "run"

    flags: SharedFlagsProtocol = _DOCS_EXPORT_FLAGS_FIELD


__all__ = ["docs_app"]
