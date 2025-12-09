"""IDE integration helpers for the CodeIntel CLI."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass

import typer

from codeintel.cli.commands._common import (
    RuntimeCliOptions,
    build_graph_runtime,
    build_runtime_or_exit,
    open_gateway_from_config,
    setup_logging,
)
from codeintel.serving.bootstrap import BackendResourceOptions, build_backend_resource

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class IdeHintsOptions:
    """Options for IDE hint resolution."""

    rel_path: str
    runtime_options: RuntimeCliOptions
    verbose: int = 0


def ide_hints_handler(options: IdeHintsOptions) -> None:
    """Emit IDE hints (module + subsystem context) for a relative file path.

    Raises
    ------
    typer.Exit
        If hints cannot be resolved.
    """
    setup_logging(options.verbose)

    runtime = build_runtime_or_exit(options.runtime_options)

    gateway = open_gateway_from_config(runtime.cfg, read_only=True)
    graph_runtime = build_graph_runtime(runtime.cfg, gateway)

    resource = build_backend_resource(
        runtime.serving,
        gateway=gateway,
        options=BackendResourceOptions(graph_runtime=graph_runtime),
    )

    response = resource.backend.get_file_hints(rel_path=options.rel_path)
    if not response.found or not response.hints:
        LOG.error("No IDE hints found for %s", options.rel_path)
        typer.secho(
            f"No hints found for: {options.rel_path}",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)

    payload = {
        "rel_path": options.rel_path,
        "hints": [hint.model_dump() for hint in response.hints],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")


__all__ = ["IdeHintsOptions", "ide_hints_handler"]
