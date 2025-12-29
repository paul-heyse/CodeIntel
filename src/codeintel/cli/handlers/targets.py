"""Target inspection handlers for the CLI."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import TargetOriginInfo, TargetOriginListResult
from codeintel.cli.handlers.runtime_helpers import compose_cli_runtime_bundle

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext
    from codeintel.runtime.runtime_bundle import RuntimeBundle


def targets_list_handler(ctx: CommandContext) -> CliResult[TargetOriginListResult]:
    """List runtime targets with optional provenance details.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    CliResult[TargetOriginListResult]
        CLI result containing target origin metadata.
    """
    show_origin = ctx.params.get_bool("show_origin")
    runtime_bundle = compose_cli_runtime_bundle(runtime=ctx.runtime, gateway=ctx.gateway)
    targets = _collect_target_origins(runtime_bundle, show_origin=show_origin)
    return CliResult.ok(TargetOriginListResult(targets=targets, count=len(targets)))


def _collect_target_origins(
    runtime_bundle: RuntimeBundle,
    *,
    show_origin: bool,
) -> list[TargetOriginInfo]:
    catalog = runtime_bundle.catalog
    provenance = runtime_bundle.module_provenance
    nodes = runtime_bundle.driver.graph.nodes
    entries: list[TargetOriginInfo] = []

    for target in catalog.all_targets:
        anchor_node = target.anchor_node
        module_name = _node_module_name(nodes.get(anchor_node))
        origin = None
        plugin_name = None
        if show_origin and module_name:
            record = provenance.get(module_name)
            if record is not None:
                origin = record.origin
                plugin_name = record.plugin_name
        entries.append(
            TargetOriginInfo(
                target=target.name,
                domain=target.module,
                anchor_module=module_name if show_origin else None,
                origin=origin,
                plugin_name=plugin_name,
            )
        )
    entries.sort(key=lambda entry: (entry.domain, entry.target))
    return entries


def _node_module_name(node: object | None) -> str | None:
    if node is None:
        return None
    callabl = getattr(node, "callable", None)
    module = getattr(callabl, "__module__", None) if callabl is not None else None
    if isinstance(module, str):
        return module
    origin_fns = getattr(node, "originating_functions", None)
    if isinstance(origin_fns, Iterable) and origin_fns:
        fn = next(iter(origin_fns), None)
        module = getattr(fn, "__module__", None)
        if isinstance(module, str):
            return module
    return None


__all__ = ["targets_list_handler"]
