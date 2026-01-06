"""CPG option loading and overlay configuration."""

from __future__ import annotations

from hamilton.function_modifiers import cache

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.graphs.cpg.constants import CPG_TARGET_NAME
from codeintel.build.hamilton.native.graphs.cpg2.types import CpgEdgeConfig, CpgOverlayOptions
from codeintel.build.hamilton.native.options.graphs import CpgOptions
from codeintel.build.hamilton.native.options.ingestion import (
    BytecodeExtractOptions,
    InspectExtractOptions,
    SymtableExtractOptions,
)
from codeintel.build.hamilton.options_loading import load_target_options


@cache(behavior="ignore")
def cpg__options(env: BuildEnv) -> CpgOptions:
    """Load CPG options from the build environment.

    Returns
    -------
    CpgOptions
        Options controlling CPG assembly behavior.
    """
    return load_target_options(
        env,
        target_name=CPG_TARGET_NAME,
        options_type=CpgOptions,
    )


@cache(behavior="ignore")
def cpg__overlay_options(env: BuildEnv) -> CpgOverlayOptions:
    """Load overlay enablement flags from ingestion options.

    Returns
    -------
    CpgOverlayOptions
        Overlay gating options for CPG edge assembly.
    """
    symtable_options = load_target_options(
        env,
        target_name="symtable",
        options_type=SymtableExtractOptions,
    )
    bytecode_options = load_target_options(
        env,
        target_name="bytecode",
        options_type=BytecodeExtractOptions,
    )
    inspect_options = load_target_options(
        env,
        target_name="inspect",
        options_type=InspectExtractOptions,
    )
    allowlist = tuple(inspect_options.module_allowlist)
    enable_inspect = inspect_options.enable and bool(allowlist)
    return CpgOverlayOptions(
        enable_symtable=symtable_options.enable,
        enable_bytecode=bytecode_options.enable,
        enable_inspect=enable_inspect,
        inspect_allowlist=allowlist,
    )


@cache(behavior="ignore")
def cpg__edge_config(
    cpg__overlay_options: CpgOverlayOptions,
    cpg__options: CpgOptions,
) -> CpgEdgeConfig:
    """Bundle overlay and CPG options for edge assembly.

    Returns
    -------
    CpgEdgeConfig
        Bundle of overlay and CPG options for edge assembly.
    """
    return CpgEdgeConfig(
        overlay_options=cpg__overlay_options,
        options=cpg__options,
    )


__all__ = [
    "cpg__edge_config",
    "cpg__options",
    "cpg__overlay_options",
]
