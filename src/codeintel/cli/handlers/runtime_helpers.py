"""Shared helpers for composing runtime bundles in CLI handlers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from codeintel.build.config import load_build_config
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_options import BuildExecutionOptions
from codeintel.build.providers import create_default_providers
from codeintel.build.run_context import BuildRunContext, BuildRunContextOverrides
from codeintel.cli.execution.bootstrap import VERBOSITY_DEBUG
from codeintel.core.execution import ExecutionContext, RunKind, new_run_context
from codeintel.core.runtime.loader import load_execution_context
from codeintel.runtime.compose import compose_runtime
from codeintel.runtime.runtime_bundle import RuntimeBundle

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.cli.resolution.types import ResolvedRuntime
    from codeintel.storage.gateway import StorageGateway


def build_execution_context(
    runtime: ResolvedRuntime,
    *,
    kind: RunKind = "full",
    requested_datasets: tuple[str, ...] = (),
    verbosity: int = 0,
) -> ExecutionContext:
    """Build an execution context for CLI-driven runtime composition.

    Returns
    -------
    ExecutionContext
        Execution context configured for the CLI runtime.
    """
    run_context = new_run_context(
        snapshot=runtime.snapshot,
        kind=kind,
        trigger="cli",
        requested_datasets=requested_datasets,
    )
    context = load_execution_context(primitives=runtime.primitives, run=run_context)
    return _apply_cli_inspection(context, verbosity=verbosity)


def planning_config(env: BuildEnv) -> dict[str, Any]:
    """Return Hamilton config overrides for planning mode.

    Returns
    -------
    dict[str, Any]
        Hamilton config overrides for planning.
    """
    config: dict[str, Any] = {}
    if env.profile:
        config["profile"] = env.profile
    config.update(env.variants.as_hamilton_config())
    config["variant_fingerprint"] = env.variants.variant_fingerprint
    return config


def _apply_cli_inspection(context: ExecutionContext, *, verbosity: int) -> ExecutionContext:
    if verbosity < VERBOSITY_DEBUG:
        return context
    build_settings = context.build_settings
    if build_settings.polars_inspect:
        return context
    updated_build = replace(build_settings, polars_inspect=True)
    updated_settings = replace(context.settings, build=updated_build)
    return replace(context, settings=updated_settings)


@dataclass(frozen=True, slots=True)
class CliRuntimeComposeOptions:
    """Options for composing CLI runtime bundles."""

    config_overrides: Mapping[str, object] | None = None
    include_planning: bool = False
    requested_datasets: tuple[str, ...] = ()
    verbosity: int = 0


def compose_cli_runtime_bundle_with_env(
    *,
    runtime: ResolvedRuntime,
    gateway: StorageGateway,
    options: CliRuntimeComposeOptions | None = None,
) -> tuple[RuntimeBundle, BuildEnv]:
    """Compose a RuntimeBundle and BuildEnv for CLI handlers.

    Returns
    -------
    tuple[RuntimeBundle, BuildEnv]
        Runtime bundle and build environment for CLI usage.
    """
    resolved_options = options or CliRuntimeComposeOptions()
    providers = create_default_providers(runtime.tools)
    config = load_build_config(runtime.snapshot.repo_root)
    execution_context = build_execution_context(
        runtime,
        requested_datasets=resolved_options.requested_datasets,
        verbosity=resolved_options.verbosity,
    )
    overrides = BuildRunContextOverrides(
        execution_options=BuildExecutionOptions(profile=runtime.project.default_profile),
    )
    context = BuildRunContext.from_execution_context(
        execution_context=execution_context,
        gateway=gateway,
        providers=providers,
        config=config,
        overrides=overrides,
    )
    env = context.build_env()
    compose_config = planning_config(env)
    if resolved_options.include_planning:
        compose_config["ci.enable_planning_nodes"] = True
    compose_config.update(dict(resolved_options.config_overrides or {}))
    bundle = compose_runtime(env=env, config=compose_config).bundle
    return bundle, env


def compose_cli_runtime_bundle(
    *,
    runtime: ResolvedRuntime,
    gateway: StorageGateway,
    options: CliRuntimeComposeOptions | None = None,
) -> RuntimeBundle:
    """Compose a RuntimeBundle for CLI handlers.

    Returns
    -------
    RuntimeBundle
        Composed runtime bundle for CLI usage.
    """
    bundle, _ = compose_cli_runtime_bundle_with_env(
        runtime=runtime,
        gateway=gateway,
        options=options,
    )
    return bundle


__all__ = [
    "CliRuntimeComposeOptions",
    "build_execution_context",
    "compose_cli_runtime_bundle",
    "compose_cli_runtime_bundle_with_env",
    "planning_config",
]
