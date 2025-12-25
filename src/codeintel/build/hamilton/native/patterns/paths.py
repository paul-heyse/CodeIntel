"""Helpers for resolving artifact output paths from DAG inventory."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers.path_templates import (
    default_formatter,
    format_path_template,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def resolve_artifact_output_path(
    env: BuildEnv,
    *,
    target: str,
    artifact: str,
    fallback_template: str | None = None,
) -> Path:
    """Resolve an artifact output path using inventory templates or fallback.

    Parameters
    ----------
    env
        Build environment containing output inventory and build paths.
    target
        Target name owning the artifact.
    artifact
        Artifact name to resolve.
    fallback_template
        Optional template string used when inventory templates are unavailable.

    Returns
    -------
    Path
        Resolved artifact path.

    Raises
    ------
    ValueError
        If no template is available for the requested artifact.
    """
    template: str | None = None
    inventory = env.output_inventory
    if inventory is not None:
        template = inventory.artifact_templates_for(target).get(artifact)
    if template is None:
        template = fallback_template
    if template is None:
        msg = f"Missing artifact template for {target}.{artifact}"
        raise ValueError(msg)
    formatter = default_formatter(
        build_dir=str(env.paths.build_dir),
        scip_dir=str(env.paths.scip_dir),
        export_dir=str(env.paths.document_output_dir),
        repo_root=str(env.snapshot.repo_root),
    )
    return Path(format_path_template(template, formatter=formatter))


def resolve_artifact_output_paths(
    env: BuildEnv,
    *,
    target: str,
    artifacts: Sequence[str],
    fallback_templates: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Resolve multiple artifact output paths for a target.

    Parameters
    ----------
    env
        Build environment containing output inventory and build paths.
    target
        Target name owning the artifacts.
    artifacts
        Artifact names to resolve.
    fallback_templates
        Optional mapping of artifact name to fallback template.

    Returns
    -------
    dict[str, Path]
        Mapping of artifact name to resolved path.
    """
    resolved: dict[str, Path] = {}
    for artifact in artifacts:
        fallback = None
        if fallback_templates is not None:
            fallback = fallback_templates.get(artifact)
        resolved[artifact] = resolve_artifact_output_path(
            env,
            target=target,
            artifact=artifact,
            fallback_template=fallback,
        )
    return resolved


__all__ = [
    "resolve_artifact_output_path",
    "resolve_artifact_output_paths",
]
