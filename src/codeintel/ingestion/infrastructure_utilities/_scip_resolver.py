"""Helper utilities to resolve SCIP ingestion inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from codeintel.config import ScipIngestStepConfig
from codeintel.ingestion.ports.discovery import ModuleRecord
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.module_index import load_module_map


@dataclass(frozen=True)
class ResolvedScipConfig:
    """Resolved, non-optional configuration for SCIP ingestion."""

    repo: str
    commit: str
    repo_root: Path
    build_dir: Path
    document_output_dir: Path
    scip_python_bin: str | None
    scip_bin: str | None
    modules: list[ModuleRecord]
    cfg_source: Literal["legacy", "explicit"]
    cfg: ScipIngestStepConfig | None


@dataclass(frozen=True)
class ScipResolverInput:
    """Input parameters for SCIP configuration resolution.

    Bundles optional parameters to reduce function argument count.
    Either provide a ScipIngestStepConfig via cfg, or provide explicit
    parameters (repo, commit, repo_root, build_dir, document_output_dir).

    Parameters
    ----------
    cfg
        Full step configuration (takes precedence if provided).
    repo
        Repository identifier.
    commit
        Commit hash.
    repo_root
        Path to repository root.
    build_dir
        Build output directory.
    document_output_dir
        Document output directory.
    scip_python_bin
        Path to scip-python binary.
    scip_bin
        Path to scip binary.
    modules
        Pre-computed module records.
    """

    cfg: ScipIngestStepConfig | None = None
    repo: str | None = None
    commit: str | None = None
    repo_root: Path | None = None
    build_dir: Path | None = None
    document_output_dir: Path | None = None
    scip_python_bin: str | None = None
    scip_bin: str | None = None
    modules: Sequence[ModuleRecord] | None = None


def resolve_scip_inputs(  # noqa: PLR0913 - convenience API; prefer ScipResolverInput param
    gateway: StorageGateway,
    modules_or_cfg: Sequence[ModuleRecord] | object,
    inputs: ScipResolverInput | None = None,
    *,
    cfg: ScipIngestStepConfig | None = None,
    repo: str | None = None,
    commit: str | None = None,
    repo_root: Path | None = None,
    build_dir: Path | None = None,
    document_output_dir: Path | None = None,
    scip_python_bin: str | None = None,
    scip_bin: str | None = None,
    modules: Sequence[ModuleRecord] | None = None,
) -> ResolvedScipConfig:
    """Normalize all SCIP inputs into a required, typed config.

    For new code, prefer using the inputs parameter with a ScipResolverInput
    to bundle parameters.

    Returns
    -------
    ResolvedScipConfig
        Normalized configuration with all required fields populated.

    Raises
    ------
    ValueError
        If required parameters are missing.
    """
    # Use inputs dataclass if provided
    if inputs is not None:
        cfg = inputs.cfg
        repo = inputs.repo
        commit = inputs.commit
        repo_root = inputs.repo_root
        build_dir = inputs.build_dir
        document_output_dir = inputs.document_output_dir
        scip_python_bin = inputs.scip_python_bin
        scip_bin = inputs.scip_bin
        modules = inputs.modules

    # Legacy config object path
    if cfg is not None or isinstance(modules_or_cfg, ScipIngestStepConfig):
        actual_cfg = cfg or modules_or_cfg
        if not isinstance(actual_cfg, ScipIngestStepConfig):
            message = "Invalid ScipIngestStepConfig"
            raise ValueError(message)

        from codeintel.ingestion.adapters.filesystem_discovery import (  # noqa: PLC0415
            FilesystemDiscoveryAdapter,
        )

        module_map = load_module_map(
            gateway,
            actual_cfg.repo,
            actual_cfg.commit,
            language="python",
            logger=None,
        )
        module_list = list(
            FilesystemDiscoveryAdapter.iter_modules(
                module_map,
                actual_cfg.repo_root,
                logger=None,
                scan_profile=None,
            )
        )
        return ResolvedScipConfig(
            repo=actual_cfg.repo,
            commit=actual_cfg.commit,
            repo_root=actual_cfg.repo_root,
            build_dir=actual_cfg.build_dir,
            document_output_dir=actual_cfg.document_output_dir,
            scip_python_bin=actual_cfg.scip_python_bin,
            scip_bin=actual_cfg.scip_bin,
            modules=module_list,
            cfg_source="legacy",
            cfg=actual_cfg,
        )

    module_list = list(modules) if modules is not None else []
    if not module_list and isinstance(modules_or_cfg, Sequence):
        module_list = list(modules_or_cfg)

    if (
        repo is None
        or commit is None
        or repo_root is None
        or build_dir is None
        or document_output_dir is None
    ):
        message = "repo, commit, repo_root, build_dir, and document_output_dir are required"
        raise ValueError(message)

    return ResolvedScipConfig(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        build_dir=build_dir,
        document_output_dir=document_output_dir,
        scip_python_bin=scip_python_bin,
        scip_bin=scip_bin,
        modules=module_list,
        cfg_source="explicit",
        cfg=None,
    )


__all__ = ["ResolvedScipConfig", "ScipResolverInput", "resolve_scip_inputs"]
