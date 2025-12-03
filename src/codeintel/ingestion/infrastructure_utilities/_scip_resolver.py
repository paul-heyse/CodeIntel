"""Helper utilities to resolve SCIP ingestion inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
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
class ScipPathConfig:
    """Path configuration for SCIP ingestion.

    Groups path-related parameters to reduce argument count in factory methods.

    Parameters
    ----------
    repo_root
        Path to repository root.
    build_dir
        Build output directory.
    document_output_dir
        Document output directory.
    """

    repo_root: Path | None = None
    build_dir: Path | None = None
    document_output_dir: Path | None = None

    @classmethod
    def from_strings(
        cls,
        *,
        repo_root: Path | str | None = None,
        build_dir: Path | str | None = None,
        document_output_dir: Path | str | None = None,
    ) -> ScipPathConfig:
        """Create from string paths with automatic coercion.

        Parameters
        ----------
        repo_root
            Path to repository root (coerced to Path).
        build_dir
            Build output directory (coerced to Path).
        document_output_dir
            Document output directory (coerced to Path).

        Returns
        -------
        ScipPathConfig
            Path configuration with coerced paths.
        """
        return cls(
            repo_root=Path(repo_root) if repo_root else None,
            build_dir=Path(build_dir) if build_dir else None,
            document_output_dir=Path(document_output_dir) if document_output_dir else None,
        )


@dataclass(frozen=True)
class ScipResolverInput:
    """Input parameters for SCIP configuration resolution.

    Bundles optional parameters to reduce function argument count.
    Either provide a ScipIngestStepConfig via cfg, or provide explicit
    parameters (repo, commit, repo_root, build_dir, document_output_dir).

    Use the `build()` factory method for convenient construction with
    automatic path coercion.

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

    @classmethod
    def build(
        cls,
        *,
        repo: str | None = None,
        commit: str | None = None,
        paths: ScipPathConfig | None = None,
        scip_python_bin: str | None = None,
        scip_bin: str | None = None,
        modules: Sequence[ModuleRecord] | None = None,
        cfg: ScipIngestStepConfig | None = None,
    ) -> ScipResolverInput:
        """Construct input with optional path configuration.

        Convenience factory that accepts grouped path configuration.

        Parameters
        ----------
        repo
            Repository identifier.
        commit
            Commit hash.
        paths
            Path configuration (repo_root, build_dir, document_output_dir).
        scip_python_bin
            Path to scip-python binary.
        scip_bin
            Path to scip binary.
        modules
            Pre-computed module records.
        cfg
            Optional config object with snapshot info.

        Returns
        -------
        ScipResolverInput
            Constructed input container.
        """
        return cls(
            cfg=cfg,
            repo=repo,
            commit=commit,
            repo_root=paths.repo_root if paths else None,
            build_dir=paths.build_dir if paths else None,
            document_output_dir=paths.document_output_dir if paths else None,
            scip_python_bin=scip_python_bin,
            scip_bin=scip_bin,
            modules=modules,
        )


def resolve_scip_inputs_from(
    gateway: StorageGateway,
    modules_or_cfg: Sequence[ModuleRecord] | object,
    inputs: ScipResolverInput,
) -> ResolvedScipConfig:
    """
    Normalize all SCIP inputs into a required, typed config using a dataclass.

    Returns
    -------
    ResolvedScipConfig
        Normalized configuration with required fields populated.

    Raises
    ------
    ValueError
        If required parameters are missing or invalid.
    """
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

        module_map = load_module_map(
            gateway,
            actual_cfg.repo,
            actual_cfg.commit,
            language="python",
            logger=None,
        )
        filesystem_adapter = import_module(
            "codeintel.ingestion.adapters.filesystem_discovery"
        ).FilesystemDiscoveryAdapter
        module_list = list(
            filesystem_adapter.iter_modules(
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


def resolve_scip_inputs(
    gateway: StorageGateway,
    modules_or_cfg: Sequence[ModuleRecord] | object,
    inputs: ScipResolverInput,
) -> ResolvedScipConfig:
    """Normalize SCIP inputs into a required, typed config.

    Parameters
    ----------
    gateway
        Storage gateway for loading module maps.
    modules_or_cfg
        Either a sequence of module records or a ScipIngestStepConfig.
    inputs
        Input container with explicit parameters.

    Returns
    -------
    ResolvedScipConfig
        Normalized configuration with required fields populated.

    Raises
    ------
    ValueError
        If required parameters are missing or invalid.
    """
    return resolve_scip_inputs_from(gateway, modules_or_cfg, inputs)


__all__ = [
    "ResolvedScipConfig",
    "ScipResolverInput",
    "resolve_scip_inputs",
    "resolve_scip_inputs_from",
]
