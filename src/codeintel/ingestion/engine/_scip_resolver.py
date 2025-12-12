"""Helper utilities to resolve SCIP ingestion inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.ingestion.ports.discovery import ModuleRecord


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
    scip_python_bin
        Path to scip-python binary.
    scip_bin
        Path to scip binary.
    """

    repo_root: Path | None = None
    build_dir: Path | None = None
    document_output_dir: Path | None = None
    scip_python_bin: str | None = None
    scip_bin: str | None = None

    @classmethod
    def from_strings(
        cls,
        *,
        repo_root: Path | str | None = None,
        build_dir: Path | str | None = None,
        document_output_dir: Path | str | None = None,
        scip_python_bin: str | None = None,
        scip_bin: str | None = None,
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
        scip_python_bin
            Path to scip-python binary.
        scip_bin
            Path to scip binary.

        Returns
        -------
        ScipPathConfig
            Path configuration with coerced paths.
        """
        return cls(
            repo_root=Path(repo_root) if repo_root else None,
            build_dir=Path(build_dir) if build_dir else None,
            document_output_dir=Path(document_output_dir) if document_output_dir else None,
            scip_python_bin=scip_python_bin,
            scip_bin=scip_bin,
        )


@dataclass(frozen=True)
class ScipResolverInput:
    """Input parameters for SCIP configuration resolution.

    Bundles optional parameters to reduce function argument count.
    Provide explicit parameters (repo, commit, repo_root, build_dir,
    document_output_dir) for resolution.

    Use the `build()` factory method for convenient construction with
    automatic path coercion.

    Parameters
    ----------
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
        modules: Sequence[ModuleRecord] | None = None,
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
            Path and binary configuration.
        modules
            Pre-computed module records.

        Returns
        -------
        ScipResolverInput
            Constructed input container.
        """
        return cls(
            repo=repo,
            commit=commit,
            repo_root=paths.repo_root if paths else None,
            build_dir=paths.build_dir if paths else None,
            document_output_dir=paths.document_output_dir if paths else None,
            scip_python_bin=paths.scip_python_bin if paths else None,
            scip_bin=paths.scip_bin if paths else None,
            modules=modules,
        )


def resolve_scip_inputs(
    modules: Sequence[ModuleRecord] | None,
    inputs: ScipResolverInput,
) -> ResolvedScipConfig:
    """Normalize SCIP inputs into a required, typed config.

    This explicit-only version requires all configuration values to be
    provided via `ScipResolverInput`. Modules can be passed either as
    the first argument or via `inputs.modules`.

    Parameters
    ----------
    modules
        Sequence of module records (takes precedence over inputs.modules).
    inputs
        Input container with explicit parameters.

    Returns
    -------
    ResolvedScipConfig
        Normalized configuration with required fields populated.

    Raises
    ------
    ValueError
        If required parameters (repo, commit, repo_root, build_dir,
        document_output_dir) are missing.
    """
    repo = inputs.repo
    commit = inputs.commit
    repo_root = inputs.repo_root
    build_dir = inputs.build_dir
    document_output_dir = inputs.document_output_dir
    scip_python_bin = inputs.scip_python_bin
    scip_bin = inputs.scip_bin

    module_list = list(modules) if modules is not None else []
    if not module_list and inputs.modules is not None:
        module_list = list(inputs.modules)

    if (
        repo is None
        or commit is None
        or repo_root is None
        or build_dir is None
        or document_output_dir is None
    ):
        msg = "repo, commit, repo_root, build_dir, and document_output_dir are required"
        raise ValueError(msg)

    return ResolvedScipConfig(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        build_dir=build_dir,
        document_output_dir=document_output_dir,
        scip_python_bin=scip_python_bin,
        scip_bin=scip_bin,
        modules=module_list,
    )


__all__ = [
    "ResolvedScipConfig",
    "ScipPathConfig",
    "ScipResolverInput",
    "resolve_scip_inputs",
]
