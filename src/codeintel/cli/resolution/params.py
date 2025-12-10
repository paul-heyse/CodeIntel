"""Canonical runtime parameters type.

This module defines RuntimeParams, the single source of truth for
runtime parameters that replaces all RuntimeCliOptions variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codeintel.cli.commands import RuntimeCLI
    from codeintel.cli.execution.context import ExecutionContext


@dataclass(frozen=True)
class BackendFlags:
    """Graph backend configuration flags.

    Parameters
    ----------
    use_gpu
        Whether to attempt GPU acceleration.
    backend
        Backend selection: "auto", "cpu", or "nx-cugraph".
    strict
        Whether to enforce strict backend compatibility.

    Examples
    --------
    >>> flags = BackendFlags()
    >>> flags.use_gpu
    False
    >>> flags.backend
    'auto'
    """

    use_gpu: bool = False
    backend: str = "auto"
    strict: bool = False


@dataclass(frozen=True)
class RuntimeParams:
    """Canonical runtime parameters from any input source.

    This is THE type for runtime parameters. All other RuntimeCliOptions
    variants are deprecated in favor of this single type.

    Parameters
    ----------
    project_root
        Root directory for project file discovery.
    repo
        Repository identifier (e.g., "org/repo").
    commit
        Commit SHA.
    db_path
        Explicit database path.
    build_dir
        Build output directory.
    repo_root
        Repository root path.
    document_output_dir
        Document export directory.
    backend
        Graph backend configuration.

    Examples
    --------
    >>> params = RuntimeParams.minimal(Path("/project"))
    >>> params.project_root
    PosixPath('/project')

    >>> params = RuntimeParams(repo="org/repo", commit="abc123")
    >>> params.repo
    'org/repo'
    """

    project_root: Path | None = None
    repo: str | None = None
    commit: str | None = None
    db_path: Path | None = None
    build_dir: Path | None = None
    repo_root: Path | None = None
    document_output_dir: Path | None = None
    backend: BackendFlags = field(default_factory=BackendFlags)

    # --- Factory Methods ---

    @classmethod
    def from_context(cls, ctx: ExecutionContext) -> RuntimeParams:
        """Extract RuntimeParams from ExecutionContext.params dict.

        The context params may contain any subset of fields.
        Missing fields use defaults.

        Parameters
        ----------
        ctx
            Execution context with params dict.

        Returns
        -------
        RuntimeParams
            Extracted parameters.

        Examples
        --------
        >>> from codeintel.cli.execution.context import ExecutionContext
        >>> ctx = ExecutionContext.for_sync("op", {"repo": "org/repo"})
        >>> params = RuntimeParams.from_context(ctx)  # doctest: +SKIP
        >>> params.repo  # doctest: +SKIP
        'org/repo'
        """
        params = ctx.params

        backend_raw = params.get("backend", {})
        backend = (
            BackendFlags(
                use_gpu=_get_bool(backend_raw, "use_gpu", default=False),
                backend=_get_str(backend_raw, "backend", default="auto"),
                strict=_get_bool(backend_raw, "strict", default=False),
            )
            if isinstance(backend_raw, dict)
            else BackendFlags()
        )

        return cls(
            project_root=_to_path(params.get("project_root")),
            repo=_to_str(params.get("repo")),
            commit=_to_str(params.get("commit")),
            db_path=_to_path(params.get("db_path")),
            build_dir=_to_path(params.get("build_dir")),
            repo_root=_to_path(params.get("repo_root")),
            document_output_dir=_to_path(params.get("document_output_dir")),
            backend=backend,
        )

    @classmethod
    def from_cyclopts(cls, runtime_cli: RuntimeCLI) -> RuntimeParams:
        """Convert Cyclopts RuntimeCLI to canonical RuntimeParams.

        RuntimeCLI is a Cyclopts-specific dataclass with Parameter
        annotations. This method extracts values into the canonical type.

        Parameters
        ----------
        runtime_cli
            Cyclopts runtime CLI dataclass.

        Returns
        -------
        RuntimeParams
            Canonical parameters.

        Examples
        --------
        >>> from codeintel.cli.commands import RuntimeCLI
        >>> cli = RuntimeCLI(repo="org/repo", commit="abc123")
        >>> params = RuntimeParams.from_cyclopts(cli)
        >>> params.repo
        'org/repo'
        """
        return cls(
            project_root=runtime_cli.project_root,
            repo=runtime_cli.repo,
            commit=runtime_cli.commit,
            db_path=runtime_cli.db_path,
            build_dir=runtime_cli.build_dir,
            repo_root=runtime_cli.repo_root,
            document_output_dir=runtime_cli.document_output_dir,
            backend=BackendFlags(),  # RuntimeCLI doesn't include backend
        )

    @classmethod
    def minimal(cls, project_root: Path | None = None) -> RuntimeParams:
        """Create minimal params for simple commands.

        Use for commands that only need project discovery (ide hints, etc).

        Parameters
        ----------
        project_root
            Optional project root path.

        Returns
        -------
        RuntimeParams
            Minimal parameters.

        Examples
        --------
        >>> params = RuntimeParams.minimal()
        >>> params.project_root is None
        True
        """
        return cls(project_root=project_root)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeParams:
        """Create RuntimeParams from a dictionary.

        Parameters
        ----------
        data
            Dictionary with parameter values.

        Returns
        -------
        RuntimeParams
            Parsed parameters.

        Examples
        --------
        >>> data = {"repo": "org/repo", "commit": "abc123"}
        >>> params = RuntimeParams.from_dict(data)
        >>> params.repo
        'org/repo'
        """
        backend_raw = data.get("backend", {})
        backend = (
            BackendFlags(
                use_gpu=_get_bool(backend_raw, "use_gpu", default=False),
                backend=_get_str(backend_raw, "backend", default="auto"),
                strict=_get_bool(backend_raw, "strict", default=False),
            )
            if isinstance(backend_raw, dict)
            else BackendFlags()
        )

        return cls(
            project_root=_to_path(data.get("project_root")),
            repo=_to_str(data.get("repo")),
            commit=_to_str(data.get("commit")),
            db_path=_to_path(data.get("db_path")),
            build_dir=_to_path(data.get("build_dir")),
            repo_root=_to_path(data.get("repo_root")),
            document_output_dir=_to_path(data.get("document_output_dir")),
            backend=backend,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.

        Examples
        --------
        >>> params = RuntimeParams(repo="org/repo")
        >>> d = params.to_dict()
        >>> d["repo"]
        'org/repo'
        """
        return {
            "project_root": str(self.project_root) if self.project_root else None,
            "repo": self.repo,
            "commit": self.commit,
            "db_path": str(self.db_path) if self.db_path else None,
            "build_dir": str(self.build_dir) if self.build_dir else None,
            "repo_root": str(self.repo_root) if self.repo_root else None,
            "document_output_dir": (
                str(self.document_output_dir) if self.document_output_dir else None
            ),
            "backend": {
                "use_gpu": self.backend.use_gpu,
                "backend": self.backend.backend,
                "strict": self.backend.strict,
            },
        }


# --- Helper Functions ---


def _to_path(value: object) -> Path | None:
    """Convert value to Path or None.

    Parameters
    ----------
    value
        Value to convert.

    Returns
    -------
    Path | None
        Converted path or None.
    """
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _to_str(value: object) -> str | None:
    """Convert value to string or None.

    Parameters
    ----------
    value
        Value to convert.

    Returns
    -------
    str | None
        Converted string or None.
    """
    if value is None:
        return None
    return str(value)


def _get_bool(data: object, key: str, *, default: bool) -> bool:
    """Get boolean from dict-like object.

    Parameters
    ----------
    data
        Dictionary to get value from.
    key
        Key to look up.
    default
        Default value if not found.

    Returns
    -------
    bool
        Boolean value.
    """
    if not isinstance(data, dict):
        return default
    return bool(data.get(key, default))


def _get_str(data: object, key: str, *, default: str) -> str:
    """Get string from dict-like object.

    Parameters
    ----------
    data
        Dictionary to get value from.
    key
        Key to look up.
    default
        Default value if not found.

    Returns
    -------
    str
        String value.
    """
    if not isinstance(data, dict):
        return default
    value = data.get(key, default)
    return str(value) if value is not None else default


__all__ = [
    "BackendFlags",
    "RuntimeParams",
]
