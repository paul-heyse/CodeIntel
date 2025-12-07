"""Fakes and builders for ingestion plugin contexts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO


@dataclass
class RecordingGateway:
    """Gateway stub that records SQL executed via its .con property."""

    result_rows: list[tuple[object, ...]] = field(default_factory=list)
    executions: list[tuple[str, list[object]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.con = _RecordingConnection(self)


class _RecordingConnection:
    def __init__(self, gateway: RecordingGateway) -> None:
        self._gateway = gateway

    def execute(self, sql: str, params: Iterable[object]) -> _RecordingConnection:
        self._gateway.executions.append((sql, list(params)))
        return self

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._gateway.result_rows


@dataclass
class RecordingResources:
    """Minimal resources bundle for plugin contexts."""

    modules: tuple[str, ...] = ()
    type_checker: object | None = None


@dataclass
class RecordingContext:
    """Lightweight stand-in for TargetExecutionContext used in plugins."""

    repo_root: Path
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    gateway: RecordingGateway = field(default_factory=RecordingGateway)
    resources: RecordingResources = field(default_factory=RecordingResources)


def make_target_context(
    *,
    repo_root: Path,
    modules: Iterable[str] | None = None,
    snapshot: tuple[str, str] = (DEFAULT_REPO, DEFAULT_COMMIT),
    type_checker: object | None = None,
    gateway: RecordingGateway | None = None,
) -> RecordingContext:
    """Construct a recording context matching plugin expectations.

    Returns
    -------
    RecordingContext
        Context populated with the provided parameters.
    """
    repo, commit = snapshot
    resources = RecordingResources(
        modules=tuple(modules) if modules is not None else (),
        type_checker=type_checker,
    )
    return RecordingContext(
        repo_root=repo_root,
        repo=repo,
        commit=commit,
        gateway=gateway or RecordingGateway(),
        resources=resources,
    )


def build_repo_tree(root: Path, files: Mapping[str, str]) -> Path:
    """Write a set of files relative to root and return the root path.

    Returns
    -------
    Path
        Repository root containing the written files.
    """
    root.mkdir(parents=True, exist_ok=True)
    for rel_path, content in files.items():
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return root


__all__ = [
    "RecordingContext",
    "RecordingGateway",
    "RecordingResources",
    "build_repo_tree",
    "make_target_context",
]
