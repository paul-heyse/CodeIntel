"""Registry for repo fixture writers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tests._helpers.fixtures.repos import (
    RepoFixture,
    write_generated_noise_fixture,
    write_large_file_fixture,
    write_monorepo_fixture,
    write_scoped_paths_fixture,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


RepoWriter = Callable[..., RepoFixture]


@dataclass(frozen=True)
class RepoFixtureEntry:
    """Fixture registry entry."""

    tag: str
    description: str
    writer: RepoWriter


_FIXTURES: dict[str, RepoFixtureEntry] = {
    "monorepo": RepoFixtureEntry(
        tag="monorepo",
        description="Multi-language monorepo with Python services and libs",
        writer=write_monorepo_fixture,
    ),
    "generated": RepoFixtureEntry(
        tag="generated",
        description="Repo with generated file noise",
        writer=write_generated_noise_fixture,
    ),
    "large-files": RepoFixtureEntry(
        tag="large-files",
        description="Repo with files exceeding size thresholds",
        writer=write_large_file_fixture,
    ),
    "scoped": RepoFixtureEntry(
        tag="scoped",
        description="Repo with scoped path filtering",
        writer=write_scoped_paths_fixture,
    ),
}


def list_repo_fixtures() -> Mapping[str, RepoFixtureEntry]:
    """Return the registered repo fixture entries.

    Returns
    -------
    Mapping[str, RepoFixtureEntry]
        Mapping of tag to fixture entry.
    """
    return dict(_FIXTURES)


def get_repo_fixture(tag: str) -> RepoFixtureEntry:
    """Return a fixture entry by tag.

    Parameters
    ----------
    tag
        Registry tag for the fixture.

    Returns
    -------
    RepoFixtureEntry
        Fixture entry for the tag.

    Raises
    ------
    KeyError
        If no fixture is registered for the tag.
    """
    entry = _FIXTURES.get(tag)
    if entry is None:
        message = f"No repo fixture registered for tag {tag!r}"
        raise KeyError(message)
    return entry


def build_repo_fixture(
    tag: str,
    repo_root: Path,
    **kwargs: object,
) -> RepoFixture:
    """Build a repo fixture by tag.

    Parameters
    ----------
    tag
        Registry tag for the fixture.
    repo_root
        Repository root to populate.
    **kwargs
        Additional keyword arguments forwarded to the writer.

    Returns
    -------
    RepoFixture
        Generated fixture metadata.
    """
    entry = get_repo_fixture(tag)
    return entry.writer(repo_root, **kwargs)


__all__ = [
    "RepoFixtureEntry",
    "build_repo_fixture",
    "get_repo_fixture",
    "list_repo_fixtures",
]
