"""Evidence bundle helpers for entrypoint analytics tests."""

from __future__ import annotations

from dataclasses import dataclass

from tests._helpers.context import QueryRow, SeedPack, TestContext
from tests._helpers.seeds import ENTRYPOINTS_PACK


@dataclass(frozen=True)
class EntrypointEvidenceBundle:
    """Collected entrypoint rows and related evidence rows."""

    entrypoints: list[QueryRow]
    tests: list[QueryRow]


def build_entrypoint_evidence(
    ctx: TestContext, *, pack: SeedPack | None = None
) -> EntrypointEvidenceBundle:
    """Ensure entrypoints are seeded and return entrypoint/test rows together."""
    ctx.require(pack or ENTRYPOINTS_PACK)
    entrypoints = ctx.query(
        """
        SELECT * FROM analytics.entrypoints
        WHERE repo = ? AND commit = ?
        """,
        [ctx.repo, ctx.commit],
    )
    tests = ctx.query(
        """
        SELECT * FROM analytics.entrypoint_tests
        WHERE repo = ? AND commit = ?
        """,
        [ctx.repo, ctx.commit],
    )
    return EntrypointEvidenceBundle(entrypoints=entrypoints, tests=tests)


__all__ = [
    "EntrypointEvidenceBundle",
    "build_entrypoint_evidence",
]
