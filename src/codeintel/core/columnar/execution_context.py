"""Execution context for Acero plans or table fallbacks."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.core.columnar.dedupe_ops import DedupeTier


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Defaults for scan/runtime behavior across columnar pipelines."""

    name: str | None = None
    scan_profile: str | None = None
    implicit_ordering: bool | None = None
    require_sequenced_output: bool | None = None
    determinism: DedupeTier | None = None
    use_threads: bool | None = None
    provenance: bool | None = None

    def resolve_use_threads(self, *, default: bool) -> bool:
        """Return the resolved use_threads value.

        Returns
        -------
        bool
            Effective use_threads toggle.
        """
        return default if self.use_threads is None else self.use_threads

    def resolve_determinism(self, default: DedupeTier) -> DedupeTier:
        """Return the resolved determinism tier.

        Returns
        -------
        DedupeTier
            Effective determinism tier.
        """
        return default if self.determinism is None else self.determinism

    def resolve_provenance(self, *, default: bool) -> bool:
        """Return the resolved provenance toggle.

        Returns
        -------
        bool
            Effective provenance toggle.
        """
        return default if self.provenance is None else self.provenance

    def resolve_implicit_ordering(self, *, default: bool | None) -> bool | None:
        """Return the resolved implicit ordering default.

        Returns
        -------
        bool | None
            Effective implicit ordering toggle.
        """
        return default if self.implicit_ordering is None else self.implicit_ordering

    def resolve_require_sequenced_output(self, *, default: bool | None) -> bool | None:
        """Return the resolved sequenced output default.

        Returns
        -------
        bool | None
            Effective sequenced output toggle.
        """
        return default if self.require_sequenced_output is None else self.require_sequenced_output


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Execution context for Acero plans or table fallbacks."""

    use_threads: bool = True
    determinism: DedupeTier = "stable_set"
    combine_chunks: bool = True
    provenance: bool = False
    runtime_profile: RuntimeProfile | None = None
