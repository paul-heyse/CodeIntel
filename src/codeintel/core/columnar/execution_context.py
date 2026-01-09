"""Execution context for Acero plans or table fallbacks."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.core.columnar.dedupe_ops import DedupeTier


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Execution context for Acero plans or table fallbacks."""

    use_threads: bool = True
    determinism: DedupeTier = "throughput"
    combine_chunks: bool = True
    provenance: bool = False
