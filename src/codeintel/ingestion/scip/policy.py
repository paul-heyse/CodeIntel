"""Decision policy for SCIP incremental rebuilds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScipIncrementalDecision:
    """Decision output for SCIP incremental indexing."""

    mode: str
    reason: str
    total_modules: int
    changed_count: int
    changed_ratio: float | None
    threshold_count: int
    threshold_ratio: float
    ratio_gate_min_modules: int
    ratio_gate_min_changed: int
    ratio_gate_applied: bool


@dataclass(frozen=True)
class ScipIncrementalInputs:
    """Inputs required to evaluate an incremental rebuild decision."""

    total_modules: int
    changed_count: int
    changed_ratio: float | None
    output_exists: bool
    options_mismatch: bool
    force_full_rebuild: bool


@dataclass(frozen=True)
class ScipIncrementalPolicy:
    """Policy object encapsulating incremental rebuild decisions."""

    full_rebuild_threshold_count: int
    full_rebuild_threshold_ratio: float
    ratio_gate_min_modules: int
    ratio_gate_min_changed: int

    def decide(self, inputs: ScipIncrementalInputs) -> ScipIncrementalDecision:
        """Decide whether to run full or incremental indexing.

        Returns
        -------
        ScipIncrementalDecision
            Decision record describing the chosen mode and rationale.
        """
        ratio_gate_applied = (
            inputs.total_modules >= self.ratio_gate_min_modules
            and inputs.changed_count >= self.ratio_gate_min_changed
        )
        if inputs.force_full_rebuild or not inputs.output_exists:
            return self._full_decision(
                reason="force_full_rebuild",
                total_modules=inputs.total_modules,
                changed_count=inputs.changed_count,
                changed_ratio=inputs.changed_ratio,
                ratio_gate_applied=ratio_gate_applied,
            )
        if inputs.options_mismatch:
            return self._full_decision(
                reason="options_mismatch",
                total_modules=inputs.total_modules,
                changed_count=inputs.changed_count,
                changed_ratio=inputs.changed_ratio,
                ratio_gate_applied=ratio_gate_applied,
            )
        if (
            self.full_rebuild_threshold_count > 0
            and inputs.changed_count >= self.full_rebuild_threshold_count
        ):
            return self._full_decision(
                reason="threshold_count",
                total_modules=inputs.total_modules,
                changed_count=inputs.changed_count,
                changed_ratio=inputs.changed_ratio,
                ratio_gate_applied=ratio_gate_applied,
            )
        if (
            ratio_gate_applied
            and self.full_rebuild_threshold_ratio > 0
            and inputs.changed_ratio is not None
            and inputs.changed_ratio >= self.full_rebuild_threshold_ratio
        ):
            return self._full_decision(
                reason="threshold_ratio",
                total_modules=inputs.total_modules,
                changed_count=inputs.changed_count,
                changed_ratio=inputs.changed_ratio,
                ratio_gate_applied=ratio_gate_applied,
            )
        return ScipIncrementalDecision(
            mode="incremental",
            reason="incremental",
            total_modules=inputs.total_modules,
            changed_count=inputs.changed_count,
            changed_ratio=inputs.changed_ratio,
            threshold_count=self.full_rebuild_threshold_count,
            threshold_ratio=self.full_rebuild_threshold_ratio,
            ratio_gate_min_modules=self.ratio_gate_min_modules,
            ratio_gate_min_changed=self.ratio_gate_min_changed,
            ratio_gate_applied=ratio_gate_applied,
        )

    def _full_decision(
        self,
        *,
        reason: str,
        total_modules: int,
        changed_count: int,
        changed_ratio: float | None,
        ratio_gate_applied: bool,
    ) -> ScipIncrementalDecision:
        return ScipIncrementalDecision(
            mode="full",
            reason=reason,
            total_modules=total_modules,
            changed_count=changed_count,
            changed_ratio=changed_ratio,
            threshold_count=self.full_rebuild_threshold_count,
            threshold_ratio=self.full_rebuild_threshold_ratio,
            ratio_gate_min_modules=self.ratio_gate_min_modules,
            ratio_gate_min_changed=self.ratio_gate_min_changed,
            ratio_gate_applied=ratio_gate_applied,
        )


__all__ = [
    "ScipIncrementalDecision",
    "ScipIncrementalInputs",
    "ScipIncrementalPolicy",
]
