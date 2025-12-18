"""Hamilton adapter for contract enforcement and validation result capture.

This module provides:
1. ContractEnforcementHook: Activates strict contract enforcement per node
2. ValidationResult: Dataclass for tracking per-node validation status
3. Validation result capture from Hamilton's @check_output_custom

When strict contracts are enabled, writes should be validated against the
currently executing target's OutputContract. Hamilton node functions are tagged
with `target=<target_name>`; this hook uses those tags to activate the
ContractEnforcer for the duration of each node execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from hamilton.lifecycle import base as lifecycle_base

from codeintel.build.hamilton.contracts.enforcement import ContractEnforcer
from codeintel.core.hamilton import tags as ht

if TYPE_CHECKING:
    from hamilton.node import Node

    from codeintel.build.targets import TargetGraph

_log = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation for a single node.

    Tracks whether Hamilton-native validation passed for a node,
    including any error messages and diagnostics.

    Parameters
    ----------
    node_name
        Name of the Hamilton node that was validated.
    passed
        Whether validation passed (True) or failed (False).
    message
        Human-readable message describing the validation result.
    error
        Exception message if validation failed.
    diagnostics
        Additional diagnostic information from validators.
    timestamp
        When the validation occurred.
    """

    node_name: str
    passed: bool
    message: str = ""
    error: str | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ValidationSummary:
    """Summary of validation results across all nodes.

    Aggregates validation results for reporting and analysis.

    Parameters
    ----------
    total_nodes
        Total number of nodes validated.
    passed_count
        Number of nodes that passed validation.
    failed_count
        Number of nodes that failed validation.
    skipped_count
        Number of nodes where validation was skipped.
    failed_nodes
        List of node names that failed validation.
    """

    total_nodes: int = 0
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    failed_nodes: list[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """Return True if all validated nodes passed.

        Returns
        -------
        bool
            True if no nodes failed validation.
        """
        return self.failed_count == 0


class ContractEnforcementHook(
    lifecycle_base.BasePreNodeExecute, lifecycle_base.BasePostNodeExecute
):
    """Hamilton lifecycle hook that activates ContractEnforcer per node.

    This hook integrates with Hamilton's lifecycle adapter protocol to
    enable schema validation during target execution. It:
    1. Activates the ContractEnforcer before each node executes
    2. Captures validation results from Hamilton's @check_output_custom
    3. Provides a summary of validation results after execution

    Parameters
    ----------
    graph
        Target graph for looking up target contracts.
    strict
        When True, validation failures raise exceptions.

    Examples
    --------
    >>> hook = ContractEnforcementHook(graph, strict=True)
    >>> driver = Builder().with_adapters(hook).build()
    >>> result = driver.execute(["my_node"])
    >>> summary = hook.get_validation_summary()
    >>> print(f"Passed: {summary.passed_count}, Failed: {summary.failed_count}")
    """

    def __init__(self, graph: TargetGraph, *, strict: bool) -> None:
        """Initialize the contract enforcement hook.

        Parameters
        ----------
        graph
            Target graph for looking up target contracts.
        strict
            When True, validation failures raise exceptions.
        """
        self._graph = graph
        self._strict = strict
        self._validation_results: dict[str, ValidationResult] = {}

    @property
    def validation_results(self) -> dict[str, ValidationResult]:
        """Get validation results by node name.

        Returns
        -------
        dict[str, ValidationResult]
            Mapping from node name to validation result.
        """
        return self._validation_results

    def pre_node_execute(
        self,
        *,
        node_: Node | None = None,
        node_tags: dict[str, str] | None = None,
        **context: object,
    ) -> None:
        """Activate contract enforcement based on `target` node tag.

        Parameters
        ----------
        node_
            Hamilton node being executed.
        node_tags
            Optional node tags (used when `node_` is not provided).
        context
            Additional Hamilton lifecycle context (ignored).
        """
        _ = context

        tags: dict[str, str] = {}
        if node_ is not None and isinstance(node_.tags, dict):
            tags = {
                k: v for k, v in node_.tags.items() if isinstance(k, str) and isinstance(v, str)
            }
        elif node_tags is not None:
            tags = dict(node_tags)

        target_raw = tags.get(ht.TAG_TARGET)
        if isinstance(target_raw, str) and target_raw:
            try:
                target = self._graph.get(target_raw)
            except KeyError:
                ContractEnforcer.deactivate()
            else:
                ContractEnforcer.activate(target, strict=self._strict)
        else:
            ContractEnforcer.deactivate()

    def post_node_execute(
        self,
        *,
        success: bool,
        node_: Node | None = None,
        node_name: str | None = None,
        error: Exception | None = None,
        **context: object,
    ) -> None:
        """Deactivate contract enforcement and capture validation results.

        Captures validation results from Hamilton's @check_output_custom
        for reporting and analysis.

        Parameters
        ----------
        success
            Whether the node execution succeeded.
        node_
            Hamilton node that was executed.
        node_name
            Node name when `node_` is not provided.
        error
            Exception if the node failed.
        context
            Additional Hamilton lifecycle context (ignored).
        """
        _ = context
        ContractEnforcer.deactivate()

        # Capture validation result
        resolved_node_name = node_.name if node_ is not None else node_name
        if not resolved_node_name:
            _log.warning(
                "Node execution completed without a node identifier; skipping validation capture"
            )
            return
        if success:
            self._validation_results[resolved_node_name] = ValidationResult(
                node_name=resolved_node_name,
                passed=True,
                message="Validation passed",
            )
        else:
            error_msg = str(error) if error else "Unknown error"
            # Check if this is a validation error from @check_output_custom
            is_validation_error = error is not None and (
                "validation" in error_msg.lower() or "validator" in error_msg.lower()
            )

            self._validation_results[resolved_node_name] = ValidationResult(
                node_name=resolved_node_name,
                passed=False,
                message="Validation failed" if is_validation_error else "Execution failed",
                error=error_msg,
                diagnostics={"is_validation_error": is_validation_error},
            )
            _log.warning(
                "Node %s failed: %s (validation_error=%s)",
                resolved_node_name,
                error_msg[:200],
                is_validation_error,
            )

    def get_validation_summary(self) -> ValidationSummary:
        """Get summary of validation results across all nodes.

        Returns
        -------
        ValidationSummary
            Aggregated validation statistics.

        Examples
        --------
        >>> summary = hook.get_validation_summary()
        >>> if not summary.all_passed:
        ...     print(f"Failed nodes: {summary.failed_nodes}")
        """
        passed = sum(1 for r in self._validation_results.values() if r.passed)
        failed = sum(1 for r in self._validation_results.values() if not r.passed)
        skipped = sum(
            1 for r in self._validation_results.values() if r.diagnostics.get("skipped", False)
        )
        failed_nodes = [name for name, r in self._validation_results.items() if not r.passed]

        return ValidationSummary(
            total_nodes=len(self._validation_results),
            passed_count=passed,
            failed_count=failed,
            skipped_count=skipped,
            failed_nodes=failed_nodes,
        )

    def clear_results(self) -> None:
        """Clear stored validation results.

        Call this before a new execution run to reset state.
        """
        self._validation_results.clear()


__all__ = [
    "ContractEnforcementHook",
    "ValidationResult",
    "ValidationSummary",
]
