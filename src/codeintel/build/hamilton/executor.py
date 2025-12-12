"""Hamilton-based build executor.

This module provides HamiltonBuildExecutor, which is a drop-in alternative
to the legacy BuildExecutor. It uses Hamilton's Driver for DAG-based
execution of build targets.

Design Principles
-----------------
1. HamiltonBuildExecutor.run() is the main entry point for execution.
2. It maps target names to Hamilton node names automatically.
3. Results are returned in a structured HamiltonBuildResult.
4. The executor integrates with existing manifest/tracking infrastructure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from codeintel.build.hamilton.driver_factory import build_driver, target_to_node_name
from codeintel.build.hamilton.manifest_hook import TargetRunRecord

if TYPE_CHECKING:
    from codeintel.build.hamilton.env import BuildEnv

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HamiltonBuildResult:
    """Result of a Hamilton-based build execution.

    This captures the outputs from Hamilton Driver execution along with
    metadata about what was requested.

    Attributes
    ----------
    requested
        Tuple of target names that were requested.
    outputs
        Dictionary mapping Hamilton node names to their outputs
        (TargetRunRecord instances).
    success
        Whether all requested targets succeeded.
    failed_targets
        Names of targets that failed during execution.

    Examples
    --------
    >>> result = executor.run(env=env, targets=["modules", "ast"])
    >>> if result.success:
    ...     print(f"Completed {len(result.requested)} targets")
    ... else:
    ...     print(f"Failed: {result.failed_targets}")
    """

    requested: tuple[str, ...]
    outputs: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    failed_targets: tuple[str, ...] = ()

    def get_record(self, target_name: str) -> TargetRunRecord | None:
        """Get the execution record for a target.

        Parameters
        ----------
        target_name
            Target name (e.g., "modules", not "t__modules").

        Returns
        -------
        TargetRunRecord | None
            Execution record if available, None otherwise.
        """
        node_name = target_to_node_name(target_name)
        if node_name is None:
            return None
        value = self.outputs.get(node_name)
        if isinstance(value, TargetRunRecord):
            return value
        return None


class HamiltonBuildExecutor:
    """Execute build targets using Hamilton Driver.

    This executor provides an alternative to the legacy BuildExecutor,
    using Hamilton for DAG-based dependency resolution and execution.

    Parameters
    ----------
    profile
        Optional policy profile name (e.g., "fast", "full", "default").

    Examples
    --------
    >>> executor = HamiltonBuildExecutor(profile="default")
    >>> result = executor.run(env=env, targets=["function_metrics"])
    >>> for target in result.requested:
    ...     record = result.get_record(target)
    ...     print(f"{target}: {record.status}")
    """

    def __init__(self, *, profile: str | None = None) -> None:
        """Initialize the Hamilton executor.

        Parameters
        ----------
        profile
            Optional policy profile name.
        """
        self._profile = profile

    @property
    def profile(self) -> str | None:
        """Return the configured profile name.

        Returns
        -------
        str | None
            Profile name or None if not configured.
        """
        return self._profile

    def run(
        self,
        *,
        env: BuildEnv,
        targets: list[str],
    ) -> HamiltonBuildResult:
        """Execute build targets using Hamilton.

        Builds a Hamilton Driver, maps target names to node names,
        and executes the DAG to compute the requested targets.

        Parameters
        ----------
        env
            Build environment with gateway, snapshot, providers, etc.
        targets
            List of target names to compute (e.g., ["function_metrics"]).

        Returns
        -------
        HamiltonBuildResult
            Result with outputs and success status.

        Examples
        --------
        >>> result = executor.run(env=env, targets=["modules", "ast"])
        >>> print(f"Completed: {result.success}")
        """
        log.info(
            "build.hamilton.executor.start targets=%s profile=%s",
            targets,
            self._profile,
        )

        # Map target names to Hamilton node names
        node_names: list[str] = []
        missing_targets: list[str] = []

        for target in targets:
            node_name = target_to_node_name(target)
            if node_name is None:
                log.warning(
                    "build.hamilton.executor.unknown_target target=%s",
                    target,
                )
                missing_targets.append(target)
            else:
                node_names.append(node_name)

        if missing_targets:
            log.error(
                "build.hamilton.executor.missing_targets targets=%s",
                missing_targets,
            )
            return HamiltonBuildResult(
                requested=tuple(targets),
                outputs={},
                success=False,
                failed_targets=tuple(missing_targets),
            )

        # Build Hamilton Driver
        config: dict[str, Any] = {"profile": self._profile or "default"}
        runtime = build_driver(config=config)

        # Execute the DAG
        try:
            # Hamilton's execute() accepts list of strings, functions, or nodes
            outputs = runtime.dr.execute(
                final_vars=list(node_names),
                inputs={"env": env, "graph": runtime.graph},
            )
        except Exception:
            log.exception("build.hamilton.executor.error")
            return HamiltonBuildResult(
                requested=tuple(targets),
                outputs={},
                success=False,
                failed_targets=tuple(targets),
            )

        # Check for failed targets
        failed: list[str] = []
        for target in targets:
            node_name = target_to_node_name(target)
            if node_name is None:
                continue
            record = outputs.get(node_name)
            if isinstance(record, TargetRunRecord) and record.status == "failed":
                failed.append(target)

        success = len(failed) == 0

        log.info(
            "build.hamilton.executor.complete success=%s failed=%s",
            success,
            failed,
        )

        return HamiltonBuildResult(
            requested=tuple(targets),
            outputs=outputs,
            success=success,
            failed_targets=tuple(failed),
        )


__all__ = [
    "HamiltonBuildExecutor",
    "HamiltonBuildResult",
]
