"""Hamilton adapter for activating strict contract enforcement per node.

When strict contracts are enabled, writes should be validated against the
currently executing target's OutputContract. Hamilton node functions are tagged
with `target=<target_name>`; this hook uses those tags to activate the
ContractEnforcer for the duration of each node execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from codeintel.build.hamilton.contracts.enforcement import ContractEnforcer

if TYPE_CHECKING:
    from codeintel.build.targets import TargetGraph


class ContractEnforcementHook:
    """Hamilton lifecycle hook that activates ContractEnforcer per node."""

    def __init__(self, graph: TargetGraph, *, strict: bool) -> None:
        self._graph = graph
        self._strict = strict

    def pre_node_execute(self, *, node_name: str, **kwargs: object) -> None:
        """Activate contract enforcement based on `target` node tag."""
        _ = node_name
        node_tags_raw = kwargs.get("node_tags")
        if isinstance(node_tags_raw, dict):
            node_tags = cast("dict[str, object] | None", node_tags_raw)
        else:
            node_tags = None
        target_raw = node_tags.get("target") if node_tags else None
        if isinstance(target_raw, str):
            try:
                target = self._graph.get(target_raw)
            except KeyError:
                ContractEnforcer.deactivate()
            else:
                ContractEnforcer.activate(target, strict=self._strict)
        else:
            ContractEnforcer.deactivate()

    @staticmethod
    def post_node_execute(*, node_name: str, **kwargs: object) -> None:
        """Deactivate contract enforcement after node execution."""
        _ = node_name
        _ = kwargs
        ContractEnforcer.deactivate()


__all__ = ["ContractEnforcementHook"]
