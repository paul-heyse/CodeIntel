"""Inventory helpers for CLI operation registry health checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from codeintel.cli.execution.registry import OperationAlias, OperationRegistry, OperationSpec


@dataclass(frozen=True)
class AliasRule:
    """Canonicalization rule for legacy operation prefixes."""

    legacy_prefix: str
    canonical_prefix: str
    note: str


@dataclass(frozen=True)
class AliasCandidate:
    """Alias candidate inferred from canonicalization rules."""

    alias_id: str
    target_id: str
    reason: str
    canonical_exists: bool
    alias_registered: bool


@dataclass(frozen=True)
class RegistryInventory:
    """Inventory summary for registry health checks."""

    operations: tuple[OperationSpec, ...]
    aliases: tuple[OperationAlias, ...]
    legacy_operations: tuple[str, ...]
    alias_candidates: tuple[AliasCandidate, ...]


DEFAULT_ALIAS_RULES: tuple[AliasRule, ...] = (
    AliasRule(
        legacy_prefix="graphs.",
        canonical_prefix="graph.",
        note="Prefer singular graph.* operation IDs.",
    ),
    AliasRule(
        legacy_prefix="datasets.",
        canonical_prefix="dataset.",
        note="Prefer singular dataset.* operation IDs.",
    ),
)


def build_registry_inventory(
    registry: OperationRegistry,
    *,
    alias_rules: Sequence[AliasRule] | None = None,
) -> RegistryInventory:
    """Build a registry inventory summary.

    Parameters
    ----------
    registry
        Operation registry to inspect.
    alias_rules
        Optional override for canonicalization rules.

    Returns
    -------
    RegistryInventory
        Inventory summary with alias candidates and legacy operations.
    """
    rules = tuple(alias_rules or DEFAULT_ALIAS_RULES)
    operations = tuple(registry.list_operations(include_hidden=True))
    aliases = tuple(registry.list_aliases())
    legacy_operations = _find_legacy_operations(operations, rules)
    alias_candidates = _find_alias_candidates(operations, registry, rules)
    return RegistryInventory(
        operations=operations,
        aliases=aliases,
        legacy_operations=legacy_operations,
        alias_candidates=alias_candidates,
    )


def _find_legacy_operations(
    operations: Iterable[OperationSpec],
    rules: Sequence[AliasRule],
) -> tuple[str, ...]:
    legacy_ids: list[str] = []
    for spec in operations:
        for rule in rules:
            if spec.operation_id.startswith(rule.legacy_prefix):
                legacy_ids.append(spec.operation_id)
                break
    return tuple(sorted(legacy_ids))


def _find_alias_candidates(
    operations: Iterable[OperationSpec],
    registry: OperationRegistry,
    rules: Sequence[AliasRule],
) -> tuple[AliasCandidate, ...]:
    op_ids = {spec.operation_id for spec in operations}
    candidates: list[AliasCandidate] = []
    for spec in operations:
        for rule in rules:
            if not spec.operation_id.startswith(rule.legacy_prefix):
                continue
            target_id = rule.canonical_prefix + spec.operation_id[len(rule.legacy_prefix) :]
            candidates.append(
                AliasCandidate(
                    alias_id=spec.operation_id,
                    target_id=target_id,
                    reason=rule.note,
                    canonical_exists=target_id in op_ids,
                    alias_registered=registry.is_alias(spec.operation_id),
                )
            )
    candidates.sort(key=lambda candidate: candidate.alias_id)
    return tuple(candidates)


__all__ = [
    "DEFAULT_ALIAS_RULES",
    "AliasCandidate",
    "AliasRule",
    "RegistryInventory",
    "build_registry_inventory",
]
