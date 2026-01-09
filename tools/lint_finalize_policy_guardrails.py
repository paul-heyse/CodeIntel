"""Guardrails for finalize policy completeness."""

from __future__ import annotations

import sys

from codeintel.core.schemas.service import get_schema_service

_DEDUPE_POLICY_REQUIRED = frozenset(
    {
        "core.file_state",
        "core.scip_external_symbols",
    }
)


def _has_list_type(column_type: str) -> bool:
    return "LIST(" in column_type.upper()


def _policy_has_list_guardrails(policy_invariants: object, policy_list_policies: object) -> bool:
    if policy_list_policies:
        return True
    if not policy_invariants:
        return False
    return any(
        getattr(invariant, "kind", None) == "list_alignment"
        for invariant in policy_invariants
    )


def main() -> int:
    service = get_schema_service()
    missing_list_policy: list[str] = []
    missing_dedupe: list[str] = []

    for schema in service.iter_table_schemas():
        table_key = schema.table_key
        has_list = any(_has_list_type(column.type) for column in schema.columns)
        policy = schema.finalize_policy
        if has_list:
            if policy is None or not _policy_has_list_guardrails(
                policy.invariants,
                policy.list_policies,
            ):
                missing_list_policy.append(table_key)
        if table_key in _DEDUPE_POLICY_REQUIRED:
            if policy is None or policy.dedupe is None:
                missing_dedupe.append(table_key)

    errors: list[str] = []
    if missing_list_policy:
        errors.append(
            "Finalize policy missing list guardrails for: "
            + ", ".join(sorted(missing_list_policy))
        )
    if missing_dedupe:
        errors.append(
            "Finalize policy missing dedupe specs for: " + ", ".join(sorted(missing_dedupe))
        )

    if errors:
        sys.stderr.write("\n".join(errors) + "\n")
        return 1

    sys.stdout.write("Finalize policy guardrails passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
