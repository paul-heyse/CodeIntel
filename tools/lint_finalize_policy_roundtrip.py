"""Round-trip validation for finalize policy serialization."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence

from codeintel.config.datasets.contracts import get_table_schemas
from codeintel.core.schemas.serde import (
    table_schema_from_json_obj,
    table_schema_to_json_obj,
)

_MAX_DIFFS = 50


def _diffs_limit_reached(diffs: list[str]) -> bool:
    return len(diffs) >= _MAX_DIFFS


def _append_diff(diffs: list[str], message: str) -> bool:
    diffs.append(message)
    return _diffs_limit_reached(diffs)


def _collect_mapping_diffs(
    left: Mapping[object, object],
    right: Mapping[object, object],
    *,
    path: str,
    diffs: list[str],
) -> None:
    left_keys = set(left.keys())
    right_keys = set(right.keys())
    for key in sorted(left_keys - right_keys):
        if _append_diff(diffs, f"{path}.{key}: missing from round-trip"):
            return
    for key in sorted(right_keys - left_keys):
        if _append_diff(diffs, f"{path}.{key}: unexpected in round-trip"):
            return
    for key in sorted(left_keys & right_keys):
        _collect_diffs(
            left[key],
            right[key],
            path=f"{path}.{key}",
            diffs=diffs,
        )
        if _diffs_limit_reached(diffs):
            return


def _collect_sequence_diffs(
    left: Sequence[object],
    right: Sequence[object],
    *,
    path: str,
    diffs: list[str],
) -> None:
    if len(left) != len(right):
        _append_diff(diffs, f"{path}: length {len(left)} != {len(right)}")
        return
    for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
        _collect_diffs(
            left_item,
            right_item,
            path=f"{path}[{index}]",
            diffs=diffs,
        )
        if _diffs_limit_reached(diffs):
            return


def _collect_diffs(
    left: object,
    right: object,
    *,
    path: str,
    diffs: list[str],
) -> None:
    if _diffs_limit_reached(diffs):
        return
    if type(left) is not type(right):
        _append_diff(diffs, f"{path}: {type(left).__name__} != {type(right).__name__}")
        return
    if isinstance(left, Mapping):
        _collect_mapping_diffs(left, right, path=path, diffs=diffs)
        return
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes, bytearray)):
        _collect_sequence_diffs(left, right, path=path, diffs=diffs)
        return
    if left != right:
        _append_diff(diffs, f"{path}: {left!r} != {right!r}")


def main() -> int:
    """Validate finalize policy serialization round-trip consistency.

    Returns
    -------
    int
        Exit code (0 on success, 1 on mismatches).
    """
    errors: list[str] = []
    for table_key, schema in sorted(get_table_schemas().items()):
        payload = table_schema_to_json_obj(schema)
        round_trip = table_schema_to_json_obj(table_schema_from_json_obj(payload))
        if payload == round_trip:
            continue
        diffs: list[str] = []
        _collect_diffs(payload, round_trip, path=table_key, diffs=diffs)
        if diffs:
            errors.extend(diffs)
        else:
            errors.append(f"{table_key}: round-trip mismatch")
    if errors:
        sys.stderr.write("Finalize policy round-trip mismatches detected:\n")
        sys.stderr.write("\n".join(errors) + "\n")
        return 1
    sys.stdout.write("Finalize policy round-trip checks passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
