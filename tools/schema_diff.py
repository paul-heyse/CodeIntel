"""Schema diff gate for manifest evolution checks."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import yaml

from codeintel.build.schemas.compile import (
    SchemaManifestContext,
    SchemaManifestRequest,
    compile_schema_manifest,
)
from codeintel.build.schemas.diff import ManifestDiffResult, SchemaDiff, compute_manifest_diffs
from codeintel.build.schemas.registry import get_schema_provider
from codeintel.cli.handlers.runtime_helpers import compose_cli_runtime_bundle
from codeintel.cli.resolution import resolve_from_params
from codeintel.core.manifests import (
    ExportArtifact,
    ExportArtifactKind,
    SchemaManifest,
    read_manifest_json,
)
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.core.schemas.serde import table_schema_from_json_obj
from codeintel.storage.gateway import open_inference_gateway

BreakingChangeKind = Literal[
    "artifact_removed",
    "table_column_nullable_changed",
    "table_column_removed",
    "table_column_type_changed",
    "table_removed",
    "view_column_nullable_changed",
    "view_column_removed",
    "view_column_type_changed",
    "view_removed",
]

_BREAKING_KINDS: tuple[str, ...] = (
    "artifact_removed",
    "table_column_nullable_changed",
    "table_column_removed",
    "table_column_type_changed",
    "table_removed",
    "view_column_nullable_changed",
    "view_column_removed",
    "view_column_type_changed",
    "view_removed",
)

_EXPORT_ARTIFACT_KINDS: frozenset[str] = frozenset({"parquet", "jsonl", "json", "csv"})


class SchemaDiffError(Exception):
    """Error raised when schema diff evaluation cannot proceed."""


@dataclass(frozen=True, slots=True)
class BreakingChange:
    """Single breaking schema change record."""

    kind: BreakingChangeKind
    table_key: str | None = None
    column_name: str | None = None
    old_value: str | bool | None = None
    new_value: str | bool | None = None
    artifact_name: str | None = None

    def summary(self) -> str:
        """Return a human-readable summary for the change.

        Returns
        -------
        str
            Summary string describing the breaking change.
        """
        if self.kind == "artifact_removed":
            message = f"artifact removed: {self.artifact_name}"
        elif self.kind == "table_removed":
            message = f"table removed: {self.table_key}"
        elif self.kind == "view_removed":
            message = f"view removed: {self.table_key}"
        elif self.kind in {"table_column_removed", "view_column_removed"}:
            prefix = "table" if self.kind.startswith("table") else "view"
            value = _format_value(value=self.old_value, fallback="unknown")
            message = f"{prefix} column removed: {self.table_key}.{self.column_name} (type={value})"
        elif self.kind in {"table_column_type_changed", "view_column_type_changed"}:
            prefix = "table" if self.kind.startswith("table") else "view"
            old_value = _format_value(value=self.old_value, fallback="unknown")
            new_value = _format_value(value=self.new_value, fallback="unknown")
            message = (
                f"{prefix} column type changed: {self.table_key}.{self.column_name} "
                f"({old_value} -> {new_value})"
            )
        elif self.kind in {"table_column_nullable_changed", "view_column_nullable_changed"}:
            prefix = "table" if self.kind.startswith("table") else "view"
            old_value = _format_value(value=self.old_value, fallback="unknown")
            new_value = _format_value(value=self.new_value, fallback="unknown")
            message = (
                f"{prefix} column nullable changed: {self.table_key}.{self.column_name} "
                f"({old_value} -> {new_value})"
            )
        else:
            message = f"breaking change: {self.kind}"
        return message


@dataclass(frozen=True, slots=True)
class SchemaBreakApproval:
    """Approved breaking schema change entry."""

    kind: BreakingChangeKind
    table_key: str | None = None
    column_name: str | None = None
    old_value: str | bool | None = None
    new_value: str | bool | None = None
    artifact_name: str | None = None
    reason: str | None = None
    owner: str | None = None
    ticket: str | None = None

    def matches(self, change: BreakingChange) -> bool:
        """Return True when this approval matches the change.

        Returns
        -------
        bool
            True when this approval matches the breaking change.
        """
        if change.kind != self.kind:
            return False
        return (
            (self.table_key is None or self.table_key == change.table_key)
            and (self.column_name is None or self.column_name == change.column_name)
            and (self.artifact_name is None or self.artifact_name == change.artifact_name)
            and (self.old_value is None or self.old_value == change.old_value)
            and (self.new_value is None or self.new_value == change.new_value)
        )


@dataclass(frozen=True, slots=True)
class SchemaBreakAllowlist:
    """Allowlist of approved breaking schema changes."""

    version: int
    approvals: tuple[SchemaBreakApproval, ...]

    def approves(self, change: BreakingChange) -> bool:
        """Return True if the allowlist approves the change.

        Returns
        -------
        bool
            True when the breaking change is allowlisted.
        """
        return any(approval.matches(change) for approval in self.approvals)


def load_schema_breaks(*, path: Path) -> SchemaBreakAllowlist:
    """Load approved schema breaks from a YAML allowlist.

    Returns
    -------
    SchemaBreakAllowlist
        Parsed allowlist (empty when missing).
    """
    if not path.is_file():
        return SchemaBreakAllowlist(version=1, approvals=())
    raw_text = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw_text) or {}
    return _parse_schema_breaks(payload, path=path)


def collect_breaking_changes(
    diff_result: ManifestDiffResult,
    *,
    expected_manifest: SchemaManifest,
) -> tuple[BreakingChange, ...]:
    """Collect breaking changes from a manifest diff.

    Returns
    -------
    tuple[BreakingChange, ...]
        Breaking changes derived from the diff results.
    """
    expected_tables = {table.table_key: table for table in expected_manifest.tables}
    expected_views = {view.table_key: view for view in expected_manifest.views}
    changes: list[BreakingChange] = []

    changes.extend(
        [
            BreakingChange(kind="table_removed", table_key=table_key)
            for table_key in diff_result.removed_tables
        ]
    )
    changes.extend(
        [
            BreakingChange(kind="view_removed", table_key=view_key)
            for view_key in diff_result.removed_views
        ]
    )
    changes.extend(
        [
            BreakingChange(kind="artifact_removed", artifact_name=artifact_name)
            for artifact_name in diff_result.removed_artifacts
        ]
    )

    changes.extend(
        _collect_column_breaks(
            diff_result.diffs,
            expected_tables,
            prefix="table",
        )
    )
    changes.extend(
        _collect_column_breaks(
            diff_result.view_diffs,
            expected_views,
            prefix="view",
        )
    )

    return tuple(changes)


def find_unapproved_breaks(
    diff_result: ManifestDiffResult,
    *,
    expected_manifest: SchemaManifest,
    allowlist: SchemaBreakAllowlist,
) -> tuple[BreakingChange, ...]:
    """Return the set of breaking changes not covered by approvals.

    Returns
    -------
    tuple[BreakingChange, ...]
        Breaking changes that lack allowlist approval.
    """
    changes = collect_breaking_changes(diff_result, expected_manifest=expected_manifest)
    return tuple(change for change in changes if not allowlist.approves(change))


def main() -> int:
    """Run schema diff gating checks.

    Returns
    -------
    int
        Exit status code (0 for success, non-zero for failures).
    """
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    expected_path = _resolve_path(repo_root, args.expected)
    allowlist_path = _resolve_path(repo_root, args.allowlist)
    actual_path = _resolve_path(repo_root, args.actual) if args.actual else None

    try:
        expected_manifest = _load_expected_manifest(expected_path)
        actual_manifest = _load_actual_manifest(repo_root, actual_path)
    except SchemaDiffError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    diff_result = compute_manifest_diffs(expected_manifest, actual_manifest)
    return _evaluate_diff(
        diff_result,
        expected_manifest=expected_manifest,
        allowlist_path=allowlist_path,
        fail_on_any=args.fail_on_any,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare schema manifests for breaking changes.")
    parser.add_argument(
        "--expected",
        type=Path,
        default=Path("build/serving/artifacts/schema_manifest.json"),
        help="Baseline schema manifest path.",
    )
    parser.add_argument(
        "--actual",
        type=Path,
        default=None,
        help="Optional actual manifest path (defaults to compiled).",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path("config/schema_breaks.yaml"),
        help="Allowlist YAML for approved breaking changes.",
    )
    parser.add_argument(
        "--fail-on-any",
        action="store_true",
        help="Fail when any schema change is detected.",
    )
    return parser.parse_args()


def _load_expected_manifest(path: Path) -> SchemaManifest:
    if not path.is_file():
        msg = f"Expected schema manifest missing: {path}"
        raise SchemaDiffError(msg)
    try:
        return _load_manifest(path)
    except (OSError, TypeError, ValueError) as exc:
        msg = f"Failed to read expected manifest: {exc}"
        raise SchemaDiffError(msg) from exc


def _load_actual_manifest(repo_root: Path, actual_path: Path | None) -> SchemaManifest:
    if actual_path is not None:
        if not actual_path.is_file():
            msg = f"Actual schema manifest missing: {actual_path}"
            raise SchemaDiffError(msg)
        try:
            return _load_manifest(actual_path)
        except (OSError, TypeError, ValueError) as exc:
            msg = f"Failed to read actual manifest: {exc}"
            raise SchemaDiffError(msg) from exc
    try:
        return _compile_current_manifest(repo_root)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        msg = f"Failed to load actual manifest: {exc}"
        raise SchemaDiffError(msg) from exc


def _evaluate_diff(
    diff_result: ManifestDiffResult,
    *,
    expected_manifest: SchemaManifest,
    allowlist_path: Path,
    fail_on_any: bool,
) -> int:
    if not diff_result.has_any_changes:
        sys.stdout.write("Schema diff clean: no changes detected.\n")
        return 0

    sys.stdout.write(diff_result.format_summary())
    sys.stdout.write("\n")

    allowlist = load_schema_breaks(path=allowlist_path)
    unapproved = find_unapproved_breaks(
        diff_result,
        expected_manifest=expected_manifest,
        allowlist=allowlist,
    )
    if unapproved:
        sys.stdout.write("Unapproved breaking changes:\n")
        for change in unapproved:
            sys.stdout.write(f"  - {change.summary()}\n")
        return 1

    if diff_result.has_breaking_changes:
        sys.stdout.write("Breaking changes approved by allowlist.\n")
    if fail_on_any and diff_result.has_any_changes:
        return 1
    return 0


def _resolve_path(repo_root: Path, value: Path) -> Path:
    if value.is_absolute():
        return value
    return repo_root / value


def _load_manifest(path: Path) -> SchemaManifest:
    payload = read_manifest_json(path)
    if not isinstance(payload, Mapping):
        msg = f"Schema manifest must be a mapping: {path}"
        raise TypeError(msg)
    return _parse_manifest_payload(payload, path=path)


def _parse_manifest_payload(payload: Mapping[str, object], *, path: Path) -> SchemaManifest:
    version = _optional_str(payload.get("version")) or "v2"
    if version != "v2":
        msg = f"Unsupported schema manifest version {version!r} in {path}"
        raise ValueError(msg)

    tables = _parse_schema_entries(payload.get("tables"), path=path, label="tables")
    views = _parse_schema_entries(payload.get("views"), path=path, label="views")
    artifacts = _parse_artifacts(payload.get("artifacts"), path=path)
    return SchemaManifest(
        version=version,
        tables=tuple(tables),
        views=tuple(views),
        artifacts=tuple(artifacts),
    )


def _parse_schema_entries(
    raw: object,
    *,
    path: Path,
    label: str,
) -> list[TableSchema]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        msg = f"Schema manifest {label} must be a list in {path}"
        raise TypeError(msg)
    tables: list[TableSchema] = []
    for item in raw:
        if not isinstance(item, Mapping):
            msg = f"Schema manifest {label} entries must be mappings in {path}"
            raise TypeError(msg)
        tables.append(table_schema_from_json_obj(item))
    return tables


def _parse_artifacts(raw: object, *, path: Path) -> list[ExportArtifact]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        msg = f"Schema manifest artifacts must be a list in {path}"
        raise TypeError(msg)
    artifacts: list[ExportArtifact] = []
    for item in raw:
        if not isinstance(item, Mapping):
            msg = f"Schema manifest artifacts entries must be mappings in {path}"
            raise TypeError(msg)
        artifacts.append(_parse_artifact_entry(item, path=path))
    return artifacts


def _parse_artifact_entry(entry: Mapping[str, object], *, path: Path) -> ExportArtifact:
    kind = _require_str(entry.get("kind"), label="artifact.kind", path=path)
    if kind not in _EXPORT_ARTIFACT_KINDS:
        msg = f"Unsupported artifact kind {kind!r} in {path}"
        raise ValueError(msg)
    filename = _require_str(entry.get("filename"), label="artifact.filename", path=path)
    table_key = _optional_str(entry.get("table_key"))
    description = _optional_str(entry.get("description"))
    return ExportArtifact(
        kind=cast("ExportArtifactKind", kind),
        filename=filename,
        table_key=table_key,
        description=description,
    )


def _require_str(value: object, *, label: str, path: Path) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    msg = f"Schema manifest {label} must be a non-empty string in {path}"
    raise TypeError(msg)


def _compile_current_manifest(repo_root: Path) -> SchemaManifest:
    runtime = resolve_from_params(
        {"project_root": repo_root, "repo_root": repo_root},
        allow_fallback=True,
    )
    gateway = open_inference_gateway(schema_provider=MappingSchemaProvider({}))
    try:
        runtime_bundle = compose_cli_runtime_bundle(runtime=runtime, gateway=gateway)
        schema_index = runtime_bundle.schema_index
        if schema_index is None:
            msg = "Runtime schema_index is required to compile schema manifests"
            raise RuntimeError(msg)
        request = SchemaManifestRequest(
            all_targets=True,
            stable=True,
            version="v2",
            include_views=True,
            include_artifacts=True,
            include_provenance=True,
            infer_native=False,
            batch_infer_native=False,
        )
        return compile_schema_manifest(
            provider=get_schema_provider(),
            context=SchemaManifestContext(
                catalog=runtime_bundle.catalog,
                schema_index=schema_index,
                tag_query=runtime_bundle.tag_query,
            ),
            request=request,
        )
    finally:
        gateway.close()


def _collect_column_breaks(
    diffs: tuple[SchemaDiff, ...],
    expected: Mapping[str, TableSchema],
    *,
    prefix: Literal["table", "view"],
) -> list[BreakingChange]:
    changes: list[BreakingChange] = []
    removed_kind = f"{prefix}_column_removed"
    type_changed_kind = f"{prefix}_column_type_changed"
    nullable_kind = f"{prefix}_column_nullable_changed"

    for diff in diffs:
        column_map = _column_map(expected.get(diff.table_key))
        for column_name in diff.removed_columns:
            column = column_map.get(column_name)
            changes.append(
                BreakingChange(
                    kind=cast("BreakingChangeKind", removed_kind),
                    table_key=diff.table_key,
                    column_name=column_name,
                    old_value=column.type if column else None,
                )
            )
        for column_name, old_type, new_type in diff.type_changes:
            changes.append(
                BreakingChange(
                    kind=cast("BreakingChangeKind", type_changed_kind),
                    table_key=diff.table_key,
                    column_name=column_name,
                    old_value=old_type,
                    new_value=new_type,
                )
            )
        for column_name, old_nullable, new_nullable in diff.nullable_changes:
            if old_nullable and not new_nullable:
                changes.append(
                    BreakingChange(
                        kind=cast("BreakingChangeKind", nullable_kind),
                        table_key=diff.table_key,
                        column_name=column_name,
                        old_value=old_nullable,
                        new_value=new_nullable,
                    )
                )
    return changes


def _column_map(schema: TableSchema | None) -> dict[str, Column]:
    if schema is None:
        return {}
    return {column.name: column for column in schema.columns}


def _parse_schema_breaks(
    payload: object,
    *,
    path: Path,
) -> SchemaBreakAllowlist:
    if not isinstance(payload, Mapping):
        msg = f"schema_breaks allowlist must be a mapping: {path}"
        raise TypeError(msg)
    version = payload.get("version", 1)
    if not isinstance(version, int):
        msg = f"schema_breaks version must be an int: {path}"
        raise TypeError(msg)
    entries_raw = payload.get("schema_breaks", [])
    approvals = _parse_break_entries(entries_raw, path=path)
    return SchemaBreakAllowlist(version=version, approvals=tuple(approvals))


def _parse_break_entries(entries_raw: object, *, path: Path) -> list[SchemaBreakApproval]:
    if entries_raw is None:
        return []
    if not isinstance(entries_raw, list):
        msg = f"schema_breaks must be a list in {path}"
        raise TypeError(msg)
    return [_parse_break_entry(entry, path=path) for entry in entries_raw]


def _parse_break_entry(entry: object, *, path: Path) -> SchemaBreakApproval:
    if not isinstance(entry, Mapping):
        msg = f"schema_breaks entries must be mappings in {path}"
        raise TypeError(msg)
    kind_raw = entry.get("kind")
    if not isinstance(kind_raw, str) or kind_raw not in _BREAKING_KINDS:
        msg = f"schema_breaks.kind must be one of {', '.join(_BREAKING_KINDS)} in {path}"
        raise ValueError(msg)
    kind = cast("BreakingChangeKind", kind_raw)
    table_key = _optional_str(entry.get("table_key"))
    column_name = _optional_str(entry.get("column_name"))
    artifact_name = _optional_str(entry.get("artifact_name"))
    old_value = _parse_optional_value(entry.get("old_value"), path=path, label="old_value")
    new_value = _parse_optional_value(entry.get("new_value"), path=path, label="new_value")
    _validate_break_entry(
        kind=kind,
        table_key=table_key,
        column_name=column_name,
        artifact_name=artifact_name,
        path=path,
    )
    reason = _optional_str(entry.get("reason"))
    owner = _optional_str(entry.get("owner"))
    ticket = _optional_str(entry.get("ticket"))
    return SchemaBreakApproval(
        kind=kind,
        table_key=table_key,
        column_name=column_name,
        artifact_name=artifact_name,
        old_value=old_value,
        new_value=new_value,
        reason=reason,
        owner=owner,
        ticket=ticket,
    )


def _validate_break_entry(
    *,
    kind: BreakingChangeKind,
    table_key: str | None,
    column_name: str | None,
    artifact_name: str | None,
    path: Path,
) -> None:
    if kind == "artifact_removed":
        if artifact_name is None:
            msg = f"schema_breaks.artifact_removed requires artifact_name in {path}"
            raise ValueError(msg)
        return
    if kind in {"table_removed", "view_removed"}:
        if table_key is None:
            msg = f"schema_breaks.{kind} requires table_key in {path}"
            raise ValueError(msg)
        return
    if table_key is None or column_name is None:
        msg = f"schema_breaks.{kind} requires table_key and column_name in {path}"
        raise ValueError(msg)


def _parse_optional_value(
    value: object,
    *,
    path: Path,
    label: str,
) -> str | bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"true", "false"}:
            return stripped.lower() == "true"
        if stripped:
            return stripped
        return None
    msg = f"schema_breaks.{label} must be a string or bool in {path}"
    raise TypeError(msg)


def _optional_str(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _format_value(*, value: str | bool | None, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return str(value).lower()
    return value


if __name__ == "__main__":
    sys.exit(main())
