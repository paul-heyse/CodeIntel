"""Dataset scaffolding helpers to seed new contract assets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ScaffoldOptions:
    """Options controlling dataset scaffolding outputs."""

    name: str
    table_key: str
    owner: str | None
    freshness_sla: str | None
    retention_policy: str | None
    schema_version: str
    stable_id: str
    validation_profile: Literal["strict", "lenient"]
    jsonl_filename: str | None
    parquet_filename: str | None
    schema_id: str
    output_dir: Path
    is_view: bool = False
    overwrite: bool = False
    dry_run: bool = False
    emit_bootstrap_snippet: bool = False


@dataclass(frozen=True)
class ScaffoldResult:
    """Artifact paths created by a scaffold run."""

    typed_dict: Path
    row_binding: Path
    json_schema: Path
    metadata: Path
    bootstrap_snippet: Path | None


def _typed_dict_content(opts: ScaffoldOptions) -> str:
    class_name = "".join(part.capitalize() for part in opts.name.split("_")) + "Row"
    return "\n".join(
        [
            '"""TypedDict row model for the new dataset scaffold."""',
            "",
            "from __future__ import annotations",
            "",
            "from typing import TypedDict",
            "",
            "",
            f"class {class_name}(TypedDict):",
            f'    """Row model for the `{opts.name}` dataset."""',
            "    # TODO: add fields",
            "    placeholder: str",
            "",
            "",
            f"def to_tuple(row: {class_name}) -> tuple[object, ...]:",
            """    \"\"\"Serialize the TypedDict into a tuple for DuckDB inserts.\"\"\"""",
            "    # TODO: update ordering to match TableSchema",
            '    return (row["placeholder"],)',
            "",
        ]
    )


def _row_binding_content(opts: ScaffoldOptions) -> str:
    class_name = "".join(part.capitalize() for part in opts.name.split("_")) + "Row"
    return "\n".join(
        [
            "# Row binding snippet for metadata bootstrap.",
            "# Add this to ROW_BINDINGS_BY_TABLE_KEY or equivalent registry.",
            f'"{opts.table_key}": _row_binding(',
            f"    row_type=row_models.{class_name},",
            "    to_tuple=row_models.to_tuple,",
            "),",
            "",
        ]
    )


def _metadata_json(opts: ScaffoldOptions) -> dict[str, object]:
    return {
        "name": opts.name,
        "table_key": opts.table_key,
        "owner": opts.owner,
        "freshness_sla": opts.freshness_sla,
        "retention_policy": opts.retention_policy,
        "schema_version": opts.schema_version,
        "stable_id": opts.stable_id,
        "validation_profile": opts.validation_profile,
        "json_schema_id": opts.schema_id,
        "jsonl_filename": opts.jsonl_filename,
        "parquet_filename": opts.parquet_filename,
        "is_view": opts.is_view,
    }


def _bootstrap_snippet_content(opts: ScaffoldOptions) -> str:
    metadata = json.dumps(_metadata_json(opts), indent=2)
    return "\n".join(
        [
            "# Paste into metadata bootstrap when registering the dataset.",
            "# metadata entry:",
            metadata,
            "",
            "# row binding (if table with inserts):",
            _row_binding_content(opts),
        ]
    )


def _ensure_paths_available(paths: list[Path], *, overwrite: bool) -> None:
    for path in paths:
        if path.exists() and not overwrite:
            message = f"Refusing to overwrite existing file: {path}"
            raise FileExistsError(message)


def scaffold_dataset(opts: ScaffoldOptions) -> ScaffoldResult:
    """
    Create a skeleton dataset contract in the target output directory.

    Notes
    -----
    The scaffold is non-destructive and writes new files only. Integration into
    the codebase (row binding registry, metadata bootstrap) is left to the caller.

    Returns
    -------
    ScaffoldResult
        Paths to the generated artifacts.
    """
    base = opts.output_dir
    typed_dict_path = base / f"{opts.name}_rows.py"
    row_binding_path = base / f"{opts.name}_binding.txt"
    schema_path = base / f"{opts.schema_id}.json"
    metadata_path = base / f"{opts.name}_metadata.json"
    bootstrap_patch_path = base / f"{opts.name}_bootstrap_patch.txt"
    targets = [typed_dict_path, row_binding_path, schema_path, metadata_path]
    if opts.emit_bootstrap_snippet:
        targets.append(bootstrap_patch_path)

    _ensure_paths_available(targets, overwrite=opts.overwrite)
    if opts.dry_run:
        return ScaffoldResult(
            typed_dict=typed_dict_path,
            row_binding=row_binding_path,
            json_schema=schema_path,
            metadata=metadata_path,
            bootstrap_snippet=bootstrap_patch_path if opts.emit_bootstrap_snippet else None,
        )

    base.mkdir(parents=True, exist_ok=True)
    typed_dict_path.write_text(_typed_dict_content(opts), encoding="utf-8")
    row_binding_path.write_text(_row_binding_content(opts), encoding="utf-8")
    schema_payload = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": opts.schema_id,
        "title": opts.name,
        "type": "object",
        "properties": {
            "placeholder": {"type": "string"},
        },
        "required": ["placeholder"],
    }
    schema_path.write_text(json.dumps(schema_payload, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(_metadata_json(opts), indent=2), encoding="utf-8")
    if opts.emit_bootstrap_snippet:
        bootstrap_patch_path.write_text(_bootstrap_snippet_content(opts), encoding="utf-8")

    return ScaffoldResult(
        typed_dict=typed_dict_path,
        row_binding=row_binding_path,
        json_schema=schema_path,
        metadata=metadata_path,
        bootstrap_snippet=bootstrap_patch_path if opts.emit_bootstrap_snippet else None,
    )
