"""
Validate exported JSONL/Parquet datasets against JSON Schemas.

Usage:
    python -m codeintel.docs_export.validate_exports --schema call_graph_edges path1.jsonl path2.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import jsonschema
import pyarrow.parquet as pq

from codeintel.services.errors import log_problem, problem

DEFAULT_SCHEMA_ROOT = Path(__file__).resolve().parent.parent / "config" / "schemas" / "export"


def _load_schema(schema_name: str, root: Path) -> tuple[dict[str, Any], jsonschema.RefResolver]:
    path = root / f"{schema_name}.json"
    if not path.is_file():
        message = f"Schema not found: {path}"
        pd = problem(code="export.schema_missing", title="Schema missing", detail=message)
        log_problem(logger=_stderr_logger(), detail=pd)
        raise FileNotFoundError(message)

    store: dict[str, dict[str, Any]] = {}
    for schema_path in root.glob("*.json"):
        with schema_path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        if "$id" in doc and isinstance(doc["$id"], str):
            store[doc["$id"]] = doc
        store[schema_path.as_uri()] = doc

    with path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    resolver = jsonschema.RefResolver(
        base_uri=root.as_uri().rstrip("/") + "/",
        referrer=schema,
        store=store,
    )
    return schema, resolver


def _stderr_logger() -> logging.Logger:
    """
    Return a stderr logger configured for error output.

    Returns
    -------
    logging.Logger
        Logger that writes to stderr.
    """
    logger = logging.getLogger("codeintel.validate_exports")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        logger.addHandler(handler)
    logger.setLevel(logging.ERROR)
    return logger


def _validate_records(
    records: list[dict[str, Any]], validator: jsonschema.Draft202012Validator
) -> list[str]:
    return [
        f"row={idx}: {error.message}"
        for idx, record in enumerate(records)
        for error in validator.iter_errors(record)
    ]


def _validate_jsonl(path: Path, validator: jsonschema.Draft202012Validator) -> list[str]:
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - user error
                errors.append(f"row={idx}: invalid JSON ({exc})")
                continue
            errors.extend(_validate_records([record], validator))
    return errors


def _validate_parquet(path: Path, validator: jsonschema.Draft202012Validator) -> list[str]:
    table = pq.read_table(path)
    records = table.to_pylist()
    return _validate_records(records, validator)


def validate_files(
    schema_name: str, paths: list[Path], *, schema_root: Path = DEFAULT_SCHEMA_ROOT
) -> int:
    """
    Validate files against the named schema.

    Returns
    -------
    int
        0 on success, non-zero on validation failure.
    """
    logger = _stderr_logger()
    try:
        schema, resolver = _load_schema(schema_name, schema_root)
    except FileNotFoundError:
        return 1

    validator = jsonschema.Draft202012Validator(schema, resolver=resolver)
    all_errors: list[str] = []
    for path in paths:
        if not path.exists():
            message = f"File not found: {path}"
            pd = problem(code="export.file_missing", title="Export file missing", detail=message)
            log_problem(logger, pd)
            all_errors.append(message)
            continue
        if path.suffix.lower() == ".jsonl":
            errors = _validate_jsonl(path, validator)
        else:
            errors = _validate_parquet(path, validator)
        all_errors.extend([f"{path}: {err}" for err in errors])

    if all_errors:
        pd = problem(
            code="export.validation_failed",
            title="Export validation failed",
            detail="; ".join(all_errors),
            extras={"errors": all_errors},
        )
        log_problem(logger, pd)
        return 1
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate exported datasets against JSON Schemas.")
    parser.add_argument("--schema", required=True, help="Schema name (without .json)")
    parser.add_argument(
        "--schema-root",
        type=Path,
        default=DEFAULT_SCHEMA_ROOT,
        help="Root directory containing export schemas",
    )
    parser.add_argument("paths", nargs="+", type=Path, help="JSONL or Parquet files to validate")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint for export validation.

    Returns
    -------
    int
        Exit code from validation (0 on success).
    """
    args = _parse_args(list(argv) if argv is not None else None)
    return validate_files(args.schema, args.paths, schema_root=args.schema_root)


if __name__ == "__main__":
    sys.exit(main())
