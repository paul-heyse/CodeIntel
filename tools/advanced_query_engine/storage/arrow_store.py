"""PyArrow persistence for advanced query engine results."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from tools.advanced_query_engine.contracts import JSONValue, QueryRequest, QueryResponse

ENGINE_VERSION = "1"
SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class PersistResult:
    """Outcome for persisting a response payload."""

    path: Path
    rows: int
    schema_name: str

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a JSON-serializable representation.

        Returns
        -------
        dict[str, JSONValue]
            Serialized persistence result.
        """
        return {
            "path": str(self.path),
            "rows": self.rows,
            "schema": self.schema_name,
        }


def _encode_metadata(metadata: dict[str, str]) -> dict[bytes, bytes]:
    """Encode string metadata to UTF-8 byte pairs.

    Returns
    -------
    dict[bytes, bytes]
        UTF-8 encoded metadata payload.
    """
    return {key.encode("utf-8"): value.encode("utf-8") for key, value in metadata.items()}


def _decode_metadata(metadata: dict[bytes, bytes] | None) -> dict[str, str]:
    """Decode UTF-8 metadata into strings.

    Returns
    -------
    dict[str, str]
        Decoded metadata payload.
    """
    if not metadata:
        return {}
    return {key.decode("utf-8"): value.decode("utf-8") for key, value in metadata.items()}


def _base_metadata() -> dict[str, str]:
    """Return base metadata for persisted datasets.

    Returns
    -------
    dict[str, str]
        Base metadata for persisted datasets.
    """
    return {
        "engine_version": ENGINE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _schema_metadata(schema_name: str) -> dict[str, str]:
    """Return schema metadata entries for persisted datasets.

    Parameters
    ----------
    schema_name:
        Schema identifier for the dataset.

    Returns
    -------
    dict[str, str]
        Schema metadata payload.
    """
    return {"schema_name": schema_name, "schema_version": SCHEMA_VERSION}


def match_record_schema(metadata: dict[str, str] | None = None) -> pa.Schema:
    """Return the PyArrow schema for match records.

    Parameters
    ----------
    metadata:
        Optional schema metadata entries.

    Returns
    -------
    pa.Schema
        PyArrow schema for match records.
    """
    merged = _base_metadata()
    merged.update(_schema_metadata("match_records"))
    if metadata:
        merged.update(metadata)
    return pa.schema(
        [
            ("engine", pa.string()),
            ("path", pa.dictionary(pa.int32(), pa.string())),
            ("start_byte", pa.int64()),
            ("end_byte", pa.int64()),
            ("start_line", pa.int32()),
            ("end_line", pa.int32()),
            ("rule_id", pa.dictionary(pa.int32(), pa.string())),
            ("pattern_id", pa.dictionary(pa.int32(), pa.string())),
            ("snippet", pa.string()),
            ("captures", pa.map_(pa.string(), pa.list_(pa.string()))),
        ],
        metadata=_encode_metadata(merged),
    )


def wiring_edge_schema(metadata: dict[str, str] | None = None) -> pa.Schema:
    """Return the PyArrow schema for wiring edges.

    Parameters
    ----------
    metadata:
        Optional schema metadata entries.

    Returns
    -------
    pa.Schema
        PyArrow schema for wiring edges.
    """
    merged = _base_metadata()
    merged.update(_schema_metadata("wiring_edges"))
    if metadata:
        merged.update(metadata)
    return pa.schema(
        [
            ("edge_id", pa.string()),
            ("pack_id", pa.dictionary(pa.int32(), pa.string())),
            ("framework", pa.dictionary(pa.int32(), pa.string())),
            ("entry_kind", pa.dictionary(pa.int32(), pa.string())),
            ("entry_key", pa.string()),
            ("path", pa.dictionary(pa.int32(), pa.string())),
            ("start_byte", pa.int64()),
            ("end_byte", pa.int64()),
            ("rule_id", pa.dictionary(pa.int32(), pa.string())),
            ("target_name", pa.string()),
            ("target_qname", pa.string()),
            ("evidence", pa.string()),
            ("captures", pa.map_(pa.string(), pa.list_(pa.string()))),
        ],
        metadata=_encode_metadata(merged),
    )


def write_match_records(
    path: Path,
    records: Iterable[dict[str, JSONValue]],
    *,
    metadata: dict[str, str] | None = None,
    partition_by: list[str] | None = None,
) -> PersistResult:
    """Persist match records to Parquet.

    Parameters
    ----------
    path:
        Output path for the dataset.
    records:
        Match record payloads.
    metadata:
        Optional metadata to attach.
    partition_by:
        Optional partition columns.

    Returns
    -------
    PersistResult
        Persistence result metadata.
    """
    rows = [_match_record_row(record) for record in records]
    schema = match_record_schema(metadata)
    table = pa.Table.from_pylist(rows, schema=schema)
    _write_table(path, table, partition_by=partition_by)
    return PersistResult(path=path, rows=len(rows), schema_name="match_records")


def write_wiring_edges(
    path: Path,
    edges: Iterable[dict[str, JSONValue]],
    *,
    metadata: dict[str, str] | None = None,
    partition_by: list[str] | None = None,
) -> PersistResult:
    """Persist wiring edges to Parquet.

    Parameters
    ----------
    path:
        Output path for the dataset.
    edges:
        Wiring edge payloads.
    metadata:
        Optional metadata to attach.
    partition_by:
        Optional partition columns.

    Returns
    -------
    PersistResult
        Persistence result metadata.
    """
    rows = [_wiring_edge_row(edge) for edge in edges]
    schema = wiring_edge_schema(metadata)
    table = pa.Table.from_pylist(rows, schema=schema)
    _write_table(path, table, partition_by=partition_by)
    return PersistResult(path=path, rows=len(rows), schema_name="wiring_edges")


def persist_query_response(
    *,
    request: QueryRequest,
    response: QueryResponse,
    output_root: Path,
    partition_by: list[str] | None = None,
) -> PersistResult | None:
    """Persist a query response to Parquet when supported.

    Parameters
    ----------
    request:
        Query request metadata.
    response:
        Query response payload.
    output_root:
        Root output directory.
    partition_by:
        Optional partition columns.

    Returns
    -------
    PersistResult | None
        Persistence result metadata when a dataset is written.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = {"request_type": request.type}
    if request.type == "pattern.scan":
        return write_match_records(
            output_root / "match_records.parquet",
            response.primary,
            metadata=metadata,
            partition_by=partition_by,
        )
    if request.type == "wiring.map":
        return write_wiring_edges(
            output_root / "wiring_edges.parquet",
            response.primary,
            metadata=metadata,
            partition_by=partition_by,
        )
    return None


def read_parquet_schema(path: Path) -> pa.Schema:
    """Read the schema for a Parquet file or dataset.

    Parameters
    ----------
    path:
        Path to a parquet file or dataset directory.

    Returns
    -------
    pa.Schema
        Schema for the persisted dataset.

    Raises
    ------
    FileNotFoundError
        If the parquet file or dataset does not exist.
    """
    if not path.exists():
        msg = f"Persisted dataset not found: {path}"
        raise FileNotFoundError(msg)
    if path.is_dir():
        dataset = ds.dataset(str(path), format="parquet")
        return dataset.schema
    return pq.read_schema(path)


def schema_compatibility_issues(path: Path, expected: pa.Schema) -> list[str]:
    """Return compatibility issues between persisted and expected schemas.

    Parameters
    ----------
    path:
        Path to the persisted dataset.
    expected:
        Expected schema definition.

    Returns
    -------
    list[str]
        Human-readable compatibility issues.
    """
    actual = read_parquet_schema(path)
    expected_meta = _decode_metadata(expected.metadata)
    actual_meta = _decode_metadata(actual.metadata)
    issues: list[str] = []
    expected_name = expected_meta.get("schema_name")
    actual_name = actual_meta.get("schema_name")
    if expected_name and not actual_name:
        issues.append("Schema name metadata is missing.")
    elif expected_name and actual_name and expected_name != actual_name:
        issues.append(f"Schema name mismatch: expected {expected_name}, got {actual_name}.")
    expected_version = expected_meta.get("schema_version")
    actual_version = actual_meta.get("schema_version")
    if expected_version and not actual_version:
        issues.append("Schema version metadata is missing.")
    elif expected_version and actual_version and expected_version != actual_version:
        issues.append(
            f"Schema version mismatch: expected {expected_version}, got {actual_version}."
        )
    for field in expected:
        try:
            actual_field = actual.field_by_name(field.name)
        except KeyError:
            issues.append(f"Missing field: {field.name}.")
            continue
        if not field.type.equals(actual_field.type):
            issues.append(
                "Field type mismatch for "
                f"{field.name}: expected {field.type}, got {actual_field.type}."
            )
    return issues


def _write_table(path: Path, table: pa.Table, *, partition_by: list[str] | None) -> None:
    """Write a table to Parquet with optional partitioning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if partition_by:
        ds.write_dataset(
            table,
            base_dir=str(path),
            format="parquet",
            partitioning=partition_by,
            existing_data_behavior="overwrite_or_ignore",
        )
        return
    pq.write_table(table, path)


def _match_record_row(record: dict[str, JSONValue]) -> dict[str, JSONValue]:
    """Project a match record payload into a flat row.

    Returns
    -------
    dict[str, JSONValue]
        Flattened match record row.
    """
    span = record.get("span")
    start_line, end_line = _span_line_bounds(span)
    return {
        "engine": record.get("engine"),
        "path": record.get("path"),
        "start_byte": _span_value(span, "start_byte"),
        "end_byte": _span_value(span, "end_byte"),
        "start_line": start_line,
        "end_line": end_line,
        "rule_id": record.get("rule_id"),
        "pattern_id": record.get("pattern_id"),
        "snippet": _snippet_text(record.get("snippet")),
        "captures": _capture_texts(record.get("captures")),
    }


def _wiring_edge_row(edge: dict[str, JSONValue]) -> dict[str, JSONValue]:
    """Project a wiring edge payload into a flat row.

    Returns
    -------
    dict[str, JSONValue]
        Flattened wiring edge row.
    """
    hook_span = edge.get("hook_span")
    target = edge.get("target")
    match = edge.get("match")
    return {
        "edge_id": edge.get("edge_id"),
        "pack_id": edge.get("pack_id"),
        "framework": edge.get("framework"),
        "entry_kind": edge.get("entry_kind"),
        "entry_key": edge.get("entry_key"),
        "path": _span_value(hook_span, "path"),
        "start_byte": _span_value(hook_span, "start_byte"),
        "end_byte": _span_value(hook_span, "end_byte"),
        "rule_id": _match_field(match, "rule_id"),
        "target_name": _target_field(target, "name"),
        "target_qname": _target_field(target, "qname"),
        "evidence": _snippet_text(edge.get("evidence")),
        "captures": _match_captures(match),
    }


def _span_value(span: JSONValue | None, key: str) -> JSONValue | None:
    """Return a span field value when available.

    Returns
    -------
    JSONValue | None
        Span field value when present.
    """
    if isinstance(span, dict):
        return span.get(key)
    return None


def _span_line_bounds(span: JSONValue | None) -> tuple[int | None, int | None]:
    """Return line bounds for a span payload.

    Returns
    -------
    tuple[int | None, int | None]
        Start/end line values when present.
    """
    if not isinstance(span, dict):
        return None, None
    start_line = span.get("start_line")
    end_line = span.get("end_line")
    return _int_or_none(start_line), _int_or_none(end_line)


def _int_or_none(value: JSONValue | None) -> int | None:
    """Return an int value when provided.

    Returns
    -------
    int | None
        Integer value when present.
    """
    return int(value) if isinstance(value, int) else None


def _snippet_text(snippet: JSONValue | None) -> str | None:
    """Extract snippet text from a snippet payload.

    Returns
    -------
    str | None
        Snippet text when present.
    """
    if isinstance(snippet, dict):
        text = snippet.get("text")
        return text if isinstance(text, str) else None
    return None


def _capture_texts(captures: JSONValue | None) -> dict[str, list[str]] | None:
    """Extract capture texts from a captures payload.

    Returns
    -------
    dict[str, list[str]] | None
        Capture texts by name when present.
    """
    if not isinstance(captures, dict):
        return None
    output: dict[str, list[str]] = {}
    for key, value in captures.items():
        if not isinstance(key, str) or not isinstance(value, list):
            continue
        texts = [_snippet_text(item) for item in value]
        output[key] = [text for text in texts if text is not None]
    return output or None


def _match_field(match: JSONValue | None, key: str) -> JSONValue | None:
    """Return a match payload field value.

    Returns
    -------
    JSONValue | None
        Match field value when present.
    """
    if isinstance(match, dict):
        return match.get(key)
    return None


def _match_captures(match: JSONValue | None) -> dict[str, list[str]] | None:
    """Return capture texts from a match payload.

    Returns
    -------
    dict[str, list[str]] | None
        Capture texts by name when present.
    """
    if not isinstance(match, dict):
        return None
    return _capture_texts(match.get("captures"))


def _target_field(target: JSONValue | None, key: str) -> str | None:
    """Return a target field string value.

    Returns
    -------
    str | None
        Target field value when present.
    """
    if isinstance(target, dict):
        value = target.get(key)
        return value if isinstance(value, str) else None
    return None


__all__ = [
    "PersistResult",
    "match_record_schema",
    "persist_query_response",
    "read_parquet_schema",
    "schema_compatibility_issues",
    "wiring_edge_schema",
    "write_match_records",
    "write_wiring_edges",
]
