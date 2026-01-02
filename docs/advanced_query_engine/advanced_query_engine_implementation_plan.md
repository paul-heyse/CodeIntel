# Advanced Query Engine — Implementation Plan (Library-Backed Enhancements)

## Purpose
Build a staged rollout plan for the advanced query engine using the approved libraries:
- PyArrow (schema-aware persistence + metadata)
- Polars (streaming analytics + profiling)
- msgspec (typed contracts + JSON schema)
- orjson (fast JSON encoding)
- intervaltree (span indexing)
- pandera (DataFrame validation)

This plan focuses on: correctness, performance, budget enforcement, and debuggability.

---

## Architecture Overview

### Data flow (high level)
1) **Input contracts** (msgspec): typed requests/options parsed and validated.
2) **Search execution** (rpygrep + ast-grep + tree-sitter): existing engine flow.
3) **Span indexing** (intervaltree): fast enclosure/overlap lookups.
4) **Results assembly** (structured records): typed or dict outputs.
5) **Persistence** (pyarrow): parquet tables with schema metadata.
6) **Analytics** (polars): lazy scans, streaming batches, profiles.
7) **Validation** (pandera): enforce schema contracts on stored/streamed frames.
8) **Serialization** (orjson): fast JSON for CLI and API responses.

### Data contracts
- Query inputs and outputs use msgspec.Struct to avoid ambiguous payloads.
- Pack validation uses msgspec + schema-derived constraints.
- Parquet schemas embed version and pack metadata in schema metadata.

---

## Phase 0 — Contract & Schema Foundation (msgspec)

### Goal
Typed request/response/packs with strict validation and JSON schema export.

### Core changes
- Define msgspec.Struct models for QueryRequest/QueryResponse and wiring pack schemas.
- Enforce `forbid_unknown_fields` for pack configs and request options.
- Export JSON Schema for documentation and validation tooling.

### Code pattern
```python
import msgspec
from typing import Annotated, Literal

class QueryBudget(msgspec.Struct, frozen=True):
    max_files: int = 300
    max_matches: int = 2000
    max_seconds: float | None = None

class QueryRequest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    type: Literal[
        "symbol.resolve",
        "refs.find",
        "callgraph.slice",
        "pattern.scan",
        "contract.lookup",
        "wiring.map",
        "precedent.search",
        "impact.slice",
    ]
    text: str
    repo_root: str
    scope_paths: list[str] | None = None
    budget: QueryBudget | None = None
    options: dict[str, object] | None = None

schema = msgspec.json.schema(QueryRequest)
```

### Checklist
- [ ] Implement msgspec structs for request/response and wiring packs
- [ ] Add JSON schema export for request and pack formats
- [ ] Replace ad-hoc dict parsing with msgspec.json.decode(type=...)
- [ ] Add tests for unknown fields and invalid enum values

---

## Phase 1 — Fast JSON Serialization (orjson)

### Goal
Speed up CLI/API responses with deterministic JSON bytes.

### Core changes
- Use `orjson.dumps` for CLI output and optional response formatting.
- Apply `OPT_SORT_KEYS` for deterministic output and `OPT_APPEND_NEWLINE` for CLI use.

### Code pattern
```python
import orjson

payload = response.to_dict()
encoded = orjson.dumps(
    payload,
    option=orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE,
)
stdout.write(encoded.decode("utf-8"))
```

### Checklist
- [ ] Update CLI serialization to orjson
- [ ] Keep deterministic output for snapshots and diffs
- [ ] Validate behavior when payload includes non-UTF-8 content

---

## Phase 2 — Span Indexing & Enclosure Lookup (intervaltree)

### Goal
Accelerate enclosing-def and overlap queries with fast interval lookup.

### Core changes
- Build an IntervalTree per file for definition spans.
- Replace linear scans with `tree.overlap` or `tree.envelop`.

### Code pattern
```python
from intervaltree import IntervalTree

# Build once per file
spans = [(rec.span.start_byte, rec.span.end_byte, rec) for rec in defs]
index = IntervalTree.from_tuples(spans)

# Enclosing definition for a byte offset
hits = index.at(byte_offset)
nearest = min(hits, key=lambda iv: iv.end - iv.begin) if hits else None
```

### Checklist
- [ ] Add span trees to SearchContext cache
- [ ] Replace enclosing_def scans with intervaltree lookups
- [ ] Add tests for edge cases (adjacent spans, nested spans)

---

## Phase 3 — Wiring Pack Validation (msgspec + derived capture checks)

### Goal
Prevent invalid wiring packs (missing captures, invalid templates) from running.

### Core changes
- Validate packs against msgspec schema and derived rule capture lists.
- Fail fast when `entry_key_template` references missing fields.
- Emit structured validation results in `related.validation`.

### Code pattern
```python
class EmitConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    entry_key_template: str
    hook_span_capture: str | None = None
    target_symbol_hint_capture: str | None = None

# Template fields must be in captures or built-ins
missing = required_fields - (captures | {"pack_id", "path", "rule_id"})
if missing:
    raise ValueError(f"Entry template missing fields: {sorted(missing)}")
```

### Checklist
- [ ] Validate pack schema with msgspec
- [ ] Check postprocess operators reference valid captures
- [ ] Validate entry_key templates for missing fields
- [ ] Return validation errors in pack debug payload

---

## Phase 4 — Persistent Results Store (pyarrow)

### Goal
Store query results in Parquet with versioned schemas and fast scans.

### Core changes
- Define Arrow schemas for results (match records, wiring edges).
- Use dictionary encoding for `path`, `rule_id`, `pack_id`.
- Attach schema metadata with pack versions and engine version.

### Code pattern
```python
import pyarrow as pa
import pyarrow.parquet as pq

schema = pa.schema(
    [
        ("path", pa.dictionary(pa.int32(), pa.string())),
        ("start_byte", pa.int64()),
        ("end_byte", pa.int64()),
        ("pack_id", pa.dictionary(pa.int32(), pa.string())),
        ("rule_id", pa.dictionary(pa.int32(), pa.string())),
        ("captures", pa.map_(pa.string(), pa.list_(pa.string()))),
    ],
    metadata={"engine_version": b"1", "pack_version": b"2026-01"},
)

rows = [
    {
        "path": rec.path,
        "start_byte": rec.span.start_byte,
        "end_byte": rec.span.end_byte,
        "pack_id": rec.pack_id,
        "rule_id": rec.rule_id,
        "captures": rec.captures,
    }
    for rec in records
]

table = pa.Table.from_pylist(rows, schema=schema)
pq.write_table(table, "build/query_results.parquet")
```

### Checklist
- [ ] Define Arrow schema for each result type
- [ ] Persist schema metadata for compatibility checks
- [ ] Add append or partition strategy (by pack_id or date)
- [ ] Implement read APIs for analytics and debug

---

## Phase 5 — Streaming Analytics & Budgets (polars)

### Goal
Enable streaming reads, incremental processing, and query profiling.

### Core changes
- Use `scan_parquet` to lazily query persisted results.
- Use `collect_batches` or `sink_batches` for early stop on budget.
- Use `profile()` for performance telemetry.

### Code pattern
```python
import polars as pl

lf = pl.scan_parquet("build/query_results.parquet")
filtered = lf.filter(pl.col("pack_id") == "wire.python.fastapi.routes")

for batch in filtered.collect_batches(chunk_size=1000):
    # Stop early on budget
    if budget_exhausted(batch):
        break

result_df, profile_df = filtered.profile()
```

### Checklist
- [ ] Add lazy scan APIs for persisted results
- [ ] Implement batch streaming with early termination
- [ ] Add profiling hooks for telemetry
- [ ] Ensure deterministic ordering when required

---

## Phase 6 — DataFrame Validation (pandera)

### Goal
Validate persisted/streamed tables against schema contracts.

### Core changes
- Define DataFrameModel for each result table.
- Validate polars/pandas DataFrames after read/stream operations.

### Code pattern
```python
import pandera.polars as pa
from pandera.typing.polars import DataFrame, Series

class MatchRow(pa.DataFrameModel):
    path: Series[str]
    start_byte: Series[int]
    end_byte: Series[int]
    pack_id: Series[str]
    rule_id: Series[str]

    class Config:
        strict = True
        coerce = True

validated: DataFrame[MatchRow] = MatchRow.validate(frame)
```

### Checklist
- [ ] Define Pandera models for each stored table
- [ ] Add validation to analytics pipeline
- [ ] Decide whether validation is always-on or feature-flagged

---

## Phase 7 — Operationalization & Rollout

### Goal
Deploy safely with metrics and feature flags.

### Core changes
- Feature flags for new persistence and validation paths.
- Metrics for pack validation errors and budget stops.
- Compatibility checks for schema version changes.

### Checklist
- [ ] Add config flags for Arrow persistence and Polars analytics
- [ ] Emit metrics on validation failures and pack errors
- [ ] Add backward-compatible schema upgrades
- [ ] Document operational knobs (budget, streaming chunk size)

---

## Deployment Checklist (summary)

1) **Contracts & JSON schema**
   - [ ] msgspec models live in `tools/advanced_query_engine/contracts.py`
   - [ ] JSON schema export for API docs

2) **Serialization**
   - [ ] orjson for CLI/API responses
   - [ ] Deterministic output enabled

3) **Span indexing**
   - [ ] intervaltree per file
   - [ ] enclosure/overlap APIs validated

4) **Pack validation**
   - [ ] msgspec schema validation
   - [ ] capture/template checks

5) **Persistence**
   - [ ] Arrow schemas finalized
   - [ ] Parquet storage implemented

6) **Analytics**
   - [ ] Polars scan/stream pipeline
   - [ ] Profiling enabled

7) **Validation**
   - [ ] Pandera models defined
   - [ ] Validation integrated in analytics path

8) **Operations**
   - [ ] Feature flags in config
   - [ ] Metrics and logs in place
   - [ ] Backward-compat checks for schema migrations
