# Quality Report Diagnostic Enrichment Plan

This document proposes a concrete diagnostic schema, enrichment pipeline, and
high-level code design to deliver best-in-class Pyright/Pyrefly diagnostics using
patterns from the advanced query engine.

---

## Goals

- Emit structured, comparable diagnostics across Pyright and Pyrefly.
- Provide actionable context: spans, snippets, and remediation guidance.
- Preserve tool outputs while layering enrichment in a deterministic, budgeted way.
- Reuse advanced query engine ideas (summary/primary/related/debug, snippets,
  symbol/precedent lookup, path-kind grouping).

Non-goals (initial scope):
- No changes to the underlying type checkers (Pyright/Pyrefly).
- No runtime code modifications or auto-fixes.

---

## Diagnostic Schema (Canonical)

This schema is a normalized wrapper for all tool diagnostics. It is intentionally
aligned with the advanced query engine response structure.

```json
{
  "summary": {
    "tool": "pyright",
    "status": "failed",
    "total": 24,
    "errors": 18,
    "warnings": 6,
    "files": 12,
    "duration_seconds": 5.2
  },
  "primary": [
    {
      "id": "pyright:reportGeneralTypeIssues",
      "severity": "error",
      "message": "Argument of type 'X' cannot be assigned...",
      "file": "src/codeintel/serving/contracts/check_operation_contracts.py",
      "span": {
        "start_line": 112,
        "start_col": 9,
        "end_line": 112,
        "end_col": 20
      },
      "snippet": {
        "text": "foo(bar)",
        "context_before": ["..."],
        "context_after": ["..."]
      },
      "rule": {
        "code": "reportGeneralTypeIssues",
        "category": "type",
        "source": "pyright"
      },
      "context": {
        "path_kind": "prod",
        "module": "codeintel.serving.contracts",
        "symbol": "foo"
      },
      "evidence": {
        "definition": {"file": "...", "span": {"start_line": 10, "start_col": 0, "end_line": 18, "end_col": 1}},
        "signature": "def foo(x: Bar) -> Baz",
        "type_origin": "foo returns Baz from ..."
      },
      "recommendations": [
        {
          "title": "Adjust call arguments",
          "detail": "Pass a Bar instance or update the overload used here.",
          "confidence": 0.72
        }
      ]
    }
  ],
  "related": {
    "groups": [
      {"by_rule": {"reportGeneralTypeIssues": 12, "reportOptionalMemberAccess": 4}},
      {"by_path_kind": {"prod": 18, "test": 6}}
    ],
    "precedents": [
      {"file": "src/...", "span": {"start_line": 88, "start_col": 4, "end_line": 90, "end_col": 16}, "why": "Same call with correct types"}
    ],
    "symbols": [
      {"name": "foo", "def_span": {"start_line": 10, "start_col": 0, "end_line": 18, "end_col": 1}}
    ]
  },
  "debug": {
    "partial": false,
    "budget_exhausted": false,
    "files_scanned": 240,
    "enrichment_latency_ms": 340
  }
}
```

### Required fields
- `summary`: counts and tool metadata.
- `primary`: ordered list of enriched diagnostics (the main user output).
- `related`: grouped/secondary outputs (precedents, symbol defs, groups).
- `debug`: enrichment budgets, partial flags, and performance metadata.

### Key normalized fields (per diagnostic)
- `id`: tool + rule or code (stable for grouping).
- `severity`: error/warning/info.
- `message`: raw tool message, unchanged.
- `file` + `span`: normalized location.
- `snippet`: contextual excerpt (before/after lines).
- `rule`: `code`, `category`, `source`.
- `context`: path kind, module, symbol name (when available).
- `evidence`: definition span, signature, type origin (when available).
- `recommendations`: ranked, short fixes (optional initially).

---

## Enrichment Pipeline

### 1) Parse raw tool output
- Pyright: consume `--outputjson` and normalize to canonical `primary` entries.
- Pyrefly: parse stderr into structured rows; map to canonical fields.

### 2) Attach spans and snippets
- Resolve file paths and line/column positions.
- Use line indexes to build `snippet` with 1-2 context lines (AQE snippet pattern).

### 3) Add semantic context
- `path_kind` classification (prod/test/doc/example) using AQE’s path classifier.
- Module name inference from path (`semantic_helpers.module_qname_from_path`).
- Basic symbol extraction from error message when possible.

### 4) Evidence enrichment (high-impact)
- For undefined/import errors: run a lightweight symbol lookup (AQE symbol.resolve idea).
- For argument mismatch: capture callsite span + signature (AQE signature parsing).
- For Optional/type mismatch: trace definition or function return origin when possible.

### 5) Precedents and groupings
- Use precedent search for 1-3 similar “correct” usage examples.
- Group counts by rule code, path kind, module, and severity.

### 6) Recommendations (ranked)
- Generate small, targeted fix suggestions based on evidence and precedents.
- Include a confidence score derived from match quality and evidence availability.

### 7) Budget/partial flags
- Enforce per-tool budgets for enrichment (max errors, max files, max time).
- Surface `partial` and `budget_exhausted` flags in `debug`.

---

## High-Level Code Design

### New modules
- `tools/quality_enrichment/contracts.py`
  - Defines canonical schema types and conversion helpers.
- `tools/quality_enrichment/parsers.py`
  - Pyright JSON parser → canonical diagnostics.
  - Pyrefly stderr parser → canonical diagnostics.
- `tools/quality_enrichment/snippets.py`
  - Snippet extraction (reuse AQE snippet logic or import it).
- `tools/quality_enrichment/enrichers.py`
  - Symbol resolver, signature analyzer, precedent search, grouping.
- `tools/quality_enrichment/pipeline.py`
  - Orchestrates parse → enrich → output.

### Integration points
- `tools/quality_report.py`
  - Add optional enrichment step after tool execution.
  - Store enriched diagnostics inside the report payload.

### Reuse from advanced query engine
- Snippets and spans: `tools.advanced_query_engine.util.snippets`.
- Path kind grouping: `tools.advanced_query_engine.util.semantic_helpers`.
- Precedent search pattern: `tools.advanced_query_engine.handlers.q7_precedent_search`.

---

## Implementation Plan

### Phase 1: Schema + parsing (Pyright/Pyrefly)
- Add canonical schema types and conversion helpers.
- Parse Pyright JSON output (`--outputjson`) into canonical diagnostics.
- Parse Pyrefly stderr output (regex-based) into canonical diagnostics.
- Emit `summary` and `primary` only (no enrichment yet).

### Phase 2: Snippet + context
- Add snippet extraction around each diagnostic span.
- Add path kind classification and module name inference.
- Populate `related.groups` (by rule, by path kind).

### Phase 3: Evidence enrichment
- For undefined symbols: attempt `symbol.resolve` candidate list.
- For callsite mismatches: attach signature + argument mapping.
- For Optional/type mismatch: annotate source of Optional when discoverable.

### Phase 4: Precedents and recommendations
- Attach 1–3 precedents for common diagnostics (type mismatch, missing import).
- Add lightweight recommendation generation with confidence scoring.

### Phase 5: Budgets and debug metadata
- Add enrichment budget config and partial flags.
- Store enrichment timing and budget data in `debug`.

---

## Success Criteria

- Diagnostics are fully structured and grouped across Pyright/Pyrefly.
- Each error includes actionable context: snippet, span, path kind.
- Evidence and precedents appear for common failure types.
- Enrichment is budgeted, deterministic, and does not slow the report beyond targets.

---

## Next Steps

- Agree on schema fields and severity mapping.
- Decide on initial enrichment budgets (max errors, time budget).
- Implement Phase 1 + Phase 2 first to get structured, enriched outputs quickly.
