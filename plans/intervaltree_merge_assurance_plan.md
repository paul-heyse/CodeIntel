# IntervalTree Merge Assurance Plan

## Goal
Deploy `intervaltree` to consolidate span-based merge logic in `src/codeintel/build` and
increase assurance for range joins (line/byte spans), while keeping Polars joins for
equality-based merges. This plan targets deterministic matching, explicit ambiguity
handling, and consistent half-open span semantics.

## Scope
- Span-based merges in build/ingestion pipelines where we currently rely on:
  - Exact line/byte equality joins.
  - Ad-hoc “closest span” heuristics.
  - Per-module or per-path Python loops without a canonical span resolver.
- Focused modules:
  - `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
  - `src/codeintel/build/hamilton/native/graphs/symbol_use.py`
  - `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
  - `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
  - `src/codeintel/build/analytics/data_models/core.py`
  - `src/codeintel/build/graphs/validation/checks/database.py`
  - Shared utilities: `src/codeintel/core/parsing/ast_index.py`, `src/codeintel/core/catalog/span_index.py`

## Design Principles
- Use **half-open** span semantics `[start, end)` everywhere; normalize inputs with
  `normalize_line_span`, `normalize_byte_span`, and `to_half_open_span`.
- Prefer **envelop/contains** match over generic overlap; use overlap only as fallback.
- Always emit **deterministic** matches:
  - Choose smallest enclosing span.
  - Use stable tie-breakers: `(span_len, start, id)` or equivalent.
- Explicitly record ambiguity:
  - Include candidate counts and a `match_kind` enum.
  - When strict mode is enabled, reject ambiguous matches.
- Keep equality joins as Polars joins; intervaltree is only for range joins.

## Phase 1: Shared Span Index + Canonical Resolution
**Objective:** Provide a single span-resolution helper usable across build modules.

### Work items
- [ ] Add `SpanResolver` helper (new file under `src/codeintel/core/` or `src/codeintel/build/`):
  - Build per-path `IntervalTree` from spans.
  - Provide `resolve_span(path, start, end, qualname=None)` with:
    1. exact match
    2. smallest enclosing match (envelop)
    3. smallest overlap match
  - Return `(match_id | None, match_kind, candidate_count)`.
- [ ] Normalize `AstSpanIndex`:
  - Replace the O(n) search in `src/codeintel/core/parsing/ast_index.py` with
    `IntervalTree` to match resolution order (exact → smallest enclosing → overlap).
  - Keep API stable (`lookup(start_line, end_line)`).
- [ ] Align `SpanIndex` in `src/codeintel/core/catalog/span_index.py` to the same
  match semantics (if any deviation remains).

### Acceptance
- `AstSpanIndex.lookup` returns identical or stricter results with deterministic ties.
- `SpanIndex` and `AstSpanIndex` share the same resolution order and half-open policy.

## Phase 2: SCIP Resolution & Syntax Augment (byte spans)
**Objective:** Standardize byte-span joins using canonical interval matching.

### Work items
- [ ] `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
  - Replace ad-hoc `IntervalTree` matching with the shared resolver helper.
  - Normalize spans via `normalize_byte_span` before insertion.
  - Ensure match kinds are consistently named: `EXACT`, `POINT`, `CONTAINS`, `OVERLAP`, `NONE`.
  - Emit candidate counts for every row.
- [ ] `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
  - Use shared resolver for matching LibCST ↔ tree-sitter nodes.
  - Ensure `split_overlaps()` or `merge_overlaps(strict=True)` is applied when overlapping
    intervals create ambiguity.

### Acceptance
- All span joins rely on shared resolver logic and emit match_kind + candidate_count.
- Ambiguity handling is consistent between SCIP resolution and syntax augmentation.

## Phase 3: Call Wiring & Symbol Use (byte/line spans)
**Objective:** Improve confidence and determinism for call target and symbol/GOID wiring.

### Work items
- [ ] `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
  - Build a per-path interval tree from SCIP occurrence spans using shared resolver.
  - Prefer `envelop` matches; use overlap only as fallback.
  - Log or mark ambiguous matches with candidate counts; update confidence formula.
- [ ] `src/codeintel/build/hamilton/native/graphs/symbol_use.py`
  - Replace direct tree overlap selection with the shared resolver (line spans).
  - Add strict mode guard: if a line matches multiple GOID spans, record ambiguity.
  - Ensure half-open line spans are used consistently.

### Acceptance
- Call target resolution and symbol GOID assignment are deterministic and emit ambiguity
  metadata.
- Confidence scores are stable under minor span shifts.

## Phase 4: Data Models + Validation checks (line spans)
**Objective:** Use intervaltree for line-span matching in analytics and validations.

### Work items
- [ ] `src/codeintel/build/analytics/data_models/core.py`
  - Replace `meta_by_line` lookups with per-path interval tree resolution.
  - Resolve class metadata via smallest enclosing span instead of start-line equality.
- [ ] `src/codeintel/build/graphs/validation/checks/database.py`
  - Use `SpanIndex` (IntervalTree-backed) to detect callsite span mismatches.
  - Emit diagnostics for ambiguous or overlapping spans.

### Acceptance
- Class metadata and validation checks no longer rely on exact start-line matches.
- Diagnostics clearly identify ambiguous matches and candidate counts.

## Phase 5: CPG Occurrence Fallback (optional)
**Objective:** Reduce row loss in CPG occurrence joins when exact line/col fails.

### Work items
- [ ] `src/codeintel/build/hamilton/native/graphs/cpg.py`
  - Add intervaltree-based fallback when exact join on line/col fails.
  - Preserve exact-join path for deterministic matches; use fallback only when
    exact join yields nulls.

### Acceptance
- CPG occurrence edges do not silently drop on minor span drift.
- Fallback joins are labeled with match_kind and candidate_count.

## Telemetry & Assurance
- Emit a small set of counters per join:
  - `interval_exact`, `interval_contains`, `interval_overlap`, `interval_none`,
    `interval_ambiguous`.
- Track ambiguity as a warning metric in build logs (not hard fail unless strict mode).

## Rollout Strategy
- Phase-by-phase gating:
  - Merge Phase 1 first (shared resolver + AstSpanIndex).
  - Apply Phase 2/3 to ingestion + graph wiring.
  - Apply Phase 4 to analytics + validations.
  - Phase 5 as optional improvement.
- Maintain feature flags for strict mode:
  - Default: warn on ambiguous matches.
  - Strict: error on ambiguous matches (per target option).

## Risks & Mitigations
- Risk: Changing span resolution might alter outputs.
  - Mitigation: introduce match_kind/candidate_count and keep exact matches preferred.
- Risk: Performance for very large span sets.
  - Mitigation: per-path trees and lazy construction; avoid global trees.
- Risk: Half-open mismatch between datasets.
  - Mitigation: centralize normalization functions and add assertions in resolver.

## Definition of Done
- Shared intervaltree resolver implemented and used by all listed targets.
- All span-based merges record match_kind and candidate_count.
- No remaining span-based merges use ad-hoc equality or manual overlap logic.
- Documentation updated with span semantics and match_kind taxonomy.
