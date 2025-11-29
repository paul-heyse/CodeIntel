## Docs Views Profiling Playbook

This note captures how to generate and interpret profiling artifacts for the subsystem docs views.

### What we profile
- `docs.v_subsystem_profile`
- `docs.v_subsystem_coverage`

Artifacts include EXPLAIN (logical) or EXPLAIN ANALYZE (with timing/rows) outputs.

### How to run
```bash
uv run python -m codeintel.storage.docs_view_profiling \
  --db-path build/db/codeintel.duckdb \
  --output-dir build/profiling \
  [--analyze]
```
- Omit `--analyze` for a fast logical plan (no scans). Include `--analyze` to capture row counts and timings (reads the DB).
- Outputs land under `build/profiling/` with one file per view plus `profile_meta.json`.

### Current recommendations
- Prefer cache + lightweight indexes over heavier materialization:
  - `analytics.test_profile` — `idx_analytics_test_profile_primary_subsystem` on `(primary_subsystem_id, repo, commit)`.
  - `analytics.subsystems` — `idx_analytics_subsystems_repo_commit_id` on `(repo, commit, subsystem_id)`.
- Refresh subsystem profile/coverage caches alongside analytics runs; rely on the indexes above for live fallbacks.

### Interpreting results
- Look for large scans on `analytics.test_profile` or joins against subsystem/graph metrics.
- If timing shows other predicates dominating (e.g., status/date filters), add targeted indexes on those columns.
- If joins re-scan the same aggregates, consider CTE reuse/materialization only when profiling shows a clear win; keep write-time overhead minimal.
