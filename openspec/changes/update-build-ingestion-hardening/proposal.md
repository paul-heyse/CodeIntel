# Change: Build and Ingestion Hardening for Deterministic Tracking

## Why
Recent failures show build runs can collide on run IDs, ingestion targets can return
coroutines, and inventory/artifact validation can accept empty outputs. These issues
reduce determinism and make correctness drift hard to detect.

## What Changes
- Enforce collision-resistant run IDs and idempotent build run tracking.
- Require Hamilton ingestion targets to return concrete results (no coroutine outputs).
- **BREAKING**: Treat empty SCIP symbol/occurrence documents as hard failures.
- **BREAKING**: Normalize coverage GOID typing and ensure coverage edges include all executed
  functions (span alignment compares subsets instead of exact equality).
- Centralize graph validation module inventory fallback to the catalog when core.modules is
  empty or unavailable.
- Make core.repo_map writes snapshot-singleton with replace/upsert semantics.
- Normalize repository GOID types and add a safe fallback for file summaries when docs
  views are empty.

## Impact
- Affected specs: build-execution, storage-boundaries
- Affected code: build/hamilton/executor.py, storage/tracking/build_tracking.py,
  build/hamilton/native/ingestion/ingest_targets.py, build/hamilton/native/ingestion/scip.py,
  analytics/testing/coverage/edges.py, graphs/validation/checks/database.py,
  storage/repositories/modules.py, storage/warehouse.py
