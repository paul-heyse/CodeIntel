# Change: Idempotent schema seeding and harness error surfacing

## Why
Ad-hoc schema creation in tests now conflicts with production schema bootstrapping and causes
flaky failures when schemas already exist. Harness record lookups also mask underlying Hamilton
build errors, slowing diagnosis and making failures harder to triage.

## What Changes
- Add a centralized, idempotent test helper for production-coupled schema seeding (docs schema at
  minimum).
- Replace ad-hoc `CREATE SCHEMA` usage in tests with the shared helper.
- Harden Hamilton build harness record access to surface underlying build errors and target
  status context when records are missing.

## Impact
- Affected specs: test-helpers
- Affected code: tests/_helpers schema helpers, tests/_helpers/harnesses, serving tests that
  create docs schemas
