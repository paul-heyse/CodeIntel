# Change: DAG-first consolidation for analytics, CLI, and storage interfaces

## Why
Non-DAG orchestration paths, duplicated config/env parsing, and inconsistent persistence/IBIS
access increase drift and make it harder to keep Hamilton as the canonical source of truth.
A DAG-first consolidation keeps outputs consistent, simplifies debugging, and enforces
clear boundaries.

## What Changes
- Remove non-DAG analytics/graph/history orchestration paths and route all execution through
  Hamilton DAG targets or cached DAG artifacts.
- Preserve CLI/debug outputs but derive them from DAG-produced datasets or cached outputs.
- Centralize runtime configuration loading so build/CLI/serving entrypoints share the same
  loader and eliminate bespoke env/path parsing, including observability and metrics gating.
- Inject observability runtime handles from the canonical loader rather than per-surface
  bootstrap logic.
- Enforce storage-owned Ibis connections and replace ad-hoc analytics persistence with a
  contract-backed writer surface.
- Consolidate ID normalization utilities to canonical core helpers.

## Impact
- Affected specs: build-execution, config-injection, storage-boundaries, interface-hygiene.
- Affected code: analytics graphs/history/metrics modules, CLI history handlers, runtime
  configuration loader, storage Ibis access, observability bootstrap, serving app factories,
  and ID normalization utilities.
