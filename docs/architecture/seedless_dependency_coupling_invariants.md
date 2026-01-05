# Seedless Dependency Coupling Invariants

## Purpose
This document records the invariants for build execution in the seedless, compute-first Hamilton DAG. These rules ensure the build reflects the current repository state without relying on historical or external snapshots.

## Invariants
- The only source of truth is the current repository snapshot; seeded datasets are not supported.
- There is no partial recompute decision-making. Cached results may prove equality but never drive compute selection.
- All datasets produced by build targets must be computed in the current run.
- Loader/query nodes must not bypass computed data nodes for produced datasets.

## External Inputs
- External inputs are only allowed when explicitly allowlisted.
- The allowlist is stored in `config/registry/external_inputs_allowlist.yaml`.
- Any non-allowlisted external input is a preflight error.

## Diagnostics
- Validation findings are emitted as diagnostics and do not abort the DAG.
- Diagnostics outputs live under `build/diagnostics/`.
