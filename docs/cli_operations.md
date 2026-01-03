# CLI Operation Registry and Canonical IDs

## Overview

CodeIntel CLI operations are registered in a single operation registry. Each
operation has a canonical ID and may expose deprecated aliases for backward
compatibility. CLI commands are thin adapters that map flags to a handler and
execute the canonical operation ID.

## Canonical naming rules

- Use singular, stable prefixes for canonical IDs:
  - `graph.*` for graph analytics targets and plans
  - `jobs.*` for background job management
  - `health.*` for environment health checks
  - `dataset.*` for dataset contract inspection and verification
- Use plural command groups only for management bundles, not for canonical IDs.
  For example, the `datasets` command group shares handlers with `dataset`,
  but its canonical operation IDs remain `dataset.*`.

## Legacy aliases (deprecated)

These aliases remain supported for backward compatibility and are marked as
deprecated in registry metadata:

- `graphs.targets.list` -> `graph.targets.list`
- `graphs.targets.plan` -> `graph.targets.plan`
- `datasets.list` -> `dataset.list`
- `datasets.describe` -> `dataset.describe`
- `datasets.verify` -> `dataset.verify`
- `datasets.info` -> `dataset.info`
- `datasets.flow` -> `dataset.flow`
- `datasets.constraints` -> `dataset.constraints`

## Migration guidance

1. Use canonical operation IDs for programmatic invocation and scripting.
2. Treat alias IDs as deprecated; they may be removed after the next CLI
   compatibility window.
3. When wiring new commands, register handlers under canonical IDs and add
   an alias only when you must preserve existing behavior.

## Related plans

- `plans/cli-handler-canonicalization-plan.md`
