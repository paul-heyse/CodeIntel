# Workspace Targets

This package is the canonical workspace root for repo-local Hamilton targets.
Add modules under `src/codeintel_targets/<domain>/` (for example,
`src/codeintel_targets/analytics/my_target.py`). Each module should define:

- a `t__<target>` anchor decorated with `codeintel.sdk.target_anchor`
- one or more saver nodes using `codeintel.sdk.save_to_table` or
  `codeintel.sdk.save_to_artifact`

The runtime module resolver will discover these modules automatically.
