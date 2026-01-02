# Tools (reference runners)

These scripts are **reference-grade glue** for consuming the pack library.

They are intentionally small and policy-free; your production system should wrap them with:
- repo-level safety rails (timeouts, per-tool budgets),
- deterministic ordering,
- caching and index reuse.

## Files

- `run_rpygrep_profile.py`: load a preset + pattern group and run `RipGrepSearch`, emitting
  normalized match records (path + line_number + submatch spans + context).

- `validate_pack_manifest.py`: sanity-check that all files referenced in `manifest.json` exist.

