# Dataset Root Configuration

## Overview
Arrow/Parquet datasets are the canonical build outputs. DuckDB reads datasets by
scanning the Parquet snapshot directories referenced by dataset manifests. The
dataset root is where those snapshot directories live.

## Default Location
By default, CodeIntel writes datasets under the build directory:

- `repo_root/build/datasets` (or `<build_dir>/datasets` when `build_dir` is overridden)

Snapshots follow the standard layout:

- `<dataset_root>/<schema>/<table>/snapshot_id=<snapshot_id>/dataset_manifest.json`

## Configure the Dataset Root
You can override the dataset root in `config/codeintel.yaml`:

```yaml
paths:
  dataset_root_dir: /absolute/path/to/datasets
```

Relative paths are resolved against `repo_root`.

## CLI Overrides for Migration
The DuckDB -> Parquet migration command supports direct overrides and optional
cleanup of legacy tables:

```bash
codeintel datasets migrate-parquet --dataset-root-dir /path/to/datasets
```

Common options:

- `--dataset-root-dir` (optional): override where datasets are written.
- `--snapshot-id` (optional): override the snapshot identifier used in parquet paths.
- `--table-key` (repeatable): migrate only specific tables.
- `--overwrite`: replace existing parquet snapshots.
- `--drop-duckdb-tables`: drop legacy DuckDB tables after a successful export.

Expected output layout:

- `<dataset_root>/<schema>/<table>/snapshot_id=<snapshot_id>/`
  - `dataset_manifest.json`
  - `part-*.parquet`

Safeguards:

- No DuckDB tables are dropped by default; `--drop-duckdb-tables` is explicit.
- With parquet-only policy enforced, missing dataset manifests will surface as
  runtime errors during serving or query execution.
