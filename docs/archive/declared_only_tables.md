# Declared-Only Tables

Declared-only tables are schemas that are **not** produced by the Hamilton DAG.
They remain defined in `codeintel.config.datasets` as static inputs for
build/storage metadata, external sources, or docs views.

The canonical list is derived from the build metadata bundle's schema registry.
Entries with `derivation_kind == "declared_source"` are declared-only and
provide the reason for why a table is not inferred from the DAG.

## How To List Declared-Only Tables

Run the following from the repo root after a build:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("build/metadata/schema/schema_registry.json")
payload = json.loads(path.read_text(encoding="utf-8"))
entries = payload.get("entries", [])
keys = [
    e.get("table_key")
    for e in entries
    if e.get("derivation_kind") == "declared_source"
]
for key in sorted(k for k in keys if isinstance(k, str)):
    print(key)
PY
```

## Why Declared-Only

Typical reasons include:
- The table is a metadata or tracking table (e.g., `metadata.*`, `build.*`).
- The table represents an external or seed dataset.
- The table is a docs-only view without a DAG producer.
