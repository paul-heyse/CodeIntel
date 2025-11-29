# Dataset Catalog and Scaffolding

## Catalog generation

Run:

```
codeintel datasets catalog --repo-root . --repo <slug> --commit <sha> --db-path build/db/db.duckdb --output-dir build/catalog
```

Outputs:
- `catalog.md` and `catalog.html` with schema columns, capabilities, ownership, freshness/retention, validation profile, filenames, schema digests, and sample rows (if requested).
- Pass `--sample-rows-strict` to fail when sampling is unavailable (e.g., missing `dataset_rows` macro); otherwise sampling issues are logged and skipped.

For CI, use `scripts/ci/contract_docs.sh` to emit the catalog and a specs snapshot; point CI artifacts to `build/catalog/`.

## Dataset scaffold

Create a new dataset skeleton:

```
codeintel datasets scaffold my_dataset --table-key analytics.my_dataset --owner team-data --freshness-sla daily --retention-policy 90d --schema-version 1 --validation-profile strict --output-dir build/dataset_scaffolds
```

Artifacts:
- TypedDict + serializer stub
- Row binding snippet
- JSON Schema stub
- Metadata JSON (owner, freshness, retention, filenames, stable id, validation profile)

Integrate by wiring the binding/metadata into the registry/bootstrap files and filling in the schema fields.
