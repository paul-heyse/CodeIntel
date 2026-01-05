# Contract Override Tags

This reference lists the supported `ci.*` tags used to override dataset contract
metadata. These tags are parsed from Hamilton saver metadata and are intended
for exceptions; defaults are derived from the table key and schema when tags are
absent.

## Supported Tags
- `ci.json_schema_id`: Override the JSON Schema identifier used for validation.
- `ci.jsonl_filename`: Override the default JSONL export filename.
- `ci.parquet_filename`: Override the default Parquet export filename.
- `ci.dataset_owner`: Override the dataset owner/team label.
- `ci.validation_profile`: Override the validation profile (`strict`, `lenient`,
  `schema-only`).

## Usage
- Provide tag values via `SaveToObjectMetadataDecorator` or `save_to_*` helpers
  using `value(...)` so tags are available at DAG compile time.
- Tags apply per-output and are read from saver metadata nodes.

## Precedence
- Observed schema (from bundle observations) overrides inferred/manifest schema.
- Inferred/manifest schema overrides declared schema.
- Override tags only change metadata fields; they do not replace schema
  inference.
