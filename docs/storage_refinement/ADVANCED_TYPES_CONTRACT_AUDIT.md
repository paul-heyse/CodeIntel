# Advanced DuckDB Types: Contract Audit

This audit reviews how advanced DuckDB types (UUID/LIST/MAP/STRUCT/
TIMESTAMP_* variants) are represented in contract schemas and highlights
follow-up work needed to support them end-to-end.

## Findings

- `codeintel.core.schemas.primitives.ColumnType` is a strict `Literal[...]`
  that only includes the legacy primitive set.
- `TableSchema` and `Column` definitions therefore cannot encode parameterized
  DuckDB types like `LIST<VARCHAR>` or `STRUCT<...>` without using casts.
- The storage roundtrip layer (`storage/schema_roundtrip.py`) now supports
  advanced types at runtime, but the contract surface still blocks them
  at type-check time.

## Implications

- Storage can roundtrip advanced types, but contract authors must either:
  - avoid them entirely, or
  - use `typing.cast(ColumnType, "<type>")` as a short-term escape hatch.
- Tooling that relies on `get_args(ColumnType)` (CLI/schema validation,
  JSON schema generation, Pandera typing) remains limited to the legacy set.

## Recommended next steps

1) Introduce a formal `DuckDBType` string surface in core schemas so
   contracts can express parameterized types without casts.
2) Move validation to runtime (explicit allowlist or regex) and keep the
   legacy ColumnType set as a compatibility guardrail.
3) Update JSON schema and Pandera generators to recognize advanced types
   (map them to safe JSON/Pandas representations where possible).
4) Add contract tests that include at least one `LIST`, `MAP`, `STRUCT`,
   and `UUID` column to ensure the end-to-end flow accepts them.

This keeps the type system strict where it adds value, while allowing
storage to take advantage of DuckDB’s advanced types.
