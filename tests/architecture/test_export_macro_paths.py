"""Ensure exporters avoid direct table reads and use macro-driven paths."""

from __future__ import annotations

from pathlib import Path

DISALLOWED_SNIPPETS = (
    "con.table(",
    ".from_df(",
)


def test_exporters_avoid_direct_table_reads() -> None:
    """Exporters should be macro-driven, not direct table/dataframe reads."""
    base = Path("src/codeintel/pipeline/export")
    targets = [
        base / "export_jsonl.py",
        base / "export_parquet.py",
    ]
    offenders: list[str] = []
    for path in targets:
        text = path.read_text(encoding="utf-8")
        for snippet in DISALLOWED_SNIPPETS:
            if snippet in text:
                offenders.append(f"{path}:{snippet}")
    if offenders:
        raise AssertionError(f"Disallowed direct table/dataframe use: {', '.join(offenders)}")
