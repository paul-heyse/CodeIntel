from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "path",
    [
        Path("src/codeintel/ingestion/py_ast_extract.py"),
        Path("src/codeintel/ingestion/cst_extract.py"),
        Path("src/codeintel/ingestion/scip_ingest.py"),
    ],
)
def test_no_ad_hoc_change_detection(path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / path).read_text(encoding="utf8")
    banned = [
        "compute_changes",
        "_should_fallback_to_full",
        "_gather_changed_paths",
        "ChangeSet",
    ]
    offenders = [token for token in banned if token in text]
    if offenders:
        message = f"{path} contains ad-hoc change detection tokens: {offenders}"
        pytest.fail(message)
