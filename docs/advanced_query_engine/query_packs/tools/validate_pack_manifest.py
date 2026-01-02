\
#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    missing: list[str] = []

    def check(path_str: str) -> None:
        p = root / path_str
        if not p.exists():
            missing.append(path_str)

    for preset in manifest.get("rpygrep_presets", []):
        check(preset["file"])
    for pg in (root / "rpygrep" / "patterns").glob("*.json"):
        # existence already implied by glob
        pass
    for pack in manifest.get("ast_grep_packs", []):
        check(pack["file"])
    for pack in manifest.get("tree_sitter_packs", []):
        check(pack["file"])
    for pack in manifest.get("wiring_packs", []):
        check(pack["file"])

    if missing:
        print("Missing referenced files:")
        for m in missing:
            print("  -", m)
        return 2

    print("OK: all manifest paths exist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
