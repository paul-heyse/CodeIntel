from __future__ import annotations

import hashlib
from typing import Any


def stable_hex_digest(*parts: Any, n: int = 16) -> str:
    """Return a stable hex digest for arbitrary JSON-serializable primitives.

    The digest is derived from the *stringified* parts joined by \x1f.
    This is deterministic across runs and Python versions (given same str()).

    Use cases:
      - edge_id
      - symbol_id
    """
    h = hashlib.sha256()
    sep = "\x1f"
    s = sep.join("" if p is None else str(p) for p in parts)
    h.update(s.encode("utf-8", errors="surrogatepass"))
    return h.hexdigest()[:n]
