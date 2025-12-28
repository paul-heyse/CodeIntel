import importlib
from contextlib import suppress

MODULES = ["pkg.mod", "pkg.util"]
for name in MODULES:
    with suppress(Exception):
        importlib.import_module(name)
