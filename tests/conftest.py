import os
import sys


def _ensure_project_root_on_path() -> None:
    """Add project root to sys.path for test imports.

    Inputs: none (uses current file location).
    Outputs: none (mutates sys.path in-place).
    Operation: resolves repository root and inserts it at the front of sys.path
    to enable absolute imports in tests.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_project_root_on_path()
