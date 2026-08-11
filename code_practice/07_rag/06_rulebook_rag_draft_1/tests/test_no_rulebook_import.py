"""§13 Phase 1: generate_synthetic.py must contain no import from rulebook/. §22.7 hard constraint."""

import ast
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent.parent / "corpus" / "generate_synthetic.py"


def test_no_rulebook_import():
    tree = ast.parse(SRC_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("rulebook"), f"forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("rulebook"), f"forbidden import: {node.module}"
