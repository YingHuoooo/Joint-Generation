"""Python AST parser for CadQuery call trees."""
from __future__ import annotations

import ast
from typing import Any, Dict


def parse_code_to_tree(code: str) -> Dict[str, Any]:
    """Parse code into a simple AST dictionary."""
    tree = ast.parse(code)
    return {"type": type(tree).__name__, "body_len": len(tree.body)}
