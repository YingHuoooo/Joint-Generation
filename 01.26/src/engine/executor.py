"""CadQuery execution sandbox."""
from __future__ import annotations

from typing import Any, Dict


def execute_code(code: str) -> Dict[str, Any]:
    scope: Dict[str, Any] = {}
    exec(code, scope, scope)
    return scope
