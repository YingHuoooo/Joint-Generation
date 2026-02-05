"""Compiler that assembles code tokens and validates syntax."""
from __future__ import annotations

import ast
from typing import Iterable


def assemble_code(tokens: Iterable[str]) -> str:
    return "".join(tokens)


def check_syntax(code: str) -> None:
    ast.parse(code)
