"""Arithmetic that is read rather than run.

``1+1=?`` should answer ``2``.  The obvious way to do that is ``eval``, and the
obvious way is wrong: the input comes from a text box, and an evaluator that can
reach ``__import__`` is a remote shell with a calculator on the front.  So the
source is parsed with ``ast`` and walked against a closed table of node types,
operators, functions and constants.  A form outside the table is refused **by
name** rather than attempted, which is this project's fail-closed default
appearing one more time.

``^`` is read as exponentiation, not as exclusive-or.  In Python it is xor, and
that is the right reading in Python; here the surrounding language is
mathematics, where nobody writing ``2^10`` means 1000₂ xor 1010₂.  The
translation happens before parsing and is stated here because a silent
reinterpretation of an operator is exactly the sort of thing that should not be
silent.
"""
from __future__ import annotations

import ast
import math
import re

#: Everything callable from an expression. Nothing here touches the filesystem,
#: the network, or any object -- they are float -> float.
FUNCTIONS = {
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10,
    "log2": math.log2, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "atan2": math.atan2, "sinh": math.sinh, "cosh": math.cosh,
    "tanh": math.tanh, "hypot": math.hypot, "degrees": math.degrees,
    "radians": math.radians, "floor": math.floor, "ceil": math.ceil,
    "factorial": math.factorial, "gcd": math.gcd,
    "abs": abs, "round": round, "min": min, "max": max, "pow": pow,
}

CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}

_BINARY = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
_UNARY = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}

#: Guards against an expression that is cheap to write and expensive to evaluate
#: (``9**9**9``). A calculator has no business locking up the bridge.
MAX_EXPONENT = 10_000

_QUERY = re.compile(r"^(?P<body>.+?)\s*[=＝]\s*[?？]\s*$")
_PURE_ARITHMETIC = re.compile(r"^[\s\d.+\-*/%^()]+$")


class MathError(ValueError):
    """An expression that will not be evaluated, and why."""


def _as_python(source: str) -> str:
    """``^`` means exponentiation here; see the module docstring."""
    return source.replace("＝", "=").replace("^", "**")


def _walk(node: ast.AST, names: dict):
    if isinstance(node, ast.Expression):
        return _walk(node.body, names)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise MathError("数値以外は書けません")
        return node.value
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise MathError(f"`{node.id}` の値が分かりません")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_walk(node.operand, names))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _walk(node.left, names), _walk(node.right, names)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise MathError("指数が大きすぎます")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise MathError("0 では割れません")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            wanted = getattr(getattr(node, "func", None), "id", "?")
            raise MathError(f"`{wanted}` という関数は使えません")
        if node.keywords:
            raise MathError("キーワード引数は使えません")
        return FUNCTIONS[node.func.id](*[_walk(item, names) for item in node.args])
    raise MathError("式として読めない書き方です")


def free_names(source: str) -> set[str]:
    """Names an expression needs and does not carry itself.

    Used by the physics language to say *which* symbols are missing instead of
    failing on the first one it happens to meet.
    """
    try:
        tree = ast.parse(_as_python(source), mode="eval")
    except SyntaxError as error:
        raise MathError(f"式として読めません: {error.msg}") from error
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in CONSTANTS:
            if not (isinstance(getattr(node, "ctx", None), ast.Load)
                    and node.id in FUNCTIONS):
                found.add(node.id)
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    return found - calls


def evaluate(source: str, names: dict | None = None):
    """The value of an expression, or ``MathError`` saying what stopped it."""
    text = _as_python(str(source).strip())
    if not text:
        raise MathError("式がありません")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise MathError(f"式として読めません: {error.msg}") from error
    return _walk(tree, dict(names or {}))


def format_number(value) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (math.inf, -math.inf):
            return str(value)
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


def is_query(message: str) -> bool:
    """Is this a question about a number rather than something someone said?

    Either it ends in ``=?``, or it is arithmetic and nothing else. Prose with a
    number in it is neither, and falls through untouched.
    """
    text = message.strip()
    if _QUERY.match(text):
        return True
    return bool(_PURE_ARITHMETIC.match(text)) and any(
        character in text for character in "+-*/%^")


def answer_query(message: str) -> str | None:
    """``1+1=?`` -> ``1 + 1 = 2``. ``None`` when this is not a question at all."""
    text = message.strip()
    if not is_query(text):
        return None
    match = _QUERY.match(text)
    body = match.group("body") if match else text
    try:
        value = evaluate(body)
    except MathError as error:
        return f"`{body}` は計算できません: {error}"
    return f"`{body.strip()}` = **{format_number(value)}**"
