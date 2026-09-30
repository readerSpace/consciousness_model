"""Program-shaped physics: declare a field, state the equation, draw the path.

    vector: B = (0, 0, Bz)
    lorentz: F exists
    calc(m d^2 x / dt^2 = lorentz(B))
    draw(x)

The interesting part of this language is not that it integrates an ODE.  It is
what it does with ``Bz``.

``Bz`` is written and never given a value, which makes it the same shape of
problem as 「フガ率」 in the dialogue layers: a symbol the person used whose
meaning the system does not have.  The rule those layers arrived at applies
unchanged -- **do not execute on an unresolved symbol** -- so the script stops,
names what is missing, and offers the lines that would bind it.  Guessing
``Bz = 1`` would draw a confident picture of a world nobody described.

Which leads to the one distinction worth stating plainly, because it is the only
place a value appears that nobody typed:

* **A symbol the person introduced has no default.**  ``Bz``, and the mass in
  their own equation, must be given values.
* **A symbol the system introduced does.**  The charge ``q`` belongs to the
  Lorentz law rather than to their text, and the initial position, velocity and
  time span are how the picture gets drawn rather than part of the physics they
  stated.  Those take defaults, and every one of them is listed in the output as
  supplied.

Everything else follows the rules the rest of the project already runs on.
Nothing is evaluated -- expressions are parsed and walked against a closed table
(:mod:`math_expression`).  A script that is not wholly readable is wholly not
run, which is L8.18's execution gate again.  An unknown keyword is refused by
name, with the nearest known one offered, rather than guessed at.

**Limits, stated rather than discovered.**  Fields are uniform: each component
must evaluate to a number, so ``B`` may not depend on position or time.  The
motion is non-relativistic.  The integrator is fixed-step RK4, and it is checked
against the closed-form cyclotron solution rather than trusted --
``analytic_error`` reports that comparison and the tests hold it to a bound.
"""
from __future__ import annotations

import base64
import difflib
import io
import math
import re
from dataclasses import dataclass, field

from .math_expression import MathError, evaluate, format_number, free_names

KINDS = ("scalar", "vector", "time", "lorentz")
LAWS = ("lorentz",)

#: Defaults the *system* brings, never the person. Every one is disclosed in the
#: output; see the module docstring for why these and not others.
SUPPLIED = {
    "q": 1.0,
    "x0": (0.0, 0.0, 0.0),
    "v0": (1.0, 0.0, 0.5),
    "t": (0.0, 20.0, 2000),
}

MAX_STEPS = 200_000


class PhysicsError(ValueError):
    def __init__(self, message: str, line: int = 0):
        super().__init__(message)
        self.line = line

    def __str__(self) -> str:
        return (f"{self.args[0]}（{self.line} 行目）" if self.line
                else str(self.args[0]))


def _suggest(word: str, known) -> str:
    close = difflib.get_close_matches(word, list(known), n=1, cutoff=0.6)
    return f"（`{close[0]}` のことですか？）" if close else ""


# --------------------------------------------------------------------------
# statements
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Statement:
    kind: str  # scalar | vector | time | law | calc | draw
    name: str
    body: str
    line: int
    source: str


_DECLARATION = re.compile(r"^(?P<kind>[A-Za-z_]\w*)\s*:\s*(?P<body>.+)$")
_ASSIGNMENT = re.compile(r"^(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<body>.+)$")
_CALLABLE = re.compile(r"^(?P<name>[A-Za-z_]\w*)\s*\((?P<body>.*)\)$", re.DOTALL)
_TUPLE = re.compile(r"^\((?P<body>.*)\)$", re.DOTALL)
_EXISTS = re.compile(r"^(?P<symbol>[A-Za-z_]\w*)\s+exists?$", re.IGNORECASE)
#: ``m d^2 x / dt^2`` and ``md^2x/dt^2`` and ``m * d^2 x / dt^2``.
_SECOND_DERIVATIVE = re.compile(
    r"^(?P<mass>[A-Za-z_]\w*)?\s*\*?\s*d\s*\^?\s*2\s*(?P<var>[A-Za-z_]\w*)"
    r"\s*/\s*d\s*(?P<time>[A-Za-z_]\w*)\s*\^?\s*2\s*$")


def parse(source: str) -> tuple[Statement, ...]:
    """Read every line, or refuse the whole script.

    Refusing the whole thing is the point: a script is never partly run because
    its last line was misspelled.
    """
    text = source.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```")
                         else lines[1:])
    found: list[Statement] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        call = _CALLABLE.match(line)
        if call and call.group("name") in ("calc", "draw"):
            found.append(Statement(call.group("name"), "", call.group("body").strip(),
                                   number, line))
            continue
        declaration = _DECLARATION.match(line)
        if declaration and not call:
            kind = declaration.group("kind").lower()
            body = declaration.group("body").strip()
            if kind in LAWS:
                if not _EXISTS.match(body):
                    raise PhysicsError(
                        f"`{kind}:` の後は `F exists` の形だけです"
                        f"{_suggest(body.split()[-1] if body.split() else '', ('exists',))}",
                        number)
                found.append(Statement("law", kind, body, number, line))
                continue
            if kind not in KINDS:
                raise PhysicsError(
                    f"`{kind}:` は使えません{_suggest(kind, KINDS)}"
                    f"（使えるのは {'、'.join(KINDS)}）", number)
            assignment = _ASSIGNMENT.match(body)
            if assignment is None:
                raise PhysicsError(f"`{kind}:` の後は `名前 = 値` の形です", number)
            found.append(Statement(kind, assignment.group("name"),
                                   assignment.group("body").strip(), number, line))
            continue
        if call:
            raise PhysicsError(
                f"`{call.group('name')}(...)` は使えません"
                f"{_suggest(call.group('name'), ('calc', 'draw'))}", number)
        assignment = _ASSIGNMENT.match(line)
        if assignment:
            found.append(Statement("scalar", assignment.group("name"),
                                   assignment.group("body").strip(), number, line))
            continue
        head = line.split()[0] if line.split() else line
        raise PhysicsError(
            f"読めない行です{_suggest(head.rstrip(':'), KINDS + ('calc', 'draw'))}",
            number)
    if not found:
        raise PhysicsError("実行できる行がありません")
    return tuple(found)


# --------------------------------------------------------------------------
# the model a script builds
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Equation:
    mass: str
    variable: str
    time: str
    law: str
    arguments: tuple[str, ...]
    line: int


@dataclass
class Model:
    scalars: dict = field(default_factory=dict)
    vectors: dict = field(default_factory=dict)   # name -> (expr, expr, expr)
    laws: set = field(default_factory=set)
    span: tuple | None = None
    equation: Equation | None = None
    supplied: dict = field(default_factory=dict)

    def components(self, name: str) -> tuple[float, float, float]:
        return tuple(evaluate(part, self.scalars)
                     for part in self.vectors[name])  # type: ignore[return-value]


def _tuple_parts(body: str, line: int, wanted: int = 3) -> tuple[str, ...]:
    match = _TUPLE.match(body.strip())
    if match is None:
        raise PhysicsError("`(a, b, c)` の形で書いてください", line)
    parts = tuple(part.strip() for part in match.group("body").split(",")
                  if part.strip())
    if len(parts) != wanted:
        raise PhysicsError(f"成分は {wanted} 個です（{len(parts)} 個ありました）", line)
    return parts


def build(statements) -> Model:
    model = Model()
    for statement in statements:
        if statement.kind == "scalar":
            model.scalars[statement.name] = statement.body
        elif statement.kind == "vector":
            model.vectors[statement.name] = _tuple_parts(statement.body, statement.line)
        elif statement.kind == "time":
            model.span = _tuple_parts(statement.body, statement.line)
        elif statement.kind == "law":
            model.laws.add(statement.name)
        elif statement.kind == "calc":
            if "=" in statement.body:
                model.equation = _equation(statement.body, statement.line)
    return model


def _equation(body: str, line: int) -> Equation:
    left, _, right = body.partition("=")
    derivative = _SECOND_DERIVATIVE.match(left.strip())
    if derivative is None:
        raise PhysicsError(
            "左辺は `m d^2 x / dt^2` の形だけが読めます"
            f"（読めなかったのは `{left.strip()}`）", line)
    call = _CALLABLE.match(right.strip())
    if call is None:
        raise PhysicsError(
            f"右辺は `lorentz(B)` のような力の呼び出しだけです"
            f"（読めなかったのは `{right.strip()}`）", line)
    law = call.group("name")
    if law not in LAWS:
        raise PhysicsError(f"`{law}` という力は知りません{_suggest(law, LAWS)}", line)
    arguments = tuple(part.strip() for part in call.group("body").split(",")
                      if part.strip())
    if not 1 <= len(arguments) <= 2:
        raise PhysicsError("`lorentz(B)` または `lorentz(B, E)` です", line)
    return Equation(derivative.group("mass") or "m", derivative.group("var"),
                    derivative.group("time"), law, arguments, line)


# --------------------------------------------------------------------------
# what is missing, and what the system is willing to supply
# --------------------------------------------------------------------------

def missing(model: Model) -> list[str]:
    """Symbols the person wrote and never bound.

    Scalars may refer to other scalars, so this is a fixed point rather than one
    pass: ``r = 2*a`` with ``a = 3`` needs nothing.
    """
    bound = set(model.scalars)
    wanted: set[str] = set()
    for expression in model.scalars.values():
        wanted |= free_names(expression)
    for parts in model.vectors.values():
        for expression in parts:
            wanted |= free_names(expression)
    if model.span is not None:
        for expression in model.span:
            wanted |= free_names(expression)
    if model.equation is not None:
        wanted.add(model.equation.mass)
        for name in model.equation.arguments:
            if name not in model.vectors:
                wanted.add(name)
    return sorted(wanted - bound - set(model.vectors))


def resolve(model: Model) -> Model:
    """Bind the scalars, and note every value the system had to supply."""
    values: dict = {}
    for _ in range(len(model.scalars) + 1):
        for name, expression in model.scalars.items():
            try:
                values[name] = evaluate(expression, values)
            except MathError:
                continue
    unresolved = [name for name in model.scalars if name not in values]
    if unresolved:
        raise PhysicsError(f"値を決められない量があります: {'、'.join(unresolved)}")
    model.scalars = values
    if model.equation is not None and "q" not in model.scalars:
        model.scalars["q"] = SUPPLIED["q"]
        model.supplied["q"] = SUPPLIED["q"]
    if model.span is None:
        model.span = SUPPLIED["t"]
        model.supplied["t"] = SUPPLIED["t"]
    else:
        model.span = tuple(evaluate(part, model.scalars) for part in model.span)
    for name in ("x0", "v0"):
        if name not in model.vectors:
            model.vectors[name] = tuple(str(value) for value in SUPPLIED[name])
            model.supplied[name] = SUPPLIED[name]
    return model


# --------------------------------------------------------------------------
# solving
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Solution:
    variable: str
    times: list
    positions: list   # list of (x, y, z)
    velocities: list
    omega: float | None
    analytic_error: float | None


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def solve(model: Model) -> Solution:
    equation = model.equation
    if equation is None:
        raise PhysicsError("解くべき式がありません（`calc(...)` を書いてください）")
    mass = model.scalars[equation.mass]
    if mass == 0:
        raise PhysicsError("質量が 0 では解けません")
    charge = model.scalars["q"]
    magnetic = model.components(equation.arguments[0])
    electric = (model.components(equation.arguments[1])
                if len(equation.arguments) > 1 else (0.0, 0.0, 0.0))

    start, stop, count = model.span
    count = int(count)
    if count < 2 or count > MAX_STEPS:
        raise PhysicsError(f"刻み数は 2 以上 {MAX_STEPS} 以下です（{count} でした）")
    if stop <= start:
        raise PhysicsError("時間の終わりは始まりより後にしてください")

    position = list(model.components("x0"))
    velocity = list(model.components("v0"))
    step = (stop - start) / (count - 1)

    def acceleration(v):
        force = _cross(v, magnetic)
        return [charge / mass * (electric[i] + force[i]) for i in range(3)]

    times, positions, velocities = [start], [tuple(position)], [tuple(velocity)]
    for index in range(1, count):
        # Fixed-step RK4 on (x, v). The field is uniform, so the derivative does
        # not depend on t and the stages differ only through v.
        k1v = acceleration(velocity)
        k1x = list(velocity)
        v2 = [velocity[i] + 0.5 * step * k1v[i] for i in range(3)]
        k2v, k2x = acceleration(v2), v2
        v3 = [velocity[i] + 0.5 * step * k2v[i] for i in range(3)]
        k3v, k3x = acceleration(v3), v3
        v4 = [velocity[i] + step * k3v[i] for i in range(3)]
        k4v, k4x = acceleration(v4), v4
        for i in range(3):
            position[i] += step / 6 * (k1x[i] + 2 * k2x[i] + 2 * k3x[i] + k4x[i])
            velocity[i] += step / 6 * (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i])
        times.append(start + index * step)
        positions.append(tuple(position))
        velocities.append(tuple(velocity))

    omega, error = None, None
    if magnetic[0] == 0 and magnetic[1] == 0 and electric == (0.0, 0.0, 0.0):
        omega = charge * magnetic[2] / mass
        error = analytic_error(model, times, positions, omega)
    return Solution(equation.variable, times, positions, velocities, omega, error)


def analytic_error(model: Model, times, positions, omega: float) -> float | None:
    """Largest gap between the integrator and the closed-form cyclotron path.

    Reported rather than assumed: a solver that is only checked against itself
    has not been checked. With ``B = (0, 0, B)`` and no electric field the motion
    is a helix and the exact path is known, so the comparison is available for
    free and the tests hold it to a bound.
    """
    if omega == 0:
        return None
    x0 = model.components("x0")
    vx0, vy0, vz0 = model.components("v0")
    worst = 0.0
    for time, found in zip(times, positions):
        phase = omega * time
        exact = (x0[0] + (vx0 / omega) * math.sin(phase)
                 - (vy0 / omega) * (math.cos(phase) - 1.0),
                 x0[1] + (vx0 / omega) * (math.cos(phase) - 1.0)
                 + (vy0 / omega) * math.sin(phase),
                 x0[2] + vz0 * time)
        worst = max(worst, math.dist(found, exact))
    return worst


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def figure(solution: Solution, title: str) -> str:
    """The trajectory as a PNG data URI: one 3-D view and one projection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [point[0] for point in solution.positions]
    ys = [point[1] for point in solution.positions]
    zs = [point[2] for point in solution.positions]

    # Kept small on purpose: the picture travels to the UI as a data URI and
    # ends up in the transcript, so a needlessly large PNG is a needlessly
    # large conversation.
    fig = plt.figure(figsize=(7.2, 3.2), dpi=96)
    space = fig.add_subplot(1, 2, 1, projection="3d")
    space.plot(xs, ys, zs, linewidth=1.4, color="#4c6ef5")
    space.scatter([xs[0]], [ys[0]], [zs[0]], s=22, color="#2f9e44", label="start")
    space.scatter([xs[-1]], [ys[-1]], [zs[-1]], s=22, color="#e03131", label="end")
    space.set_xlabel("x"); space.set_ylabel("y"); space.set_zlabel("z")
    space.set_title(title)
    space.legend(loc="upper left", fontsize=7, frameon=False)

    plane = fig.add_subplot(1, 2, 2)
    plane.plot(xs, ys, linewidth=1.4, color="#4c6ef5")
    plane.scatter([xs[0]], [ys[0]], s=22, color="#2f9e44")
    plane.scatter([xs[-1]], [ys[-1]], s=22, color="#e03131")
    plane.set_xlabel("x"); plane.set_ylabel("y")
    plane.set_title("xy projection")
    plane.set_aspect("equal", adjustable="datalim")
    plane.grid(alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# --------------------------------------------------------------------------
# running a script
# --------------------------------------------------------------------------

def looks_like_physics(source: str) -> bool:
    text = source.strip()
    if text.startswith("```physics"):
        return True
    if not any(cue in text for cue in
               ("calc(", "draw(", "vector:", "scalar:", "time:", "lorentz:")):
        return False
    try:
        parse(text)
    except PhysicsError:
        # A misspelling inside something that is plainly this language should be
        # reported *by* this language, not handed to the chat agent.
        return any(cue in text for cue in ("calc(", "draw(", "lorentz:", "vector:"))
    return True


def _needs_report(model: Model, wanted: list[str]) -> str:
    lines = ["**値が決まっていない量があるので実行しませんでした**", "",
             "次の記号はスクリプトの中で使われていますが、値が与えられていません。",
             "推測すると、誰も書いていない世界の絵を自信たっぷりに描くことになります。",
             ""]
    lines += [f"- `{name}`" for name in wanted]
    lines += ["", "この行を足すと動きます:", "", "```physics"]
    lines += [f"{name} = 1.0" for name in wanted]
    lines += ["```"]
    return "\n".join(lines)


def run(source: str) -> dict:
    """Parse the whole script, resolve it, and only then solve and draw."""
    try:
        statements = parse(source)
        model = build(statements)
    except (PhysicsError, MathError) as error:
        return {"ok": False, "error": str(error), "needs": [], "supplied": {},
                "results": [], "figures": [],
                "output": f"**実行しませんでした**: {error}"}

    wanted = missing(model)
    if wanted:
        return {"ok": False, "error": None, "needs": wanted, "supplied": {},
                "results": [], "figures": [], "output": _needs_report(model, wanted)}

    results: list[dict] = []
    figures: list[dict] = []
    try:
        model = resolve(model)
        solution = None
        for statement in statements:
            if statement.kind == "calc" and "=" not in statement.body:
                value = evaluate(statement.body, model.scalars)
                results.append({"line": statement.line, "source": statement.source,
                                "detail": f"{statement.body} = {format_number(value)}"})
            elif statement.kind == "calc":
                solution = solve(model)
                detail = (f"{solution.variable}(t) を {len(solution.times)} 点で解きました"
                          f"（t = {model.span[0]} … {model.span[1]}）")
                if solution.omega is not None:
                    detail += f"、サイクロトロン角振動数 ω = {format_number(solution.omega)}"
                results.append({"line": statement.line, "source": statement.source,
                                "detail": detail})
            elif statement.kind == "draw":
                name = statement.body.strip()
                if solution is None:
                    raise PhysicsError("先に `calc(...)` で解いてください", statement.line)
                if name != solution.variable:
                    raise PhysicsError(
                        f"`{name}` は解かれていません"
                        f"（解いたのは `{solution.variable}`）", statement.line)
                uri = figure(solution, f"{name}(t)")
                figures.append({"name": name, "title": f"{name}(t)", "data_uri": uri})
                results.append({"line": statement.line, "source": statement.source,
                                "detail": "軌跡を描きました"})
    except (PhysicsError, MathError) as error:
        return {"ok": False, "error": str(error), "needs": [],
                "supplied": model.supplied, "results": results, "figures": [],
                "output": f"**途中で止めました**: {error}"}

    lines = ["**スクリプトを実行しました**", "",
             "| 行 | 文 | 結果 |", "| --- | --- | --- |"]
    for item in results:
        lines.append(f"| {item['line']} | `{item['source']}` | {item['detail']} |")
    if model.supplied:
        lines += ["", "**こちらで補った値**（書かれていなかったもの）", "",
                  "| 記号 | 値 | 理由 |", "| --- | --- | --- |"]
        reasons = {"q": "ローレンツ力の電荷。式ではなく法則の側の量",
                   "t": "描画のための時間範囲（開始, 終了, 点数）",
                   "x0": "初期位置", "v0": "初速度"}
        for name, value in model.supplied.items():
            lines.append(f"| `{name}` | {value} | {reasons.get(name, '既定値')} |")
    if solution is not None and solution.analytic_error is not None:
        lines += ["", f"解析解（らせん）との最大ずれ: "
                      f"**{solution.analytic_error:.3e}**"
                      "（ソルバを自分自身とだけ比べても検証にならないので出しています）"]
    for item in figures:
        lines += ["", f"![{item['title']}]({item['data_uri']})"]
    return {"ok": True, "error": None, "needs": [], "supplied": model.supplied,
            "results": results, "figures": figures, "output": "\n".join(lines)}
