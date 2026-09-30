"""L4.7: family-agnostic compositional search over general code primitives.

One composer receives a general primitive vocabulary -- including primitives
that are useless for every contract in this experiment -- and no repair rule
for any contract family.  For each unseen contract it searches compositions
of those primitives, selects with real contract tests, abstracts the
successful trace into a parameterized operator whose arguments are stored as
observation roles rather than concrete names, and reuses that operator on
hold-outs that rename state, rename parameters, change literals, or change
repository structure.

The search, the abstraction, and the reuse code paths never branch on the
contract family.  ``search_is_family_agnostic`` re-checks that mechanically
by inspecting the source of the functions that perform them.  Operator names
such as ``ROLLBACK`` are attached post hoc from the discovered primitive
signature and are never used to select or guide search.

Known limitations, kept explicit rather than hidden: the composition grammar
forbids repeating an identical primitive instance inside one program, bounds
program length, and bounds loop iterations, so this measures composition
search inside a bounded general grammar, not unbounded program synthesis.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import ast
import inspect
import json
import shutil
import subprocess
import sys
import tempfile
import types

DISTRACTOR_PRIMITIVES = ("LOG", "SLEEP", "TOUCH")
PRIMITIVES = ("OPERATION", "EFFECT", "RESULT", "TRY", "EXCEPT", "FINALLY", "END", "GUARD", "REPEAT", "RAISE",
              "SAVE", "RESTORE", "READ", "WRITE", "REMOVE") + DISTRACTOR_PRIMITIVES
MAX_PROGRAM_LENGTH = 6
SEARCH_BUDGET = 400_000
POST_HOC_LABELS = {
    "SAVE>TRY>OPERATION>EXCEPT>RESTORE>RAISE": "ROLLBACK",
    "TRY>SAVE>OPERATION>EXCEPT>RESTORE>RAISE": "ROLLBACK",
    "EFFECT>WRITE>RESULT": "SYNC",
    "WRITE>OPERATION": "SYNC",
    "TRY>OPERATION>FINALLY>WRITE": "GUARANTEED_CLEANUP",
    "GUARD>OPERATION": "IDEMPOTENT_GUARD",
}


# --------------------------------------------------------------------------- observation


@dataclass(frozen=True)
class Container:
    expression: str
    kind: str
    value_types: tuple[str, ...]
    mutated: bool
    asserted: bool


@dataclass(frozen=True)
class Observation:
    alias: str
    function: str
    parameter: str
    body: tuple[str, ...]
    terminates: bool
    containers: tuple[Container, ...]
    literals: tuple[tuple[str, str], ...]
    indent: str
    body_range: tuple[int, int]


def _expression(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _expression(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


def _literal_kind(node: ast.AST) -> str:
    if isinstance(node, ast.Constant):
        return type(node.value).__name__
    return ""


def _collect_literals(tree: ast.AST) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and not isinstance(node.value, type(None)):
            entry = (type(node.value).__name__, repr(node.value))
            if entry not in found:
                found.append(entry)
    return found


def _container_kind(node: ast.AST) -> str:
    return "dict" if isinstance(node, ast.Dict) else "list" if isinstance(node, ast.List) else ""


def _container_value_types(node: ast.AST) -> tuple[str, ...]:
    values = node.values if isinstance(node, ast.Dict) else node.elts if isinstance(node, ast.List) else []
    return tuple(dict.fromkeys(kind for kind in (_literal_kind(value) for value in values) if kind))


def _classes(sources: dict[str, str]) -> dict[str, ast.ClassDef]:
    table: dict[str, ast.ClassDef] = {}
    for source in sources.values():
        for node in ast.parse(source).body:
            if isinstance(node, ast.ClassDef):
                table[node.name] = node
    return table


def _attribute_containers(definition: ast.ClassDef) -> list[tuple[str, str, tuple[str, ...]]]:
    found = []
    for method in definition.body:
        if isinstance(method, ast.FunctionDef) and method.name == "__init__":
            for statement in method.body:
                if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Attribute):
                    kind = _container_kind(statement.value)
                    if kind:
                        found.append((statement.targets[0].attr, kind, _container_value_types(statement.value)))
    return found


def _mutated_expressions(function: ast.FunctionDef, instances: dict[str, str],
                         classes: dict[str, ast.ClassDef]) -> set[str]:
    mutated: set[str] = set()
    for node in ast.walk(function):
        target = None
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        if isinstance(target, ast.Subscript):
            mutated.add(_expression(target.value))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = _expression(node.func.value)
            if node.func.attr in {"append", "update", "pop", "clear", "remove", "insert", "extend"}:
                mutated.add(owner)
            definition = classes.get(instances.get(owner, ""))
            if definition is not None:
                for method in definition.body:
                    if isinstance(method, ast.FunctionDef) and method.name == node.func.attr:
                        for attribute in _mutated_expressions(method, {}, classes):
                            if attribute.startswith("self."):
                                mutated.add(f"{owner}.{attribute[len('self.'):]}")
    return mutated


def observe(sources: dict[str, str], alias: str, script: str) -> Observation:
    """Derive every search parameter from the repository and its contract test."""
    entry = sources[alias]
    tree = ast.parse(entry)
    classes = _classes(sources)
    called = [node.func.attr for node in ast.walk(ast.parse(script))
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and _expression(node.func.value) == alias]
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in called)
    instances: dict[str, str] = {}
    plain: list[Container] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            kind = _container_kind(node.value)
            if kind:
                plain.append(Container(name, kind, _container_value_types(node.value), False, False))
            elif isinstance(node.value, ast.Call) and _expression(node.value.func) in classes:
                instances[name] = _expression(node.value.func)
                for attribute, attribute_kind, types_ in _attribute_containers(classes[instances[name]]):
                    plain.append(Container(f"{name}.{attribute}", attribute_kind, types_, False, False))
    mutated = _mutated_expressions(function, instances, classes)
    resolved = tuple(replace(container, mutated=container.expression in mutated,
                             asserted=container.expression in script) for container in plain)
    containers = tuple(sorted(resolved, key=lambda container: (0 if container.mutated else 1 if container.asserted else 2,
                                                               plain.index(next(item for item in plain
                                                                                if item.expression == container.expression)))))
    lines = entry.splitlines()
    first, last = function.body[0].lineno - 1, function.body[-1].end_lineno
    body = lines[first:last]
    indent = body[0][:len(body[0]) - len(body[0].lstrip())]
    literals: list[tuple[str, str]] = []
    for source in [entry] + [text for name, text in sources.items() if name != alias]:
        for entry_literal in _collect_literals(ast.parse(source)):
            if entry_literal not in literals:
                literals.append(entry_literal)
    parameter = next(argument.arg for argument in function.args.args if argument.arg != "self")
    terminal = isinstance(function.body[-1], (ast.Return, ast.Raise))
    return Observation(alias, function.name, parameter,
                       tuple(line[len(indent):] if line.startswith(indent) else line.strip() for line in body),
                       terminal, containers, tuple(literals), indent, (first, last))


# --------------------------------------------------------------------------- primitives


@dataclass(frozen=True)
class Token:
    primitive: str
    roles: tuple[str, ...]
    kind: str
    header: tuple[str, ...] = ()
    code: tuple[str, ...] = ()
    terminal: bool = False
    binds: str = ""
    requires: tuple[str, ...] = ()
    parts: tuple[str, ...] = ()


def _resolve_container(observation: Observation, role: str) -> Container | None:
    index = int(role[1:])
    return observation.containers[index] if index < len(observation.containers) else None


def _resolve_value(observation: Observation, role: str) -> str:
    if role == "PARAM":
        return observation.parameter
    if role.startswith("L:"):
        _, kind, rank = role.split(":")
        matching = [text for literal_kind, text in observation.literals if literal_kind == kind]
        return matching[int(rank)] if int(rank) < len(matching) else ""
    if role.startswith("R:"):
        container = _resolve_container(observation, role[2:])
        return f"{container.expression}[{observation.parameter}]" if container else ""
    if role.startswith("B:"):
        return f"observed_{role[2:][1:]}"
    return ""


def build_token(primitive: str, roles: tuple[str, ...], observation: Observation) -> Token | None:
    """Render one primitive instance against an observation; roles stay abstract."""
    if primitive == "OPERATION":
        return Token(primitive, roles, "statement", code=observation.body, terminal=observation.terminates,
                     parts=("EFFECT", "RESULT"))
    if primitive in {"EFFECT", "RESULT"}:
        if not observation.terminates or len(observation.body) < 2:
            return None
        boundary = next(index for index, line in reversed(list(enumerate(observation.body)))
                        if line and not line.startswith(" "))
        code = observation.body[:boundary] if primitive == "EFFECT" else observation.body[boundary:]
        if not code:
            return None
        return Token(primitive, roles, "statement", code=code, terminal=primitive == "RESULT",
                     parts=(primitive,))
    if primitive == "TRY":
        return Token(primitive, roles, "open", header=("try:",))
    if primitive == "EXCEPT":
        return Token(primitive, roles, "handler", header=("except Exception:",))
    if primitive == "FINALLY":
        return Token(primitive, roles, "handler", header=("finally:",))
    if primitive == "END":
        return Token(primitive, roles, "close")
    if primitive == "RAISE":
        return Token(primitive, roles, "statement", code=("raise",), terminal=True)
    if primitive == "LOG":
        return Token(primitive, roles, "statement", code=(f"_log = repr({observation.parameter})",))
    if primitive == "SLEEP":
        return Token(primitive, roles, "statement", code=("pass",))
    if primitive == "TOUCH":
        return Token(primitive, roles, "statement", code=("_touched = True",))
    container = _resolve_container(observation, roles[0]) if roles and roles[0].startswith("C") else None
    if primitive in {"SAVE", "RESTORE", "READ", "REMOVE", "WRITE"} and container is None:
        return None
    index = roles[0][1:] if container else ""
    if primitive == "SAVE":
        return Token(primitive, roles, "statement", code=(f"snapshot_{index} = {container.expression}.copy()",),
                     binds=f"snapshot_{index}")
    if primitive == "RESTORE":
        code = ((f"{container.expression}.clear()", f"{container.expression}.update(snapshot_{index})")
                if container.kind == "dict" else (f"{container.expression}[:] = list(snapshot_{index})",))
        return Token(primitive, roles, "statement", code=code, requires=(f"snapshot_{index}",))
    if primitive == "READ":
        if container.kind != "dict":
            return None
        return Token(primitive, roles, "statement",
                     code=(f"observed_{index} = {container.expression}[{observation.parameter}]",),
                     binds=f"observed_{index}")
    if primitive == "REMOVE":
        code = ((f"{container.expression}.pop({observation.parameter}, None)",) if container.kind == "dict"
                else (f"{container.expression}[:] = [item for item in {container.expression} "
                      f"if item != {observation.parameter}]",))
        return Token(primitive, roles, "statement", code=code)
    if primitive == "WRITE":
        value = _resolve_value(observation, roles[1])
        if not value:
            return None
        code = ((f"{container.expression}[{observation.parameter}] = {value}",) if container.kind == "dict"
                else (f"{container.expression}.append({value})",))
        requires = (value,) if roles[1].startswith("B:") else ()
        return Token(primitive, roles, "statement", code=code, requires=requires)
    if primitive in {"GUARD", "REPEAT"}:
        operator, target = roles[0].split(":")
        container = _resolve_container(observation, target)
        if container is None:
            return None
        parameter = observation.parameter
        condition = {"IN": f"{parameter} in {container.expression}",
                     "NOTIN": f"{parameter} not in {container.expression}",
                     "TRUE": f"{container.expression}[{parameter}]",
                     "FALSE": f"not {container.expression}[{parameter}]",
                     "DUP": f"{container.expression}.count({parameter}) > 1"}.get(operator, "")
        if not condition:
            return None
        if primitive == "GUARD":
            return Token(primitive, roles, "open", header=(f"if {condition}:",))
        return Token(primitive, roles, "open",
                     header=(f"for _repeat in range(len({container.expression}) + 1):",
                             f"    if not ({condition}):", "        break"))
    return None


def build_vocabulary(observation: Observation) -> tuple[Token, ...]:
    """Every primitive instance the observation supports, distractors included."""
    proposals: list[tuple[str, tuple[str, ...]]] = [("OPERATION", ()), ("EFFECT", ()), ("RESULT", ()), ("TRY", ()),
                                                    ("EXCEPT", ()), ("FINALLY", ()), ("END", ()), ("RAISE", ())]
    for index, container in enumerate(observation.containers):
        role = f"C{index}"
        proposals += [("GUARD", (f"IN:{role}",)), ("GUARD", (f"NOTIN:{role}",))]
        if "bool" in container.value_types:
            proposals += [("GUARD", (f"TRUE:{role}",)), ("GUARD", (f"FALSE:{role}",))]
        if container.kind == "list":
            proposals += [("REPEAT", (f"DUP:{role}",))]
        proposals += [("SAVE", (role,)), ("RESTORE", (role,)), ("READ", (role,)), ("REMOVE", (role,))]
        ranks: dict[str, int] = {}
        for kind, _ in observation.literals:
            rank = ranks.get(kind, 0)
            ranks[kind] = rank + 1
            if kind in container.value_types:
                proposals.append(("WRITE", (role, f"L:{kind}:{rank}")))
        for other in range(len(observation.containers)):
            if other != index:
                proposals.append(("WRITE", (role, f"R:C{other}")))
        proposals.append(("WRITE", (role, f"B:C{index}")))
    proposals += [(primitive, ()) for primitive in DISTRACTOR_PRIMITIVES]
    tokens = [build_token(primitive, roles, observation) for primitive, roles in proposals]
    return tuple(token for token in tokens if token is not None)


# --------------------------------------------------------------------------- composition grammar


@dataclass(frozen=True)
class Frame:
    kind: str
    statements: int
    terminated: bool


@dataclass(frozen=True)
class State:
    stack: tuple[Frame, ...]
    used: frozenset
    bound: frozenset
    consumed: frozenset
    parts: frozenset


INITIAL = State((Frame("ROOT", 0, False),), frozenset(), frozenset(), frozenset(), frozenset())


def advance(state: State, token: Token) -> State | None:
    """Apply one primitive to a partial program, rejecting ill-formed extensions."""
    key = (token.primitive, token.roles)
    if key in state.used:
        return None
    if set(token.parts) & state.parts:
        return None
    if not set(token.requires) <= state.bound:
        return None
    top = state.stack[-1]
    if token.kind == "handler":
        if top.kind != "TRY" or top.statements == 0:
            return None
        stack = state.stack[:-1] + (Frame("EXCEPT" if token.header[0].startswith("except") else "FINALLY", 0, False),)
        return State(stack, state.used | {key}, state.bound, state.consumed, state.parts)
    if token.kind == "close":
        if len(state.stack) == 1 or top.statements == 0 or top.kind == "TRY":
            return None
        return State(state.stack[:-1], state.used | {key}, state.bound, state.consumed, state.parts)
    if top.terminated:
        return None
    if token.primitive == "RAISE" and not any(frame.kind == "EXCEPT" for frame in state.stack):
        return None
    stack = state.stack[:-1] + (replace(top, statements=top.statements + 1),)
    if token.kind == "open":
        stack = stack + (Frame("TRY" if token.primitive == "TRY" else token.primitive, 0, False),)
    elif token.terminal:
        stack = stack[:-1] + (replace(stack[-1], terminated=True),)
    return State(stack, state.used | {key}, state.bound | ({token.binds} if token.binds else frozenset()),
                 state.consumed | set(token.requires), state.parts | set(token.parts))


def complete(state: State) -> bool:
    return (state.parts == {"EFFECT", "RESULT"} and all(frame.kind != "TRY" for frame in state.stack)
            and all(frame.statements > 0 for frame in state.stack) and state.bound <= state.consumed)


def feasible(state: State, remaining: int) -> bool:
    needed = (0 if state.parts == {"EFFECT", "RESULT"} else 1) + 2 * sum(frame.kind == "TRY" for frame in state.stack)
    return needed + len(state.bound - state.consumed) <= remaining


def render(program: tuple[Token, ...]) -> tuple[str, ...]:
    lines: list[str] = []
    depth = 0
    for token in program:
        if token.kind == "handler":
            depth -= 1
            lines += ["    " * depth + line for line in token.header]
            depth += 1
        elif token.kind == "close":
            depth -= 1
        elif token.kind == "open":
            lines += ["    " * depth + line for line in token.header]
            depth += 1
        else:
            lines += ["    " * depth + line for line in token.code]
    return tuple(lines)


def apply_program(source: str, observation: Observation, program: tuple[Token, ...]) -> str:
    lines = source.splitlines()
    first, last = observation.body_range
    body = [observation.indent + line if line else "" for line in render(program)]
    return "\n".join(lines[:first] + body + lines[last:]) + "\n"


def signature(program: tuple[Token, ...]) -> str:
    return ">".join(token.primitive for token in program)


# --------------------------------------------------------------------------- contract evaluation


def evaluate_fast(source: str, alias: str, script: str) -> bool:
    """Run a single-module candidate and its contract script in a fresh namespace."""
    module = types.ModuleType(alias)
    try:
        exec(compile(source, "<candidate>", "exec"), module.__dict__)
        exec(compile(script, "<contract>", "exec"), {alias: module})
    except Exception:
        return False
    return True


def evaluate_repo(sources: dict[str, str], alias: str, script: str) -> bool:
    """Run the same contract as a real unittest process against a real repository."""
    root = Path(tempfile.mkdtemp(prefix="l47-contract-"))
    try:
        for name, text in sources.items():
            root.joinpath(f"{name}.py").write_text(text, encoding="utf-8")
        root.joinpath("unrelated.py").write_text("UNRELATED = 1\n", encoding="utf-8")
        body = "\n".join("        " + line if line else "" for line in script.splitlines())
        root.joinpath("test_contract.py").write_text(
            f"import unittest\nimport {alias}\n\n\nclass Contract(unittest.TestCase):\n"
            f"    def test_contract(self):\n{body}\n", encoding="utf-8")
        return subprocess.run([sys.executable, "-m", "unittest", "discover", "-p", "test_*.py"],
                              cwd=root, capture_output=True, encoding="utf-8", errors="replace", check=False).returncode == 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------- search


@dataclass
class SearchOutcome:
    program: tuple[Token, ...] | None
    trials: int
    generated: int
    trace: tuple[tuple[str, str], ...]
    exhausted: bool


def search(sources: dict[str, str], alias: str, script: str, observation: Observation,
           extra: tuple[tuple[str, tuple[tuple[str, tuple[str, ...]], ...]], ...] = (),
           max_length: int = MAX_PROGRAM_LENGTH, budget: int = SEARCH_BUDGET) -> SearchOutcome:
    """Depth-bounded composition search selected only by the contract test."""
    vocabulary = build_vocabulary(observation)
    source = sources[alias]
    seen: set[str] = set()
    trace: list[tuple[str, str]] = []
    counters = {"trials": 0, "generated": 0}

    def attempt(program: tuple[Token, ...]) -> bool:
        counters["generated"] += 1
        candidate = apply_program(source, observation, program)
        if candidate in seen:
            return False
        seen.add(candidate)
        counters["trials"] += 1
        passed = evaluate_fast(candidate, alias, script)
        if len(trace) < 8 or passed:
            trace.append((signature(program), "PASS" if passed else "FAIL"))
        return passed

    for name, steps in extra:
        program = instantiate(steps, observation)
        if program is not None and attempt(program):
            return SearchOutcome(program, counters["trials"], counters["generated"], tuple(trace), False)

    def walk(program: tuple[Token, ...], state: State, remaining: int) -> tuple[Token, ...] | None:
        if counters["trials"] >= budget:
            return None
        if remaining == 0:
            return program if complete(state) and attempt(program) else None
        if not feasible(state, remaining):
            return None
        for token in vocabulary:
            following = advance(state, token)
            if following is None:
                continue
            found = walk(program + (token,), following, remaining - 1)
            if found is not None:
                return found
        return None

    for length in range(1, max_length + 1):
        found = walk((), INITIAL, length)
        if found is not None:
            return SearchOutcome(found, counters["trials"], counters["generated"], tuple(trace), False)
        if counters["trials"] >= budget:
            return SearchOutcome(None, counters["trials"], counters["generated"], tuple(trace), True)
    return SearchOutcome(None, counters["trials"], counters["generated"], tuple(trace), True)


# --------------------------------------------------------------------------- operator library


@dataclass(frozen=True)
class Operator:
    label: str
    steps: tuple[tuple[str, tuple[str, ...]], ...]
    support: int
    reuse_count: int
    acquisition_trials: int
    exact_body: tuple[str, ...]


def abstract(program: tuple[Token, ...], trials: int) -> Operator:
    """Compress a successful trace into role-parameterized steps."""
    steps = tuple((token.primitive, token.roles) for token in program)
    label = POST_HOC_LABELS.get(signature(program), "OPERATOR_" + signature(program).replace(">", "_"))
    return Operator(label, steps, 1, 0, trials, render(program))


def instantiate(steps: tuple[tuple[str, tuple[str, ...]], ...], observation: Observation) -> tuple[Token, ...] | None:
    """Re-render stored steps against a new observation, then re-check well-formedness."""
    program: list[Token] = []
    state = INITIAL
    for primitive, roles in steps:
        token = build_token(primitive, roles, observation)
        if token is None:
            return None
        state_or_none = advance(state, token)
        if state_or_none is None:
            return None
        state = state_or_none
        program.append(token)
    return tuple(program) if complete(state) else None


def description_length(operator: Operator, uses: int) -> int:
    """MDL gain of storing the operator instead of every raw trace."""
    return uses * len(operator.steps) - (len(operator.steps) + uses)


# --------------------------------------------------------------------------- contracts


@dataclass(frozen=True)
class Contract:
    identifier: str
    family: str
    role: str
    alias: str
    sources: dict[str, str]
    script: str


def contracts() -> tuple[Contract, ...]:
    rollback = (
        ("rollback-acquire",
         "inventory = {'widget': 10, 'gadget': 4, 'trinket': 3}\nattempts = 0\n\n\ndef operate(key):\n"
         "    global attempts\n    attempts += 1\n    inventory[key] -= 1\n    inventory['gadget'] += 1\n"
         "    if key == 'widget':\n        raise RuntimeError('payment declined')\n    return inventory[key]\n",
         "assert service.operate('trinket') == 2\n"
         "assert service.inventory == {'widget': 10, 'gadget': 5, 'trinket': 2}\n"
         "try:\n    service.operate('widget')\nexcept RuntimeError:\n    pass\nelse:\n"
         "    raise AssertionError('expected RuntimeError')\n"
         "assert service.attempts == 2\n"
         "assert service.inventory == {'widget': 10, 'gadget': 5, 'trinket': 2}\n"),
        ("rollback-holdout-a",
         "stock = {'panel': 7, 'spare': 2, 'bolt': 9}\ncalls = 0\n\n\ndef reserve(item_id):\n"
         "    global calls\n    calls += 1\n    stock[item_id] -= 1\n    stock['spare'] += 1\n"
         "    if item_id == 'panel':\n        raise ValueError('downstream failure')\n    return stock[item_id]\n",
         "assert service.reserve('bolt') == 8\n"
         "assert service.stock == {'panel': 7, 'spare': 3, 'bolt': 8}\n"
         "try:\n    service.reserve('panel')\nexcept ValueError:\n    pass\nelse:\n"
         "    raise AssertionError('expected ValueError')\n"
         "assert service.calls == 2\n"
         "assert service.stock == {'panel': 7, 'spare': 3, 'bolt': 8}\n"),
        ("rollback-holdout-b",
         "balances = {'checking': 250, 'fees': 0, 'savings': 80}\naudit = 0\n\n\ndef settle(reference):\n"
         "    global audit\n    audit += 1\n    balances[reference] -= 50\n    balances['fees'] += 5\n"
         "    if reference == 'checking':\n        raise LookupError('ledger unavailable')\n"
         "    return balances[reference]\n",
         "assert service.settle('savings') == 30\n"
         "assert service.balances == {'checking': 250, 'fees': 5, 'savings': 30}\n"
         "try:\n    service.settle('checking')\nexcept LookupError:\n    pass\nelse:\n"
         "    raise AssertionError('expected LookupError')\n"
         "assert service.audit == 2\n"
         "assert service.balances == {'checking': 250, 'fees': 5, 'savings': 30}\n"),
    )
    coherence = (
        ("coherence-acquire", "persistent = {'target': 'old'}\ncache = {'target': 'old'}\n\n\ndef operate(key):\n"
         "    persistent[key] = key.upper()\n    return persistent[key]\n",
         "assert service.operate('target') == 'TARGET'\nassert service.cache['target'] == 'TARGET'\n"),
        ("coherence-holdout-a", "store = {'row': 'stale'}\nmirror = {'row': 'stale'}\n\n\ndef refresh(identifier):\n"
         "    store[identifier] = identifier.title()\n    return store[identifier]\n",
         "assert service.refresh('row') == 'Row'\nassert service.mirror['row'] == 'Row'\n"),
        ("coherence-holdout-b", "primary = {'sku': 'draft'}\nreplica = {'sku': 'draft'}\n\n\ndef publish(record):\n"
         "    primary[record] = record + '-live'\n    return primary[record]\n",
         "assert service.publish('sku') == 'sku-live'\nassert service.replica['sku'] == 'sku-live'\n"),
    )
    cleanup = (
        ("cleanup-acquire", "resources = {'ok': True, 'bad': True}\nflaky = {'ok': False, 'bad': True}\n\n\n"
         "def operate(key):\n    handle = resources[key]\n    if flaky[key]:\n"
         "        raise RuntimeError('operation failed')\n    return handle\n",
         "assert service.operate('ok') is True\nassert service.resources['ok'] is False\n"
         "try:\n    service.operate('bad')\nexcept RuntimeError:\n    pass\nelse:\n"
         "    raise AssertionError('expected RuntimeError')\nassert service.resources['bad'] is False\n"),
        ("cleanup-holdout-a", "handles = {'alpha': True, 'beta': True}\nunstable = {'alpha': False, 'beta': True}\n\n\n"
         "def acquire(name):\n    resource = handles[name]\n    if unstable[name]:\n"
         "        raise OSError('acquire failed')\n    return resource\n",
         "assert service.acquire('alpha') is True\nassert service.handles['alpha'] is False\n"
         "try:\n    service.acquire('beta')\nexcept OSError:\n    pass\nelse:\n"
         "    raise AssertionError('expected OSError')\nassert service.handles['beta'] is False\n"),
        ("cleanup-holdout-b", "leases = {'north': True, 'south': True}\nbroken = {'north': False, 'south': True}\n\n\n"
         "def borrow(zone):\n    lease = leases[zone]\n    if broken[zone]:\n"
         "        raise KeyError('lease refused')\n    return lease\n",
         "assert service.borrow('north') is True\nassert service.leases['north'] is False\n"
         "try:\n    service.borrow('south')\nexcept KeyError:\n    pass\nelse:\n"
         "    raise AssertionError('expected KeyError')\nassert service.leases['south'] is False\n"),
    )
    idempotency = (
        ("idempotency-acquire", "state = ['first', 'target']\n\n\ndef operate(item):\n"
         "    state.append(item)\n    return state\n",
         "service.operate('target')\nassert service.state == ['first', 'target']\n"
         "service.operate('second')\nassert service.state == ['first', 'target', 'second']\n"),
        ("idempotency-holdout-a", "registry = ['alpha', 'beta']\n\n\ndef enroll(member):\n"
         "    registry.append(member)\n    return registry\n",
         "service.enroll('beta')\nassert service.registry == ['alpha', 'beta']\n"
         "service.enroll('gamma')\nassert service.registry == ['alpha', 'beta', 'gamma']\n"),
        ("idempotency-holdout-b", "queue = ['job-1']\n\n\ndef submit(job):\n"
         "    queue.append(job)\n    return queue\n",
         "service.submit('job-1')\nassert service.queue == ['job-1']\n"
         "service.submit('job-2')\nassert service.queue == ['job-1', 'job-2']\n"),
    )
    built: list[Contract] = []
    for family, group in (("rollback", rollback), ("coherence", coherence),
                          ("cleanup", cleanup), ("idempotency", idempotency)):
        for position, (identifier, source, script) in enumerate(group):
            built.append(Contract(identifier, family, "acquisition" if position == 0 else "holdout",
                                  "service", {"service": source}, script))
    return tuple(built)


def structural_holdout() -> Contract:
    """Same rollback contract, different repository structure and state shape."""
    sources = {
        "inventory": "class Inventory:\n    def __init__(self):\n"
                     "        self.counts = {'widget': 5, 'spare': 1, 'trinket': 4}\n\n"
                     "    def take(self, item):\n        self.counts[item] -= 1\n"
                     "        self.counts['spare'] += 1\n",
        "payment": "class Payment:\n    def charge(self, item):\n        if item == 'widget':\n"
                   "            raise RuntimeError('gateway declined')\n        return True\n",
        "coordinator": "from inventory import Inventory\nfrom payment import Payment\n\n"
                       "inventory = Inventory()\npayment = Payment()\nattempts = 0\n\n\ndef place(item):\n"
                       "    global attempts\n    attempts += 1\n"
                       "    inventory.take(item)\n    return payment.charge(item)\n",
    }
    script = ("assert coordinator.place('trinket') is True\n"
              "assert coordinator.inventory.counts == {'widget': 5, 'spare': 2, 'trinket': 3}\n"
              "try:\n    coordinator.place('widget')\nexcept RuntimeError:\n    pass\nelse:\n"
              "    raise AssertionError('expected RuntimeError')\n"
              "assert coordinator.attempts == 2\n"
              "assert coordinator.inventory.counts == {'widget': 5, 'spare': 2, 'trinket': 3}\n")
    return Contract("rollback-structural", "rollback", "holdout", "coordinator", sources, script)


# --------------------------------------------------------------------------- conditions


def _apply_exact(contract: Contract, observation: Observation, body: tuple[str, ...]) -> bool:
    source = contract.sources[contract.alias]
    lines = source.splitlines()
    first, last = observation.body_range
    patched = "\n".join(lines[:first] + [observation.indent + line if line else "" for line in body] + lines[last:]) + "\n"
    return evaluate_fast(patched, contract.alias, contract.script)


def solve(contract: Contract, library: tuple[Operator, ...], condition: str) -> dict[str, object]:
    """Solve one contract under one ablation condition; trials are the cost unit."""
    observation = observe(contract.sources, contract.alias, contract.script)
    if condition == "memorize_exact":
        for operator in library:
            if _apply_exact(contract, observation, operator.exact_body):
                return {"success": True, "trials": 1, "used": operator.label, "signature": ""}
        return {"success": False, "trials": len(library), "used": "", "signature": ""}
    if condition in {"full", "parameterized_operator"}:
        for index, operator in enumerate(library):
            program = instantiate(operator.steps, observation)
            if program is None:
                continue
            candidate = apply_program(contract.sources[contract.alias], observation, program)
            if evaluate_fast(candidate, contract.alias, contract.script):
                return {"success": True, "trials": index + 1, "used": operator.label,
                        "signature": signature(program)}
        if condition == "parameterized_operator":
            return {"success": False, "trials": max(1, len(library)), "used": "", "signature": ""}
    outcome = search(contract.sources, contract.alias, contract.script, observation)
    if outcome.program is None:
        return {"success": False, "trials": outcome.trials, "used": "", "signature": ""}
    return {"success": True, "trials": outcome.trials, "used": "", "signature": signature(outcome.program)}


# --------------------------------------------------------------------------- experiment


def _family_agnostic() -> bool:
    text = "".join(inspect.getsource(function) for function in
                   (observe, build_token, build_vocabulary, advance, complete, feasible, render,
                    apply_program, search, abstract, instantiate, solve)).lower()
    return not any(word in text for word in ("rollback", "coherence", "cleanup", "idempot", "sync"))


def run_compositional_operator_experiment() -> dict[str, object]:
    tasks = contracts()
    families: dict[str, dict[str, object]] = {}
    library: list[Operator] = []
    for family in ("rollback", "coherence", "cleanup", "idempotency"):
        group = [task for task in tasks if task.family == family]
        acquisition = group[0]
        observation = observe(acquisition.sources, acquisition.alias, acquisition.script)
        outcome = search(acquisition.sources, acquisition.alias, acquisition.script, observation)
        if outcome.program is None:
            families[family] = {"acquired": False, "search_trials": outcome.trials,
                                "candidates_generated": outcome.generated, "search_exhausted": outcome.exhausted}
            continue
        operator = abstract(outcome.program, outcome.trials)
        patched = dict(acquisition.sources)
        patched[acquisition.alias] = apply_program(acquisition.sources[acquisition.alias], observation, outcome.program)
        verified = evaluate_repo(patched, acquisition.alias, acquisition.script)
        reuse = []
        for holdout in group[1:]:
            reuse.append(solve(holdout, (operator,), "full"))
        uses = 1 + sum(bool(entry["success"]) for entry in reuse)
        operator = replace(operator, support=uses, reuse_count=uses - 1)
        library.append(operator)
        families[family] = {
            "acquired": True,
            "operator": operator.label,
            "label_source": "post_hoc",
            "signature": signature(outcome.program),
            "program": list(render(outcome.program)),
            "search_trials": outcome.trials,
            "candidates_generated": outcome.generated,
            "search_trace": [list(entry) for entry in outcome.trace],
            "verified_by_unittest_subprocess": verified,
            "holdout_success": all(entry["success"] for entry in reuse),
            "holdout_trials": [entry["trials"] for entry in reuse],
            "support": operator.support,
            "reuse_count": operator.reuse_count,
            "delta_description_length": description_length(operator, uses),
            "promoted_by_mdl": description_length(operator, uses) > 0,
            "search_cost_saved": operator.acquisition_trials - max(entry["trials"] for entry in reuse),
            "acquisition_cost_exceeds_reuse_cost":
                operator.acquisition_trials > max(entry["trials"] for entry in reuse),
        }
    holdouts = [task for task in tasks if task.role == "holdout"]
    ablation: dict[str, dict[str, float]] = {}
    for condition in ("full", "parameterized_operator", "memorize_exact", "no_compression"):
        rows = [solve(task, tuple(library), condition) for task in holdouts]
        ablation[condition] = {
            "holdouts": float(len(rows)),
            "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
            "mean_trials": sum(int(row["trials"]) for row in rows) / len(rows),
            "max_trials": float(max(int(row["trials"]) for row in rows)),
        }
    structural = structural_holdout()
    structural_observation = observe(structural.sources, structural.alias, structural.script)
    structural_result: dict[str, object] = {"success": False, "trials": 0, "operator": "",
                                            "verified_by_unittest_subprocess": False}
    for index, operator in enumerate(library):
        program = instantiate(operator.steps, structural_observation)
        if program is None:
            continue
        patched = dict(structural.sources)
        patched[structural.alias] = apply_program(structural.sources[structural.alias],
                                                  structural_observation, program)
        if evaluate_repo(patched, structural.alias, structural.script):
            structural_result = {"success": True, "trials": index + 1, "operator": operator.label,
                                 "state_expression": structural_observation.containers[0].expression,
                                 "program": list(render(program)),
                                 "verified_by_unittest_subprocess": True}
            break
    used_primitives = {primitive for operator in library for primitive, _ in operator.steps}
    return {
        "primitives": list(PRIMITIVES),
        "distractor_primitives": list(DISTRACTOR_PRIMITIVES),
        "search_is_family_agnostic": _family_agnostic(),
        "shared_composer": True,
        "max_program_length": MAX_PROGRAM_LENGTH,
        "operators_acquired": len(library),
        "operator_labels": [operator.label for operator in library],
        "distinct_operator_signatures": len({">".join(primitive for primitive, _ in operator.steps)
                                             for operator in library}),
        "all_families_acquired": all(entry.get("acquired") for entry in families.values()),
        "mdl_promoted_operators": sum(bool(entry.get("promoted_by_mdl")) for entry in families.values()),
        "unused_primitives": sorted(set(PRIMITIVES) - used_primitives),
        "distractors_unused": all(primitive not in used_primitives for primitive in DISTRACTOR_PRIMITIVES),
        "families": families,
        "ablation": ablation,
        "structural_holdout": structural_result,
    }


if __name__ == "__main__":
    print(json.dumps(run_compositional_operator_experiment(), ensure_ascii=False, indent=2, sort_keys=True))
