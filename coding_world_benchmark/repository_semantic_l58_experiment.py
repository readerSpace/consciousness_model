"""L5.8: repository semantic understanding for the local coding agent.

The index is deliberately conservative: Python AST definitions, local calls,
simple assignment provenance, and source-line evidence.  It does not pretend
to execute a full Python type system.  Every explanation claim keeps a file
and line reference so a later LLM can summarize a grounded context graph.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str
    file: str
    line: int
    signature: str
    parent: str | None = None
    docstring: str = ""


@dataclass(frozen=True)
class CodeEdge:
    source: str
    target: str
    relation: str
    file: str
    line: int


@dataclass(frozen=True)
class CodeEvidence:
    claim: str
    file: str
    line: int
    symbol: str
    evidence_type: str


@dataclass(frozen=True)
class RepositoryGraph:
    symbols: tuple[Symbol, ...]
    edges: tuple[CodeEdge, ...]
    evidence: tuple[CodeEvidence, ...]

    def symbol(self, qualified_name: str) -> Symbol | None:
        return next((item for item in self.symbols if item.name == qualified_name), None)


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = [arg.arg for arg in node.args.args]
    return f"{node.name}({', '.join(args)})"


def _qualified(parent: str | None, name: str) -> str:
    return f"{parent}.{name}" if parent else name


def _names(node: ast.AST, context: ast.expr_context | None = None) -> tuple[str, ...]:
    return tuple(sorted({item.id for item in ast.walk(node) if isinstance(item, ast.Name) and (context is None or isinstance(item.ctx, context))}))


def index_repository(root: Path) -> RepositoryGraph:
    root = root.resolve()
    symbols: list[Symbol] = []
    edges: list[CodeEdge] = []
    evidence: list[CodeEvidence] = []
    python_files = sorted(path for path in root.rglob("*.py") if not {".git", ".venv", "__pycache__"}.intersection(path.parts))
    local_functions: dict[str, tuple[str, str, int]] = {}

    for path in python_files:
        relative = str(path.relative_to(root))
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=relative)
        except (OSError, SyntaxError):
            continue
        module_name = relative[:-3].replace("\\", ".").replace("/", ".")
        if module_name.endswith(".__init__"):
            module_name = module_name[:-9]
        symbols.append(Symbol(module_name, "module", relative, 1, module_name))
        parent_by_node: dict[int, str] = {}

        def assign_parents(parent: str, current: ast.AST) -> None:
            for child in ast.iter_child_nodes(current):
                parent_by_node[id(child)] = parent
                next_parent = _qualified(module_name, child.name) if isinstance(child, ast.ClassDef) else parent
                assign_parents(next_parent, child)

        assign_parents(module_name, tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parent = parent_by_node.get(id(node), module_name)
                qualified = _qualified(parent, node.name)
                kind = "experiment" if node.name.startswith(("exp", "test_")) else "function"
                symbols.append(Symbol(qualified, kind, relative, node.lineno, _signature(node), parent, ast.get_docstring(node) or ""))
                local_functions[node.name] = (qualified, relative, node.lineno)
            elif isinstance(node, ast.ClassDef):
                symbols.append(Symbol(_qualified(module_name, node.name), "class", relative, node.lineno, f"class {node.name}", module_name, ast.get_docstring(node) or ""))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        qualified = _qualified(module_name, target.id)
                        symbols.append(Symbol(qualified, "constant", relative, target.lineno, target.id, module_name))

        source_lines = source.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            function_name = _qualified(module_name, node.name)
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in local_functions:
                    callee = local_functions[call.func.id][0]
                    edges.append(CodeEdge(function_name, callee, "calls", relative, call.lineno))
                    evidence.append(CodeEvidence(f"{function_name} calls {callee}", relative, call.lineno, function_name, "call"))
            for assignment in ast.walk(node):
                if not isinstance(assignment, ast.Assign):
                    continue
                written = _names(assignment.targets[0], ast.Store) if assignment.targets else ()
                read = _names(assignment.value, ast.Load)
                for variable in written:
                    edges.append(CodeEdge(function_name, f"{module_name}.{variable}", "writes", relative, assignment.lineno))
                    evidence.append(CodeEvidence(f"{function_name} writes {variable}", relative, assignment.lineno, function_name, "dataflow"))
                    for upstream in read:
                        edges.append(CodeEdge(f"{module_name}.{upstream}", f"{module_name}.{variable}", "flows_to", relative, assignment.lineno))
                        evidence.append(CodeEvidence(f"{variable} receives {upstream}", relative, assignment.lineno, function_name, "provenance"))
                for variable in read:
                    edges.append(CodeEdge(function_name, f"{module_name}.{variable}", "reads", relative, assignment.lineno))
            for returned in ast.walk(node):
                if isinstance(returned, ast.Return) and returned.value is not None:
                    for variable in _names(returned.value, ast.Load):
                        edges.append(CodeEdge(f"{module_name}.{variable}", function_name, "returns", relative, returned.lineno))
                        evidence.append(CodeEvidence(f"{function_name} returns {variable}", relative, returned.lineno, function_name, "return"))
            if node.lineno <= len(source_lines):
                evidence.append(CodeEvidence(f"{function_name} defined as {_signature(node)}", relative, node.lineno, function_name, "definition"))
    return RepositoryGraph(tuple(symbols), tuple(edges), tuple(evidence))


def search_symbols(graph: RepositoryGraph, query: str, *, limit: int = 12) -> tuple[Symbol, ...]:
    tokens = [token.lower() for token in re.findall(r"[a-z][a-z0-9_]{2,}|[一-龥ぁ-んァ-ヶ]{2,}", query)]
    scored = []
    for symbol in graph.symbols:
        haystack = " ".join((symbol.name, symbol.signature, symbol.docstring, symbol.file)).lower()
        score = sum(1 for token in tokens if token in haystack)
        if score:
            scored.append((score, symbol.kind == "experiment", symbol))
    return tuple(item[2] for item in sorted(scored, key=lambda item: (item[0], item[1], -item[2].line), reverse=True)[:limit])


def context_for(graph: RepositoryGraph, query: str, *, limit: int = 6) -> dict[str, object]:
    candidates = search_symbols(graph, query, limit=limit)
    candidate_names = {item.name for item in candidates}
    selected_edges = tuple(edge for edge in graph.edges if edge.source in candidate_names or edge.target in candidate_names)
    selected_evidence = tuple(item for item in graph.evidence if item.symbol in candidate_names or any(item.claim.startswith(edge.source) for edge in selected_edges))
    interpretations = [{"target": item.name, "kind": item.kind, "file": item.file, "line": item.line,
                        "signature": item.signature, "confidence": round(1.0 / max(1, len(candidates)), 2)} for item in candidates]
    return {
        "query": query,
        "ambiguous": len(candidates) > 1,
        "interpretations": interpretations,
        "call_graph": [edge.__dict__ for edge in selected_edges if edge.relation == "calls"],
        "dataflow": [edge.__dict__ for edge in selected_edges if edge.relation in {"writes", "reads", "flows_to", "returns"}],
        "code_evidence": [item.__dict__ for item in selected_evidence],
        "irrelevant_symbol_count": max(0, len(graph.symbols) - len(candidates)),
    }


def explain_repository_question(root: Path, question: str) -> dict[str, object]:
    graph = index_repository(root)
    context = context_for(graph, question)
    context["symbol_count"] = len(graph.symbols)
    context["edge_count"] = len(graph.edges)
    context["file_count"] = len({symbol.file for symbol in graph.symbols})
    return context
