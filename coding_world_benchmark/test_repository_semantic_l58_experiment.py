from pathlib import Path

from coding_world_benchmark.repository_semantic_l58_experiment import (
    context_for,
    explain_repository_question,
    index_repository,
    search_symbols,
)


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "cosmos.py").write_text(
        "def make_initial_state():\n"
        "    rho = 0.45\n"
        "    information = rho\n"
        "    return information\n\n"
        "def expA1_single_initial_state_full_cosmic_evolution():\n"
        "    state = make_initial_state()\n"
        "    return evolve(state)\n\n"
        "def exp445_pre_geometric_initial_state():\n"
        "    rho = 0.4\n"
        "    return rho\n\n"
        "def evolve(state):\n"
        "    return state\n",
        encoding="utf-8",
    )
    (tmp_path / "test_cosmos.py").write_text(
        "from cosmos import make_initial_state\n\n"
        "def test_state():\n"
        "    assert make_initial_state() == 0.45\n",
        encoding="utf-8",
    )
    return tmp_path


def test_ast_index_creates_symbols_and_experiment_entries(tmp_path):
    graph = index_repository(_repo(tmp_path))
    names = {symbol.name for symbol in graph.symbols}
    assert "cosmos.expA1_single_initial_state_full_cosmic_evolution" in names
    assert "cosmos.exp445_pre_geometric_initial_state" in names
    assert any(symbol.kind == "experiment" for symbol in graph.symbols)


def test_call_graph_and_dataflow_preserve_grounded_lineage(tmp_path):
    graph = index_repository(_repo(tmp_path))
    edges = {(edge.source, edge.target, edge.relation) for edge in graph.edges}
    assert ("cosmos.expA1_single_initial_state_full_cosmic_evolution", "cosmos.make_initial_state", "calls") in edges
    assert ("cosmos.rho", "cosmos.information", "flows_to") in edges
    assert any(item.evidence_type == "call" and item.line > 0 for item in graph.evidence)


def test_context_graph_reports_ambiguity_and_code_evidence(tmp_path):
    graph = index_repository(_repo(tmp_path))
    context = context_for(graph, "initial state")
    names = {item["target"] for item in context["interpretations"]}
    assert context["ambiguous"]
    assert "cosmos.expA1_single_initial_state_full_cosmic_evolution" in names
    assert "cosmos.exp445_pre_geometric_initial_state" in names
    assert context["code_evidence"]
    assert context["irrelevant_symbol_count"] > 0


def test_search_is_separate_from_semantic_context_assembly(tmp_path):
    graph = index_repository(_repo(tmp_path))
    symbols = search_symbols(graph, "expA1 initial state")
    context = explain_repository_question(tmp_path, "expA1 initial state")
    assert symbols[0].name == "cosmos.expA1_single_initial_state_full_cosmic_evolution"
    assert context["call_graph"]
    assert context["edge_count"] >= len(context["call_graph"])
