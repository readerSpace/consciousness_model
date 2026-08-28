from functional_consciousness_benchmark import run


def test_ablation_pattern_matches_the_stated_architecture():
    scores = run()
    assert scores["C0"]["mean"] < scores["C1"]["mean"]
    assert scores["C1"]["mean"] < scores["C2"]["mean"] < scores["C3"]["mean"]
    assert scores["C2"]["novel_problem_solving"] == 1.0
    assert scores["C2"]["global_availability"] == 0.0
    assert scores["C3"]["flexible_planning"] == 1.0
    assert scores["C3"]["metacognition"] == 1.0
