from coding_world_benchmark.external_repair_l81_experiment import parse_repair_proposal, plan_repair
from coding_world_benchmark.semantic_repair_l83_experiment import decompose_repair, format_repair_decomposition


def test_repair_changes_become_dependency_ordered_dag(tmp_path):
    (tmp_path / "router.py").write_text("class Router:\n    pass\n", encoding="utf-8")
    instruction = parse_repair_proposal("""Problem: repair routing
Target files: router.py
Suggested changes:
- add enum
- change router condition
- add helper
- add regression test
Regression tests:
- test_router.py
""")
    plan = plan_repair(tmp_path, instruction)
    decomposition = decompose_repair(instruction, plan)

    assert decomposition.cycle_free
    assert [step.id for step in decomposition.steps] == ["step_1", "step_2", "step_3", "step_4"]
    assert decomposition.steps[1].depends_on == ("step_1",)
    assert decomposition.steps[3].depends_on == ("step_1", "step_2", "step_3")
    assert "Semantic Repair Decomposition" in format_repair_decomposition(decomposition)
