import gzip
import json

from coding_world_benchmark.conversation_memory_l73_experiment import ConversationKnowledgeMemory


def test_memory_compresses_and_reloads_research_facts(tmp_path):
    path = tmp_path / ".knowledge_memory.json.gz"
    memory = ConversationKnowledgeMemory(path)
    memory.learn(
        "量子情報から時空が創発するか検証して",
        "【結論】\nentanglement geometry relation is a hypothesis.\n$$ S_A = Area / 4G_N $$\nsource: https://arxiv.org/abs/hep-th/0603001",
    )
    memory.save()

    assert path.is_file()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    assert payload["version"] == 1
    assert len(payload["facts"]) >= 2

    restored = ConversationKnowledgeMemory(path)
    context = restored.context("量子情報と時空のentanglement")
    assert "entanglement geometry" in context
    assert "arxiv" in context


def test_repeated_fact_is_compressed_by_use_count(tmp_path):
    memory = ConversationKnowledgeMemory(tmp_path / "memory.gz")
    memory.learn("wave equation", "equation: u_tt = c^2 u_xx")
    memory.learn("wave equation", "equation: u_tt = c^2 u_xx")

    assert len(memory.facts) == 1
    assert memory.facts[0].uses == 2


def test_task_contract_memory_policy_excludes_scientific_validation_noise(tmp_path):
    memory = ConversationKnowledgeMemory(tmp_path / "memory.gz")
    memory.learn("gauge_theory_validation", "conclusion: SU2/SU3 gauge validation passed")
    memory.learn("behavior monitor", "status: TaskSpec ExpectedBehavior ObservedBehavior mismatch contract")

    context = memory.context("TaskSpec ExpectedBehavior Behavior Monitor", policy="task_contract_only")

    assert "ExpectedBehavior" in context
    assert "SU2" not in context
    assert "gauge" not in context


def test_scientific_modeling_memory_policy_excludes_repair_and_gauge_noise(tmp_path):
    memory = ConversationKnowledgeMemory(tmp_path / "memory.gz")
    memory.learn("gauge_theory_validation", "conclusion: SU2/SU3 gauge validation passed")
    memory.learn("router repair", "status: router fallback patch applied")
    memory.learn("magnetic particle", "equation: Lorentz force particle ODE with RK4 simulation")

    context = memory.context("一様磁場 荷電粒子 simulation", policy="scientific_modeling_only")

    assert "Lorentz force" in context
    assert "SU2" not in context
    assert "router" not in context
