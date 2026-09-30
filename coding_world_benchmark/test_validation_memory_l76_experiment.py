import gzip
import json

from coding_world_benchmark.coding_agent_app import LocalWorkspaceAgent
from coding_world_benchmark.validation_memory_l76_experiment import (
    ValidationCompressor,
    ValidationMemoryRetriever,
)


def _workspace(tmp_path):
    folder = tmp_path / "su3_tuned_wilson_critical_benchmark"
    folder.mkdir()
    (folder / "test_validation.py").write_text(
        "def initialize_state(seed=0):\n    return seed\n\n"
        "def run_parameter_sweep(beta):\n    return beta\n\n"
        "def test_gauge_invariance():\n    assert True  # success condition\n\n"
        "# failure: finite-size instability; fix: increase system size\n",
        encoding="utf-8",
    )
    return tmp_path


def test_validation_files_are_compressed_into_concept_and_provenance(tmp_path):
    root = _workspace(tmp_path)
    store = root / ".conscious_coding_agent" / "validation_knowledge.jsonl.gz"
    knowledge = ValidationCompressor().compress(root, store)

    assert len(knowledge) == 1
    item = knowledge[0]
    assert item.concept == "gauge_theory_validation"
    assert "run_parameter_sweep" in item.reusable_operators
    assert item.failure_modes
    assert item.fixes
    assert item.provenance[0].file.endswith("test_validation.py")
    with gzip.open(store, "rt", encoding="utf-8") as stream:
        assert json.loads(stream.readline())["concept"] == "gauge_theory_validation"


def test_validation_memory_retrieves_reusable_pattern(tmp_path):
    knowledge = ValidationCompressor().compress(_workspace(tmp_path))
    result = ValidationMemoryRetriever(knowledge).retrieve("保存則とSU3の不変性を検証")
    assert result
    assert result[0].concept == "gauge_theory_validation"


def test_agent_refreshes_validation_memory_and_reuses_it(tmp_path):
    root = _workspace(tmp_path)
    result = LocalWorkspaceAgent(root).handle("SU3の検証パターンを再利用して保存則を検証")
    assert "Reusable Validation Knowledge" in result.text
    assert "gauge_theory_validation" in result.text
